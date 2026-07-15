import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('..')
import os
import time
import psutil
import gc
import json
import torch

import numpy as np
import scipy.sparse as sp
import scipy.linalg
import pandas as pd
import anndata as ad
import scanpy as sc
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

import STAligner
from STAligner import ST_utils


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def load_and_preprocess(dataset_ids, hvg_genes, preprocess_variant):
    """Load datasets and apply preprocessing."""
    file_fold = os.path.join("..", "..", "RAW_SLICE", "DLPFC")
    Batch_list = []
    adj_list = []
    total_cells = 0

    for dataset in dataset_ids:
        print(f"   Processing dataset: {dataset}")
        adata = sc.read_visium(
            os.path.join(file_fold, dataset),
            count_file=f"{dataset}_filtered_feature_bc_matrix.h5",
            load_images=True,
        )
        adata.var_names_make_unique()

        # read the annotation
        Ann_df = pd.read_csv(
            os.path.join(file_fold, dataset, f"{dataset}_truth.txt"),
            sep="\t",
            header=None,
            index_col=0,
        )
        Ann_df.columns = ["Ground Truth"]
        Ann_df[Ann_df.isna()] = "unknown"
        adata.obs["Ground Truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"].astype("category")

        # make spot name unique
        adata.obs_names = [x + "_" + dataset for x in adata.obs_names]
        adata.obs["batch"] = dataset

        # Constructing the spatial network
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=150)
        # STAligner.Stats_Spatial_Net(adata)

        # Preprocessing
        if preprocess_variant == "log1p":
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_genes)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        elif preprocess_variant == "sct":
            adata.layers["counts"] = adata.X.copy()
            sc.experimental.pp.recipe_pearson_residuals(
                adata,
                n_top_genes=hvg_genes,
                theta=100,
                clip=None,
                inplace=True,
            )
        else:
            raise ValueError(f"Unknown preprocess variant: {preprocess_variant}")

        if "highly_variable" not in adata.var:
            raise ValueError("Preprocessing did not produce highly_variable genes.")

        adata = adata[:, adata.var["highly_variable"]].copy()

        adj_list.append(adata.uns["adj"])
        Batch_list.append(adata)
        total_cells += adata.n_obs
        print(f"    {dataset}: {adata.n_obs} cells processed")

    # Concatenate
    adata_concat = ad.concat(Batch_list, label="slice_name", keys=dataset_ids)
    adata_concat.obs["celltype"] = adata_concat.obs["Ground Truth"].astype("category")
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype("category")
    print(f"   Combined shape: {adata_concat.shape}")

    # Build adjacency edge list from sparse block-diagonal graph
    adj_concat = sp.block_diag(adj_list, format="coo")
    adj_concat.eliminate_zeros()
    adj_concat = adj_concat.tocoo()
    adata_concat.uns["edgeList"] = (adj_concat.row.astype(np.int64), adj_concat.col.astype(np.int64))

    return adata_concat, total_cells


def run_group(group_name, dataset_ids, n_clusters, hvg_genes, preprocess_variant, output_dir, device):
    """Run STAligner for a specific group and configuration."""
    print(f"\n=== Running group={group_name}, variant={preprocess_variant}, hvg={hvg_genes} ===")

    adata_concat, total_cells = load_and_preprocess(dataset_ids, hvg_genes, preprocess_variant)

    # Training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_before = get_memory_usage()
    training_start_time = time.time()

    print("Initializing STAligner model...")
    adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=100, device=device)
    edge_list = [
        [left, right]
        for left, right in zip(adata_concat.uns["edgeList"][0], adata_concat.uns["edgeList"][1])
    ]
    adata_concat.uns["edgeList"] = edge_list

    training_end_time = time.time()
    memory_after = get_memory_usage()
    training_time = training_end_time - training_start_time
    memory_used = memory_after - memory_before

    print("Training completed!")

    # Clustering
    ST_utils.mclust_R(adata_concat, num_cluster=n_clusters, used_obsm="STAligner")
    adata_concat = adata_concat[adata_concat.obs["celltype"] != "unknown"]
    adata_concat.obs["new_batch"] = adata_concat.obs["batch_name"].astype(str).astype("category")

    # Save results
    output_base = f"staligner_{group_name}_{preprocess_variant}_hvg{hvg_genes}"
    output_h5ad = os.path.join(output_dir, f"{output_base}.h5ad")
    output_json = os.path.join(output_dir, f"{output_base}_benchmark.json")

    os.makedirs(output_dir, exist_ok=True)
    adata_concat.write(output_h5ad)

    benchmark_results = {
        "method_name": "STAligner",
        "dataset_group": group_name,
        "preprocess_variant": preprocess_variant,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "training_time_hours": training_time / 3600,
        "memory_usage_mb": memory_used,
        "memory_usage_gb": memory_used / 1024,
        "total_cells": total_cells,
        "total_genes": adata_concat.n_vars,
        "embedding_dim": int(adata_concat.obsm["STAligner"].shape[1]),
        "n_datasets": len(dataset_ids),
        "random_seed": 50,
        "hvg_genes": hvg_genes,
        "knn_neigh": 100,
        "rad_cutoff": 150,
        "n_clusters": n_clusters,
        "timestamp": pd.Timestamp.now().isoformat(),
        "device": str(device),
    }

    with open(output_json, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"  Saved {output_h5ad}")
    print(f"  Saved {output_json}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    group_configs = [
        # {"name": "1new", "datasets": ["151673", "151674", "151675", "151676"], "n_clusters": 7},
        # {"name": "2new", "datasets": ["151669", "151670", "151671", "151672"], "n_clusters": 5},
        # {"name": "3new", "datasets": ["151507", "151508", "151509", "151510"], "n_clusters": 7},
        # {"name": "7374new", "datasets": ["151673", "151674"], "n_clusters": 7},
        {"name": "all_new", "datasets": ["151673", "151669", "151507"], "n_clusters": 7},
    ]

    hvg_values = [1000, 3000, 5000]
    preprocess_variants = ["log1p", "sct"]
    output_dir = os.path.join("..", "results")

    for group in group_configs:
        for variant in preprocess_variants:
            for hvg_genes in hvg_values:
                run_group(
                    group_name=group["name"],
                    dataset_ids=group["datasets"],
                    n_clusters=group["n_clusters"],
                    hvg_genes=hvg_genes,
                    preprocess_variant=variant,
                    output_dir=output_dir,
                    device=device,
                )


if __name__ == "__main__":
    main()
