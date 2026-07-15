import os
import sys
import time
import json
import gc
import psutil

import torch
import scanpy as sc
import pandas as pd

sys.path.append('../')
from GraphST import GraphST


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def load_and_preprocess(dataset_ids, hvg_genes, preprocess_variant, file_fold):
    adatas = []
    total_cells = 0

    for dataset in dataset_ids:
        print(f"  Loading dataset {dataset}...")
        adata = sc.read_visium(
            os.path.join(file_fold, dataset),
            count_file=f"filtered_feature_bc_matrix.h5",
            load_images=True,
        )
        adata.var_names_make_unique()
        adata.obs["batch"] = dataset
        total_cells += adata.n_obs

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

        if "highly_variable" in adata.var:
            adata = adata[:, adata.var["highly_variable"]].copy()
        else:
            raise ValueError("Preprocessing did not produce highly_variable genes for selection.")

        adatas.append(adata)
        print(f"    {dataset}: {adata.n_obs} cells, {adata.n_vars} genes")

    combined = adatas[0].concatenate(adatas[1:], batch_key="batch")
    print(f"  Combined shape: {combined.shape}")
    return combined, total_cells


def run_hbc(hvg_genes, preprocess_variant, output_dir, device):
    print(f"\n=== Running HBC, variant={preprocess_variant}, hvg={hvg_genes} ===")
    dataset_ids = ["section1", "section2"]
    n_clusters = 10
    file_fold = os.path.join("..", "..", "RAW_SLICE", "hbc")

    adata, total_cells = load_and_preprocess(dataset_ids, hvg_genes, preprocess_variant, file_fold)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_before = get_memory_usage()
    training_start_time = time.time()

    model = GraphST.GraphST(adata, device=device, random_seed=50)
    adata = model.train()

    training_end_time = time.time()
    memory_after = get_memory_usage()
    training_time = training_end_time - training_start_time
    memory_used = memory_after - memory_before

    print("\nContinuing with clustering...")
    from GraphST.utils import clustering
    tool = "mclust"
    if tool == "mclust":
        clustering(adata, n_clusters, method=tool)
    elif tool in ["leiden", "louvain"]:
        clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)

    batch_mapping = {
        "0": "section1",
        "1": "section2",
    }
    adata.obs["new_batch"] = adata.obs["batch"].replace(batch_mapping)

    # Adjust obs_names for metadata mapping
    new_obs_names = []
    for i, obs_name in enumerate(adata.obs_names):
        base_name = obs_name.split("-")[0]
        batch = adata.obs["batch"].iloc[i]
        suffix = "-1" if batch == "0" else "-2"
        new_obs_names.append(base_name + suffix)

    adata.obs_names = new_obs_names
    adata.obs["celltype"] = "Unknown"

    # Load metadata and assign celltypes
    for dataset in dataset_ids:
        suffix = "-1" if dataset == "section1" else "-2"
        meta_path = os.path.join(file_fold, dataset, "metadata.csv")
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path, index_col=0)
            meta.index = [idx.split("-")[0] + suffix for idx in meta.index]
            common_barcodes = adata.obs_names.intersection(meta.index)
            for barcode in common_barcodes:
                adata.obs.loc[barcode, "celltype"] = meta.loc[barcode, "celltype"]

    adata.obs["celltype"] = adata.obs["celltype"].fillna("Unknown")

    output_base = f"hbc_adata_{preprocess_variant}_hvg{hvg_genes}"
    output_h5ad = os.path.join(output_dir, f"{output_base}.h5ad")
    output_json = os.path.join(output_dir, f"graphst_benchmark_hbc_{preprocess_variant}_hvg{hvg_genes}.json")

    os.makedirs(output_dir, exist_ok=True)
    adata.write(output_h5ad)

    benchmark_results = {
        "method_name": "GraphST",
        "dataset_group": "hbc",
        "preprocess_variant": preprocess_variant,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "training_time_hours": training_time / 3600,
        "memory_usage_mb": memory_used,
        "memory_usage_gb": memory_used / 1024,
        "total_cells": total_cells,
        "final_cells": adata.n_obs,
        "total_genes": adata.n_vars,
        "embedding_dim": int(adata.obsm["emb"].shape[1]),
        "n_datasets": len(dataset_ids),
        "device": str(device),
        "random_seed": 50,
        "hvg_genes": hvg_genes,
        "n_clusters": n_clusters,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    with open(output_json, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"  Saved {output_h5ad}")
    print(f"  Saved {output_json}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hvg_values = [1000, 3000, 5000]
    preprocess_variants = ["log1p", "sct"]
    output_dir = os.path.join("..", "results")

    for variant in preprocess_variants:
        for hvg_genes in hvg_values:
            run_hbc(
                hvg_genes=hvg_genes,
                preprocess_variant=variant,
                output_dir=output_dir,
                device=device,
            )


if __name__ == "__main__":
    main()
