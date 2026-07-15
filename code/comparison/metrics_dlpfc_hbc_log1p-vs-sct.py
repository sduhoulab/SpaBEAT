import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import scanpy as sc
import scib
import anndata as ad
import gc
import numpy as np
import scib_metrics

sys.path.append('..')
from metrics.spatial_metrics import spatialbench

# Enable memory optimization
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'


group_configs = {
    "1new": {
        "raw": "../DATA_RAW/raw_adata1.h5ad",
        "graphst_base": "../GraphST/results/DLPFC_adata_1new_{variant}_hvg{hvg}.h5ad",
        "staligner_base": "../STAligner/results/staligner_1new_{variant}_hvg{hvg}.h5ad",
        "graphst_cluster_key": "mclust",
        "graphst_batch_key": "new_batch",
        "graphst_celltype_key": "ground_truth",
        "graphst_rep": "emb_pca",
        "staligner_cluster_key": "mclust",
        "staligner_batch_key": "batch_name",
        "staligner_celltype_key": "celltype",
        "staligner_rep": "STAligner",
        "raw_cluster_key": "mclust",
        "raw_batch_key": "new_batch",
        "raw_celltype_key": "celltype",
        "raw_rep": "X_pca",
    },
    "2new": {
        "raw": "../DATA_RAW/raw_adata2.h5ad",
        "graphst_base": "../GraphST/results/DLPFC_adata_2new_{variant}_hvg{hvg}.h5ad",
        "staligner_base": "../STAligner/results/staligner_2new_{variant}_hvg{hvg}.h5ad",
        "graphst_cluster_key": "domain",
        "graphst_batch_key": "new_batch",
        "graphst_celltype_key": "ground_truth",
        "graphst_rep": "emb_pca",
        "staligner_cluster_key": "mclust",
        "staligner_batch_key": "batch_name",
        "staligner_celltype_key": "celltype",
        "staligner_rep": "STAligner",
        "raw_cluster_key": "mclust",
        "raw_batch_key": "new_batch",
        "raw_celltype_key": "celltype",
        "raw_rep": "X_pca",
    },
    "3new": {
        "raw": "../DATA_RAW/raw_adata3.h5ad",
        "graphst_base": "../GraphST/results/DLPFC_adata_3new_{variant}_hvg{hvg}.h5ad",
        "staligner_base": "../STAligner/results/staligner_3new_{variant}_hvg{hvg}.h5ad",
        "graphst_cluster_key": "domain",
        "graphst_batch_key": "new_batch",
        "graphst_celltype_key": "ground_truth",
        "graphst_rep": "emb_pca",
        "staligner_cluster_key": "mclust",
        "staligner_batch_key": "batch_name",
        "staligner_celltype_key": "celltype",
        "staligner_rep": "STAligner",
        "raw_cluster_key": "mclust",
        "raw_batch_key": "new_batch",
        "raw_celltype_key": "celltype",
        "raw_rep": "X_pca",
    },
    "7374new": {
        "raw": "../DATA_RAW/raw_adata_7374.h5ad",
        "graphst_base": "../GraphST/results/DLPFC_adata_7374new_{variant}_hvg{hvg}.h5ad",
        "staligner_base": "../STAligner/results/staligner_7374new_{variant}_hvg{hvg}.h5ad",
        "graphst_cluster_key": "domain",
        "graphst_batch_key": "new_batch",
        "graphst_celltype_key": "ground_truth",
        "graphst_rep": "emb_pca",
        "staligner_cluster_key": "mclust",
        "staligner_batch_key": "batch_name",
        "staligner_celltype_key": "celltype",
        "staligner_rep": "STAligner",
        "raw_cluster_key": "mclust",
        "raw_batch_key": "new_batch",
        "raw_celltype_key": "celltype",
        "raw_rep": "X_pca",
    },
    "all_new": {
        "raw": "../DATA_RAW/raw_adata_all.h5ad",
        "graphst_base": "../GraphST/results/DLPFC_adata_all_new_{variant}_hvg{hvg}.h5ad",
        "staligner_base": "../STAligner/results/staligner_all_new_{variant}_hvg{hvg}.h5ad",
        "graphst_cluster_key": "domain",
        "graphst_batch_key": "new_batch",
        "graphst_celltype_key": "ground_truth",
        "graphst_rep": "emb_pca",
        "staligner_cluster_key": "mclust",
        "staligner_batch_key": "batch_name",
        "staligner_celltype_key": "celltype",
        "staligner_rep": "STAligner",
        "raw_cluster_key": "mclust",
        "raw_batch_key": "new_batch",
        "raw_celltype_key": "celltype",
        "raw_rep": "X_pca",
    },
    "hbc": {
        "raw": "../DATA_RAW/raw_adata_hbc.h5ad",
        "graphst_base": "../GraphST/results/hbc_adata_{variant}_hvg{hvg}.h5ad",
        "staligner_base": "../STAligner/results/staligner_hbc_{variant}_hvg{hvg}.h5ad",
        "graphst_cluster_key": "domain",
        "graphst_batch_key": "new_batch",
        "graphst_celltype_key": "celltype",
        "graphst_rep": "emb_pca",
        "staligner_cluster_key": "mclust",
        "staligner_batch_key": "new_batch",
        "staligner_celltype_key": "celltype",
        "staligner_rep": "STAligner",
        "raw_cluster_key": "mclust",
        "raw_batch_key": "new_batch",
        "raw_celltype_key": "celltype",
        "raw_rep": "X_pca",
    },
}

hvg_values = [1000, 3000, 5000]
variants = ["log1p", "sct"]

metrics_columns = [
    "dataset_group",
    "hvg",
    "model",
    "variant",
    "SCS",
    # "Bio_MoranI",
    # "Bio_GearyC",
    # "Batch_MoranI",
    # "Batch_GearyC",
    "Overall_MoranI",
    "Overall_GearyC",
    "graph_connectivity",
    "iLISI",
    "cLISI",
    "kBET",
    "ASW_batch",
    "ASW_celltype",
    "ARI",
]


def ensure_representation(adata, rep):
    """Safely compute representation with memory checks."""
    if rep == "X_pca":
        if "X_pca" not in adata.obsm:
            try:
                n_comps = min(50, adata.n_obs // 2, adata.n_vars // 2)
                sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=False, svd_solver="arpack")
            except Exception as e:
                print(f"Warning: PCA computation failed: {e}")
                pass
    elif rep not in adata.obsm:
        raise ValueError(f"Representation '{rep}' not found in AnnData object.")


def compute_metrics(adata, cluster_key, batch_key, celltype_key, use_rep):
    results = {}
    ensure_representation(adata, use_rep)

    res = spatialbench.benchmark_comprehensive(
        adata,
        cluster_key=cluster_key,
        batch_key=batch_key,
        celltype_key=celltype_key,
        degree=6,
        rep_time=1000,
        top_n=5,
        seed=0,
        use_rep=use_rep,
    )

    sc.pp.neighbors(adata, n_neighbors=30, use_rep=use_rep, random_state=666)
    results["SCS"] = res["SCS"]
    # results["Bio_MoranI"] = res["bio_preservation"]["MoranI"]
    # results["Bio_GearyC"] = res["bio_preservation"]["GearyC"]
    # results["Batch_MoranI"] = res["batch_correction"]["MoranI"]
    # results["Batch_GearyC"] = res["batch_correction"]["GearyC"]
    results["Overall_MoranI"] = res["overall_preservation"]["MoranI"]
    results["Overall_GearyC"] = res["overall_preservation"]["GearyC"]
    results["graph_connectivity"] = scib.me.graph_connectivity(adata, label_key=celltype_key)
    results["iLISI"] = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=use_rep)
    results["cLISI"] = scib.me.clisi_graph(adata, label_key=celltype_key, type_="embed", use_rep=use_rep)
    results["kBET"] = scib.me.kBET(
        adata,
        batch_key=batch_key,
        label_key=celltype_key,
        type_="embed",
        embed=use_rep,
    )
    results["ASW_batch"] = scib.me.silhouette_batch(
        adata,
        batch_key=batch_key,
        label_key=celltype_key,
        embed=use_rep,
    )
    results["ASW_celltype"] = scib.me.silhouette(
        adata,
        label_key=celltype_key,
        embed=use_rep,
    )
    results["ARI"] = scib.me.ari(adata, cluster_key=cluster_key, label_key=celltype_key)

    return results


def load_adata(path):
    """Load AnnData with memory optimization and validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    gc.collect()
    try:
        adata = sc.read_h5ad(path)
        if adata.X is None or adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError(f"Invalid data in {path}")
        # Convert to float32 to reduce memory
        try:
            adata.X = adata.X.astype(np.float32)
        except:
            pass
        return adata
    except Exception as e:
        print(f"Error loading {path}: {e}")
        raise


def safe_compute(adata, cluster_key, batch_key, celltype_key, use_rep, method_name):
    """Safely compute metrics with error recovery and memory cleanup."""
    try:
        if cluster_key not in adata.obs or batch_key not in adata.obs or celltype_key not in adata.obs:
            missing = [k for k in [cluster_key, batch_key, celltype_key] if k not in adata.obs]
            raise ValueError(f"Missing keys in obs: {missing}")
        result = compute_metrics(adata, cluster_key, batch_key, celltype_key, use_rep)
        gc.collect()
        return result
    except Exception as exc:
        print(f"Warning: failed metrics for {method_name}: {exc}")
        gc.collect()
        return {k: None for k in metrics_columns if k not in ["dataset_group", "hvg", "model", "variant"]}


def main():
    records = []
    checkpoint_file = "../results/metrics_dlpfc_hbc_checkpoint.csv"
    
    # Load checkpoint if exists
    processed_items = set()
    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            processed_items = set(zip(df_checkpoint['dataset_group'], df_checkpoint['hvg'], df_checkpoint['model'], df_checkpoint['variant']))
            records = df_checkpoint.to_dict('records')
            print(f"Loaded {len(records)} records from checkpoint.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting fresh.")

    for group_name, config in group_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing group: {group_name}")
        print(f"{'='*60}")
        
        try:
            raw_path = config["raw"]
            raw_adata = load_adata(raw_path)
            print(f"Loaded raw data: {raw_adata.shape}")
        except Exception as e:
            print(f"ERROR: Failed to load raw data for {group_name}: {e}")
            continue

        for hvg in hvg_values:
            print(f"\n  HVG={hvg}...")
            
            if (group_name, hvg, "RAW", "raw") in processed_items:
                print(f"    (Skipping RAW - already processed)")
            else:
                try:
                    raw_metrics = safe_compute(
                        raw_adata.copy(),
                        config["raw_cluster_key"],
                        config["raw_batch_key"],
                        config["raw_celltype_key"],
                        config["raw_rep"],
                        f"RAW/{group_name}",
                    )
                    records.append(
                        {
                            "dataset_group": group_name,
                            "hvg": hvg,
                            "model": "RAW",
                            "variant": "raw",
                            **raw_metrics,
                        }
                    )
                    print(f"    ✓ RAW metrics computed")
                    # Save results with checkpoint
                    df_checkpoint = pd.DataFrame(records, columns=metrics_columns)
                    df_checkpoint.to_csv(checkpoint_file, index=False)
                except Exception as e:
                    print(f"    ✗ RAW failed: {e}")
                    continue

            for variant in variants:
                graphst_path = config["graphst_base"].format(variant=variant, hvg=hvg)

                if (group_name, hvg, "GraphST", variant) in processed_items:
                    print(f"    (Skipping GraphST {variant} - already processed)")
                elif os.path.exists(graphst_path):
                    try:
                        graphst_adata = load_adata(graphst_path)
                        graphst_metrics = safe_compute(
                            graphst_adata,
                            config["graphst_cluster_key"],
                            config["graphst_batch_key"],
                            config["graphst_celltype_key"],
                            config["graphst_rep"],
                            f"GraphST_{variant}/{group_name}/hvg{hvg}",
                        )
                        records.append(
                            {
                                "dataset_group": group_name,
                                "hvg": hvg,
                                "model": "GraphST",
                                "variant": variant,
                                **graphst_metrics,
                            }
                        )
                        print(f"    ✓ GraphST {variant} metrics computed")
                        # Save results with checkpoint
                        df_checkpoint = pd.DataFrame(records, columns=metrics_columns)
                        df_checkpoint.to_csv(checkpoint_file, index=False)
                        del graphst_adata
                        gc.collect()
                    except Exception as e:
                        print(f"    ✗ GraphST {variant} failed: {e}")
                else:
                    print(f"    - GraphST {variant} file not found")

                if "staligner_base" in config:
                    staligner_path = config["staligner_base"].format(variant=variant, hvg=hvg)
                else:
                    staligner_path = config["staligner"].get(variant)

                if (group_name, hvg, "STAligner", variant) in processed_items:
                    print(f"    (Skipping STAligner {variant} - already processed)")
                elif staligner_path:
                    if os.path.exists(staligner_path):
                        try:
                            staligner_adata = load_adata(staligner_path)
                            staligner_metrics = safe_compute(
                                staligner_adata,
                                config["staligner_cluster_key"],
                                config["staligner_batch_key"],
                                config["staligner_celltype_key"],
                                config["staligner_rep"],
                                f"STAligner_{variant}/{group_name}",
                            )
                            records.append(
                                {
                                    "dataset_group": group_name,
                                    "hvg": hvg,
                                    "model": "STAligner",
                                    "variant": variant,
                                    **staligner_metrics,
                                }
                            )
                            print(f"    ✓ STAligner {variant} metrics computed")
                            # Save results with checkpoint
                            df_checkpoint = pd.DataFrame(records, columns=metrics_columns)
                            df_checkpoint.to_csv(checkpoint_file, index=False)
                            del staligner_adata
                            gc.collect()
                        except Exception as e:
                            print(f"    ✗ STAligner {variant} failed: {e}")
                    else:
                        print(f"    - STAligner {variant} file not found")
                else:
                    print(f"    - No STAligner {variant} path defined")

            

    output_csv = os.path.join("..", "results", "metrics_dlpfc_hbc_log1p-vs-sct.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records, columns=metrics_columns)  
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Saved metrics summary to {output_csv}")
    print(f"✓ Records processed: {len(records)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
