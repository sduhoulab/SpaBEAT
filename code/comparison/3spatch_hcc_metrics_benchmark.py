import os
import pandas as pd
import numpy as np
import scanpy as sc
import scib
from tqdm import tqdm
import sys
import scipy.sparse as sp
import warnings

sys.path.append("/data_hou/ST_data_new/model-zn/comparison/")
from spatial_metrics import spatialbench

warnings.filterwarnings("ignore")

METHODS = {
    "RAW": {"cluster_key": "mclust", "batch_key": "new_batch", "celltype_key": "celltype", "use_rep": "X_pca"},
    "DeepST": {"cluster_key": "DeepST_refine_domain", "batch_key": "new_batch", "celltype_key": "celltype", "use_rep": "DeepST_embed"},
    "GraphST": {"cluster_key": "domain", "batch_key": "new_batch", "celltype_key": "celltype", "use_rep": "emb_pca"},
    "STAligner": {"cluster_key": "mclust","batch_key": "new_batch","celltype_key": "celltype","use_rep": "STAligner"},
    "SPIRAL": {"cluster_key": "mclust","batch_key": "new_batch","celltype_key": "celltype","use_rep": "spiral"},
   "STitch3D": {"cluster_key": "cluster","batch_key": "new_batch","celltype_key": "celltype","use_rep": "latent"},
    "Spatialign": {"cluster_key": "mclust","batch_key": "new_batch","celltype_key": "celltype","use_rep": "correct"},
    "PRECAST": {"cluster_key": "ident","batch_key": "slice_id","celltype_key": "ground_truth","use_rep": "X_PRECAST"},
    "SpaMask": {"cluster_key": "mclust", "batch_key": "batch_name", "celltype_key": "celltype", "use_rep": "SpaMask_embed"},    
    "SpaCross": {"cluster_key": "mclust","batch_key": "batch_name","celltype_key": "celltype","use_rep": "SpaCross_embed"},
    "SpaBatch": {"cluster_key": "mclust","batch_key": "batch_name","celltype_key": "celltype","use_rep": "SpaBatch_embed"}
    }

DATA_PATHS = {
    "RAW": "/data_hou/ST_data_new/model-zn/data_raw/spatch_hcc_raw_adata1.h5ad",
    "DeepST": "/data_hou/ST_data_new/model-zn/deepst/results/spatch_hcc_deepst_adata1.h5ad",
    "GraphST": "/data_hou/ST_data_new/model-zn/graphst/results1/spatch_hcc_graphst_adata1.h5ad",
    "STAligner": "/data_hou/ST_data_new/model-zn/STAligner/results/spatch_hcc_sta_adata1.h5ad",
    "Spatialign": "/data_hou/ST_data_new/model-zn/spatialign/results/spatch_hcc/spatch_hcc_spa_multiple_adata1.h5ad",
    "SPIRAL": "/data_hou/ZhaoMH/new_model/Spiral/results/spatch_hcc_spiral_adata1.h5ad",
    "STitch3D": "/data_hou/ST_data_new/model-zn/STitch3D/results/spatch_hcc/spatch_hcc_stitch3d_adata1.h5ad",
    "PRECAST": "/data_hou/ST_data_new/model-zn/precast/results/spatch_hcc_seuInt_with_all_spatial.h5ad",
    "SpaMask":  "/data_hou/ST_data_new/model-zn/spamask/results/multiple_adata_spatch_hcc_spamask.h5ad",
    "SpaCross": "/data_hou/ST_data_new/model-zn/spacross/results/multiple_adata_spatch_hcc_spacross.h5ad",
    "SpaBatch": "/data_hou/ST_data_new/model-zn/spabatch/results/multiple_adata_spatch_hcc_spabatch.h5ad"
}

BATCH_METRICS = ["GC", "iLISI", "kBET", "ASW_batch"]
BIO_METRICS = ["SCS", "MoranI", "GearyC", "cLISI", "ASW_domain", "ARI"]
ALL_METRICS = BIO_METRICS + BATCH_METRICS

def clear_memory():
    import gc
    gc.collect()
    print("Memory cleared")

def compute_metrics(adata, config, degree=4, seed=0, top_n=3, safe_mode=True):
    def clean_matrix(mat):
        if sp.issparse(mat):
            mat = mat.astype(np.float32).tocoo(copy=True)
            mat.data = np.nan_to_num(mat.data, nan=0.0, posinf=0.0, neginf=0.0)
            return mat.tocsr()
        else:
            mat = np.array(mat, dtype=np.float32, copy=True)
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            return mat

    if safe_mode:
        adata.X = clean_matrix(adata.X)

    results = {}
    if adata.raw is None:
        adata.raw = adata.copy()
    else:
        adata.raw._X = clean_matrix(adata.raw.X)

    adata_scib = adata.copy()
    adata_raw_clean = adata.raw.to_adata()
    adata_raw_clean.X = clean_matrix(adata_raw_clean.X)

    if 'spatial' in adata.obsm:
        adata_raw_clean.obsm['spatial'] = adata.obsm['spatial']
    elif 'X_spatial' in adata.obsm:
         adata_raw_clean.obsm['spatial'] = adata.obsm['X_spatial']
    
    try:
        sc.pp.neighbors(adata_raw_clean, use_rep='spatial', key_added='spatial')
    except Exception as e:
        print(f"  Warning: Spatial graph construction failed: {e}")

    use_rep = config.get("use_rep", "X_pca")
    if use_rep in adata_scib.obsm:
        adata_scib.obsm[use_rep] = clean_matrix(adata_scib.obsm[use_rep])

    try:
        print("  Computing SCS...")
        results["SCS"] = spatialbench.spatial_coherence_score(adata_raw_clean, annotation_key=config["cluster_key"], degree=degree, seed=seed)
        
        print("  Computing  Moran I & Geary C...")
        if 'count' in adata.layers:
            adata_for_moran = sc.AnnData(
                X=sp.csr_matrix(adata.layers['count'], dtype=float),
                obs=adata.obs.copy(),
                var=adata.var.copy(),
                obsm={'spatial': adata.obsm['spatial'].copy()}
            )
        else:
            adata_for_moran = adata_raw_clean
            if adata_for_moran.X.dtype != float:
                adata_for_moran.X = adata_for_moran.X.astype(float)

        moranI, gearyC = spatialbench.moran_geary_preservation(
            adata_for_moran, celltype_key=config["celltype_key"], top_n=top_n
        )
        results["MoranI"], results["GearyC"] = moranI, gearyC
    except Exception as e:
        results["SCS"], results["MoranI"], results["GearyC"] = np.nan, np.nan, np.nan

    try:
        sc.pp.neighbors(adata_scib, use_rep=use_rep, n_neighbors=15, random_state=seed)
        results["ARI"] = scib.me.ari(adata_scib, cluster_key=config["cluster_key"], label_key=config["celltype_key"])
        results["GC"] = scib.me.graph_connectivity(adata_scib, label_key=config["celltype_key"])
        results["ASW_domain"] = scib.me.silhouette(adata_scib, label_key=config["celltype_key"], embed=use_rep)
        results["ASW_batch"] = scib.me.silhouette_batch(adata_scib, batch_key=config["batch_key"], label_key=config["celltype_key"], embed=use_rep)
        results["iLISI"] = scib.me.ilisi_graph(adata_scib, batch_key=config["batch_key"], type_="embed", use_rep=use_rep)
        results["cLISI"] = scib.me.clisi_graph(adata_scib, label_key=config["celltype_key"], type_="embed", use_rep=use_rep)
        results["kBET"] = scib.me.kBET(adata_scib, batch_key=config["batch_key"], label_key=config["celltype_key"], type_="embed", embed=use_rep)
    except Exception as e:
        print(f"Metrics calculation failed: {e}")
        for k in ["ARI", "GC", "ASW_domain", "ASW_batch", "iLISI", "cLISI", "kBET"]:
            results[k] = np.nan
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark spatch_hcc only")
    parser.add_argument("--method", "-m", type=str, help="Run specific method (e.g., DeepST)")
    args = parser.parse_args()

    results = []
    methods_to_run = {args.method: METHODS[args.method]} if args.method else METHODS

    os.makedirs("/data_hou/ST_data_new/model-zn/comparison/results/individual_models", exist_ok=True)

    for method_name, config in methods_to_run.items():
        if method_name not in DATA_PATHS:
            print(f"Method {method_name} not configured.")
            continue
            
        file_path = DATA_PATHS[method_name]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"\n>>> Processing: {method_name}")
        adata = sc.read_h5ad(file_path)
        metrics = compute_metrics(adata, config)
        metrics["Method"] = method_name
        metrics["Sample"] = "spatch_hcc"
        results.append(metrics)
        del adata
        clear_memory()

    if results:
        df = pd.DataFrame(results)
        df[["Method", "Sample"] + ALL_METRICS].to_csv("/data_hou/ST_data_new/model-zn/comparison/results/individual_models/spatch_hcc_all_metrics0711.csv", index=False)
        print("\nResults saved to /data_hou/ST_data_new/model-zn/comparison/results/individual_models/spatch_hcc_all_metrics0711.csv")

if __name__ == "__main__":
    main()