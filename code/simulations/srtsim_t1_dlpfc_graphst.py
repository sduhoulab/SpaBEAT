import os
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing as mp
import sys
sys.path.append('/data_hou/BE/GraphST')

from GraphST import GraphST
import time
import psutil
import gc
import json
import anndata as ad
import numpy as np
import scipy.sparse as sp

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run_graphst_integration(ref_slice, target_slice, data_path, base_save_path, n_clusters=7):
    datasets = [ref_slice, target_slice]
    pair_name = f"{ref_slice}_vs_{target_slice}"
    
    save_path = base_save_path
    os.makedirs(save_path, exist_ok=True)
    
    adatas = []
    total_cells = 0

    print(f"\nStarting data preprocessing for {pair_name}...")
    
    for dataset in datasets:
        print(f"Processing {dataset}...")
        batch_name = dataset.split('/')[-1]
        adata = sc.read_h5ad(os.path.join(data_path, dataset, f"{batch_name}.h5ad"))
        adata.var_names_make_unique()
        adata.obs['new_batch'] = dataset
        
        if sp.issparse(adata.X):
            adata.X = adata.X.toarray()
        
        if 'ground_truth' not in adata.obs:
            metadata_file = os.path.join(data_path, dataset, 'metadata.csv')
            if os.path.exists(metadata_file):
                print(f"[Info] No ground_truth found in {dataset}, loading from metadata.csv...")
                df_meta = pd.read_csv(metadata_file, index_col=0)
                
                possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label', 'Cell_class']
                for col in possible_cols:
                    if col in df_meta.columns:
                        adata.obs['ground_truth'] = adata.obs_names.map(df_meta[col])
                        print(f"  [✔] Successfully loaded labels from column: '{col}'")
                        break
                else:
                    print(f"  [Warning] metadata.csv found but no valid label columns detected!")
            else:
                print(f"  [Error] No labels in adata and metadata.csv not found for {dataset}!")
        
        if 'ground_truth' in adata.obs.columns:
            adata = adata[~pd.isnull(adata.obs['ground_truth'])]
        
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        
        adatas.append(adata)
        total_cells += adata.n_obs
        print(f"  {dataset}: {adata.n_obs} cells, {adata.n_vars} genes")

    print(f"Total cells across datasets: {total_cells}")

    adata = adatas[0].concatenate(adatas[1:], batch_key='batch')
    print(f"Concatenated data shape: {adata.shape}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_before = get_memory_usage()
    training_start_time = time.time()

    model = GraphST.GraphST(adata, device=device, random_seed=50)
    print("Training GraphST model...")
    adata = model.train()

    training_end_time = time.time()
    memory_after = get_memory_usage()
    training_time = training_end_time - training_start_time
    memory_used = memory_after - memory_before

    benchmark_results = {
        'method_name': 'GraphST',
        'dataset': pair_name,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'training_time_hours': training_time / 3600,
        'memory_usage_mb': memory_used,
        'memory_usage_gb': memory_used / 1024,
        'total_cells': total_cells,
        'final_cells': adata.n_obs,
        'total_genes': adata.n_vars,
        'embedding_dim': adata.obsm['emb'].shape[1],
        'n_datasets': len(datasets),
        'device': str(device),
        'random_seed': 50,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    print("\nContinuing with clustering...")

    print("Reviewing adata.obsm keys:", list(adata.obsm.keys()))
    print("======= adata.obsm['emb'] details =======")
    print("Type :", type(adata.obsm['emb']))
    print("Shape:", adata.obsm['emb'].shape)
    print("Ndim :", adata.obsm['emb'].ndim)
    print("Dtype:", adata.obsm['emb'].dtype)
    print("======================================================")

    adata.obsm['emb'] = adata.obsm['emb'].astype(np.float64)

    from GraphST.utils import clustering
    tool = 'mclust'
    if tool == 'mclust':
        clustering(adata, n_clusters, method=tool)
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)

    if 'ground_truth' in adata.obs.columns:
        adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')

    adata.write(os.path.join(save_path, f"multiple_adata_{pair_name}.h5ad"))

    with open(os.path.join(save_path, f"graphst_benchmark_{pair_name}.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n🎉 Task {pair_name} GraphST integration finished! Results saved to: {save_path}\n")
    
    del model, adata, adatas
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    DATA_PATH = "/data_hou/ZhaoMH/situmulation_results/data_model/1_1DLPFC_InterSlice/SRTsim_Converted/"
    BASE_SAVE_PATH = "/data_hou/ZhaoMH/new_model/GraphST/results/SRTsim_dlpfc_t1/"
    
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    
    REF_SLICE = "Sim_Slice_1"
    
    N_CLUSTERS = 7
    
    for i in range(2, 7):
        target_slice = f"Sim_Slice_{i}"
        
        print("="*60)
        print(f"🚀 Starting GraphST consecutive slice alignment task: {REF_SLICE} + {target_slice}")
        print("="*60)
        
        try:
            run_graphst_integration(
                ref_slice=REF_SLICE,
                target_slice=target_slice,
                data_path=DATA_PATH,
                base_save_path=BASE_SAVE_PATH,
                n_clusters=N_CLUSTERS
            )
        except Exception as e:
            print(f"❌ Task {REF_SLICE}_vs_{target_slice} failed. Error message: {str(e)}")
            continue
            
    print("All slice pair integration tasks have been completed!")