import os
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing as mp
import sys
sys.path.append('/data/ZhaoMH/ST0507/GraphST-main')

from GraphST import GraphST
import time
import psutil
import gc
import json
import anndata as ad 
import numpy as np

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Starting data preprocessing...")
n_clusters = 5
datasets = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
file_fold = "/root/ST0507/data/Her2_tumor_converted/A/" 
adatas = []
total_cells = 0

# ==================== New 1: Find common genes across all slices ====================
common_genes = None
for sample in datasets:
    temp_batch = sample.split('/')[-1]
    temp_adata = sc.read_h5ad(os.path.join(file_fold, sample, f"{temp_batch}.h5ad"))
    if common_genes is None:
        common_genes = set(temp_adata.var_names)
    else:
        common_genes &= set(temp_adata.var_names)
common_genes = list(common_genes)
print(f"✅ Found {len(common_genes)} common genes")
# =========================================================================

for dataset in datasets:  
    print(f"Processing {dataset}...")
    batch_name = dataset.split('/')[-1]
    adata = sc.read_h5ad( os.path.join(file_fold, dataset, f"{batch_name}.h5ad") )
    adata.var_names_make_unique()
    adata.obs['new_batch'] = dataset
   
    # ==================== Subset adata to common genes ====================
    adata = adata[:, common_genes].copy()
    # =========================================================================
    
    # Load ground truth if not present in adata
    if 'ground_truth' not in adata.obs:
        metadata_file = os.path.join(file_fold, dataset, 'metadata.csv')
        if os.path.exists(metadata_file):
            print(f"[Info] No ground_truth found in {dataset}, loading from metadata.csv...")
            df_meta = pd.read_csv(metadata_file, index_col=0)
            
            # Check all possible label columns for compatibility
            possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label']
            for col in possible_cols:
                if col in df_meta.columns:
                    # Map labels using cell barcodes to avoid order mismatch
                    adata.obs['ground_truth'] = adata.obs_names.map(df_meta[col])
                    print(f"  [✔] Successfully loaded labels from column: '{col}'")
                    break
            else:
                print(f"  [Warning] metadata.csv found but no valid label columns detected!")
        else:
            print(f"  [Error] No labels in adata and metadata.csv not found for {dataset}!")
    
    # ==================================================================
    
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

# =============== GraphST training ===============

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

# =============== Saving benchmarking results ===============
benchmark_results = {
    'method_name': 'GraphST',
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
    'hvg_genes': 5000,
    'timestamp': pd.Timestamp.now().isoformat()
}

print("\nContinuing with clustering...")


adata.obsm['emb'] = adata.obsm['emb'].astype(np.float64)

from GraphST.utils import clustering
tool = 'mclust'  # mclust, leiden, and louvain
if tool == 'mclust':
    clustering(adata, n_clusters, method=tool)
elif tool in ['leiden', 'louvain']:
    clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)


adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
adata.write("/root/ST0507/new_model/GraphST/results/multiple_adata_her2_A.h5ad")

with open("/root/ST0507/new_model/GraphST/results/multiple_adata_her2_A.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)
