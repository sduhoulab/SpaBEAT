import os
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing as mp
import sys
sys.path.append('/root/BE/GraphST')


from GraphST import GraphST
import time
import psutil
import gc
import json
import anndata as ad 
import numpy as np
import scipy.sparse as sp

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Starting data preprocessing...")
n_clusters = 14
datasets = ['slice_39', 'slice_44']
file_fold = "/root/ST0507/data/visumST/Visium_converted/" 

save_path = "/root/ST0507/new_model/GraphST/results/"
os.makedirs(save_path, exist_ok=True)

adatas = []
total_cells = 0


for dataset in datasets:  
    print(f"Processing {dataset}...")
    batch_name = dataset.split('/')[-1]
    adata = sc.read_h5ad(os.path.join(file_fold, dataset, f"{batch_name}.h5ad"))
    adata.var_names_make_unique()
    adata.obs['new_batch'] = dataset
   
    
    # Convert to dense matrix to prevent PyG calculation errors
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    # =========================================================================
    
    # Load ground truth if not present in adata
    if 'ground_truth' not in adata.obs:
        metadata_file = os.path.join(file_fold, dataset, 'metadata.csv')
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
    
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    
    # ==================== Modification: Skip HVG for Targeted Panel ====================
    # MERFISH contains a targeted 155 gene panel. HVG selection is unnecessary and will fail.
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # adata = adata[:, adata.var['highly_variable']]
    # ===================================================================================
    
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
    'dataset': 'MERFISH_Mouse1',
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

print("✅ Reviewing adata.obsm keys:", list(adata.obsm.keys()))
print("======= adata.obsm['emb'] details =======")
print("Type :", type(adata.obsm['emb']))
print("Shape:", adata.obsm['emb'].shape)
print("Ndim :", adata.obsm['emb'].ndim)
print("Dtype:", adata.obsm['emb'].dtype)
print(adata.obsm['emb'][:5])
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

adata.write(os.path.join(save_path, "multiple_adata_visiumST.h5ad"))

with open(os.path.join(save_path, "multiple_adata_visumST.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\n🎉 visiumST GraphST integration complete.")