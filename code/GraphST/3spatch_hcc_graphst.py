import os
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing as mp
import sys
sys.path.append('/data_hou/BE/GraphST/')
from GraphST import GraphST
import time
import psutil
import gc
import json
import anndata as ad 

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_clusters = set()
adatas = []
total_cells = 0

print("Starting preprocessing Visium HD data")

hd_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_hcc_hd_ff/adata.h5ad"
hd_adata = sc.read_h5ad(hd_adata_path)
#hd_celltypes = hd_adata.obs['annotation'].dropna().astype(str).unique().tolist()
#hd_celltypes.sort()
#print(f"Total {len(hd_celltypes)} celltypes:") 15
#print(hd_celltypes)
hd_celltypes = hd_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(hd_celltypes)

print("Starting preprocessing Xe data")

xe_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_hcc_xenium/adata.h5ad"
xe_adata = sc.read_h5ad(xe_adata_path)
#xe_celltypes = xe_adata.obs['annotation'].dropna().astype(str).unique().tolist()
#xe_celltypes.sort()
#print(f"Total {len(xe_celltypes)} celltypes:") 17
#print(xe_celltypes)
xe_celltypes = xe_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(xe_celltypes)
n_clusters = len(all_clusters)
print(f"Clusters {n_clusters}")

name_to_adata = {
    'hd': hd_adata,
    'xe': xe_adata
}

for name, adata in name_to_adata.items():
    print(f"Processing {name}...")
    
    adata.var_names_make_unique()
    adata.obs['batch'] = name # Add batch information
    print("--- adata summary ---")
    print(adata)
    print("\n--- Adata Barcode (Obs Names) head ---")
    print(adata.obs_names[:5].tolist())
    
    if name == 'xe':
        control_genes = adata.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
        adata = adata[:, ~control_genes].copy()
        print("--- Filtered adata summary ---")
        print(adata)
    
    adata.obs.loc[adata.obs['batch'] == name, 'ground_truth'] = adata.obs['annotation']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print(adata.obs[['batch', 'ground_truth']].reset_index().head(5))
    
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('ground_truth', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()
    print(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    print(adata)
    adatas.append(adata)
    total_cells += adata.n_obs
    print(f"  {name}: {adata.n_obs} cells, {adata.n_vars} genes")

print(f"Total cells across datasets: {total_cells}")

adata = adatas[0].concatenate(adatas[1:], batch_key='batch')
print(f"Concatenated data shape: {adata.shape}")

# =============== GraphST training ===============

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

model = GraphST.GraphST(adata, datatype='Stereo', device=device, random_seed=50)
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
    'n_datasets': len(name_to_adata),
    'device': str(device),
    'random_seed': 50,
    'hvg_genes': 5000,
    'timestamp': pd.Timestamp.now().isoformat()
}

print("\nContinuing with clustering...")

from GraphST.utils import clustering
tool = 'mclust'  # mclust, leiden, and louvain
if tool == 'mclust':
    clustering(adata, n_clusters, method=tool)
elif tool in ['leiden', 'louvain']:
    clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)

print(f" Batch head {adata.obs['batch'].head()}")

batch_mapping = {
    '0': 'hd',
    '1': 'xe'
}


adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
print(f" New Batch head {adata.obs['new_batch'].head()}")

adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
adata.write("/data_hou/ST_data_new/model-zn/graphst/results1/spatch_hcc_graphst_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/graphst/results1/spatch_hcc_graphst_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)