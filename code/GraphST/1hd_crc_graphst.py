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

print("Starting data preprocessing...")
n_clusters = 3
datasets = ['P1CRC', 'P2CRC', 'P5CRC']
file_fold = '/data_hou/ST_data_new/Visium_crc/'
file_bin = 'binned_outputs/square_008um/'
file_meta = 'HumanColonCancer_VisiumHD/MetaData/'

adatas = []
total_cells = 0

for dataset in datasets:  
    print(f"Processing {dataset}...")
    adata_path = os.path.join(file_fold, dataset, file_bin)
    adata = sc.read_visium(adata_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    adata.obs['batch'] = dataset  # Add batch information
    print("--- adata summary ---")
    print(adata)
    print("\n--- Adata Barcode (Obs Names) head ---")
    print(adata.obs_names[:5].tolist())
    
    df_meta_path = os.path.join(file_fold, file_meta)
    df_meta = pd.read_parquet(df_meta_path + dataset + '_Metadata.parquet')
    print(f"meta shape (rows, columns): {df_meta.shape}")
    print("\n--- df_meta head ---")
    print(df_meta.head())
    
    mapping_dict = dict(zip(df_meta['barcode'], df_meta['Periphery']))
    adata.obs.loc[adata.obs['batch'] == dataset, 'ground_truth'] = adata.obs_names.map(mapping_dict)
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
    'n_datasets': len(datasets),
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
    '0': 'P1CRC',
    '1': 'P2CRC',
    '2': 'P5CRC'
}

adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
print(f" New Batch head {adata.obs['new_batch'].head()}")

adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
adata.write("/data_hou/ST_data_new/model-zn/graphst/results/hd_crc_graphst_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/graphst/results/hd_crc_graphst_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)