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

file_fold = '/data_hou/ST_data_new/Xenium_breast1/'
datasets = ['Rep1_outs', 'Rep2_outs']
celltype_path = os.path.join(file_fold, "Cell_Barcode_Type_Matrices.xlsx")

sample_to_sheet = {
    'Rep1_outs': 'Xenium R1 Fig1-5 (supervised)',
    'Rep2_outs': 'Xenium R2 Fig1-5 (supervised)'
}
print("Reading designated Excel sheets...")
all_clusters = set()
sample_cell_type_maps = {}

for dataset in datasets:
    target_sheet = sample_to_sheet[dataset]
    print(f"Loading sheet '{target_sheet}' for sample '{dataset}'...")
    
    try:
        df_meta = pd.read_excel(celltype_path, sheet_name=target_sheet)
        df_meta['Barcode'] = df_meta['Barcode'].astype(str)
        all_clusters.update(df_meta['Cluster'].dropna().unique())

        sample_cell_type_maps[dataset] = dict(zip(df_meta['Barcode'], df_meta['Cluster']))
        print(f"-> Successfully loaded. Total cell annotations in sheet: {len(df_meta)}")
    except Exception as e:
        raise ValueError(f"Error loading sheet '{target_sheet}' from {celltype_path}. Please check sheet names. Error: {e}")

n_clusters = len(all_clusters)
print(f"Clusters {n_clusters}")
adatas = []
total_cells = 0

for dataset in datasets:  
    print(f"Processing {dataset}...")
    cur_sample_path = os.path.join(file_fold, dataset)
    
    adata_path = os.path.join(cur_sample_path, "cell_feature_matrix.h5")
    adata = sc.read_10x_h5(adata_path)
    adata.var_names_make_unique()
    adata.obs['batch'] = dataset  # Add batch information
    print("--- adata summary ---")
    print(adata)
    print("\n--- Adata Barcode (Obs Names) head ---")
    print(adata.obs_names[:5].tolist())
    
    parquet_path = os.path.join(cur_sample_path, "cells.parquet")
    df_cells = pd.read_parquet(parquet_path)
    df_cells['cell_id'] = df_cells['cell_id'].astype(str)
    df_cells = df_cells.set_index('cell_id')
    common_cells = adata.obs_names.intersection(df_cells.index)
    print(f"Matched {len(common_cells)} cells between H5 and Parquet.")
    
    adata = adata[common_cells].copy()
    x_col = 'x_centroid'
    y_col = 'y_centroid'
    adata.obsm["spatial"] = df_cells.loc[adata.obs_names, [x_col, y_col]].values
    #adata.obs["imagecol"] = adata.obsm["spatial"][:, 0]
    #adata.obs["imagerow"] = adata.obsm["spatial"][:, 1]
    adata.obs["array_col"] = adata.obsm["spatial"][:, 0]
    adata.obs["array_row"] = adata.obsm["spatial"][:, 1]
    print("--- Merge adata summary ---")
    print(adata)
    
    control_genes = adata.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
    adata = adata[:, ~control_genes].copy()
    print("--- Filtered adata summary ---")
    print(adata)
    
    mapping_dict = sample_cell_type_maps[dataset]
    adata.obs.loc[adata.obs['batch'] == dataset, 'ground_truth'] = adata.obs_names.map(mapping_dict)
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print(adata.obs[['batch', 'ground_truth']].head(5))
    
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
    'hvg_genes': adata.n_vars,
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
    '0': 'Rep1',
    '1': 'Rep2'
}


adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
print(f" New Batch head {adata.obs['new_batch'].head()}")

adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
adata.write("/data_hou/ST_data_new/model-zn/graphst/results1/xe_breast_rep1-rep2_graphst_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/graphst/results1/xe_breast_rep1-rep2_graphst_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)