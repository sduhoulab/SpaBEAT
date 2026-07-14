import scanpy as sc
import pandas as pd
import os

file_fold = '/data_hou/ST_data_new/Xenium_breast1/'
datasets = ['Rep1_outs', 'Rep2_outs']
save_path = "/data_hou/ST_data_new/model-zn/spatialign/results/xe_breast/" 
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
        
        df_meta['Cluster'] = df_meta['Cluster'].fillna("unknown")
        sample_cell_type_maps[dataset] = dict(zip(df_meta['Barcode'], df_meta['Cluster']))
        print(f"-> Successfully loaded. Total cell annotations in sheet: {len(df_meta)}")
    except Exception as e:
        raise ValueError(f"Error loading sheet '{target_sheet}' from {celltype_path}. Please check sheet names. Error: {e}")

data_list = []
Batch_list = []

####scanpy=1.9.1 preprocessing
for dataset in datasets:
    cur_sample_path = os.path.join(file_fold, dataset)
    h5_path = os.path.join(cur_sample_path, "cell_feature_matrix.h5")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()  
    
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
    
    control_genes = adata.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
    adata = adata[:, ~control_genes].copy()
    
    current_map = sample_cell_type_maps[dataset]
    adata.obs['celltype'] = adata.obs_names.map(current_map).astype('category')
    adata = adata[adata.obs['celltype'] != 'unknown']
    
    adata.X = adata.X.astype('float32')
    if 'spatial' in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['spatial'].astype('float32')
    min_gene = 20
    min_cell = 20
    sc.pp.filter_cells(adata, min_genes=min_gene)
    sc.pp.filter_genes(adata, min_cells=min_cell)
    
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('celltype', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()
    
    sc.pp.normalize_total(adata, target_sum=1e4)  
    sc.pp.log1p(adata)
    h5ad_path = os.path.join(save_path, f"{dataset}.h5ad")
    adata.write_h5ad(h5ad_path)  
    data_list.append(h5ad_path)
    Batch_list.append(adata)
    print(f"Saved {h5ad_path}")


####  conda activate Spatialign   
import sys
sys.path.append('/data_hou/BE/Spatialign/')
import os
import scanpy as sc
from spatialign import Spatialign
from warnings import filterwarnings
from anndata import AnnData
import h5py
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
filterwarnings("ignore")
import torch
torch.set_default_dtype(torch.float32)
import time
import psutil
import gc
import json
from sklearn.mixture import GaussianMixture

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("Starting data preprocessing...")

data_list = [
    '/data_hou/ST_data_new/model-zn/spatialign/results/xe_breast/Rep1_outs.h5ad',
    '/data_hou/ST_data_new/model-zn/spatialign/results/xe_breast/Rep2_outs.h5ad'
]
dataset_names = ['Rep1_outs', 'Rep2_outs']

print("Loading datasets for cell counting...")
total_cells = 0
for i, data_path in enumerate(data_list):
    temp_adata = sc.read_h5ad(data_path)
    cells_count = temp_adata.n_obs
    total_cells += cells_count
    print(f"  {dataset_names[i]}: {cells_count} cells")
    del temp_adata  

print(f"Total cells across datasets: {total_cells}")

model = Spatialign(
    *data_list,
    batch_key='batch',
    is_norm_log=True,
    is_scale=False,
    n_neigh=15,
    is_undirected=True,
    latent_dims=100,
    seed=42,
    gpu=0,
    save_path="/data_hou/ST_data_new/model-zn/spatialign/results/xe_breast/" ,
    is_verbose=False
)
raw_merge = AnnData.concatenate(*model.dataset.data_list)
print(raw_merge.n_obs)
print(raw_merge.obs_names.is_unique)
# =============== spatiAlign training ===============
print("\nStarting core training benchmarking...")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training spatialign model...")
model.train(0.05, 1, 0.1)
model.alignment()

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before
print("Training completed!")

correct1 = sc.read_h5ad("/data_hou/ST_data_new/model-zn/spatialign/results/xe_breast/res/correct_data0.h5ad")
correct2 = sc.read_h5ad("/data_hou/ST_data_new/model-zn/spatialign/results/xe_breast/res/correct_data1.h5ad")

merge_data = correct1.concatenate(correct2)
print(merge_data.n_obs)
print(merge_data.obs_names.is_unique)

# =============== Saving benchmarking results ===============
benchmark_results = {
    'method_name': 'Spatialign',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': merge_data.n_obs,
    'total_genes': merge_data.n_vars,
    'embedding_dim': merge_data.obsm["correct"].shape[1],
    'n_datasets': len(data_list),
    'datasets': dataset_names,
    'device': 'GPU 0',
    'random_seed': 42,
    'latent_dims': 100,
    'timestamp': pd.Timestamp.now().isoformat()
}

batch_mapping = {
    '0': 'Rep1_outs',
    '1': 'Rep2_outs'
}

merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)
merge_data.obs['new_batch'] = merge_data.obs['new_batch'].astype('category')
merge_data = merge_data[~merge_data.obs['celltype'].isna()]
merge_data.X = np.nan_to_num(merge_data.X, nan=0.0)

print("Performing clustering...")
sc.pp.scale(merge_data)
X = merge_data.obsm['correct']
n_components = len(all_clusters)
gmm = GaussianMixture(n_components=n_components, random_state=42)
merge_data.obs['mclust'] = gmm.fit_predict(X)
merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")
merge_data.write("/data_hou/ST_data_new/model-zn/spatialign/results/xe_breast/xe_breast_Rep1-2_spa_multiple_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/spatialign/results/xe_breast/xe_breast_Rep1-2_spa_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)
