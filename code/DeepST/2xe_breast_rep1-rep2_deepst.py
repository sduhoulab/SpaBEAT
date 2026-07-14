import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data_hou/BE/DeepST/DeepST')
import os 
from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import community as louvain
import pandas as pd
import anndata as ad 
import time
import psutil
import gc
import json
import torch

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

file_fold = '/data_hou/ST_data_new/Xenium_breast1/'
data_name_list = ['Rep1_outs', 'Rep2_outs']
save_path = "/data_hou/ST_data_new/model-zn/deepst/results/" 
celltype_path = os.path.join(file_fold, "Cell_Barcode_Type_Matrices.xlsx")

sample_to_sheet = {
    'Rep1_outs': 'Xenium R1 Fig1-5 (supervised)',
    'Rep2_outs': 'Xenium R2 Fig1-5 (supervised)'
}
print("Reading designated Excel sheets...")
all_clusters = set()
sample_cell_type_maps = {}

for dataname in data_name_list:
    target_sheet = sample_to_sheet[dataname]
    print(f"Loading sheet '{target_sheet}' for sample '{dataname}'...")
    
    try:
        df_meta = pd.read_excel(celltype_path, sheet_name=target_sheet)
        df_meta['Barcode'] = df_meta['Barcode'].astype(str)
        all_clusters.update(df_meta['Cluster'].dropna().unique())

        sample_cell_type_maps[dataname] = dict(zip(df_meta['Barcode'], df_meta['Cluster']))
        print(f"-> Successfully loaded. Total cell annotations in sheet: {len(df_meta)}")
    except Exception as e:
        raise ValueError(f"Error loading sheet '{target_sheet}' from {celltype_path}. Please check sheet names. Error: {e}")

n_domains = len(all_clusters)
print(f"all clusters {n_domains}")

deepen = run(save_path = save_path, 
	task = "Integration",
	pre_epochs = 500, 
	epochs = 500, 
	use_gpu = True,
)

augement_data_list = []
graph_list = []
total_cells = 0

for dataname in data_name_list:  
    print(f"Processing {dataname}...")
    
    cur_sample_path = os.path.join(file_fold, dataname)
    adata_path = os.path.join(cur_sample_path, "cell_feature_matrix.h5")
    adata = sc.read_10x_h5(adata_path)
    adata.var_names_make_unique()
    adata.obs['new_batch'] = dataname
    
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
    
    mapping_dict = sample_cell_type_maps[dataname]
    adata.obs.loc[adata.obs['new_batch'] == dataname, 'ground_truth'] = adata.obs_names.map(mapping_dict)
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print(adata.obs[['new_batch', 'ground_truth']].reset_index().head(5))
    
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('ground_truth', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()
    
    adata = deepen._get_augment(adata, spatial_type="BallTree", use_morphological = False)
    
    total_cells += adata.n_obs
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
    graph_list.append(graph_dict)
    augement_data_list.append(adata)

print(f"data_name_list length: {len(data_name_list)}")
print(f"graph_list length: {len(graph_list)}")

torch.cuda.empty_cache()

multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list, data_name_list=data_name_list, graph_list=graph_list)
data = deepen._data_process(multiple_adata, pca_n_comps=200)

# =============== DeepST training ===============
print("Starting core training benchmarking...")

gc.collect()
torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training DeepST model...")
deepst_embed = deepen._fit(
    data = data,
    graph_dict = multiple_graph,
    domains = multiple_adata.obs["batch"].values,
    n_domains = len(data_name_list)
)

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before

print("Training completed!")

# =============== Saving benchmarking results ===============
benchmark_results = {
    'method_name': 'DeepST',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': multiple_adata.n_obs,
    'total_genes': multiple_adata.n_vars,
    'embedding_dim': deepst_embed.shape[1],
    'n_datasets': len(data_name_list),
    'pre_epochs': 500,
    'epochs': 500,
    'timestamp': pd.Timestamp.now().isoformat()
}

multiple_adata.obsm["DeepST_embed"] = deepst_embed
multiple_adata = deepen._get_cluster_data(multiple_adata, n_domains=n_domains, priori = True)
multiple_adata.obs['celltype'] = multiple_adata.obs['ground_truth'].astype('category')
multiple_adata.write("/data_hou/ST_data_new/model-zn/deepst/results/xe_breast_rep1-rep2_deepst_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/deepst/results/xe_breast_rep1-rep2_deepst_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

