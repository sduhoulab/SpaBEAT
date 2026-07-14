import sys
sys.path.append("/data_hou/BE/SpaBatch-main/SpaBatch")

import os
import torch
import time
import psutil
import gc
import json
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from adj import main, combine_graph_dict
from train import train_model
from utils import mclust_R, fix_seed
from sklearn.decomposition import PCA

# ====================== BenchmarkTracker ======================
class BenchmarkTracker:
    def __init__(self, method_name):
        self.method_name = method_name
        self.process = psutil.Process(os.getpid())

    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.mem_before = self.process.memory_info().rss / (1024 * 1024)
        self.start_time = time.time()
        print(f"\n[{self.method_name}] Starting core training benchmark...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.train_time = time.time() - self.start_time
        self.mem_used = (self.process.memory_info().rss / (1024 * 1024)) - self.mem_before
        self.gpu_mem = torch.cuda.max_memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0
        print(f"Time: {self.train_time:.2f}s | CPU RAM: {self.mem_used:.2f}MB | GPU Peak: {self.gpu_mem:.2f}MB\n")

    def save_report(self, json_path, adata, embed_key, extra_meta={}):
        results = {
            'method_name': self.method_name,
            'training_time_seconds': self.train_time,
            'memory_usage_mb': self.mem_used,
            'gpu_peak_memory_mb': self.gpu_mem,
            'total_cells': adata.n_obs,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        results.update(extra_meta)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

fix_seed(42)

# ======================================================================
# Task Configuration Switch
# ======================================================================
TASKS = {
    'xe': {'root': '/data_hou/ST_data_new/Xenium_breast1/', 'slices': ['Rep1_outs', 'Rep2_outs'], 'n_domains': 20, 'type': 'xe', 'suffix': 'xe_breast'}
}


# ======================================================================
# Argument Parsing Setup
# ======================================================================
parser = argparse.ArgumentParser(description="Run SpaBatch Benchmark for a specific task.")
parser.add_argument(
    '--task',
    type=str,
    required=True,
    choices=list(TASKS.keys()),
    help="Specify the task to run. Available tasks: " + ", ".join(TASKS.keys())
)
parser.add_argument('--n_domains', type=int, default=None, help='Number of domains for clustering')

args_cli = parser.parse_args()
CURRENT_TASK = args_cli.task
config = TASKS[CURRENT_TASK]

final_n_domains = args_cli.n_domains if args_cli.n_domains is not None else config['n_domains']

data_root = Path(config['root'])
proj_list = config['slices']
save_path = Path('/data_hou/ST_data_new/model-zn/spabatch/results/')
save_path.mkdir(parents=True, exist_ok=True)

print(f"=== Running SpaBatch Benchmark for: {CURRENT_TASK} (n_domains = {final_n_domains}) ===")
celltype_path = os.path.join(data_root, "Cell_Barcode_Type_Matrices.xlsx")

sample_to_sheet = {
    'Rep1_outs': 'Xenium R1 Fig1-5 (supervised)',
    'Rep2_outs': 'Xenium R2 Fig1-5 (supervised)'
}
print("Reading designated Excel sheets...")
all_clusters = set()
sample_cell_type_maps = {}

for proj_name in tqdm(proj_list):
    target_sheet = sample_to_sheet[proj_name]
    print(f"Loading sheet '{target_sheet}' for sample '{proj_name}'...")
    
    try:
        df_meta = pd.read_excel(celltype_path, sheet_name=target_sheet)
        df_meta['Barcode'] = df_meta['Barcode'].astype(str)
        all_clusters.update(df_meta['Cluster'].dropna().unique())
        sample_cell_type_maps[proj_name] = dict(zip(df_meta['Barcode'], df_meta['Cluster']))
        print(f"-> Successfully loaded. Total cell annotations in sheet: {len(df_meta)}")
    except Exception as e:
        raise ValueError(f"Error loading sheet '{target_sheet}' from {celltype_path}. Please check sheet names. Error: {e}")
n_clusters = len(all_clusters)
print(f"Clusters {n_clusters}")

# 1. Automatically load and concatenate datasets
print("Loading data, building graphs, and concatenating...")
for i, proj_name in enumerate(tqdm(proj_list)):
    print(f"Processing {proj_name}...")
    
    cur_sample_path = os.path.join(data_root, proj_name)
    adata_path = os.path.join(cur_sample_path, "cell_feature_matrix.h5")
    adata_tmp = sc.read_10x_h5(adata_path)
    adata_tmp.var_names_make_unique()
    adata_tmp.obs['batch_name'] = proj_name
    
    parquet_path = os.path.join(cur_sample_path, "cells.parquet")
    df_cells = pd.read_parquet(parquet_path)
    df_cells['cell_id'] = df_cells['cell_id'].astype(str)
    df_cells = df_cells.set_index('cell_id')
    common_cells = adata_tmp.obs_names.intersection(df_cells.index)
    print(f"Matched {len(common_cells)} cells between H5 and Parquet.")
    
    adata_tmp = adata_tmp[common_cells].copy()
    x_col = 'x_centroid'
    y_col = 'y_centroid'
    adata_tmp.obsm["spatial"] = df_cells.loc[adata_tmp.obs_names, [x_col, y_col]].values
    
    control_genes = adata_tmp.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
    adata_tmp = adata_tmp[:, ~control_genes].copy()
    
    mapping_dict = sample_cell_type_maps[proj_name]
    adata_tmp.obs.loc[adata_tmp.obs['batch_name'] == proj_name, 'ground_truth'] = adata_tmp.obs_names.map(mapping_dict).astype('category')
    adata_tmp = adata_tmp[~pd.isnull(adata_tmp.obs['ground_truth'])]
    print(adata_tmp.obs[['batch_name', 'ground_truth']].reset_index().head(5))
    
    target_cells = 15000
    if adata_tmp.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata_tmp.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata_tmp.n_obs
        df_obs = adata_tmp.obs.copy()
        stratified_indices = df_obs.groupby('ground_truth', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata_tmp = adata_tmp[stratified_indices].copy()
    # Construct spatial graph using cleaned adata
    graph_dict_tmp = main(adata_tmp, adj_cons_by='coordinate', distType='KNN', k_cutoff=12, rad_cutoff=250)

    # Merge datasets
    if proj_name == proj_list[0]:
        adata = adata_tmp
        graph_dict = graph_dict_tmp
    else:
        var_names = adata.var_names.intersection(adata_tmp.var_names)
        adata_tmp = adata_tmp[:, var_names]
        var_names = adata.var_names.intersection(adata_tmp.var_names)
        adata = adata[:, var_names].copy()         # 加上 .copy()
        adata_tmp = adata_tmp[:, var_names].copy() # 加上 .copy()
        adata = adata.concatenate(adata_tmp, batch_key="concat_batch")
        graph_dict = combine_graph_dict(graph_dict, graph_dict_tmp)

# 2. Preprocessing
adata.layers['count'] = adata.X.toarray()
#sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=5000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
#adata = adata[:, adata.var['highly_variable'] == True]
adata.raw = adata.copy()
sc.pp.scale(adata)
adata.obsm['X_pca'] = PCA(n_components=50, random_state=42).fit_transform(adata.X)

# 3. Benchmark training
with BenchmarkTracker('SpaBatch') as tracker:
    SpaBatch_net = train_model(adata, graph_dict, pre_epochs=500, epochs=1000, mask_rate=0.2)
    SpaBatch_net.train_with_dec(num_aggre=1)
    SpaBatch_feat, q = SpaBatch_net.process()

if not (np.isnan(SpaBatch_feat).any() or np.isinf(SpaBatch_feat).any()):
    adata.obsm['SpaBatch_embed'] = np.ascontiguousarray(SpaBatch_feat, dtype=np.float64)

if 'SpaBatch_embed' in adata.obsm:
    try:
        print(f"Clustering with mclust (n_domains = {final_n_domains})...")
        mclust_R(adata, num_cluster=final_n_domains, used_obsm='SpaBatch_embed')
    except Exception as e:
        print(f"Mclust Error: {e}")

# Assign ground truth to celltype for benchmark metrics
if 'ground_truth' in adata.obs:
    adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
else:
    print("[Warning] ground_truth column not found, biological metrics may fail!")

# ======================================================================
# 4. Save results
# ======================================================================
tracker.save_report(
    json_path=save_path / f"spabatch_benchmark_{config['suffix']}.json",
    adata=adata,
    embed_key='SpaBatch_embed',
    extra_meta={'n_datasets': len(proj_list)}
)

adata.write(save_path / f"multiple_adata_{config['suffix']}_spabatch.h5ad")
print(f"Successfully finished {CURRENT_TASK} and saved to {save_path}/multiple_adata_{config['suffix']}_spabatch.h5ad")
