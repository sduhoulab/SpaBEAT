import sys
sys.path.append("/data_hou/BE/SpaBatch-main/SpaBatch/")

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
    'spatch_ov': {'root': None, 'slices': ['hd', 'xe'], 'n_domains': 14, 'type': 'spatch', 'suffix': 'spatch_ov'}
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


# ======================================================================

#data_root = Path(config['root'])
proj_list = config['slices']
save_path = Path('/data_hou/ZhaoMH/new_model/SpaBatch-main/Results/')
save_path.mkdir(parents=True, exist_ok=True)

print(f"=== Running SpaBatch Benchmark for: {CURRENT_TASK} (n_domains = {final_n_domains}) ===")

all_clusters = set()

print("Starting preprocessing Visium HD data")
hd_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_ov_hd_ff/adata.h5ad"
hd_adata = sc.read_h5ad(hd_adata_path)
hd_celltypes = hd_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(hd_celltypes)

print("Starting preprocessing Xe data")
xe_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_ov_xenium/adata.h5ad"
xe_adata = sc.read_h5ad(xe_adata_path)
xe_celltypes = xe_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(xe_celltypes)
n_clusters = len(all_clusters)
print(f"Clusters {n_clusters}")
name_to_adata = {
    'hd': hd_adata,
    'xe': xe_adata
}
# 1. Automatically load and concatenate datasets
print("Loading data, building graphs, and concatenating...")
adata_list = []
for name, adata_tmp in name_to_adata.items():
    print(f"Processing {name}...")
    
    adata_tmp.var_names_make_unique()
    adata_tmp.obs['batch_name'] = name

    if name == 'xe':
        control_genes = adata_tmp.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
        adata_tmp = adata_tmp[:, ~control_genes].copy()
    
    adata_tmp.obs.loc[adata_tmp.obs['batch_name'] == name, 'ground_truth'] = adata_tmp.obs['annotation'].astype('category')
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
    adata_list.append(adata_tmp)

import anndata
common_genes = adata_list[0].var_names
for ad in adata_list[1:]:
    common_genes = common_genes.intersection(ad.var_names)
print(f"Overlap genes: {len(common_genes)}")
for i in range(len(adata_list)):
    adata_list[i] = adata_list[i][:, common_genes].copy()

adata = anndata.concat(
    adata_list, 
    keys=proj_list, 
    label="concat_batch"
)
print(f"Concatenated adata shape: {adata.shape}")

sc.pp.filter_cells(adata, min_genes=20)
sc.pp.filter_genes(adata, min_cells=3)
adata.layers['count'] = adata.X.toarray()
# Construct spatial graph using cleaned adata
graph_dict = main(adata, adj_cons_by='coordinate', distType='KNN', k_cutoff=12, rad_cutoff=250)

# 2. Preprocessing
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=5000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, adata.var['highly_variable'] == True]
adata.raw = adata.copy()
#sc.pp.scale(adata)
###################
sc.pp.scale(adata, max_value=10)
#############################
adata.obsm['X_pca'] = PCA(n_components=50, random_state=42).fit_transform(adata.X)

# 3. Benchmark training
with BenchmarkTracker('SpaBatch') as tracker:
    SpaBatch_net = train_model(adata, graph_dict, pre_epochs=500, epochs=1000, mask_rate=0.2)
    SpaBatch_net.train_with_dec(num_aggre=1)
    SpaBatch_feat, q = SpaBatch_net.process()
    
    print("feature NaN:", np.isnan(SpaBatch_feat).any())
    print("feature Inf:", np.isinf(SpaBatch_feat).any())
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
#tracker.save_report(
#    json_path=save_path / f"spabatch_benchmark_{config['suffix']}.json",
#    adata=adata,
#    embed_key='SpaBatch_embed',
#    extra_meta={'n_datasets': len(proj_list)}
#)

# ======================================================================
# 4. Save results
# ======================================================================
tracker.save_report(
    json_path=save_path / f"spabatch_benchmark_{config['suffix']}.json",
    adata=adata,
    embed_key='SpaBatch_embed',
    extra_meta={'n_datasets': len(proj_list)}
)

# =============== 新增代码：清理 uns 中导致保存失败的非规范 Key ===============
print("Cleaning adata.uns before saving...")
if 'Spatial_graphList' in adata.uns:
    del adata.uns['Spatial_graphList']

# 稳妥起见，把 uns 里面所有非 str 类型的 key 的元素都清理掉或转换掉
keys_to_delete = []
for k, v in adata.uns.items():
    if isinstance(v, dict):
        if any(not isinstance(sub_k, str) for sub_k in v.keys()):
            keys_to_delete.append(k)

for k in keys_to_delete:
    del adata.uns[k]
# =============================================================================


adata.write(save_path / f"multiple_adata_{config['suffix']}_spabatch.h5ad")
print(f"Successfully finished {CURRENT_TASK} and saved to {save_path}/multiple_adata_{config['suffix']}_spabatch.h5ad")
