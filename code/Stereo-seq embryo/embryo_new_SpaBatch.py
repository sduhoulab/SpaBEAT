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
import scipy.sparse as sp

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
    'merfish1': {'root': '/data/ZhaoMH/ST0507/RAW_SLICE/merfish_mouse1/data/MERFISH_converted/', 'slices': ['Slice_7', 'Slice_8', 'Slice_9', 'Slice_10', 'Slice_11'], 'n_domains': 8, 'type': 'merfish1', 'suffix': 'merfish1'},
    'starmap': {'root': '/data/ZhaoMH/ST0507/RAW_SLICE/STARmap_mouse/STARMAP_converted/', 'slices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'n_domains': 5, 'type': 'starmap', 'suffix': 'starmap'},
    'PFC': {'root': '/root/ST0507/data/starmap_PFC/STARMAP_converted/', 'slices': ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control'], 'n_domains': 4, 'type': 'PFC', 'suffix': 'PFC'},
    'embyro': {'root': '/data_hou/ZhaoMH/data/stereo_embryo/Embryo_converted/', 'slices': ['Slice_5', 'Slice_6', 'Slice_7', 'Slice_8', 'Slice_9'], 'n_domains': 25, 'type': 'embryo', 'suffix': 'embryo'}
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

data_root = Path(config['root'])
proj_list = config['slices']
save_path = Path('/data_hou/ZhaoMH/new_model/SpaBatch-main/Results')
save_path.mkdir(parents=True, exist_ok=True)

print(f"=== Running SpaBatch Benchmark for: {CURRENT_TASK} (n_domains = {final_n_domains}) ===")


# 1. Automatically load and concatenate datasets
print("Loading data, downsampling, building graphs, and concatenating...")
np.random.seed(42)

for i, proj_name in enumerate(tqdm(proj_list)):
    batch_name = proj_name.split('/')[-1] 
    
    file_path = data_root / proj_name / f"{batch_name}.h5ad"
    metadata_file = data_root / proj_name / 'metadata.csv'
    
    adata_tmp = sc.read_h5ad(file_path)
    adata_tmp.var_names_make_unique()
    adata_tmp.obs['batch_name'] = batch_name

    adata_tmp = adata_tmp[~pd.isnull(adata_tmp.obs['ground_truth'])].copy()

    keep_idx = np.random.choice(
        adata_tmp.n_obs,
        size=int(adata_tmp.n_obs * 0.5),
        replace=False
    )
    keep_idx = np.sort(keep_idx)
    adata_tmp = adata_tmp[keep_idx].copy()

    graph_dict_tmp = main(adata_tmp, adj_cons_by='coordinate', distType='KNN', k_cutoff=12, rad_cutoff=250)

    if proj_name == proj_list[0]:
        adata = adata_tmp
        graph_dict = graph_dict_tmp
    else:
        var_names = adata.var_names.intersection(adata_tmp.var_names)
        adata_tmp = adata_tmp[:, var_names]
        var_names = adata.var_names.intersection(adata_tmp.var_names)
        adata = adata[:, var_names].copy()
        adata_tmp = adata_tmp[:, var_names].copy()
        adata = adata.concatenate(adata_tmp, batch_key="concat_batch")
        
        graph_dict = combine_graph_dict(graph_dict, graph_dict_tmp)



# 2. Preprocessing
adata.layers['count'] = adata.X.toarray()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=5000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, adata.var['highly_variable'] == True]
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