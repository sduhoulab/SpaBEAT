import sys
sys.path.append("/data_hou/BE/SpaCross-main/")

import warnings
warnings.filterwarnings("ignore")

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
from sklearn.decomposition import PCA
import yaml

import SpaCross as TOOLS

import random
import torch
from torch.backends import cudnn

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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


# ====================== Task Configuration ======================
TASKS = {
    'hd': {'root': '/data_hou/ST_data_new/Visium_crc/', 'slices': ['P1CRC', 'P2CRC', 'P5CRC'], 'n_domains': 3, 'type': 'hd', 'suffix': 'hd_crc'}
}

# ====================== Argument Parsing Setup ======================
parser = argparse.ArgumentParser(description="Run SpaCross Benchmark for a specific task.")
parser.add_argument(
    '--task',
    type=str,
    required=True,
    choices=list(TASKS.keys()),
    help="Specify the task to run. Available tasks: " + ", ".join(TASKS.keys())
)
parser.add_argument('--n_domains', type=int, default=None, help='Number of domains for clustering')

args = parser.parse_args()
CURRENT_TASK = args.task
config_task = TASKS[CURRENT_TASK]

# Determine final n_domains: use CLI argument if provided, otherwise use the value from TASKS
final_n_domains = args.n_domains if args.n_domains is not None else config_task['n_domains']

data_root = Path(config_task['root'])
file_bin = 'binned_outputs/square_008um/'
file_meta = 'HumanColonCancer_VisiumHD/MetaData/'

slices_list = config_task['slices']
save_path = Path('/data_hou/ST_data_new/model-zn/spacross/results')
save_path.mkdir(parents=True, exist_ok=True)

print(f"=== Running SpaCross Benchmark for: {CURRENT_TASK} (n_domains = {final_n_domains}) ===")

# Load model configuration
config_type = config_task['type'].upper() 
yaml_dir = Path('/data_hou/BE/SpaCross-main/Config/')
expected_yaml = yaml_dir / f"{config_type}.yaml"
default_yaml = yaml_dir / "DLPFC.yaml"

if expected_yaml.exists():
    print(f"Loading specific model config from: {expected_yaml.name}")
    yaml_to_load = expected_yaml
else:
    print(f"{expected_yaml.name} not found, loading DLPFC.yaml")
    yaml_to_load = default_yaml

with open(yaml_to_load, 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

# ====================== 1. Data Loading and Alignment ======================
Batch_list = []
for proj_name in tqdm(slices_list):

    print(f"Processing {proj_name}...")
    adata_path = os.path.join(data_root, proj_name, file_bin)
    adata = sc.read_visium(adata_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    adata.obs['slice_id'] = proj_name
    adata.obs['batch_name'] = proj_name
    
    df_meta_path = os.path.join(data_root, file_meta)
    df_meta = pd.read_parquet(df_meta_path + proj_name + '_Metadata.parquet')
    
    mapping_dict = dict(zip(df_meta['barcode'], df_meta['Periphery']))
    adata.obs.loc[adata.obs['batch_name'] == proj_name, 'ground_truth'] = adata.obs_names.map(mapping_dict).astype('category')
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print(adata.obs[['batch_name', 'ground_truth']].reset_index().head(5))
    adata.obs['Ground Truth'] = adata.obs['ground_truth']
    
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('ground_truth', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()
    # 4. Preprocessing
    adata.layers['count'] = adata.X.toarray()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    Batch_list.append(adata)
    

# Perform ICP alignment using SpaCross
print("Performing ICP alignment...")
Batch_list = TOOLS.align_spots(Batch_list, method='icp', data_type="ST", plot=True)
adata, edge_index = TOOLS.graph_construction3D(Batch_list, section_ids=slices_list, k_cutoff=12, mode='KNN')
adata.obsm['X_pca'] = PCA(n_components=50, random_state=42).fit_transform(adata.X)

# ====================== 2. Model Training and Benchmark ======================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize model with dynamic number of domains
net = TOOLS.SC_pipeline(adata, edge_index=edge_index, num_clusters=final_n_domains,
                        device=device, config=config, imputation=False)

with BenchmarkTracker('SpaCross') as tracker:
    net.trian()
    enc_rep, recon = net.process()

# Check for invalid values in embeddings
if np.isnan(enc_rep).any() or np.isinf(enc_rep).any():
    print("Warning: Embeddings contain invalid values!")
else:
    # 1. Save embedding
    adata.obsm['SpaCross_embed'] = np.ascontiguousarray(enc_rep, dtype=np.float64)
    # 2. Clustering using mclust
    print(f"Clustering with mclust (n_domains = {final_n_domains})...")
    adata.obs['mclust'] = TOOLS.clustering(z=adata.obsm['SpaCross_embed'], n_clust=final_n_domains, num_seed=1, method="mclust")

# ====================== 3. Standardize Metadata ======================
# Add batch information
adata.obs['batch_name'] = adata.obs['slice_id'].astype('category')

# Add ground truth cell type labels for benchmark evaluation
if 'Ground Truth' in adata.obs:
    adata.obs['celltype'] = adata.obs['Ground Truth'].astype('category')
else:
    print("[Warning] Ground Truth not found, biological metrics may not be computable!")

# ====================== 4. Save Results ======================
tracker.save_report(save_path / f"spacross_benchmark_{config_task['suffix']}.json", adata, 'SpaCross_embed', {'n_datasets': len(slices_list)})
adata.write(save_path / f"multiple_adata_{config_task['suffix']}_spacross.h5ad")

print(f"Successfully finished {CURRENT_TASK} and saved to {save_path}/multiple_adata_{config_task['suffix']}_spacross.h5ad")