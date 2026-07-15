import sys
sys.path.append("/data/ZhaoMH/ST0507/SpaCross-main")

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
    'merfish1': {'root': '/data/ZhaoMH/ST0507/RAW_SLICE/merfish_mouse1/data/MERFISH_converted/', 'slices': ['Slice_7', 'Slice_8', 'Slice_9', 'Slice_10', 'Slice_11'], 'n_domains': 8, 'type': 'merfish1', 'suffix': 'merfish1'},
    'her2_B': {'root': '/root/ST0507/data/Her2_tumor_converted/B/', 'slices': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6'], 'n_domains': 5, 'type': 'her2_B', 'suffix': 'B'}
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
slices_list = config_task['slices']
save_path = Path('/data/ZhaoMH/ST0507/SpaCross-main/Results')
save_path.mkdir(parents=True, exist_ok=True)

print(f"=== Running SpaCross Benchmark for: {CURRENT_TASK} (n_domains = {final_n_domains}) ===")

# Load model configuration
config_type = config_task['type'].upper() 
yaml_dir = Path('/data/ZhaoMH/ST0507/SpaCross-main/Config')
expected_yaml = yaml_dir / f"{config_type}.yaml"
default_yaml = yaml_dir / "DLPFC.yaml"

if expected_yaml.exists():
    print(f"Loading specific model config from: {expected_yaml.name}")
    yaml_to_load = expected_yaml
else:
    print(f"[提示] 未找到专属配置文件 {expected_yaml.name}，默认使用通用的 DLPFC.yaml")
    yaml_to_load = default_yaml

with open(yaml_to_load, 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

# ====================== 1. Data Loading and Alignment ======================
Batch_list = []
for idxxx, section_id in enumerate(slices_list):
    input_dir = data_root / section_id

    batch_name = str(section_id).split('/')[-1]
    
    file_path = input_dir / f"{batch_name}.h5ad"
    metadata_file = input_dir / 'metadata.csv'

    adata = sc.read_h5ad(file_path)
    adata.var_names_make_unique(join="++")
    adata.obs['slice_id'] = section_id
    adata.obs['batch_name'] = batch_name

    if 'ground_truth' in adata.obs.columns and 'Ground Truth' not in adata.obs.columns:
        adata.obs['Ground Truth'] = adata.obs['ground_truth']

    if 'Ground Truth' not in adata.obs.columns:
        if metadata_file.exists():
            print(f"\n[Info] No ground_truth found in {section_id}, loading from metadata.csv...")
            df_meta = pd.read_csv(metadata_file, index_col=0)
            
            possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label', 'Cell_class']
            for col in possible_cols:
                if col in df_meta.columns:
                    adata.obs['Ground Truth'] = adata.obs_names.map(df_meta[col])
                    print(f"  [✔] Successfully loaded labels from column: '{col}'")
                    break
            else:
                print(f"  [Warning] metadata.csv found but no valid label columns detected for {section_id}!")
        else:
            print(f"  [Error] No labels in adata and metadata.csv not found for {section_id}!")
    
    if 'Ground Truth' in adata.obs.columns:
        adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
        matched_count = adata.obs['Ground Truth'].notna().sum()
        print(f"👉 [{section_id}] Total cells: {adata.n_obs} | Matched valid cells: {matched_count}\n")
    else:
        print(f"👉 [{section_id}] Total cells: {adata.n_obs} | No Ground Truth available.\n")

    adata = adata[~pd.isnull(adata.obs['Ground Truth'])].copy()

    adata.layers['count'] = adata.X.toarray()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    Batch_list.append(adata)

# Perform ICP alignment using SpaCross
print("Performing ICP alignment...")
Batch_list = TOOLS.align_spots(Batch_list, method='icp', data_type="merfish", plot=False)
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
    print("❌ Warning: Embeddings contain invalid values!")
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

print(f"✅ Successfully finished {CURRENT_TASK} and saved to {save_path}/multiple_adata_{config_task['suffix']}_spacross.h5ad")