import sys
sys.path.append("/root/ST0507/new_model/SpaCross-main")

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
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
import yaml

import SpaCross as TOOLS

import random
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



def run_spacross_integration(ref_slice, target_slice, data_path, base_save_path, yaml_dir, n_domains=4):
    slices_list = [ref_slice, target_slice]
    pair_name = f"{ref_slice}_vs_{target_slice}"
    
    data_root = Path(data_path)
    save_path = Path(base_save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running SpaCross Benchmark for: {pair_name} (n_domains = {n_domains}) ===")

    # ====================== Load model configuration ======================
    yaml_dir_path = Path(yaml_dir)
    default_yaml = yaml_dir_path / "DLPFC.yaml"

    if default_yaml.exists():
        print(f"Loading model config from: {default_yaml.name}")
        yaml_to_load = default_yaml
    else:
        print(f"❌ [Error] 未找到配置文件 {default_yaml}，请检查路径！")
        return

    with open(yaml_to_load, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    # ====================== 1. Data Loading and Alignment ======================
    Batch_list = []
    
    for section_id in slices_list:
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
                df_meta = pd.read_csv(metadata_file, index_col=0)
                possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label', 'Cell_class']
                for col in possible_cols:
                    if col in df_meta.columns:
                        adata.obs['Ground Truth'] = adata.obs_names.map(df_meta[col])
                        break
        
        if 'Ground Truth' in adata.obs.columns:
            adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
            matched_count = adata.obs['Ground Truth'].notna().sum()
            print(f"👉 [{section_id}] Total cells: {adata.n_obs} | Matched valid cells: {matched_count}")
        else:
            print(f"👉 [{section_id}] Total cells: {adata.n_obs} | No Ground Truth available.")

        adata = adata[~pd.isnull(adata.obs['Ground Truth'])].copy()
        adata.layers['count'] = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()
        
        n_top = min(5000, adata.n_vars)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)

        Batch_list.append(adata)

    print("Performing ICP alignment...")
    Batch_list = TOOLS.align_spots(Batch_list, method='icp', data_type="merfish", plot=False)
    adata, edge_index = TOOLS.graph_construction3D(Batch_list, section_ids=slices_list, k_cutoff=12, mode='KNN')
    adata.obsm['X_pca'] = PCA(n_components=50, random_state=42).fit_transform(adata.X)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = TOOLS.SC_pipeline(adata, edge_index=edge_index, num_clusters=n_domains,
                            device=device, config=config, imputation=False)

    with BenchmarkTracker('SpaCross') as tracker:
        net.trian()
        enc_rep, recon = net.process()

    if np.isnan(enc_rep).any() or np.isinf(enc_rep).any():
        print("❌ Warning: Embeddings contain invalid values!")
    else:
        adata.obsm['SpaCross_embed'] = np.ascontiguousarray(enc_rep, dtype=np.float64)
        print(f"Clustering with mclust (n_domains = {n_domains})...")
        adata.obs['mclust'] = TOOLS.clustering(z=adata.obsm['SpaCross_embed'], n_clust=n_domains, num_seed=1, method="mclust")


    adata.obs['batch_name'] = adata.obs['slice_id'].astype('category')
    if 'Ground Truth' in adata.obs:
        adata.obs['celltype'] = adata.obs['Ground Truth'].astype('category')


    json_path = save_path / f"spacross_benchmark_{pair_name}.json"
    tracker.save_report(
        json_path=str(json_path), 
        adata=adata, 
        embed_key='SpaCross_embed', 
        extra_meta={'n_datasets': len(slices_list), 'dataset': pair_name, 'hvg_genes': n_top}
    )
    
    h5ad_path = save_path / f"multiple_adata_{pair_name}_spacross.h5ad"
    adata.write(str(h5ad_path))
    
    print(f"✅ Successfully finished {pair_name} and saved to {h5ad_path}\n")
    

    del net, adata, Batch_list, enc_rep, recon, edge_index
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



if __name__ == "__main__":

    DATA_PATH = "/root/ST0507/simulations_model/data_model/1_DLPFC_InterSlice/SRTsim_Converted/" 
    BASE_SAVE_PATH = "/root/ST0507/new_model/SpaCross-main/Results" 
    YAML_DIR = "/root/ST0507/new_model/SpaCross-main/Config"
    
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    
    REF_SLICE = "Sim_Slice_1"
    

    N_DOMAINS = 7
    for i in range(2, 7):
        target_slice = f"Sim_Slice_{i}"
        
        print("="*60)
        print(f"🚀 start: {REF_SLICE} + {target_slice}")
        print("="*60)
        
        try:
            run_spacross_integration(
                ref_slice=REF_SLICE, 
                target_slice=target_slice, 
                data_path=DATA_PATH, 
                base_save_path=BASE_SAVE_PATH, 
                yaml_dir=YAML_DIR,
                n_domains=N_DOMAINS
            )
        except Exception as e:
            print(f"❌ Task {REF_SLICE}_vs_{target_slice} failed. Error message: {str(e)}")
            continue
            
    print("🎯 All STAligner slice pair integration tasks have been completed!")