import sys
sys.path.append("/root/ST0507/new_model/SpaBatch-main/")

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
import matplotlib.pyplot as plt

from SpaBatch.adj import main, combine_graph_dict
from SpaBatch.train import train_model
from SpaBatch.utils import mclust_R, fix_seed
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

def run_spabatch_integration(ref_slice, target_slice, data_path, base_save_path, n_domains=4):
    proj_list = [ref_slice, target_slice]
    pair_name = f"{ref_slice}_vs_{target_slice}"
    
    save_path = Path(base_save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_path)

    print(f"\n=== Running SpaBatch Benchmark for: {pair_name} (n_domains = {n_domains}) ===")
    print("Loading data, building graphs, and concatenating...")
    
    adata = None
    graph_dict = None

    for i, proj_name in enumerate(tqdm(proj_list)):
        batch_name = proj_name.split('/')[-1] 
        file_path = data_root / proj_name / f"{batch_name}.h5ad"
        metadata_file = data_root / proj_name / 'metadata.csv'
        
        adata_tmp = sc.read_h5ad(file_path)
        adata_tmp.var_names_make_unique()
        adata_tmp.obs['batch_name'] = batch_name

        if 'ground_truth' not in adata_tmp.obs.columns:
            if metadata_file.exists():
                print(f"\n[Info] No ground_truth found in {proj_name}, loading from metadata.csv...")
                df_meta = pd.read_csv(metadata_file, index_col=0)
                
                possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label']
                for col in possible_cols:
                    if col in df_meta.columns:
                        adata_tmp.obs['ground_truth'] = adata_tmp.obs_names.map(df_meta[col])
                        print(f"  [✔] Successfully loaded labels from column: '{col}'")
                        break
                else:
                    print(f"  [Warning] metadata.csv found but no valid label columns detected for {proj_name}!")
            else:
                print(f"  [Error] No labels in adata and metadata.csv not found for {proj_name}!")
        else:
            adata_tmp.obs['ground_truth'] = adata_tmp.obs['ground_truth'].astype('category')

        # Remove NA background spots
        adata_tmp = adata_tmp[~pd.isnull(adata_tmp.obs['ground_truth'])].copy()

        # Construct spatial graph
        graph_dict_tmp = main(adata_tmp, adj_cons_by='coordinate', distType='KNN', k_cutoff=12, rad_cutoff=250)

        # Merge datasets
        if i == 0:
            adata = adata_tmp
            graph_dict = graph_dict_tmp
        else:
            var_names = adata.var_names.intersection(adata_tmp.var_names)
            adata_tmp = adata_tmp[:, var_names]
            adata = adata[:, var_names].copy()
            adata_tmp = adata_tmp[:, var_names].copy()
            adata = adata.concatenate(adata_tmp, batch_key="concat_batch")
            graph_dict = combine_graph_dict(graph_dict, graph_dict_tmp)

    # ==========================================
    # 2. Preprocessing
    # ==========================================

    adata.layers['count'] = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()
    
    n_top = min(5000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable'] == True]
    adata.raw = adata.copy()
    sc.pp.scale(adata)
    adata.obsm['X_pca'] = PCA(n_components=50, random_state=42).fit_transform(adata.X)

    # ==========================================
    # 3. Benchmark training
    # ==========================================
    with BenchmarkTracker('SpaBatch') as tracker:
        SpaBatch_net = train_model(adata, graph_dict, pre_epochs=500, epochs=1000, mask_rate=0.2)
        SpaBatch_net.train_with_dec(num_aggre=1)
        SpaBatch_feat, q = SpaBatch_net.process()

    if not (np.isnan(SpaBatch_feat).any() or np.isinf(SpaBatch_feat).any()):
        adata.obsm['SpaBatch_embed'] = np.ascontiguousarray(SpaBatch_feat, dtype=np.float64)

    if 'SpaBatch_embed' in adata.obsm:
        try:
            print(f"Clustering with mclust (n_domains = {n_domains})...")
            mclust_R(adata, num_cluster=n_domains, used_obsm='SpaBatch_embed')
        except Exception as e:
            print(f"Mclust Error: {e}")

    if 'ground_truth' in adata.obs:
        adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
    else:
        print("[Warning] ground_truth column not found, biological metrics may fail!")

    # ==========================================
    # 4. Save results 
    # ==========================================
    json_path = save_path / f"spabatch_benchmark_{pair_name}.json"
    tracker.save_report(
        json_path=str(json_path),
        adata=adata,
        embed_key='SpaBatch_embed',
        extra_meta={'n_datasets': len(proj_list), 'dataset': pair_name, 'hvg_genes': n_top}
    )

    h5ad_path = save_path / f"multiple_adata_{pair_name}_spabatch.h5ad"
    adata.write(str(h5ad_path))
    print(f"Successfully finished {pair_name} and saved to {h5ad_path}")
    
    del SpaBatch_net, adata, adata_tmp, graph_dict, graph_dict_tmp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

=
if __name__ == "__main__":
    DATA_PATH = "/root/ST0507/simulations_model/data_model/1_DLPFC_InterSlice/SRTsim_Converted/" 
    BASE_SAVE_PATH = "/root/ST0507/new_model/SpaBatch-main/results" 
    
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)

    REF_SLICE = "Sim_Slice_1"
    
    N_DOMAINS = 7
    
    for i in range(6, 7):
        target_slice = f"Sim_Slice_{i}"
        
        print("="*60)
        print(f"🚀 start: {REF_SLICE} + {target_slice}")
        print("="*60)
        
        try:
            run_spabatch_integration(
                ref_slice=REF_SLICE, 
                target_slice=target_slice, 
                data_path=DATA_PATH, 
                base_save_path=BASE_SAVE_PATH, 
                n_domains=N_DOMAINS
            )
        except Exception as e:
            print(f"❌ Task {REF_SLICE}_vs_{target_slice} failed. Error message: {str(e)}")
            continue
            
    print("🎯 All STAligner slice pair integration tasks have been completed!")