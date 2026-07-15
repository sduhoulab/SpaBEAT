import sys
# 确保填入你自己的 Spatialign 路径
sys.path.append('/root/BE/Spatialign/spatialign')

import os
from spatialign import Spatialign
from warnings import filterwarnings
from anndata import AnnData
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import h5py
import matplotlib.pyplot as plt
import torch
import time
import psutil
import gc
import json
from sklearn.mixture import GaussianMixture

filterwarnings("ignore")
torch.set_default_dtype(torch.float32)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_spatialign_integration(ref_slice, target_slice, data_path, base_save_path, n_components=4):
    datasets = [ref_slice, target_slice]
    pair_name = f"{ref_slice}_vs_{target_slice}"
    
    os.makedirs(base_save_path, exist_ok=True)
    temp_input_files = []
    total_cells = 0

    print(f"\nStarting data preprocessing for {pair_name}...")
    
    for dataset in datasets:
        print(f"Processing {dataset}...")
        adata = sc.read_h5ad(os.path.join(data_path, dataset, f"{dataset}.h5ad"))
        adata.var_names_make_unique()
        
        metadata_file = os.path.join(data_path, dataset, 'metadata.csv')
        if os.path.exists(metadata_file):
            df_meta = pd.read_csv(metadata_file, index_col=0)
            possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label', 'Cell_class']
            for col in possible_cols:
                if col in df_meta.columns:
                    adata.obs['ground_truth'] = adata.obs_names.map(df_meta[col])
                    break
        
        if 'ground_truth' in adata.obs.columns:
            adata = adata[~adata.obs['ground_truth'].isna()]
            adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
            adata = adata[adata.obs['celltype'] != 'unknown']

        adata.layers['count'] = adata.X.copy()
        
        min_gene = 20
        min_cell = 20
        sc.pp.filter_cells(adata, min_genes=min_gene)
        sc.pp.filter_genes(adata, min_cells=min_cell)
        
        sc.pp.normalize_total(adata, target_sum=1e4)  
        sc.pp.log1p(adata)

        if sp.issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = np.array(adata.X, dtype=np.float32)

        if 'spatial' in adata.obsm:
            adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
            
        total_cells += adata.n_obs
        print(f"  {dataset}: {adata.n_obs} cells")
        
        temp_path = os.path.join(base_save_path, f"temp_{dataset}.h5ad")
        adata.write_h5ad(temp_path)
        temp_input_files.append(temp_path)
        
    print(f"Total cells across {pair_name}: {total_cells}")
    print("=======================================================")
    
    model = Spatialign(
        *temp_input_files,
        batch_key='batch',
        is_norm_log=True,
        is_scale=False,
        n_neigh=15,
        is_undirected=True,
        latent_dims=100,
        seed=42,
        gpu=0,
        save_path=base_save_path,
        is_verbose=False
    )

    print("\nStarting core training benchmarking...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_before = get_memory_usage()
    training_start_time = time.time()

    print("Training Spatialign model...")
    model.train(0.05, 1, 0.1)
    model.alignment()

    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    memory_after = get_memory_usage()
    memory_used = memory_after - memory_before
    print("Training completed!")


    res_dir = os.path.join(base_save_path, "res")
    correct_path1 = os.path.join(res_dir, "correct_data0.h5ad")
    correct_path2 = os.path.join(res_dir, "correct_data1.h5ad")
    
    correct1 = sc.read_h5ad(correct_path1)
    correct2 = sc.read_h5ad(correct_path2)
    merge_data = correct1.concatenate(correct2)


    batch_mapping = {'0': datasets[0], '1': datasets[1]}
    merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)
    merge_data.obs['new_batch'] = merge_data.obs['new_batch'].astype('category')
    
    merge_data.obs['celltype'] = merge_data.obs['ground_truth']
    merge_data = merge_data[~merge_data.obs['celltype'].isna()]
    merge_data.X = np.nan_to_num(merge_data.X, nan=0.0)

    print("Performing GaussianMixture clustering...")
    if 'count' not in merge_data.layers:
        merge_data.layers['count'] = merge_data.X.copy()
        
    sc.pp.scale(merge_data)
    X_emb = merge_data.obsm['correct']
    
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    merge_data.obs['mclust'] = gmm.fit_predict(X_emb)
    merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")

    benchmark_results = {
        'method_name': 'Spatialign',
        'dataset': pair_name,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'training_time_hours': training_time / 3600,
        'memory_usage_mb': memory_used,
        'memory_usage_gb': memory_used / 1024,
        'total_cells': total_cells,
        'final_cells': merge_data.n_obs,
        'total_genes': merge_data.n_vars,
        'embedding_dim': merge_data.obsm["correct"].shape[1],
        'n_datasets': len(datasets),
        'device': 'GPU 0',
        'random_seed': 42,
        'latent_dims': 100,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    merge_data.write(os.path.join(base_save_path, f"multiple_adata_{pair_name}.h5ad"))

    with open(os.path.join(base_save_path, f"spatialign_benchmark_{pair_name}.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n🎉 任务 {pair_name} Spatialign 整合分析完毕！结果已保存至: {base_save_path}\n")

    for temp_file in temp_input_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    if os.path.exists(correct_path1): os.remove(correct_path1)
    if os.path.exists(correct_path2): os.remove(correct_path2)
    
    del model, merge_data, correct1, correct2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



if __name__ == "__main__":

    DATA_PATH = "/root/ST0507/simulations_model/data_model/1_DLPFC_InterSlice/SRTsim_Converted/" 
    BASE_SAVE_PATH = "/root/ST0507/new_model/spatialign/results" 
    
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    
    REF_SLICE = "Sim_Slice_1"
    
    N_COMPONENTS = 7
    
    for i in range(2, 7):
        target_slice = f"Sim_Slice_{i}"
        
        print("="*60)
        print(f"🚀 start: {REF_SLICE} + {target_slice}")
        print("="*60)
        
        try:
            run_spatialign_integration(
                ref_slice=REF_SLICE, 
                target_slice=target_slice, 
                data_path=DATA_PATH, 
                base_save_path=BASE_SAVE_PATH, 
                n_components=N_COMPONENTS
            )
        except Exception as e:
            print(f"❌ Task {REF_SLICE}_vs_{target_slice} failed. Error message: {str(e)}")
            continue
            
    print("🎯 All STAligner slice pair integration tasks have been completed!")
    
    
    