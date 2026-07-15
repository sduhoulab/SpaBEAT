import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/root/BE/DeepST/DeepST')
from DeepST import run
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad
import time
import psutil
import gc
import json
import torch

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_deepst_integration(ref_slice, target_slice, data_path, base_save_path, n_domains=4):
    data_name_list = [ref_slice, target_slice]
    pair_name = f"{ref_slice}_vs_{target_slice}"
    
    save_path = os.path.join(base_save_path, pair_name)
    os.makedirs(save_path, exist_ok=True)

    deepen = run(
        save_path = save_path,
        task = "Integration",
        pre_epochs = 500,
        epochs = 500,
        use_gpu = True,
    )

    augement_data_list = []
    graph_list = []
    total_cells = 0

    for i in range(len(data_name_list)):
        sample_path = data_name_list[i]
        print(f">>> Processing slice: {sample_path}")
        adata = sc.read_h5ad(os.path.join(data_path, sample_path, f"{sample_path}.h5ad"))
        
        if sp.issparse(adata.X):
            adata.X = adata.X.toarray()
        
        adata.obs["array_col"] = adata.obsm["spatial"][:, 0]
        adata.obs["array_row"] = adata.obsm["spatial"][:, 1]
        
        adata.obs["imagecol"] = adata.obsm["spatial"][:, 0]
        adata.obs["imagerow"] = adata.obsm["spatial"][:, 1]
        
        adata = deepen._get_augment(
            adata,
            spatial_type="BallTree",
            use_morphological=False
        )
        
        adata.obs['new_batch'] = sample_path
        
        if 'ground_truth' not in adata.obs:
            metadata_file = os.path.join(data_path, sample_path, 'metadata.csv')
            if os.path.exists(metadata_file):
                print(f"[Info] Trying to load labels from external metadata.csv...")
                df_meta = pd.read_csv(metadata_file, index_col=0)
                possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label', 'Cell_class']
                for col in possible_cols:
                    if col in df_meta.columns:
                        adata.obs['ground_truth'] = adata.obs_names.map(df_meta[col])
                        break
        
        if 'ground_truth' in adata.obs.columns:
            adata = adata[~pd.isnull(adata.obs['ground_truth'])]
            
        total_cells += adata.n_obs
        
        graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
        graph_list.append(graph_dict)
        augement_data_list.append(adata)

    print(f"data_name_list length: {len(data_name_list)}")
    print(f"graph_list length: {len(graph_list)}")

    torch.cuda.empty_cache()

    multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list, data_name_list=data_name_list, graph_list=graph_list)

    data = deepen._data_process(multiple_adata, pca_n_comps=64)

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
    memory_used = get_memory_usage() - memory_before

    print("Training completed!")

    benchmark_results = {
        'method_name': 'DeepST',
        'dataset': pair_name,
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

    if 'ground_truth' in multiple_adata.obs.columns:
        multiple_adata.obs['celltype'] = multiple_adata.obs['ground_truth'].astype('category')
        
    multiple_adata.write(os.path.join(save_path, f"multiple_adata_{pair_name}.h5ad"))

    with open(os.path.join(save_path, f"deepst_benchmark_{pair_name}.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)
        
    print(f"\n🎉 Task {pair_name} integration finished! Results saved to: {save_path}\n")
    
    del deepen, adata, multiple_adata, multiple_graph, data, deepst_embed
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    DATA_PATH = "/root/ST0507/simulations_model/data_model/1_DLPFC_InterSlice/SRTsim_Converted/"
    BASE_SAVE_PATH = "/root/ST0507/new_model/DeepST/results"
    
    REF_SLICE = "Sim_Slice_1"
    n_domains = 7
    
    for i in range(2, 7):
        target_slice = f"Sim_Slice_{i}"
        
        print("="*60)
        print(f"🚀 Starting consecutive slice alignment task: {REF_SLICE} + {target_slice}")
        print("="*60)
        
        try:
            run_deepst_integration(
                ref_slice=REF_SLICE,
                target_slice=target_slice,
                data_path=DATA_PATH,
                base_save_path=BASE_SAVE_PATH,
                n_domains=n_domains
            )
        except Exception as e:
            print(f"❌ Task {REF_SLICE}_vs_{target_slice} failed. Error message: {str(e)}")
            continue
            
    print("All slice pair integration tasks have been completed!")