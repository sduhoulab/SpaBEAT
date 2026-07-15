import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data/ZhaoMH/ST0507/DeepST-main/DeepST')
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

data_path = "/data/ZhaoMH/ST0507/RAW_SLICE/STARmap_mouse/STARMAP_converted/" 
data_name_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

save_path = "/data/ZhaoMH/ST0507/DeepST-main/results" 
os.makedirs(save_path, exist_ok=True)

n_domains = 5

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
    adata = sc.read_h5ad(os.path.join(data_path, sample_path, f"slice_{sample_path}.h5ad"))
    
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    
    adata.obs["array_col"] = adata.obsm["spatial"][:, 0]
    adata.obs["array_row"] = adata.obsm["spatial"][:, 1]
    
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0]
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1]
    
    adata = deepen._get_augment(
        adata, 
        spatial_type="LinearRegress",
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


# ==========================================
multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list, data_name_list=data_name_list, graph_list=graph_list)

data = deepen._data_process(multiple_adata, pca_n_comps=20)

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

# ==========================================
# 5. 保存并导出指标
# ==========================================
benchmark_results = {
    'method_name': 'DeepST',
    'dataset': 'MERFISH_Mouse1',
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
    
multiple_adata.write(os.path.join(save_path, "multiple_adata_starmap.h5ad"))

with open(os.path.join(save_path, "deepst_benchmark_starmap.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)
    

