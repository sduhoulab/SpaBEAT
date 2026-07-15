import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/root/BE/DeepST/DeepST')
import os 
from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import community as louvain
import pandas as pd
import anndata as ad 
import time
import psutil
import gc
import json
import torch

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

data_path = "/root/ST0507/data/Her2_tumor_converted/A/" 

data_name_list = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
save_path = "/root/ST0507/results/DeepST" 
n_domains = 5
deepen = run(save_path = save_path, 
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
    batch_name = sample_path.split('/')[-1]
    adata = sc.read_h5ad( os.path.join(data_path, sample_path, f"{batch_name}.h5ad") )
    
    
    # ==================== Supplement ====================
    # 1. Get library_id of current sample (e.g., 'A1')
    library_id = list(adata.uns["spatial"].keys())[0]
    # Explicitly assign high-resolution image ('hires') for DeepST
    adata.uns["spatial"][library_id]["use_quality"] = "hires"
    # 2. Extract imagerow and imagecol from spatial matrix
    # Note: In image coordinate system, col corresponds to X (width), row corresponds to Y (height)
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0]
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1]

    # 3. Provide physical array coordinates for _get_augment
    # (DeepST calculates node distances using these two columns during graph construction; pixel coordinates are directly used)
    adata.obs["array_col"] = adata.obsm["spatial"][:, 0]
    adata.obs["array_row"] = adata.obsm["spatial"][:, 1]

    # ==================================================================

    adata = deepen._get_image_crop(adata, data_name=data_name_list[i])
    adata = deepen._get_augment(adata, spatial_type="LinearRegress")

    # ==================== Supplement ====================
    # 2. Load batch label
    adata.obs['new_batch'] = data_name_list[i]

    # [Condition check] Load labels externally if ground_truth does not exist inside anndata
    if 'ground_truth' not in adata.obs:
        metadata_file = os.path.join(data_path, sample_path, 'metadata.csv')
        if os.path.exists(metadata_file):
            print(f"[Notice] No ground_truth found inside {sample_path}, trying to load from external metadata.csv...")
            df_meta = pd.read_csv(metadata_file, index_col=0)

            # Scan potential label column names (compatible with Her2, DLPFC and other datasets)
            possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label']
            for col in possible_cols:
                if col in df_meta.columns:
                    # Map via cell barcodes (index) to avoid order mismatch
                    adata.obs['ground_truth'] = adata.obs_names.map(df_meta[col])
                    print(f"  [✔] Successfully loaded external label column: '{col}'")
                    break
            else:
                print(f"  [Warning] metadata.csv exists, but no matched known label columns found!")
        else:
            print(f"  [Error] No internal labels in {sample_path}, and fallback metadata.csv not found!")

    # ==================================================================
    
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    total_cells += adata.n_obs
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
    graph_list.append(graph_dict)
    augement_data_list.append(adata)

print(f"data_name_list length: {len(data_name_list)}")
print(f"graph_list length: {len(graph_list)}")

torch.cuda.empty_cache()

multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list, data_name_list=data_name_list, graph_list=graph_list)
data = deepen._data_process(multiple_adata, pca_n_comps=200)

# =============== DeepST training ===============
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
memory_after = get_memory_usage()
memory_used = memory_after - memory_before

print("Training completed!")

# =============== Saving benchmarking results ===============
benchmark_results = {
    'method_name': 'DeepST',
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
multiple_adata.obs['celltype'] = multiple_adata.obs['ground_truth'].astype('category')
multiple_adata.write("/root/ST0507/new_model/DeepST/results/multiple_adata_her2_A111.h5ad")

with open("/root/ST0507/new_model/DeepST/results/deepst_benchmark_her2_A111.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

