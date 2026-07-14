import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data_hou/BE/DeepST/DeepST')
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

all_clusters = set()

print("Starting preprocessing Visium HD data")
hd_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_hcc_hd_ff/adata.h5ad"
hd_adata = sc.read_h5ad(hd_adata_path)
#hd_celltypes = hd_adata.obs['annotation'].dropna().astype(str).unique().tolist()
#hd_celltypes.sort()
#print(f"Total {len(hd_celltypes)} celltypes:") 15
#print(hd_celltypes)
hd_celltypes = hd_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(hd_celltypes)

print("Starting preprocessing Xe data")
xe_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_hcc_xenium/adata.h5ad"
xe_adata = sc.read_h5ad(xe_adata_path)
#xe_celltypes = xe_adata.obs['annotation'].dropna().astype(str).unique().tolist()
#xe_celltypes.sort()
#print(f"Total {len(xe_celltypes)} celltypes:") 17
#print(xe_celltypes)
xe_celltypes = xe_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(xe_celltypes)

n_domains = len(all_clusters)
print(f"Clusters {n_domains}")

data_name_list = ['hd', 'xe']
save_path = "/data_hou/ST_data_new/model-zn/deepst/results/" 

deepen = run(save_path = save_path, 
	task = "Integration",
	pre_epochs = 500, 
	epochs = 500, 
	use_gpu = False,
)

name_to_adata = {
    'hd': hd_adata,
    'xe': xe_adata
}

augement_data_list = []
graph_list = []
total_cells = 0

for name, adata in name_to_adata.items():
    print(f"Processing {name}...")
    
    adata.var_names_make_unique()
    adata.obs['new_batch'] = name # Add batch information
    print("--- adata summary ---")
    print(adata)
    print("\n--- Adata Barcode (Obs Names) head ---")
    print(adata.obs_names[:5].tolist())
    
    if name == 'xe':
        control_genes = adata.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
        adata = adata[:, ~control_genes].copy()
        print("--- Filtered adata summary ---")
        print(adata)
    
    adata.obs.loc[adata.obs['new_batch'] == name, 'ground_truth'] = adata.obs['annotation']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print(adata.obs[['new_batch', 'ground_truth']].reset_index().head(5))
    
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('ground_truth', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()
    
    adata = deepen._get_augment(adata, spatial_type="BallTree", use_morphological = False)
    
    total_cells += adata.n_obs
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
    graph_list.append(graph_dict)
    augement_data_list.append(adata)
    
    print(f"  {name}: {adata.n_obs} cells, {adata.n_vars} genes")

print(f"Total cells across datasets: {total_cells}")
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
multiple_adata.write("/data_hou/ST_data_new/model-zn/deepst/results/spatch_hcc_deepst_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/deepst/results/spatch_hcc_deepst_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

