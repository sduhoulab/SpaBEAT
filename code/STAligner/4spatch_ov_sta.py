import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data_hou/BE/STAligner/')
import STAligner
from STAligner import ST_utils
from STAligner.ST_utils import match_cluster_labels
import os
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix
import time
import psutil
import gc
import json
import torch

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
used_device = device  

total_cells = 0
Batch_list = []
adj_list = []
datasets = ['hd', 'xe']

all_clusters = set()
print("Starting preprocessing Visium HD data")
hd_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_ov_hd_ff/adata.h5ad"
hd_adata = sc.read_h5ad(hd_adata_path)
#hd_celltypes = hd_adata.obs['annotation'].dropna().astype(str).unique().tolist()
#hd_celltypes.sort()
#print(f"Total {len(hd_celltypes)} celltypes:") 15
#print(hd_celltypes)
hd_celltypes = hd_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(hd_celltypes)
#all_clusters.discard('Unknown')
print("Starting preprocessing Xe data")

xe_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_ov_xenium/adata.h5ad"
xe_adata = sc.read_h5ad(xe_adata_path)
#xe_celltypes = xe_adata.obs['annotation'].dropna().astype(str).unique().tolist()
#xe_celltypes.sort()
#print(f"Total {len(xe_celltypes)} celltypes:") 17
#print(xe_celltypes)
xe_celltypes = xe_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(xe_celltypes)

n_clusters = len(all_clusters)
print(f"Clusters {n_clusters}")

adatas = []
name_to_adata = {
    'hd': hd_adata,
    'xe': xe_adata
}

for name, adata in name_to_adata.items():
    print(f"Processing {name}...")
    
    adata.var_names_make_unique()
    adata.obs['batch'] = name # Add batch information
    print("--- Adata summary ---")
    print(adata)
    
    if name == 'xe':
        control_genes = adata.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
        adata = adata[:, ~control_genes].copy()
        print("--- Filtered adata summary ---")
        print(adata)
    if adata.obs['annotation'].dtype.name == 'category':
        adata.obs['annotation'] = adata.obs['annotation'].cat.add_categories(["unknown"])
    adata.obs['annotation'] = adata.obs['annotation'].fillna("unknown")
    adata.obs.loc[adata.obs['batch'] == name, 'Ground Truth'] = adata.obs['annotation'].astype('category')
    print(adata.obs[['batch', 'Ground Truth']].reset_index().head(5))
    
    adata.obs_names = [x+'_'+name for x in adata.obs_names]
    
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('Ground Truth', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()
    
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=80)
    STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
    
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    total_cells += adata.n_obs
    print(f"  {name}: {adata.n_obs} cells, {adata.n_vars} genes")

print(f"Total cells across datasets: {total_cells}")
adata_concat = ad.concat(Batch_list, label="slice_name", keys=datasets)
adata_concat.obs['celltype'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print(f'adata_concat.shape: {adata_concat.shape}')

# adj
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1, len(datasets)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

# =============== STAligner training ===============
print("Starting STAligner Core Training Benchmark...")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Initializing STAligner model...")
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=80, device=used_device)
edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
adata_concat.uns['edgeList'] = edge_list

training_end_time = time.time()
memory_after = get_memory_usage()
training_time = training_end_time - training_start_time
memory_used = memory_after - memory_before

print("Training completed!")

# =============== Saving benchmarking results ===============
benchmark_results = {
    'method_name': 'STAligner',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'total_genes': adata_concat.n_vars,
    'embedding_dim': adata_concat.obsm['STAligner'].shape[1],
    'n_datasets': len(datasets),
    'random_seed': 50,
    'hvg_genes': 5000,
    'knn_neigh': 100,
    'rad_cutoff': 80,
    'timestamp': pd.Timestamp.now().isoformat(),
    'device': str(used_device)
}

ST_utils.mclust_R(adata_concat, num_cluster=n_clusters, used_obsm='STAligner')
adata_concat = adata_concat[adata_concat.obs['celltype']!='unknown']
adata_concat.obs["new_batch"] = adata_concat.obs["batch_name"].astype('category')

adata_concat.write("/data_hou/ST_data_new/model-zn/STAligner/results/spatch_ov_sta_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/STAligner/results/spatch_ov_sta_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)