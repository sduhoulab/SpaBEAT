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
datasets = ['P1CRC', 'P2CRC', 'P5CRC']
file_fold = '/data_hou/ST_data_new/Visium_crc/'
file_bin = 'binned_outputs/square_008um/'
file_meta = 'HumanColonCancer_VisiumHD/MetaData/'
adatas = []

for dataset in datasets:   
    print(f"Processing dataset: {dataset}")
    adata_path = os.path.join(file_fold, dataset, file_bin)
    adata = sc.read_visium(adata_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    adata.obs['batch'] = dataset  # Add batch information
    
    # read the annotation
    Ann_df_path = os.path.join(file_fold, file_meta)
    Ann_df = pd.read_parquet(Ann_df_path + dataset + '_Metadata.parquet')
    Ann_df['Periphery'] = Ann_df['Periphery'].fillna("unknown")
    
    mapping_dict = dict(zip(Ann_df['barcode'], Ann_df['Periphery']))
    adata.obs.loc[adata.obs['batch'] == dataset, 'Ground Truth'] = adata.obs_names.map(mapping_dict).astype('category')
    print(adata.obs[['batch', 'Ground Truth']].reset_index().head(5))
    
    # make spot name unique
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    
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
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=250)
    STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
    
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    total_cells += adata.n_obs
    print(f"   {dataset}: {adata.n_obs} cells processed")

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
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=100, device=used_device)
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
    'rad_cutoff': 250,
    'timestamp': pd.Timestamp.now().isoformat(),
    'device': str(used_device)
}

ST_utils.mclust_R(adata_concat, num_cluster=3, used_obsm='STAligner')
adata_concat = adata_concat[adata_concat.obs['celltype']!='unknown']
adata_concat.obs["new_batch"] = adata_concat.obs["batch_name"].astype('category')

adata_concat.write("/data_hou/ST_data_new/model-zn/STAligner/results/hd_crc_sta_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/STAligner/results/hd_crc_sta_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)
