import scanpy as sc
import pandas as pd
import os

file_fold = '/data_hou/ST_data_new/Visium_crc/'
file_bin = 'binned_outputs/square_008um'
datasets = ['P1CRC', 'P2CRC', 'P5CRC']
file_meta = 'HumanColonCancer_VisiumHD/MetaData/'
save_path = "/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/" 

data_list = []
Batch_list = []
####scanpy=1.9.1 preprocessing
for dataset in datasets:  
    adata_path = os.path.join(file_fold, dataset, file_bin)
    adata = sc.read_visium(adata_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    
    Ann_df_path = os.path.join(file_fold, file_meta)
    Ann_df = pd.read_parquet(Ann_df_path + dataset + '_Metadata.parquet')
    Ann_df['Periphery'] = Ann_df['Periphery'].fillna("unknown")
    
    mapping_dict = dict(zip(Ann_df['barcode'], Ann_df['Periphery']))
    adata.obs['celltype'] = adata.obs_names.map(mapping_dict).astype('category')
    unmapped_barcodes = adata.obs_names[adata.obs['celltype'].isna()]
    print(f"[{dataset}] discover {len(unmapped_barcodes)} unmapped Barcode:")
    print(adata.obs[['celltype']].reset_index().head(5))
    adata = adata[adata.obs['celltype'] != 'unknown'].copy()
    adata.X = adata.X.astype('float32')
    if 'spatial' in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['spatial'].astype('float32')
    min_gene = 20
    min_cell = 20
    sc.pp.filter_cells(adata, min_genes=min_gene)
    sc.pp.filter_genes(adata, min_cells=min_cell)
    
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('celltype', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()
    
    sc.pp.normalize_total(adata, target_sum=1e4)  
    sc.pp.log1p(adata)
    h5ad_path = os.path.join(save_path, f"{dataset}.h5ad")
    adata.write_h5ad(h5ad_path)  
    data_list.append(h5ad_path)
    Batch_list.append(adata)
    print(f"Saved {h5ad_path}")


####  conda activate Spatialign   
import sys
sys.path.append('/data_hou/BE/Spatialign/')
import os
import scanpy as sc
from spatialign import Spatialign
from warnings import filterwarnings
from anndata import AnnData
import h5py
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
filterwarnings("ignore")
import torch
torch.set_default_dtype(torch.float32)
import time
import psutil
import gc
import json
from sklearn.mixture import GaussianMixture

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("Starting data preprocessing...")

data_list = [
    '/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/P1CRC.h5ad',
    '/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/P2CRC.h5ad',
    '/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/P5CRC.h5ad'
]
dataset_names = ['P1CRC', 'P2CRC', 'P5CRC']

print("Loading datasets for cell counting...")
total_cells = 0
for i, data_path in enumerate(data_list):
    temp_adata = sc.read_h5ad(data_path)
    cells_count = temp_adata.n_obs
    total_cells += cells_count
    print(f"  {dataset_names[i]}: {cells_count} cells")
    del temp_adata  

print(f"Total cells across datasets: {total_cells}")

model = Spatialign(
    *data_list,
    batch_key='batch',
    is_norm_log=True,
    is_scale=False,
    n_neigh=15,
    is_undirected=True,
    latent_dims=100,
    seed=42,
    gpu=0,
    save_path="/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/",
    is_verbose=False
)
raw_merge = AnnData.concatenate(*model.dataset.data_list)
print(raw_merge.n_obs)
print(raw_merge.obs_names.is_unique)
# =============== spatiAlign training ===============
print("\nStarting core training benchmarking...")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training spatialign model...")
model.train(0.05, 1, 0.1)
model.alignment()

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before
print("Training completed!")

correct1 = sc.read_h5ad("/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/res/correct_data0.h5ad")
correct2 = sc.read_h5ad("/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/res/correct_data1.h5ad")
correct3 = sc.read_h5ad("/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/res/correct_data2.h5ad")

merge_data = correct1.concatenate(correct2, correct3)

# =============== Saving benchmarking results ===============
benchmark_results = {
    'method_name': 'Spatialign',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': merge_data.n_obs,
    'total_genes': merge_data.n_vars,
    'embedding_dim': merge_data.obsm["correct"].shape[1],
    'n_datasets': len(data_list),
    'datasets': dataset_names,
    'device': 'GPU 0',
    'random_seed': 42,
    'latent_dims': 100,
    'timestamp': pd.Timestamp.now().isoformat()
}

batch_mapping = {
    '0': 'P1CRC',
    '1': 'P2CRC',
    '2': 'P5CRC'
}

merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)
merge_data.obs['new_batch'] = merge_data.obs['new_batch'].astype('category')
merge_data = merge_data[~merge_data.obs['celltype'].isna()]
merge_data.X = np.nan_to_num(merge_data.X, nan=0.0)

print("Performing clustering...")
sc.pp.scale(merge_data)
X = merge_data.obsm['correct']
n_components = 3
gmm = GaussianMixture(n_components=n_components, random_state=42)
merge_data.obs['mclust'] = gmm.fit_predict(X)
merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")
merge_data.write("/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/hd_crc_spa_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/spatialign/results/hd_crc/hd_crc_spa_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)
