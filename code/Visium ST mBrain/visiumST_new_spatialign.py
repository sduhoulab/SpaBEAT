import scanpy as sc
import pandas as pd
import os


file_fold = "/root/ST0507/data/visumST/Visium_converted/" 
datasets = ['slice_39', 'slice_44']
save_path = "/root/ST0507/new_model/spatialign/results" 
data_list = []
Batch_list = []
####scanpy=1.9.1 preprocessing
for dataset in datasets:  
    adata = sc.read_h5ad(os.path.join(file_fold + dataset, f"{dataset}.h5ad"))
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(os.path.join(file_fold, dataset, 'metadata.csv'), index_col=0)
    print("\n=== Ann_df 的列名 ===")
    print(Ann_df.columns)
    print("\n=== Ann_df 的前5行 ===")
    print(Ann_df.head())
    Ann_df = Ann_df[['celltype']]
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'celltype'].astype('category')
    adata = adata[adata.obs['celltype']!='unknown']
    adata.X = adata.X.astype('float32')
    if 'spatial' in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['spatial'].astype('float32')
    
    adata.layers['count'] = adata.X.copy()
    min_gene = 20
    min_cell = 20
    sc.pp.filter_cells(adata, min_genes=min_gene)
    sc.pp.filter_genes(adata, min_cells=min_cell)
    sc.pp.normalize_total(adata, target_sum=1e4)  
    sc.pp.log1p(adata)
    
    import numpy as np
    adata.X = adata.X.astype(np.float32)
    
    h5ad_path = os.path.join(save_path, f"{dataset}.h5ad")
    adata.write_h5ad(h5ad_path)  
    data_list.append(h5ad_path)
    Batch_list.append(adata)
    print(f"Saved {h5ad_path}")


####  conda activate Spatialign   
import sys
sys.path.append('/root/BE/Spatialign/spatialign')
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

#data_list = [
#    '/data/ZhaoMH/ST0507/RAW_SLICE/merfish_mouse1/data/MERFISH_converted/Slice_7/Slice_7.h5ad',
#    '/data/ZhaoMH/ST0507/RAW_SLICE/merfish_mouse1/data/MERFISH_converted/Slice_8/Slice_8.h5ad',
#    '/data/ZhaoMH/ST0507/RAW_SLICE/merfish_mouse1/data/MERFISH_converted/Slice_9/Slice_9.h5ad',
#    '/data/ZhaoMH/ST0507/RAW_SLICE/merfish_mouse1/data/MERFISH_converted/Slice_10/Slice_10.h5ad',
#    '/data/ZhaoMH/ST0507/RAW_SLICE/merfish_mouse1/data/MERFISH_converted/Slice_11/Slice_11.h5ad'
#]
dataset_names = ['slice_39', 'slice_44']


print("Loading datasets for cell counting...")
total_cells = 0
for i, data_path in enumerate(data_list):
    temp_adata = sc.read_h5ad(data_path)
    cells_count = temp_adata.n_obs
    total_cells += cells_count
    print(f"  {dataset_names[i]}: {cells_count} cells")
    del temp_adata  

print(f"Total cells across datasets: {total_cells}")



import numpy as np
import scipy.sparse as sp

adata_list = []

for data_path in data_list:

    adata = sc.read_h5ad(data_path)
    print(f"Before: {adata.X.dtype}")

    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.array(adata.X, dtype=np.float32)
    print(f"After: {adata.X.dtype}")

    adata_list.append(adata)

import scanpy as sc
import numpy as np
import scipy.sparse as sp
import os

new_data_list = []

for data_path in data_list:

    adata = sc.read_h5ad(data_path)

    print(f"Before: {adata.X.dtype}")

    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.array(adata.X, dtype=np.float32)

    print(f"After: {adata.X.dtype}")
    new_path = data_path.replace(".h5ad", "_float32.h5ad")
    adata.write(new_path)

    new_data_list.append(new_path)

    print(f"Saved: {new_path}")
    
print("=======================================================")
    
model = Spatialign(
    *new_data_list,
    batch_key='batch',
    is_norm_log=True,
    is_scale=False,
    n_neigh=15,
    is_undirected=True,
    latent_dims=100,
    seed=42,
    gpu=0,
    save_path="/root/ST0507/new_model/spatialign/results/",
    is_verbose=False
)

raw_merge = AnnData.concatenate(*model.dataset.data_list)

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

correct1 = sc.read_h5ad("/root/ST0507/new_model/spatialign/results/res/correct_data0.h5ad")
correct2 = sc.read_h5ad("/root/ST0507/new_model/spatialign/results/res/correct_data1.h5ad")


merge_data = correct1.concatenate(correct2)

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
    '0': 'slice_39',
    '1': 'slice_44'
}

merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)
merge_data.obs['new_batch'] = merge_data.obs['new_batch'].astype('category')
print(list(merge_data.obs.columns))
#merge_data.obs['celltype'] = merge_data.obs['ground_truth']
merge_data = merge_data[~merge_data.obs['celltype'].isna()]
merge_data.X = np.nan_to_num(merge_data.X, nan=0.0)

print("Performing clustering...")
if 'count' not in merge_data.layers:
    merge_data.layers['count'] = merge_data.X.copy()
    
sc.pp.scale(merge_data)
X = merge_data.obsm['correct']
n_components = 14
gmm = GaussianMixture(n_components=n_components, random_state=42)
merge_data.obs['mclust'] = gmm.fit_predict(X)
merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")
merge_data.write("/root/ST0507/new_model/spatialign/results/multiple_adata_visiumST.h5ad")

with open("/root/ST0507/new_model/spatialign/results/spatialign_benchmark_visiumST.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

