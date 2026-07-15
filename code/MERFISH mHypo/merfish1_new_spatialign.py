
import sys
sys.path.append('/data/ZhaoMH/ST0507/Spatialign-main/spatialign')
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
import scipy.sparse as sp

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("Starting data preprocessing...")


file_fold = "/data/ZhaoMH/ST0507/RAW_SLICE/merfish_mouse1/data/MERFISH_converted/"
datasets = ['Slice_7', 'Slice_8', 'Slice_9', 'Slice_10', 'Slice_11']
save_path = "/data/ZhaoMH/ST0507/Spatialign-main/results"
os.makedirs(save_path, exist_ok=True)

h5ad_path_list = []
dataset_names = []
total_cells = 0


for dataset in datasets:
    adata = sc.read_h5ad(os.path.join(file_fold, dataset, f"{dataset}.h5ad"))
    adata.var_names_make_unique()


    Ann_df = pd.read_csv(os.path.join(file_fold, dataset, "metadata.csv"), index_col=0)
    Ann_df = Ann_df[['ground_truth']]

    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'ground_truth'].astype('category')


    adata = adata[adata.obs['celltype'] != 'unknown'].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if sp.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = np.array(adata.X, dtype=np.float32)

    if 'spatial' in adata.obsm:
        adata.obsm['spatial'] = np.array(adata.obsm['spatial'], dtype=np.float32)

    adata.obs['batch'] = dataset

    temp_h5ad = os.path.join(save_path, f"{dataset}_prep.h5ad")
    adata.write_h5ad(temp_h5ad)
    
    h5ad_path_list.append(temp_h5ad)
    dataset_names.append(dataset)
    total_cells += adata.n_obs

    print(f"{dataset} done: {adata.n_obs} cells")


model = Spatialign(
    *h5ad_path_list,
    batch_key='batch',
    is_norm_log=False,
    is_scale=False,
    n_neigh=15,
    is_undirected=True,
    latent_dims=50,
    seed=42,
    gpu=0,
    save_path=save_path + "/", 
    is_verbose=False
)


raw_merge = AnnData.concatenate(*model.dataset.data_list)


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

correct1 = sc.read_h5ad("/data/ZhaoMH/ST0507/Spatialign-main/results/res/correct_data0.h5ad")
correct2 = sc.read_h5ad("/data/ZhaoMH/ST0507/Spatialign-main/results/res/correct_data1.h5ad")
correct3 = sc.read_h5ad("/data/ZhaoMH/ST0507/Spatialign-main/results/res/correct_data2.h5ad")
correct4 = sc.read_h5ad("/data/ZhaoMH/ST0507/Spatialign-main/results/res/correct_data3.h5ad")
correct5 = sc.read_h5ad("/data/ZhaoMH/ST0507/Spatialign-main/results/res/correct_data4.h5ad")

merge_data = correct1.concatenate(correct2, correct3, correct4, correct5)

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
    'n_datasets': len(datasets),
    'datasets': dataset_names,
    'device': 'GPU 0',
    'random_seed': 42,
    'latent_dims': 100,
    'timestamp': pd.Timestamp.now().isoformat()
}

batch_mapping = {
    '0': 'Slice_7',
    '1': 'Slice_8',
    '2': 'Slice_9',
    '3': 'Slice_10',
    '4': 'Slice_11'
}

merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)
merge_data.obs['new_batch'] = merge_data.obs['new_batch'].astype('category')
print(list(merge_data.obs.columns))
merge_data.obs['celltype'] = merge_data.obs['ground_truth']
merge_data = merge_data[~merge_data.obs['celltype'].isna()]
merge_data.X = np.nan_to_num(merge_data.X, nan=0.0)

print("Performing clustering...")
sc.pp.scale(merge_data)
X = merge_data.obsm['correct']
n_components = 5
gmm = GaussianMixture(n_components=n_components, random_state=42)
merge_data.obs['mclust'] = gmm.fit_predict(X)
merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")
merge_data.write("/data/ZhaoMH/ST0507/Spatialign-main/results/multiple_adata_merfish1.h5ad")

with open("/data/ZhaoMH/ST0507/Spatialign-main/results/spatialign_benchmark_merfish1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

