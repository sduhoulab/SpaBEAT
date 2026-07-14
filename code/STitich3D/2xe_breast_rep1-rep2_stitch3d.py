import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data_hou/ST_data_new/model-zn/STitch3D/STitch3D/')
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.io
import matplotlib.pyplot as plt
import os
import sys
import STitch3D
import warnings
warnings.filterwarnings("ignore")
import time
import psutil
import gc
import json
import torch

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
used_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(1234)

adata_ref = sc.read_10x_h5("/data_hou/ST_data_new/Xenium_breast1/Sample1_scRNA/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5")
adata_ref.var_names_make_unique()

file_fold = '/data_hou/ST_data_new/Xenium_breast1/'
datasets = ['Rep1_outs', 'Rep2_outs']
save_path = "/data_hou/ST_data_new/model-zn/STitch3D/results/xe_breast/"

celltype_path = os.path.join(file_fold, "Cell_Barcode_Type_Matrices.xlsx")
sample_to_sheet = {
    'Rep1_outs': 'Xenium R1 Fig1-5 (supervised)',
    'Rep2_outs': 'Xenium R2 Fig1-5 (supervised)',
    'sc': 'scFFPE-Seq'
}

print("Reading designated Excel sheets...")
target_sheet = sample_to_sheet['sc']
meta = pd.read_excel(celltype_path, sheet_name=target_sheet, engine='openpyxl')
celltype_list_use = meta['Annotation'].unique().tolist()
sc_dict = dict(zip(meta['Barcode'], meta['Annotation']))
adata_ref.obs['celltype'] = adata_ref.obs_names.map(sc_dict)
print(adata_ref)
adata_ref = adata_ref[adata_ref.obs['celltype'].notna()]
print(adata_ref)

all_clusters = set()
sample_cell_type_maps = {}

for dataset in datasets:
    target_sheet = sample_to_sheet[dataset]
    print(f"Loading sheet '{target_sheet}' for sample '{dataset}'...")
    
    try:
        df_meta = pd.read_excel(celltype_path, sheet_name=target_sheet, engine='openpyxl')
        df_meta['Barcode'] = df_meta['Barcode'].astype(str)
        all_clusters.update(df_meta['Cluster'].dropna().unique())

        sample_cell_type_maps[dataset] = dict(zip(df_meta['Barcode'], df_meta['Cluster']))
        print(f"-> Successfully loaded. Total cell annotations in sheet: {len(df_meta)}")
    except Exception as e:
        raise ValueError(f"Error loading sheet '{target_sheet}' from {celltype_path}. Please check sheet names. Error: {e}")
        

adatas = []
total_cells = 0
for dataset in datasets:
    print(f"Processing {dataset}...")
    cur_sample_path = os.path.join(file_fold, dataset)
    h5_path = os.path.join(cur_sample_path, "cell_feature_matrix.h5")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    adata.obs['slice_id'] = dataset
    
    parquet_path = os.path.join(cur_sample_path, "cells.parquet")
    df_cells = pd.read_parquet(parquet_path)
    df_cells['cell_id'] = df_cells['cell_id'].astype(str)
    df_cells = df_cells.set_index('cell_id')
    common_cells = adata.obs_names.intersection(df_cells.index)
    print(f"Matched {len(common_cells)} cells between H5 and Parquet.")
    
    adata = adata[common_cells].copy()
    x_col = 'x_centroid'
    y_col = 'y_centroid'
    adata.obsm["spatial"] = df_cells.loc[adata.obs_names, [x_col, y_col]].values
    adata.obs["array_col"] = adata.obsm["spatial"][:, 0]
    adata.obs["array_row"] = adata.obsm["spatial"][:, 1]
    
    mapping = sample_cell_type_maps[dataset]
    adata.obs['layer'] = adata.obs.index.map(mapping)
    adata = adata[adata.obs['layer'].notna()]
    
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('layer', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()
        
    adatas.append(adata)
    total_cells += adata.n_obs

adata_stitched = STitch3D.utils.align_spots(adatas, data_type = 'Xenium', plot=True)

adata, adata_basis = STitch3D.utils.preprocess(adata_stitched,
                                                  adata_ref,
                                                  celltype_ref=celltype_list_use,
                                                  sample_col=None,
                                                  slice_dist_micron=[10.],
                                                  n_hvg_group=500)

# =============== STicth training ===============

gc.collect()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training STitch3D model...")
model = STitch3D.model.Model(adata, adata_basis)
model.train()

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before

print("Training completed!")
result = model.eval(adatas, save=True, output_path=save_path)


# =============== Saving benchmarking results ===============
benchmark_results = {
    'method_name': 'STitch3D',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'total_genes': adata.n_vars,
    'final_cells': model.adata_st.n_obs,
    'embedding_dim': model.adata_st.obsm['latent'].shape[1],
    'n_datasets': len(datasets),
    'random_seed': 1234,
    'n_hvg_group': 500,
    'slice_dist_micron': [10.],
    'device': str(used_device),
    'timestamp': pd.Timestamp.now().isoformat()
}

from sklearn.mixture import GaussianMixture
np.random.seed(1234)
gm = GaussianMixture(n_components=len(all_clusters), covariance_type='tied', init_params='kmeans')
y = gm.fit_predict(model.adata_st.obsm['latent'], y=None)
model.adata_st.obs["GM"] = y
model.adata_st.obs["GM"].to_csv(os.path.join(save_path, "clustering_result.csv"))
adata.obs["new_batch"] = adata.obs["slice_id"].astype(str)
adata.obs['new_batch'] = adata.obs['new_batch'].str.replace('.0', '', regex=False)
adata.obs['cluster'] = adata.obs['GM'].astype('category')
adata.obs['celltype'] = adata.obs['layer'].astype('category')

adata.write(os.path.join(save_path, "xe_breast_rep1-rep2_stitch3d_adata1.h5ad"))

with open(os.path.join(save_path, "xe_breast_rep1-rep2_stitch3d_benchmark1.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)
