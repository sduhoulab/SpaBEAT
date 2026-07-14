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

adata_ref = sc.read_10x_h5("/data_hou/ST_data_new/Flex_crc/HumanColonCancer_Flex_Multiplex_count_filtered_feature_bc_matrix.h5")
adata_ref.var_names_make_unique()

meta = pd.read_csv("/data_hou/ST_data_new/Visium_crc/HumanColonCancer_VisiumHD/MetaData/SingleCell_MetaData.csv", index_col=0)
meta = meta[meta['QCFilter'] != "Remove"]
celltype_list_use = meta['Level1'].unique().tolist()
common_cells = adata_ref.obs_names.intersection(meta.index)
adata_ref = adata_ref[common_cells].copy()
meta = meta.loc[common_cells]
adata_ref.obs = meta
adata_ref.obs["group"] = adata_ref.obs["Patient"]
adata_ref.obs["celltype"] = adata_ref.obs["Level1"]

file_fold = '/data_hou/ST_data_new/Visium_crc/'
file_bin = 'binned_outputs/square_008um'
file_meta = 'HumanColonCancer_VisiumHD/MetaData/'
datasets = ['P1CRC', 'P2CRC', 'P5CRC']

adatas = []
total_cells = 0

for dataset in datasets:
    adata_path = os.path.join(file_fold, dataset, file_bin)
    adata = sc.read_visium(adata_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    
    df_meta_path = os.path.join(file_fold, file_meta)
    df_meta = pd.read_parquet(df_meta_path + dataset + '_Metadata.parquet')
    print(f"meta shape (rows, columns): {df_meta.shape}")
    print("\n--- df_meta head ---")
    print(df_meta.head())
    
    mapping_dict = dict(zip(df_meta['barcode'], df_meta['Periphery']))
    adata.obs['layer'] = adata.obs_names.map(mapping_dict)
    adata = adata[adata.obs['layer'].notna()]
    adata.obs['slice_id'] = dataset
    
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

adata_stitched = STitch3D.utils.align_spots(adatas, data_type = 'HD', plot=True)


adata, adata_basis = STitch3D.utils.preprocess(adata_stitched,
                                                  adata_ref,
                                                  celltype_ref=celltype_list_use,
                                                  sample_col="group",
                                                  slice_dist_micron=[10., 300.],
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
save_path = "/data_hou/ST_data_new/model-zn/STitch3D/results/hd_crc/"
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
    'slice_dist_micron': [10., 300.],
    'device': str(used_device),
    'timestamp': pd.Timestamp.now().isoformat()
}

from sklearn.mixture import GaussianMixture
np.random.seed(1234)
gm = GaussianMixture(n_components=3, covariance_type='tied', init_params='kmeans')
y = gm.fit_predict(model.adata_st.obsm['latent'], y=None)
model.adata_st.obs["GM"] = y
model.adata_st.obs["GM"].to_csv(os.path.join(save_path, "clustering_result.csv"))
adata.obs["new_batch"] = adata.obs["slice_id"].astype(str)
adata.obs['new_batch'] = adata.obs['new_batch'].str.replace('.0', '', regex=False)
adata.obs['cluster'] = adata.obs['GM'].astype('category')
adata.obs['celltype'] = adata.obs['layer'].astype('category')

adata.write("/data_hou/ST_data_new/model-zn/STitch3D/results/hd_crc/hd_crc_stitch3d_adata1.h5ad")

with open("/data_hou/ST_data_new/model-zn/STitch3D/results/hd_crc/hd_crc_stitch3d_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)
