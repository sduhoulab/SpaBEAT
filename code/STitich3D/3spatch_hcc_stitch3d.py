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

adata_ref_path = "/data_hou/ST_data_new/Spatch/scrna/HCC/adata.h5ad"
adata_ref = sc.read_h5ad(adata_ref_path)
adata_ref.var_names_make_unique()
print("--- AnnData Summary ---")
print(adata_ref)
print(adata_ref.var_names[:5])
adata_ref.var = pd.DataFrame(index=adata_ref.var_names)
adata_ref.obs["celltype"] = adata_ref.obs["major_annotation"]
celltype_list_use = adata_ref.obs["celltype"].unique().tolist()

all_clusters = set()
adatas = []
total_cells = 0

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
n_clusters = len(all_clusters)
print(f"Clusters {n_clusters}")
#print(f"Xenium coordinate range (min/max): {xe_adata.obsm['spatial'].min()} to {xe_adata.obsm['spatial'].max()}")
#print(f"Visium HD coordinate range (min/max): {hd_adata.obsm['spatial'].min()} to {hd_adata.obsm['spatial'].max()}")

name_to_adata = {
    'hd': hd_adata,
    'xe': xe_adata
}
for name, adata in name_to_adata.items():
    print(f"Processing {name}...")
    adata.var_names_make_unique()
    adata.obs['slice_id'] = name # Add batch information
    print("--- adata summary ---")
    print(adata)
    print("\n--- Adata Barcode (Obs Names) head ---")
    print(adata.obs_names[:5].tolist())
    
    if name == 'xe':
        control_genes = adata.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
        adata = adata[:, ~control_genes].copy()
        print("--- Filtered adata summary ---")
        print(adata)
    
    adata.obs.loc[adata.obs['slice_id'] == name, 'layer'] = adata.obs['annotation']
    adata = adata[adata.obs['layer'].notna()]
    print(adata.obs[['slice_id', 'layer']].reset_index().head(5))
    
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
    print(f"  {name}: {adata.n_obs} cells, {adata.n_vars} genes")

adata_stitched = STitch3D.utils.align_spots(adatas, data_type = 'Xenium', plot=True)


adata, adata_basis = STitch3D.utils.preprocess(adata_stitched,
                                                  adata_ref,
                                                  celltype_ref=celltype_list_use,
                                                  sample_col=None,
                                                  slice_dist_micron=[10.],
                                                  n_hvg_group=500)

save_path = "/data_hou/ST_data_new/model-zn/STitch3D/results/spatch_hcc/"

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
    'n_datasets': len(name_to_adata),
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

adata.write(os.path.join(save_path, "spatch_hcc_stitch3d_adata1.h5ad"))

with open(os.path.join(save_path, "spatch_hcc_stitch3d_benchmark1.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)
