import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import anndata
import scipy as sp
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

samples=['P1CRC', 'P2CRC', 'P5CRC']
file_fold = '/data_hou/ST_data_new/Visium_crc/'
file_bin = 'binned_outputs/square_008um/'
file_meta = 'HumanColonCancer_VisiumHD/MetaData/'

Batch_list = []

# ================== 1. Data Loading and Integration ==================
for sample in samples:

    print(f"Processing {sample}...")
    
    adata_path = os.path.join(file_fold, sample, file_bin)
    adata = sc.read_visium(adata_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    adata.obs['batch'] = sample
    
    df_meta_path = os.path.join(file_fold, file_meta)
    df_meta = pd.read_parquet(df_meta_path + sample + '_Metadata.parquet')
    
    mapping_dict = dict(zip(df_meta['barcode'], df_meta['Periphery']))
    adata.obs.loc[adata.obs['batch'] == sample, 'ground_truth'] = adata.obs_names.map(mapping_dict)
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print(adata.obs[['batch', 'ground_truth']].reset_index().head(5))
    
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('ground_truth', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()
    adata.obs_names = [sample + '-' + x for x in adata.obs_names]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    Batch_list.append(adata)
    
adata_concat = anndata.concat(Batch_list, label="slice_name", keys=samples)
vf=np.array(adata_concat.var.index)
for i in np.arange(len(samples)):
    adata1=adata_concat[adata_concat.obs['slice_name']==samples[i],:]
    cells=adata1.obs_names
    mat=pd.DataFrame(adata1.X.todense(),index=cells,columns=vf)
    coord=pd.DataFrame(adata1.obsm['spatial'],index=cells,columns=['x','y'])
    meta=pd.DataFrame(adata1.obs[['ground_truth', 'batch']])
    meta = meta.rename(columns={'ground_truth': 'celltype'})
    mat.to_csv("/data_hou/ST_data_new/model-zn/spiral/data/hd_crc/"+samples[i]+"_mat.csv")
    meta.to_csv("/data_hou/ST_data_new/model-zn/spiral/data/hd_crc/"+samples[i]+"_meta.csv")
    coord.to_csv("/data_hou/ST_data_new/model-zn/spiral/data/hd_crc/"+samples[i]+"_coord.csv")
