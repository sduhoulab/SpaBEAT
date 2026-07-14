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

file_fold = '/data_hou/ST_data_new/Xenium_breast1/'
samples=['Rep1_outs', 'Rep2_outs']

celltype_path = os.path.join(file_fold, "Cell_Barcode_Type_Matrices.xlsx")

sample_to_sheet = {
    'Rep1_outs': 'Xenium R1 Fig1-5 (supervised)',
    'Rep2_outs': 'Xenium R2 Fig1-5 (supervised)'
}
print("Reading designated Excel sheets...")
sample_cell_type_maps = {}

for sample in samples:
    target_sheet = sample_to_sheet[sample]
    print(f"Loading sheet '{target_sheet}' for sample '{sample}'...")
    
    try:
        df_meta = pd.read_excel(celltype_path, sheet_name=target_sheet)
        df_meta['Barcode'] = df_meta['Barcode'].astype(str)

        sample_cell_type_maps[sample] = dict(zip(df_meta['Barcode'], df_meta['Cluster']))
        print(f"-> Successfully loaded. Total cell annotations in sheet: {len(df_meta)}")
    except Exception as e:
        raise ValueError(f"Error loading sheet '{target_sheet}' from {celltype_path}. Please check sheet names. Error: {e}")

Batch_list = []
# ================== 1. Data Loading and Integration ==================
for sample in samples:

    print(f"Processing {sample}...")
    cur_sample_path = os.path.join(file_fold, sample)
    
    adata_path = os.path.join(cur_sample_path, "cell_feature_matrix.h5")
    adata = sc.read_10x_h5(adata_path)
    adata.var_names_make_unique()
    adata.obs['batch'] = sample  # Add batch information
    
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
    
    control_genes = adata.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
    adata = adata[:, ~control_genes].copy()
    
    mapping_dict = sample_cell_type_maps[sample]
    adata.obs.loc[adata.obs['batch'] == sample, 'ground_truth'] = adata.obs_names.map(mapping_dict)
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print(adata.obs[['batch', 'ground_truth']].head(5))
    
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
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    #sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    #adata = adata[:, adata.var['highly_variable']]
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
    mat.to_csv("/data_hou/ST_data_new/model-zn/spiral/data/xe_breast/"+samples[i]+"_mat.csv")
    meta.to_csv("/data_hou/ST_data_new/model-zn/spiral/data/xe_breast/"+samples[i]+"_meta.csv")
    coord.to_csv("/data_hou/ST_data_new/model-zn/spiral/data/xe_breast/"+samples[i]+"_coord.csv")
