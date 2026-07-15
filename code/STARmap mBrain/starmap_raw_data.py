import scanpy as sc
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import anndata as ad


file_fold = '/data/ZhaoMH/ST0507/RAW_SLICE/STARmap_mouse/STARMAP_converted/'
datasets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
save_path = f'/data/ZhaoMH/ST0507/DATA_RAW/raw_adata_starmap.h5ad'

adatas = [] 
for dataset in datasets:
    print(f"Processing {dataset}...")
    batch_name = dataset.split('/')[-1]
    
    adata = sc.read_h5ad(os.path.join(file_fold, dataset, f"{batch_name}.h5ad"))
    adata.var_names_make_unique()
    adata.obs['new_batch'] = dataset

    Ann_df = pd.read_csv(os.path.join(file_fold, dataset, 'metadata.csv'), index_col=0)
    Ann_df = Ann_df[['ground_truth']]
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'ground_truth'].astype('category')
    adata = adata[adata.obs['celltype'] != 'unknown']

    adata.X = adata.X.astype('float32')
    if 'spatial' in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['spatial'].astype('float32')

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]

    adatas.append(adata)


raw_adata = ad.concat(adatas, join='outer', label='new_batch', keys=datasets)

sc.tl.pca(raw_adata, n_comps=20)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=5, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")

raw_adata.write(save_path)
