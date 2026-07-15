import scanpy as sc
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import anndata as ad


def run_raw_integration(ref_slice, target_slice, file_fold, base_save_path):
    datasets = [ref_slice, target_slice]
    pair_name = f"{ref_slice}_vs_{target_slice}"
    
    save_path = os.path.join(base_save_path, f"multiple_adata_{pair_name}_raw.h5ad")
    
    
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

    sc.tl.pca(raw_adata, n_comps=50)
    X_pca = raw_adata.obsm["X_pca"]
    gmm = GaussianMixture(n_components=7, random_state=42)
    raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
    raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
    raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")

    raw_adata.write(save_path)


if __name__ == "__main__":
    DATA_PATH = "/root/ST0507/simulations_model/data_model/1_DLPFC_InterSlice/SRTsim_Converted/"
    BASE_SAVE_PATH = "/root/ST0507/DATA_RAW/results/"
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    
    REF_SLICE = "Sim_Slice_1"
    
    for i in range(2, 7):
        target_slice = f"Sim_Slice_{i}"
        
        try:
            run_raw_integration(
                ref_slice=REF_SLICE, 
                target_slice=target_slice, 
                file_fold=DATA_PATH, 
                base_save_path=BASE_SAVE_PATH
            )
        except Exception as e:
            print(f"❌ Task {REF_SLICE}_vs_{target_slice} failed. Error message: {str(e)}")
            continue
                    
    print("\n🎯 All RAW slice pair extraction tasks have been completed!")