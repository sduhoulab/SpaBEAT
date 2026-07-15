import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.mixture import GaussianMixture
import anndata as ad


def process_her2_type1(sample_letter):

    if sample_letter in ["A", "B", "C", "D"]:
        n_slices = 6
        datasets = [f"{sample_letter}{i}" for i in range(1, n_slices+1)]
    elif sample_letter in ["E", "F", "G", "H"]:
        n_slices = 3
        datasets = [f"{sample_letter}{i}" for i in range(1, n_slices+1)]
    else:
        raise ValueError("Only samples A-H are supported!")

    file_fold = f'/root/ST0507/data/Her2_tumor_converted/{sample_letter}/'
    save_path = f'/root/ST0507/DATA_RAW/raw_adata_her2_{sample_letter}.h5ad'

    print(f"\n===== start HER2 {sample_letter} | slice：{datasets} =====")

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
    gmm = GaussianMixture(n_components=5, random_state=42)
    raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
    raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
    raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")

    raw_adata.write(save_path)
    print(f"✅ saved：{save_path}")


def process_her2_sample():
    """
    Run only once
    Load all 8 slices: A/A1, B/B1, C/C1, D/D1, E/E1, F/F1, G/G2, H/H1
    Concatenate all slices → output one merged h5ad file
    """
    # List of all slice paths
    data_name_list = ['A/A1', 'B/B1', 'C/C1', 'D/D1', 'E/E1', 'F/F1', 'G/G2', 'H/H1']
    
    # Root directory
    base_fold = '/root/ST0507/data/Her2_tumor_converted/'
    save_path = '/root/ST0507/DATA_RAW/raw_adata_her2_sample.h5ad'

    print(f"\n===== Start processing all HER2 samples | total {len(data_name_list)} slices =====")

    adatas = []
    for slice_path in data_name_list:
        print(f"Processing {slice_path}...")
        
        # Split A/A1 → sample_letter=A, slice_name=A1
        sample_letter, slice_name = slice_path.split('/')
        batch_name = slice_path  # Batch identifier: A/A1
        
        # Path to h5ad file: A/A1/A1.h5ad
        h5ad_path = os.path.join(base_fold, sample_letter, slice_name, f"{slice_name}.h5ad")
        meta_path = os.path.join(base_fold, sample_letter, slice_name, "metadata.csv")

        # 1. Read h5ad file
        adata = sc.read_h5ad(h5ad_path)
        adata.var_names_make_unique()
        adata.obs['new_batch'] = batch_name

        # 2. Load cell type annotations
        Ann_df = pd.read_csv(meta_path, index_col=0)
        Ann_df = Ann_df[['ground_truth']]
        adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'ground_truth'].astype('category')
        adata = adata[adata.obs['celltype'] != 'unknown']

        # 3. Data type conversion
        adata.X = adata.X.astype('float32')
        if 'spatial' in adata.obsm:
            adata.obsm['spatial'] = adata.obsm['spatial'].astype('float32')

        # 4. Normalization and feature selection
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]

        adatas.append(adata)

    # ==============================
    # Concatenate all slices (perform concatenation only once)
    # ==============================
    raw_adata = ad.concat(adatas, join='outer', label='new_batch', keys=data_name_list)

    # GMM clustering (named as mclust for consistency)
    sc.tl.pca(raw_adata, n_comps=50)
    X_pca = raw_adata.obsm["X_pca"]
    gmm = GaussianMixture(n_components=5, random_state=42)
    raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
    raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
    raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")

    # Save output
    os.makedirs('/root/ST0507/DATA_RAW/', exist_ok=True)
    raw_adata.write(save_path)
    print(f"\n✅ All tasks finished!")
    print(f"📁 Single output file generated：{save_path}")
    return raw_adata


# ==============================================
# Execution entry
# ==============================================
if __name__ == "__main__":
    process_her2_sample()  
    
    process_her2_type1("A")
    process_her2_type1("B")
    process_her2_type1("C")
    process_her2_type1("D")
    process_her2_type1("E")
    process_her2_type1("F")
    process_her2_type1("G")
    # Process sample H
    process_her2_type1("H")