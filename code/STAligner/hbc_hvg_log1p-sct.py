import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('..')
import os
import time
import psutil
import gc
import json
import torch

import numpy as np
import scipy.linalg
import pandas as pd
import anndata as ad
import scanpy as sc
import STAligner
from STAligner import ST_utils


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def load_and_preprocess(hvg_genes, preprocess_variant):
    file_fold = os.path.join('..', '..', 'RAW_SLICE', 'hbc')
    datasets = ['section1', 'section2']
    Batch_list = []
    adj_list = []
    total_cells = 0

    for dataset in datasets:
        print(f"   Processing dataset: {dataset}")
        adata = sc.read_visium(
            os.path.join(file_fold, dataset),
            count_file='filtered_feature_bc_matrix.h5',
            load_images=True,
        )
        adata.var_names_make_unique()
        adata.obs_names = [x + '_' + dataset for x in adata.obs_names]
        adata.obs['batch'] = dataset

        STAligner.Cal_Spatial_Net(adata, rad_cutoff=500)
        # STAligner.Stats_Spatial_Net(adata)

        if preprocess_variant == 'log1p':
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=hvg_genes)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        elif preprocess_variant == 'sct':
            adata.layers['counts'] = adata.X.copy()
            sc.experimental.pp.recipe_pearson_residuals(
                adata,
                n_top_genes=hvg_genes,
                theta=100,
                clip=None,
                inplace=True,
            )
        else:
            raise ValueError(f"Unknown preprocess variant: {preprocess_variant}")

        if 'highly_variable' not in adata.var:
            raise ValueError('Preprocessing did not produce highly_variable genes.')

        adata = adata[:, adata.var['highly_variable']].copy()
        adj_list.append(adata.uns['adj'])
        Batch_list.append(adata)
        total_cells += adata.n_obs
        print(f"    {dataset}: {adata.n_obs} cells processed")

    adata_concat = ad.concat(Batch_list, label='slice_name', keys=datasets)
    adata_concat.obs['batch_name'] = adata_concat.obs['slice_name'].astype('category')
    print(f'   Combined shape: {adata_concat.shape}')

    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1, len(datasets)):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

    adata_concat.uns['edgeList'] = np.nonzero(adj_concat)
    return adata_concat, total_cells


def run_hbc(hvg_genes, preprocess_variant, output_dir, device):
    print(f"\n=== Running HBC variant={preprocess_variant}, hvg={hvg_genes} ===")
    adata_concat, total_cells = load_and_preprocess(hvg_genes, preprocess_variant)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_before = get_memory_usage()
    training_start_time = time.time()

    print('Initializing STAligner model...')
    adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=100, device=device)
    edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
    adata_concat.uns['edgeList'] = edge_list

    training_end_time = time.time()
    memory_after = get_memory_usage()
    training_time = training_end_time - training_start_time
    memory_used = memory_after - memory_before

    print('Training completed!')

    ST_utils.mclust_R(adata_concat, num_cluster=10, used_obsm='STAligner')
    new_obs_names = []
    for i, obs_name in enumerate(adata_concat.obs_names):
        base_name = obs_name.rsplit('-', 1)[0]
        batch = adata_concat.obs['batch'].iloc[i]
        suffix = '-1' if batch == 'section1' else '-2'
        new_obs_names.append(base_name + suffix)

    adata_concat.obs_names = new_obs_names
    adata_concat.obs['celltype'] = 'Unknown'

    file_fold = os.path.join('..', '..', 'RAW_SLICE', 'hbc')
    for dataset in ['section1', 'section2']:
        suffix = '-1' if dataset == 'section1' else '-2'
        meta_path = os.path.join(file_fold, dataset, 'metadata.csv')
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path, index_col=0)
            meta.index = [idx.rsplit('-', 1)[0] + suffix for idx in meta.index]
            common_barcodes = adata_concat.obs_names.intersection(meta.index)
            for barcode in common_barcodes:
                adata_concat.obs.loc[barcode, 'celltype'] = meta.loc[barcode, 'celltype']

    adata_concat.obs['celltype'] = adata_concat.obs['celltype'].fillna('Unknown')
    adata_concat.obs['new_batch'] = adata_concat.obs['batch_name'].astype('category')

    output_base = f'staligner_hbc_{preprocess_variant}_hvg{hvg_genes}'
    output_h5ad = os.path.join(output_dir, f'{output_base}.h5ad')
    output_json = os.path.join(output_dir, f'{output_base}_benchmark.json')

    os.makedirs(output_dir, exist_ok=True)
    adata_concat.write(output_h5ad)

    benchmark_results = {
        'method_name': 'STAligner',
        'dataset_group': 'hbc',
        'preprocess_variant': preprocess_variant,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'training_time_hours': training_time / 3600,
        'memory_usage_mb': memory_used,
        'memory_usage_gb': memory_used / 1024,
        'total_cells': total_cells,
        'total_genes': adata_concat.n_vars,
        'embedding_dim': int(adata_concat.obsm['STAligner'].shape[1]),
        'n_datasets': 2,
        'random_seed': 50,
        'hvg_genes': hvg_genes,
        'knn_neigh': 100,
        'rad_cutoff': 500,
        'n_clusters': 10,
        'timestamp': pd.Timestamp.now().isoformat(),
        'device': str(device),
    }

    with open(output_json, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    print(f'  Saved {output_h5ad}')
    print(f'  Saved {output_json}')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    hvg_values = [1000, 3000, 5000]
    preprocess_variants = ['log1p', 'sct']
    output_dir = os.path.join('..', 'results')

    for variant in preprocess_variants:
        for hvg_genes in hvg_values:
            run_hbc(
                hvg_genes=hvg_genes,
                preprocess_variant=variant,
                output_dir=output_dir,
                device=device,
            )


if __name__ == '__main__':
    main()
