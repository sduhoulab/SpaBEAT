import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/root/BE/STAligner')
import STAligner
from STAligner import ST_utils
from STAligner.ST_utils import match_cluster_labels
import os
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix
import time
import psutil
import gc
import json
import torch
from sklearn.metrics import pairwise_distances

def mclust_R_local(adata, num_cluster, modelNames='EEE', used_obsm='STAligner', key_added_pred="mclust", random_seed=2024):
    import numpy as np
    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri

    np.random.seed(random_seed)
    robjects.r['set.seed'](random_seed)

    numpy2ri.activate()
    robjects.r.library("mclust")

    embed = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    embed = np.nan_to_num(embed)
    if embed.ndim == 1:
        embed = embed[:, np.newaxis]

    robjects.globalenv['mat'] = numpy2ri.numpy2rpy(embed)
    robjects.globalenv['num_cluster'] = num_cluster
    robjects.globalenv['modelNames'] = modelNames
    
    robjects.r('''
        mat <- as.matrix(mat)
        dimnames(mat) <- NULL
        res <- mclust::Mclust(mat, G=num_cluster, modelNames=modelNames)
    ''')

    res = robjects.globalenv['res']
    mclust_res = np.array(res.rx2('classification'), dtype=int)

    adata.obs[key_added_pred] = mclust_res
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    
    numpy2ri.deactivate()
    
    return adata
    
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
used_device = device  

def run_staligner_integration(ref_slice, target_slice, data_path, base_save_path, n_clusters=4):
    datasets = [ref_slice, target_slice]
    pair_name = f"{ref_slice}_vs_{target_slice}"
    
    save_path = base_save_path
    os.makedirs(save_path, exist_ok=True)
    
    total_cells = 0
    Batch_list = []
    adj_list = []
    
    print(f"\nStarting data preprocessing for {pair_name}...")

    for dataset in datasets:   
        print(f"   Processing dataset: {dataset}")
        adata = sc.read_h5ad(os.path.join(data_path, dataset, f"{dataset}.h5ad"))
        adata.var_names_make_unique()
        
        # Diagnostics
        print(adata.obsm.keys())
        print(adata.obsm['spatial'][:5] if 'spatial' in adata.obsm else "NO SPATIAL")
        if 'spatial' in adata.obsm:
            print("Spatial Min:", np.min(adata.obsm['spatial'], axis=0))
            print("Spatial Max:", np.max(adata.obsm['spatial'], axis=0))
            coords = adata.obsm['spatial']
            dist = pairwise_distances(coords)
            print("min dist:", np.min(dist[np.nonzero(dist)]))
            print("median dist:", np.median(dist))

        # Robust Metadata Loading
        metadata_file = os.path.join(data_path, dataset, 'metadata.csv')
        if os.path.exists(metadata_file):
            df_meta = pd.read_csv(metadata_file, index_col=0)
            possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label', 'Cell_class']
            for col in possible_cols:
                if col in df_meta.columns:
                    adata.obs['ground_truth'] = adata.obs_names.map(df_meta[col]).astype('category')
                    break
        
        # Filter cells missing ground truth
        if 'ground_truth' in adata.obs.columns:
            adata = adata[~pd.isnull(adata.obs['ground_truth'])]
        
        # make spot name unique
        adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
        adata.obs['batch'] = dataset  # Add batch information
        
        # Constructing the spatial network
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=500)
        STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
        
        # Normalization
        n_top = min(5000, adata.n_vars)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]
        
        adj_list.append(adata.uns['adj'])
        Batch_list.append(adata)
        total_cells += adata.n_obs
        print(f"   {dataset}: {adata.n_obs} cells processed")

    # =========================
    # Concatenation and Block Diagonal Matrix
    # =========================
    adata_concat = ad.concat(Batch_list, label="slice_name", keys=datasets)
    adata_concat.obs['celltype'] = adata_concat.obs['ground_truth'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
    print(f'adata_concat.shape: {adata_concat.shape}')

    # adj
    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1, len(datasets)):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

    adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

    # =============== STAligner training ===============
    print("\nStarting STAligner Core Training Benchmark...")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_before = get_memory_usage()
    training_start_time = time.time()

    print("Initializing STAligner model...")
    adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=100, device=used_device)
    edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
    adata_concat.uns['edgeList'] = edge_list

    training_end_time = time.time()
    memory_after = get_memory_usage()
    training_time = training_end_time - training_start_time
    memory_used = memory_after - memory_before

    print("Training completed!")

    # =============== Diagnostics & Post-processing ===============
    X = adata_concat.obsm['STAligner']
    print(f"STAligner Shape: {X.shape}")
    print(f"STAligner Dtype: {X.dtype}")
    print("NaN:", np.isnan(X).sum())
    print("Inf:", np.isinf(X).sum())

    # Mclust Clustering
    mclust_R_local(adata_concat, num_cluster=n_clusters, used_obsm='STAligner')
    adata_concat = adata_concat[adata_concat.obs['celltype'] != 'unknown']
    adata_concat.obs["new_batch"] = adata_concat.obs["batch_name"].astype('category')

    # =============== Saving benchmarking results ===============
    benchmark_results = {
        'method_name': 'STAligner',
        'dataset': pair_name,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'training_time_hours': training_time / 3600,
        'memory_usage_mb': memory_used,
        'memory_usage_gb': memory_used / 1024,
        'total_cells': total_cells,
        'final_cells': adata_concat.n_obs,
        'total_genes': adata_concat.n_vars,
        'embedding_dim': adata_concat.obsm['STAligner'].shape[1],
        'n_datasets': len(datasets),
        'random_seed': 50,
        'hvg_genes': n_top,
        'knn_neigh': 100,
        'rad_cutoff': 500,
        'timestamp': pd.Timestamp.now().isoformat(),
        'device': str(used_device)
    }

    adata_concat.write(os.path.join(save_path, f"multiple_adata_{pair_name}.h5ad"))

    with open(os.path.join(save_path, f"staligner_benchmark_{pair_name}.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n🎉 Task {pair_name} STAligner integration finished! Results saved to: {save_path}\n")
    
    del adata_concat, Batch_list, adj_list, X
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    DATA_PATH = "/root/ST0507/simulations_model/data_model/1_DLPFC_InterSlice/SRTsim_Converted/" 
    BASE_SAVE_PATH = "/root/ST0507/new_model/STAligner/results" 
    
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
   
    REF_SLICE = "Sim_Slice_1"
    
    N_CLUSTERS = 7
    for i in range(2, 7):
        target_slice = f"Sim_Slice_{i}"
        
        print("="*60)
        print(f"start: {REF_SLICE} + {target_slice}")
        print("="*60)
        
        try:
            run_staligner_integration(
                ref_slice=REF_SLICE, 
                target_slice=target_slice, 
                data_path=DATA_PATH, 
                base_save_path=BASE_SAVE_PATH, 
                n_clusters=N_CLUSTERS
            )
        except Exception as e:
            print(f"❌ Task {REF_SLICE}_vs_{target_slice} failed. Error message: {str(e)}")
            continue
            
    print("🎯 All STAligner slice pair integration tasks have been completed!")