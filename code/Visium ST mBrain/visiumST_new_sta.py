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

total_cells = 0
Batch_list = []
adj_list = []
datasets = ['slice_39', 'slice_44']
file_fold = '/root/ST0507/data/visumST/Visium_converted/'
adatas = []

for dataset in datasets:   
    print(f"   Processing dataset: {dataset}")
    temp_batch = dataset.split('/')[-1]
    adata = sc.read_h5ad(os.path.join(file_fold, dataset, f"{temp_batch}.h5ad"))
    adata.var_names_make_unique()
    print(adata.obsm.keys())
    print(adata.obsm['spatial'][:5] if 'spatial' in adata.obsm else "NO SPATIAL")
    print(np.min(adata.obsm['spatial'], axis=0))
    print(np.max(adata.obsm['spatial'], axis=0))
    from sklearn.metrics import pairwise_distances

    coords = adata.obsm['spatial']
    dist = pairwise_distances(coords)

    print("min dist:", np.min(dist[np.nonzero(dist)]))
    print("median dist:", np.median(dist))

    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold, dataset, 'metadata.csv'), index_col=0)
    if 'ground_truth' in Ann_df.columns:
        label_col = 'ground_truth'
    else:
        label_col = 'celltype'

    Ann_df = Ann_df[[label_col]]

    adata.obs[label_col] = Ann_df.loc[adata.obs_names, label_col].astype('category')
    
    # make spot name unique
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  # Add batch information
    
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=500)
    STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
    
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    total_cells += adata.n_obs
    print(f"   {dataset}: {adata.n_obs} cells processed")

adata_concat = ad.concat(Batch_list, label="slice_name", keys=datasets)
adata_concat.obs['celltype'] = adata_concat.obs['celltype'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print(f'adata_concat.shape: {adata_concat.shape}')

# adj
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1, len(datasets)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

# =============== STAligner training ===============
print("Starting STAligner Core Training Benchmark...")

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

# =============== Saving benchmarking results ===============
benchmark_results = {
    'method_name': 'STAligner',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'total_genes': adata_concat.n_vars,
    'embedding_dim': adata_concat.obsm['STAligner'].shape[1],
    'n_datasets': len(datasets),
    'random_seed': 50,
    'hvg_genes': 5000,
    'knn_neigh': 100,
    'rad_cutoff': 150,
    'timestamp': pd.Timestamp.now().isoformat(),
    'device': str(used_device)
}
print(adata_concat.obsm['STAligner'].shape)
print(adata_concat.obsm['STAligner'].dtype)

import numpy as np

X = adata_concat.obsm['STAligner']

print("NaN:", np.isnan(X).sum())
print("Inf:", np.isinf(X).sum())

print(len(X.shape))

mclust_R_local(adata_concat, num_cluster=14, used_obsm='STAligner')
adata_concat = adata_concat[adata_concat.obs['celltype']!='unknown']
adata_concat.obs["new_batch"] = adata_concat.obs["batch_name"].astype('category')

adata_concat.write("/root/ST0507/new_model/STAligner/results/staligner_visiumST.h5ad")

with open("/root/ST0507/new_model/STAligner/results/staligner_benchmark_visiumST.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)
