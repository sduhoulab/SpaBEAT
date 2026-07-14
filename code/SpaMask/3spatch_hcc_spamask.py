import sys
sys.path.append("/data_hou/BE/SpaMask-main/")

import os
import torch
import time
import psutil
import gc
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import SpaMask as stm

import warnings
warnings.filterwarnings('ignore')

from SpaMask import utils

utils.fix_seed(seed=42)

# ====================== BenchmarkTracker ======================
class BenchmarkTracker:
    def __init__(self, method_name):
        self.method_name = method_name
        self.process = psutil.Process(os.getpid())

    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.mem_before = self.process.memory_info().rss / (1024 * 1024)
        self.start_time = time.time()
        print(f"\n[{self.method_name}] Starting core training benchmark...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.train_time = time.time() - self.start_time
        self.mem_used = (self.process.memory_info().rss / (1024 * 1024)) - self.mem_before
        self.gpu_mem = torch.cuda.max_memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0
        print(f"Time: {self.train_time:.2f}s | CPU RAM: {self.mem_used:.2f}MB | GPU Peak: {self.gpu_mem:.2f}MB\n")

    def save_report(self, json_path, adata, embed_key, extra_meta={}):
        results = {'method_name': self.method_name, 'training_time_seconds': self.train_time,
                   'memory_usage_mb': self.mem_used, 'gpu_peak_memory_mb': self.gpu_mem,
                   'total_cells': adata.n_obs, 'timestamp': pd.Timestamp.now().isoformat()}
        results.update(extra_meta)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

# ====================== SpaMask Utility Functions ======================
def best_fit_transform(A, B):
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t

def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    m = A.shape[1]
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0
    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)
        src = np.dot(T, src)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    T, _, _ = best_fit_transform(A, src[:m, :].T)
    return T, distances, i

def transform(point_cloud, T):
    point_cloud_align = np.ones((point_cloud.shape[0], 3))
    point_cloud_align[:, 0:2] = np.copy(point_cloud)
    point_cloud_align = np.dot(T, point_cloud_align.T).T
    return point_cloud_align[:, :2]

def align_spots(adata_st_list_input, method="icp", data_type="Visium", coor_key="spatial", tol=0.01, test_all_angles=False, plot=False):
    adata_st_list = adata_st_list_input.copy()
    if (method == "icp") or (method == "ICP"):
        point_cloud_list = []
        for adata in adata_st_list:
            if 'in_tissue' in adata.obs.columns:
                adata = adata[adata.obs['in_tissue'] == 1]
            if data_type=="Visium":    
              if 'array_row' in adata.obs.columns and 'array_col' in adata.obs.columns:
                  print("array_row, array_col yes")
                  loc_x = np.array(adata.obs.loc[:, ["array_row"]]) * np.sqrt(3)
                  loc_y = np.array(adata.obs.loc[:, ["array_col"]])
                  loc = np.concatenate((loc_x, loc_y), axis=1)
                  pairwise_loc_distsq = np.sum((loc.reshape([1, -1, 2]) - loc.reshape([-1, 1, 2])) ** 2, axis=2)
                  n_neighbors = np.sum(pairwise_loc_distsq < 5, axis=1) - 1
                  edge = ((n_neighbors > 1) & (n_neighbors < 5)).astype(np.float32)
            elif data_type=="Visium HD":
                if 'array_row' in adata.obs.columns and 'array_col' in adata.obs.columns:
                    print("array_row, array_col yes")
                    loc_x = adata.obs.loc[:, ["array_row"]]
                    loc_x = np.array(loc_x)
                    loc_y = adata.obs.loc[:, ["array_col"]]
                    loc_y = np.array(loc_y)
                    loc = np.concatenate((loc_x, loc_y), axis=1)
                    pairwise_loc_distsq = np.sum((loc.reshape([1,-1,2]) - loc.reshape([-1,1,2])) ** 2, axis=2)
                    min_distsq = np.sort(np.unique(pairwise_loc_distsq), axis=None)[1]
                    n_neighbors = np.sum(pairwise_loc_distsq < (min_distsq * 3), axis=1) - 1
                    edge = ((n_neighbors > 1) & (n_neighbors < 7)).astype(np.float32)
            else:
                print("array_row, array_col not found")
                edge = np.ones(adata.shape[0], dtype=np.float32)
            
            point_cloud_list.append(adata.obsm[coor_key][edge == 1].copy())

        trans_list = []
        adata_st_list[0].obsm["spatial_aligned"] = adata_st_list[0].obsm[coor_key].copy()
        for i in range(len(adata_st_list) - 1):
            T, _, _ = icp(point_cloud_list[i + 1], point_cloud_list[i], tolerance=tol)
            trans_list.append(T)
        for i in range(len(adata_st_list) - 1):
            point_cloud_align = adata_st_list[i + 1].obsm[coor_key].copy()
            for T in trans_list[:(i + 1)][::-1]:
                point_cloud_align = transform(point_cloud_align, T)
            adata_st_list[i + 1].obsm["spatial_aligned"] = point_cloud_align
    return adata_st_list

def preprocess(adata_st_list, section_ids=None, three_dim_coor=None, coor_key="spatial_aligned", rad_cutoff=None, rad_coef=1.5, k_cutoff=12, slice_dist_micron=None, c2c_dist=100, model='KNN'):
    adata_st = ad.concat(adata_st_list, label="slice_name", keys=section_ids)
    adata_st.obs['Ground Truth'] = adata_st.obs['Ground Truth'].astype('category')
    adata_st.obs["batch_name"] = adata_st.obs["slice_name"].astype('category')

    adata_st_ref = adata_st_list[0].copy()
    loc_ref = np.array(adata_st_ref.obsm[coor_key])
    min_dist_ref = np.sort(np.unique(pairwise_distances(loc_ref)), axis=None)[1]
    if rad_cutoff is None:
        rad_cutoff = min_dist_ref * rad_coef

    loc_xy = pd.DataFrame(adata_st.obsm['spatial_aligned']).values
    loc_z = np.zeros(adata_st.shape[0])
    if slice_dist_micron is not None:
        dim = 0
        for i in range(len(slice_dist_micron)):
            dim += adata_st_list[i].shape[0]
            loc_z[dim:] += slice_dist_micron[i] * (min_dist_ref / c2c_dist)
    loc = np.concatenate([loc_xy, loc_z.reshape(-1, 1)], axis=1)
    loc = pd.DataFrame(loc)
    loc.index = adata_st.obs.index

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(loc)
        distances, indices = nbrs.kneighbors(loc)
        KNN_list = [pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])) for it in range(indices.shape[0])]

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = KNN_df.loc[KNN_df['Distance'] > 0,].copy()
    id_cell_trans = dict(zip(range(loc.shape[0]), np.array(loc.index)))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    adata_st.uns['Spatial_Net'] = Spatial_Net
    return adata_st

def train_one(args, adata, num_clusters):
    net = stm.spaMask.SPAMASK(adata,
                     tissue_name='Donor',
                     num_clusters=num_clusters,
                     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                     learning_rate=args.learning_rate,
                     weight_decay=args.weight_decay,
                     max_epoch=args.max_epoch,
                     gradient_clipping=args.gradient_clipping,
                     feat_mask_rate=args.feat_mask_rate,
                     edge_drop_rate=args.edge_drop_rate,
                     hidden_dim=args.hidden_dim,
                     latent_dim=args.latent_dim,
                     bn=args.bn,
                     att_dropout_rate=args.att_dropout_rate,
                     fc_dropout_rate=args.fc_dropout_rate,
                     use_token=args.use_token,
                     rep_loss=args.rep_loss,
                     rel_loss=args.rel_loss,
                     alpha=args.alpha,
                     lam=args.lam,
                     random_seed=args.seed,
                     nps=args.nps)
    net.train()
    net.process(method="mclust")
    return net.get_adata()

# ======================================================================
# Task Configuration & CLI Setup
# ======================================================================
TASKS = {
    'spatch_hcc': {'root': None, 'slices': ['hd', 'xe'], 'n_domains': 17, 'type': 'spatch', 'suffix': 'spatch_hcc'}
}

parser = argparse.ArgumentParser(description="Run SpaMask Benchmark for a specific task.")
parser.add_argument(
    '--task',
    type=str,
    required=True,
    choices=list(TASKS.keys()),
    help="Specify the task to run. Available tasks: " + ", ".join(TASKS.keys())
)
parser.add_argument('--n_domains', type=int, default=None, help='Number of domains for clustering')

args_cli = parser.parse_args()
CURRENT_TASK = args_cli.task
config = TASKS[CURRENT_TASK]

final_n_domains = args_cli.n_domains if args_cli.n_domains is not None else config['n_domains']
#data_root = Path(config['root'])
proj_list = config['slices']
save_path = Path('/data_hou/ST_data_new/model-zn/spamask/results/')
save_path.mkdir(parents=True, exist_ok=True)

all_clusters = set()

print("Starting preprocessing Visium HD data")
hd_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_hcc_hd_ff/adata.h5ad"
hd_adata = sc.read_h5ad(hd_adata_path)
hd_celltypes = hd_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(hd_celltypes)

print("Starting preprocessing Xe data")
xe_adata_path = "/data_hou/ST_data_new/Spatch/transcriptome_hcc_xenium/adata.h5ad"
xe_adata = sc.read_h5ad(xe_adata_path)
xe_celltypes = xe_adata.obs['annotation'].dropna().astype(str).unique()
all_clusters.update(xe_celltypes)
n_clusters = len(all_clusters)
print(f"Clusters {n_clusters}")


print(f"=== Running SpaMask Benchmark for: {CURRENT_TASK} (n_domains = {final_n_domains}) ===")

# ================== SpaMask Hyperparameters ==================
args = stm.utils.build_args()
args.hidden_dim, args.latent_dim = 512, 256
args.max_epoch = 1000
args.lam = 2
args.feat_mask_rate = 0.5
args.edge_drop_rate = 0.2
args.top_genes = 5000
args.rad_cutoff = 200
args.k_cutoff = 12
args.model = 'KNN'

Batch_list = []

# ================== 1. Data Loading and Integration ==================
name_to_adata = {
    'hd': hd_adata,
    'xe': xe_adata
}

for name, adata in name_to_adata.items():
    print(f"Processing {name}...")
    
    adata.var_names_make_unique()
    adata.obs['batch_name'] = name
    
    if name == 'xe':
        control_genes = adata.var_names.str.startswith(('BLANK_', 'NegControl_', 'Control_', 'antisense_'))
        adata = adata[:, ~control_genes].copy()
        print("--- Filtered adata summary ---")
        print(adata)
    
    adata.obs.loc[adata.obs['batch_name'] == name, 'ground_truth'] = adata.obs['annotation'].astype('category')
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print(adata.obs[['batch_name', 'ground_truth']].reset_index().head(5))
    adata.obs['Ground Truth'] = adata.obs['ground_truth']
    target_cells = 15000
    if adata.n_obs > target_cells:
        print(f"Stratified Subsampling from {adata.n_obs} to approximately {target_cells} cells...")
        sampling_fraction = target_cells / adata.n_obs
        df_obs = adata.obs.copy()
        stratified_indices = df_obs.groupby('ground_truth', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=50)
        ).index
        adata = adata[stratified_indices].copy()

    Batch_list.append(adata)

# ================== 2. Spatial Alignment and 3D Graph Construction ==================
print("Performing ICP alignment and 3D graph construction...")
Batch_list = align_spots(Batch_list, data_type='Xenium', method='icp', plot=False)

slice_dist_micron = [10] * (len(proj_list) - 1) if len(proj_list) > 1 else None
adata = preprocess(Batch_list, section_ids=proj_list, k_cutoff=args.k_cutoff, rad_cutoff=args.rad_cutoff, model=args.model, slice_dist_micron=slice_dist_micron)

# ================== 3. Standard Preprocessing ==================
adata.layers['count'] = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X.copy()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=args.top_genes)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)

adata.obsm['feat'] = PCA(n_components=50, random_state=42).fit_transform(adata.X)

# ================== 4. Core Model Training with Benchmarking ==================
with BenchmarkTracker('SpaMask') as tracker:
    adata = train_one(args, adata, final_n_domains)

if 'eval_pred' in adata.obsm:
    adata.obsm['SpaMask_embed'] = np.ascontiguousarray(adata.obsm['eval_pred'], dtype=np.float64)
    print("Successfully extracted and saved embedding to adata.obsm['SpaMask_embed']")
else:
    print("Warning: 'eval_pred' not found in model output! Embedding extraction failed.")

# ================== 5. Standardize Metadata for Evaluation ==================
if 'ground_truth' in adata.obs:
    adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
elif 'Ground Truth' in adata.obs:
    adata.obs['celltype'] = adata.obs['Ground Truth'].astype('category')
else:
    print("[Warning] No ground truth/celltype found, biological metrics may fail!")

# ================== 6. Save Results ==================
tracker.save_report(
    json_path=save_path / f"spamask_benchmark_{config['suffix']}.json",
    adata=adata,
    embed_key='SpaMask_embed',
    extra_meta={'n_datasets': len(proj_list)}
)

adata.write(save_path / f"multiple_adata_{config['suffix']}_spamask.h5ad")
print(f"Successfully finished {CURRENT_TASK} and saved to {save_path}/multiple_adata_{config['suffix']}_spamask.h5ad")