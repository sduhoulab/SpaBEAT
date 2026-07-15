import sys
sys.path.append("/data_hou/BE/SpaMask-main")

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
                
            if 'array_row' in adata.obs.columns and 'array_col' in adata.obs.columns:
                loc_x = np.array(adata.obs.loc[:, ["array_row"]]) * np.sqrt(3)
                loc_y = np.array(adata.obs.loc[:, ["array_col"]])
                loc = np.concatenate((loc_x, loc_y), axis=1)
                pairwise_loc_distsq = np.sum((loc.reshape([1, -1, 2]) - loc.reshape([-1, 1, 2])) ** 2, axis=2)
                n_neighbors = np.sum(pairwise_loc_distsq < 5, axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 5)).astype(np.float32)
            else:
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
    # SpaMask relies on the 'Ground Truth' column name
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
    net.process(method="mclust") # mclust results are stored in adata.obs['mclust']
    return net.get_adata()


# ======================================================================
# Task Configuration & CLI Setup
# ======================================================================
TASKS = {
    'merfish1': {'root': '/data/ZhaoMH/ST0507/RAW_SLICE/merfish_mouse1/data/MERFISH_converted/', 'slices': ['Slice_7', 'Slice_8', 'Slice_9', 'Slice_10', 'Slice_11'], 'n_domains': 8, 'type': 'merfish1', 'suffix': 'merfish1'},
    'starmap': {'root': '/data/ZhaoMH/ST0507/RAW_SLICE/STARmap_mouse/STARMAP_converted/', 'slices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'n_domains': 5, 'type': 'starmap', 'suffix': 'starmap'},
    'PFC': {'root': '/root/ST0507/data/starmap_PFC/STARMAP_converted/', 'slices': ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control'], 'n_domains': 4, 'type': 'PFC', 'suffix': 'PFC'},
    'embyro': {'root': '/data_hou/ZhaoMH/data/stereo_embryo/Embryo_converted/', 'slices': ['Slice_5', 'Slice_6', 'Slice_7', 'Slice_8', 'Slice_9'], 'n_domains': 25, 'type': 'embryo', 'suffix': 'embryo'}
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

data_root = Path(config['root'])
proj_list = config['slices']
save_path = Path('/data_hou/BE/SpaMask-main/Results')
save_path.mkdir(parents=True, exist_ok=True)

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

# ================== 1. Data Loading and Integration ==================
Batch_list = []
for proj_name in tqdm(proj_list):
    batch_name = str(proj_name).split('/')[-1]
    
    file_path = data_root / proj_name / f"{batch_name}.h5ad"
    metadata_file = data_root / proj_name / 'metadata.csv'

    adata_tmp = sc.read_h5ad(file_path)
    adata_tmp.var_names_make_unique()
    adata_tmp.obs['batch_name'] = batch_name

    if 'ground_truth' in adata_tmp.obs.columns and 'Ground Truth' not in adata_tmp.obs.columns:
        adata_tmp.obs['Ground Truth'] = adata_tmp.obs['ground_truth']

    if 'Ground Truth' not in adata_tmp.obs.columns:
        if metadata_file.exists():
            print(f"\n[Info] No ground_truth found in {proj_name}, loading from metadata.csv...")
            df_meta = pd.read_csv(metadata_file, index_col=0)
            
            possible_cols = ['ground_truth', 'layer_guess', 'celltype', 'Ground Truth', 'label', 'Cell_class']
            for col in possible_cols:
                if col in df_meta.columns:
                    adata_tmp.obs['Ground Truth'] = adata_tmp.obs_names.map(df_meta[col])
                    print(f"  [✔] Successfully loaded labels from column: '{col}'")
                    break
            else:
                print(f"  [Warning] metadata.csv found but no valid label columns detected for {proj_name}!")
        else:
            print(f"  [Error] No labels in adata and metadata.csv not found for {proj_name}!")
    
    if 'Ground Truth' in adata_tmp.obs.columns:
        adata_tmp.obs['Ground Truth'] = adata_tmp.obs['Ground Truth'].astype('category')
        matched_count = adata_tmp.obs['Ground Truth'].notna().sum()
        print(f"[{proj_name}] Total cells: {adata_tmp.n_obs} | Matched valid cells: {matched_count}\n")
    else:
        print(f"[{proj_name}] Total cells: {adata_tmp.n_obs} | No Ground Truth available.\n")
        
    fraction = 0.5
    sc.pp.subsample(adata_tmp, fraction=fraction, random_state=42)
    print(f"  [Downsample] Retained {adata_tmp.n_obs} cells ({fraction*100}%) for {proj_name}")

    adata_tmp = adata_tmp[~pd.isnull(adata_tmp.obs['Ground Truth'])].copy()

    adata_tmp.obs['ground_truth'] = adata_tmp.obs['Ground Truth']

    Batch_list.append(adata_tmp)

# ================== 2. Spatial Alignment and 3D Graph Construction ==================
print("Performing ICP alignment and 3D graph construction...")
Batch_list = align_spots(Batch_list, method='icp', plot=False)

slice_dist_micron = [10] * (len(proj_list) - 1) if len(proj_list) > 1 else None
adata = preprocess(Batch_list, section_ids=proj_list, k_cutoff=args.k_cutoff, rad_cutoff=args.rad_cutoff, model=args.model, slice_dist_micron=slice_dist_micron)

# ================== 3. Standard Preprocessing ==================
adata.layers['count'] = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X.copy()

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=args.top_genes)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)

# SpaMask requires features stored in 'feat'
adata.obsm['feat'] = PCA(n_components=50, random_state=42).fit_transform(adata.X)

# ================== 4. Core Model Training with Benchmarking ==================
with BenchmarkTracker('SpaMask') as tracker:
    adata = train_one(args, adata, final_n_domains)

# Save embedding to adata.obsm['SpaMask_embed']
if 'eval_pred' in adata.obsm:
    # Convert to contiguous float64 array to avoid H5AD writing errors
    adata.obsm['SpaMask_embed'] = np.ascontiguousarray(adata.obsm['eval_pred'], dtype=np.float64)
    print("✅ Successfully extracted and saved embedding to adata.obsm['SpaMask_embed']")
else:
    print("❌ Warning: 'eval_pred' not found in model output! Embedding extraction failed.")

# ================== 5. Standardize Metadata for Evaluation ==================
# Ensure celltype label exists for benchmark scripts
if 'ground_truth' in adata.obs:
    adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
elif 'Ground Truth' in adata.obs:
    adata.obs['celltype'] = adata.obs['Ground Truth'].astype('category')
else:
    print("[Warning] No ground truth/celltype found, biological metrics may fail!")

# SpaMask automatically generates adata.obs['mclust'] via net.process()

# ================== 6. Save Results ==================
tracker.save_report(
    json_path=save_path / f"spamask_benchmark_{config['suffix']}.json",
    adata=adata,
    embed_key='SpaMask_embed',
    extra_meta={'n_datasets': len(proj_list)}
)

adata.write(save_path / f"multiple_adata_{config['suffix']}_spamask.h5ad")
print(f"✅ Successfully finished {CURRENT_TASK} and saved to {save_path}/multiple_adata_{config['suffix']}_spamask.h5ad")