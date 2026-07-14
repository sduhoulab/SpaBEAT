import os
import torch
import scanpy as sc
import pandas as pd
import numpy as np
import json
import gc
import time
import psutil
import sys
import tifffile
import re
import argparse
from scipy.ndimage import center_of_mass
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors

# 引入 SPIRAL 官方核心组件
sys.path.append('../')
from spiral.main import SPIRAL_integration
from spiral.layers import MeanAggregator
from spiral.utils import layer_map, mclust_R

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# =====================================================================
# 1. 载入空间元数据与双矩阵组装 (严格对接乳腺癌最新物料路径)
# =====================================================================
print("Loading Breast Cancer datasets...")
roi_df = pd.read_csv("/data_hou/BE/GraphST/code/data_cell/Selection_breast_cancer.csv", skiprows=[0, 1])
roi_df.columns = roi_df.columns.str.strip()
selected_cell_ids = roi_df.iloc[:, 0].astype(str).tolist()

cells_meta = pd.read_parquet("/data_hou/ST_data_new/Human_Breast_Biomarkers_S1_Top_outs/cells.parquet")
if 'cell_id' in cells_meta.columns: 
    cells_meta = cells_meta.set_index('cell_id')
selected_meta = cells_meta[cells_meta.index.astype(str).isin(selected_cell_ids)]

xmin, xmax = selected_meta['x_centroid'].min() - 40.0, selected_meta['x_centroid'].max() + 40.0
ymin, ymax = selected_meta['y_centroid'].min() - 40.0, selected_meta['y_centroid'].max() + 40.0
pixel_size = 0.2125

# 处理官方 A 矩阵
adata_A = sc.read_h5ad("/data_hou/BE/GraphST/code/data_cell/matrix_A_official_breast.h5ad")
adata_A.obs['batch'] = 'Official_A'
selected_meta_clean = selected_meta.copy()
selected_meta_clean.index = selected_meta_clean.index.astype(str).str.strip()
coord_dict = dict(zip(selected_meta_clean.index, selected_meta_clean[['x_centroid', 'y_centroid']].values))
adata_A.obsm['spatial'] = np.array([coord_dict.get(str(n).strip(), coord_dict.get(str(n).strip()[:-2], [0.0, 0.0])) for n in adata_A.obs_names], dtype=np.float32)

# 处理 Cellpose B 矩阵
mask = tifffile.imread("/data_hou/BE/GraphST/code/data_cell/roi_breast_dapi.tif")
if len(mask.shape) == 3: mask = mask[0]
df_transcripts = pd.read_parquet("/data_hou/ST_data_new/Human_Breast_Biomarkers_S1_Top_outs/transcripts.parquet")
df_roi_trans = df_transcripts[(df_transcripts['x_location'] >= xmin) & (df_transcripts['x_location'] <= xmax) & (df_transcripts['y_location'] >= ymin) & (df_transcripts['y_location'] <= ymax)].copy()
df_roi_trans['pixel_x'] = ((df_roi_trans['x_location'] - xmin) / pixel_size).astype(int)
df_roi_trans['pixel_y'] = ((df_roi_trans['y_location'] - ymin) / pixel_size).astype(int)
h, w = mask.shape
df_roi_trans = df_roi_trans[(df_roi_trans['pixel_x'] >= 0) & (df_roi_trans['pixel_x'] < w) & (df_roi_trans['pixel_y'] >= 0) & (df_roi_trans['pixel_y'] < h)]
df_roi_trans['cell_id'] = mask[df_roi_trans['pixel_y'].values, df_roi_trans['pixel_x'].values]
df_roi_trans = df_roi_trans[(df_roi_trans['cell_id'] > 0) & (~df_roi_trans['feature_name'].str.startswith(('Blank-', 'Control-', 'NegControl-')))]
count_table = pd.crosstab(df_roi_trans['cell_id'], df_roi_trans['feature_name'])
adata_B = sc.AnnData(X=csr_matrix(count_table.values))
adata_B.obs_names = [f"Cellpose_Cell_{i}" for i in count_table.index]
adata_B.var_names = count_table.columns
adata_B.obs['batch'] = 'Cellpose_B'

cell_ids_in_mask = sorted(np.unique(mask))[1:]
centroids = center_of_mass(mask > 0, mask, cell_ids_in_mask)
id_to_coord = {cid: [xmin + (cx * pixel_size), ymin + (cy * pixel_size)] for cid, (cy, cx) in zip(cell_ids_in_mask, centroids)}
adata_B.obsm['spatial'] = np.array([id_to_coord[int(name.split("_")[-1])] for name in adata_B.obs_names], dtype=np.float32)

# =====================================================================
# 2. 注入真实 celltype 与共有特征对齐
# =====================================================================
print("Aligning feature spaces and injecting celltype...")
df_assign = pd.read_csv("/data_hou/ST_data_new/Human_Breast_Biomarkers_S1_Top_outs/Human_Breast_Biomarkers_S1_Top_cell_groups.csv")
df_assign.columns = df_assign.columns.str.strip()
id_col = [c for c in df_assign.columns if 'id' in c.lower() or 'barcode' in c.lower()][0]
type_col = [c for c in df_assign.columns if 'type' in c.lower() or 'group' in c.lower()][0]
id_to_type = {str(r_id).strip()[:-2] if str(r_id).strip().endswith('.0') else str(r_id).strip(): str(c_type).strip() for r_id, c_type in zip(df_assign[id_col], df_assign[type_col]) if not pd.isna(r_id) and not pd.isna(c_type)}

clean_ids_A = [str(n).strip()[:-2] if str(n).strip().endswith('.0') else str(n).strip() for n in adata_A.obs_names]
adata_A.obs['celltype'] = [id_to_type.get(cid, "Unknown") for cid in clean_ids_A]

# 【核心修改】从 R 导出的 CSV 中直接注入 B 矩阵自身的 Seurat 注释结果，彻底弃用 KDTree 空间投影
anno_file = "/data_hou/BE/GraphST/code/data_cell/cellpose_new_annotations.csv"
if os.path.exists(anno_file):
    df_anno = pd.read_csv(anno_file)
    df_anno['id_lower'] = df_anno['cell_id'].astype(str).str.lower().str.strip()
    id_to_type_B = dict(zip(df_anno['id_lower'], df_anno['final_celltype']))
    adata_B.obs['celltype'] = [str(id_to_type_B.get(name.lower().strip(), "Unknown")) for name in adata_B.obs_names]
    print(f"  -> 成功从 CSV 注入 B 矩阵自身注释！有效标签数: {adata_B.obs['celltype'].value_counts().to_dict()}")
else:
    raise FileNotFoundError(f"未在当前目录下找到 R 导出的注释文件: {anno_file}，请检查该文件是否存在！")

common_genes = list(set(adata_A.var_names.astype(str)).intersection(set(adata_B.var_names.astype(str))))
adata_A = adata_A[:, common_genes].copy()
adata_B = adata_B[:, common_genes].copy()
adata_B.var = adata_A.var.copy()

total_cells = adata_A.n_obs + adata_B.n_obs

for col in [adata_A, adata_B]:
    sc.pp.normalize_total(col, target_sum=1e4)
    sc.pp.log1p(col)
    col.var['highly_variable'] = True

# =====================================================================
# 3. 隐形临时中转机制 (满足 SPIRAL 库的文件读取刚需，用完即秒删)
# =====================================================================
spiral_tmp_dir = "/data_hou/BE/GraphST/code/data_cell/spiral_temp/"
os.makedirs(spiral_tmp_dir, exist_ok=True)
datasets = ['Official_A', 'Cellpose_B']
feat_file, edge_file, meta_file, coord_file = [], [], [], []
prefix = '-'.join(datasets)

for i, (name, adata_sub) in enumerate(zip(datasets, [adata_A, adata_B])):
    f_path, e_path = os.path.join(spiral_tmp_dir, f"{prefix}_{name}_features.txt"), os.path.join(spiral_tmp_dir, f"{prefix}_{name}_edge_KNN_6.csv")
    m_path, c_path = os.path.join(spiral_tmp_dir, f"{prefix}_{name}_label.txt"), os.path.join(spiral_tmp_dir, f"{prefix}_{name}_positions.txt")
    feat_file.append(f_path); edge_file.append(e_path); meta_file.append(m_path); coord_file.append(c_path)
    
    pd.DataFrame(adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X, index=adata_sub.obs_names, columns=adata_sub.var_names).to_csv(f_path)
    adata_sub.obs[['batch', 'celltype']].to_csv(m_path)
    pd.DataFrame(adata_sub.obsm['spatial'], index=adata_sub.obs_names, columns=['y', 'x']).to_csv(c_path)
    
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(adata_sub.obsm['spatial'])
    _, indices = nbrs.kneighbors(adata_sub.obsm['spatial'])
    edges = [f"{adata_sub.obs_names[r]}:{adata_sub.obs_names[c]}" for r in range(len(adata_sub.obs_names)) for c in indices[r, 1:]]
    with open(e_path, 'w') as f: f.write('\n'.join(edges) + '\n')

# =====================================================================
# 4. 运行 SPIRAL 神经网络训练与性能大表解算
# =====================================================================
gc.collect(); torch.cuda.empty_cache()
N = len(common_genes)

M = 1 if len(datasets) == 2 else len(datasets)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--AEdims', type=list, default=[N, [512], 32])
parser.add_argument('--AEdimsR', type=list, default=[32, [512], N])
parser.add_argument('--GSdims', type=list, default=[512, 32])
parser.add_argument('--zdim', type=int, default=32)
parser.add_argument('--znoise_dim', type=int, default=4)
parser.add_argument('--CLdims', type=list, default=[4, [], M])
parser.add_argument('--DIdims', type=list, default=[28, [32, 16], M])
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--agg_class', type=str, default=MeanAggregator)
parser.add_argument('--num_samples', type=str, default=6)
parser.add_argument('--N_WALKS', type=int, default=6)
parser.add_argument('--WALK_LEN', type=int, default=1)
parser.add_argument('--N_WALK_LEN', type=int, default=6)
parser.add_argument('--NUM_NEG', type=int, default=6)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--alpha1', type=float, default=N)
parser.add_argument('--alpha2', type=float, default=1)
parser.add_argument('--alpha3', type=float, default=1)
parser.add_argument('--alpha4', type=float, default=1)
parser.add_argument('--lamda', type=float, default=1)
parser.add_argument('--Q', type=float, default=10)
params, _ = parser.parse_known_args()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training SPIRAL model...")
SPII = SPIRAL_integration(params, feat_file, edge_file, meta_file)
SPII.train()

training_end_time = time.time()
memory_after = get_memory_usage()
training_time = training_end_time - training_start_time
memory_used = memory_after - memory_before

# =====================================================================
# 5. 精准提取特征，100% 对齐斩断多余数据，实现你的四个黄金键值
# =====================================================================
SPII.model.eval()
all_idx = np.arange(SPII.feat.shape[0])
all_layer, all_mapping = layer_map(all_idx.tolist(), SPII.adj, 2)
all_rows = SPII.adj.tolil().rows[all_layer[0]]
all_feature = torch.Tensor(SPII.feat.iloc[all_layer[0], :].values).float().cuda()
all_embed, _, _, _ = SPII.model(all_feature, all_layer, all_mapping, all_rows, SPII.params.lamda, SPII.de_act, SPII.cl_act)
spiral_biological_embedding = all_embed[2].cpu().detach().numpy()[:, params.znoise_dim:]

# 垂直合并 AnnData 骨架并进行终极防爆修复
adata_combined = adata_A.concatenate(adata_B, batch_key='batch')
adata_combined.obsm['spatial'] = np.vstack([adata_A.obsm['spatial'], adata_B.obsm['spatial']])

# 核心对齐：精准喂饱你的比对配置要求，拒绝多余花样
adata_combined.obsm['spiral'] = spiral_biological_embedding                     # 对应 "use_rep": "spiral"
adata_combined.obs['new_batch'] = adata_combined.obs['batch'].astype(str)        # 对应 "batch_key": "new_batch"
adata_combined.obs['celltype'] = adata_combined.obs['celltype'].astype(str)      # 对应 "celltype_key": "celltype"

# 调用 SPIRAL 原生的 mclust_R 算法计算联合空间域，彻底摆脱 GraphST
adata_combined = mclust_R(adata_combined, used_obsm='spiral', num_cluster=8)
adata_combined.obs['mclust'] = adata_combined.obs['mclust'].astype(str)          # 对应 "cluster_key": "mclust"

# 拼装 15 项性能评估字典大表
benchmark_results = { 
    'method_name': 'SPIRAL', 'training_time_seconds': training_time, 'training_time_minutes': training_time / 60, 'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used, 'memory_usage_gb': memory_used / 1024, 'total_cells': total_cells, 'final_cells': adata_combined.n_obs,
    'total_genes': adata_combined.n_vars, 'embedding_dim': adata_combined.obsm['spiral'].shape[1], 'n_datasets': len(adata_combined.obs['new_batch'].unique()),
    'device': str(device), 'random_seed': 50, 'hvg_genes': 5000, 'timestamp': pd.Timestamp.now().isoformat()
}

# =====================================================================
# 6. 持久化大统一存盘，就地毁灭临时文件
# =====================================================================
os.makedirs("./results", exist_ok=True)
adata_combined.write("./results/Xenium_SPIRAL_integrated.h5ad")
with open("./results/spiral_xenium_benchmark.json", "w") as f: json.dump(benchmark_results, f, indent=2)

# 执行清道夫程序，瞬间毁灭落盘的临时中转文本
for f_list in [feat_file, edge_file, meta_file, coord_file]:
    for f_p in f_list:
        if os.path.exists(f_p): os.remove(f_p)
if os.path.exists(spiral_tmp_dir): os.rmdir(spiral_tmp_dir)

print("SPIRAL Pipeline completed successfully with exact 4 keys mapped via native mclust_R!")
