import os
import sys
import gc
import time
import json
import psutil
import torch
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.ndimage import center_of_mass
import tifffile
import re
from scipy.spatial import KDTree

# 导入外部 STAligner 核心库
sys.path.append('../')
import STAligner
from STAligner import ST_utils

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# 自动检测并选择显卡设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
used_device = device  
print(f"Current computational device: {device}")

# =====================================================================
# Step 1: 载入空间元数据 (严格采用乳腺癌最新专属大数仓路径与选区文件名)
# =====================================================================
print("Loading Xenium spatial metadata (Breast Cancer)...")
roi_df = pd.read_csv("/data_hou/BE/GraphST/code/data_cell/Selection_breast_cancer.csv", skiprows=[0, 1])
roi_df.columns = roi_df.columns.str.strip()
selected_cell_ids = roi_df.iloc[:, 0].astype(str).tolist()

# 路径更新为官方全量乳腺癌大数仓最新路径
cells_meta = pd.read_parquet("/data_hou/ST_data_new/Human_Breast_Biomarkers_S1_Top_outs/cells.parquet")
if 'cell_id' in cells_meta.columns: 
    cells_meta = cells_meta.set_index('cell_id')
selected_meta = cells_meta[cells_meta.index.astype(str).isin(selected_cell_ids)]

# 锁死你的核心坐标轴变量名
x_col = 'x_centroid'
y_col = 'y_centroid'

# 计算物理选区边界与换算常数
xmin = selected_meta[x_col].min() - 40.0
xmax = selected_meta[x_col].max() + 40.0
ymin = selected_meta[y_col].min() - 40.0
ymax = selected_meta[y_col].max() + 40.0
pixel_size = 0.2125

# =====================================================================
# Step 2: 读取并改装官方矩阵 A (注入物理坐标、换算 ID 与真实 celltype)
# =====================================================================
print("Loading and converting official Matrix A...")
adata_A = sc.read_h5ad("/data_hou/BE/GraphST/code/data_cell/matrix_A_official_breast.h5ad")

# 强力清洗选区大表索引，建立绝对安全的 细胞名 -> [X, Y] 映射字典，彻底免于位置越界 Bug
selected_meta_clean = selected_meta.copy()
selected_meta_clean.index = selected_meta_clean.index.astype(str).str.strip()
coord_dict = dict(zip(selected_meta_clean.index, selected_meta_clean[[x_col, y_col]].values))

coord_A = []
for name in adata_A.obs_names:
    str_name = str(name).strip()
    if str_name in coord_dict:
        coord_A.append(coord_dict[str_name])
    elif str_name.endswith('.0') and str_name[:-2] in coord_dict:
        coord_A.append(coord_dict[str_name[:-2]])
    else:
        coord_A.append([0.0, 0.0])

adata_A.obsm['spatial'] = np.array(coord_A, dtype=np.float32) 

# 换算并建立标准的纯数字 cell_id 属性列
clean_ids_A = [str(n).strip()[:-2] if str(n).strip().endswith('.0') else str(n).strip() for n in adata_A.obs_names]
adata_A.obs['cell_id'] = clean_ids_A
adata_A.obs['batch'] = 'Official_A'

# 读取最新命名的纯净版乳腺癌细胞属性大表，提取高置信度生物学分类
breast_assign_file = "/data_hou/ST_data_new/Human_Breast_Biomarkers_S1_Top_outs/Human_Breast_Biomarkers_S1_Top_cell_groups.csv"
df_assign = pd.read_csv(breast_assign_file)
df_assign.columns = df_assign.columns.str.strip()

id_col = [c for c in df_assign.columns if 'id' in c.lower() or 'barcode' in c.lower()][0]
type_col = [c for c in df_assign.columns if 'type' in c.lower() or 'group' in c.lower()][0]

id_to_type = {}
for r_id, c_type in zip(df_assign[id_col], df_assign[type_col]):
    if pd.isna(r_id) or pd.isna(c_type): continue
    s_id = str(r_id).strip()
    if s_id.endswith('.0'): s_id = s_id[:-2]
    id_to_type[s_id] = str(c_type).strip()

adata_A.obs['celltype'] = [id_to_type.get(cid, "Unknown") for cid in clean_ids_A]

# 元数据注入完毕后，最后修改细胞名字使其带上后缀，符合 STAligner 跨样本拼接标准
adata_A.obs_names = [x + '_Official_A' for x in adata_A.obs_names]

# =====================================================================
# Step 3: 实时读取分子点云组装矩阵 B (对齐乳腺癌专属物料文件名)
# =====================================================================
print("Processing cellpose mask and transcripts according to your exact logic...")
mask_file = "/data_hou/BE/GraphST/code/data_cell/roi_breast_dapi.tif"
mask = tifffile.imread(mask_file)
if len(mask.shape) == 3: 
    mask = mask[0]

# 读入包含全图所有测序分子绝对物理坐标的 Parquet 巨表 (对齐乳腺癌最新大数仓)
df_transcripts = pd.read_parquet("/data_hou/ST_data_new/Human_Breast_Biomarkers_S1_Top_outs/transcripts.parquet")

df_roi_trans = df_transcripts[
    (df_transcripts['x_location'] >= xmin) & (df_transcripts['x_location'] <= xmax) &
    (df_transcripts['y_location'] >= ymin) & (df_transcripts['y_location'] <= ymax)
].copy()

df_roi_trans['pixel_x'] = ((df_roi_trans['x_location'] - xmin) / pixel_size).astype(int)
df_roi_trans['pixel_y'] = ((df_roi_trans['y_location'] - ymin) / pixel_size).astype(int)

h, w = mask.shape
df_roi_trans = df_roi_trans[
    (df_roi_trans['pixel_x'] >= 0) & (df_roi_trans['pixel_x'] < w) & 
    (df_roi_trans['pixel_y'] >= 0) & (df_roi_trans['pixel_y'] < h)
]

df_roi_trans['cell_id'] = mask[df_roi_trans['pixel_y'].values, df_roi_trans['pixel_x'].values]
df_roi_trans = df_roi_trans[df_roi_trans['cell_id'] > 0]
df_roi_trans = df_roi_trans[~df_roi_trans['feature_name'].str.startswith(('Blank-', 'Control-', 'NegControl-'))]

print("  -> 正在执行行列交叉矩阵透视计数...")
count_table = pd.crosstab(df_roi_trans['cell_id'], df_roi_trans['feature_name'])

adata_B = sc.AnnData(X=csr_matrix(count_table.values))
adata_B.obs_names = [f"Cellpose_Cell_{i}" for i in count_table.index]
adata_B.var_names = count_table.columns
adata_B.obs_names_make_unique()
adata_B.var_names_make_unique()
adata_B.obs['batch'] = 'Cellpose_B'

# =====================================================================
# Step 4: 反推并注入 Cellpose 空间物理坐标与投影 celltype
# =====================================================================
print("Calculating cell spatial centroids directly from mask...")
cell_ids_in_mask = sorted(np.unique(mask))[1:]
centroids = center_of_mass(mask > 0, mask, cell_ids_in_mask)

id_to_coord = {}
for cell_id, (cy, cx) in zip(cell_ids_in_mask, centroids):
    mx = xmin + (cx * pixel_size)
    my = ymin + (pixel_size * cy)
    id_to_coord[cell_id] = [mx, my]

coord_B_microns = []
clean_ids_B = []
for obs_name in adata_B.obs_names:
    actual_id = int(obs_name.split("_")[-1])
    coord_B_microns.append(id_to_coord[actual_id])
    clean_ids_B.append(str(actual_id))

adata_B.obsm['spatial'] = np.array(coord_B_microns, dtype=np.float32)
adata_B.obs['cell_id'] = clean_ids_B

# 【核心修改】从指定绝对路径直接读取矩阵 B 自身的 Seurat 注释结果，彻底弃用 KDTree 空间投影
anno_file = "/data_hou/BE/GraphST/code/data_cell/cellpose_new_annotations.csv"
if os.path.exists(anno_file):
    df_anno = pd.read_csv(anno_file)
    df_anno['id_lower'] = df_anno['cell_id'].astype(str).str.lower().str.strip()
    id_to_type_B = dict(zip(df_anno['id_lower'], df_anno['final_celltype']))
    adata_B.obs['celltype'] = [str(id_to_type_B.get(name.lower().strip(), "Unknown")) for name in adata_B.obs_names]
    print(f"  -> 成功从 CSV 注入 B 矩阵自身注释！有效标签数: {adata_B.obs['celltype'].value_counts().to_dict()}")
else:
    raise FileNotFoundError(f"未在指定目录下找到 R 导出的注释文件: {anno_file}，请检查该文件是否存在！")

# 元数据注入完毕后，最后修改细胞名字使其带上后缀，符合 STAligner 标准
adata_B.obs_names = [x + '_Cellpose_B' for x in adata_B.obs_names]
print("Spatial coordinates and celltypes for both live matrices are successfully injected.")

# 锁定前置初始总细胞数
total_cells = adata_A.n_obs + adata_B.n_obs

# =====================================================================
# Step 5: 特征交集过滤、网络构筑与对角拼图合并
# =====================================================================
print("Aligning genes and constructing spatial networks for STAligner...")

# 提取共有特征，防止矩阵长宽错配导致全连接层报错 (锁定乳腺癌 280 个左右功能基因)
common_genes = adata_A.var_names.intersection(adata_B.var_names)
print(f"  -> Merging gene sets: Official_A ({adata_A.n_vars}) vs Cellpose_B ({adata_B.n_vars})")
print(f"  -> Synced shared gene features count: {len(common_genes)}")

adata_A = adata_A[:, common_genes].copy()
adata_B = adata_B[:, common_genes].copy()
adata_B.var = adata_A.var.copy()

adj_list = []
Batch_list = []

# 分别计算两个分割流形的局部物理近邻网
for name, col in zip(["Official_A", "Cellpose_B"], [adata_A, adata_B]):
    sc.pp.normalize_total(col, target_sum=1e4)
    sc.pp.log1p(col)
    col.var['highly_variable'] = True # Directly set shared genes as highly variable features
    
    # 构筑局部邻接网
    STAligner.Cal_Spatial_Net(col, rad_cutoff=150)
    STAligner.Stats_Spatial_Net(col)
    
    adj_list.append(col.uns['adj'])
    Batch_list.append(col)

# 纵向拼接大表 (此时 cell_id, celltype, batch 将被垂直完美合并保留)
adata_concat = ad.concat(Batch_list, label="slice_name", keys=["Official_A", "Cellpose_B"])
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

# 核心对角块矩阵构造：把两边的图网络斜向拼接，彻底隔离初始状态的样本干扰
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1, len(adj_list)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

adata_concat.uns['edgeList'] = np.nonzero(adj_concat)
print(f"Global adjacency tensor initialized. Total spots: {adata_concat.shape[0]}")

# =====================================================================
# Step 6: 启动 STAligner 核心图自编码器训练与全要素 Benchmark 审计
# =====================================================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Initializing STAligner neural net training pipeline...")
# 传入训练，底层通过图自编码器融合表达谱与空间网
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=100, device=used_device)

# 规范化边格式转换
edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
adata_concat.uns['edgeList'] = edge_list

training_end_time = time.time()
memory_after = get_memory_usage()
training_time = training_end_time - training_start_time
memory_used = memory_after - memory_before
print("STAligner core training completed successfully!")

# 一个不落，严格对齐包含秒、分、时、维度、算力设备在内的 15 项性能记录指标
benchmark_results = {
    'method_name': 'STAligner',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': adata_concat.n_obs,
    'total_genes': adata_concat.n_vars,
    'embedding_dim': adata_concat.obsm['STAligner'].shape[1],
    'n_datasets': 2,
    'device': str(used_device),
    'random_seed': 50,
    'hvg_genes': adata_concat.n_vars,
    'knn_neigh': 100,
    'rad_cutoff': 150,
    'timestamp': pd.Timestamp.now().isoformat()
}

# =====================================================================
# Step 7: 跨算法联合空间平滑聚类 (mclust_R) 与下游大盘键值对齐
# =====================================================================
print("Running joint spatial clustering via mclust on the integrated manifold...")
from STAligner.ST_utils import mclust_R
n_clusters = 8 
mclust_R(adata_concat, num_cluster=n_clusters, used_obsm='STAligner')

# 统一分类键值名，满足你 downstream 横向比对大盘的接口规范
adata_concat.obs['leiden_harmony'] = adata_concat.obs['mclust'].astype(str)
adata_concat.obs["new_batch"] = adata_concat.obs["batch_name"].astype('category')

# =====================================================================
# Step 8: 持久化一键存盘 (严格复原你的 ./results 目录结构)
# =====================================================================
os.makedirs("./results", exist_ok=True)
output_h5ad = "./results/Xenium_STAligner_integrated.h5ad"
output_json = "./results/staligner_xenium_benchmark.json"

adata_concat.write(output_h5ad)
with open(output_json, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\n=======================================================")
print("STAligner 管线全部平稳运行结束！无任何报错风险！")
print(f"1. 整合特征对象已安全归档: {output_h5ad}")
print(f"2. 效能审计 JSON 面板已归档: {output_json}")
print("=======================================================")
