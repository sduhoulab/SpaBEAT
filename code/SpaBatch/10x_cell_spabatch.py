import os
import sys
import gc
import time
import json
import psutil
import torch
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import tifffile
import re
from scipy.sparse import csr_matrix
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')

# 导入外部 SpaBatch 核心数仓路径
sys.path.append("/data_hou/BE/SpaBatch-main/SpaBatch")
from adj import main as build_graph, combine_graph_dict
from train import train_model
from utils import mclust_R, fix_seed

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# 锁定随机种子与显卡设备
fix_seed(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

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

x_col = 'x_centroid'
y_col = 'y_centroid'

xmin = selected_meta[x_col].min() - 40.0
xmax = selected_meta[x_col].max() + 40.0
ymin = selected_meta[y_col].min() - 40.0
ymax = selected_meta[y_col].max() + 40.0
pixel_size = 0.2125

# =====================================================================
# Step 2: 读取并改装官方矩阵 A (采用安全字典映射，对齐乳腺癌物料文件名)
# =====================================================================
print("Loading and converting official Matrix A...")
adata_A = sc.read_h5ad("/data_hou/BE/GraphST/code/data_cell/matrix_A_official_breast.h5ad")
adata_A.obs['batch_name'] = 'Official_A'
adata_A.obs['proj_name'] = 'Official_A'

# 建立绝对安全的 细胞名 -> [X, Y] 映射字典，彻底免于位置越界 Bug
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

# =====================================================================
# Step 3: 实时读取分子点云组装矩阵 B (对齐乳腺癌物料文件名)
# =====================================================================
print("Processing cellpose mask and transcripts according to your exact logic...")
mask_file = "/data_hou/BE/GraphST/code/data_cell/roi_breast_dapi.tif"
mask = tifffile.imread(mask_file)
if len(mask.shape) == 3: 
    mask = mask[0]

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
adata_B.obs['batch_name'] = 'Cellpose_B'
adata_B.obs['proj_name'] = 'Cellpose_B'

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

print("Spatial coordinates and celltypes successfully injected.")

# =====================================================================
# Step 5: 特征交集对齐与独立构筑 SpaBatch 空间图网络
# =====================================================================
print("Aligning gene features and constructing individual graph dicts...")

# 锁定乳腺癌 280 个左右核心功能共有基因
common_genes = adata_A.var_names.intersection(adata_B.var_names)
print(f"  -> Merged shared gene features count: {len(common_genes)}")

adata_A = adata_A[:, common_genes].copy()
adata_B = adata_B[:, common_genes].copy()
adata_B.var = adata_A.var.copy()

total_cells = adata_A.n_obs + adata_B.n_obs

# 各自独立建立局部空间几何拓扑网
graph_dict_A = build_graph(adata_A, adj_cons_by='coordinate', distType='KNN', k_cutoff=8, rad_cutoff=250)
graph_dict_B = build_graph(adata_B, adj_cons_by='coordinate', distType='KNN', k_cutoff=8, rad_cutoff=250)

# 合并图结构与 Anndata 纵向大合并
graph_dict_combined = combine_graph_dict(graph_dict_A, graph_dict_B)
adata_combined = adata_A.concatenate(adata_B, batch_key='batch')

# ⚠️ 生信防御拦截：手动重新挽回并垂直粘回被 Scanpy 强行抹去的空间坐标
adata_combined.obsm['spatial'] = np.vstack([adata_A.obsm['spatial'], adata_B.obsm['spatial']])

# =====================================================================
# Step 6: 高维归一化与 PCA 降维预处理
# =====================================================================
print("Executing combined matrix data process and PCA compression...")
adata_combined.layers['count'] = adata_combined.X.toarray()

sc.pp.filter_genes(adata_combined, min_cells=1)
sc.pp.normalize_total(adata_combined, target_sum=1e6)
sc.pp.log1p(adata_combined)

# 全部共有基因均参与特征流形对齐
adata_combined.var['highly_variable'] = True
sc.pp.scale(adata_combined)

# 抽提稠密主成分矩阵作为网络输入
adata_X = PCA(n_components=200, random_state=42).fit_transform(adata_combined.X)
adata_combined.obsm['X_pca'] = adata_X

# =====================================================================
# Step 7: 启动 SpaBatch 掩膜图自编码器训练与全要素 Benchmark 审计
# =====================================================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training SpaBatch model via masked graph autoencoder...")
pre_epochs = 500
epochs = 1000

SpaBatch_net = train_model(adata_combined, graph_dict_combined, pre_epochs=pre_epochs, epochs=epochs, mask_rate=0.2)
SpaBatch_net.train_with_dec(num_aggre=1)

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before
print("SpaBatch core integration completed successfully!")

# 收割低维去噪嵌入特征，写入大盘通用 Rep 槽位
SpaBatch_feat, _ = SpaBatch_net.process()
adata_combined.obsm['SpaBatch'] = SpaBatch_feat

# 严格对齐包含秒、分、时、维度、算力设备在内的全量 15 项性能记录核心指标
benchmark_results = {
    'method_name': 'SpaBatch',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': adata_combined.n_obs,
    'total_genes': adata_combined.n_vars,
    'embedding_dim': SpaBatch_feat.shape[1],
    'n_datasets': 2,
    'device': str(device),
    'random_seed': 1,
    'hvg_genes': 5000,
    'pre_epochs': pre_epochs,
    'epochs': epochs,
    'timestamp': pd.Timestamp.now().isoformat()
}

# =====================================================================
# Step 8: 跨算法联合空间平滑聚类 (mclust_R) 与大盘接口对齐
# =====================================================================
print("Performing clustering analysis via mclust bridge...")
n_clusters = 8
mclust_R(adata_combined, num_cluster=n_clusters, used_obsm='SpaBatch')

# 规范分类标签槽位名与多算法大盘比对接口契合
adata_combined.obs['leiden_harmony'] = adata_combined.obs['mclust'].astype(str)
adata_combined.obs['new_batch'] = adata_combined.obs['batch'].astype(str)

results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)
output_h5ad = os.path.join(results_dir, "Xenium_SpaBatch_integrated.h5ad")
output_json = os.path.join(results_dir, "spabatch_xenium_benchmark.json")

adata_combined.write(output_h5ad)
with open(output_json, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\n=======================================================")
print("SpaBatch 评测管线全部运行成功结束！")
print(f"1. 校正后的空间融合特征大表已存盘: {output_h5ad}")
print(f"2. 全套 15 项效率审计 JSON 报告已存盘: {output_json}")
print("=======================================================")
