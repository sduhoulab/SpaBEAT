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
from scipy.ndimage import center_of_mass
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# 自动选择驱动设备 (优先使用 GPU 加速)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# =====================================================================
# Step 1: 载入空间元数据 (严格采用乳腺癌最新专属大数仓路径与选区文件名)
# =====================================================================
print("Loading Xenium spatial metadata (Breast Cancer)...")
roi_df = pd.read_csv("./data_cell/Selection_breast_cancer.csv", skiprows=[0, 1])
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

# 计算图纸的物理原点与拦截边界
xmin = selected_meta[x_col].min() - 40.0
xmax = selected_meta[x_col].max() + 40.0
ymin = selected_meta[y_col].min() - 40.0
ymax = selected_meta[y_col].max() + 40.0
pixel_size = 0.2125

# =====================================================================
# Step 2: 读取并改装官方矩阵 A (对齐乳腺癌专属文件名)
# =====================================================================
print("Loading and converting official Matrix A...")
adata_A = sc.read_h5ad("./data_cell/matrix_A_official_breast.h5ad")
adata_A.obs['batch'] = 'Official_A'
# 提取官方细胞对应的原始微米坐标
coord_A = selected_meta.loc[adata_A.obs_names, [x_col, y_col]].values
adata_A.obsm['spatial'] = coord_A.astype(np.float32) 

# =====================================================================
# Step 3: 实时读取分子点云组装矩阵 B (对齐乳腺癌专属路径与文件名)
# =====================================================================
print("Processing cellpose mask and transcripts according to your exact logic...")
mask_file = "./data_cell/roi_breast_dapi.tif"
mask = tifffile.imread(mask_file)
if len(mask.shape) == 3: 
    mask = mask[0]

# 读入包含全图所有测序分子绝对物理坐标的 Parquet 巨表 (对齐乳腺癌最新大数仓)
df_transcripts = pd.read_parquet("/data_hou/ST_data_new/Human_Breast_Biomarkers_S1_Top_outs/transcripts.parquet")

# 空间物理拦截
df_roi_trans = df_transcripts[
    (df_transcripts['x_location'] >= xmin) & (df_transcripts['x_location'] <= xmax) &
    (df_transcripts['y_location'] >= ymin) & (df_transcripts['y_location'] <= ymax)
].copy()

# 物理跨尺度换算
df_roi_trans['pixel_x'] = ((df_roi_trans['x_location'] - xmin) / pixel_size).astype(int)
df_roi_trans['pixel_y'] = ((df_roi_trans['y_location'] - ymin) / pixel_size).astype(int)

# 过滤掉因四舍五入超出图片边界的边缘点
h, w = mask.shape
df_roi_trans = df_roi_trans[
    (df_roi_trans['pixel_x'] >= 0) & (df_roi_trans['pixel_x'] < w) & 
    (df_roi_trans['pixel_y'] >= 0) & (df_roi_trans['pixel_y'] < h)
]

# 核心投射：让每个 RNA 分子低头看脚下踩到了 Cellpose 划分的哪个细胞 ID 上
df_roi_trans['cell_id'] = mask[df_roi_trans['pixel_y'].values, df_roi_trans['pixel_x'].values]

# 数据纯化：过滤流浪分子与质控探针
df_roi_trans = df_roi_trans[df_roi_trans['cell_id'] > 0]
df_roi_trans = df_roi_trans[~df_roi_trans['feature_name'].str.startswith(('Blank-', 'Control-', 'NegControl-'))]

print("  -> 正在执行行列交叉矩阵透视计数...")
# 计算每个 8 位细胞里截获的各基因分子总数
count_table = pd.crosstab(df_roi_trans['cell_id'], df_roi_trans['feature_name'])

# 华丽打包为标准的单细胞 AnnData 矩阵对象
adata_B = sc.AnnData(X=csr_matrix(count_table.values))
adata_B.obs_names = [f"Cellpose_Cell_{i}" for i in count_table.index]
adata_B.var_names = count_table.columns

adata_B.obs_names_make_unique()
adata_B.var_names_make_unique()
adata_B.obs['batch'] = 'Cellpose_B'

# =====================================================================
# Step 4: 基于相同的 8 位图像，反推并注入 Cellpose 空间物理坐标
# =====================================================================
print("Calculating cell spatial centroids directly from mask...")
cell_ids_in_mask = sorted(np.unique(mask))[1:]
centroids = center_of_mass(mask > 0, mask, cell_ids_in_mask)

id_to_coord = {}
for cell_id, (cy, cx) in zip(cell_ids_in_mask, centroids):
    mx = xmin + (cx * pixel_size)
    my = ymin + (cy * pixel_size)
    id_to_coord[cell_id] = [mx, my]

coord_B_microns = []
for obs_name in adata_B.obs_names:
    actual_id = int(obs_name.split("_")[-1])
    coord_B_microns.append(id_to_coord[actual_id])

adata_B.obsm['spatial'] = np.array(coord_B_microns, dtype=np.float32)
print("Spatial coordinates for both live matrices are successfully injected.")

# =====================================================================
# [新增核心信息注入]：打通细胞类型大表，建立 cell_id 与 celltype 金标准
# =====================================================================
print("Enriching pre-integration identity fields (cell_id & celltype)...")

# 1. 换算并建立 A 和 B 的标准纯数字 cell_id 属性列
clean_ids_A = [str(n).strip()[:-2] if str(n).strip().endswith('.0') else str(n).strip() for n in adata_A.obs_names]
adata_A.obs['cell_id'] = clean_ids_A

clean_ids_B = []
for name in adata_B.obs_names:
    numbers = re.findall(r'\d+', str(name))
    clean_ids_B.append(numbers[0] if numbers else "0")
adata_B.obs['cell_id'] = clean_ids_B

# 2. 读取最新命名的纯净版乳腺癌细胞属性大表
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

# 3. 激活官方矩阵 A 的真实乳腺癌类别
adata_A.obs['celltype'] = [id_to_type.get(cid, "Unknown") for cid in clean_ids_A]

# 4. 【已更新】从 R 导出的 CSV 中读取矩阵 B 自身的 Seurat 注释结果（防大小写不一致增强版）
# 请确保 "cellpose_new_annotations.csv" 就在您的当前工作目录下，或者改为它的绝对路径
anno_file = "/data_hou/BE/GraphST/code/data_cell/cellpose_new_annotations.csv"
if os.path.exists(anno_file):
    df_anno = pd.read_csv(anno_file)
    # 将 R 导出的 cell_id 统一转为小写，建立干净的映射字典
    df_anno['id_lower'] = df_anno['cell_id'].astype(str).str.lower().str.strip()
    id_to_type = dict(zip(df_anno['id_lower'], df_anno['final_celltype']))
    
    # 将新组装的 adata_B.obs_names 转换后去字典中精准匹配标签
    adata_B.obs['celltype'] = [str(id_to_type.get(name.lower().strip(), "Unknown")) for name in adata_B.obs_names]
    print(f"  -> 成功从 CSV 注入 B 矩阵自身注释！有效标签数: {adata_B.obs['celltype'].value_counts().to_dict()}")
else:
    raise FileNotFoundError(f"未在当前目录下找到 R 导出的注释文件: {anno_file}，请检查该文件是否存在！")

# 锁定前置初始总细胞数（用于下游精确填入测试性能大表中）
total_cells = adata_A.n_obs + adata_B.n_obs
print(f"Pre-integration metadata setup complete. Raw total cells sum: {total_cells}")

# =====================================================================
# Step 5: 高维归一化与横向矩阵大合并 (增加特征交集切片与坐标防御)
# =====================================================================
print("Executing gene expression normalization and preprocessing...")

# 学术对齐：将特征空间收缩锁死在 280 个左右的共有局部功能基因上
common_genes = list(set(adata_A.var_names.astype(str)).intersection(set(adata_B.var_names.astype(str))))
print(f"Aligning feature spaces. Total shared local panel genes: {len(common_genes)}")
adata_A = adata_A[:, common_genes].copy()
adata_B = adata_B[:, common_genes].copy()
adata_B.var = adata_A.var.copy()

for col in [adata_A, adata_B]:
    sc.pp.normalize_total(col, target_sum=1e4)
    sc.pp.log1p(col)
    col.var['highly_variable'] = True

# 纵向拼接两个数据集
adata_combined = adata_A.concatenate(adata_B, batch_key='batch')

# 重大技术防御：手动挽回并重建被 Scanpy concatenate 自动抹掉的 .obsm['spatial']
adata_combined.obsm['spatial'] = np.vstack([adata_A.obsm['spatial'], adata_B.obsm['spatial']])
print(f"Concatenated spatial AnnData shape: {adata_combined.shape}")

# =====================================================================
# Step 6: 启动 GraphST 神经网络训练
# =====================================================================
gc.collect()
if torch.cuda.is_available(): 
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training GraphST model to harmonize spatial manifolds...")
sys.path.append('../')
from GraphST import GraphST

model = GraphST.GraphST(adata_combined, device=device, random_seed=50)
adata_combined = model.train()

training_end_time = time.time()
memory_after = get_memory_usage()
training_time = training_end_time - training_start_time
memory_used = memory_after - memory_before

# 动态拉取当前合并批次的独特数据集名称列表
datasets = adata_combined.obs['batch'].unique()

# 严格对齐用户指定的全要素高规格性能评估结果大表
benchmark_results = { 
    'method_name': 'GraphST',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60, 
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': adata_combined.n_obs,
    'total_genes': adata_combined.n_vars,
    'embedding_dim': adata_combined.obsm['emb'].shape[1],
    'n_datasets': len(datasets),
    'device': str(device),
    'random_seed': 50, 
    'hvg_genes': 5000,
    'timestamp': pd.Timestamp.now().isoformat()
}

# =====================================================================
# Step 7: 跨算法联合空间平滑聚类 (mclust)
# =====================================================================
print("Running joint spatial clustering via mclust on the integrated embedding...")
from GraphST.utils import clustering
n_clusters = 8 
clustering(adata_combined, n_clusters, method='mclust')

# =====================================================================
# Step 8: 一键持久化存盘 (严格复原你的 ./results 目录结构)
# =====================================================================
os.makedirs("./results", exist_ok=True)
output_h5ad = "./results/Xenium_GraphST_integrated.h5ad"
output_json = "./results/graphst_xenium_benchmark.json"

adata_combined.write(output_h5ad)
with open(output_json, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("Pipeline completed successfully!")
print(f"Integrated AnnData saved to: {output_h5ad}")
print(f"Performance metrics saved to: {output_json}")
