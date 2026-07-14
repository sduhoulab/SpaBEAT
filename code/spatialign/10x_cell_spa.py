import os
import sys
import gc
import time
import json
import psutil
import torch
import scanpy as sc
import anndata as ad
from anndata import AnnData
import pandas as pd
import numpy as np
import tifffile
import re
from scipy.sparse import csr_matrix
from scipy.ndimage import center_of_mass
from sklearn.mixture import GaussianMixture
from scipy.spatial import KDTree
from warnings import filterwarnings
filterwarnings("ignore")

# 导入外部 Spatialign 核心组件
from spatialign import Spatialign

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

torch.set_default_dtype(torch.float32)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# 计算物理原点与拦截边界
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
adata_A.obs['batch'] = 'Official_A'

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

print("Spatial coordinates and celltypes successfully injected.")

# =====================================================================
# Step 5: 特征交集过滤、高维规范化与构建 Spatialign 文件缓存流
# =====================================================================
print("Executing gene expression normalization and data synchronization...")

common_genes = adata_A.var_names.intersection(adata_B.var_names)
print(f"  -> Merging features: Official_A ({adata_A.n_vars}) vs Cellpose_B ({adata_B.n_vars})")
print(f"  -> Shared synced genes count: {len(common_genes)}")

adata_A = adata_A[:, common_genes].copy()
adata_B = adata_B[:, common_genes].copy()
adata_B.var = adata_A.var.copy()

for col in [adata_A, adata_B]:
    sc.pp.normalize_total(col, target_sum=1e4)
    sc.pp.log1p(col)

temp_dir = "./spatialign_temp/"
os.makedirs(temp_dir, exist_ok=True)

data_list = [
    os.path.join(temp_dir, "Official_A.h5ad"),
    os.path.join(temp_dir, "Cellpose_B.h5ad")
]
dataset_names = ['Official_A', 'Cellpose_B']
total_cells = adata_A.n_obs + adata_B.n_obs

adata_A.write_h5ad(data_list[0])
adata_B.write_h5ad(data_list[1])
print(f"Temporary h5ad input caches written. Total cells: {total_cells}")

# =====================================================================
# Step 6: 实例化 Spatialign 模型大对象
# =====================================================================
print("Initializing Spatialign model container...")
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)

model = Spatialign(
    *data_list,
    batch_key='batch',
    is_norm_log=True,
    is_scale=False,
    n_neigh=15,
    is_undirected=True,
    latent_dims=100,
    seed=42,
    gpu=0,
    save_path=results_dir,
    is_verbose=False
)
raw_merge = AnnData.concatenate(*model.dataset.data_list)

# =====================================================================
# Step 7: 启动 Spatialign 深度对齐训练并进行硬核 Benchmarking 审计
# =====================================================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training Spatialign model to align cross-segmentation manifolds...")
model.train(0.05, 1, 0.1)
model.alignment()

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before
print("Spatialign core alignment completed successfully!")

# 从模型自动输出的结果目录中，重新载入被擦除批次噪声后的校正数据文件
correct1 = sc.read_h5ad(os.path.join(results_dir, "res/correct_data0.h5ad"))
correct2 = sc.read_h5ad(os.path.join(results_dir, "res/correct_data1.h5ad"))

# 严格对齐包含秒、分、时、维度、算力设备在内的 15 项性能记录指标
benchmark_results = {
    'method_name': 'Spatialign',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': correct1.n_obs + correct2.n_obs, 
    'total_genes': correct1.n_vars,
    'embedding_dim': correct1.obsm["correct"].shape[1],
    'n_datasets': len(data_list),
    'device': 'GPU 0',
    'random_seed': 42,
    'hvg_genes': correct1.n_vars,
    'timestamp': pd.Timestamp.now().isoformat()
}

# =====================================================================
# Step 8: 【核心恢复】：基于留存细胞，执行安全的字典一对一精准元数据恢复
# =====================================================================
print("Enriching pre-integration identity fields onto retained cells...")
# 建立 细胞名 -> 属性 的高能映射字典（完美防御过滤导致的维度冲突）
A_celltype_map = dict(zip(adata_A.obs_names, adata_A.obs['celltype']))
A_cellid_map = dict(zip(adata_A.obs_names, adata_A.obs['cell_id']))
A_spatial_map = dict(zip(adata_A.obs_names, adata_A.obsm['spatial']))

B_celltype_map = dict(zip(adata_B.obs_names, adata_B.obs['celltype']))
B_cellid_map = dict(zip(adata_B.obs_names, adata_B.obs['cell_id']))
B_spatial_map = dict(zip(adata_B.obs_names, adata_B.obsm['spatial']))

# 为官方批次留存下来的细胞精准反向找回真标签
correct1.obs['celltype'] = [A_celltype_map.get(name, "Unknown") for name in correct1.obs_names]
correct1.obs['cell_id'] = [A_cellid_map.get(name, "0") for name in correct1.obs_names]
correct1.obsm['spatial'] = np.array([A_spatial_map[name] for name in correct1.obs_names], dtype=np.float32)

# 为Cellpose批次留存下来的细胞精准反向找回真标签
correct2.obs['celltype'] = [B_celltype_map.get(name, "Unknown") for name in correct2.obs_names]
correct2.obs['cell_id'] = [B_cellid_map.get(name, "0") for name in correct2.obs_names]
correct2.obsm['spatial'] = np.array([B_spatial_map[name] for name in correct2.obs_names], dtype=np.float32)

# 此时执行纵向合并，Scanpy会自动安全合并 obs 内部已被对齐的自定义属性列
merge_data = correct1.concatenate(correct2)

batch_mapping = {
    '0': 'Official_A',
    '1': 'Cellpose_B'
}
raw_merge.obs['new_batch'] = raw_merge.obs['batch'].replace(batch_mapping)
merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)

# 手动召回并垂直粘回被 Scanpy concatenate 自动吃掉的空间二维微米绝对坐标
merge_data.obsm['spatial'] = np.vstack([correct1.obsm['spatial'], correct2.obsm['spatial']])

print("Performing continuous probabilistic clustering via GMM...")
sc.pp.scale(merge_data)
X = merge_data.obsm['correct']
n_components = 8 

gmm = GaussianMixture(n_components=n_components, random_state=42)
merge_data.obs['mclust'] = gmm.fit_predict(X)
merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")

# 统一分类键值名，确保直接兼容下游评测大盘
merge_data.obs['leiden_harmony'] = merge_data.obs['mclust'].astype(str)

# =====================================================================
# Step 9: 数据一键持久化与临时中转站粉碎回收 (保持你的 ./results 结构)
# =====================================================================
output_h5ad = "./results/Xenium_Spatialign_integrated.h5ad"
output_json = "./results/spatialign_xenium_benchmark.json"

merge_data.write(output_h5ad)
with open(output_json, "w") as f:
    json.dump(benchmark_results, f, indent=2)

# 全自动粉碎临时生成的 h5ad 文件，释放服务器磁盘空间
import shutil
shutil.rmtree(temp_dir)

print("\n=======================================================")
print("Spatialign 管线全部顺利冲过终点！")
print(f"1. 流形校正后的联合对象已归档: {output_h5ad}")
print(f"2. 硬核评测指标 JSON 面板已归档: {output_json}")
print("=======================================================")
