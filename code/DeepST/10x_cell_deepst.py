# =====================================================================
# Xenium单细胞原位成像矩阵定制版 DeepST 算法脚本 (纯文字逻辑完全修复版)
# 文件名：10x_cell_deepst.py
# =====================================================================
import warnings
warnings.filterwarnings("ignore")
import os 
import sys
import gc
import time
import json
import re
import torch
import scanpy as sc
import anndata as ad 
import pandas as pd
import numpy as np
import tifffile
import psutil
from scipy.sparse import csr_matrix
from scipy.ndimage import center_of_mass

# 核心修正：严格恢复师姐原版跑通的导入骨架，彻底粉碎 AttributeError 报错
sys.path.append('../DeepST')
from DeepST import run

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# 自动检测计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Current computational device: {device}")

# 全局大统一：细胞大类标签规范化清洗字典
global_label_mapping = {
    '9_Tumor_Cells': 'Tumor_Cells', '8_Tumor_Cells': 'Tumor_Cells', '7_Tumor_Cells': 'Tumor_Cells',
    '4_Tumor_Cells': 'Tumor_Cells', '12_Tumor_Cells': 'Tumor_Cells', '15_Tumor_Cells': 'Tumor_Cells',
    'Endothelial': '5_Endothelial_Cells',
    'Stromal': '2_Stromal_Cells',
    'Perivascular-Like': '3_Perivascular_Cells',
    'Macrophages_1': '1_Macrophages',
    'Macrophages_2': '1_Macrophages',
    'Myoepi_ACTA2+': '10_Tumor_Adjacent_Myoepithelial_Cells',
    'Myoepi_KRT15+': '14_Normal_Myoepithelial_Cells',
    'CD4+_T_Cells': '6_T_Lymphocytes',
    'CD8+_T_Cells': '6_T_Lymphocytes',
    'B_Cells': '6_T_Lymphocytes',
    'LAMP3+_DCs': '1_Macrophages',
    'IRF7+_DCs': '1_Macrophages',
    'Mast_Cells': '1_Macrophages',
    'Stromal_&_T_Cell_Hybrid': '2_Stromal_Cells',
    'T_Cell_&_Tumor_Hybrid': '6_T_Lymphocytes',
    'Prolif_Invasive_Tumor': '0_Proliferative_Tumor_Cells',
    'Invasive_Tumor': 'Tumor_Cells', 'DCIS 1': 'Tumor_Cells', 'DCIS 2': 'Tumor_Cells',
    'Unlabeled': 'Unknown'
}

# =====================================================================
# Step 1: 载入空间元数据 (严格保持实际服务器路径)
# =====================================================================
print("Loading Xenium spatial metadata...")
data_dir = "/data_hou/BE/GraphST/code/data_cell/"
roi_df = pd.read_csv(os.path.join(data_dir, "Selection_breast_cancer.csv"), skiprows=[0, 1])
roi_df.columns = roi_df.columns.str.strip()
selected_cell_ids = roi_df.iloc[:, 0].astype(str).tolist()

cells_meta = pd.read_parquet("/data_hou/ST_data_new/Xenium_breast2/S1_Top/cells.parquet", engine="fastparquet")
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
# Step 2: 读取并改装官方矩阵 A
# =====================================================================
print("Loading and converting official Matrix A...")
path_mat_A = os.path.join(data_dir, "matrix_A_official_breast.h5ad")
adata_A = sc.read_h5ad(path_mat_A)
adata_A.obs['new_batch'] = 'Official_A'
adata_A.obs['ground_truth'] = 'Unknown'

# 物理洗涤 A 矩阵的原生细胞标签
if "celltype" in adata_A.obs.columns:
    adata_A.obs["celltype"] = [global_label_mapping.get(str(lbl), str(lbl)) for lbl in adata_A.obs["celltype"]]
else:
    adata_A.obs["celltype"] = "Unknown"

coord_A = selected_meta.loc[adata_A.obs_names, [x_col, y_col]].values
adata_A.obsm['spatial'] = coord_A.astype(np.float32) 

# =====================================================================
# Step 3: 实时读取分子点云组装矩阵 B (保留您的原位纯化拦截逻辑)
# =====================================================================
print("Processing cellpose mask and transcripts according to your exact logic...")
mask_file = os.path.join(data_dir, "roi_breast_dapi.tif")
mask = tifffile.imread(mask_file)
if len(mask.shape) == 3: 
    mask = mask[0]

df_transcripts = pd.read_parquet("/data_hou/ST_data_new/Xenium_breast2/S1_Top/transcripts.parquet", engine="fastparquet")

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
adata_B.obs['new_batch'] = 'Cellpose_B'
adata_B.obs['ground_truth'] = 'Unknown'

# 核心亮点：通过细胞ID跨文件就地召回并洗涤 B 矩阵的权威专家细胞标签
try:
    path_mat_B_anno = os.path.join(data_dir, "matrix_B_cellpose_breast.h5ad")
    if os.path.exists(path_mat_B_anno):
        adata_B_anno = sc.read_h5ad(path_mat_B_anno)
        lbl_col = 'final_celltype' if 'final_celltype' in adata_B_anno.obs.columns else 'celltype'
        
        anno_dict = {}
        for name, row in adata_B_anno.obs.iterrows():
            lbl = str(row[lbl_col])
            nums = re.findall(r'\d+', str(name))
            if nums:
                anno_dict[int(nums[0])] = lbl
                
        b_celltypes = []
        for obs_name in adata_B.obs_names:
            cell_id = int(obs_name.split("_")[-1])
            b_celltypes.append(global_label_mapping.get(anno_dict.get(cell_id, "Unknown"), anno_dict.get(cell_id, "Unknown")))
        adata_B.obs['celltype'] = b_celltypes
    else:
        adata_B.obs['celltype'] = 'Unknown'
except Exception as e:
    print(f"提取 B 矩阵现成标签失败，执行降级处理: {e}")
    adata_B.obs['celltype'] = 'Unknown'

# =====================================================================
# Step 4: 反推并注入 Cellpose 空间物理坐标
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
for obs_name in adata_B.obs_names:
    actual_id = int(obs_name.split("_")[-1])
    coord_B_microns.append(id_to_coord[actual_id])

adata_B.obsm['spatial'] = np.array(coord_B_microns, dtype=np.float32)
print("Spatial coordinates for both live matrices successfully injected.")

# =====================================================================
# Step 5: 特征交集过滤与 DeepST 整合前置预处理
# =====================================================================
print("Synchronizing gene features and preparing DeepST container...")

common_genes = adata_A.var_names.intersection(adata_B.var_names)
print(f"  -> Features aligned: Official_A ({adata_A.n_vars}) vs Cellpose_B ({adata_B.n_vars})")
print(f"  -> Merged shared gene size: {len(common_genes)}")

adata_A = adata_A[:, common_genes].copy()
adata_B = adata_B[:, common_genes].copy()

total_cells = adata_A.n_obs + adata_B.n_obs

# 物理修改输出目录到大盘指定的特定 results 文件夹下
save_path = "/data_hou/BE/DeepST/results/"
os.makedirs(save_path, exist_ok=True)

# 恢复标准 run 实例化语法
deepen = run(
    save_path = save_path, 
    task = "Integration",
    pre_epochs = 500, 
    epochs = 500, 
    use_gpu = True,
)

augement_data_list = []
graph_list = []
data_name_list = ['Official_A', 'Cellpose_B']

for adata_item in [adata_A, adata_B]:
    adata_item.obs["imagerow"] = adata_item.obsm["spatial"][:, 1] 
    adata_item.obs["imagecol"] = adata_item.obsm["spatial"][:, 0] 
    adata_item.obs["array_row"] = adata_item.obsm["spatial"][:, 1] 
    adata_item.obs["array_col"] = adata_item.obsm["spatial"][:, 0] 
    
    adata_item.obsm["image_feat_pca"] = np.zeros((adata_item.n_obs, 50), dtype=np.float32)
    
    adata_aug = deepen._get_augment(adata_item, spatial_type="LinearRegress")
    graph_dict = deepen._get_graph(adata_aug.obsm["spatial"], distType="KDTree")
    
    graph_list.append(graph_dict)
    augement_data_list.append(adata_aug)

multiple_adata, multiple_graph = deepen._get_multiple_adata(
    adata_list=augement_data_list, 
    data_name_list=data_name_list, 
    graph_list=graph_list
)
data = deepen._data_process(multiple_adata, pca_n_comps=200)

# =====================================================================
# Step 6: 启动 DeepST 核心模型训练与硬核 Benchmarking 效能监控
# =====================================================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training DeepST model via GCN-VAE and domain adversarial learning...")
deepst_embed = deepen._fit(
    data = data,
    graph_dict = multiple_graph,
    domains = multiple_adata.obs["batch"].values,
    n_domains = len(data_name_list)
)

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before
print("DeepST core alignment completed successfully!")

# 性能审计元数据指标面板 (100% 结构兼容师姐原版)
benchmark_results = {
    'method_name': 'DeepST',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': multiple_adata.n_obs,
    'total_genes': multiple_adata.n_vars,
    'embedding_dim': deepst_embed.shape[1],
    'n_datasets': len(data_name_list),
    'pre_epochs': 500,
    'epochs': 500,
    'timestamp': pd.Timestamp.now().isoformat()
}

# =====================================================================
# Step 7: 隐空间特征提取与大盘槽位规范化物理落盘 (彻底移除了多余聚类代码)
# =====================================================================
print("正在规范化重构流形特征槽位并执行 H5AD 持久化存储...")
adata_final = multiple_adata.copy()

# 将训练好的低维表征精准塞入大盘指定的特征流形槽位
adata_final.obsm["deepst"] = deepst_embed

# 双保险清洗并重构大盘检验所需的合并批次键与细胞类型属性
adata_final.obs['new_batch'] = adata_final.obs['batch'].astype(str)
adata_final.obs['celltype'] = adata_final.obs['celltype'].astype('category')

output_h5ad = os.path.join(save_path, "Xenium_DeepST_integrated.h5ad")
output_json = os.path.join(save_path, "deepst_xenium_benchmark.json")

adata_final.write(output_h5ad)
with open(output_json, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\n=======================================================")
print("DeepST 横向评测管线全部运行圆满结束！")
print(f"1. 流形校正特征 AnnData 已安全保存: {output_h5ad}")
print(f"2. 完整的 Benchmark 效率指标报告已保存: {output_json}")
print("=======================================================")
