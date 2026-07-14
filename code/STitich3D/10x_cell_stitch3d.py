# =====================================================================
# Xenium单细胞原位成像矩阵定制版 STitch3D 算法脚本 (坐标槽位补齐版)
# 文件名：xenium_stitch3d.py
# =====================================================================
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import gc
import json
import torch
import psutil
import numpy as np
import pandas as pd
import pandas.core.indexes.base as pd_base
import scanpy as sc
import anndata as ad
import STitch3D

# 针对新版 Pandas 的兼容性补丁，恢复旧版 Index 的 & 交集运算行为
def patched_and(self, other):
    return self.intersection(other)

pd.Index.__and__ = patched_and
pd_base.Index.__and__ = patched_and

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# 硬件与随机种子配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
used_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(used_device)
np.random.seed(1234)

# =====================================================================
# 1. 数据读入 (路径指向实际 data_cell 目录)
# =====================================================================
print("正在读取原始 Xenium 空间单细胞矩阵...")
data_dir = "/data_hou/BE/GraphST/code/data_cell/"
path_A = os.path.join(data_dir, "matrix_A_official_breast.h5ad")
path_B = os.path.join(data_dir, "matrix_B_cellpose_breast.h5ad")

adata_A = sc.read_h5ad(path_A)
adata_B = sc.read_h5ad(path_B)

if 'spatial' not in adata_A.obsm and 'X_spatial' in adata_A.obsm:
    adata_A.obsm['spatial'] = adata_A.obsm['X_spatial']
if 'spatial' not in adata_B.obsm and 'X_spatial' in adata_B.obsm:
    adata_B.obsm['spatial'] = adata_B.obsm['X_spatial']

# 核心修正：由于两个矩阵天然对齐，直接将原生坐标复制给 STitch3D 必需的 spatial_aligned 槽位
adata_A.obsm['spatial_aligned'] = adata_A.obsm['spatial'].copy()
adata_B.obsm['spatial_aligned'] = adata_B.obsm['spatial'].copy()

# =====================================================================
# 2. 细胞标签前置对齐 (STitch3D 算法参考集 anchors 锚定硬性要求)
# =====================================================================
print("正在执行前置细胞大类标签规范化洗涤...")
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

# 物理清洗 A/B 标签，强转 category 属性
adata_A.obs["celltype"] = [global_label_mapping.get(lbl, lbl) for lbl in adata_A.obs["celltype"].astype(str)]
if "final_celltype" in adata_B.obs.columns:
    raw_labels_B = adata_B.obs.apply(
        lambda r: str(r["final_celltype"]) if pd.notna(r["final_celltype"]) and str(r["final_celltype"]) != 'nan' else str(r["celltype"]), axis=1
    ).astype(str)
else:
    raw_labels_B = adata_B.obs["celltype"].astype(str)
adata_B.obs["celltype"] = [global_label_mapping.get(lbl, lbl) for lbl in raw_labels_B]

adata_A.obs['slice_id'] = 'Official_A'
adata_B.obs['slice_id'] = 'Cellpose_B'
adata_A.obs['batch'] = 'Official_A'
adata_B.obs['batch'] = 'Cellpose_B'

# =====================================================================
# 3. 构建单细胞参考集 (借用对齐后的高精度 A 矩阵)
# =====================================================================
print("正在构建单细胞解离参考集...")
adata_ref = adata_A.copy()
adata_ref.obs['group'] = 'Official_A'

celltype_list_use = sorted([lbl for lbl in adata_ref.obs['celltype'].unique() if lbl != 'Unknown'])

common_genes = adata_A.var_names.intersection(adata_B.var_names)
adata_A = adata_A[:, common_genes].copy()
adata_B = adata_B[:, common_genes].copy()

adatas = [adata_A, adata_B]
total_cells = adata_A.n_obs + adata_B.n_obs

# =====================================================================
# 4. STitch3D空间预处理
# =====================================================================
print("正在进行三维图网格空间交叉预处理...")

hvg_group_param = 300
c2c_dist_param = 15.0  # 真实单细胞平均中心距离设定为15微米

# 直接向预处理注入adatas列表，利用coor_key指定原生坐标槽位
adata_input, adata_basis = STitch3D.utils.preprocess(
    adatas,
    adata_ref,
    celltype_ref=celltype_list_use,
    sample_col="group",
    coor_key="spatial",
    c2c_dist=c2c_dist_param,
    n_hvg_group=hvg_group_param,
    slice_dist_micron=None  # 设为None代表两矩阵为空间完全重叠的平行重复样本
)

# =====================================================================
# 5. 模型训练与基准性能审计
# =====================================================================
gc.collect()
memory_before = get_memory_usage()
training_start_time = time.time()

print("开始训练 STitch3D 深度整合网络模型...")
model = STitch3D.model.Model(adata_input, adata_basis)
model.train()

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before
print("STitch3D 训练完成！")

save_path = "/data_hou/BE/STitch3D/results/"
os.makedirs(save_path, exist_ok=True)
result = model.eval(adatas, save=True, output_path=save_path)

# =====================================================================
# 6. 隐空间提取、槽位规范化物理落盘
# =====================================================================
print("正在构建最终对接评测大盘的 Anndata 对象...")
adata_final = model.adata_st.copy()

# 完美对齐大盘指定的特征流形槽位与批次键名
adata_final.obsm['stitch3d'] = adata_final.obsm['latent']
adata_final.obs['new_batch'] = adata_final.obs['slice_id'].astype(str)
adata_final.obs['batch'] = adata_final.obs['slice_id'].astype(str)
adata_final.obs['celltype'] = adata_final.obs['celltype'].astype('category')

# 强力恢复原生的细胞二维物理坐标，防止细胞索引顺序形变
if 'spatial' not in adata_final.obsm:
    spatial_map = pd.concat([
        pd.DataFrame(adata_A.obsm['spatial'], index=adata_A.obs_names),
        pd.DataFrame(adata_B.obsm['spatial'], index=adata_B.obs_names)
    ])
    adata_final.obsm['spatial'] = spatial_map.loc[adata_final.obs_names].values

output_h5ad_file = os.path.join(save_path, "Xenium_STitch3D_integrated.h5ad")
adata_final.write(output_h5ad_file)
print(f"整合完毕！矩阵已物理落盘至: {output_h5ad_file}")

# =====================================================================
# 7. 性能审计日志 (已全量补齐师姐原版全量审计键值)
# =====================================================================
benchmark_results = {
    'method_name': 'STitch3D',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'total_genes': adata_input.n_vars,
    'final_cells': model.adata_st.n_obs,
    'embedding_dim': model.adata_st.obsm['latent'].shape[1],
    'n_datasets': len(adatas),
    'random_seed': 1234,
    'n_hvg_group': hvg_group_param,
    'slice_dist_micron': None,
    'device': str(used_device),
    'timestamp': pd.Timestamp.now().isoformat()
}

with open(os.path.join(save_path, "stitch3d_xenium_benchmark.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)
print("性能审计日志已完整导出。流水线安全交卷！")
