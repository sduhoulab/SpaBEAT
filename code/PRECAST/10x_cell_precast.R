setwd("D:/Xenium Explorer/Breast_biomarker_2/")
rm(list = ls())

install.packages('zellkonverter')
BiocManager::install("zellkonverter")
install.packages("harmony")

library(PRECAST)
library(Seurat)

suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(PRECAST))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(zellkonverter))

# 内存监控函数配置
get_memory_usage <- function() {
  mem_info <- gc(verbose = FALSE)
  used_memory <- sum(mem_info[, 2]) 
  return(used_memory * 8 / 1024^2) 
}

# 设定随机种子确保流形结果可重复
set.seed(1234)


# 全局大统一：细胞大类标签规范化清洗字典
global_label_mapping <- c(
  '9_Tumor_Cells' = 'Tumor_Cells', '8_Tumor_Cells' = 'Tumor_Cells', '7_Tumor_Cells' = 'Tumor_Cells',
  '4_Tumor_Cells' = 'Tumor_Cells', '12_Tumor_Cells' = 'Tumor_Cells', '15_Tumor_Cells' = 'Tumor_Cells',
  'Endothelial' = '5_Endothelial_Cells',
  'Stromal' = '2_Stromal_Cells',
  'Perivascular-Like' = '3_Perivascular_Cells',
  'Macrophages_1' = '1_Macrophages',
  'Macrophages_2' = '1_Macrophages',
  'Myoepi_ACTA2+' = '10_Tumor_Adjacent_Myoepithelial_Cells',
  'Myoepi_KRT15+' = '14_Normal_Myoepithelial_Cells',
  'CD4+_T_Cells' = '6_T_Lymphocytes',
  'CD8+_T_Cells' = '6_T_Lymphocytes',
  'B_Cells' = '6_T_Lymphocytes',
  'LAMP3+_DCs' = '1_Macrophages',
  'IRF7+_DCs' = '1_Macrophages',
  'Mast_Cells' = '1_Macrophages',
  'Stromal_&_T_Cell_Hybrid' = '2_Stromal_Cells',
  'T_Cell_&_Tumor_Hybrid' = '6_T_Lymphocytes',
  'Prolif_Invasive_Tumor' = '0_Proliferative_Tumor_Cells',
  'Invasive_Tumor' = 'Tumor_Cells', 'DCIS 1' = 'Tumor_Cells', 'DCIS 2' = 'Tumor_Cells',
  'Unlabeled' = 'Unknown'
)

# =====================================================================
# 1. 数据本地读入 (路径指向您 Windows 的实际选区存放目录)
# =====================================================================
print("正在从本地读取 Xenium H5AD 原始矩阵...")
data_dir <- "D:/Xenium Explorer/两个矩阵_乳腺癌完全体/"
path_A <- file.path(data_dir, "matrix_A_official_breast.h5ad")
path_B <- file.path(data_dir, "matrix_B_cellpose_breast_annotated_final.h5ad")
# 利用 zellkonverter 算子在内存中直接解算 H5AD
sce_A <- zellkonverter::readH5AD(path_A)
sce_B <- zellkonverter::readH5AD(path_B)

# 物理剥离表达矩阵与元数据表格
counts_A <- assay(sce_A, 1)
counts_B <- assay(sce_B, 1)
meta_A <- as.data.frame(colData(sce_A))
meta_B <- as.data.frame(colData(sce_B))

# 提取并交叉对齐低维原位空间坐标
spatial_A <- if ("spatial" %in% reducedDimNames(sce_A)) reducedDim(sce_A, "spatial") else reducedDim(sce_A, 1)
spatial_B <- if ("spatial" %in% reducedDimNames(sce_B)) reducedDim(sce_B, "spatial") else reducedDim(sce_B, 1)

# 核心修正：PRECAST 要求空间单细胞原位坐标在元数据中必须严格命名为 row 和 col
meta_A$row <- spatial_A[, 2]
meta_A$col <- spatial_A[, 1]
meta_B$row <- spatial_B[, 2]
meta_B$col <- spatial_B[, 1]

meta_A$slice_id <- "Official_A"
meta_B$slice_id <- "Cellpose_B"

# =====================================================================
# 2. 细胞标签前置物理洗涤 (确保跨样本锚点匹配绝对稳健)
# =====================================================================
print("正在执行前置细胞大类标签规范化清洗...")
if ("celltype" %in% colnames(meta_A)) {
  mapped_labels_A <- global_label_mapping[as.character(meta_A$celltype)]
  mapped_labels_A[is.na(mapped_labels_A)] <- as.character(meta_A$celltype)[is.na(mapped_labels_A)]
  meta_A$celltype <- mapped_labels_A
} else {
  meta_A$celltype <- "Unknown"
}

if ("final_celltype" %in% colnames(meta_B)) {
  raw_b <- as.character(meta_B$final_celltype)
} else if ("celltype" %in% colnames(meta_B)) {
  raw_b <- as.character(meta_B$celltype)
} else {
  raw_b <- rep("Unknown", nrow(meta_B))
}
mapped_labels_B <- global_label_mapping[raw_b]
mapped_labels_B[is.na(mapped_labels_B)] <- raw_b[is.na(mapped_labels_B)]
meta_B$celltype <- mapped_labels_B

# 实例化构建标准的 Seurat 容器列表
seu_A <- CreateSeuratObject(counts = counts_A, meta.data = meta_A)
seu_B <- CreateSeuratObject(counts = counts_B, meta.data = meta_B)
seuList <- list(seu_A, seu_B)

total_cells <- ncol(seu_A) + ncol(seu_B)
total_genes <- nrow(seu_A)

# =====================================================================
# 3. PRECAST 核心图网络模型建立与性能打点测试
# =====================================================================
print("正在准备启动 PRECAST 模型训练与效能打点...")
gc(verbose = FALSE)
memory_before <- get_memory_usage()
training_start_time <- Sys.time()

# 筛选 2000 个高变基因作为图特征基底
PRECASTObj <- CreatePRECASTObject(seuList = seuList, selectGenesMethod = "HVGs", gene.number = 2000)

# 核心修正：由于 Xenium 属于连续不规则单细胞成像技术，必须指定 platform = "ST"
PRECASTObj <- AddAdjList(PRECASTObj, platform = "ST")

# 设定条件随机场模型参数与对冲噪声迭代步数
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)

# 启动核心期望最大化算法进行图流形整合，分群数 K 设定为乳腺癌的标准 8 类空间域
PRECASTObj <- PRECAST(PRECASTObj, K = 8)

training_end_time <- Sys.time()
training_time <- as.numeric(difftime(training_end_time, training_start_time, units = "secs"))
memory_after <- get_memory_usage()
memory_used <- memory_after - memory_before
print("PRECAST 核心整合阶段运行成功！")

# =====================================================================
# 4. 后处理与大盘槽位规范化物理落盘
# =====================================================================
print("正在抽取校正特征低维流形并重构大盘检验槽位...")
PRECASTObj <- SelectModel(PRECASTObj)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")

# 核心修正：由于 PRECAST 的 IntegrateSpaData 函数默认只保留基础的 batch 和 cluster 标签
# 导致原输入对象的自定义元数据 celltype 丢失。我们需要手动将其从原始对象中召回映射。

# 1. 汇总原始矩阵 A 和 B 的所有细胞条形码名称与对应的细胞类型标签
meta_combined <- data.frame(
  cell_name = c(rownames(seu_A@meta.data), rownames(seu_B@meta.data)),
  celltype = c(as.character(seu_A$celltype), as.character(seu_B$celltype)),
  stringsAsFactors = FALSE
)

# 2. 尝试直接通过细胞名匹配召回
matched_idx <- match(colnames(seuInt), meta_combined$cell_name)
seuInt$celltype <- meta_combined$celltype[matched_idx]

# 3. 鲁棒性防御：如果 Seurat 在集成时自动为细胞名添加了批次后缀（如 _1, _2 或 -1, -2），则剥离后缀后再执行匹配
if (any(is.na(seuInt$celltype))) {
  clean_names <- sub("_[123456789]$", "", colnames(seuInt))
  clean_names <- sub("-[123456789]$", "", clean_names)
  matched_idx_clean <- match(clean_names, meta_combined$cell_name)
  seuInt$celltype <- meta_combined$celltype[matched_idx_clean]
}

# 4. 对未匹配上的极端样本进行降级填充，并强转为大盘指定的 factor 属性
seuInt$celltype[is.na(seuInt$celltype)] <- "Unknown"
seuInt$celltype <- as.factor(seuInt$celltype)

# 规范化大盘所需的键名与合并批次槽位，防止 downstream 评测 KeyError 闪退
seuInt$new_batch <- seuInt$batch
seuInt$batch <- seuInt$batch

# 物理创建本地输出路径
save_path <- "D:/Xenium Explorer/PRECAST_results/"
if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}

final_cells <- ncol(seuInt)
embedding_dim <- ncol(PRECASTObj@resList$hZ[[1]])

# 性能能耗审计日志字典 (100% 完全结构兼容大盘)
benchmark_results <- list(
  method_name = "PRECAST",
  training_time_seconds = training_time,
  training_time_minutes = training_time / 60,
  training_time_hours = training_time / 3600,
  memory_usage_mb = memory_used,
  memory_usage_gb = memory_used / 1024,
  total_cells = total_cells,
  final_cells = final_cells,
  total_genes = total_genes,
  n_datasets = 2,
  max_iterations = 30,
  gene_number = 2000,
  embedding_dim = embedding_dim, 
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)

# 转换为 SingleCellExperiment 导出通用的 H5AD 数据
sce <- as.SingleCellExperiment(seuInt)

# 双保险：重定义降维嵌入名称，确保大盘读取 rep="precast" 时完美命中
if ("PRECAST" %in% reducedDimNames(sce)) {
  reducedDim(sce, "precast") <- reducedDim(sce, "PRECAST")
}

output_h5ad <- file.path(save_path, "Xenium_PRECAST_integrated.h5ad")
output_json <- file.path(save_path, "precast_xenium_benchmark.json")

zellkonverter::writeH5AD(sce, output_h5ad)
write_json(benchmark_results, output_json, pretty = TRUE, auto_unbox = TRUE)

print("=======================================================")
print("PRECAST 本地 Windows 评测管线全部圆满结束！")
print(paste("1. 流形校正特征已成功写入:", output_h5ad))
print(paste("2. 能耗审计报告已成功写入:", output_json))
print("=======================================================")



# =====================================================================
# DeepST 隐空间表征正统 mclust 聚类本地物理修复脚本
# 文件名：patch_deepst_mclust.R
# =====================================================================
suppressPackageStartupMessages(library(zellkonverter))
suppressPackageStartupMessages(library(mclust))
suppressPackageStartupMessages(library(SingleCellExperiment))

# 设定随机种子，确保聚类划分结果与大盘中其他算法具有可重复对比性
set.seed(1234)

# 1. 读取下载到本地的 DeepST 整合矩阵
h5ad_path <- "D:/Xenium Explorer/两个矩阵_乳腺癌完全体/Xenium_DeepST_integrated.h5ad"
print("正在读取本地 DeepST H5AD 对象...")
sce <- zellkonverter::readH5AD(h5ad_path)

# 2. 提取 DeepST 在隐空间中产出的低维特征流形
# 当 readH5AD 读取 obsm['deepst'] 时，会将其自动存入 reducedDim 的相应槽位中
print("正在提取 deepst 隐空间低维特征流形...")
embedding <- reducedDim(sce, "deepst")

# 3. 运行严格的 mclust 聚类，空间域数量 K 与大盘完全对齐设定为 8
print("正在激活正统 R 语言 mclust 混合高斯聚类算子 (5万细胞级别计算预计需要1-2分钟)...")
fit <- Mclust(embedding, G = 8)

# 4. 将聚类编号严格转换为字符串，并精准注入到大盘硬编码期待的 mclust 属性列中
colData(sce)$mclust <- as.character(fit$classification)
print("-> mclust 空间域聚类计算成功，标签已成功写入元数据！")

# 5. 覆盖写回原 H5AD 文件
print("正在执行 H5AD 文件就地持久化覆盖落盘...")
zellkonverter::writeH5AD(sce, h5ad_path)

print("=======================================================")
# 强行清除系统环境变量干扰，让 basilisk 恢复自主沙盒环境控制
print("DeepST 矩阵正统 mclust 标签修复补丁全部运行圆满结束！")
print("=======================================================")






