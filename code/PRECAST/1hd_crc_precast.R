suppressPackageStartupMessages(library(Seurat))
options(Seurat.object.assay.version = "v3")

suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(PRECAST))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(zellkonverter))
suppressPackageStartupMessages(library(arrow))

# =============== Memory Monitoring Function ===============
get_memory_usage <- function() {
    mem_info <- gc(verbose = FALSE)
    used_memory <- sum(mem_info[, 2]) 
    return(used_memory * 8 / 1024^2)  
}

# =============== Loading Data===============
file_fold <- '/data_hou/ST_data_new/Visium_crc/'
file_bin <- 'binned_outputs/square_008um/'
out_dir <- "/data_hou/ST_data_new/model-zn/precast/results/"
file_meta <- 'HumanColonCancer_VisiumHD/MetaData/'
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

project_name <- "hd_crc"
datasets <- c("P1CRC", "P2CRC", "P5CRC")
n_domains <- 3

seuList <- list()
total_cells <- 0
total_genes <- 0

cat("Loading h5 data for", project_name, "...\n")

for (j in 1:length(datasets)) {
    dataset <- datasets[j]
    file_path <- file.path(file_fold, dataset, file_bin, 'filtered_feature_bc_matrix.h5')
    cat("  Loading slice", dataset, "from", file_path, "\n")
    counts <- Read10X_h5(file_path)
    spatial_path <- file.path(file_fold, dataset, file_bin, "spatial/tissue_positions.parquet")
    sp_df <- as.data.frame(arrow::read_parquet(spatial_path))
    rownames(sp_df) <- sp_df$barcode
    
    meta_path <- file.path(file_fold, file_meta, paste0(dataset, '_Metadata.parquet'))
    df_meta <- as.data.frame(arrow::read_parquet(meta_path))
    gt_map <- setNames(df_meta$Periphery, df_meta$barcode)
    all_cells <- colnames(counts)
    meta_data <- data.frame(row.names = all_cells)
    
    meta_data$row <- sp_df[all_cells, "array_row"]
    meta_data$col <- sp_df[all_cells, "array_col"]
    meta_data$ground_truth <- gt_map[all_cells]
    
    keep_idx <- !is.na(meta_data$ground_truth)
    meta_data <- meta_data[keep_idx, , drop=FALSE]
    counts <- counts[, keep_idx, drop=FALSE]
    print(colnames(meta_data))
    print(head(meta_data$ground_truth))
    
    seu <- CreateSeuratObject(counts = counts, project = dataset)
    seu <- AddMetaData(seu, metadata = meta_data)
    if (!"ground_truth" %in% colnames(seu@meta.data)) {
        stop("Warning: metadata loading error")
    }
    target_cells <- 15000
    if (ncol(seu) > target_cells) {
        cat("  Stratified Subsampling for", dataset, "...\n")
        sampling_fraction <- target_cells / ncol(seu)
        
        cells_to_keep <- unlist(lapply(unique(seu$ground_truth), function(gt) {
            cells_in_gt <- colnames(seu)[seu$ground_truth == gt]
            n_sample <- ceiling(length(cells_in_gt) * sampling_fraction)
            sample(cells_in_gt, min(length(cells_in_gt), n_sample))
        }))
        seu <- subset(seu, cells = cells_to_keep)
    }
    seu <- RenameCells(seu, add.cell.id = paste0("slice", dataset))
	
    total_cells <- total_cells + ncol(seu)
    if (j == 1) total_genes <- nrow(seu)
    
    seuList[[j]] <- seu
    
    rm(counts, sp_df, df_meta, meta_data)
    gc()
}

# =============== Preprocessing ===============
cat("Preprocessing datasets...\n")

min_genes <- min(sapply(seuList, nrow))
num_features <- min(5000, min_genes)
cat("Using", num_features, "highly variable features for integration.\n")

seuList <- lapply(seuList, function(x) {
    print(x)
    assay_name <- DefaultAssay(x)
    cat("Default Assay name:", assay_name, "\n")
    x[["RNA"]] <- as(object = x[["RNA"]], Class = "Assay")
    x <- NormalizeData(x)
    x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = num_features)
    x <- ScaleData(x) 
    return(x)
})

memory_before <- get_memory_usage()
start_time <- Sys.time()

# =============== PRECAST Integration ===============
cat("Creating PRECAST Object...\n")
PRECASTObj <- CreatePRECASTObject(seuList, project = project_name, gene.number = num_features, 
                                  selectGenesMethod = "SPARK-X", premin.spots = 20, 
                                  premin.features = 20, postmin.spots = 1, postmin.features = 10)

cat("Adding Adjacency List...\n")
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium HD") 

cat("Setting Parameters...\n")
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)
PRECASTObj <- PRECAST(PRECASTObj, K = n_domains)

end_time <- Sys.time()
training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
memory_after <- get_memory_usage()
memory_used <- memory_after - memory_before

cat("Training completed! Merging and exporting results...\n")

# =============== Post-processing ===============
resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human") 

saveRDS(PRECASTObj, file = paste0(out_dir, "PRECASTObj_", project_name, ".rds"))

# =============== Benchmark JSON ===============
final_cells <- sum(unlist(lapply(PRECASTObj@seuList, function(x) ncol(x))))
embedding_dim <- ncol(PRECASTObj@resList$hZ[[1]])

benchmark_results <- list(
    method_name = "PRECAST",
    dataset = project_name,
    training_time_seconds = training_time,
    training_time_minutes = training_time / 60,
    memory_usage_mb = memory_used,
    total_cells = total_cells,
    final_cells = final_cells,
    total_genes = total_genes,
    n_datasets = length(datasets),
    embedding_dim = embedding_dim, 
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)

write_json(benchmark_results, paste0(out_dir, "precast_benchmark_", project_name, ".json"), pretty = TRUE, auto_unbox = TRUE)


cat("Loading raw data directly from H5 for export...\n")
raw_obj_list <- list()

for (dataset in datasets) {
    file_path <- file.path(file_fold, dataset, file_bin, 'filtered_feature_bc_matrix.h5')
    counts <- Read10X_h5(file_path)
    spatial_path <- file.path(file_fold, dataset, file_bin, "spatial/tissue_positions.parquet")
    sp_df <- as.data.frame(arrow::read_parquet(spatial_path))
    rownames(sp_df) <- sp_df$barcode
    
    meta_path <- file.path(file_fold, file_meta, paste0(dataset, '_Metadata.parquet'))
    df_meta <- as.data.frame(arrow::read_parquet(meta_path))
    gt_map <- setNames(df_meta$Periphery, df_meta$barcode)
    all_cells <- colnames(counts)
    meta_data <- data.frame(row.names = all_cells)
    
    meta_data$row <- sp_df[all_cells, "array_row"]
    meta_data$col <- sp_df[all_cells, "array_col"]
    meta_data$ground_truth <- gt_map[all_cells]
    
    keep_idx <- !is.na(meta_data$ground_truth)
    meta_data <- meta_data[keep_idx, , drop=FALSE]
    counts <- counts[, keep_idx, drop=FALSE]
    
    obj <- CreateSeuratObject(counts = counts, project = dataset)
    obj <- AddMetaData(obj, metadata = meta_data)
    if (!"ground_truth" %in% colnames(obj@meta.data)) {
        stop("Warning: metadata loading error")
    }
    target_cells <- 15000
    if (ncol(obj) > target_cells) {
        cat("  Stratified Subsampling for", dataset, "...\n")
        sampling_fraction <- target_cells / ncol(obj)
        
        cells_to_keep <- unlist(lapply(unique(obj$ground_truth), function(gt) {
            cells_in_gt <- colnames(obj)[obj$ground_truth == gt]
            n_sample <- ceiling(length(cells_in_gt) * sampling_fraction)
            sample(cells_in_gt, min(length(cells_in_gt), n_sample))
        }))
        obj <- subset(obj, cells = cells_to_keep)
    }
    obj <- RenameCells(obj, add.cell.id = paste0("slice", dataset))
    raw_obj_list[[dataset]] <- obj
    
    rm(counts, sp_df, df_meta, meta_data)
    gc()
}

raw_combined <- merge(raw_obj_list[[1]], y = raw_obj_list[-1])
raw_counts <- GetAssayData(raw_combined, assay = "RNA", layer = "counts")

common_cells <- intersect(colnames(raw_counts), colnames(seuInt))

cat(paste("Successfully Mapping", length(common_cells), "cells \n"))

raw_counts <- raw_counts[, common_cells, drop=FALSE]
seuInt <- subset(seuInt, cells = common_cells)

sce_int <- SingleCellExperiment(
    assays = list(counts = raw_counts),
    colData = colData(as.SingleCellExperiment(seuInt))
)

if ("PRECAST" %in% names(seuInt@reductions)) {
    mat <- Embeddings(seuInt, "PRECAST")
} else {
    mat <- do.call(rbind, PRECASTObj@resList$hZ)
    rownames(mat) <- unlist(lapply(PRECASTObj@seuList, colnames))
}
reducedDim(sce_int, "X_PRECAST") <- mat[colnames(sce_int), ]

meta <- raw_combined@meta.data[colnames(sce_int), , drop = FALSE]
spatial_coords <- as.matrix(cbind(meta$col, meta$row))
rownames(spatial_coords) <- colnames(sce_int)
reducedDim(sce_int, "spatial") <- spatial_coords

if ("ground_truth" %in% colnames(raw_combined@meta.data)) {
    colData(sce_int)$ground_truth <- raw_combined@meta.data[colnames(sce_int),"ground_truth"]
}

if (!is.null(PRECASTObj@resList$ident)) {
    ident_vec <- unlist(PRECASTObj@resList$ident)
    names(ident_vec) <- unlist(lapply(PRECASTObj@seuList, colnames))
    ident_vec <- ident_vec[colnames(sce_int)]
    colData(sce_int)$precast_cluster <- factor(ident_vec)
}

colData(sce_int)$slice_id <- colData(sce_int)$orig.ident

cat("Saving Final H5AD...\n")
if (!inherits(assay(sce_int, "counts"), "CsparseMatrix")) {
    assay(sce_int, "counts") <- as(assay(sce_int, "counts"), "CsparseMatrix")
}

final_h5ad_path <- paste0(out_dir, "hd_crc_seuInt_with_all_spatial.h5ad")
zellkonverter::writeH5AD(sce_int, final_h5ad_path)

cat("Done")