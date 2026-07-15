suppressPackageStartupMessages(library(Seurat))
options(Seurat.object.assay.version = "v3")

suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(PRECAST))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(zellkonverter))

# =============== Memory Monitoring Function ===============
get_memory_usage <- function() {
    mem_info <- gc(verbose = FALSE)
    used_memory <- sum(mem_info[, 2]) 
    return(used_memory * 8 / 1024^2)  
}

# =============== Data Setup ===============
file_fold <- "/root/ST0507/data/Her2_tumor_converted/A/"
datasets <- c('A1', 'A2', 'A3', 'A4', 'A5', 'A6')
out_dir <- "/root/ST0507/new_model/PRECAST/results/"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

seuList <- list()
total_cells <- 0
total_genes <- 0

print("Loading Her2_A h5ad data...")

for (i in 1:length(datasets)) {
    dataset <- datasets[i]
    file_path <- file.path(file_fold, dataset, paste0(dataset, ".h5ad"))
    cat("Loading", dataset, "from", file_path, "\n")
    
    sce <- readH5AD(file_path)
    
    cnts <- assay(sce, "X")
    if (is.null(cnts)) {
        cnts <- assay(sce, "counts")
    }
    
    meta_data <- as.data.frame(colData(sce))
    
    sp_coords <- reducedDim(sce, "spatial")
    meta_data$row <- sp_coords[, 1]
    meta_data$col <- sp_coords[, 2]
    
    seu <- CreateSeuratObject(counts = cnts, meta.data = meta_data, project = dataset)
    
    total_cells <- total_cells + ncol(seu)
    if (i == 1) total_genes <- nrow(seu)
    
    seuList[[i]] <- seu
}

# =============== Preprocessing ===============
print("Preprocessing datasets...")
seuList <- lapply(seuList, function(x) {
    x[["RNA"]] <- as(object = x[["RNA"]], Class = "Assay")
    
    x <- NormalizeData(x)
    x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
    x <- ScaleData(x) 
    
    return(x)
})

memory_before <- get_memory_usage()
start_time <- Sys.time()

# =============== PRECAST Integration ===============
print("Creating PRECAST Object...")
PRECASTObj <- CreatePRECASTObject(seuList, project = "Her2_A", gene.number = 2000, 
                                  selectGenesMethod = "SPARK-X", premin.spots = 20, 
                                  premin.features = 20, postmin.spots = 1, postmin.features = 10)

print("Adding Adjacency List...")
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium")

print("Setting Parameters...")
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)
PRECASTObj <- PRECAST(PRECASTObj, K = 5)

end_time <- Sys.time()
training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
memory_after <- get_memory_usage()
memory_used <- memory_after - memory_before

print("Training completed! Merging and exporting results...")

# =============== Export and Post-processing ===============
saveRDS(PRECASTObj, file =  paste0(out_dir, "her2_A_PRECASTObj.rds"))

# =============== Post-processing (not included in benchmark) ===============
print("Performing post-processing...")

## backup the fitting results in resList 
resList <- PRECASTObj@resList
print(PRECASTObj@resList)
PRECASTObj <- SelectModel(PRECASTObj)
print(PRECASTObj@seuList)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")
print(seuInt)
saveRDS(seuInt, file =  paste0(out_dir, "her2_A_seuInt.rds"))

# =============== Calculate final statistics ===============
final_cells <- sum(unlist(lapply(PRECASTObj@seuList, function(x) ncol(x))))
final_genes <- nrow(PRECASTObj@seuList[[1]])
embedding_dim <- ncol(PRECASTObj@resList$hZ[[1]])


# =============== Benchmark JSON ===============

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
    n_datasets = length(datasets),
    max_iterations = 30,
    embedding_dim = embedding_dim, 
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)

write_json(benchmark_results, paste0(out_dir, "her2_A_precast_benchmark.json"), pretty = TRUE, auto_unbox = TRUE)


library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)
library(Matrix)

PRECASTObj <- readRDS(paste0(out_dir, "her2_A_PRECASTObj.rds"))
seuInt <- readRDS(paste0(out_dir, "her2_A_seuInt.rds"))
raw_data_root <- "/root/ST0507/data/Her2_tumor_converted/A/" 
sample_ids <- c('A1', 'A2', 'A3', 'A4', 'A5', 'A6') 
raw_obj_list <- list()

print("Loading Her2_A data directly from H5ADs...")
raw_obj_list <- list()

for (dataset in datasets) {
    file_path <- file.path(file_fold, dataset, paste0(dataset, ".h5ad"))
    cat("Processing", dataset, "...\n")
    sce <- readH5AD(file_path)
    
    cnts <- assay(sce, "X")
    if (is.null(cnts)) cnts <- assay(sce, "counts")
    
    meta_data <- as.data.frame(colData(sce))
    
    sp_coords <- reducedDim(sce, "spatial")
    meta_data$row <- sp_coords[, 1]
    meta_data$col <- sp_coords[, 2]
    
    if ("ground_truth" %in% colnames(meta_data)) {
        colnames(meta_data)[colnames(meta_data) == "ground_truth"] <- "ground_truth"
    }
   
    obj <- CreateSeuratObject(counts = cnts, meta.data = meta_data, project = dataset)

    raw_obj_list[[dataset]] <- obj
}


print("Merging...")
raw_combined <- merge(
    raw_obj_list[[1]],
    raw_obj_list[-1]
)
raw_counts <- GetAssayData(
    raw_combined,
    assay = "RNA",
    slot = "counts"
)

colnames(raw_counts) <- gsub("_", "", colnames(raw_counts))
common_cells <- intersect(
    colnames(raw_counts),
    colnames(seuInt)
)
length(common_cells)

raw_counts <- raw_counts[, common_cells]

seuInt <- subset(
    seuInt,
    cells = common_cells
)

raw_counts <- raw_counts[, colnames(seuInt)]

print(all(colnames(raw_counts) == colnames(seuInt)))

sce_int <- SingleCellExperiment(
    assays = list(counts = raw_counts),
    colData = colData(as.SingleCellExperiment(seuInt))
)

colnames(raw_counts) <- sub(
    "_(\\d+)$",
    "\\1",
    colnames(raw_counts)
)

rownames(raw_combined@meta.data) <- sub(
    "_(\\d+)$",
    "\\1",
    rownames(raw_combined@meta.data)
)

if ("PRECAST" %in% names(seuInt@reductions)) {

    emb <- Embeddings(seuInt, "PRECAST")
    emb <- emb[colnames(sce_int), ]
    reducedDim(sce_int, "X_PRECAST") <- emb

} else if ("position" %in% names(seuInt@reductions)) {

    emb <- Embeddings(seuInt, "position")
    emb <- emb[colnames(sce_int), ]
    reducedDim(sce_int, "X_PRECAST") <- emb

} else {

    mat <- do.call(rbind, PRECASTObj@resList$hZ)
    reducedDim(sce_int, "X_PRECAST") <- mat[colnames(sce_int), ]

}

if (all(c("row", "col") %in% names(colData(sce_int)))) {

    spatial_coords <- as.matrix(
        cbind(
            colData(sce_int)$col,
            colData(sce_int)$row
        )
    )

} else {
    meta <- raw_combined@meta.data[
        colnames(sce_int),
        ,
        drop = FALSE
    ]
    spatial_coords <- as.matrix(
        cbind(meta$col, meta$row)
    )
}

rownames(spatial_coords) <- colnames(sce_int)
reducedDim(sce_int, "spatial") <- spatial_coords


if ("ground_truth" %in% colnames(raw_combined@meta.data)) {
    gt <- raw_combined@meta.data[colnames(sce_int),"ground_truth"]
    colData(sce_int)$ground_truth <- gt
    print("Ground Truth added successfully.")
}


if (!is.null(PRECASTObj@resList$ident)) {
    ident_vec <- unlist(PRECASTObj@resList$ident)
    names(ident_vec) <- unlist(
        lapply(PRECASTObj@seulist, colnames)
    )
    names(ident_vec) <- gsub("_", "", names(ident_vec))
    ident_vec <- ident_vec[colnames(sce_int)]
    colData(sce_int)$precast_cluster <- factor(ident_vec)
}

if ("orig.ident" %in% colnames(colData(sce_int))) {
    colData(sce_int)$slice_id <- colData(sce_int)$orig.ident
}

print("Saving H5AD...")
if (!inherits(assay(sce_int, "counts"), "CsparseMatrix")) {
    assay(sce_int, "counts") <- as(
        assay(sce_int, "counts"),
        "CsparseMatrix"
    )
}
dim(sce_int)
dim(reducedDim(sce_int, "X_PRECAST"))
dim(reducedDim(sce_int, "spatial"))
nrow(colData(sce_int))

zellkonverter::writeH5AD(sce_int, paste0(out_dir, "her2_A_seuInt_with_all_spatial.h5ad"))


#### python
import scanpy as sc
import squidpy as sq
import numpy as np

file_path = '/root/ST0507/new_model/PRECAST/results/her2_A_seuInt_with_all_spatial.h5ad'
print(f"Loading {file_path}...")
adata = sc.read_h5ad(file_path)

if 'ground_truth' in adata.obs:
    adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
    adata = adata[~adata.obs['celltype'].isna()]
    adata = adata[adata.obs['celltype'] != "unknown"].copy()

if 'spatial' in adata.obsm:
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=False)
    print("Spatial graph computed.")
else:
    print("Error: No spatial coordinates found!")

batch_mapping = {
    '1': 'A1', 
    '2': 'A2',
    '3': 'A3',
    '4': 'A4',
    '5': 'A5',
    '6': 'A6'
}

if 'batch' in adata.obs:
    mapped_batches = adata.obs['batch'].astype(str).map(batch_mapping)
    adata.obs['new_batch'] = mapped_batches.fillna(adata.obs['batch'])
else:
    adata.obs['new_batch'] = adata.obs['orig.ident']


if 'PRECAST' in adata.obsm:
    adata.obsm['PRECAST_embed'] = adata.obsm['PRECAST']


adata.write(file_path)
print("Python evaluation preparation done! File updated.")


