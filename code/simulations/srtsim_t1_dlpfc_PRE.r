suppressPackageStartupMessages(library(Seurat))
options(Seurat.object.assay.version = "v3")

suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(PRECAST))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(zellkonverter))

get_memory_usage <- function() {
    mem_info <- gc(verbose = FALSE)
    used_memory <- sum(mem_info[, 2]) 
    return(used_memory * 8 / 1024^2)  
}

file_fold <- "/root/ST0507/simulations_model/data_model/1_DLPFC_InterSlice/SRTsim_Converted/" 
out_dir <- "/root/ST0507/new_model/PRECAST/results/SRTsim_dlpfc_t1"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

ref_slice <- "Sim_Slice_1"
n_domains <- 7
for (i in 2:6) {
    target_slice <- paste0("Sim_Slice_", i)
    datasets <- c(ref_slice, target_slice)
    pair_name <- paste0(ref_slice, "_vs_", target_slice)
    
    cat("\n=======================================================\n")
    cat("🚀 Starting PRECAST consecutive slice alignment task:", pair_name, "\n")
    cat("=======================================================\n")
    
    seuList <- list()
    total_cells <- 0
    total_genes <- 0

    cat("Loading h5ad data for", pair_name, "...\n")

    for (j in 1:length(datasets)) {
        dataset <- datasets[j]
        file_path <- file.path(file_fold, dataset, paste0(dataset, ".h5ad"))
        cat("  Loading", dataset, "from", file_path, "\n")
        
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
        if (j == 1) total_genes <- nrow(seu)
        
        seuList[[j]] <- seu
    }

    cat("Preprocessing datasets...\n")
    seuList <- lapply(seuList, function(x) {
        x[["RNA"]] <- as(object = x[["RNA"]], Class = "Assay")
        
        x <- NormalizeData(x)
        x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
        x <- ScaleData(x) 
        
        return(x)
    })

    memory_before <- get_memory_usage()
    start_time <- Sys.time()

    cat("Creating PRECAST Object...\n")
    PRECASTObj <- CreatePRECASTObject(seuList, project = pair_name, gene.number = 2000, 
                                      selectGenesMethod = "SPARK-X", premin.spots = 20, 
                                      premin.features = 20, postmin.spots = 1, postmin.features = 10)

    cat("Adding Adjacency List...\n")
    PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium")

    cat("Setting Parameters...\n")
    PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)
    PRECASTObj <- PRECAST(PRECASTObj, K = n_domains)

    end_time <- Sys.time()
    training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
    memory_after <- get_memory_usage()
    memory_used <- memory_after - memory_before

    cat("Training completed! Merging and exporting results...\n")

    resList <- PRECASTObj@resList
    PRECASTObj <- SelectModel(PRECASTObj)
    seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")
    
    saveRDS(PRECASTObj, file = paste0(out_dir, "PRECASTObj_", pair_name, ".rds"))
    saveRDS(seuInt, file = paste0(out_dir, "seuInt_", pair_name, ".rds"))

    final_cells <- sum(unlist(lapply(PRECASTObj@seuList, function(x) ncol(x))))
    final_genes <- nrow(PRECASTObj@seuList[[1]])
    embedding_dim <- ncol(PRECASTObj@resList$hZ[[1]])

    benchmark_results <- list(
        method_name = "PRECAST",
        dataset = pair_name,
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

    write_json(benchmark_results, paste0(out_dir, "precast_benchmark_", pair_name, ".json"), pretty = TRUE, auto_unbox = TRUE)

    cat("Loading raw data directly from H5ADs for export...\n")
    raw_obj_list <- list()

    for (dataset in datasets) {
        file_path <- file.path(file_fold, dataset, paste0(dataset, ".h5ad"))
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

    raw_combined <- merge(raw_obj_list[[1]], raw_obj_list[-1])
    raw_counts <- GetAssayData(raw_combined, assay = "RNA", layer = "counts")

    colnames(raw_counts) <- gsub("_", "", colnames(raw_counts))
    common_cells <- intersect(colnames(raw_counts), colnames(seuInt))

    raw_counts <- raw_counts[, common_cells]
    seuInt <- subset(seuInt, cells = common_cells)
    raw_counts <- raw_counts[, colnames(seuInt)]

    sce_int <- SingleCellExperiment(
        assays = list(counts = raw_counts),
        colData = colData(as.SingleCellExperiment(seuInt))
    )

    colnames(raw_counts) <- sub("_(\\d+)$", "\\1", colnames(raw_counts))
    rownames(raw_combined@meta.data) <- sub("_(\\d+)$", "\\1", rownames(raw_combined@meta.data))

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
        spatial_coords <- as.matrix(cbind(colData(sce_int)$col, colData(sce_int)$row))
    } else {
        meta <- raw_combined@meta.data[colnames(sce_int), , drop = FALSE]
        spatial_coords <- as.matrix(cbind(meta$col, meta$row))
    }

    rownames(spatial_coords) <- colnames(sce_int)
    reducedDim(sce_int, "spatial") <- spatial_coords

    if ("ground_truth" %in% colnames(raw_combined@meta.data)) {
        gt <- raw_combined@meta.data[colnames(sce_int),"ground_truth"]
        colData(sce_int)$ground_truth <- gt
    }

    if (!is.null(PRECASTObj@resList$ident)) {
        ident_vec <- unlist(PRECASTObj@resList$ident)
        names(ident_vec) <- unlist(lapply(PRECASTObj@seuList, colnames))
        names(ident_vec) <- gsub("_", "", names(ident_vec))
        ident_vec <- ident_vec[colnames(sce_int)]
        colData(sce_int)$precast_cluster <- factor(ident_vec)
    }

    if ("orig.ident" %in% colnames(colData(sce_int))) {
        colData(sce_int)$slice_id <- colData(sce_int)$orig.ident
    }

    cat("Saving Final H5AD...\n")
    if (!inherits(assay(sce_int, "counts"), "CsparseMatrix")) {
        assay(sce_int, "counts") <- as(assay(sce_int, "counts"), "CsparseMatrix")
    }
    
    final_h5ad_path <- paste0(out_dir, "multiple_adata_", pair_name, "_precast.h5ad")
    zellkonverter::writeH5AD(sce_int, final_h5ad_path)
    
    cat("✅ Task", pair_name, "finished and exported successfully.\n")
    
    rm(PRECASTObj, seuInt, raw_obj_list, raw_combined, sce_int, sce, seuList)
    gc()
}
cat("\n🎯 All PRECAST slice pair integration tasks have been completed!\n")