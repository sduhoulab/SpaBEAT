Dataset Preprocessing
====================



To standardize the benchmarking process, we established a baseline data processing and format-conversion workflow. The raw gene expression matrices (.h5) and spatial metadata (tissue_position_list.csv) served as the base inputs. These were initially loaded into the AnnData format using the scanpy package (e.g., sc.read_visium), which supports the simultaneous integration of expression matrices, spatial coordinates, and H&E imagery. Unique gene identifiers were ensured via adata.var_names_make_unique().

To accommodate the idiosyncratic input requirements of various benchmarking methods, we implemented the following customized conversions:

PRECAST: Slices were preserved and converted into R-native .rds files.

STitch3D: Supplemented with external or matched single-cell reference datasets. These reference data were handled in accordance with the specific input formats required by the model: either as self-contained .h5 (.h5ad) files or as a structured suite of expression matrices (.mtx), barcode metadata (.csv), and mapping files (.tsv). 

SPIRAL: Generated four core files: 

  ①*_features.txt/.csv (gene expression matrix); 

  ② *_edge_KNN_*.csv (K-nearest neighbor spatial topological graph based on pixel coordinates); 

  ③ *_label.txt/.csv (regional or cell-type annotations); 

  ④ *_positions.txt/.csv (raw pixel coordinates).

Cell-type or region-specific annotations were curated and mapped according to the unique format and schema of each individual dataset. To ensure spatial integrity, we performed a rigorous alignment process using barcodes and grid coordinates as identifiers. Expression matrices, spatial coordinates, H&E images, and reference annotations were intersected to prune the datasets, effectively filtering out invalid spots and ensuring downstream consistency across all methods.

All spatial transcriptomics datasets underwent a unified base preprocessing workflow before algorithm-specific customization, including total-count normalization, log1p transformation, and general quality control filtering of low-quality cells/genes. Unless explicitly stated in each method subsection, the top 5,000 highly variable genes (HVGs) identified via the vst strategy were retained for downstream integration. 

High-resolution or cell-level datasets with large numbers of observations, including HD_crc, Xenium breast, spatch_ov, and spatch_hcc, were downsampled where necessary for computational feasibility. For datasets exceeding 15,000 cells or spots in method-compatible input form, stratified subsampling was performed to obtain approximately 15,000 observations. Sampling was stratified by reference annotation labels, so that each cell-type or spatial-domain group was sampled at a unified proportional ratio and major annotation categories were proportionally retained. A fixed random seed of 50 was used to ensure reproducibility. The resulting downsampled matrices, spatial coordinates, metadata, and reference annotations were then exported into the input formats required by each integration method.

