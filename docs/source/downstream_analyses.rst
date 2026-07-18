Downstream Analyses
===================



We introduced a quantitative assessment framework to evaluate the biological interpretability of corrected representations, focusing on spatial domain resolution consistency and the fidelity of intercellular communication.

8.1 Spatial Correction Visualization Pipeline
----

To intuitively evaluate STAligner's correction performance on the integrated 12-slice DLPFC dataset, we established a comparative visualization framework encompassing UMAP dimensionality reduction, PAGA graph abstraction, and spatial distribution mapping. The pipeline was applied independently to both RAW and STAligner  datasets:

**①Data preprocessing and dimensionality reduction**: For the RAW dataset, we computed a neighborhood graph using use_rep="X_pca" with random_state=22, followed by UMAP embedding via sc.tl.umap (random_state=22). For the STAligner dataset, the identical procedure was applied using use_rep="STAligner" as the input representation, enabling direct comparison of low-dimensional structures before and after batch correction.

**②UMAP visualization**: UMAP embeddings were visualized using sc.pl.umap with two distinct color mappings: (a) Ground Truth annotations (celltype): layer-specific anatomical labels representing biological ground truth; (b) Clustering results (mclust): method-derived clustering outputs from the respective embedding spaces. These visualizations allowed assessment of both layer separation quality and batch mixing patterns.

**③ PAGA domain-level topology inference**: For each dataset, we performed PAGA (Partition-based Graph Abstraction) analysis via sc.tl.paga with groups="celltype". For RAW, the neighbor graph was constructed using use_rep="X_pca" with n_pcs=50. For STAligner, the neighbor graph was constructed using use_rep="STAligner". The resulting abstracted graphs were visualized using sc.pl.paga with nodes colored by celltype annotations, enabling characterization of the connectivity structure between cortical layers and assessment of whether spatial alignment improved the continuity of laminar organization.

**④Spatial distribution visualization**: Using sc.pl.spatial with spot_size=100, we projected both celltype and mclust annotations back onto the original tissue coordinates. This directly illustrated the spatial concordance between clustering assignments and anatomical annotations, validating whether STAligner enhanced layer-wise spatial consistency without disrupting the tissue architecture.

8.2 Marker Gene Heatmap Pipeline
----

We established three heatmap categories to validate clustering discrimination and anatomical alignment:

**① Reference-annotation heatmaps**: Using slice 151507, we utilized layer annotations as Ground Truth. Steps: (a) QC: remove invalid spots; (b) Normalization: sc.pp.normalize_total (target_sum=1e4) and log1p; (c) DE Analysis: Wilcoxon rank-sum test (padj < 0.05, log FC > 0.25); (d) Gene screening: extract Top 5 enriched markers per layer to establish a candidate gene pool. A subset of representative genes ['MALAT1', 'HPCAL1', 'NEFL', 'PCP4', 'TMSB10', 'SCGB2A2', 'MBP', 'PLP1', 'CRYAB'] was manually selected from these candidates for visualization, followed by row Z-score calculation prior to plotting.

**② STAligner clustering comparison heatmap**: Constructed on the identical integrated 12-slice dataset using STAligner batch-corrected clustering outputs. Clusters were ordered as ['5', '1', '4', '7', '6', '3', '2'] and annotated according to corresponding anatomical laminar regions, enabling direct side-by-side evaluation of layer separation after spatial alignment.

**③ RAW clustering comparison heatmap**: Generated on the integrated 12-slice dataset using uncorrected RAW clustering results. Clusters were arranged in predefined order ['1', '4', '6', '7', '2', '5', '3'] and mapped to anatomical layers (Layer_1~Layer_6/WM), to characterize laminar expression patterns without spatial correction.

8.3 Spatial Cellular Communication Analysis
----

To quantitatively evaluate STAligner's fidelity in preserving intercellular signaling patterns, we conducted space-constrained CellChat analysis across 8 DLPFC slices (151507, 151508, 151509, 151510, 151673, 151674, 151675, 151676), comparing signaling communication patterns among reference annotations, RAW clustering results, and STAligner-corrected clustering results.

8.3.1 Data Preparation and Preprocessing
~~~~

For each slice, we integrated three data sources to construct a unified analysis framework. The raw gene expression matrix for each slice was loaded from the 10x Genomics filtered feature-barcode matrix using sc.read_10x_h5(), with gene symbols deduplicated via var_names_make_unique(). For label extraction, the RAW annotations were subsetted from raw_adata_12.h5ad by matching batch labels, from which we extracted mclust and celltype annotations, while the STAligner annotations were subsetted from 12_staligner_DLPFC.h5ad by matching slice names, extracting mclust and Ground Truth annotations. Spots were then aligned by barcode intersection across the 10x expression matrix, RAW annotations, and STAligner annotations, retaining only common barcodes for downstream analysis. Spatial coordinates were extracted from raw_adata_12.h5ad with barcodes as index, containing imagecol and imagerow columns, and filtered to match the aligned spot set. For each slice, we generated a unified AnnData object containing the full gene expression matrix with three annotation columns, spatial coordinates CSV, three metadata CSV files for CellChat input, and Matrix Market format files including the count matrix, gene list, and spot list.

8.3.2 CellChat Analysis Pipeline
~~~~

We performed CellChat analysis (V2 spatial mode) independently on each slice and each annotation scheme (Ground Truth, RAW, STAligner) using a standardized pipeline implemented in R. Low-expressed genes were first filtered where fewer than 10% of spots expressed the gene, followed by normalization using Seurat's SCTransform with the glmGamPoi method for efficient variance stabilization, yielding normalized expression values. For spatial constraint configuration, coordinates were converted from pixels to micrometers using a conversion factor computed from the minimum non-zero pixel distance (conversion.factor = 100 / min.pixel.dist), with the spatial interaction range set to 2800 μm based on tissue architecture considerations, and a spatial.factors object was created with ratio = conversion.factor and tol = 65/2 for V2 spatial mode compatibility. CellChat objects were constructed using createCellChat() with the normalized expression as the object, metadata with group annotations, the group column specification, spatial datatype, spatial coordinates matrix, and the spatial conversion factors. Using the human CellChat database (CellChatDB.human) containing curated ligand-receptor pairs, the pipeline proceeded through subsetting data to genes expressed in at least 2 cells, identifying overexpressed genes, identifying overexpressed ligand-receptor interactions, and computing communication probabilities via computeCommunProb() with the triMean method, raw expression usage, spatial distance weighting enabled (distance.use = TRUE), interaction range of 2800 μm, and distance scaling factor of 0.01. Following probability computation, communications were filtered, pathway activities were inferred via computeCommunProbPathway(), and networks were aggregated at the group level. To validate spatial constraints, we checked for the presence of the distance column in the communication results, and when spatial constraints did not fully apply, manual secondary filtering was performed by computing group center distances and retaining only interactions within the 2800 μm range. For each analysis, we saved the CellChat object as an RDS file, ligand-receptor pair communication tables, and network count matrices.

8.3.3 Communication Network Visualization
~~~~

For each slice and annotation scheme, we generated ordered heatmaps to visualize inter-group communication patterns using the network count matrices. For Ground Truth heatmaps, groups were ordered as ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6', 'WM'] to reflect anatomical layering from superficial to deep cortex. For RAW heatmaps, clusters were ordered as ['RAW_C0', 'RAW_C3', 'RAW_C5', 'RAW_C6', 'RAW_C1', 'RAW_C4', 'RAW_C2'] and annotated with corresponding anatomical layers (Layer_1, Layer_2, Layer_1/Layer_4/Layer_5, Layer_4, Layer_5, Layer_1/WM, WM) based on cluster-to-layer mapping. For STAligner heatmaps, clusters were ordered as ['STA_C5', 'STA_C1', 'STA_C4', 'STA_C7', 'STA_C6', 'STA_C3', 'STA_C2'] and annotated with anatomical layers (Layer_1, Layer_2, Layer_1/Layer_2, Layer_4, Layer_5, Layer_6/WM, WM) based on cluster-to-layer mapping. 

8.3.4 Quantitative Comparison of Communication Preservation
~~~~

To systematically compare communication pattern preservation between RAW and STAligner, we computed four quantitative metrics for each slice. The LR-pair retention rate was defined as the proportion of ligand-receptor pairs detected in the Ground Truth analysis that were also detected in the target analysis (RAW or STAligner). Similarly, the pathway retention rate was defined as the proportion of signaling pathways detected in the Ground Truth analysis that were also detected in the target analysis. Additionally, we computed the total communication weight as the sum of all communication probabilities across all group pairs, representing the overall signaling intensity, and the total communication count as the number of significant ligand-receptor interactions detected across all group pairs. Paired Wilcoxon signed-rank tests were performed to compare RAW and STAligner retention rates across the 8 slices, accounting for the paired nature of the data where each slice serves as its own control. For visualization, we generated paired line plots for both LR-pair and pathway retention rates with individual slices connected by lines colored uniquely by slice. We also generated a variance bar plot showing the variance of retention rates across slices for four conditions (LR_RAW, LR_STAligner, Pathway_RAW, Pathway_STAligner), with variance values labeled on top of each bar.

