Robustness Analyses

=================


To systematically evaluate the performance stability of spatial transcriptomic integration algorithms under varying preprocessing strategies, input data characteristics, and technical parameter shifts, this study conducted comprehensive robustness validation experiments across three core dimensions: preprocessing sensitivity, targeted-gene overlap, and cell segmentation. All experiments adhered strictly to the control variable method—fixing core algorithm hyperparameters and downstream evaluation metrics while isolating the target validation variable. Detailed experimental procedures and quantitative metric results are documented in the corresponding supplementary output files.

7.1 Robustness to Normalization
----

To assess the sensitivity of integration algorithms to different standardization and preprocessing protocols, we designed a preprocessing sensitivity framework focusing on normalization strategy and the number of retained highly variable genes (HVGs).

Testing Baseline: We selected two representative algorithms that allow configurable preprocessing options (GraphST and STAligner) for focused evaluation, using unaligned RAW results as the baseline reference. The testing datasets included DLPFC Sample 1, DLPFC Sample 2, DLPFC Sample 3, DLPFC_7374, DLPFC_all, and HBC.

While strictly controlling algorithm hyperparameters, we implemented two normalization strategies, LogNorm and SCTransform, orthogonally combined with three HVG-retention settings using the top 1,000, 3,000, and 5,000 highly variable genes. This generated six preprocessing configurations per method: 

LogNorm: Total-count normalization was first applied to scale each cell/spot to a fixed library size, followed by log1p transformation. This strategy was used to reduce library-size effects and stabilize the global variance structure of gene expression.

SCTransform: SCTransform was used as an alternative normalization strategy based on regularized negative-binomial regression, providing a variance-stabilizing transformation that accounts for sequencing-depth-related technical variation.

HVG Dimensionality Control: The two normalization strategies were paired with feature-selection schemes retaining 1,000, 3,000, and 5,000 HVGs, respectively. By systematically comparing the fluctuation amplitudes of clustering metrics (e.g., ARI), biological-preservation metrics, and batch-removal metrics across these schemes, we quantitatively evaluated the robustness boundaries of GraphST and STAligner to upstream normalization strategies and feature-number choices.

7.2 Robustness to Genes Overlap
----

Addressing the common issue of partially overlapping targeted-gene panels in multi-sample integration, we designed a "Cross-Masking" experiment on the Xenium in situ single-cell breast cancer (Xenium breast) consecutive slice dataset to evaluate the spatial network recovery capability of algorithms when facing feature dimension deficiencies.

Experimental Setup and Masking Strategy: We selected representative methods with contrasting performance patterns, including DeepST and spatiAlign, together with the RAW baseline for comparison. The two original consecutive Xenium slices share precisely 300 targeted genes. We artificially constructed 4 decreasing overlap gradients: full overlap (300 genes), removal of 50 genes, removal of 100 genes, and removal of 200 genes.

To circumvent batch shifts caused by single-slice masking, we adopted a rigorous symmetric cross-masking logic. Taking the "removal of 50 genes" as an example: For Slice 1, the first 25 genes were randomly masked, exposing the remaining 275 genes to the model. Concurrently, for Slice 2, the last 25 genes were masked, exposing a different subset of 275 genes. When the underlying integration algorithm executed feature alignment and automatically extracted the intersection, the 25 unique genes of Slice 1 and the 25 unique genes of Slice 2 were mutually excluded, ensuring the actual intersection dimension explicitly participating in spatial graph network construction and dimensional reduction alignment was strictly controlled at 250 genes.

We evaluated the degradation patterns of RAW, DeepST, and spatiAlign across the four overlap gradients, thereby revealing the direct impact of feature dimension loss on spatial domain alignment performance.

7.3 Robustness to Cell Segmentation
----

Downstream analysis of single-cell spatial transcriptomics (e.g., Xenium) heavily relies on the accuracy of upstream image-based cell segmentation. To quantify the segmentation-associated batch-like variation induced by diverse segmentation algorithms and its subsequent impact on integration models, we conducted a rigorous methodological comparison on a human breast cancer Xenium section.

7.3.1 Segmentation Matrix Construction and Cell Alignment
~~~~

Official Segmentation Matrix (Matrix A): Utilizing Xenium Explorer 4, we delineated a target tissue region (3,873,426.5 µm²) containing 27,713 cells. Expression profiles, unique Cell IDs, and spatial coordinates (x,y) were directly extracted from the official cell_feature_matrix.h5 to form the baseline Matrix A.

  

Cellpose Segmentation Matrix (Matrix B): Leveraging the accompanying morphology.ome.tif tissue staining image, original physical spatial coordinates of cells were mapped to the pixel level and imported into the U-Net-based Cellpose single-cell image recognition pipeline for re-segmentation. To accommodate the massive cell volume, Cellpose generated a 16-bit mask image to guarantee unique Cell ID capacity. Subsequently, based on geometric spatial overlap, the masks redrawn by Cellpose were strictly mapped back to the official initial Cell IDs. This accomplished ID alignment of heterogeneous segmentation schemes at the single-cell level, yielding reconstructed Matrix B.

 



7.3.2 Unified Gold Standard Annotation
~~~~

Because shifts in segmentation boundaries directly trigger alterations in transcript assignments, native expression feature disparities inherently exist between Matrix A and B. To ensure fair evaluation, prior to merging the matrices, we executed a unified single-cell clustering re-annotation on the consolidated data domain using the Seurat standard pipeline, anchored by default spatial coordinates and official cell types. This served as the unified Gold Standard for downstream ARI and cLISI calculations.

7.3.3 Segmentation Batch Effect Integration Evaluation Pipeline
~~~~

Within the unaligned RAW fused matrix (direct concatenation of A and B), massive batch clustering phenomena induced solely by segmentation methodological differences could be clearly observed via UMAP dimensionality reduction. For a comprehensive systemic evaluation, this dual-matrix dataset was fed into 10 spatial integration algorithms and validated according to the following core principles:

Segmentation Baseline Feature Deviation: We first statistically profiled the baseline physical deviations between the official pipeline and the Cellpose pipeline concerning cell counts, transcripts per cell, genes per cell, cell geometric area, and spatial cell-type proportions. This elucidates the biological and geometric origins of the integration difficulty.

Batch Removal Capability: Treating "Official" and "Cellpose" as batch labels, we calculated and compared the Batch ASW, iLISI, kBET, and Graph Connectivity of each integration algorithm.

Biological Feature Preservation: Anchored by the unified gold standard annotation, we calculated the Cell-type/Domain ASW, cLISI, ARI, SCS scores, and marker gene Moran’s I indices for each algorithm. This evaluated whether the algorithms could safeguard the underlying authentic cytological clustering boundaries while forcefully pulling together the heterogeneous segmentation matrices.

Over-correction Risk: By plotting a comprehensive evaluation scatterplot, we determined whether an algorithm showed a potential over-correction pattern—meaning that it successfully integrated the official and Cellpose-derived data, but compromised the integrity of reference-defined tumor, stromal, or immune microenvironment structures as a trade-off.

