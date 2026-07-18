Controlled simulations
========


To comprehensively and objectively evaluate the robustness of spatial transcriptomics alignment algorithms under diverse technical noise and biological variations, we employed two advanced statistical simulation frameworks: SRTsim and scDesign3. SRTsim excels at generating high-fidelity domain-specific expression patterns via generative modeling, whereas scDesign3 uses flexible statistical models (e.g., Copula and smoothing splines) to capture spatial covariates and complex gene co-expression networks. This orthogonal combination generates multi-gradient benchmarking datasets with rigorous theoretical foundations and high biological authenticity. The simulation inputs, parameter settings, and downstream data utilization are detailed below.

6.1 Inter-Slice Simulations
----

We utilized real tissue slices from DLPFC (151507), HBC (section1), and Xenium Breast (Rep1) as baseline references to perform multi-gradient cross-slice simulations.

6.1.1 Raw Data Preprocessing
~~~~

DLPFC and HBC datasets: Standard filtered expression matrices were loaded, gene names deduplicated, and matched with true cell-type labels. Invalid annotated spots were discarded, outputting full-slice "cell × gene" feature matrices.

Xenium Breast dataset: Owing to the large number of cells in the original Xenium dataset, 
only Rep1 was used for benchmarking to reduce computational and memory overhead. 
Cell-type annotations were first matched to the Xenium cell barcodes, and cells without valid annotations were excluded. 
To further reduce memory consumption while preserving the native spatial organization, 
a deterministic spatial cropping strategy was applied based on the global coordinate distribution. 
Specifically, the median values of the x- and y-centroid coordinates were calculated across all annotated cells, 
and only cells located in the upper-right quadrant `(x > median(x) and y > median(y))` were retained, resulting in approximately 25% of the original annotated cells. The filtered expression matrix was then exported in a gene × cell format together with the corresponding spatial coordinates and cell-type labels for downstream analysis..

6.1.2 SRTsim-based Gradient Simulation
~~~~

Top 5000 highly expressed genes were retained for Visium platforms, and all non-zero targeted genes for Xenium. 
Incorporating 2D spatial coordinates and cell labels, spatial hierarchical expression models were fitted using the domain-based simulation scheme `(sim_schem = "domain")`, 
where ground-truth tissue labels were used to preserve spatial domain-specific transcriptional patterns. 
Model fitting and simulation were performed with fixed random seeds to ensure reproducibility.

**Parameters and Output**: A baseline slice without technical bias (capture efficiency = 1.0) was first generated from the fitted SRTsim model. 
Global binomial downsampling was then applied to every count value in the simulated expression matrix to create a decreasing capture efficiency gradient {1.0, 0.9, 0.7, 0.5, 0.3, 0.1}, mimicking technical degradation from perfect capture to extreme sparsity. During this process, spatial coordinates, cell identities, and biological labels remained unchanged, ensuring that only technical variation was introduced while preserving the underlying biological structure.

**Model Input Pairing**: The simulated slice at gradient 1.0 was set as the "reference slice" and paired successively with the degraded slices (0.9, 0.7, 0.5, 0.3, 0.1), 
forming 5 pairs of "consecutive slices" with varying technical noise gaps for downstream alignment testing. Each slice pair therefore shared identical spatial organization and biological composition, differing only in the degree of simulated technical degradation.

6.1.3 scDesign3-based Gradient Simulation
~~~~

Following identical gene filtering, a SingleCellExperiment object was constructed. The fit_marginal function, combined with thin-plate splines, modeled non-linear associations of gene expression means using the formula s(spatial_x,spatial_y,bs="ts") + ground_truth (using Negative Binomial for Visium, Poisson for Xenium), while Gaussian Copula captured the gene covariance structure. The fitted model therefore jointly preserved spatial smoothness, tissue-domain labels, and gene–gene dependency. Model fitting and simulation were performed with fixed random seeds to ensure reproducibility.

**Parameters and Output**: A decreasing mean shift gradient {0, -0.1, -0.3, -0.5, -0.7, -0.9} was introduced by perturbing the intercept coefficient of each gene-specific marginal model with Gaussian random noise (standard deviation = 0.1), simulating non-uniform capture efficiency decay while preserving spatial structures. The modified marginal parameters were subsequently combined with the original Gaussian Copula to generate new count matrices, thereby maintaining the learned gene correlation structure across all simulated slices. Six slices were generated, with spatial coordinates, biological labels, and cell identities kept unchanged, while actively clearing memory between iterations.

**Model Input Pairing**: The baseline slice (shift 0) was paired with the remaining degraded slices (-0.1 to -0.9), forming 5 sets of simulated consecutive slices. Each slice pair therefore differed only in the degree of simulated technical degradation while sharing identical spatial organization and biological composition.

6.2 Inter-Sample Simulations
----

This section constructs cross-sample integration scenarios using dual-factor perturbations (localized biological lesions and global technical batch noise) to evaluate algorithmic robustness in eliminating technical bias while preserving true biological variation. The modeling baseline reuses the single-slice datasets generated in Section 6.1: DLPFC (151507), HBC (section1), and Xenium Breast (Rep1).

6.2.1 Dual-Factor Perturbation via SRTsim
~~~~

Feature Extraction and Baseline Generation: Visium datasets were filtered for the Top 5000 genes, while Xenium data retained the Top 3000 highly abundant genes. Spatial coordinates and cell types were input into SRTsim to fit a spatial hierarchical expression model using the domain-based simulation scheme (sim_schem = "domain"), where ground-truth tissue labels were incorporated to preserve spatial domain-specific expression patterns. Model fitting and simulation were performed with fixed random seeds to ensure reproducibility..The fitted model generated the pure, technical-noise-free reference Slice A. 

Biological Variation Injection: The spatial domain with the highest cellular abundance in the baseline slice was designated as the "targeted lesion region." For the biological perturbation gradient, no perturbation was applied at FC 1.0, whereas 50, 100, and 300 highly expressed genes were perturbed with FC 1.5, 2.0, and 3.0, respectively. The selected genes were modified only within the targeted lesion region, while all remaining genes and spatial domains were left unchanged, generating variant Slice B. This design introduces localized biological variation while preserving the global tissue architecture and spatial organization.

.. code-block:: python
    :linenos:

    # Core code for localized biological lesion injection
    if (current_fc > 1.0) {
    current_counts_B[target_gene_names, target_cell_names] <- 
        round(current_counts_B[target_gene_names, target_cell_names] * current_fc)
    }

**Sequencing Batch Decay**: Capture rate gradients {1.0, 0.7, 0.3, 0.1} were enforced via global binomial downsampling to generate heterogeneous matrices featuring decreased sequencing depth and increased Dropouts. 
Binomial sampling was applied to every count value across the entire expression matrix while leaving spatial coordinates, cell identities, and biological perturbations unchanged. Consequently, Slice B simultaneously contained localized biological differences and controllable global technical batch effects. The two perturbation factors were combined in a full factorial design, producing 16 (4×4) simulated inter-sample scenarios covering all combinations of biological and technical perturbation levels.

.. code-block:: python
    :linenos:

    # Core code for technical batch decay (Binomial downsampling)
    if (current_eff < 1.0) {
        sampled_vec <- rbinom(n = length(current_counts_B), 
                                size = as.numeric(current_counts_B), 
                                prob = current_eff)
        current_counts_B <- matrix(sampled_vec, nrow = nrow(current_counts_B), 
                                    ncol = ncol(current_counts_B))
    }


6.2.2 Dual-Factor Perturbation via scDesign3
~~~~

**Baseline Modeling**: The fit_marginal function was utilized to fit gene marginal distributions, employing thin-plate smoothing spline functions to capture spatial coordinate trends. Specifically, the marginal model was fitted using the formula ground_truth + s(spatial_x, spatial_y, bs = "ts"), jointly modeling tissue-domain effects and nonlinear spatial variation. The model family was specified as negative binomial (nb), and a Gaussian Copula network extracted baseline parameters (mean_mat, sigma_mat) to generate Slice A. The fitted Gaussian Copula was subsequently retained throughout all simulations to preserve the baseline gene–gene dependency structure. Model fitting and simulation were performed with fixed random seeds to ensure reproducibility.

.. code-block:: python
    :linenos:

    # Core code for marginal distribution fitting
    ST_marginal <- fit_marginal(
        data = ST_data, predictor = "gene",
        mu_formula = "ground_truth + s(spatial_x, spatial_y, bs = 'ts')",
        sigma_formula = "1", family_use = "nb", usebam = TRUE
    )

**Dual-Factor Variant Slice B Generation**: Biological variation was achieved by multiplying the targeted intersection regions on the underlying mean matrix (mean_mat) by the FC multipliers {1.0, 1.5, 2.0, 3.0}. Only the selected genes within the targeted spatial region were modified, whereas all remaining genes and spatial domains retained their original marginal parameters. Technical batch decay was simulated by setting log-mean decay gradients {0, -0.1, -0.5, -0.9}. For groups experiencing decay, a Gaussian random multiplier following N(0, 0.1²) was superimposed to simulate non-uniform global decay. Specifically, each cell was assigned an independent multiplicative scaling factor, introducing heterogeneous technical degradation across the entire sample while preserving the underlying spatial coordinates, cell identities, and Gaussian Copula-derived gene correlation structure. The biological and technical perturbation factors were combined in a full factorial design, generating 16 (4 × 4) simulated inter-sample scenarios spanning all combinations of perturbation strengths.

.. code-block:: python
    :linenos:

    # Core code for heterogeneous batch noise injection
    if (current_cr < 1.0 || current_sd > 0.0) {
        # current_cr represents the intercept decay; current_sd = 0.1
        safe_multipliers <- exp(rnorm(ncol(current_mean_mat), mean = 0, sd = current_sd))
        scale_factors <- safe_multipliers * current_cr
        
        for (g in seq_len(ncol(current_mean_mat))) {
            current_mean_mat[, g] <- current_mean_mat[, g] * scale_factors[g]
        }
    }

6.2.3 Downstream Evaluation Pairing Strategy (Model Inputs)
~~~~

To objectively quantify model performance, S00 was used as the common baseline reference, and six representative paired comparisons were selected for downstream algorithms for each reference dataset and simulation framework:

**S01 batch-only settings**: Slice A paired with Slice B containing technical perturbation only, including a moderate technical perturbation and a strong technical perturbation. For SRTsim, these corresponded to capture efficiencies of 0.7 and 0.1. For scDesign3, these corresponded to log-mean decay gradients of -0.1 and -0.9.

**S10 biology-only settings:** Slice A paired with Slice B containing biological perturbation only, including 50 genes with FC 1.5, 100 genes with FC 2.0, and 300 genes with FC 3.0, while no additional technical perturbation was introduced.

**S11 combined setting**: Slice A paired with Slice B containing the strongest biological perturbation and the strongest technical perturbation. This corresponded to 300 genes with FC 3.0 plus capture efficiency 0.1 for SRTsim, and 300 genes with FC 3.0 plus log-mean decay gradient -0.9 for scDesign3.

Together, these six paired comparisons represented batch-only, biology-only, and combined batch-plus-biological states, enabling evaluation of both technical batch removal and preservation of predefined biological differences.

6.3 Cross-Platform Simulations
----

We constructed simulated datasets covering controlled cross-resolution simulations using ovarian cancer spatch_ov Xenium data.

6.3.1 Spatch_OV Cross-Resolution Dataset Construction
~~~~

To accurately simulate the dual challenges of "resolution discrepancy" and "sequencing capture heterogeneity" encountered when integrating different spatial omics technologies (e.g., Visium-like pseudo-spot representation and subcellular-resolution Xenium), this section details the generation pipeline for the Spatch_OV cross-resolution simulation dataset.

**Data Preprocessing and Physical Cropping**

This pipeline used the ovarian cancer spatch_ov Xenium data as the high-resolution cell-level reference for simulation.

First, Xenium (.h5ad) and Visium HD data were loaded, utilizing var_names_make_unique to ensure gene name singularity. Subsequently, a dimensional reduction on the gene axis was executed: by calculating the intersection, only the targeted genes shared by both platforms were retained, reducing the matrix from tens of thousands of genes to the shared subset (e.g., ~300 genes).

To prevent Memory Out-of-Bound errors during single-cell level simulations, a spatial cropping strategy based on physical coordinates was introduced. Targeting the spatial coordinates (x, y) of the Xenium data, a hard threshold truncation was applied using the 75th percentile. This retained only the high-density valid cellular regions in the "upper-right quadrant" of the original tissue section. While perfectly preserving the tissue's spatial topology, this operation scaled down the observational volume to approximately 6.25% of the raw data. The cropped count matrices and their corresponding spatial metadata were exported independently as CSV files to feed into the downstream simulation engines.

**SRTsim-based Spatial Binning and Gradient Simulation**

**Data Aggregation (Pseudo-Visium Generation)**: The filtered Xenium single-cell matrix was read. To bridge the physical scale gap, a grid size (bin_size = 50) was defined, and the high-resolution single-cell coordinates were forcibly grid-mapped using a floor transformation. Sparse matrix multiplication (sparseMatrix) was then leveraged to rapidly aggregate the single-cell counts within the same grid, generating low-resolution "Pseudo-Visium" Spots. Edge spots containing fewer than 3 cells were discarded, and a Majority Voting strategy was applied to assign a representative cell type (ground_truth) to each aggregated Spot. The spatial coordinates of each pseudo-spot were defined as the mean x- and y-coordinates of all constituent cells, thereby preserving the global tissue geometry after aggregation.

**Model Fitting and Baseline Generation**: The Pseudo-Visium matrix and the merged spatial coordinates were fed into SRTsim. Depending on the abundance of the aggregated spots, either domain or tissue mode was adaptively invoked to fit the spatial expression distribution. Specifically, the domain mode was attempted first, while the tissue mode was automatically adopted if the aggregated spot number was insufficient for stable domain-level fitting. Model fitting was performed with fixed random seeds to ensure reproducibility. The slice generated at this stage is denoted as P0 (P0_NoBatch), representing a pure baseline with "physical scale differences" but devoid of "technical batch noise." The original high-resolution Xenium dataset was simultaneously retained as the cell-level reference for downstream cross-resolution alignment.

**Streamlined Batch Injection**: Building upon the P0 slice, three continuous global capture efficiency decay gradients were established: 1.0 (P0_NoBatch), 0.7 (P1_WeakBatch), 0.3 (P2_MediumBatch), and 0.1 (P3_StrongBatch). For each gradient, the rbinom function was called to execute binomial downsampling, precisely mimicking the Dropout effect caused by a plummet in sequencing depth. During this process, the aggregated spot coordinates and majority-voted cell identities remained unchanged, ensuring that only technical batch effects were introduced after spatial aggregation. The matrices generated for each group were exported alongside the original high-resolution Xenium data, which served as the cell-level reference.

6.3.2 scDesign3-based Cross-Scale Modeling and Gradient Simulation
~~~~

**Modeling and Marginal Distribution Fitting**: The identical size-50 Pseudo-Visium spatial binning logic was reused to construct the SingleCellExperiment object. Pseudo-spots containing fewer than three cells were excluded, the spatial coordinates of each pseudo-spot were defined as the mean coordinates of all constituent cells, and representative cell identities were assigned by majority voting. When fitting the gene marginal distributions (fit_marginal), thin-plate smoothing spline terms of spatial coordinates (s(spatial_x, spatial_y, bs='ts')) and cell-type covariates were rigorously incorporated. The Negative Binomial (nb) family was selected, and a Gaussian Copula accurately captured inter-gene co-expression correlations. Model fitting was performed using one computational core (n_cores = 1) with fixed random seeds to ensure reproducibility. The base parameters extracted at this point (mean_mat, sigma_mat, etc.) generated the P0 (P0_NoBatch) noise-free baseline group, functionally equivalent to SRTsim's. The original high-resolution Xenium dataset was simultaneously retained as the cell-level reference for downstream cross-resolution alignment.

**Mean Shift and Heterogeneous Noise Injection**: Complex noise was injected into the baseline mean matrix (mean_mat). Four equivalent decay gradients (CR) were set: exp(0), exp(-0.1), exp(-0.5), and exp(-0.9), corresponding to capture rates of 1.00, ~0.90, ~0.61, and ~0.41, respectively. For the non-baseline groups (P1~P3), not only was a CR scalar decay executed, but random normal noise (log-normal scale multiplier) controlled by a standard deviation SD = 0.1 was also superimposed. This simulated the gene-specific, non-uniform decay (rather than global uniform downsampling) encountered in real cross-platform sequencing. To ensure numerical stability during simulation, extremely small or invalid mean estimates were truncated to a minimum positive value before data generation. The modified marginal parameters, together with the fitted Gaussian Copula and extracted baseline covariance structure, were subsequently used by simu_new to generate the final simulated count matrices while preserving the underlying spatial organization and gene dependency structure. The resulting four sets of heterogeneous expression profiles were exported independently, whereas the spatial coordinates and spot annotations remained identical across all simulated groups.
