Ranking and Coverage
====================


This supplementary section elaborates the full technical implementation details of the hierarchical rank-based benchmarking framework, including standardized metric preprocessing, multi-stage aggregation logic and score normalization. The RAW uncorrected baseline was retained solely for reference visualization and fully excluded from all ranking, aggregation and composite score calculations throughout the pipeline.

4.1 Metric Standardization Preprocessing
----

Raw evaluation metrics exhibit heterogeneous scales and opposing optimal directions (higher-is-better vs. lower-is-better). A dataset-wise standardization pipeline was applied prior to ranking, with a dedicated transformation for the SCS metric:

**General Min–Max scaling function**: A reusable `min_max_scale()` function normalizes each metric to the range [0,1]. 
For metrics where lower values indicate superior performance, scaled outputs are inverted via 1-scaled value. 
Notably, while most metrics (e.g., *ARI*, *NMI*, *ASW*, and *kBET*) were retrieved directly from the scib framework in pre-normalized formats, 
the Spatial Coherence Score (SCS) was manually calculated and subsequently underwent this `min_max_scale()` normalization to align with the global benchmarking scale.
In contrast, Moran’s I and Geary’s C statistics were calculated independently from raw representations and were excluded from this normalization procedure
to preserve their original statistical significance; these were only subject to transformation (e.g., Geary’s C inversion) 
during visualization to ensure consistent directional interpretation.

**SCS transformation**: SCS requires a non-linear pre-transformation step before Min–Max normalization, performed independently per dataset_name:

*Compute group-wise scale* factor: scale = median(|SCS|); set :math:`scale=10^{-9}` if the median equals zero.

*Exponential mapping*: SCStrans = 1 - exp(SCS/scale).

Linearly rescale SCStrans to [0,1] with min_max_scale(higher_is_better=True).

4.2 Per-Dataset Rank Calculation (Task-Level Ranking)
----

All ranking operations were executed stratified by Scenario and dataset_name, following the hierarchical rank logic outlined in the main manuscript. Metrics were partitioned into two functional groups with predefined optimization directions: Batch-effect removal metrics and
Biological signal preservation metrics.

4.3 Category-averaged ranks and weighted overall rank
----

Rank each single metric individually using rank (method='min') aligned to its optimization direction within each independent dataset. Compute mean ranks across all batch metrics (Dataset_Batch_mean) and biological metrics (Dataset_Bio_mean), then re-rank these means to obtain integer (Dataset_Batch_Rank) and (Dataset_Bio_Rank) (smaller rank = better performance).

Compute task-level weighted overall rank following Equation (13):

Dataset_Weighted_Overall_Rank = 0.6 × Dataset_Batch_Rank + 0.4 × Dataset_Bio_Rank

Batch-effect removal receives a slightly higher weight (0.6) as a fundamental prerequisite for spatial data integration, while biological fidelity retains substantial weighting (0.4).

The resulting Dataset_Weighted_Overall_Rank served as the input for the subsequent family-, scenario-, and global-level aggregation procedures.

4.4 Family-Level Rank Aggregation (Family-Level Ranking)
----

Following the per-dataset ranking described in Section 4.3, a family-level aggregation was performed to summarize method performance across related benchmark datasets. Benchmark families consist of multiple datasets derived from the same biological benchmark (e.g., HER2_A, HER2_B, HER2_C, and HER2_D belonging to the HER2 family).

For each Scenario, the rankings obtained for individual datasets were first aggregated within each family by calculating the arithmetic mean of the dataset-level ranks. This produced a single representative family-level mean rank for each method and benchmark family.

After obtaining the family-level mean ranks for all benchmark families within a scenario, ranking stability was evaluated across families rather than across individual datasets. Specifically, the overall mean family rank was calculated as:

.. math::
    \bar{F}=\frac{1}{M}\sum_{j=1}^{M}F_j

where :math:`F_j` denotes the mean rank of the j-th benchmark family and M is the total number of benchmark families in the scenario.

The standard deviation of the family-level mean ranks was then computed, from which the standard error (SE) and coefficient of variation (CV) were derived:

.. math::
    \mathrm{SE}
    =
    \frac{\mathrm{SD}(F_1,\ldots,F_M)}{\sqrt{M}},
    \qquad
    \mathrm{CV}
    =
    \frac{\mathrm{SD}(F_1,\ldots,F_M)}{\bar{F}}.

where SD(~) denotes the sample standard deviation. The family-level mean rank reflects the average performance of each method across benchmark families, 
whereas SE quantifies the uncertainty of the estimated mean rank and CV measures the relative variability of rankings across different benchmark families. 
Lower mean ranks indicate better overall performance, while lower SE and CV indicate more consistent performance across benchmark families.

For simulated datasets, family-level aggregation followed a hierarchical procedure. Rankings were first averaged across simulation gradients within each simulation framework, yielding one mean rank for each benchmark family under SRTsim and scDesign3, respectively. These two simulator-specific mean ranks were subsequently averaged with equal weights (1:1 macro averaging) to obtain the final family-level rank. This hierarchical aggregation prevents simulation frameworks with different numbers of simulation settings from disproportionately influencing the overall evaluation.

4.5 Multi-Level Macro-Averaged Rank Aggregation
----

To prevent benchmark collections with abundant sub-tasks from dominating global results, a hierarchical macro-averaging strategy was applied across hierarchical levels (dataset - family - scenario - global), with equal weighting at each tier.

**Family-level aggregation**: Family-level mean ranks were generated as described in Section 4.4 and subsequently used as the input for scenario-level aggregation.

**Scenario-level aggregation**: All family-level mean ranks within one scenario were averaged to generate scenario-level mean ranks, including Scenario_Batch_mean, Scenario_Bio_mean and Scenario_Overall_mean. Integer scenario ranks were derived by re-ranking these mean values within each scenario group. This step ensures every benchmark family contributes equally to its parent scenario, regardless of internal task count.

**Global aggregation across three scenarios**: Mean scenario ranks from Inter-slice, Inter-sample, and Cross-protocol/platform were averaged with identical weights to compute global summary ranks, namely Global_Batch_Rank_Mean, Global_Bio_Rank_Mean and Global_Overall_Rank_Mean. Integer global batch ranks, biological ranks, and final overall ranks were generated for visualization.

4.6 Coverage Ratio and Final Score Calculation (Equation 14)
-----

The final quantitative score balances integration performance and algorithm robustness, which refers to the task completion rate. Coverage Ratio equals the number of datasets successfully processed by a method divided by the total unique benchmark datasets. Methods with frequent runtime failures exhibit reduced coverage, penalizing their final score.

Performance metrics and coverage ratio were combined with a 70:30 weighting scheme to prioritize correction quality while taking practical usability into consideration. All candidate methods were sorted in descending order based on Final Score to produce the global benchmark ranking.

4.7 Statistical Validation Note
----

Statistical testing was not used as the primary basis for final method ranking. Instead, the benchmark relied on hierarchical rank aggregation, scenario-level macro-averaging, and coverage-adjusted final scores to summarize method performance across heterogeneous datasets and simulation settings. This design was adopted because not all methods were applicable to all benchmark tasks, and incomplete task coverage could otherwise introduce bias into formal paired statistical tests. Therefore, rank summaries, SE/CV estimates, coverage ratios, and complete task-level metric outputs were reported to support transparent interpretation of method performance.
