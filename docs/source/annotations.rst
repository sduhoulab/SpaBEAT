Annotation and ground truth
===========================

This page summarises the annotation source, granularity, completeness, and
recommended evaluation metrics for each benchmark dataset in SpaBEAT.

Summary
-------

.. list-table::
   :widths: 5 16 12 16 16 16 14 14 14
   :header-rows: 1

   * - ID
     - Dataset
     - Annotation type
     - Labels / domains
     - Annotation source
     - Annotation method
     - Completeness
     - Recommended metrics
     - Notes
   * - D1
     - DLPFC
     - Domain / layer
     - Layer1–Layer6, WM / 7 domains
     - Original study manual annotation
     - Histology-informed manual layer annotation
     - Complete
     - ARI, cLISI, ASW_domain, SCS, marker validation
     - Structured brain tissue, relatively clear layers
   * - D2
     - HBC
     - Tumor / domain
     - Section-level annotations (TBD)
     - Original / manual (TBD)
     - Manual / original label (TBD)
     - Partial / complete (TBD)
     - ARI, cLISI, ASW (if available)
     - Less-structured tumor; labels may be partial
   * - D5
     - Xenium breast2
     - Cell type / region
     - Cell type / domain (TBD)
     - Matched scRNA-seq / reference
     - Reference mapping / label transfer
     - Partial
     - ARI, cLISI, ASW, segmentation sensitivity
     - Annotation quality depends on reference
   * - D5-seg
     - Xenium breast (Cellpose)
     - Cell type
     - Cell type (TBD)
     - Matched scRNA-seq reference
     - Cellpose segmentation + reference annotation
     - Partial
     - Segmentation sensitivity metrics
     - Segmentation changes cell boundaries and counts
   * - D10
     - MERFISH
     - Cell type
     - Cell types (TBD)
     - Original dataset
     - Original annotation
     - Partial / complete (TBD)
     - ARI, cLISI, ASW
     - Targeted panel only
   * - D11
     - STARmap
     - Cell type
     - Cell types (TBD)
     - Original dataset
     - Original annotation
     - Partial / complete (TBD)
     - ARI, cLISI, ASW
     - Targeted panel only
   * - D7
     - OV Xenium / Visium HD
     - Cell type / region
     - TBD
     - Original / reference
     - Reference mapping / original labels
     - Partial
     - Cross-platform metrics
     - Feature-space mismatch across platforms
   * - D13
     - HER2
     - Tumor region / domain
     - TBD
     - Original publication / manual / marker
     - TBD
     - Partial / complete (TBD)
     - ARI, cLISI, ASW (if labels available)
     - Tumor regions less stereotyped
   * - SIM-S2
     - DLPFC / HBC / Xenium simulations
     - Known perturbation labels
     - Bio_0/1/2/3 and Batch_0/1/2/3/4
     - Simulation design
     - Known controlled labels
     - Complete
     - Over-correction retention, simulation metrics
     - Top 50 genes; observed FC used as empirical oracle


Per-dataset details
-------------------

D1 — DLPFC
~~~~~~~~~~

* Annotation type: domain / layer
* Labels: Layer1–Layer6, WM (7 domains)
* Source: original study manual annotation
* Method: histology-informed manual layer annotation
* Completeness: complete
* Recommended metrics: ARI, cLISI, ASW_domain, SCS, marker validation
* Notes: structured brain tissue with relatively clear laminar
  organisation, providing a strong reference for domain-identification
  benchmarks.

D2 — HBC
~~~~~~~~

* Annotation type: tumor / domain
* Labels: section-level annotations (TBD)
* Source: original / manual (to be confirmed)
* Method: manual / original label (TBD)
* Completeness: partial or complete (to be confirmed)
* Recommended metrics: ARI, cLISI, ASW if labels are available
* Notes: less-structured tumor tissue; label coverage may be partial.

D5 — Xenium breast2
~~~~~~~~~~~~~~~~~~~

* Annotation type: cell type / region
* Labels: cell type / domain (TBD)
* Source: matched scRNA-seq / reference
* Method: reference mapping / label transfer
* Completeness: partial
* Recommended metrics: ARI, cLISI, ASW, segmentation sensitivity
* Notes: annotation quality depends on the chosen reference.

D5-seg — Xenium breast (Cellpose)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Annotation type: cell type
* Labels: cell type (TBD)
* Source: matched scRNA-seq reference
* Method: Cellpose segmentation followed by reference annotation
* Completeness: partial
* Recommended metrics: segmentation sensitivity metrics
* Notes: segmentation alters cell boundaries and counts, and downstream
  labels inherit that variability.

D10 — MERFISH
~~~~~~~~~~~~~

* Annotation type: cell type
* Labels: cell types (TBD)
* Source: original dataset
* Method: original annotation
* Completeness: partial or complete (to be confirmed)
* Recommended metrics: ARI, cLISI, ASW
* Notes: analysis restricted to the targeted gene panel.

D11 — STARmap
~~~~~~~~~~~~~

* Annotation type: cell type
* Labels: cell types (TBD)
* Source: original dataset
* Method: original annotation
* Completeness: partial or complete (to be confirmed)
* Recommended metrics: ARI, cLISI, ASW
* Notes: analysis restricted to the targeted gene panel.

D7 — OV Xenium / Visium HD
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Annotation type: cell type / region
* Labels: TBD
* Source: original / reference
* Method: reference mapping / original labels
* Completeness: partial
* Recommended metrics: cross-platform metrics
* Notes: feature-space mismatch across platforms requires shared or
  matched genes for evaluation.

D13 — HER2
~~~~~~~~~~

* Annotation type: tumor region / domain
* Labels: TBD
* Source: original publication / manual / marker-based
* Method: TBD
* Completeness: partial or complete (to be confirmed)
* Recommended metrics: ARI, cLISI, ASW when labels are available
* Notes: tumor regions are less stereotyped, so shared reference
  annotations across samples may be limited.

SIM-S2 — DLPFC / HBC / Xenium simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Annotation type: known perturbation labels
* Labels: Bio_0 / Bio_1 / Bio_2 / Bio_3 and Batch_0 / Batch_1 / Batch_2 /
  Batch_3 / Batch_4
* Source: simulation design
* Method: known controlled labels
* Completeness: complete
* Recommended metrics: over-correction retention and other simulation
  metrics
* Notes: top 50 genes are used; the observed fold change serves as the
  empirical oracle for retention evaluation.
