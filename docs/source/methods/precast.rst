PRECAST
=======

Introduction
----
PRECAST (Probabilistic embedding, clustering, and alignment for integrating ST) 

PRECAST is a probabilistic model that integrates slices from multiple samples by directly taking normalized gene expression matrices as input. It aligns and estimates joint embeddings for biological effects across different domains. The method applies a simple projection strategy combined with spatial dimension reduction and spatial clustering. It then utilizes aligned representations, intrinsic CAR components, estimated labels, and Potts models to deal with batch effects in spatial data, ensuring that biological signals are preserved while removing unwanted technical variations. 

Environment configuration
----
Analyses were performed using PRECAST (v1.9), an open-source R package available via CRAN or GitHub (https://github.com/feiyoung/PRECAST). The GitHub repository provides comprehensive documentation, configuration guidelines for diverse operating systems, and illustrative tutorials. Since PRECAST relies on C++ based statistical optimization, the computational benchmarking and modeling were executed entirely on high-performance CPU architectures.

Input data
----
Required .rds format, containing high-dimensional spatial transcriptomic expression matrices, corresponding spot- or cell-level metadata, and associated experimental annotations.


Unique preprocessing and parameter configuration
----
Processed via the standard Seurat pipeline. This workflow included library size normalization (NormalizeData), stratified subsampling for high-resolution slices, and data standardization (ScaleData). Highly variable genes (HVGs) were identified using either the vst (variance-stabilizing transformation) method or SPARK-X to capture spatially variable features. For the HD_crc and Xenium breast datasets, we utilized the SPARK-X method with the number of HVGs set to 5,000 to improve spatial feature detection; for all other datasets, the number of HVGs was set to the default of 2,000. Subsequently, a PRECASTObj was initialized via CreatePRECASTObject, followed by AddAdjList and AddParSetting to configure the model parameters. Finally, data integration was performed using IntegrateSpaData, and spatial clustering was executed by specifying the cluster count (K) in the PRECAST function.

Output results
----
Batch-corrected unified latent embeddings (stored in the ‘X_PRECAST’ field) and spatial domain clustering results (derived via PRECAST).


Installation
------------

`PRECAST <https://github.com/feiyoung/PRECAST>`_ is an R package for
probabilistic embedding, clustering and alignment of multi-section ST
data.

This is the **only R-based** method in the SpaBEAT benchmark.

PRECAST is installed inside an R environment:

.. code-block:: console

   $ conda create -n precast -c conda-forge r-base=4.2.2 -y
   $ conda activate precast

.. code-block:: R

   install.packages("remotes")
   remotes::install_github("feiyoung/PRECAST")
   install.packages(c("Seurat", "ggplot2", "dplyr", "anndata"))

Reproduction scripts
--------------------

R scripts live in ``code/PRECAST/``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1dlpfc_new_PRE.r`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor
   * - ``all_new_dlpfc_PRE.r``
     - DLPFC, all 12 sections
   * - ``hbc_new_PRE.r``
     - Human breast cancer
   * - ``mob_new_PRE.r``
     - Mouse olfactory bulb
   * - ``coronal_new_PRE.r``
     - Mouse coronal brain

Run individual scripts with:

.. code-block:: console

   $ cd code/PRECAST
   $ Rscript 1dlpfc_new_PRE.r

PRECAST outputs an :file:`.RDS` or :file:`.h5ad` file containing the
integrated low-dimensional embedding, which is loaded by the comparison
scripts in :doc:`../metrics`.
