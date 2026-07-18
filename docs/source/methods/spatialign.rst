spatiAlign
==========

Introduction
----

spatiAlign (Unsupervised contrastive learning model) 

spatiAlign performs self-supervised comparative learning through the Deep Graph Information Maximisation(DGI) technique for integrating slices from multiple samples. This approach uses across-domain adaptation techniques to align joint embeddings, effectively accounting for batch effects across tissue sections. By leveraging the DGI framework, spatiAlign enhances the integration process while preserving the spatial context and minimizing the influence of batch effects on biological interpretations. 

Environment configuration
----

spatiAlign can be installed from PyPI or Github (https://github.com/STOmics/Spatialign). spatiAlign is built upon torch, torch_geometric, torch_cluster, torch_scatter and torch_sparse dependencies. To ensure optimal computational efficiency, the package was executed on a GPU-accelerated environment, although it remains compatible with CPU-based execution.

Input data
----

Raw gene expression matrix, spatial coordinates, and reference annotations (e.g., spatial ground truth labels or cell types).

Unique preprocessing and parameter configuration
----
Follow universal preprocessing standard (target_sum=1e4 normalization, log1p, stratified cell downsampling for high-resolution samples); retain only cells and genes with a minimum of 20 features and 20 observations, respectively. Notably, feature scaling was explicitly disabled (is_scale=False) during model initialization. The Spatialign function was performed with the latent representation dimension set to 100 and a k-nearest neighbor graph constructed. Model training was conducted with hyperparameter settings configured to (0.05, 1, 0.1) to optimize the alignment of spatial transcriptomic slices. Post-integration, the unified embeddings were subsequently scaled prior to spatial domain clustering.

Output results
----
Batch-corrected unified latent embeddings (stored in the ‘correct’ field) and spatial domain clustering results (derived via GMM).



Installation
------------


`spatiAlign <https://github.com/STOmics/Spatialign>`_ is an unsupervised
contrastive learning approach for integrating multi-batch ST data,
developed by BGI/STOmics.

.. code-block:: console

   $ conda create -n spatialign python=3.8 -y
   $ conda activate spatialign
   $ pip install spatialign==0.0.3
   $ pip install torch==1.13.1
   $ pip install scanpy==1.9.3 anndata==0.9.2

Reproduction scripts
--------------------

Scripts live in ``code/spatialign/``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1dlpfc_new_spa.py`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor
   * - ``all_dlpfc_new_spa.py``
     - DLPFC, all 12 sections
   * - ``hbc_new_spa.py``
     - Human breast cancer
   * - ``mob_new_spa.py``
     - Mouse olfactory bulb
   * - ``coromal_new_spa.py``
     - Mouse coronal brain

Run everything
--------------

.. code-block:: console

   $ cd code/spatialign
   $ bash run_all_spa.sh
