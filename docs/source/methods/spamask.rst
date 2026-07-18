SpaMask
=======

Introduction
----
SpaMask is a dual-masking graph autoencoder with contrastive learning dedicated to spatial transcriptomics clustering and multi-slice integration, composed of two parallel shared-weight GNN branches: Masked Graph Autoencoder (MGAE) and Masked Graph Contrastive Learning (MGCL). The MGAE branch performs node masking on gene expression features and uses GCN decoder and scaled cosine error loss to reconstruct masked spot information by aggregating spatial neighbors. The MGCL branch implements edge masking on spatial adjacency graphs, constructing positive/negative edge pairs and edge-based NCE contrastive loss to tighten embeddings of spatially adjacent spots. The two branches are jointly optimized by weighted total loss. 

Environment configuration
----
SpaMask is available at Github (https://github.com/wenwenmin/SpaMask), which provides guidelines for installation. To ensure optimal computational efficiency, the package was executed on a GPU-accelerated environment, although it remains compatible with CPU-based execution.

Input data
----
Raw gene expression matrix, spatial coordinates, and reference annotations (e.g., spatial ground truth labels or cell types).


Unique preprocessing and parameter configuration
----
Strictly comply with unified preprocessing standard (HVG filtering, 1e4 normalization, log transform, stratified subsampling for high-resolution samples), without additional operations. Parameter configuration follows default settings.

Output results
----
Batch-corrected unified latent embeddings and spatial domain clustering results (derived via mclust). 


Installation
------------

`SpaMask <https://github.com/JinmiaoChenLab/SpaMask>`_ is a
masking-based self-supervised framework for spatial transcriptomics.
It learns spot embeddings by reconstructing randomly masked gene
expression and graph edges over the spatial neighborhood graph,
yielding representations that are robust to batch effects between
slices.


.. code-block:: console

   $ conda create -n spamask python=3.9 -y
   $ conda activate spamask
   $ pip install torch==1.13.1
   $ pip install torch-geometric==2.3.0
   $ pip install scanpy==1.9.3 anndata==0.9.2
   $ git clone https://github.com/JinmiaoChenLab/SpaMask.git
   $ pip install -e SpaMask

Reproduction scripts
--------------------

Scripts live in ``code/SpaMask/``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1new_dlpfc_spamask.py`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor (inter-slice)
   * - ``all_new_dlpfc_spamask.py``
     - DLPFC, all 12 sections (inter-sample)
   * - ``hbc_new_spamask.py``
     - Human breast cancer (inter-slice)
   * - ``mob_new_spamask.py``
     - Mouse olfactory bulb (cross-platform)
   * - ``coronal_new_spamask.py``
     - Mouse coronal brain (cross-protocol)

Run everything
--------------

.. code-block:: console

   $ cd code/SpaMask
   $ bash run_all_spamask.sh

The script writes integrated AnnData objects (``.h5ad``) containing the
SpaMask embedding in ``adata.obsm['emb']``. These embeddings are loaded
by the metric scripts under ``code/comparison/`` (see
:doc:`../metrics`).
