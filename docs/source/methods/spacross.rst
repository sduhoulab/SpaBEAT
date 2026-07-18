SpaCross
========

Introduction
----
SpaCross proposes cross-masked graph autoencoder to build two complementary feature views, and a Cross-Masked Latent Consistency (CML) module to impose implicit latent space supervision across dual masked views. Its core innovation is the adaptive hybrid spatial-semantic graph (AHSG), which fuses local spatial neighbors and global semantically similar spots to balance spatial continuity and inter-domain semantic consistency, paired with contrastive aggregation loss to eliminate technical batch noise. 

Environment configuration
----
SpaCross is available at Github (https://github.com/wenwenmin/SpaCross). To ensure optimal computational efficiency, the package was executed on a GPU-accelerated environment, although it remains compatible with CPU-based execution.

Input data
----
Raw gene expression matrix, spatial coordinates, dataset-specific YAML configuration files, and reference annotations (e.g., spatial ground truth labels or cell types).

Unique preprocessing and parameter configuration
----
Strictly comply with unified preprocessing standard (HVG filtering, 1e4 normalization, log transform, stratified subsampling for high-resolution samples), without additional operations. For datasets with corresponding YAML configuration files, the original settings were retained; otherwise, the DLPFC configuration was applied as the default.

Output results
----
`enc_rep, recon = net.process()`. Batch-corrected unified latent embeddings and spatial domain clustering results (derived via mclust).


Installation
------------

`SpaCross <https://github.com/zhanglabtools/SpaCross>`_ integrates
spatial transcriptomics slices via a cross-attention graph neural
network. It models cross-slice spot correspondences alongside the
intra-slice spatial graph, yielding a shared embedding that aligns
slices across donors, platforms, and protocols.

.. code-block:: console

   $ conda create -n spacross python=3.9 -y
   $ conda activate spacross
   $ pip install torch==1.13.1
   $ pip install torch-geometric==2.3.0
   $ pip install scanpy==1.9.3 anndata==0.9.2
   $ git clone https://github.com/zhanglabtools/SpaCross.git
   $ pip install -e SpaCross

Reproduction scripts
--------------------

Scripts live in ``code/SpaCross/``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1new_dlpfc_spacross.py`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor (inter-slice)
   * - ``all_new_dlpfc_spacross.py``
     - DLPFC, all 12 sections (inter-sample)
   * - ``hbc_new_spacross.py``
     - Human breast cancer (inter-slice)
   * - ``mob_new_spacross.py``
     - Mouse olfactory bulb (cross-platform)
   * - ``coronal_new_spacross.py``
     - Mouse coronal brain (cross-protocol)

Run everything
--------------

.. code-block:: console

   $ cd code/SpaCross
   $ bash run_all_spacross.sh

The script writes integrated AnnData objects (``.h5ad``) containing the
SpaCross embedding in ``adata.obsm['emb']``. These embeddings feed the
benchmark metric scripts under ``code/comparison/`` (see
:doc:`../metrics`).
