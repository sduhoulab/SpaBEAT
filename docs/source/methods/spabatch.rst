SpaBatch
========

Introduction
----
SpaBatch is a variational graph autoencoder (VGAE)-based multi-slice integration tool for 3D spatial domain identification and batch effect removal, applicable to diverse tissues, species and sequencing platforms including MERFISH and Stereo-seq. It adopts a two-stage training pipeline: pre-training via masked VGAE to capture spatial-transcriptional latent features with scaled cosine reconstruction loss and KL regularization; fine-tuning combines Deep Embedded Clustering (DEC) for compact cluster formation and readout-aggregated triplet contrastive loss to correct cross-slice batch effects. The framework constructs block-diagonal multi-slice spatial graphs and supports semi-supervised spatial domain prediction using limited annotated slices. It can denoise gene expression signals to strengthen marker gene spatial patterns.

Environment configuration
----
SpaBatch is available at Github (https://github.com/wenwenmin/SpaBatch). To ensure optimal computational efficiency, the package was executed on a GPU-accelerated environment, although it remains compatible with CPU-based execution.

Input data
----
Raw gene expression matrix and spatial coordinates, and ground truth annotations (strictly reserved for downstream spatial domain clustering and benchmarking evaluation, rather than model training)

Unique preprocessing and parameter configuration
----
Strictly comply with unified preprocessing standard (HVG filtering, 1e4 normalization, log transform, stratified subsampling for high-resolution samples), without additional operations. Parameter configuration follows default settings.

Output results
----
`SpaBatch_feat, q = SpaBatch_net.process()`. Batch-corrected unified latent embeddings (stored in the ‘SpaBatch’ fields) and spatial domain clustering results (derived via mclust).




Installation
------------

`SpaBatch <https://github.com/QSong-github/SpaBatch>`_ is a deep
learning framework tailored to batch-effect correction in spatial
transcriptomics. It combines a graph autoencoder over the spatial
neighborhood graph with an explicit batch-adversarial objective, so the
learned embedding mixes slices while retaining spatial domain
structure.

.. code-block:: console

   $ conda create -n spabatch python=3.9 -y
   $ conda activate spabatch
   $ pip install torch==1.13.1
   $ pip install torch-geometric==2.3.0
   $ pip install scanpy==1.9.3 anndata==0.9.2
   $ git clone https://github.com/QSong-github/SpaBatch.git
   $ pip install -e SpaBatch

(GPU is recommended; the same ``pip`` invocation works on CPU.)

Reproduction scripts
--------------------

Scripts live in ``code/SpaBatch/``. Each script targets one dataset and
follows the same naming convention as the other methods in this
benchmark:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1new_dlpfc_spabatch.py`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor (inter-slice)
   * - ``all_new_dlpfc_spabatch.py``
     - DLPFC, all 12 sections (inter-sample)
   * - ``hbc_new_spabatch.py``
     - Human breast cancer (inter-slice)
   * - ``mob_new_spabatch.py``
     - Mouse olfactory bulb (cross-platform)
   * - ``coronal_new_spabatch.py``
     - Mouse coronal brain (cross-protocol)

Run everything
--------------

.. code-block:: console

   $ cd code/SpaBatch
   $ bash run_all_spabatch.sh

The script writes integrated AnnData objects (``.h5ad``) containing the
SpaBatch embedding in ``adata.obsm['emb']`` to the configured output
directory. These embeddings are consumed by the metric scripts under
``code/comparison/`` (see :doc:`../metrics`).
