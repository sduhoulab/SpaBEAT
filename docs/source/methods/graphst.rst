GraphST
=======

Introduction
-------------
GraphST (Self-Supervised Contrastive Learning) 

GraphST introduces the PASTE technique for aligning spatial coordinates to reduce batch effects during horizontal integration. When vertically integrating slices from multiple samples, the method first aligns H&E images and constructs a shared neighborhood graph. Through iterative aggregation of neighbor representations, GraphST smooths feature distributions and mitigates batch differences. A self-supervised contrastive learning strategy is then employed to reinforce spot embeddings, ensuring that spatially neighboring spots exhibit similar representations, further reducing batch effects and preserving spatial structures. 

Environment configuration
--------------
The GraphST package was obtained from the official `repository <https://github.com/JinmiaoChenLab/GraphST>`_ and installed following the provided documentation. GraphST integrates Scanpy for spatial data processing with the PyTorch and cuDNN frameworks to execute its core GNN-based contrastive learning modules. To ensure optimal computational efficiency, the package was executed on a GPU-accelerated environment, although it remains compatible with CPU-based execution.


Input data
----

Raw gene expression matrix, spatial coordinates, and reference annotations (e.g., spatial ground truth labels or cell types).

Unique preprocessing and parameter configuration
----
Adopt unified baseline preprocessing workflow including HVG selection, 1e4 normalization, log transformation and stratified downsampling for high-resolution samples; no extra custom preprocessing operations. Default parameters were used.

Output results
----
Batch-corrected unified latent embeddings (stored in the ‘emb’ or ‘emb_pca’ field) and spatial domain clustering results (derived via mclust).





Installation
------------

`GraphST <https://github.com/JinmiaoChenLab/GraphST>`_ is a graph
self-supervised contrastive learning method for spatial clustering and
batch integration of ST data.

Reference implementation:
https://github.com/JinmiaoChenLab/GraphST/tree/main

.. code-block:: console

   $ conda create -n graphst python=3.9 -y
   $ conda activate graphst
   $ pip install GraphST==1.1.1
   $ pip install torch==1.13.1 torchvision==0.14.1
   $ pip install scanpy==1.9.3 anndata==0.9.2 rpy2==3.5.13

(GPU is recommended; the same ``pip`` invocation works on CPU.)

Reproduction scripts
--------------------

Scripts live in ``code/GraphST/``. Each script targets one dataset:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1new_dlpfc_graphst.py`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor (inter-slice)
   * - ``all_new_dlpfc_graphst.py``
     - DLPFC, all 12 sections (inter-sample)
   * - ``hbc_new_graphst.py``
     - Human breast cancer (inter-slice)
   * - ``mob_new_graphst.py``
     - Mouse olfactory bulb (cross-platform)
   * - ``coronal_new_graphst.py``
     - Mouse coronal brain (cross-protocol)

Run everything
--------------

.. code-block:: console

   $ cd code/GraphST
   $ bash run_all_graphst.sh

The script writes integrated AnnData objects (``.h5ad``) containing the
GraphST embedding in ``adata.obsm['emb']`` to the configured output
directory. These embeddings are the input to the metric scripts under
``code/comparison/`` (see :doc:`../metrics`).
