STAligner 
=========

Introduction
------------------------------

STAligner (Graph Attention Neural Network for ST Data)

STAligner, built on the STAGATE model, is an advanced integration method for ST data that aligns slices from the same sample, 
different samples, and different technology platforms/protocols. It constructs a spatial neighbor graph based on spot coordinates 
and uses a graph attention autoencoder to create spatially-aware embeddings. The method incorporates spot triplet learning 
based on mutual nearest neighbor (MNN) principles to minimize the distance between anchor-positive pairs while maximizing 
the distance between anchor-negative pairs during slice alignment. This strategy effectively reduces batch effects in the latent space, 
enhancing integration quality and spatial coherence. 

Environment configuration
--------------------------

The STAligner package was obtained from the official `repository <https://github.com/zhanglabtools/STAligner>`_ and installed following the provided documentation. STAligner is built upon the Scanpy, PyTorch, and PyTorch Geometric (PyG) frameworks. To ensure optimal computational efficiency, the package was executed on a GPU-accelerated environment, although it remains compatible with CPU-based execution.

Input data
----------

Raw gene expression matrix, spatial coordinates, and reference annotations, such as spatial domain labels or cell-type labels.

Unique preprocessing and parameter configuration
---------

Follow unified standard pipeline (5000 HVGs, target_sum=1e4 normalization, log1p, stratified downsampling to 15,000 cells for high-resolution samples); no extra scaling or custom normalization steps added. Default parameters were used.


Output results
-----------

Batch-corrected unified latent embeddings (stored in the ‘STAligner’ field) and spatial domain clustering results (derived via mclust).



Installation
------------

`STAligner <https://github.com/zhanglabtools/STAligner>`_ uses a graph
attention auto-encoder with triplet learning to align and integrate
multiple ST slices.

.. code-block:: console

   $ conda create -n staligner python=3.8 -y
   $ conda activate staligner
   $ pip install STAligner==1.0.0
   $ pip install torch==1.13.1 torch-geometric==2.3.0
   $ pip install scanpy==1.9.3 anndata==0.9.2

Reproduction scripts
--------------------

Scripts live in ``code/STAligner/``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1new_dlpfc_sta.py`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor
   * - ``all_new_dlpfc_sta.py``
     - DLPFC, all 12 sections
   * - ``hbc_new_sta.py``
     - Human breast cancer
   * - ``mob_new_sta.py``
     - Mouse olfactory bulb
   * - ``coronal_new_sta.py``
     - Mouse coronal brain

Run everything
--------------

.. code-block:: console

   $ cd code/STAligner
   $ bash run_all_sta.sh

A dataset-specific metric helper (``STAligner_metrics.py``) is also
available under ``code/comparison/`` to score STAligner outputs.
