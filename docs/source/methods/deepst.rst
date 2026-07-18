DeepST
======

Introduction
----

DeepST (Deep Learning Framework for ST) 

DeepST is a customizable deep learning framework that integrates ST data across multiple batches and technology platforms/protocols. It learns joint embeddings across batches and maps them into a shared latent space for integration. The method combines a graph neural network (GNN) autoencoder with a denoising autoencoder to generate a latent representation of augmented ST data. Additionally, DeepST introduces a gradient reversal layer (GRL) and domain discriminator through domain adversarial neural networks (DAN), effectively eliminating batch effects and adapting the model to cross-domain differences in data. 

Environment configuration
----
To install DeepST, make sure PyTorch and PyG are installed. For more details on dependencies, refer to Github (https://github.com/JiangBioLab/DeepST). To ensure optimal computational efficiency, the package was executed on a GPU-accelerated environment, although it remains compatible with CPU-based execution.

Input data
----
Raw gene expression matrix, spatial coordinates, optional H&E histology images and reference annotations (e.g., spatial ground truth labels or cell types).

Unique preprocessing and parameter configuration
----
Stratified downsampling for high-resolution samples; adopt target_sum=1 normalization, followed by log1p and global scaling; no HVG filtering step. The model training was executed via the main.run function (task = "Integration"). For the HD_crc and Xenium breast datasets, the _get_augment function utilized the BallTree algorithm to calculate spatial distances and perform neighbor searches. Furthermore, for samples lacking corresponding H&E-stained tissue images, the use_morphological parameter was set to False.

Output results
----
Batch-corrected unified latent embeddings (stored in the ‘DeepST’ field) and spatial domain clustering results (derived via mclust).


Installation
------------

`DeepST <https://github.com/JiangBioLab/DeepST>`_ identifies spatial
domains and integrates multiple ST slices via a deep learning model that
combines histology, gene expression and spatial location.

.. code-block:: console

   $ conda create -n deepst python=3.8 -y
   $ conda activate deepst
   $ pip install torch==1.13.1
   $ pip install scanpy==1.9.3 anndata==0.9.2
   $ pip install torch-geometric==2.3.0
   $ git clone https://github.com/JiangBioLab/DeepST.git
   $ pip install -e DeepST

DeepST expects histology images for 10x Visium datasets. Make sure the
``spatial/`` subfolder produced by Space Ranger is present.

Reproduction scripts
--------------------

Scripts live in ``code/DeepST/`` and mirror the per-dataset layout of the
other methods:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1new_dlpfc_deepst.py`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor
   * - ``all_new_dlpfc_deepst.py``
     - DLPFC, all 12 sections
   * - ``hbc_new_deepst.py``
     - Human breast cancer
   * - ``mob_deepst.py``
     - Mouse olfactory bulb
   * - ``coronal_new_deepst.py``
     - Mouse coronal brain

Run everything
--------------

.. code-block:: console

   $ cd code/DeepST
   $ bash run_all_deepst.sh
