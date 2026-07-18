STitch3D
========

Introduction
----

STitch3D (3D Cellular Structure Reconstruction) 

STitch3D is a unified framework designed to integrate multiple ST slices to reconstruct 3D cellular structures. This method focuses on identifying spatial domains and aligns slices through iterative closest point (ICP) or PASTE techniques, converting spatial point coordinates into a common coordinate system. It then constructs a 3D neighbor graph using a Combined Coordinate System (CCS) and generates a shared latent space using a Graph Attention Network (GAT). By accounting for slice- and gene-specific effects, STitch3D adjusts gene expression, effectively removing batch effects and integrating spatial information across multiple slices.

Environment configuration
----
STitch3D can be installed from PyPI or Github (https://github.com/YangLabHKUST/STitch3D). To ensure optimal computational efficiency, the package was executed on a GPU-accelerated environment, although it remains compatible with CPU-based execution.

Input data
----
Raw gene expression matrix, spatial coordinates and optional single-cell reference data. Notably, because the execution of STitch3D in our benchmarking pipeline relied on matched single-cell references to guide representation learning, this method was explicitly excluded from evaluations on datasets lacking paired single-cell data (e.g., the HER2, MERFISH mHypo, and simulated datasets).

Unique preprocessing and parameter configuration
----
Execute standard normalization and log transformation, stratified subsampling for high-resolution samples; select top 500 highly variable genes from external single-cell reference data to enhance the model's representation learning capacity. For the HD_crc matched single-cell reference data, we excluded samples labeled ['QCFilter'] != "Remove" to ensure input quality. Additionally, we performed spatial domain clustering using a Gaussian Mixture Model (GMM) implemented via the Python scikit-learn library.

Output results
----
Batch-corrected unified latent embeddings (stored in the ‘latent’ field) and spatial domain clustering results (derived via GMM).




Installation
------------

`STitch3D <https://github.com/YangLabHKUST/STitch3D>`_ integrates multiple
2D ST slices into a unified 3D representation while correcting for batch
effects between slices.

.. code-block:: console

   $ conda create -n stitch3d python=3.9 -y
   $ conda activate stitch3d
   $ pip install torch==1.13.1
   $ pip install scanpy==1.9.3 anndata==0.9.2
   $ pip install harmonypy==0.0.9
   $ git clone https://github.com/YangLabHKUST/STitch3D.git
   $ pip install -e STitch3D

Reproduction scripts
--------------------

Scripts live in ``code/STitch3D/`` and are dataset-specific. Run the
script that corresponds to the dataset you want to reproduce:

.. code-block:: console

   $ cd code/STitch3D
   $ python <script>.py

The output is a ``.h5ad`` file with the STitch3D 3D embedding in
``adata.obsm['stitch3d']`` that the comparison scripts under
``code/comparison/`` consume.

.. note::

   STitch3D was originally designed for *aligned* slices of the same
   tissue. For the cross-platform (Dataset 3) and cross-protocol
   (Dataset 4) benchmarks we additionally pre-align slices with PASTE
   before running STitch3D — see :doc:`../reproducibility`.
