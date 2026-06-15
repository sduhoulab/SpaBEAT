Installation
============

The **SpaBEC** benchmark reproduces seven spatial transcriptomics integration
methods and ten evaluation metrics. Because the seven methods have
mutually incompatible Python/R environments, we recommend creating one
:command:`conda` environment **per method**.

.. _general-prerequisites:

Prerequisites
-------------

* Linux or macOS (most methods are tested on Ubuntu 20.04 / 22.04)
* `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or
  `Mamba <https://mamba.readthedocs.io/>`_
* Python ≥ 3.8 (per-method versions listed in each tutorial)
* R ≥ 4.1 (only required for :doc:`methods/precast`)
* (Optional) CUDA 11.x for GPU acceleration of GraphST, DeepST, STAligner,
  SPIRAL and spatiAlign

.. _clone-repo:

Clone the repository
--------------------

.. code-block:: console

   $ git clone https://github.com/sduhoulab/SpaBEC.git
   $ cd SpaBEC

The reproducibility scripts live under ``code/``:

.. code-block:: text

   code/
   ├── GraphST/        # GraphST reproduction scripts
   ├── DeepST/         # DeepST reproduction scripts
   ├── PRECAST/        # PRECAST (R) reproduction scripts
   ├── STAligner/      # STAligner reproduction scripts
   ├── SPIRAL/         # SPIRAL reproduction scripts
   ├── STitch3D/       # STitch3D reproduction scripts
   ├── spatialign/     # spatiAlign reproduction scripts
   └── comparison/     # Metrics, UMAPs, ranking and plotting

Create a base environment
-------------------------

A minimal environment that can run the metric and plotting scripts in
``code/comparison/`` is sufficient if you only want to **score existing
embeddings**:

.. code-block:: console

   $ conda create -n spabe-eval python=3.9 -y
   $ conda activate spabe-eval
   $ pip install \
       scanpy==1.9.3 \
       anndata==0.9.2 \
       numpy==1.23.5 \
       pandas==1.5.3 \
       scikit-learn==1.2.2 \
       scib==1.1.4 \
       matplotlib==3.7.1 \
       seaborn==0.12.2

For per-method environments, see the *Installation* section of each method
page under :doc:`Integration Methods <methods/graphst>`.

Hardware notes
--------------

* The largest dataset (12 DLPFC slices, ~50k spots × 33k genes) requires
  roughly **32 GB RAM** for the comparison/UMAP scripts.
* GPU is optional but speeds up GraphST, DeepST, STAligner, SPIRAL and
  spatiAlign by 5–20×. PRECAST is CPU/R only.
