GraphST
=======

`GraphST <https://github.com/JinmiaoChenLab/GraphST>`_ is a graph
self-supervised contrastive learning method for spatial clustering and
batch integration of ST data.

Reference implementation:
https://github.com/JinmiaoChenLab/GraphST/tree/main

Installation
------------

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
