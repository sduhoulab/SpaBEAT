DeepST
======

`DeepST <https://github.com/JiangBioLab/DeepST>`_ identifies spatial
domains and integrates multiple ST slices via a deep learning model that
combines histology, gene expression and spatial location.

Installation
------------

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
