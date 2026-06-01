spatiAlign
==========

`spatiAlign <https://github.com/STOmics/Spatialign>`_ is an unsupervised
contrastive learning approach for integrating multi-batch ST data,
developed by BGI/STOmics.

Installation
------------

.. code-block:: console

   $ conda create -n spatialign python=3.8 -y
   $ conda activate spatialign
   $ pip install spatialign==0.0.3
   $ pip install torch==1.13.1
   $ pip install scanpy==1.9.3 anndata==0.9.2

Reproduction scripts
--------------------

Scripts live in ``code/spatialign/``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1dlpfc_new_spa.py`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor
   * - ``all_dlpfc_new_spa.py``
     - DLPFC, all 12 sections
   * - ``hbc_new_spa.py``
     - Human breast cancer
   * - ``mob_new_spa.py``
     - Mouse olfactory bulb
   * - ``coromal_new_spa.py``
     - Mouse coronal brain

Run everything
--------------

.. code-block:: console

   $ cd code/spatialign
   $ bash run_all_spa.sh
