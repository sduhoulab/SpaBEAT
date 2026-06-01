STAligner
=========

`STAligner <https://github.com/zhanglabtools/STAligner>`_ uses a graph
attention auto-encoder with triplet learning to align and integrate
multiple ST slices.

Installation
------------

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
