SPIRAL
======

`SPIRAL <https://github.com/guott15/SPIRAL>`_ integrates ST data via a
graph domain adaptation neural network that disentangles biological
signal from batch effects.

Installation
------------

.. code-block:: console

   $ conda create -n spiral python=3.9 -y
   $ conda activate spiral
   $ git clone https://github.com/guott15/SPIRAL.git
   $ cd SPIRAL
   $ pip install -r requirements.txt
   $ pip install -e .

Reproduction scripts
--------------------

Scripts live in ``code/SPIRAL/``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1dlpfc_new_spiral.py`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor
   * - ``all_dlpfc_new_spiral.py``
     - DLPFC, all 12 sections
   * - ``hbc_new_spiral.py``
     - Human breast cancer
   * - ``mob_new_spiral.py``
     - Mouse olfactory bulb
   * - ``coronal_new_spiral.py``
     - Mouse coronal brain

Run everything
--------------

.. code-block:: console

   $ cd code/SPIRAL
   $ bash run_all_spiral.sh
