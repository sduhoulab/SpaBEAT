PRECAST
=======

`PRECAST <https://github.com/feiyoung/PRECAST>`_ is an R package for
probabilistic embedding, clustering and alignment of multi-section ST
data.

This is the **only R-based** method in the SpaBE benchmark.

Installation
------------

PRECAST is installed inside an R environment:

.. code-block:: console

   $ conda create -n precast -c conda-forge r-base=4.2.2 -y
   $ conda activate precast

.. code-block:: R

   install.packages("remotes")
   remotes::install_github("feiyoung/PRECAST")
   install.packages(c("Seurat", "ggplot2", "dplyr", "anndata"))

Reproduction scripts
--------------------

R scripts live in ``code/PRECAST/``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Script
     - Dataset
   * - ``1dlpfc_new_PRE.r`` / ``2`` / ``3`` / ``7374``
     - DLPFC, individual donor
   * - ``all_new_dlpfc_PRE.r``
     - DLPFC, all 12 sections
   * - ``hbc_new_PRE.r``
     - Human breast cancer
   * - ``mob_new_PRE.r``
     - Mouse olfactory bulb
   * - ``coronal_new_PRE.r``
     - Mouse coronal brain

Run individual scripts with:

.. code-block:: console

   $ cd code/PRECAST
   $ Rscript 1dlpfc_new_PRE.r

PRECAST outputs an :file:`.RDS` or :file:`.h5ad` file containing the
integrated low-dimensional embedding, which is loaded by the comparison
scripts in :doc:`../metrics`.
