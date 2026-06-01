Reproducing the benchmark
=========================

This page walks through the full pipeline end-to-end. It assumes you
have already followed :doc:`installation` and downloaded the data as
described in :doc:`datasets`.

1. Pre-process the raw data
---------------------------

The script ``code/comparison/raw_data.py`` converts each downloaded
dataset into a uniform :file:`AnnData` object:

.. code-block:: console

   $ cd code/comparison
   $ python raw_data.py --dataset DLPFC   --in data/DLPFC   --out h5ad/DLPFC
   $ python raw_data.py --dataset HBC     --in data/HBC     --out h5ad/HBC
   $ python raw_data.py --dataset MOB     --in data/MOB     --out h5ad/MOB
   $ python raw_data.py --dataset Coronal --in data/Coronal --out h5ad/Coronal

For Dataset 2 (HBC) ``hbc_metadata.py`` adds the manual annotations.
For Dataset 3 (MOB) ``mob_h5ad.py`` harmonises spot coordinates across
platforms.

2. Run each integration method
------------------------------

Activate the per-method conda environment (see each
:doc:`Integration Methods <methods/graphst>` page) and execute the
corresponding ``run_all_*.sh`` script:

.. code-block:: console

   $ conda activate graphst    && bash code/GraphST/run_all_graphst.sh
   $ conda activate deepst     && bash code/DeepST/run_all_deepst.sh
   $ conda activate precast    && bash code/PRECAST/run_all_precast.sh  # via Rscript
   $ conda activate staligner  && bash code/STAligner/run_all_sta.sh
   $ conda activate spiral     && bash code/SPIRAL/run_all_spiral.sh
   $ conda activate stitch3d   && python code/STitch3D/<dataset>.py
   $ conda activate spatialign && bash code/spatialign/run_all_spa.sh

Each script writes an integrated :file:`.h5ad` per dataset.

3. Compute the ten metrics
--------------------------

Switch to the evaluation environment (``spabe-eval``) and run:

.. code-block:: console

   $ conda activate spabe-eval
   $ cd code/comparison
   $ python metrics_benchmark.py        # 5 batch-removal + 5 bio metrics
   $ python spatial_metrics.py          # spatially-aware ARI / NMI
   $ python "four metrcis.py"           # headline 4 metrics
   $ python time_memory.py              # runtime + RAM

Outputs are written as CSV files alongside the integrated AnnData.

4. Generate plots
-----------------

.. code-block:: console

   $ bash plots_run.sh   # bar plots, radar plots, ranks
   $ bash umap_run.sh    # per-method UMAPs and side-by-side comparisons

The notebook-style entry points
``plots_metrics_benchmark.py``, ``rank_metrics_plot.py`` and
``comparison_umap.py`` can also be run individually if you only need a
subset of figures.

5. STAligner downstream analysis
--------------------------------

For the case study of STAligner on DLPFC, the additional script
``Downstream_sta.py`` reproduces the differential-expression and
cell-cell communication figures from the paper.

Expected runtime
----------------

On a workstation with an NVIDIA RTX 3090 and 64 GB RAM, the full
benchmark (all four datasets × seven methods) takes roughly **18–24
hours of wall time**, dominated by DeepST on Dataset 1 (all 12 DLPFC
sections).
