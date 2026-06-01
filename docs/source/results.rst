Results
=======

This page summarises the high-level findings of the SpaBE benchmark and
points to the figures / tables produced by the scripts in
``code/comparison/``.

Summary tables
--------------

After running :doc:`reproducibility`, the following CSV files are written
under ``code/comparison/results/``:

* ``metrics_<dataset>_<method>.csv`` — raw metric values per method
  and dataset.
* ``ranks_<dataset>.csv`` — ranks of each method per metric.
* ``time_memory.csv`` — runtime and peak RAM per method.

Headline figures
----------------

* **Radar plot** of the ten metrics per method, faceted by dataset
  (``rank_metrics_plot.py``).
* **UMAP grids** comparing batches before and after integration
  (``comparison_umap.py`` / ``umap_run.sh``).
* **Bar plots** ranking methods by aggregate score
  (``plots_metrics_benchmark.py`` / ``plots_run.sh``).

Take-home messages
------------------

* No single method dominates all four batch-effect categories.
* Graph-based contrastive methods (GraphST, SPIRAL, spatiAlign) tend
  to perform best on **inter-slice** and **inter-sample** tasks.
* Probabilistic models (PRECAST) and graph attention models
  (STAligner) are most robust on **cross-platform** data.
* **Cross-protocol** integration remains the hardest setting — all
  methods lose biological signal as protocols diverge.

For the precise numbers and discussion, see the accompanying
manuscript referenced on the :doc:`citation` page.
