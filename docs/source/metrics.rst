Evaluation metrics
==================

SpaBEAT evaluates each integration method along **two axes** with **ten
metrics in total**. Five metrics quantify *batch-effect removal*, and
five quantify *biological-signal preservation*. The implementations live
in ``code/comparison/``.

Batch-effect removal
--------------------

These metrics are higher when batches are well mixed in the embedding.

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Metric
     - Range
     - Description
   * - kBET
     - [0, 1]
     - k-nearest-neighbour batch effect test; fraction of neighbourhoods
       with batch composition indistinguishable from background.
   * - iLISI
     - [1, B]
     - Integration LISI; effective number of batches in a local
       neighbourhood. Higher = better mixing.
   * - Batch ASW
     - [0, 1]
     - Silhouette of batch labels, rescaled so 1 = perfect mixing.
   * - Graph connectivity
     - [0, 1]
     - Fraction of cells of the same biological label connected in a kNN
       graph built on the integrated embedding.

Spatial structure preservation
------------------------------

These metrics are higher when known biology (manual annotations or
clusters) is preserved.

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Metric
     - Range
     - Description
   * - SCS
     - [-∞,+∞]
     - Quantifies the spatial coherence of biological signals by measuring how consistently similar expression patterns are preserved among neighboring spots.
   * - Moran’s I
     - [-1, 1]
     - Measures the global spatial autocorrelation of gene expression, indicating whether similar expression values are spatially clustered.
   * - Geary’s C
     - [-1, 1]
     - Quantifies the spatial coherence of biological signals by measuring how consistently similar expression patterns are preserved among neighboring spots
   * - ARI
     - [-1, 1]
     - Adjusted Rand Index between Leiden / mclust clusters and manual
       layer annotations.
   * - cLISI
     - [1, C]
     - Cell-type LISI; lower = better preservation of cell-type identity.
   * - Cell-type ASW
     - [-1, 1]
     - Silhouette of biological labels in the integrated embedding.

Running the metrics
-------------------

The main entry point is:

.. code-block:: console

   $ cd code/comparison
   $ python metrics_benchmark.py \
       --input  <method_output.h5ad> \
       --label  Layer \
       --batch  batch \
       --output results/<method>.csv

Additional helpers:

* ``spatial_metrics.py`` — adds spatially-aware variants of ARI.
* ``four metrcis.py`` — runs the four headline metrics in one shot.
* ``time_memory.py`` — records runtime and peak RAM for each method.
* ``rank_metrics_plot.py`` — produces the radar / rank plots shown in
  the paper.

The same scripts are wrapped by ``plots_run.sh`` and ``umap_run.sh`` for
batch generation of all figures.
