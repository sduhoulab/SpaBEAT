SpaBEC Reproducibility
=====================

**SpaBEC** is a reproducibility framework that systematically defines and
benchmarks **batch effects in spatial transcriptomics (ST)**.

Spatial transcriptomics enables high-resolution profiling of gene expression
within tissue slices, but technical and experimental variations introduce
batch effects that obscure biological signals and complicate data integration.
This site documents the framework, datasets, methods, and metrics used in our
study *"A systematic framework for defining and evaluating batch
correction in spatial transcriptomics"*.

We categorize batch effects in ST into four types:

1. **Inter-slice** — batch effects between consecutive slices of the same sample.
2. **Inter-sample** — batch effects between different biological samples.
3. **Cross-protocol / cross-platform** — batch effects across sequencing
   technologies (10x Visium, Stereo-seq, Slide-seqV2, etc.).
4. **Intra-slice** — batch effects within a single slice.

We use this framework to evaluate seven representative batch-correction /
integration methods across human and mouse datasets, scoring each with ten
metrics that capture both batch-effect removal and biological signal preservation.

.. note::

   This project is under active development. The reproducibility code lives
   in the ``code/`` directory of the SpaBEC repository.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   datasets

.. toctree::
   :maxdepth: 2
   :caption: Integration Methods

   methods/graphst
   methods/deepst
   methods/precast
   methods/staligner
   methods/spiral
   methods/stitch3d
   methods/spatialign

.. toctree::
   :maxdepth: 2
   :caption: Benchmark

   metrics
   reproducibility
   results

.. toctree::
   :maxdepth: 1
   :caption: About

   citation
