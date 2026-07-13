SpaBEAT (Spatial Batch Effect Assessment and Testing)
================================================

**SpaBEAT** is a reproducibility framework that systematically defines and
benchmarks batch effects in spatial transcriptomics (ST).

We categorize ST batch effects into four types (inter-slice, inter-sample,
cross-protocol/platform, and intra-slice) and evaluate seven representative
integration methods — GraphST, DeepST, PRECAST, STAligner, SPIRAL,
STitch3D and spatiAlign — across four human and mouse datasets using ten
metrics that capture both batch-effect removal and biological-signal
preservation.

Documentation
-------------

The full documentation, including installation, datasets, per-method
tutorials, metrics and reproduction steps, is built with Sphinx from the
``docs/`` directory and hosted on Read the Docs.

To build the docs locally::

   pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/build/html
   open docs/build/html/index.html

To auto-build the docs on file changes::

   pip install -r docs/requirements.txt
   sphinx-autobuild docs/source docs/build/html


Reproducibility code
--------------------

The ``code/`` directory contains one subfolder per benchmarked method
plus a ``comparison/`` folder with the metric, UMAP, ranking and
plotting scripts. See ``docs/source/reproducibility.rst`` for the
end-to-end pipeline.
