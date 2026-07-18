SPIRAL
======

Introduction
----
SPIRAL (Integrating and aligning ST data) 

SPIRAL consists of two consecutive modules: spiral-integration and spiral-alignment. The spiral-integration module integrates slices from different samples and technology platforms/protocols, correcting for batch effects by combining an inductive graph neural network (GraphSAGE) with a domain adaptation network. By using the GraphSAGE encoder and performing gradient inversion in a decoupled network, SPIRAL effectively removes batch effects during data integration, ensuring that the integrated ST data maintains biological coherence.

Environment configuration
----
SPIRAL based on pyG (pytorch Geometric) framework is available at https://github.com/guott15/SPIRAL. SPIRAL primarily runs on the GPU but also relies on the CPU for partial data processing.

Input data
----
Raw gene expression matrix, spatial coordinates, pre-constructed spatial neighbor graphs (e.g., KNN edges), and reference annotations (e.g., spatial ground truth labels or cell types).

Unique preprocessing and parameter configuration
----
Strictly comply with unified preprocessing standard (HVG filtering, 1e4 normalization, log transform, stratified subsampling for high-resolution samples), with an additional step to intersect shared genes across slices to generate four core documents as input. Parameter configuration follows default settings.

Output results
----
Batch-corrected unified latent embeddings (excluding designated noise dimensions, stored in the 'spiral' field) and spatial domain clustering results (derived via mclust).



Installation
------------


`SPIRAL <https://github.com/guott15/SPIRAL>`_ integrates ST data via a
graph domain adaptation neural network that disentangles biological
signal from batch effects.

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
