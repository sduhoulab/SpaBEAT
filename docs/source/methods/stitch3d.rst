STitch3D
========

`STitch3D <https://github.com/YangLabHKUST/STitch3D>`_ integrates multiple
2D ST slices into a unified 3D representation while correcting for batch
effects between slices.

Installation
------------

.. code-block:: console

   $ conda create -n stitch3d python=3.9 -y
   $ conda activate stitch3d
   $ pip install torch==1.13.1
   $ pip install scanpy==1.9.3 anndata==0.9.2
   $ pip install harmonypy==0.0.9
   $ git clone https://github.com/YangLabHKUST/STitch3D.git
   $ pip install -e STitch3D

Reproduction scripts
--------------------

Scripts live in ``code/STitch3D/`` and are dataset-specific. Run the
script that corresponds to the dataset you want to reproduce:

.. code-block:: console

   $ cd code/STitch3D
   $ python <script>.py

The output is a ``.h5ad`` file with the STitch3D 3D embedding in
``adata.obsm['stitch3d']`` that the comparison scripts under
``code/comparison/`` consume.

.. note::

   STitch3D was originally designed for *aligned* slices of the same
   tissue. For the cross-platform (Dataset 3) and cross-protocol
   (Dataset 4) benchmarks we additionally pre-align slices with PASTE
   before running STitch3D — see :doc:`../reproducibility`.
