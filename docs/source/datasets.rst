Data availability
=================

SpaBE benchmarks four spatial transcriptomics datasets that span the four
batch-effect categories introduced on the :doc:`home page <index>`.

Summary
-------

.. list-table::
   :widths: 6 26 18 18 32
   :header-rows: 1

   * - #
     - Dataset
     - Tissue / organism
     - Platforms
     - Batch effect category
   * - 1
     - Human DLPFC (12 sections)
     - Dorsolateral prefrontal cortex, human
     - 10x Visium
     - Inter-slice & inter-sample
   * - 2
     - Human breast cancer (Block A)
     - HER2+ breast tumor, human
     - 10x Visium
     - Inter-slice
   * - 3
     - Mouse olfactory bulb
     - Olfactory bulb, mouse
     - 10x Visium / Stereo-seq / Slide-seqV2
     - Cross-platform
   * - 4
     - Mouse coronal brain
     - Coronal brain, mouse
     - 10x Visium (Normal / DAPI / FFPE)
     - Cross-protocol

Dataset 1 — Human DLPFC (12 sections)
-------------------------------------

Twelve dorsolateral prefrontal cortex sections from three donors with
manual layer annotations.

* Source: `spatialLIBD <https://research.libd.org/spatialLIBD/>`_
* Suitable for both **inter-slice** (within donor) and **inter-sample**
  (across donor) evaluation.
* Used by scripts named ``*new_dlpfc_*`` and ``all_new_dlpfc_*`` in every
  method directory under ``code/``.

Dataset 2 — Human breast cancer (Block A)
-----------------------------------------

Two consecutive sections of HER2+ breast cancer, used as an **inter-slice**
benchmark.

* Block A Section 1:
  https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1
* Block A Section 2:
  https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_2
* Used by ``hbc_*`` scripts.

Dataset 3 — Mouse olfactory bulb (cross-platform)
-------------------------------------------------

Three mouse olfactory bulb slices acquired on three different platforms,
used to evaluate **cross-platform** batch effects.

* 10x Visium:
  https://www.10xgenomics.com/datasets/adult-mouse-olfactory-bulb-1-standards
* Stereo-seq and Slide-seqV2 (processed):
  https://drive.google.com/drive/folders/1Omte1adVFzyRDw7VloOAQYwtv_NjdWcG?usp=share_link
* Used by ``mob_*`` scripts.

Dataset 4 — Mouse coronal brain (cross-protocol)
------------------------------------------------

Three coronal sections of adult mouse brain acquired with three 10x Visium
**experimental protocols**, used to evaluate **cross-protocol** batch
effects.

* 10x Normal:
  https://www.10xgenomics.com/datasets/mouse-brain-section-coronal-1-standard-1-1-0
* 10x DAPI:
  https://www.10xgenomics.com/datasets/adult-mouse-brain-section-1-coronal-stains-dapi-anti-neu-n-1-standard-1-1-0
* 10x FFPE:
  https://www.10xgenomics.com/datasets/adult-mouse-brain-ffpe-1-standard-1-3-0
* Used by ``coronal_*`` scripts.

Recommended layout
------------------

We recommend the following directory layout for the downloaded data so the
reproduction scripts can be run with minimal edits:

.. code-block:: text

   data/
   ├── DLPFC/
   │   ├── 151507/
   │   ├── 151508/
   │   └── ... (12 sections)
   ├── HBC/
   │   ├── Block_A_Section_1/
   │   └── Block_A_Section_2/
   ├── MOB/
   │   ├── 10x/
   │   ├── stereo/
   │   └── slide/
   └── Coronal/
       ├── normal/
       ├── dapi/
       └── ffpe/

Each script in ``code/<method>/`` exposes a ``data_path`` variable at the
top that should be pointed at the corresponding folder.
