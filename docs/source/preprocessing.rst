Preprocessing and QC
====================

This page documents the preprocessing and quality-control (QC) settings
used for every benchmark dataset. Each dataset is mapped to a single
section that summarises the raw input, QC thresholds, normalization,
feature selection, coordinate handling, labels, output format, and the
script used to reproduce the preprocessing.

Summary
-------

.. list-table::
   :widths: 14 20 14 16 16 20
   :header-rows: 1

   * - Dataset
     - Raw input
     - QC (spots/cells)
     - Normalization / log / scaling
     - HVG / feature selection
     - Notes
   * - DLPFC
     - Spot-by-gene raw counts
     - TBD
     - Library-size norm (TBD) + log1p (TBD) + scaling (TBD)
     - Top HVGs; top 5000 for simulation
     - Methods must specify normalization and HVG number
   * - HBC
     - Spot-by-gene raw counts
     - TBD
     - TBD
     - TBD
     - Non-brain tissue; annotation may be partial
   * - Xenium breast2
     - Cell-by-gene targeted matrix
     - Cell QC (TBD)
     - TBD
     - All genes; overlap 300 / 250 / 200 / 100 for sensitivity
     - Targeted panel — do not call HVG top-3000 unless used
   * - Visium HD
     - Bin-by-gene / spot-by-gene
     - Bin QC (TBD)
     - TBD
     - HVG / top genes (TBD)
     - Requires bin size specification
   * - OV pseudo-Visium
     - Xenium cell counts aggregated to 50 µm grid
     - Min 3 cells per pseudo-spot
     - TBD
     - Top 5000 or all if fewer than 5000
     - 50 µm grid-based binning; not Giotto
   * - MOB cross-platform
     - Platform-specific matrices
     - TBD
     - TBD
     - Top 1000 for scDesign3
     - SRTsim domain / tissue-level fitting by platform
   * - MERFISH
     - Cell-by-gene targeted matrix
     - TBD
     - TBD
     - All detected targeted genes (TBD)
     - Image-based targeted panel
   * - STARmap
     - Cell-by-gene targeted matrix
     - TBD
     - TBD
     - All detected targeted genes (TBD)
     - Image-based targeted panel
   * - HER2 breast cancer
     - Spot-by-gene raw counts
     - TBD
     - TBD
     - HVG / top genes (TBD)
     - Requires script and annotations


DLPFC
-----

* Raw input: spot-by-gene raw counts
* QC (spots): TBD
* QC (genes): TBD
* Normalization: library-size normalization (TBD)
* Log transform: log1p (TBD)
* Scaling: TBD
* HVG / feature selection: top HVGs; top 5000 for simulation
* Number of features used: TBD
* Shared gene handling: intersection of genes across slices
* Spatial coordinates: array coordinates
* Batch label: slice / sample
* Biological label: layer / domain
* Output format: h5ad / RDS (TBD)
* Script path: ``code/preprocessing/...`` (TBD)
* Notes: methods must specify normalization and HVG number.
* Status: TBD

HBC
---

* Raw input: spot-by-gene raw counts
* QC (spots): TBD
* QC (genes): TBD
* Normalization: TBD
* Log transform: TBD
* Scaling: TBD
* HVG / feature selection: TBD
* Number of features used: TBD
* Shared gene handling: intersection of genes
* Spatial coordinates: array coordinates
* Batch label: section / slice
* Biological label: tumor / domain label
* Output format: TBD
* Script path: TBD
* Notes: non-brain tissue; annotation may be partial.
* Status: TBD

Xenium breast2
--------------

* Raw input: cell-by-gene targeted matrix
* QC (cells): cell QC (TBD)
* QC (genes): all targeted genes / detected genes
* Normalization: TBD
* Log transform: TBD
* Scaling: TBD
* HVG / feature selection: all genes; overlap 300 / 250 / 200 / 100 for
  sensitivity
* Number of features used: ~300
* Shared gene handling: shared targeted genes
* Spatial coordinates: cell centroid x/y
* Batch label: slice / sample / segmentation
* Biological label: cell type / domain
* Output format: h5ad / csv (TBD)
* Script path: TBD
* Notes: targeted panel — do not call HVG top-3000 unless used.
* Status: TBD

Visium HD
---------

* Raw input: bin-by-gene / spot-by-gene
* QC (bins): TBD
* QC (genes): TBD
* Normalization: TBD
* Log transform: TBD
* Scaling: TBD
* HVG / feature selection: HVG / top genes (TBD)
* Number of features used: TBD
* Shared gene handling: shared genes
* Spatial coordinates: bin coordinates
* Batch label: sample
* Biological label: TBD
* Output format: TBD
* Script path: TBD
* Notes: bin size must be specified.
* Status: TBD

OV pseudo-Visium
----------------

* Raw input: Xenium cell counts aggregated to a 50 µm grid
* QC (pseudo-spots): minimum 3 cells per pseudo-spot
* QC (genes): top 5000 or all if fewer than 5000
* Normalization: TBD
* Log transform: TBD
* Scaling: TBD
* HVG / feature selection: high-abundance genes / all genes if fewer than
  5000
* Number of features used: TBD
* Shared gene handling: shared with Xenium
* Spatial coordinates: grid center coordinates
* Batch label: platform / condition
* Biological label: majority cell type / domain
* Output format: genes × pseudo-spots matrix + metadata
* Script path: ``code/simulation/pseudo_visium_OV.R`` (TBD)
* Notes: 50 µm grid-based binning; not Giotto.
* Status: TBD

MOB cross-platform
------------------

* Raw input: platform-specific matrices
* QC (spots/cells): TBD
* QC (genes): TBD
* Normalization: TBD
* Log transform: TBD
* Scaling: TBD
* HVG / feature selection: top 1000 for scDesign3
* Number of features used: 1000 for scDesign3
* Shared gene handling: shared genes
* Spatial coordinates: platform coordinates
* Batch label: platform id
* Biological label: domain / cell type
* Output format: TBD
* Script path: TBD
* Notes: SRTsim performs domain / tissue-level fitting by platform.
* Status: TBD

MERFISH
-------

* Raw input: cell-by-gene targeted matrix
* QC (cells): TBD
* QC (genes): all / selected targeted genes (TBD)
* Normalization: TBD
* Log transform: TBD
* Scaling: TBD
* HVG / feature selection: all detected targeted genes (TBD)
* Number of features used: TBD
* Shared gene handling: intersection across slices
* Spatial coordinates: cell coordinates
* Batch label: slice
* Biological label: cell type
* Output format: TBD
* Script path: TBD
* Notes: image-based targeted dataset.
* Status: TBD

STARmap
-------

* Raw input: cell-by-gene targeted matrix
* QC (cells): TBD
* QC (genes): all / selected targeted genes (TBD)
* Normalization: TBD
* Log transform: TBD
* Scaling: TBD
* HVG / feature selection: all detected targeted genes (TBD)
* Number of features used: TBD
* Shared gene handling: intersection across slices
* Spatial coordinates: cell coordinates
* Batch label: slice
* Biological label: cell type
* Output format: TBD
* Script path: TBD
* Notes: image-based targeted dataset.
* Status: TBD

HER2 breast cancer
------------------

* Raw input: spot-by-gene raw counts
* QC (spots): TBD
* QC (genes): TBD
* Normalization: TBD
* Log transform: TBD
* Scaling: TBD
* HVG / feature selection: HVG / top genes (TBD)
* Number of features used: TBD
* Shared gene handling: intersection of genes
* Spatial coordinates: spot coordinates
* Batch label: slice / sample
* Biological label: tumor / domain
* Output format: TBD
* Script path: TBD
* Notes: script and annotations still required.
* Status: TBD
