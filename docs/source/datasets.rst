Data and code availability
==========================

SpaBEAT benchmarks fourteen spatial transcriptomics datasets that span the
batch-effect categories introduced on the :doc:`home page <index>`.


Real Datasets
-------------

HER2+ breast tumor (ST)
~~~~~~~~~~~~~~~~~~~~~~~

Eight HER2-positive breast tumor patients (A–H)
profiled with Spatial Transcriptomics (ST). Patients A–D contribute 6
evenly spaced sections per tumor and patients E–H contribute 3 adjacent
sections per tumor (total n = 36 sections). Section sizes range from
~176 to ~712 spots and 14,861 to 15,842 genes per section.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - One section from each tumor was examined and annotated by a pathologist (A.E.) based on the morphology of the associated H&E image.
   * - **Ground truth definition**
     -
      - ``in situ cancer``
      - ``invasive cancer``
      - ``adipose tissue``
      - ``immune infiltrate``
      - ``connective tissue``
   * - **Domain/cell type count**
     - 5 categories
   * - **Application scenarios**
     -
      - Definition 1 — Inter-slice, Non-Consecutive;
      - Definition 1 — Inter-slice, Consecutive;
      - Definition 2 — Inter-sample, Different-Samples
   * - **Reference**
     - https://www.nature.com/articles/s41467-021-26271-2
   * - **Data link (RAW)**
     - | https://github.com/almaan/HER2st/
       | https://doi.org/10.5281/zenodo.21318241
   * - **Data link (Converted)**
     - 
       

Visium ST mouse brain (slice_39 / slice_44)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

75 consecutive coronal ST sections spanning a single adult mouse brain
hemisphere. Sections 39 (620 spots, 23,371 genes) and 44 (639 spots,
23,371 genes) are used.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - Clusters were named on the basis of the anatomical region (Allen Brain Atlas definition) according to the main position identity of spots.
   * - **Ground truth definition**
     -
      - ``cerebellum``
      - ``cortical subplate``
      - ``fiber tracts``
      - ``hindbrain``
      - ``hippocampal region``
      - ``hypothalamus``
      - ``isocortex``
      - ``midbrain``
      - ``olfactory areas``
      - ``pallidum``
      - ``retrohippocampal region``
      - ``striatum``
      - ``thalamus``
      - ``ventricular systems``
   * - **Domain/cell type count**
     - ``domains = 14``
   * - **Application scenarios**
     - Definition 1 — Inter-slice, Consecutive Slices
   * - **Reference**
     - https://www.science.org/doi/10.1126/sciadv.abb3446#sec-4
   * - **Data link (RAW)**
     - https://zenodo.org/records/8167488
   * - **Data link (Converted)**
     - https://doi.org/10.5281/zenodo.21319528

MERFISH mouse hypothalamus (Slice_7–Slice_11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Five adjacent consecutive slices at Bregma −0.04, −0.09, −0.14, −0.19,
−0.24 with 5,488 / 5,557 / 5,926 / 5,803 / 5,543 cells respectively,
measured over a common panel of 155 genes.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - Cells across sections were annotated to eight distinct anatomical structures. BASS was used to obtain cell-type clustering.
   * - **Ground truth definition**
     -
      - ``V3`` (third ventricle)
      - ``BST`` (bed nuclei of the stria terminalis)
      - ``fx`` (columns of the fornix)
      - ``MPA`` (medial preoptic area)
      - ``MPN`` (medial preoptic nucleus)
      - ``PV`` (periventricular hypothalamic nucleus)
      - ``PVH`` (paraventricular hypothalamic nucleus)
      - ``PVT`` (paraventricular nucleus of the thalamus)
   * - **Domain/cell type count**
     - ``domains = 8``; ``celltypes = 15`` (used)
   * - **Application scenarios**
     -  Definition 1 — Inter-slice, Consecutive Slices
   * - **Reference**
     - https://link.springer.com/article/10.1186/s13059-022-02734-7
   * - **Data link (RAW)**
     - https://zenodo.org/records/6814510
   * - **Data link (Converted)**
     - https://doi.org/10.5281/zenodo.21319752

STARmap mouse brain (10 slices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

10 consecutive 10 µm-thick slices (0–9) from a 1400 × 1700 × 100 µm
tissue block, with 2,103 to 3,663 spots per slice and 28 genes.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - Dimensionality reduction plus graph clustering (SPACEL and the STARmap 3D reference).
   * - **Ground truth definition**
     -
      - ``L2/3`` (cortical excitatory layer)
      - ``L4`` (cortical excitatory layer)
      - ``L5`` (cortical excitatory layer)
      - ``L6`` (cortical excitatory layer)
      - ``PV`` (inhibitory interneuron)
      - ``Astro`` (astrocyte)
      - ``Olig1`` (immature oligodendrocyte)
      - ``Olig2`` (mature oligodendrocyte)
   * - **Domain/cell type count**
     - ``domains = 5``
   * - **Application scenarios**
     -  Definition 1 — Inter-slice, Consecutive Slices
   * - **Reference**
     - https://www.nature.com/articles/s41467-023-43220-3#Sec10
   * - **Data link (RAW)**
     - | https://www.starmapresources.org/data
       | https://doi.org/10.5281/zenodo.21318919
   * - **Data link (Converted)**
     - https://doi.org/10.5281/zenodo.21319822

Stereo-seq mouse embryo E16.5 (Slice_5–Slice_9)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single E16.5 mouse embryo from the MOSTA atlas (13 total whole-embryo
slices, ~7.1 to ~76.1 mm²). Slices 5–9 contain 15,105 to 19,389 spots
and 1,923 genes per slice.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - Default MOSTA parameters plus anatomic annotation from the eHistology Kaufman Annotations (http://www.emouseatlas.org/emap/home.html) and the Allen Brain Atlas (http://mouse.brainmap.org/), combined with the spatiotemporal transcriptomic atlas of mouse organogenesis clustering.
   * - **Ground truth definition**
     -
      - ``Liver``
      - ``Heart``
      - ``Connective tissue``
      - ``Meninges``
      - ``Mucosal epithelium``
      - ``Spinal cord``
      - ``Bone``
      - ``Muscle``
      - ``Cavity``
      - ``Jaw and tooth``
      - ``Brain``
      - additional tissue-level compartments
   * - **Domain/cell type count**
     - 25
   * - **Application scenarios**
     - Definition 1 — Inter-slice, Consecutive Slices
   * - **Reference**
     - https://www.nature.com/articles/s41467-023-43220-3#Sec10
   * - **Data link (RAW)**
     - https://db.cngb.org/stomics/mosta/download/
   * - **Data link (Converted)**
     - https://doi.org/10.5281/zenodo.21319948


STARmap PLUS mouse mPFC (BZ5 / BZ9 / BZ14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three independent mPFC slices from distinct mice, with 166 shared genes
and 1,049 (BZ5), 1,053 (BZ9), 1,088 (BZ14) cells, respectively.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - Cells across sections were annotated to four distinct cortical layer structures (L1, L2/3, L5, L6); BASS was used to derive cell-type clustering.
   * - **Ground truth definition**
     -
      - ``L1``
      - ``L2/3``
      - ``L5``
      - ``L6``
      - ~80–90% excitatory pyramidal neurons and ~10–20% inhibitory GABAergic interneurons; final evaluation uses BASS cell types.
   * - **Domain/cell type count**
     - ``domains = 4``; ``celltypes = 15`` (used)
   * - **Application scenarios**
     - Definition 2 — Inter-sample, Different-Samples
   * - **Reference**
     - https://link.springer.com/article/10.1186/s13059-022-02734-7
   * - **Data link (RAW)**
     - https://zenodo.org/records/10698914
   * - **Data link (Converted)**
     - https://doi.org/10.5281/zenodo.21320238

Human CRC Visium HD (hd_crc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Human colorectal cancer patients (n = 3; P1CRC / P2CRC / P5CRC), FFPE
tissue profiled with Visium HD at 8 µm bin size.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - The authors identified tumor cells via spot deconvolution, then used a custom distance-based analysis to define all barcoded 8 µm bins within 50 µm of these regions as the tumor periphery.
   * - **Ground truth definition**
     -
      - ``Tumor``
      - ``Tissue``
      - ``Periphery``
   * - **Domain/cell type count**
     - 3
   * - **Application scenarios**
     - Definition 2 — Inter-sample, Different-Samples
   * - **Reference**
     - High-definition spatial transcriptomic profiling of immune cell populations in colorectal cancer.
   * - **Data link (RAW)**
     - https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc
   * - **Data link (Converted)**
     - https://doi.org/10.5281/zenodo.21320852

Xenium human HER2+ breast cancer (Rep1 / Rep2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Human HER2+ breast cancer including ductal carcinoma in situ (DCIS)
profiled with Xenium (labelled ``Rep1_outs`` and ``Rep2_outs``).

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - Supervised cell-type annotations derived from matched Flex single-cell data.
   * - **Ground truth definition**
     -
      - ``Macrophages_1``
      - ``Stromal``
      - ``CD8+_T_Cells``
      - ``CD4+_T_Cells``
      - ``B_Cells``
      - ``Invasive_Tumor``
      - ``T_Cell_&_Tumor_Hybrid``
      - ``Prolif_Invasive_Tumor``
      - ``Unlabeled``
      - ``Macrophages_2``
      - ``Stromal_&_T_Cell_Hybrid``
      - ``DCIS_1``
      - ``Endothelial``
      - ``Myoepi_ACTA2+``
      - ``DCIS_2``
      - ``Mast_Cells``
      - ``Perivascular-Like``
      - ``IRF7+_DCs``
      - ``LAMP3+_DCs``
      - ``Myoepi_KRT15+``
   * - **Domain/cell type count**
     - 20
   * - **Application scenarios**
     -
      - Definition 1 — Inter-slice, Consecutive
   * - **Reference**
     - High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis.
   * - **Data link (RAW)**
     - https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast

Spatch human HCC (Visium HD + Xenium 5K)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Untreated human hepatocellular carcinoma (HCC) tissue samples. Each
specimen was analysed using both the Visium HD and Xenium 5K platforms.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - Cell-type annotations provided in the original studies (details not provided).
   * - **Ground truth definition**
     -
      - ``CD8T``
      - ``CD4T``
      - ``Macrophage``
      - ``Tprolif``
      - ``cDC2``
      - ``SMC``
      - ``NK``
      - ``Monocyte``
      - ``Hepatocyte``
      - ``Mast``
      - ``Kupffer``
      - ``Plasma``
      - ``mregDC``
      - ``cDC1``
      - ``Endothelial``
      - ``B``
      - ``Neutrophil``
   * - **Domain/cell type count**
     - 17
   * - **Application scenarios**
     -
      - Definition 3 — Cross-platform
   * - **Reference**
     - Systematic benchmarking of high-throughput subcellular spatial transcriptomics platforms across human tumors.
   * - **Data link (RAW)**
     - http://spatch.pku-genomics.org/

Spatch human OV (Visium HD + Xenium 5K)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Untreated human ovarian (OV) tissue samples. Each specimen was analysed
using both the Visium HD and Xenium 5K platforms.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Ground truth source**
     - Cell-type annotations provided in the original studies (details not provided).
   * - **Ground truth definition**
     -
      - ``CD8T``
      - ``CD4T``
      - ``Macrophage``
      - ``Unknown``
      - ``SMC``
      - ``NK``
      - ``DC``
      - ``Monocyte``
      - ``Fibroblast``
      - ``Mast``
      - ``Epithelial``
      - ``Plasma``
      - ``Endothelial``
      - ``Neutrophil``
   * - **Domain/cell type count**
     - 14
   * - **Application scenarios**
     -
      - Definition 3 — Cross-platform
   * - **Reference**
     - Systematic benchmarking of high-throughput subcellular spatial transcriptomics platforms across human tumors.
   * - **Data link (RAW)**
     - http://spatch.pku-genomics.org/


Simulated Datasets
------------------

Simulations use the real reference slices listed below as source data and
apply either **SRTsim** (binomial subsampling with a capture-efficiency
gradient) or **scDesign3** (negative-binomial + 2D spatial spline model
with intercept shift and Gaussian noise) to generate paired slices for
each batch-effect scenario. Simulated datasets do not have a separate
publication or data link — refer to the corresponding real datasets in
the section above.

DLPFC (simulated)
~~~~~~~~~~~~~~~~~

Human DLPFC Visium reference (spatialLIBD) — 12 serial coronal slices
from 3 healthy human donors; simulations use section 151507 (1 slice).

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Domain/cell type count**
     - 7 domains
   * - **Application scenarios**
     -
      - Definition 1 — Inter-slice, Consecutive (SRTsim gradient {1.0, 0.9, 0.7, 0.5, 0.3, 0.1}; scDesign3 gradient {0, -0.1, -0.3, -0.5, -0.7, -0.9})
      - Definition 2 — Inter-sample, Different-Samples (SRTsim biological fold change {1.0, 1.5, 2.0, 3.0} × capture gradient {1.0, 0.7, 0.3, 0.1}; scDesign3 biological fold change {1.0, 1.5, 2.0, 3.0} × batch gradient {0, -0.1, -0.5, -0.9})

HBC (simulated)
~~~~~~~~~~~~~~~

HBC Visium reference — Block A Section 1 and Section 2 are 10 µm
cryosectioned slices of invasive ductal carcinoma (IDC) breast tissue
(BioIVT Asterand) processed with the Visium Spatial protocol.
Section 1: 3,798 spots; Section 2: 3,987 spots. Simulations use
Section 1.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Domain/cell type count**
     - 10 domains
   * - **Application scenarios**
     -
      - Definition 1 — Inter-slice, Consecutive (SRTsim + scDesign3 gradients as for DLPFC)
      - Definition 2 — Inter-sample, Different-Samples (SRTsim + scDesign3 combined biological × batch gradients as for DLPFC)

Xenium breast (simulated)
~~~~~~~~~~~~~~~~~~~~~~~~~

Xenium human HER2+ breast cancer Rep1 reference, cropped to
X > median X and Y > median Y (top-right region), all genes retained.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Domain/cell type count**
     - 20 cell types
   * - **Application scenarios**
     -
      - Definition 1 — Inter-slice, Consecutive (SRTsim + scDesign3 gradients as for DLPFC)
      - Definition 2 — Inter-sample, Different-Samples (SRTsim + scDesign3 combined biological × batch gradients as for DLPFC)

MOB cross-platform (simulated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mouse olfactory bulb reference profiled on three platforms (10X Visium,
BGI Stereomics, Slide-seqV2). SRTsim performs platform-specific fitting
— domain-level for 10X Visium and tissue-level for the sparser BGI and
Slide-seqV2 slices — and outputs three slices, one per platform, with
no gradient.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Domain/cell type count**
     - 9 domains
   * - **Application scenarios**
     -
      - Definition 3 — Cross-platform

Spatch OV (simulated pseudo-Visium)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ovarian Xenium slice from the Spatch dataset cropped to the top-right
region above the XY median, with the top 5,000 abundant genes retained.
A 50 × 50 spatial-bin aggregation with majority-vote cell labels
downsamples the single-cell Xenium data to Visium-HD-like spot
resolution. SRTsim (Xenium → HD baseline P0, gradient {0.7, 0.3, 0.1})
and scDesign3 (baseline P0, gradient {-0.1, -0.5, -0.9}) generate the
paired slices.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Details
   * - **Domain/cell type count**
     - 13 domains
   * - **Application scenarios**
     -
      - Definition 3 — Cross-platform


Recommended layout
------------------

We recommend the following directory layout for the downloaded data so
the reproduction scripts can be run with minimal edits:

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
   ├── Coronal/
   │   ├── normal/
   │   ├── dapi/
   │   └── ffpe/
   ├── HER2/
   ├── VisiumST/
   ├── MERFISH/
   ├── STARmap/
   ├── Stereo_seq_embryo/
   ├── PFC/
   ├── hd_crc/
   ├── Xenium_breast/
   ├── Spatch_HCC/
   └── Spatch_OV/

Each script in ``code/<method>/`` exposes a ``data_path`` variable at the
top that should be pointed at the corresponding folder.
