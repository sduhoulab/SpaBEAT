import scanpy as sc
import squidpy as sq
import numpy as np
import os
import gc

file_path = '/data/ZhaoMH/ST0507/PRECAST-main/results/starmap_seuInt_with_all_spatial.h5ad'

if not os.path.exists(file_path):
    print(f"File not found, verify whether the R script ran successfully: {file_path}")
    exit()
    
print("="*60)
print(f"Start Python postprocessing: STARmap PRECAST")
print("="*60)

adata = sc.read_h5ad(file_path)

if 'ground_truth' in adata.obs:
    adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
    adata = adata[~adata.obs['celltype'].isna()]
    adata = adata[adata.obs['celltype'] != "unknown"].copy()

if 'spatial' in adata.obsm:
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=False)
    print("  [✔] Spatial graph computed.")
else:
    print("  [❌] Error: No spatial coordinates found!")

batch_mapping = {str(i+1): str(i) for i in range(10)}

if 'orig.ident' in adata.obs:
    mapped_batches = adata.obs['orig.ident'].astype(str).map(batch_mapping)
    adata.obs['new_batch'] = mapped_batches.fillna(adata.obs['orig.ident'])
else:
    print("  [⚠️] orig.ident batch information not detected.")

if 'PRECAST' in adata.obsm:
    adata.obsm['PRECAST_embed'] = adata.obsm['PRECAST']
elif 'X_PRECAST' in adata.obsm:
    adata.obsm['PRECAST_embed'] = adata.obsm['X_PRECAST']

adata.write(file_path)
print(f"STARmap PRECAST Python evaluation preparation done! File updated.\n")

del adata
gc.collect()