import scanpy as sc
import squidpy as sq
import numpy as np
import os
import gc

base_dir = '/root/ST0507/new_model/PRECAST/results/SRTsim_dlpfc_t1'
ref_slice = "Sim_Slice_1"


for i in range(2, 7):
    target_slice = f"Sim_Slice_{i}"
    pair_name = f"{ref_slice}_vs_{target_slice}"
    file_path = os.path.join(base_dir, f"multiple_adata_{pair_name}_precast.h5ad")
    
    if not os.path.exists(file_path):
        print(f"⚠️ File not found, skip: {file_path}")
        continue
        
    print("="*60)
    print(f"🚀 Starting Python post-processing: {pair_name}")
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

    batch_mapping = {
        '1': ref_slice, 
        '2': target_slice
    }

    if 'batch' in adata.obs:
        mapped_batches = adata.obs['batch'].astype(str).map(batch_mapping)
        adata.obs['new_batch'] = mapped_batches.fillna(adata.obs['batch'])
    else:
        adata.obs['new_batch'] = adata.obs['orig.ident']

    if 'PRECAST' in adata.obsm:
        adata.obsm['PRECAST_embed'] = adata.obsm['PRECAST']
    elif 'X_PRECAST' in adata.obsm:
        adata.obsm['PRECAST_embed'] = adata.obsm['X_PRECAST']

    adata.write(file_path)
    print(f"✅ {pair_name} Python evaluation preparation done! File updated.\n")
    
    del adata
    gc.collect()

print("🎯 All Python post-processing tasks completed!")