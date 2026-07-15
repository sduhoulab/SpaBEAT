import sys
sys.path.insert(0, '/data/ZhaoMH/ST0507/SPIRAL-main')

import os
import gc
import time
import json
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import anndata
import scanpy as sc
import argparse
from sklearn.decomposition import PCA
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import umap.umap_ as umap

if 'spiral' in sys.modules:
    del sys.modules['spiral']

import spiral

print("spiral path:")
print(spiral.__file__)

from spiral.main import SPIRAL_integration
from spiral.layers import *
from spiral.utils import *
from spiral.CoordAlignment import CoordAlignment
from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_spiral_integration(ref_slice, target_slice, data_path, base_save_path, n_domains=4):
    datasets = [ref_slice, target_slice]
    pair_name = f"{ref_slice}_vs_{target_slice}"
    
    save_path = base_save_path
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n=== Running SPIRAL Benchmark for: {pair_name} (n_domains = {n_domains}) ===")
    knn = 8
    N_WALKS = knn
    WALK_LEN = 1
    N_WALK_LEN = knn
    NUM_NEG = knn

    feat_file = []
    edge_file = []
    meta_file = []
    coord_file = []

    for dataset in datasets:
        base_path_ds = os.path.join(data_path, dataset, "GNN_format")
        feat_file.append(os.path.join(base_path_ds, f"{dataset}_features.txt"))
        edge_file.append(os.path.join(base_path_ds, f"{dataset}_edge_KNN_{knn}.csv"))
        meta_file.append(os.path.join(base_path_ds, f"{dataset}_label.txt"))
        coord_file.append(os.path.join(base_path_ds, f"{dataset}_positions.txt"))

    print("Aligning and formatting input files securely...")
    dfs = []
    for f_path in feat_file:
        df = pd.read_csv(f_path, sep=None, index_col=0, engine='python')
        df.index = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()
        dfs.append(df)

    common_genes = set(dfs[0].columns)
    for df in dfs[1:]:
        common_genes = common_genes.intersection(set(df.columns))

    common_genes = sorted(list(common_genes))
    print(f"Common genes for {pair_name}: {len(common_genes)}")

    new_feat_files = []
    for i, df in enumerate(dfs):
        df_common = df.loc[:, common_genes]
        new_file = os.path.join(save_path, f"temp_{datasets[i]}_features_common_{pair_name}.txt")
        df_common.to_csv(new_file, sep=',')
        new_feat_files.append(new_file)

    feat_file = new_feat_files
    N = len(common_genes)
    M = 1 if len(datasets) == 2 else len(datasets)

    total_cells = sum([pd.read_csv(f, header=0, index_col=0).shape[0] for f in feat_file])
    print(f"Total cells to process: {total_cells:,} | Total genes: {N:,}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--AEdims', type=list, default=[N,[512],32])
    parser.add_argument('--AEdimsR', type=list, default=[32,[512],N])
    parser.add_argument('--GSdims', type=list, default=[512,32])
    parser.add_argument('--zdim', type=int, default=32)
    parser.add_argument('--znoise_dim', type=int, default=4)
    parser.add_argument('--CLdims', type=list, default=[4,[],M])
    parser.add_argument('--DIdims', type=list, default=[28,[32,16],M])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--agg_class', type=str, default=MeanAggregator)
    parser.add_argument('--num_samples', type=int, default=knn)
    parser.add_argument('--N_WALKS', type=int, default=N_WALKS)
    parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN)
    parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN)
    parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--alpha1', type=float, default=N)
    parser.add_argument('--alpha2', type=float, default=1)
    parser.add_argument('--alpha3', type=float, default=1)
    parser.add_argument('--alpha4', type=float, default=1)
    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--Q', type=float, default=10)

    params, unknown = parser.parse_known_args([])
    
    params.AEdims = [N, [512], 32]
    params.AEdimsR = [32, [512], N]
    params.CLdims = [4, [], M]
    params.DIdims = [28, [32, 16], M]
    params.alpha1 = N

    # =============== SPIRAL Training ===============
    print("\nStarting SPIRAL training benchmarking...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_before = get_memory_usage()
    training_start_time = time.time()

    SPII = SPIRAL_integration(params, feat_file, edge_file, meta_file)
    SPII.train()

    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    memory_used = get_memory_usage() - memory_before
    print("Training completed!")

    SPII.model.eval()
    all_idx = np.arange(SPII.feat.shape[0])
    all_layer, all_mapping = layer_map(all_idx.tolist(), SPII.adj, len(SPII.params.GSdims))
    all_rows = SPII.adj.tolil().rows[all_layer[0]]
    all_feature = torch.Tensor(SPII.feat.iloc[all_layer[0],:].values).float().to(device)

    all_embed, ae_out, clas_out, disc_out = SPII.model(all_feature, all_layer, all_mapping, all_rows, SPII.params.lamda, SPII.de_act, SPII.cl_act)
    [ae_embed, gs_embed, embed] = all_embed

    embed = embed.cpu().detach()
    names = ['GTT_'+str(i) for i in range(embed.shape[1])]
    embed1 = pd.DataFrame(np.array(embed), index=SPII.feat.index, columns=names)
    
    embed_file = os.path.join(save_path, f"SPIRAL_embed_{pair_name}.csv")
    embed1.to_csv(embed_file)
    embed_df = pd.DataFrame(embed.numpy(), index=SPII.feat.index)
    znoise_dim = SPII.params.znoise_dim
    embed_new_df = pd.concat([
        pd.DataFrame(np.zeros((embed_df.shape[0], znoise_dim)), index=embed_df.index),
        embed_df.iloc[:, znoise_dim:]
    ], axis=1)
    
    embed_new = torch.tensor(embed_new_df.values).float().to(device)
    xbar_new = np.array(SPII.model.agc.ae.de(embed_new, nn.Sigmoid())[1].cpu().detach())
    xbar_new1 = pd.DataFrame(xbar_new, index=SPII.feat.index, columns=SPII.feat.columns)

    xbar_file = os.path.join(save_path, f"SPIRAL_correct_{pair_name}.csv")
    xbar_new1.to_csv(xbar_file)

    print("Performing clustering analysis...")
    ann = anndata.AnnData(SPII.feat)
    ann.obsm['spiral'] = embed1.iloc[:, SPII.params.znoise_dim:].values
    sc.pp.neighbors(ann, use_rep='spiral')

    ann = mclust_R(ann, used_obsm='spiral', num_cluster=n_domains)
    ann.X = SPII.feat
    ann.obs['batch'] = SPII.meta.loc[:, 'batch'].values
    ann.obs['celltype'] = SPII.meta.loc[:, 'celltype'].values

    coord = pd.read_csv(coord_file[0], header=0, index_col=0)
    for i in np.arange(1, len(datasets)):
        coord = pd.concat((coord, pd.read_csv(coord_file[i], header=0, index_col=0)))
    coord.columns = ['y', 'x']

    if len(coord) == ann.n_obs:
        ann.obsm['spatial'] = coord.values
    else:
        raise ValueError(f"❌ 长度不匹配！坐标表有 {len(coord)} 行，但细胞有 {ann.n_obs} 个！")

    ann.obs["new_batch"] = ann.obs["batch"].astype(str)
    
    ann.write(os.path.join(save_path, f"multiple_adata_{pair_name}_spiral.h5ad"))

    benchmark_results = {
        'method_name': 'SPIRAL',
        'dataset': pair_name,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'training_time_hours': training_time / 3600,
        'memory_usage_mb': memory_used,
        'memory_usage_gb': memory_used / 1024,
        'total_cells': total_cells,
        'final_cells': SPII.feat.shape[0],
        'total_genes': N,
        'embedding_dim': embed.shape[1],
        'n_datasets': len(datasets),
        'epochs': params.epochs,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    with open(os.path.join(save_path, f"spiral_benchmark_{pair_name}.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"✅ 任务 {pair_name} SPIRAL 整合完毕！结果已保存。")

    for f in new_feat_files:
        if os.path.exists(f):
            os.remove(f)

    del SPII, ann, coord, embed1, xbar_new1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    DATA_PATH = "/data/ZhaoMH/ST0507/simulations_model/1_DLPFC_InterSlice/SRTsim_Converted/" 
    BASE_SAVE_PATH = "/data/ZhaoMH/ST0507/SPIRAL-main/results/SRTsim_dlpfc_t1" 
    
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    REF_SLICE = "Sim_Slice_1"

    N_DOMAINS = 7
    for i in range(2, 7):
        target_slice = f"Sim_Slice_{i}"
        
        print("="*60)
        print(f"🚀 start: {REF_SLICE} + {target_slice}")
        print("="*60)
        
        try:
            run_spiral_integration(
                ref_slice=REF_SLICE, 
                target_slice=target_slice, 
                data_path=DATA_PATH, 
                base_save_path=BASE_SAVE_PATH, 
                n_domains=N_DOMAINS
            )
        except Exception as e:
            print(f"❌ Task {REF_SLICE}_vs_{target_slice} failed. Error message: {str(e)}")
            continue
                    
    print("🎯 All STAligner slice pair integration tasks have been completed!")