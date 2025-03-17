import os
import sys
sys.path.append('/home/wfa/project/single_cell_multimodal')
import anndata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import numpy as np
import scanpy as sc
import matplotlib
import h5py
from scipy.sparse import csr_matrix
import anndata
import torch
from utils.data_processing import load_TFbinding, extract_overlap_regions
import torch

dataset_name = 'GSE241468'
#
atac_data = anndata.read_h5ad(f'filter_data/{dataset_name}/ATAC_filter.h5ad')
TF_data = anndata.read_h5ad(f'filter_data/{dataset_name}/TF_filter.h5ad')

Element_name = atac_data.var.index

# 替换每个字符串中的第一个 '-' 为 ':' 只有BMMC用
Element_name = Element_name.str.replace('-', ':', 1)

pd.DataFrame(Element_name).to_csv('Peaks.txt',header=None,index=None)

GRNdir = 'GRN/data_bulk/'

Match2 = pd.read_csv(GRNdir + 'Match2.txt', sep='\t')
Match2 = Match2.values

motifWeight = pd.read_csv(GRNdir + 'motifWeight.txt', index_col=0, sep='\t')

TFName = TF_data.var.index.values

outdir = f'./{dataset_name}_TF_Binding/'

extract_overlap_regions('hg38', GRNdir, outdir, 'LINGER')
load_TFbinding(GRNdir, motifWeight, Match2, TFName, Element_name, outdir)

# TF_binding = pd.read_csv(outdir + 'TF_binding.txt', sep='\t', header=None)
# print(TF_binding)
# print(TF_binding.shape)
# print(atac_data.obs_names)
# index_ = atac_data.obs.index.str[:5]
# atac_data.obs['batch'] = index_
# print(atac_data.obs['batch'].unique())
# atac_data.obs.rename(columns={'cell_annotation': 'cell_type'}, inplace=True)
# # atac_data.obs.rename(columns={'ct_subtype': 'cell_subtype'}, inplace=True)
#
# print(atac_data.obs.columns)
# atac_data.write(f'filter_data/{dataset_name}/RNA_raw.h5ad')

# rna_data = anndata.read_h5ad('filter_data/Xie_2023/H3K27me3.h5ad')
# # T_data = anndata.read_h5ad('filter_data/GSE243917/GSE243917_T_RNA_processed.h5ad')
# # B_data = anndata.read_h5ad('filter_data/GSE243917/GSE243917_B_RNA_processed.h5ad')
# # fib_data = anndata.read_h5ad('filter_data/GSE243917/GSE243917_fibroblast_RNA_processed.h5ad')
# # my_data = anndata.read_h5ad('filter_data/GSE243917/GSE243917_myeloid_RNA_processed.h5ad')
#
# print(rna_data)
# print(rna_data.obs)
# print(rna_data.obs.columns)
# print(data.obs)
# print(data.obs.columns)

# print(data.obs['ct_subtype'])
# print(data.obs['donor'])
# # 检查是否有可用的GPU
# if torch.cuda.is_available():
#     # 获取GPU的数量
#     num_gpus = torch.cuda.device_count()
#     print(f"Number of available GPUs: {num_gpus}\n")
#
#     for i in range(num_gpus):
#         print(f"GPU {i} details:")
#
#         # 获取GPU名称
#         gpu_name = torch.cuda.get_device_name(i)
#         print(f"  Name: {gpu_name}")
#
#         # 获取GPU计算能力
#         gpu_capability = torch.cuda.get_device_capability(i)
#         print(f"  Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
#
#         # 获取GPU总内存
#         gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # 转换为GB
#         print(f"  Total Memory: {gpu_memory_total:.2f} GB")
#
#         # 获取GPU的当前温度（如果支持）
#         try:
#             gpu_temp = torch.cuda.utilization_stats(i).current_temperature
#             print(f"  Current Temperature: {gpu_temp}°C")
#         except AttributeError:
#             print(f"  Current Temperature: Not available")


# matplotlib.use('Agg')  # 设置为无界面模式


# data = sc.read_10x_h5('10xgenomics/10k_Human_PBMCs/10k_PBMC_Multiome_nextgem_Chromium_X_filtered_feature_bc_matrix.h5')
# data = sc.read_10x_h5('GSE151302/GSM4572188_Control2_filtered_peak_bc_matrix.h5')
# atac_data = sc.read_10x_h5('GSE151302/GSM4572193_Control2_filtered_feature_bc_matrix.h5')
# # adata = anndata.read_h5ad('')
#
# print(data)
#
# print(atac_data)
# import h5py
#
#
# PTH = "GSE151302/GSM4572193_Control2_filtered_feature_bc_matrix.h5"

# adata = anndata.read_h5ad('GSE194122/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad')
# # 使用scanpy进行基因表达数据过滤
# # 1. 过滤掉低质量的细胞
# sc.pp.filter_cells(adata, min_genes=200)  # 至少有200个基因在每个细胞中表达
# sc.pp.filter_cells(adata, min_counts=1000)  # 每个细胞中至少有1000个UMI（唯一分子标记）
#
# batches_info = adata.obs['batch'].values.tolist()
# # print(batches_info)
#
# one_hot_encoded_batches = pd.get_dummies(batches_info, prefix='batch', dtype=float)
#
# batch_dims = one_hot_encoded_batches.shape[1]

# # 从adata.var中提取特征类型为GEX的数据
# gex_var = adata.var[adata.var['feature_types'] == 'GEX']
# gex_data = adata[:, gex_var.index]
#
# # 从adata.var中提取特征类型为ATAC的数据
# atac_var = adata.var[adata.var['feature_types'] == 'ATAC']
# atac_data = adata[:, atac_var.index]

# print(adata.var)
# def print_hdf5_structure(group, indent=0):
#     """ Recursively prints the structure of an HDF5 group. """
#     for key in group.keys():
#         item = group[key]
#         print(' ' * indent + str(key) + ' (' + type(item).__name__ + ')')
#         if isinstance(item, h5py.Group):
#             print_hdf5_structure(item, indent + 2)
#
#
# # file_path = 'your_file.h5'
# with h5py.File(file_path, 'r') as f:
#     print_hdf5_structure(f)
#     print(f['features']['genome'].data)

# Open the HDF5 file
# with h5py.File(file_path, 'r') as f:
#     data = f['matrix']
#     for key in data.keys():
#         print(data[key], key, data[key].name)
#
#     data = f['matrix']['data']
#     features = f['matrix']['features']
#     print(features)
#     print(data[:])
#     indices = f['matrix']['indices']
#     print(indices[:])
#     for key in features.keys():
#         print(features[key][:], key, features[key].shape)

# Open the HDF5 file
# with h5py.File(file_path, 'r') as f:
#     # Access the 'matrix' group
#     matrix_group = f['matrix']
#
#     # Extract datasets
#     barcodes = np.array(matrix_group['barcodes']).astype(str)
#     genes = np.array(matrix_group['features']['id']).astype(str)
#     print(genes)
#     data = np.array(matrix_group['data']).astype(float)
#     indices = np.array(matrix_group['indices']).astype(int)
#     print(indices.shape)
#     indptr = np.array(matrix_group['indptr']).astype(int)
#     print(indptr.shape)
#     shape = np.array(matrix_group['shape']).astype(int)
#     shape = (shape[1], shape[0])
#     print(shape)
# # Create a sparse matrix
# from scipy.sparse import csr_matrix
#
# X = csr_matrix((data, indices, indptr), shape=shape)
#
# # Create AnnData object
# adata = anndata.AnnData(X)
# adata.obs_names = barcodes
# adata.var_names = genes

# Save to AnnData format
# adata.write('H_Kidney_Cancer_data.h5ad')
# Create a sparse matrix
# adata = anndata.read_h5ad('H_Kidney_Cancer_data.h5ad')
#
# print(adata)
# sc.pp.normalize_total(adata, target_sum=1e4)
# # sc.pp.log1p(adata)
#
# sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000, subset=True)
#
# adata = adata[adata.layers["counts"].sum(1) != 0]  # Remove cells with all zeros.

# # 基因和细胞的过滤
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)
#
# # 归一化和对数变换
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)

# multiomics = adata.var
# cell_types = adata.obs
#
# print(multiomics)
# print(cell_types)
# print(cell_types.columns)
#
# print(adata)
# # 获取表达矩阵
# expression_matrix = adata.X
#
# # 转换为 Pandas DataFrame
# df_expression_matrix = pd.DataFrame(expression_matrix.todense(), index=adata.obs_names, columns=adata.var_names)
#
# # 输出 Pandas DataFrame
# print(df_expression_matrix)
#
# # 降维分析，如PCA
# sc.tl.pca(adata)
#
# # 邻近图构建
# sc.pp.neighbors(adata)
#
# # 聚类
# # sc.tl.louvain(adata)
# sc.tl.umap(adata)
# # 可视化UMAP结果并保存
# sc.pl.umap(adata, color='cell_type', save='umap_cell_types.pdf')
# sc.pl.umap(adata, color='batch', save='umap_batch.pdf')
#
#
# df_np = np.array(df_expression_matrix.values)
#
# umap_reducer = umap.UMAP(n_components=2, random_state=66, n_neighbors=200)
# tsne = TSNE(n_components=2, random_state=66, learning_rate='auto', init='random')
#
# embedding_2d = tsne.fit_transform(df_np)

# data = pd.read_csv('eqtl/2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt', sep='\t')
# print(data.columns)
# print(data.head())
# print(data['GeneSymbol'].unique())
# gene1 = data[data['GeneSymbol'] == 'GPR153']
# print(gene1)

# data = 'GSE236903/GSE236903_filtered_feature_bc_matrix.h5'
# data = h5py.op(data)
# print(data)