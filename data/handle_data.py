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
from utils.data_processing import load_TFbinding, extract_overlap_regions, preload_TF_binding
import torch

dataset_name = 'human_PBMC'
#
# atac_data = anndata.read_h5ad(f'filter_data/{dataset_name}/ATAC_filter.h5ad')
# TF_data = anndata.read_h5ad(f'filter_data/{dataset_name}/TF_filter.h5ad')
#
# Element_name = atac_data.var.index
#
# # 替换每个字符串中的第一个 '-' 为 ':' 只有BMMC用
# Element_name = Element_name.str.replace('-', ':', 1)
#
# # pd.DataFrame(Element_name).to_csv('Peaks.txt',header=None,index=None)
#
GRNdir = 'GRN/data_bulk/'
#
# Match2 = pd.read_csv(GRNdir + 'Match2.txt', sep='\t')
# Match2 = Match2.values
#
# motifWeight = pd.read_csv(GRNdir + 'motifWeight.txt', index_col=0, sep='\t')
#
# TFName = TF_data.var.index.values
#
# outdir = f'./{dataset_name}_TF_Binding/'
#
# extract_overlap_regions('hg38', GRNdir, outdir, 'scTFBridge')
# load_TFbinding(GRNdir, motifWeight, Match2, TFName, Element_name, outdir)
filter_data_path = f'filter_data/{dataset_name}/'
output_path = f'filter_data/{dataset_name}/TF_binding/'
preload_TF_binding(filter_data_path, GRNdir, output_path)
