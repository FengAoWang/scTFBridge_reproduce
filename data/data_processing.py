import sys
sys.path.append('/home/wfa/project/single_cell_multimodal')
from utils.data_processing import five_fold_split_dataset, adata_multiomics_processing
import anndata
import scanpy as sc

# adata = anndata.read_h5ad('GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
# #
# print(adata)
# print(adata.var)
# # 提取 GEX 数据
# gex_mask = adata.var['feature_types'] == 'GEX'
# print("GEX mask shape:", gex_mask.shape)
#
# # 提取 ATAC 数据
# atac_mask = adata.var['feature_types'] == 'ATAC'
#
#
# rna_adata = adata[:, gex_mask]
# atac_adata = adata[:, atac_mask]
#
#
# # 打印提取的数据
# print("GEX 数据:")
# print(rna_adata)
# rna_adata.write('filter_data/BMMC/BMMC_RNA_processed.h5ad')
#
#
# print("ATAC 数据:")
# print(atac_adata)
# atac_adata.write('filter_data/BMMC/BMMC_ATAC_processed.h5ad')


cell_types_annotations = [
    'NK',
    'Lymphatic',
    'AT2',
    'AT1',
    'T',
    'Club',
    'Ciliated',
    'Artery',
    'Vein',
    'Macrophage',
    'Goblet',
    'Fibroblast',
    'Monocyte',
    'Basal',
    'Sm. Mus.',
    'Capillary',
    'Mesothelial',
    'B',
    'Dendritic',
    'AT1/AT2',
    'NK/T',
    'Myofib.',
    'AT2-pro'
]



dataset = 'GSE243917'
rna_adata = anndata.read_h5ad(f'filter_data/{dataset}/GSE243917_genesXcells_ALL_batch.h5ad')
atac_data = anndata.read_h5ad(f'filter_data/{dataset}/GSE243917_peaksXcells_ALL_batch.h5ad')

TF_name = '../data/GRN/data_bulk/TFName.txt'
TF_name = open(TF_name, 'r').readlines()
for i in range(len(TF_name)):
    TF_name[i] = TF_name[i].replace('\n', '')

# adata = sc.concat((rna_adata, atac_data), axis=0)
# print(adata.var.columns)
TF_list = ['RUNX1', 'STAT1', 'IRF1', 'CTCF', 'REST', 'SPl1']
adata_multiomics_processing([rna_adata, atac_data],
                            [f'filter_data/{dataset}/RNA_filter.h5ad',
                             f'filter_data/{dataset}/ATAC_filter.h5ad',
                             f'filter_data/{dataset}/TF_filter.h5ad',
                             f'filter_data/{dataset}/batch_info.csv'],
                            2000, 0.01, TF_name, TF_list)

rna_adata = anndata.read_h5ad(f'filter_data/{dataset}/RNA_filter.h5ad')
five_fold_split_dataset(rna_adata, f'filter_data/{dataset}/fold_split_info.csv')
