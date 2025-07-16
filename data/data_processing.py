from utils.data_processing import five_fold_split_dataset, adata_multiomics_processing
import anndata


dataset = 'human_PBMC'
# rna_adata = anndata.read_h5ad(f'filter_data/{dataset}/GSE243917_genesXcells_ALL_batch.h5ad')
# atac_data = anndata.read_h5ad(f'filter_data/{dataset}/GSE243917_peaksXcells_ALL_batch.h5ad')

rna_adata = anndata.read_h5ad(f'filter_data/{dataset}/10x-Multiome-Pbmc10k-RNA.h5ad')
atac_data = anndata.read_h5ad(f'filter_data/{dataset}/10x-Multiome-Pbmc10k-ATAC.h5ad')

TF_name = '../data/GRN/data_bulk/TFName.txt'
TF_name = open(TF_name, 'r').readlines()
for i in range(len(TF_name)):
    TF_name[i] = TF_name[i].replace('\n', '')

output_path = f'filter_data/{dataset}/'
adata_multiomics_processing([rna_adata, atac_data],
                            dataset,
                            TF_name,
                            3000,
                            0.01)

rna_adata = anndata.read_h5ad(f'filter_data/{dataset}/RNA_filter.h5ad')
five_fold_split_dataset(rna_adata, f'filter_data/{dataset}/fold_split_info.csv')


