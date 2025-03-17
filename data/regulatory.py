import pandas as pd
import scipy.io
import anndata

# data = 'eqtl/PECA/TFTG_corr_human.mat'
# mat_data = scipy.io.loadmat(data)
# print('TFTG', mat_data)
#
# print(mat_data.keys())
# print(len(mat_data['List']))
# print(mat_data['List'])
# print(len(mat_data['TFName']))

TF_TG = pd.read_csv('eqtl/PECA/RE_gene_corr_hg19.bed', sep='\t')

print(TF_TG)
data = TF_TG.iloc[:, 3].unique()
print(data)
print(len(data))
# TF_names = TF_TG['TF (regulator)'].unique()
# print(TF_names)
#
# TF_names = TF_TG['Gene (target)'].unique()
# # TF_names = open('eqtl/PECA/TFName_human.txt', 'r').readlines()
# # for i in range(len(TF_names)):
# #     TF_names[i] = TF_names[i].strip('\n')
# #
# gex_data = anndata.read_h5ad('filter_data/human_PBMC/filter_10x-Multiome-Pbmc10k-RNA.h5ad')
# gex_name = gex_data.var_names.tolist()
# print(gex_name)
#
# set1 = set(gex_name)
# set2 = set(TF_names)
#
# common_features = list(set1.intersection(set2))
# common_features.sort()
#
# print(common_features)
# print(len(common_features))
