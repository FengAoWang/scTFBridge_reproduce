import pandas as pd
import numpy as np
import anndata

peak_score = pd.read_csv('../../motif_scan_results_demo/pbmc_motif_TF_scan_output.txt', sep='\t')
# peak_score.to_csv('motif_scan_results.csv')
print(peak_score.columns)
df = pd.DataFrame([peak_score.columns.tolist()]).T
df.to_csv('col.csv')
peak_score.sort_values(by=['Peak Score'], ascending=False, inplace=True)
print(peak_score)
print(peak_score['Peak Score'].unique())
# atac_data = anndata.read_h5ad('filter_data/human_PBMC/filter_10x-Multiome-Pbmc10k-ATAC.h5ad')
# atac_data_var = atac_data.var.copy()
# print(atac_data_var)
# atac_data_var['chrom'] = atac_data_var.index.str.split(':').str[0]
# 
# temp_split = pd.DataFrame(list(atac_data_var.index.str.split(':')))
# 
# #
# # print(temp_split)
# # 第二步：按'-'分割并取第一部分
# chorm = pd.DataFrame(atac_data_var.index.str.split(':').str[0])
# start = temp_split[1].str.split('-').str[0].astype(int)
# end = temp_split[1].str.split('-').str[1].astype(int)
# 
# new_df = pd.concat([chorm, start, end], axis=1)
# new_df.columns = ['chrom', 'chromStart', 'chromEnd']
# print(new_df.head())
# print(new_df['chrom'].unique())
# 
# # 筛选只保留包含'chr'的行
# valid_chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
# filtered_df = new_df[new_df['chrom'].isin(valid_chromosomes)]
# 
# filtered_df.reset_index(inplace=True)
# filtered_df['Strand'] = 0
# filtered_df = filtered_df.iloc[:5, :]
# print(filtered_df)
# 
# filtered_df.to_csv('../../homer/pbmc_peak_demo.bed', sep='\t', header=False, index=False)

# atac_data_var['chromStart'] = temp_split[1].str.split('-').str[0].astype(int)
# atac_data_var['chromEnd'] = temp_split[1].str.split('-').str[1].astype(int)
# print(atac_data_var)
# new_df = atac_data_var[['chrom', 'chromStart', 'chromEnd']]
# print(new_df)


# chr1 = atac_data_var[atac_data_var['chr'] == 'chr1']


# data1 = pd.read_csv('GRN/data_bulk/Primary_TF_RE_chr1.txt', sep='\t')
# col = data1.columns.tolist()
# col[0] = 'peak'
# data1.columns = col
# # data1.sort_values(by='peak', inplace=True)
# # print(data1)
# data1['chr'] = data1['peak'].str.split(':').str[0]
# temp_split = data1['peak'].str.split(':', expand=True)
# print(temp_split)
#
# # 第二步：按'-'分割并取第一部分
# data1['start'] = temp_split[1].str.split('-', expand=True)[0].astype(int)
# data1['end'] = temp_split[1].str.split('-', expand=True)[1].astype(int)
#
# # print(data1['peak'].str.split(':').str[1].str.split('-')[0])
# # print(data1['peak'].str.split(':').str[1].str.split('-')[1])
# data1.sort_values(by='start', inplace=True)
# print(data1)


# data2 = pd.read_csv('GRN/data_bulk/Primary_TF_RE_chr2.txt', sep='\t')
#
# print(data1)


