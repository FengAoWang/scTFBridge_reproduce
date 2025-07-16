import os.path
import torch
import torch.nn.functional as F
import sys
# sys.path.append('/home/wfa/project/single_cell_multimodal')
from sklearn.cluster import KMeans
from model.scTFBridge import scTFBridge, explainModel
import anndata
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset.dataset import scMultiDataset
from torch.utils.data import DataLoader
import ast
import time
from model.pathexplainer import PathExplainerTorch
import torch.multiprocessing as mp
import pybedtools

sceqtl_mapping = [
    {"Cell type": "CD14 Mono", "eqtl_path": "Perez2022Science_7.sig_qtl.tsv"},
    {"Cell type": "CD8 Naive", "eqtl_path": "Perez2022Science_17.sig_qtl.tsv"},
    {"Cell type": "CD4 TCM", "eqtl_path": "YAZAR2022Sci_5.sig_qtl.tsv"},
    {"Cell type": "CD4 TEM", "eqtl_path": "YAZAR2022Sci_1.sig_qtl.tsv"},
    {"Cell type": "Memory B", "eqtl_path": "YAZAR2022Sci_9.sig_qtl.tsv"},
    {"Cell type": "CD8 TEM_1", "eqtl_path": "YAZAR2022Sci_4.sig_qtl.tsv"},
    {"Cell type": "cDC", "eqtl_path": "Perez2022Science_3.sig_qtl.tsv"},
    {"Cell type": "CD4 Naive", "eqtl_path": "Perez2022Science_15.sig_qtl.tsv"},
]


# loading GEX ATAC adata
dataset_name = 'human_PBMC'
cell_key = 'cell_type'
training_mode = 'debug0620'
cell_type = 'CD4 Naive'
eqtl_path = next((item["eqtl_path"] for item in sceqtl_mapping if item["Cell type"] == cell_type), None)

single_cell_eqtl_path = f'../data/eqtl/single_cell_eqtl/{eqtl_path}'

single_cell_gene_eqtl = pd.read_csv(single_cell_eqtl_path, sep='\t')
single_cell_gene_eqtl.rename(columns={'geneName': 'GeneSymbol'}, inplace=True)


gex_data = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/RNA_filter.h5ad')
atac_adata = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/ATAC_filter.h5ad')
TF_adata = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/TF_filter.h5ad')

fold_split_info = pd.read_csv(f'../data/filter_data/{dataset_name}/fold_split_info.csv')
mask = pd.read_csv(f'../data/{dataset_name}_TF_Binding/TF_binding.txt', sep='\t', header=None).values


if 'batch' in gex_data.obs.columns:
    batches_info = gex_data.obs['batch'].values.tolist()
else:
    batches_info = [0 for _ in range(gex_data.shape[0])]

one_hot_encoded_batches = pd.get_dummies(batches_info, prefix='batch', dtype=float)

batch_dims = one_hot_encoded_batches.shape[1]


#   common_eqtl


# cellular_gene_eqtl = pd.read_csv('../data/eqtl/2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt', sep='\t')


gene_feature = gex_data.var.index.values
gene_eqtl_feature = single_cell_gene_eqtl['GeneSymbol'].unique()
common_features = intersection = list(set(gene_feature) & set(gene_eqtl_feature))
# print(common_features)
print(len(common_features))

# gene1_eqtl = gene_eqtl[gene_eqtl['GeneSymbol'] == 'PER3']

# gene1_eqtl.sort_values(by='SNPPos', inplace=True)
# gene1_eqtl.to_csv('PER3_eqtl.csv')


# print(gex_data.var)


# print(gex_data.var)
# print(atac_adata.var)
# atac_peak_feature = atac_adata.var.index.values.tolist()
# for i in range(len(atac_peak_feature)):
#     atac_peak_feature[i] = atac_peak_feature[i].replace(':', '-')
# mean_atac_average = atac_adata.X.toarray().mean(axis=0)
# print(atac_peak_feature)
# peak_chr = [feature.split('-')[0] for feature in atac_peak_feature]
# peak_start = [int(feature.split('-')[1]) for feature in atac_peak_feature]
# peak_end = [int(feature.split('-')[2]) for feature in atac_peak_feature]
# print(atac_peak_feature)


# print(batches_info)

id_list = []
for i in range(5):
    train_id = ast.literal_eval(fold_split_info.loc[i, 'train_id'])
    val_id = ast.literal_eval(fold_split_info.loc[i, 'validation_id'])
    test_id = ast.literal_eval(fold_split_info.loc[i, 'test_id'])
    id_list.append([train_id, val_id, test_id])

# print(ast.literal_eval(fold_split_info.loc[0, 'train_id']))
kmeans = KMeans(n_clusters=3)
dim1 = gex_data.X.shape[1]
dim2 = atac_adata.X.shape[1]


def get_all_representation(data_loader, model: scTFBridge):
    joint_share_representations = torch.Tensor([])
    rna_private_representations = torch.Tensor([])
    atac_private_representations = torch.Tensor([])
    rna_share_representations = torch.Tensor([])
    atac_share_representations = torch.Tensor([])

    all_recon_rna = torch.Tensor([])
    all_recon_atac = torch.Tensor([])

    with torch.no_grad():
        with tqdm(data_loader, unit='batch') as tepoch:
            for batch, data, batch_id in enumerate(tepoch):
                rna_data, atac_data, batch_id = data
                rna_data = rna_data.cuda()
                atac_data = atac_data.cuda()
                batch_id = batch_id.cuda()

                output = model([rna_data, atac_data], batch_id)
                share_embedding = output['share_embedding'].cpu()
                rna_embedding = output['rna_private_embedding'].cpu()
                atac_embedding = output['atac_private_embedding'].cpu()
                rna_share = output['rna_share_embedding'].cpu()
                atac_share = output['atac_share_embedding'].cpu()

                joint_share_representations = torch.cat((joint_share_representations, share_embedding), dim=0)
                rna_private_representations = torch.cat((rna_private_representations, rna_embedding), dim=0)
                atac_private_representations = torch.cat((atac_private_representations, atac_embedding), dim=0)
                rna_share_representations = torch.cat((rna_share_representations, rna_share), dim=0)
                atac_share_representations = torch.cat((atac_share_representations, atac_share), dim=0)

                recon_rna, recon_atac = model.cross_modal_generation([rna_data, atac_data], batch_id)
                all_recon_rna = torch.cat((all_recon_rna, recon_rna.cpu()), dim=0)
                all_recon_atac = torch.cat((all_recon_atac, recon_atac.cpu()), dim=0)
    output = {
        'joint_share_representations': joint_share_representations.detach().cpu().numpy(),
        'rna_private_representations': rna_private_representations.detach().cpu().numpy(),
        'atac_private_representations': atac_private_representations.detach().cpu().numpy(),
        'rna_share_representations': rna_share_representations.detach().cpu().numpy(),
        'atac_share_representations': atac_share_representations.detach().cpu().numpy(),
    }
    return output


def get_sample_data(dataloader, num_samples=4):
    samples_rna, samples_atac, sample_batch = [], [], []
    with tqdm(dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            rna_data, atac_data, TF_data, batch_id = data
            samples_rna.append(rna_data)
            samples_atac.append(atac_data)
            sample_batch.append(batch_id)
            if len(samples_rna) * rna_data.shape[0] >= num_samples:
                break
        rna_data = torch.cat(samples_rna, dim=0)
        atac_data = torch.cat(samples_atac, dim=0)
        batch_id = torch.cat(sample_batch, dim=0)
        all_data = torch.cat((rna_data, atac_data, batch_id), dim=1).cuda()
        rna_data = rna_data.cuda()
        return all_data, rna_data



gene_ids = []

gex_data_var = gex_data.var.copy()
for gene in common_features:
    # TSS = int(gene_eqtl[gene_eqtl['GeneSymbol'] == gene]['GenePos'].tolist()[0])
    gene_id = gex_data_var.index.get_loc(gene)
    gene_ids.append(gene_id)
    # print(gene, gene_id)
gene_ids = sorted(gene_ids)
# print(gene_ids)


def compute_peak_score(fold, cuda_device_id, cell_type):
    torch.cuda.set_device(cuda_device_id)
    mask_tensor = torch.tensor(mask).float()

    sc_multi_demo = scTFBridge([dim1, dim2], [1024], [1024],
                               128, 1, ['gaussian', 'bernoulli'], batch_dims, 1, mask_tensor)
    model_dict = torch.load(f'model_dict/sc_multi_{dataset_name}_{training_mode}_fold{fold}.pt', map_location='cpu')
    sc_multi_demo.load_state_dict(model_dict)
    sc_multi_demo.cuda()
    sc_multi_demo.eval()
    sc_multi_demo.latent_mode = 'latent_z'

    train_gex_adata = gex_data[id_list[fold][0]].copy()
    validation_gex_adata = gex_data[id_list[fold][1]].copy()
    test_gex_adata = gex_data[id_list[fold][2]].copy()

    train_TF_adata = TF_adata[id_list[fold][0]].copy()
    validation_TF_adata = TF_adata[id_list[fold][1]].copy()
    test_TF_adata = TF_adata[id_list[fold][2]].copy()


    if cell_type != 'cellular':
        train_gex_adata = train_gex_adata[train_gex_adata.obs['cell_type'] == cell_type]
        test_gex_adata = test_gex_adata[test_gex_adata.obs['cell_type'] == cell_type]

    train_atac_adata = atac_adata[id_list[fold][0]].copy()
    validation_atac_adata = atac_adata[id_list[fold][1]].copy()
    test_atac_adata = atac_adata[id_list[fold][2]].copy()
    if cell_type != 'cellular':
        train_atac_adata = train_atac_adata[train_atac_adata.obs['cell_type'] == cell_type]
        test_atac_adata = test_atac_adata[test_atac_adata.obs['cell_type'] == cell_type]

    train_batch_info = one_hot_encoded_batches.iloc[id_list[fold][0]]
    validation_batch_info = one_hot_encoded_batches.iloc[id_list[fold][1]]
    test_batch_info = one_hot_encoded_batches.iloc[id_list[fold][2]]

    train_gex_data_np = train_gex_adata.X.toarray()
    train_atac_data_np = train_atac_adata.X.toarray()
    train_TF_adata_np = train_TF_adata.X.toarray()

    validation_gex_data_np = validation_gex_adata.X.toarray()
    validation_atac_data_np = validation_atac_adata.X.toarray()
    validation_TF_adata_np = validation_TF_adata.X.toarray()

    test_gex_data_np = test_gex_adata.X.toarray()
    test_atac_data_np = test_atac_adata.X.toarray()
    test_TF_adata_np = test_TF_adata.X.toarray()

    sc_dataset = scMultiDataset([train_gex_data_np, train_atac_data_np, train_TF_adata_np], train_batch_info)
    val_sc_dataset = scMultiDataset([validation_gex_data_np, validation_atac_data_np, validation_TF_adata_np], validation_batch_info)
    test_sc_dataset = scMultiDataset([test_gex_data_np, test_atac_data_np, test_TF_adata_np], test_batch_info)

    sc_train_dataloader = DataLoader(sc_dataset, batch_size=5, shuffle=True, num_workers=8, pin_memory=True)
    sc_test_dataloader = DataLoader(test_sc_dataset, batch_size=5, shuffle=True, num_workers=8, pin_memory=True)

    explain_model = explainModel(sc_multi_demo, 'ATAC2RNA', 128, gene_ids)
    explain_model.eval()
    explain_model.cuda()

    all_data, train_rna_data = get_sample_data(sc_train_dataloader, num_samples=5)
    new_data, test_rna_data = get_sample_data(sc_test_dataloader, num_samples=50)

    start_time = time.time()

    # 创建SHAP解释器 n to n
    # explainer = shap.DeepExplainer(explain_model, all_data)
    # shap_values = explainer.shap_values(new_data, check_additivity=False)
    #
    # print('time used: ', time.time() - start_time)
    # shap_values = np.array(shap_values)
    # shap_values = shap_values[:, :, dim1:dim1+dim2]
    # shap_values = np.abs(shap_values)
    # shap_values = shap_values.mean(axis=1)
    # shap_values = np.abs(shap_values).mean(axis=0)

    all_attributions = []
    for gene in gene_ids:
        explain_model.gene_ids = [gene]
        def model_loss_wrapper(z):
            rna_recon = explain_model(z)
            return rna_recon
            # return F.mse_loss(rna_recon, test_rna_data[:, [gene]], reduction='none').mean(1).view(-1, 1)

        explainer = PathExplainerTorch(model_loss_wrapper)

        baseline_data = all_data[0, :]  # define a baseline, in this case the zeros vector
        baseline_data = torch.zeros_like(baseline_data)
        baseline_data.requires_grad = True
        attributions = explainer.attributions(new_data,
                                              baseline=baseline_data,
                                              num_samples=50,
                                              use_expectation=False)
        # 将attributions合并到all_attributions
        attributions = attributions.cpu().detach()
        if len(all_attributions) == 0:
            all_attributions = attributions.unsqueeze(0)  # 添加第一个维度
        else:
            all_attributions = torch.cat((all_attributions, attributions.unsqueeze(0)), dim=0)
        torch.cuda.empty_cache()  # Clear cache to reduce memory consumption

    all_attributions = all_attributions.numpy()
    shap_values = np.array(all_attributions)
    shap_values = shap_values[:, :, dim1:dim1+dim2]
    shap_values = np.abs(shap_values)
    shap_values = shap_values.mean(axis=1)


    # distance = np.array(distance)
    # distance = np.expand_dims(distance, axis=0)

    # shap_values = shap_values.reshape(-1)

    # distance = distance.reshape(-1)

    # cis_score = mean_atac_average[:, np.newaxis] * shap_values
    # cis_score = cis_score.T
    # df = pd.DataFrame({'feature': atac_peak_feature, 'peak start': peak_start,
    #                    'peak end': peak_end, 'chr': peak_chr})
    # # df.sort_values(by='value', ascending=False, inplace=True)
    if not os.path.exists(f'peak_shap/{dataset_name}'):
        os.makedirs(f'peak_shap/{dataset_name}')
    np.save(f'peak_shap/{dataset_name}/{cell_type}_atac_peak_shap_no_loss_value_fold{fold}.npy', shap_values)
    # df.to_csv(f'peak_shap/atac_feature_values_fold{fold}.csv', index=False)

    # explainer = IntegratedGradients(explain_model)
    # with torch.no_grad():
    #     with tqdm(test_dataloader, unit='batch') as tepoch:
    #         for batch, data in enumerate(tepoch):
    #             rna_data, atac_data, batch_id = data
    #             rna_data = rna_data.cuda()
    #             atac_data = atac_data.cuda()
    #             multi_data = torch.concat((rna_data, atac_data), dim=1)
    #             attr_info = explainer.attribute(multi_data)
    #             print(attr_info)
    # # 计算SHAP值
    # shap_values = explainer.shap_values(all_data)
    # print(shap_values)

    # train_output_embedding = get_all_representation(test_dataloader, sc_multi_demo)
    # test_gex_adata.obsm['sc_rna_private'] = train_output_embedding['rna_private_representations']
    #
    # sc.pp.neighbors(test_gex_adata, use_rep='sc_rna_private')
    # sc.tl.umap(test_gex_adata)
    # sc.tl.leiden(test_gex_adata)
    # sc.tl.louvain(test_gex_adata)
    #
    # # 假设你已经有了 PCA 降维后的数据
    # kmeans = KMeans(n_clusters=5, random_state=0).fit(test_gex_adata.obsm['sc_rna_private'])
    # test_gex_adata.obs['kmeans'] = kmeans.labels_.astype(str)
    #
    # sc.pl.umap(test_gex_adata, color="batch", save=f'_train_batch_{fold}.pdf')
    # sc.pl.umap(test_gex_adata, color="cell_type", save=f'_train_cell_type_{fold}.pdf')
    # sc.pl.umap(test_gex_adata, color="leiden", save=f'_train_leiden_{fold}.pdf')
    # sc.pl.umap(test_gex_adata, color="louvain", save=f'_train_louvain_{fold}.pdf')
    # sc.pl.umap(test_gex_adata, color="kmeans", save=f'_train_kmeans_{fold}.pdf')


def multiprocessing_train_fold(folds, function, func_args_list):
    processes = []
    return_queue = mp.Queue()

    for i in range(folds):
        p = mp.Process(target=function, args=func_args_list[i])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# cell = 'Naive B'
device_id_list = [7, 1, 6, 6, 7]
functions = [(fold, device_id_list[fold], cell_type) for fold in range(5)]

multiprocessing_train_fold(5, compute_peak_score, functions)


def cellular_compute_TSS_distance(cell_type, gene_eqtl, dataset_name, gex_data, atac_adata):
    RE_region = pybedtools.example_bedtool(os.path.abspath(f'../data/{dataset_name}_TF_Binding/Region.bed')).sort()

    gene_feature = gex_data.var.index.values
    atac_peak_feature = atac_adata.var.index.values.tolist()
    peak_mean_access = atac_adata.X.mean(axis=0).tolist()
    # print(peak_mean_access)
    peak_info = atac_adata.var
    # print(peak_info)
    peak_info['mean_access'] = peak_mean_access[0]
    peak_info['peak'] = peak_info.index
    # print(peak_mean_access)
    gene_eqtl_feature = gene_eqtl['GeneSymbol'].unique()
    common_features = list(set(gene_feature) & set(gene_eqtl_feature))

    gene_eqtl = gene_eqtl[gene_eqtl['GeneSymbol'].isin(common_features)]
    gene_eqtl_TSS = gene_eqtl[['SNPChr', 'GenePos', 'GeneSymbol']]
    gene_eqtl_TSS.drop_duplicates(subset=['SNPChr', 'GeneSymbol', 'GenePos'], inplace=True)
    gene_eqtl_TSS['GenePos+'] = gene_eqtl_TSS['GenePos'] + 1
    gene_eqtl_bed = gene_eqtl_TSS[['SNPChr', 'GenePos', 'GenePos+', 'GeneSymbol']]
    gene_eqtl_bed['SNPChr'] = 'chr' + gene_eqtl_bed['SNPChr'].astype(str)
    gene_eqtl_bed.sort_values(['SNPChr'], inplace=True)
    gene_eqtl_bed.to_csv(f'cis_regulatory/{cell_type}_gene_eqtl_bed.bed', index=False, header=False, sep='\t')
    print(gene_eqtl_bed)

    gene_eqtl_bed = pybedtools.example_bedtool(os.path.abspath(f'cis_regulatory/{cell_type}_gene_eqtl_bed.bed')).sort()
    # gene_eqtl_bed = gene_eqtl_bed.filter(lambda x: x.chrom == chr_name)

    eqtl_pos = gene_eqtl[['SNPChr', 'SNPPos', 'GeneSymbol']]
    eqtl_pos['SNPPos+'] = eqtl_pos['SNPPos'] + 1
    eqtl_pos['SNPChr'] = 'chr' + eqtl_pos['SNPChr'].astype(str)
    eqtl_pos_bed = eqtl_pos[['SNPChr', 'SNPPos', 'SNPPos+', 'GeneSymbol']]
    eqtl_pos_bed.to_csv(f'cis_regulatory/{cell_type}_eqtl_pos_bed.bed', index=False, header=False, sep='\t')
    eqtl_pos_bed = pybedtools.example_bedtool(os.path.abspath(f'cis_regulatory/{cell_type}_eqtl_pos_bed.bed')).sort()
    print(len(eqtl_pos_bed))
    # print(eqtl_pos_bed)
    # print(gene_eqtl_bed)
    # print(gene_eqtl_TSS)
    print('tss distance')
    closest = RE_region.closest(gene_eqtl_bed, d=True, k=len(common_features))
    closest.saveas(f'cis_regulatory/{cell_type}_gene_RE_distance.bed')
    print(len(closest))
    print('eqtl overlap')
    over_lap = RE_region.intersect(eqtl_pos_bed, wa=True, wb=True)
    over_lap.saveas(f'cis_regulatory/{cell_type}_RE_eqtl_overlap.bed')


gene_eqtl = pd.read_csv('../data/eqtl/2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt',
                        sep='\t')
gene_eqtl = gene_eqtl[['GeneSymbol', 'GenePos', 'SNPChr', 'SNPPos']]
gene_eqtl.drop_duplicates(subset=['GeneSymbol'], inplace=True)

# gene_eqtl.drop_duplicates(subset=['SNPPos'], inplace=True)
# gene_eqtl.rename(columns={'SNPChr': 'GeneChr'}, inplace=True)
# gene_eqtl.to_csv('gene_genome_info.csv', index=False)
print(gene_eqtl)


single_cell_gene_eqtl = pd.read_csv(single_cell_eqtl_path, sep='\t')

single_cell_gene_eqtl.rename(columns={'geneName': 'GeneSymbol'}, inplace=True)
single_cell_gene_eqtl.rename(columns={'chrom': 'SNPChr'}, inplace=True)
single_cell_gene_eqtl.rename(columns={'position': 'SNPPos'}, inplace=True)
# 假设 gene_eqtl 中 GeneSymbol 是唯一的
mapping = gene_eqtl.set_index('GeneSymbol')['GenePos']
single_cell_gene_eqtl['GenePos'] = single_cell_gene_eqtl['GeneSymbol'].map(mapping)
single_cell_gene_eqtl.dropna(subset=['GenePos'], inplace=True)
single_cell_gene_eqtl['GenePos'] = single_cell_gene_eqtl['GenePos'].astype(int)
print(single_cell_gene_eqtl)
# single_cell_gene_eqtl.rename(columns={'geneName': 'GeneSymbol'}, inplace=True)


gex_data = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/RNA_filter.h5ad')
atac_adata = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/ATAC_filter.h5ad')

cellular_compute_TSS_distance(cell_type, single_cell_gene_eqtl, dataset_name,  gex_data, atac_adata)

