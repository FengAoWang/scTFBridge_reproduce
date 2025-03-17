import os.path
import torch
import torch.nn.functional as F
import shap
import sys
sys.path.append('/home/wfa/project/single_cell_multimodal')
from sklearn.cluster import KMeans
from model.scTFBridge import scMulti, explainModelLatentZ, explainModelLatentTF
import anndata
import scanpy as sc
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import episcanpy.api as epi
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset.dataset import scMultiDataset
from torch.utils.data import DataLoader
import ast
import shap
from captum.attr import DeepLiftShap, IntegratedGradients
import hotspot
import matplotlib.pyplot as plt
import time
from sklearn.feature_selection import mutual_info_classif
from model.pathexplainer import PathExplainerTorch
import random
import torch.multiprocessing as mp



def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')


set_seed(3407)
# loading GEX ATAC adata
dataset_name = 'human_PBMC'
print(dataset_name)
gex_data = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/RNA_filter.h5ad')
atac_adata = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/ATAC_filter.h5ad')
TF_adata = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/TF_filter.h5ad')

TF_length = TF_adata.var.shape[0]


fold_split_info = pd.read_csv(f'../data/filter_data/{dataset_name}/fold_split_info.csv')
mask = pd.read_csv(f'../data/{dataset_name}_TF_Binding/TF_binding.txt', sep='\t', header=None).values


if 'batch' in gex_data.obs.columns:
    batches_info = gex_data.obs['batch'].values.tolist()
else:
    batches_info = [0 for _ in range(gex_data.shape[0])]

one_hot_encoded_batches = pd.get_dummies(batches_info, prefix='batch', dtype=float)

batch_dims = one_hot_encoded_batches.shape[1]
dim1 = gex_data.X.shape[1]
dim2 = atac_adata.X.shape[1]


id_list = []
for i in range(5):
    train_id = ast.literal_eval(fold_split_info.loc[i, 'train_id'])
    val_id = ast.literal_eval(fold_split_info.loc[i, 'validation_id'])
    test_id = ast.literal_eval(fold_split_info.loc[i, 'test_id'])
    id_list.append([train_id, val_id, test_id])


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
        return all_data


# def get_latent_z(model:scMulti, dataloader, num_samples=4):
#     share_embedding, private_embedding, sample_batch = [], [], []
#     rna_input = []
#     with tqdm(dataloader, unit='batch') as tepoch:
#         for batch, data in enumerate(tepoch):
#             rna_data, atac_data, TF_data, batch_id = data
#             rna_data = rna_data.cuda()
#             atac_data = atac_data.cuda()
#             TF_data = TF_data.cuda()
#             batch_id = batch_id.cuda()
#
#             output = model.ATACShareEncoder(atac_data)
#             # output = output + TF_data
#
#             share_embedding.append(output)
#             sample_batch.append(batch_id)
#             rna_input.append(rna_data)
#             if len(share_embedding) * rna_data.shape[0] >= num_samples:
#                 break
#
#         share_data = torch.cat(share_embedding, dim=0)
#         batch_id = torch.cat(sample_batch, dim=0)
#         all_data = torch.cat((share_data, batch_id), dim=1)
#         rna_input = torch.cat(rna_input, dim=0)
#         # all_data = torch.cat((all_data, rna_input), dim=1).cuda()
#         return all_data, rna_input

def get_latent_z(model, dataloader, modal, num_samples=4):
    share_embedding, private_embedding, sample_batch = [], [], []
    input = []
    with tqdm(dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            rna_data, atac_data, TF_data, batch_id = data
            rna_data = rna_data.cuda()
            atac_data = atac_data.cuda()
            TF_data = TF_data.cuda()
            batch_id = batch_id.cuda()

            output = model([rna_data, atac_data, TF_data], batch_id)

            share_embedding.append(output[f'share_embedding'])
            private_embedding.append(output[f'{modal}_private_embedding'])
            sample_batch.append(batch_id)
            if modal == 'rna':
                input.append(rna_data)
            else:
                input.append(atac_data)

            if len(share_embedding) * atac_data.shape[0] >= num_samples:
                break

        share_data = torch.cat(share_embedding, dim=0)
        private_data = torch.cat(private_embedding, dim=0)
        batch_id = torch.cat(sample_batch, dim=0)
        all_data = torch.cat((share_data, private_data, batch_id), dim=1)
        rna_input = torch.cat(input, dim=0)
        # all_data = torch.cat((all_data, rna_input), dim=1).cuda()
        return all_data, rna_input


def compute_TF_value(fold, device_id, cell_type):
    torch.cuda.set_device(device_id)
    mask_tensor = torch.tensor(mask).float()

    sc_multi_demo = scMulti([dim1, dim2], [1024], [1024],
                            128, 1, ['gaussian', 'bernoulli'], batch_dims, 1, mask_tensor)
    model_dict = torch.load(f'model_dict/sc_multi_{dataset_name}_fold{fold}.pt', map_location='cpu')
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
        print(cell_type)
        train_gex_adata = train_gex_adata[train_gex_adata.obs['cell_type'] == cell_type]
        test_gex_adata = test_gex_adata[test_gex_adata.obs['cell_type'] == cell_type]

    train_atac_adata = atac_adata[id_list[fold][0]].copy()
    validation_atac_adata = atac_adata[id_list[fold][1]].copy()
    test_atac_adata = atac_adata[id_list[fold][2]].copy()
    if cell_type != 'cellular':
        print(cell_type)
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
    sc_test_dataloader = DataLoader(test_sc_dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True)

    train_latent_z, train_rna = get_latent_z(sc_multi_demo, sc_train_dataloader, num_samples=20, modal='rna')
    baseline_latent_z = torch.zeros_like(train_latent_z)
    test_latent_z, test_rna = get_latent_z(sc_multi_demo, sc_test_dataloader, num_samples=300, modal='rna')
    start_time = time.time()
    all_attributions = []
    explain_model = explainModelLatentZ(sc_multi_demo, 'rna', 128, 0)
    explain_model.eval()
    explain_model.cuda()

    for dim in range(dim1):
        explain_model.dimension_num = dim


        def model_loss_wrapper(z):
            rna_recon = explain_model(z)
            return F.mse_loss(rna_recon, test_rna[:, [dim]], reduction='none').mean(1).view(-1, 1)


        explainer = PathExplainerTorch(model_loss_wrapper)

        baseline_data = train_latent_z[0, :]  # define a baseline, in this case the zeros vector
        baseline_data = torch.zeros_like(baseline_data)
        baseline_data.requires_grad = True
        attributions = explainer.attributions(test_latent_z,
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

        # print(dim, attributions.shape)

        #   SHAP value
        # explainer = shap.DeepExplainer(explain_model, train_latent_z)
        # attributions = explainer.shap_values(test_latent_z)

        # explainer = shap.DeepExplainer(explain_model, train_latent_z)
        #
        # # with torch.no_grad():
        # shap_values = explainer.shap_values(test_latent_z)
    all_attributions = all_attributions.numpy()
    shap_values = np.array(all_attributions)
    shap_values = shap_values[:, :, :TF_length]
    mean_shap_values = np.mean(np.abs(shap_values), axis=1)
    if not os.path.exists(f'TF_shap/{dataset_name}'):
        os.makedirs(f'TF_shap/{dataset_name}')
    np.save(f'TF_shap/{dataset_name}/{cell_type}_TF_shap_value_fold{fold}_v2.npy', mean_shap_values)
    np.save(f'TF_shap/{dataset_name}/{cell_type}_all_sample_TF_shap_value_fold{fold}.npy', shap_values)
    print(mean_shap_values.shape)

    # shap_share = shap_values[:, :128]
    # shap_private = shap_values[:, 128:256]
    # # shap_input = shap_values[:, 256:]
    #
    # shap_share_sum = np.sum(np.abs(shap_share).mean(0))
    # shap_private_sum = np.sum(np.abs(shap_private).mean(0))
    # # shap_input_sum = np.sum(np.abs(shap_input).mean(axis=0))
    #
    # print('share', shap_share_sum)
    # print('private', shap_private_sum)
    # # print('input', shap_input_sum)
    #
    # shap.summary_plot(shap_values[:, :256], test_latent_z[:, :256].detach().cpu().numpy())

    print('time used: ', time.time() - start_time)


def multiprocessing_train_fold(folds, function, func_args_list):
    processes = []
    return_queue = mp.Queue()

    for i in range(folds):
        p = mp.Process(target=function, args=func_args_list[i])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


cell_list = gex_data.obs['cell_type'].unique().tolist()
print(cell_list)
for cell in cell_list:
    device_id_list = [3, 6, 5, 0, 1]
    functions = [(fold, device_id_list[fold], cell) for fold in range(5)]

    multiprocessing_train_fold(5, compute_TF_value, functions)
