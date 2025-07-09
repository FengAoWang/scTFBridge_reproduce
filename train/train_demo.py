import os.path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import sys
sys.path.append('/data2/wfa/project/single_cell_multimodal')
import scanpy as sc
import anndata
from dataset.dataset import scMultiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.scTFBridge import scTFBridge, calculate_r_squared_torch, calculate_pcc_torch
import time
import episcanpy.api as epi
import numpy as np
import scipy
import pandas as pd
from sklearn import metrics
import random
from sklearn.metrics import mutual_info_score
import pyinform
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score
import ast
import torch.multiprocessing as mp
import os

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

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



# 计算两个嵌入矩阵之间的互信息
def compute_mutual_information(embedding1, embedding2, bins=30):
    # 将嵌入矩阵投影到一维
    embedding1_flattened = embedding1.flatten()
    embedding2_flattened = embedding2.flatten()

    # 使用直方图将数据分箱
    counts1, _ = np.histogram(embedding1_flattened, bins=bins)
    counts2, _ = np.histogram(embedding2_flattened, bins=bins)

    # 计算互信息
    mi = mutual_info_score(counts1, counts2)

    return mi


# 计算两个嵌入张量之间的互信息
def compute_mutual_information_tensor(embedding1, embedding2, bins=30):
    # 将PyTorch张量转换为NumPy数组
    embedding1_np = embedding1.detach().cpu().numpy()
    embedding2_np = embedding2.detach().cpu().numpy()

    # 将嵌入矩阵展平
    embedding1_flattened = embedding1_np.flatten()
    embedding2_flattened = embedding2_np.flatten()

    # 使用直方图将数据分箱
    counts1, _ = np.histogram(embedding1_flattened, bins=bins)
    counts2, _ = np.histogram(embedding2_flattened, bins=bins)

    # 计算互信息
    mi = mutual_info_score(counts1, counts2)

    return mi


set_seed(3407)




# loading GEX ATAC adata
dataset_name = 'human_PBMC'
training_mode = 'no_prior_strain'
cell_key = 'cell_type'

common_TG = pd.read_csv('/data2/ycx/LINGER/data/TG/common_TG_new.csv')['TG'].values.tolist()
# print('common TG', common_TG)

gex_data = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/RNA_filter.h5ad')
# gex_data = gex_data[:, gex_data.var_names.isin(common_TG)]
print(gex_data)


atac_adata = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/ATAC_filter.h5ad')
TF_adata = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/TF_filter.h5ad')

TF_length = TF_adata.var.shape[0]

fold_split_info = pd.read_csv(f'../data/filter_data/{dataset_name}/fold_split_info.csv')
mask = pd.read_csv(f'../data/filter_data/{dataset_name}/TF_binding/TF_binding.txt', sep='\t', header=None).values

if 'batch' in gex_data.obs.columns:
    batches_info = gex_data.obs['batch'].values.tolist()
else:
    batches_info = [0 for _ in range(gex_data.shape[0])]

# print(batches_info)

one_hot_encoded_batches = pd.get_dummies(batches_info, prefix='batch', dtype=float)

batch_dims = one_hot_encoded_batches.shape[1]


id_list = []
for i in range(5):
    train_id = ast.literal_eval(fold_split_info.loc[i, 'train_id'])
    val_id = ast.literal_eval(fold_split_info.loc[i, 'validation_id'])
    test_id = ast.literal_eval(fold_split_info.loc[i, 'test_id'])
    id_list.append([train_id, val_id, test_id])


# all_ARI = [0 for i in range(5)]
# all_NMI = [0 for i in range(5)]
# all_HOM = [0 for i in range(5)]
# all_AMI = [0 for i in range(5)]
# all_r2 = [0 for i in range(5)]
#
# all_atac_ARI = [0 for i in range(5)]
# all_atac_NMI = [0 for i in range(5)]
# all_atac_HOM = [0 for i in range(5)]
# all_atac_AMI = [0 for i in range(5)]


# for fold in range(5):
def scDM_training(fold: int, device_id: int):
    torch.cuda.set_device(device_id)

    train_gex_adata = gex_data[id_list[fold][0]].copy()
    validation_gex_adata = gex_data[id_list[fold][1]].copy()
    test_gex_adata = gex_data[id_list[fold][2]].copy()

    train_atac_adata = atac_adata[id_list[fold][0]].copy()
    validation_atac_adata = atac_adata[id_list[fold][1]].copy()
    test_atac_adata = atac_adata[id_list[fold][2]].copy()

    train_TF_adata = TF_adata[id_list[fold][0]].copy()
    validation_TF_adata = TF_adata[id_list[fold][1]].copy()
    test_TF_adata = TF_adata[id_list[fold][2]].copy()


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

    print(len(sc_dataset))
    print(sc_dataset[0][0], sc_dataset[0][1])
    print(sc_dataset[0][0].shape, sc_dataset[0][1].shape)

    sc_dataloader = DataLoader(sc_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_sc_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_sc_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

    dim1 = gex_data.X.shape[1]
    dim2 = atac_adata.X.shape[1]
    TF_dim = TF_adata.X.shape[1]

    mask_tensor = torch.tensor(mask).float()
    one_mask_tensor = torch.ones_like(mask_tensor)

    # mask_tensor_ = torch.ones_like(mask_tensor)
    # 计算最小值和最大值
    min_val = mask_tensor.min()
    max_val = mask_tensor.max()

    # 进行全局归一化
    normalized_mask_tensor = (mask_tensor - min_val) / (max_val - min_val)

    sc_multi_demo = scTFBridge([dim1, dim2], [1024], [1024],
                               TF_dim, 0.1, ['gaussian', 'bernoulli'], batch_dims, 1, 1, one_mask_tensor)

    epochs = 150
    best_val_loss = float('inf')
    patience = 10  # 设定 early stopping 的 patience
    no_improve_epochs = 0
    best_model_dict = sc_multi_demo.state_dict()

    sc_multi_demo.cuda()
    sc_multi_demo.train()
    
    optimizer = torch.optim.Adam(sc_multi_demo.parameters(), lr=1e-3)

    for epoch in range(epochs):
        train_total_loss = 0
        start_time = time.time()
        sc_multi_demo.train()
        with tqdm(sc_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                rna_data, atac_data, TF_data, batch_id = data
                rna_data = rna_data.cuda()
                atac_data = atac_data.cuda()
                TF_data = TF_data.cuda()
                batch_id = batch_id.cuda()

                optimizer.zero_grad()
                loss = sc_multi_demo.compute_loss([rna_data, atac_data, TF_data], batch_id)
                # print(loss)
                loss.backward()

                optimizer.step()
                train_total_loss += loss.item()
        train_total_loss /= len(sc_dataloader)

        sc_multi_demo.eval()
        val_total_loss = 0
        with torch.no_grad():
            with tqdm(val_dataloader, unit='batch') as tepoch:
                for batch, data in enumerate(tepoch):
                    rna_data, atac_data, TF_data, batch_id = data
                    rna_data = rna_data.cuda()
                    atac_data = atac_data.cuda()
                    TF_data = TF_data.cuda()
                    batch_id = batch_id.cuda()
                    loss = sc_multi_demo.compute_loss([rna_data, atac_data, TF_data], batch_id)
                    val_total_loss += loss.item()
            val_total_loss /= len(sc_dataloader)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_total_loss:.4f}, Val Loss: {val_total_loss:.4f}')
        # 早停机制
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            no_improve_epochs = 0
            best_model_dict = sc_multi_demo.state_dict()
            print(f'Best model saved with val loss: {best_val_loss:.4f}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print('Early stopping triggered')
                break
        print(f'time used: {time.time() - start_time:.4f}')

    #   start eval
    sc_multi_demo.load_state_dict(best_model_dict)
    torch.save(best_model_dict, f'model_dict/sc_multi_{dataset_name}_{training_mode}_fold{fold}.pt')
    torch.cuda.empty_cache()
    sc_multi_demo.eval()
    latent_representations = torch.Tensor([])
    rna_representations = torch.Tensor([])
    atac_representations = torch.Tensor([])
    rna_share_representations = torch.Tensor([])
    atac_share_representations = torch.Tensor([])

    all_recon_rna = torch.Tensor([])
    all_recon_atac = torch.Tensor([])

    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                rna_data, atac_data, TF_data, batch_id = data
                rna_data = rna_data.cuda()
                atac_data = atac_data.cuda()
                batch_id = batch_id.cuda()
                TF_data = TF_data.cuda()

                output = sc_multi_demo([rna_data, atac_data, TF_data], batch_id)
                share_embedding = output['share_embedding'].cpu()
                rna_embedding = output['RNA_private_embedding'].cpu()
                atac_embedding = output['ATAC_private_embedding'].cpu()
                rna_share = output['RNA_share_embedding'].cpu()
                atac_share = output['ATAC_share_embedding'].cpu()

                latent_representations = torch.cat((latent_representations, share_embedding), dim=0)
                rna_representations = torch.cat((rna_representations, rna_embedding), dim=0)
                atac_representations = torch.cat((atac_representations, atac_embedding), dim=0)
                rna_share_representations = torch.cat((rna_share_representations, rna_share), dim=0)
                atac_share_representations = torch.cat((atac_share_representations, atac_share), dim=0)

                recon_rna, recon_atac = sc_multi_demo.cross_modal_generation([rna_data, atac_data, TF_data], batch_id)
                all_recon_rna = torch.cat((all_recon_rna, recon_rna.cpu()), dim=0)
                all_recon_atac = torch.cat((all_recon_atac, recon_atac.cpu()), dim=0)

    all_share_embedding = torch.concat((latent_representations, rna_representations, atac_representations), dim=1)
    all_rna_embedding = torch.concat((latent_representations, rna_representations), dim=1)
    all_atac_embedding = torch.concat((latent_representations, atac_representations), dim=1)

    gex_data_recon = test_gex_adata.copy()
    atac_data_recon = test_atac_adata.copy()

    gex_data_recon.X = all_recon_rna.detach().cpu().numpy()
    atac_data_recon.X = all_recon_atac.detach().cpu().numpy()

    # latent_representation
    test_gex_adata.obsm["X_share"] = latent_representations.detach().cpu().numpy()  # 将潜在表示存储回adata用于后续分析
    test_gex_adata.obsm["X_rna"] = rna_representations.detach().cpu().numpy()  # 将潜在表示存储回adata用于后续分析
    test_gex_adata.obsm["X_atac"] = atac_representations.detach().cpu().numpy()  # 将潜在表示存储回adata用于后续分析

    # mi_latent_rna = pyinform.mutualinfo.mutual_info(latent_representations, rna_representations)
    # mi_latent_atac = pyinform.mutualinfo.mutual_info(latent_representations, atac_representations)
    # mi_latent_atac_rna = pyinform.mutualinfo.mutual_info(rna_representations, atac_representations)
    # mi_share_atac_rna = pyinform.mutualinfo.mutual_info(rna_share_representations, atac_share_representations)
    #
    # print(f"Mutual Information between latent and RNA representations: {mi_latent_rna}")
    # print(f"Mutual Information between latent and ATAC representations: {mi_latent_atac}")
    # print(f"Mutual Information between latent RNA and ATAC representations: {mi_latent_atac_rna}")
    # print(f"Mutual Information between share RNA and ATAC representations: {mi_share_atac_rna}")

    # 进行UMAP降维并可视化
    representations = ["X_share", "X_atac",  "X_rna"]
    save_prefixes = ["share", "atac_private", "rna_private"]

    # for rep, prefix in zip(representations, save_prefixes):
    #     sc.pp.neighbors(test_gex_adata, use_rep=rep)
    #     sc.tl.tsne(test_gex_adata, use_rep=rep)
    #
    #     # sc.pl.tsne(test_gex_adata, color="batch", save=f'figures/{dataset_name}/_{prefix}_batch_{fold}.pdf')
    #     sc.pl.tsne(test_gex_adata, color="cell_type", show=False)
    #     plt.savefig(f'figures/{dataset_name}/tsne_{prefix}_cell_type_{fold}.pdf', dpi=600, bbox_inches='tight')
    #
    #     sc.tl.leiden(test_gex_adata)
    #
    #     ARI = metrics.adjusted_rand_score(test_gex_adata.obs['cell_type'], test_gex_adata.obs['leiden'])
    #     AMI = metrics.adjusted_mutual_info_score(test_gex_adata.obs['cell_type'], test_gex_adata.obs['leiden'])
    #     NMI = metrics.normalized_mutual_info_score(test_gex_adata.obs['cell_type'], test_gex_adata.obs['leiden'])
    #     HOM = metrics.homogeneity_score(test_gex_adata.obs['cell_type'], test_gex_adata.obs['leiden'])
    #
    #     print('ARI', ARI)
    #     print('AMI', AMI)
    #     print('NMI', NMI)
    #     print('HOM', HOM)

    sc.pp.pca(gex_data_recon)
    sc.pp.neighbors(gex_data_recon, use_rep='X')
    sc.tl.leiden(gex_data_recon)
    # sc.tl.louvain(gex_data)

    ARI = metrics.adjusted_rand_score(gex_data_recon.obs[cell_key], gex_data_recon.obs['leiden'])
    AMI = metrics.adjusted_mutual_info_score(gex_data_recon.obs[cell_key], gex_data_recon.obs['leiden'])
    NMI = metrics.normalized_mutual_info_score(gex_data_recon.obs[cell_key], gex_data_recon.obs['leiden'])
    HOM = metrics.homogeneity_score(gex_data_recon.obs[cell_key], gex_data_recon.obs['leiden'])

    print('\nrecon RNA')
    print('ARI', ARI)
    # shared_dict['all_ARI'][fold] = ARI
    print('AMI', AMI)
    # shared_dict['all_AMI'][fold] = AMI
    print('NMI', NMI)
    # shared_dict['all_NMI'][fold] = NMI
    print('HOM', HOM)
    # shared_dict['all_HOM'][fold] = HOM

    # 将 AnnData 对象的 .X 属性转换为 numpy 数组
    actual_rna = test_gex_adata.X.toarray()
    reconstructed_rna = all_recon_rna.detach().cpu().numpy()

    average_pcc = calculate_pcc_torch(torch.from_numpy(actual_rna), torch.from_numpy(reconstructed_rna))

    print("Average R^2 value for RNA reconstruction:", average_pcc.mean(dim=0))
    # shared_dict['all_r2'][fold] = average_r2.mean(dim=0).item()
    rna_metric = [fold, ARI, AMI, NMI, HOM, average_pcc.mean(dim=0).item()]

    sc.tl.umap(gex_data_recon)
    sc.pl.umap(gex_data_recon, color=cell_key, show=False)
    if not os.path.exists(f'figures/{dataset_name}/'):
        os.makedirs(f'figures/{dataset_name}/')
    plt.savefig(f'figures/{dataset_name}/umap_rna_recon_cell_type_{training_mode}_{fold}.pdf', dpi=600, bbox_inches='tight')

    sc.pp.pca(atac_data_recon)
    sc.pp.neighbors(atac_data_recon, use_rep='X')
    sc.tl.leiden(atac_data_recon)

    ARI = metrics.adjusted_rand_score(atac_data_recon.obs[cell_key], atac_data_recon.obs['leiden'])
    AMI = metrics.adjusted_mutual_info_score(atac_data_recon.obs[cell_key], atac_data_recon.obs['leiden'])
    NMI = metrics.normalized_mutual_info_score(atac_data_recon.obs[cell_key], atac_data_recon.obs['leiden'])
    HOM = metrics.homogeneity_score(atac_data_recon.obs[cell_key], atac_data_recon.obs['leiden'])

    atac_metric = [fold, ARI, AMI, NMI, HOM]

    print('\nrecon atac')
    print('ARI', ARI)
    # shared_dict['all_atac_ARI'][fold] = ARI
    print('AMI', AMI)
    # shared_dict['all_atac_AMI'][fold] = AMI
    print('NMI', NMI)
    # shared_dict['all_atac_NMI'][fold] = NMI
    print('HOM', HOM)
    # shared_dict['all_atac_HOM'][fold] = HOM

    sc.tl.umap(atac_data_recon)
    sc.pl.umap(atac_data_recon, color=cell_key, show=False)
    plt.savefig(f'figures/{dataset_name}/umap_atac_recon_cell_type_{training_mode}_{fold}.pdf', dpi=1000, bbox_inches='tight')

    return rna_metric, atac_metric


def worker_function(func_args, return_queue):
    # 假设 function 是你的目标函数
    result = scDM_training(*func_args)
    return_queue.put(result)


def multiprocessing_train_fold(folds, function, func_args_list):
    processes = []
    return_queue = mp.Queue()

    for i in range(folds):
        p = mp.Process(target=worker_function, args=(func_args_list[i], return_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not return_queue.empty():
        results.append(return_queue.get())

    return results


all_fold = [f'fold{fold}' for fold in range(1, 6)]
device_list = [7, 2, 7, 2, 7]

manager = mp.Manager()
process_shared_dict = manager.dict()

process_shared_dict['all_ARI'] = [0 for _ in range(5)]
process_shared_dict['all_NMI'] = [0 for _ in range(5)]
process_shared_dict['all_HOM'] = [0 for _ in range(5)]
process_shared_dict['all_AMI'] = [0 for _ in range(5)]
process_shared_dict['all_pcc'] = [0 for _ in range(5)]
process_shared_dict['all_atac_ARI'] = [0 for _ in range(5)]
process_shared_dict['all_atac_NMI'] = [0 for _ in range(5)]
process_shared_dict['all_atac_HOM'] = [0 for _ in range(5)]
process_shared_dict['all_atac_AMI'] = [0 for _ in range(5)]

training_function = [(fold, device_list[fold]) for fold in range(5)]

results = multiprocessing_train_fold(5, scDM_training, training_function)

print(results)

all_rna_metric = []
all_atac_metric = []

for result in results:
    rna_metric, atac_metric = result
    all_rna_metric.append(rna_metric)
    all_atac_metric.append(atac_metric)

#
df_rna_metric = pd.DataFrame(all_rna_metric)
# df_rna_metric = df_rna_metric.T
df_rna_metric.columns = ['fold', 'ARI', 'AMI', 'NMI', 'HOM', 'pcc']
df_rna_metric.sort_values(by='fold', inplace=True)
df_rna_metric.to_csv(f'metric_performance/all_{dataset_name}_{training_mode}_rna_metrics.csv', index=False)

df_atac_metric = pd.DataFrame(all_atac_metric)
# df_atac_metric = df_atac_metric.T
df_atac_metric.columns = ['fold', 'ARI', 'AMI', 'NMI', 'HOM']
df_atac_metric.sort_values(by='fold', inplace=True)
df_atac_metric.to_csv(f'metric_performance/all_{dataset_name}_{training_mode}_atac_metrics.csv', index=False)

