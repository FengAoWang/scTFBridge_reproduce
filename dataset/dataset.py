import numpy
from torch.utils.data import Dataset
import anndata
import torch
import numpy as np
import pandas as pd


class scMultiDataset(Dataset):
    def __init__(self, anndata_list, batch_info: pd.DataFrame):
        self.rna_tensor = torch.Tensor(anndata_list[0])
        self.atac_tensor = torch.Tensor(anndata_list[1])
        self.TF_tensor = torch.Tensor(anndata_list[2])
        self.batch_info = torch.Tensor(batch_info.to_numpy())

    def __len__(self):
        return self.rna_tensor.shape[0]

    def __getitem__(self, idx):
        rna_data = self.rna_tensor[idx, :]
        atac_data = self.atac_tensor[idx, :]
        TF_data = self.TF_tensor[idx, :]
        batch_info = self.batch_info[idx, :]
        return rna_data, atac_data, TF_data, batch_info
