# scTFBridge

**scTFBridge: A Disentangled Deep Generative Model Informed by TF-Motif Binding for Gene Regulation Inference in Single-Cell Multi-Omics**

scTFBridge is a novel single-cell multi-omics integration method designed for modality disentanglement and gene regulatory network (GRN) inference. It leverages transcription factor (TF)-motif binding information to model complex regulatory relationships in single-cell data, enabling researchers to uncover insights into gene regulation across multiple omics layers.


## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training and GRN inference](#model-architecture)
- [License](#license)
- [Citing](#citing)
---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA (optional, for GPU acceleration)
- Required libraries:
  - `torch >= 1.10.0`
  - `numpy`
  - `pandas`
  - `scanpy`
  - `anndata`
  - `scipy`

### Install from source
To install scTFBridge from source, follow these steps:
```bash
# Clone the repository
git clone https://github.com/your-username/scTFBridge.git

# install with pip
Note: PyPI package coming soon. Stay tuned for updates

```

## Tutorials

## GRN 

## Data Preparation
Here, we provide a demo

```bash
from utils.data_processing import five_fold_split_dataset, adata_multiomics_processing
import anndata


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
```

## License


## Citing
