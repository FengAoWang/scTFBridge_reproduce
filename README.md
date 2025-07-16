# scTFBridge: Reproduction Repository

This repository contains the code and instructions necessary to reproduce the analyses presented in the publication: **"scTFBridge: A Disentangled Deep Generative Model Informed by TF-Motif Binding for Gene Regulation Inference in Single-Cell Multi-Omics"**.

<br>

> **⭐ IMPORTANT NOTE**
>
> This repository is specifically for reproducing the original study's results. For a more user-friendly and streamlined version of the scTFBridge model to apply to your own data, please visit the main project repository:
>
> 👉 [**FengAoWang/scTFBridge**](https://github.com/FengAoWang/scTFBridge)
>
<br>

scTFBridge is a novel deep generative model for single-cell multi-omics integration. It uniquely incorporates transcription factor (TF)-motif binding information to disentangle chromatin accessibility and gene expression, enabling robust inference of gene regulatory networks (GRNs).

![scTFBridge Overview](https://raw.githubusercontent.com/FengAoWang/scTFBridge_reproduce/main/figure1.png)
*Figure: An overview of the scTFBridge model architecture and its approach to integrating scRNA-seq and scATAC-seq data.*

## Table of Contents
* [Installation](#installation)
* [Workflow for Reproducing Results](#workflow-for-reproducing-results)
* [Citation](#-citation)


---

## Installation

### Prerequisites

- Python 3.12 or higher
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
git clone https://github.com/your-username/scTFBridge_reproduce.git

# install with pip
pip install sctfbridge
```




## Workflow for Reproducing Results
Here, we provide a demo to preprocess and train scTFBridge using a single-cell multi-omics dataset from PBMC. 
### Dataset filtering
```bash
from utils.data_processing import five_fold_split_dataset, adata_multiomics_processing
import anndata


dataset = 'PBMC'


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

```
### Prepare TF-Motif Binding file
```bash
from utils.data_processing import preload_TF_binding

dataset_name = 'PBMC'
#

GRNdir = 'GRN/data_bulk/'

filter_data_path = f'filter_data/{dataset_name}/'
output_path = f'filter_data/{dataset_name}/TF_binding/'
preload_TF_binding(filter_data_path, GRNdir, output_path)

```


### loading GEX ATAC adata
```bash
dataset_name = 'PBMC'
gex_data = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/RNA_filter.h5ad')
atac_adata = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/ATAC_filter.h5ad')
TF_adata = anndata.read_h5ad(f'../data/filter_data/{dataset_name}/TF_filter.h5ad')

TF_length = TF_adata.var.shape[0]

fold_split_info = pd.read_csv(f'../data/filter_data/{dataset_name}/fold_split_info.csv')
mask = pd.read_csv(f'../data/{dataset_name}_TF_Binding/TF_binding.txt', sep='\t', header=None).values
```

### loading scTFBridge
```bash
mask_tensor = torch.tensor(mask).float()
    sc_multi_demo = scTFBridge([dim1, dim2], [1024], [1024],
                               TF_dim, 1, ['gaussian', 'bernoulli'], batch_dims, 1, 1, mask_tensor)
 ```
The training process can be found in train/train_demo.py, then you will get the trained model for GRN inference.


### Trans-Regulatory Inference

Trans-regulatory elements are typically transcription factors that can influence the expression of distant genes. The inference of these genome-wide regulatory interactions is performed using the following script:

* **`train/explain_TF.py`**: This script takes the trained scTFBridge model and utilizes it to identify and score the regulatory influence of transcription factors on their potential target genes across the genome.

### Cis-Regulatory Inference

Cis-regulatory elements, such as promoters and enhancers, affect the expression of nearby genes on the same chromosome. The script dedicated to inferring these local regulatory interactions is:

* **`train/explain_cis_RE.py`**: This script leverages the trained model to analyze the relationships between regulatory elements and their proximal genes, providing insights into the cis-regulatory landscape.



## Citation
If you use scTFBridge in your research, please cite our publication:

> **scTFBridge: A Disentangled Deep Generative Model Informed by TF-Motif Binding for Gene Regulation Inference in Single-Cell Multi-Omics**
>
> Feng, A. Wang, *et al*. (2025). *bioRxiv*.
>
> doi: [`10.1101/2025.01.16.633293v1`](https://www.biorxiv.org/content/10.1101/2025.01.16.633293v1)


