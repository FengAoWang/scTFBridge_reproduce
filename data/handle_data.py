
from utils.data_processing import load_TFbinding, extract_overlap_regions, preload_TF_binding

dataset_name = 'human_PBMC'
#

GRNdir = 'GRN/data_bulk/'

filter_data_path = f'filter_data/{dataset_name}/'
output_path = f'filter_data/{dataset_name}/TF_binding/'
preload_TF_binding(filter_data_path, GRNdir, output_path)
