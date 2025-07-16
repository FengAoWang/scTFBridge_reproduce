from utils.data_processing import preload_TF_binding

dataset_name = 'PBMC'
#

GRNdir = 'GRN/data_bulk/'

filter_data_path = f'filter_data/{dataset_name}/'
output_path = f'filter_data/{dataset_name}/TF_binding/'
preload_TF_binding(filter_data_path, GRNdir, output_path)
