from heatmaps import *
from model_comparison_core.model_parameters.GPDM_params import output_dir_name

# Usage
output_name = 'heatmaps'
csv_directory = 'C:/Users/Jesse/Documents/Python/GPy/HGPLVM_output_repository/model_summaries/GPDM_params/'+output_dir_name  # Replace with your actual directory path
save_directory = 'C:/Users/Jesse/Documents/Python/GPy/HGPLVM_output_repository/compiled_model_summaries/heatmaps/'+output_dir_name+'/'
results = process_csv_files(csv_directory)
create_heatmap(results, save_directory, output_name)

