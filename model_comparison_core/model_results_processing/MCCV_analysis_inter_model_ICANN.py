from MCCV_analysis_base import run_mccv_analysis, print_raw_data, save_results_to_csv
import os

data_set = "Movements CMU"

if data_set == "Bimanual 3D":
   data_set_init = "BM"
elif data_set == "Movements CMU":
   data_set_init = "CMU"    

param_dir = 'inter_model'
sub_dir = 'ICANN'
test_type = 'ttest'
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Example usage
control_path = dir+r"\output_repository\model_summaries\\" + param_dir + r"\\" + sub_dir + r"\GPDM_MCCV_"+data_set_init
test_paths = [
   dir+r"\output_repository\model_summaries\\" + param_dir + r"\\" + sub_dir +  r"\GPDM_geo_4_MCCV_"+data_set_init,
    dir+r"\output_repository\model_summaries\\" + param_dir + r"\\" + sub_dir +  r"\GPDM_50_MCCV_"+data_set_init,
    dir+r"\output_repository\model_summaries\\" + param_dir + r"\\" + sub_dir +  r"\GPDM_50_50_MCCV_"+data_set_init,
    dir+r"\output_repository\model_summaries\\" + param_dir + r"\\" + sub_dir + r"\RNN_MCCV_"+data_set_init,
    dir+r"\output_repository\model_summaries\\" + param_dir + r"\\" + sub_dir + r"\transformer_MCCV_"+data_set_init,
    dir+r"\output_repository\model_summaries\\" + param_dir + r"\\" + sub_dir + r"\VAE_MCCV_"+data_set_init
]

results = run_mccv_analysis(control_path, test_paths, test_type=test_type)
#print_raw_data(results)
save_path = dir+r"\output_repository\\MCCV\\" + param_dir + "\\" + sub_dir + "\\" +data_set + "\\" + test_type + "\\"
if not os.path.exists(save_path):
   os.makedirs(save_path)
save_results_to_csv(results, save_path)