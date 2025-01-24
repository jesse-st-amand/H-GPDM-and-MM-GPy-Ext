from MCCV_analysis_base import run_mccv_analysis, print_raw_data, save_results_to_csv
import os

data_set = "Bimanual 3D"
sub_folder = 'MCCV_AISTATS_ldj'
test_type = 'ttest'
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Example usage
control_path = dir+r"\HGPLVM_output_repository\model_summaries\\" + sub_folder + r"\GPDM_MCCV_BM_50\MCCV_f1_dist_ldj_Bimanual 3D_GPDMM_Bayesian_50"
test_paths = [
    dir+r"\HGPLVM_output_repository\model_summaries\\" + sub_folder + r"\RNN_MCCV_BM\MCCV_f1_dist_ldj_Bimanual 3D_Bayesian_RNN",
    dir+r"\HGPLVM_output_repository\model_summaries\\" + sub_folder + r"\transformer_MCCV_BM\MCCV_f1_dist_ldj_Bimanual 3D_Bayesian_Transformer",
    dir+r"\HGPLVM_output_repository\model_summaries\\" + sub_folder + r"\VAE_MCCV_BM\MCCV_f1_dist_ldj_Bimanual 3D_Bayesian_VAE"
]

results = run_mccv_analysis(control_path, test_paths, test_type=test_type)
#print_raw_data(results)
save_path = dir+r"\HGPLVM_output_repository\\MCCV\\" + sub_folder + "\\" +data_set + "\\" + test_type + "\\"
if not os.path.exists(save_path):
   os.makedirs(save_path)
save_results_to_csv(results, save_path)