from MCCV_analysis_grouping_base import run_mccv_analysis, print_raw_data, save_results_to_csv
from MCCV_analysis_control_param_func import find_control_and_test_paths
import os
import glob
 


param_dir = 'MCCV_ICANN'
save_param_dir = 'intra_model'
test_type = 'ttest'
sub_dir = 'geometries_selected_ICANN'  
data_set = 'CMU'
name = 'geos_selected'
model = 'GPDMM'

control_params = {'model': 'best per score',}

dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Example usage
main_dir = dir+r"\output_repository\model_summaries\\" + param_dir + r"\\" + sub_dir + r"\\" + name+"_"+data_set
print(f"Main directory: {main_dir}")

# Find control and test directories using the utility function
control_path, test_paths = find_control_and_test_paths(main_dir, control_params)

# Check if we found valid paths
if not control_path or not test_paths:
    print("Error: Could not find valid control path or test paths!")
    exit(1)

# Run the analysis
print("\nRunning MCCV analysis...")
results = run_mccv_analysis(control_path, test_paths, test_type=test_type)

# Save results
print("\nSaving results...")
save_path = dir+r"\output_repository\\MCCV\\" + save_param_dir + "\\" + sub_dir + r"\\" + name+"_"+data_set + "\\" + test_type + "\\"
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_results_to_csv(results, save_path)
print(f"Analysis complete. Results saved to {save_path}")