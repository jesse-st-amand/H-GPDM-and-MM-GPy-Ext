from MCCV_analysis_grouping_base import run_mccv_analysis, print_raw_data, save_results_to_csv
import os
import glob
 


param_dir = 'MCCV_ICANN'
save_param_dir = 'intra_model'
test_type = 'ttest'
sub_dir = 'geometries'  
data_set = 'BM'
name = 'geos_all'
model = 'GPDMM'

control_params = {'init': 'kernel_pca_rbf',
                  'geometry': 'fourier_basis',
                  'geo_params': 11}

dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Example usage
main_dir = dir+r"\output_repository\model_summaries\\" + param_dir + r"\\" + sub_dir + r"\\" + name+"_"+data_set
print(f"Main directory: {main_dir}")
control_path = ''
test_paths = []

# Find control and test directories in main_dir
if os.path.exists(main_dir) and os.path.isdir(main_dir):
    # List all directories in main_dir
    subdirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    print(f"Found {len(subdirs)} subdirectories in main_dir")
    
    if not subdirs:
        print("Error: No subdirectories found in main_dir!")
        exit(1)
    
    # Function to check if a directory name matches control parameters
    def matches_control_params(dir_name):
        dir_basename = os.path.basename(dir_name)
        print(f"Checking directory: {dir_basename}")
        
        # Temporarily remove any ";control" suffix for parameter extraction
        if ";control" in dir_basename:
            dir_basename = dir_basename.replace(";control", "")
            
        # Parse directory name into parameters
        params = {}
        param_pairs = dir_basename.split(';')
        
        for pair in param_pairs:
            if '-' in pair:
                category, value = pair.split('-', 1)  # Split on first hyphen only
                
                # Handle numeric parameters (including those with decimal points)
                if category == 'geo_params':
                    if value.replace('.', '', 1).isdigit():  # Handles both integers and floats
                        # Convert to same type as in control_params for comparison
                        if isinstance(control_params.get(category), int):
                            # If control parameter is integer, convert to int (removing decimal)
                            try:
                                value = int(float(value))
                            except (ValueError, TypeError):
                                pass
                        elif isinstance(control_params.get(category), float):
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                pass
                
                params[category] = value
        
        # Debug: Print extracted parameters
        print(f"  Extracted parameters: {params}")
        print(f"  Control parameters: {control_params}")
        
        # Check if all control parameters are present with matching values
        for key, value in control_params.items():
            if key not in params:
                print(f"  Missing parameter: {key}")
                return False
            
            # Compare values (convert to string for consistent comparison)
            ctrl_value = str(value)
            dir_value = str(params[key])
            
            if ctrl_value != dir_value:
                print(f"  Parameter mismatch: {key} = {dir_value}, control = {ctrl_value}")
                return False
        
        print(f"  ✓ MATCH: This directory matches all control parameters")
        return True
    
    # Find control directory based on control_params
    print("\nSearching for control directory matching parameters:")
    for k, v in control_params.items():
        print(f"  {k}: {v}")
    
    control_dirs = [d for d in subdirs if matches_control_params(d)]
    if control_dirs:
        control_path = control_dirs[0]  # Use the first matching directory
        print(f"\nControl path set to: {control_path}")
    else:
        print("\nWarning: No directory matching control parameters!")
        # Check if there's a directory with ";control" suffix as fallback
        control_suffix_dirs = [d for d in subdirs if ";control" in os.path.basename(d)]
        if control_suffix_dirs:
            control_path = control_suffix_dirs[0]
            print(f"Falling back to directory with ;control suffix: {control_path}")
        else:
            # If no control directory found but directories exist, use the first one as control
            control_path = subdirs[0]
            print(f"Using {control_path} as default control path (first directory)")
    
    # Verify the control path has CSV files
    csv_files = glob.glob(os.path.join(control_path, '*.csv'))
    if not csv_files:
        print(f"Warning: No CSV files found in control path: {control_path}")
        # Try to find a different control path with CSV files
        for potential_control in subdirs:
            csv_files = glob.glob(os.path.join(potential_control, '*.csv'))
            if csv_files:
                control_path = potential_control
                print(f"Using alternate control path with CSV files: {control_path}")
                break
    
    print(f"Found {len(csv_files)} CSV files in control path")
    
    # Set test_paths to all other directories with CSV files
    test_paths = []
    for test_path in [d for d in subdirs if d != control_path]:
        csv_files = glob.glob(os.path.join(test_path, '*.csv'))
        if csv_files:
            test_paths.append(test_path)
            print(f"Found {len(csv_files)} CSV files in test path: {test_path}")
        else:
            print(f"Skipping test path with no CSV files: {test_path}")
    
    print(f"Using {len(test_paths)} valid test directories")
    
    if not test_paths:
        print("Warning: No valid test directories found with CSV files")
        exit(1)
else:
    print(f"Error: main_dir does not exist or is not a directory: {main_dir}")
    exit(1)

# Debug information before running analysis
print(f"Final control path: {control_path}")
print(f"Final test paths: {[os.path.basename(path) for path in test_paths]}")

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