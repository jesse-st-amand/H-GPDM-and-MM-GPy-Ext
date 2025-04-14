import os
import shutil
import datetime
import pandas as pd
import io
import glob

def try_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value

def unpack_directory(source_dir):
    print(f"Attempting to unpack directory: {source_dir}")
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"The source directory does not exist: {source_dir}")

    parent_dir = os.path.dirname(source_dir)
    source_dir_name = os.path.basename(source_dir)
    copy_dir_name = f"unpacked_{source_dir_name}"
    copy_dir = os.path.join(parent_dir, copy_dir_name)

    print(f"Creating copy directory: {copy_dir}")
    os.makedirs(copy_dir, exist_ok=True)

    print(f"Copying files from {source_dir} to {copy_dir}")
    shutil.copytree(source_dir, copy_dir, dirs_exist_ok=True)

    working_dir = copy_dir
    sub_dirs = [f.path for f in os.scandir(working_dir) if f.is_dir()]

    for dir_path in sub_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(working_dir, file)
                shutil.move(src_path, dest_path)
        shutil.rmtree(dir_path)

    print(f"Finished unpacking all folders to {working_dir}")
    print("Original subdirectories have been removed.")
    return copy_dir

def process_model(model_name, model_summaries_sub_dir, grouping_params, smoothness_metric):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Path to the original files
    base_path = base_dir + '/output_repository/model_summaries/' + model_summaries_sub_dir

    print(f"Processing model: {model_name}")
    print(f"Directory name: {model_name}")

    # Construct the full path to the source directory
    full_path = os.path.join(base_path, model_name)
    print(f"Full path to unpack: {full_path}")

    try:
        # Unpack the directory to work with all files
        unpacked_dir = unpack_directory(full_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the directory exists and the path is correct.")
        return

    # Create path for the reorganized files
    # Replace param_dir with save_dir in the path
    output_path = base_path.replace(param_dir, save_dir if save_dir else 'reorganized')
    output_model_path = os.path.join(output_path, model_name)
    
    print(f"Creating output directory: {output_model_path}")
    os.makedirs(output_model_path, exist_ok=True)

    # Copy files to reorganized directory structure
    copy_files_to_reorganized_dirs(unpacked_dir, output_model_path, grouping_params)

    # Clean up the unpacked directory
    try:
        shutil.rmtree(unpacked_dir)
        print(f"Cleaned up temporary directory: {unpacked_dir}")
    except Exception as e:
        print(f"Warning: Could not remove temporary directory {unpacked_dir}: {e}")

def copy_files_to_reorganized_dirs(directory_path, output_path, grouping_params):
    """Copy files to reorganized directory structure based on grouping parameters"""
    for file in glob.glob(os.path.join(directory_path, '*.csv')):
        try:
            with open(file, 'r') as f:
                content = f.read()
            
            # Get the first table from the file (parameter table)
            first_table = pd.read_csv(io.StringIO(content.split('--- ITERATION RESULTS ---')[0]))
            
            # Extract hyperparameters
            hyperparams = {row['Parameter']: try_float(row['Optimized Value']) 
                         for _, row in first_table.iterrows()}
            
            # Process parameters that need special handling
            processed_params = {}
            
            # Handle constraints parameter specially
            if 'constraints' in grouping_params:
                constraint_params = sorted(
                    [p for p in hyperparams.keys() if p.startswith('constraints-')],
                    key=lambda x: int(x.split('-')[1])
                )
                if constraint_params:
                    combined_constraints = tuple(hyperparams[param] for param in constraint_params)
                    processed_params['constraints'] = combined_constraints
                    for p in constraint_params:
                        hyperparams.pop(p, None)
            
            # Handle other parameters
            for param in grouping_params:
                if param == 'constraints':
                    continue  # Already handled above
                
                # Check if parameter has a numeric suffix
                param_parts = param.rsplit('-', 1)
                base_param = param_parts[0]
                
                # Get all matching parameters and their values in order of appearance
                param_values = []
                for _, row in first_table.iterrows():
                    if row['Parameter'] == base_param:
                        param_values.append(try_float(row['Optimized Value']))
                
                if len(param_parts) > 1:
                    try:
                        # If it's a numeric suffix, get the nth occurrence (0-based index)
                        index = int(param_parts[1])
                        if param_values:
                            processed_params[param] = param_values[index]
                    except ValueError:
                        # If suffix is not numeric, treat as regular parameter
                        if param in hyperparams:
                            processed_params[param] = hyperparams[param]
                else:
                    # No suffix - handle potential duplicates
                    if param_values:
                        if len(param_values) == 1:
                            processed_params[param] = param_values[0]
                        else:
                            # Multiple occurrences - combine into tuple preserving order
                            processed_params[param] = tuple(param_values)
            
            # Update hyperparams with processed parameters
            hyperparams.update(processed_params)
            
            # Create subdirectory path based on grouping parameters
            sub_dir_parts = []
            for param in grouping_params:
                param_value = hyperparams.get(param, "NA")
                # Convert tuples or lists to string if needed
                if isinstance(param_value, (list, tuple)):
                    param_value = "-".join(str(v) for v in param_value)
                
                # Replace spaces with underscores and colons with hyphens
                param_str = str(param).replace(" ", "_").replace(":", "_")
                param_value_str = str(param_value).replace(" ", "_").replace(":", "_")
                
                sub_dir_parts.append(f"{param_str}-{param_value_str}")
            
            # Combine all parts and sanitize the final directory name
            group_dir_name = ";".join(sub_dir_parts)
            # Additional sanitization for the entire directory name
            group_dir_name = group_dir_name.replace(" ", "_").replace(":", "_")
            
            group_sub_dir = os.path.join(output_path, group_dir_name)
            os.makedirs(group_sub_dir, exist_ok=True)
            
            # Copy the file to the appropriate subdirectory
            file_name = os.path.basename(file)
            dest_file_path = os.path.join(group_sub_dir, file_name)
            shutil.copy2(file, dest_file_path)
            print(f"Copied {file} to {dest_file_path}")
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

def main():
    global param_dir, save_dir
    
    save_dir = 'MCCV_ICANN'
    param_dir = 'intra_model'
    sub_dir = 'hierarchies'  
    data_set = 'CMU'
    name = 'H1_all'
    smoothness_metric = 'smoothness'  # Kept for compatibility
    models = [
        {
            'name': name+"_"+data_set,
            'model_summaries_sub_dir': param_dir+'/'+sub_dir+'/',
            'grouping_params': ['input_dim',"geo params"]
        },
    ]

    for model in models:
        process_model(model['name'], model['model_summaries_sub_dir'], model['grouping_params'], smoothness_metric)

if __name__ == "__main__":
    main()