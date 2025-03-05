import os
import shutil
import datetime
from compile_data_funcs import main as compile_main
import pandas as pd
import io
import re
import glob
import numpy as np

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

    base_path = base_dir + '/output_repository/model_summaries/' + model_summaries_sub_dir

    print(f"Processing model: {model_name}")
    
    # Use model_name directly instead of importing output_dir_name
    print(f"Directory name: {model_name}")

    # Construct the full path
    full_path = os.path.join(base_path, model_name)
    print(f"Full path to unpack: {full_path}")

    try:
        # Unpack the directory
        unpacked_dir = unpack_directory(full_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the directory exists and the path is correct.")
        return

    # Output directory - modified to use full model_summaries_sub_dir path
    output_base = os.path.abspath(
        base_dir+r'\output_repository\compiled_model_summaries')
    # Create the full subdirectory path
    output_subdir = os.path.join(output_base, model_summaries_sub_dir.strip('/'), model_name)
    
    print(f"Creating output directory: {output_subdir}")
    # Create output directory if it doesn't exist
    os.makedirs(output_subdir, exist_ok=True)

    # Output file name
    output_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S" + '.csv')
    output_file = os.path.join(output_subdir, output_filename)

    # Ensure the path uses forward slashes
    output_file = output_file.replace('\\', '/')
    print(f"Output file: {output_file}")

    # Process the directory and calculate statistics
    results = process_directory(unpacked_dir, grouping_params)
    stats = calculate_statistics(results)
    save_results_to_csv(stats, output_file, grouping_params)

    # Clean up the unpacked directory
    try:
        shutil.rmtree(unpacked_dir)
        print(f"Cleaned up temporary directory: {unpacked_dir}")
    except Exception as e:
        print(f"Warning: Could not remove temporary directory {unpacked_dir}: {e}")

def find_best_iteration(filepath):
    """First pass to find best iteration based on validation data"""
    with open(filepath, 'r') as file:
        content = file.read()

    parts = content.split('--- ITERATION RESULTS ---')
    second_table = pd.read_csv(io.StringIO(parts[1].strip()))
    
    # Filter for validation rows
    validation_rows = second_table[second_table['label'] == 'validation']
    
    # Find the maximum F1 score
    max_f1 = validation_rows['f1'].max()
    
    # Filter rows with maximum F1 score
    max_f1_rows = validation_rows[validation_rows['f1'] == max_f1]
    
    # Among these rows, find the one with minimum score
    best_validation_row = max_f1_rows.loc[max_f1_rows['score'].idxmin()]
    
    return best_validation_row['iteration']

def process_file(filepath, target_iteration):
    """Second pass to get test metrics at the closest iteration to target"""
    with open(filepath, 'r') as file:
        content = file.read()

    parts = content.split('--- ITERATION RESULTS ---')
    first_table = pd.read_csv(io.StringIO(parts[0]))
    second_table = pd.read_csv(io.StringIO(parts[1].strip()))

    # Filter for test rows and create a copy to avoid the warning
    test_rows = second_table[second_table['label'] == 'test'].copy()
    
    # Use loc to set values
    test_rows.loc[:, 'iteration_diff'] = abs(test_rows['iteration'] - target_iteration)
    closest_test_row = test_rows.loc[test_rows['iteration_diff'].idxmin()]
    
    # Extract hyperparameters and metrics
    hyperparams = {}
    for _, row in first_table.iterrows():
        param = row['Parameter']
        value = try_float(row['Optimized Value'])
        hyperparams[param] = value

    metrics = {
        'score': closest_test_row['score'],
        'f1': closest_test_row['f1'],
        'smoothness': closest_test_row['smoothness'],
        'avg_freeze': closest_test_row['avg_freeze'],
        'iteration': closest_test_row['iteration']
    }

    return hyperparams, metrics

def process_directory(directory_path, grouping_params):
    """
    Process all files in a directory and group them according to specified parameters.
    Modified to match MCCV_analysis approach: each file uses its own best iteration.
    """
    results = {}
    
    # Process each file individually, finding the best iteration for each file
    for file in glob.glob(os.path.join(directory_path, '*.csv')):
        try:
            with open(file, 'r') as f:
                content = f.read()
            first_table = pd.read_csv(io.StringIO(content.split('--- ITERATION RESULTS ---')[0]))
            
            # Extract hyperparameters
            hyperparams = {row['Parameter']: try_float(row['Optimized Value']) 
                         for _, row in first_table.iterrows()}
            
            # Process parameters that need special handling
            processed_params = {}
            
            # First, handle constraints specially
            if 'constraints' in grouping_params:
                constraint_params = sorted(
                    [p for p in hyperparams.keys() if p.startswith('constraints-')],
                    key=lambda x: int(x.split('-')[1])
                )
                if constraint_params:
                    combined_constraints = tuple(hyperparams[param] for param in constraint_params)
                    processed_params['constraints'] = combined_constraints
                    # Remove individual constraint parameters
                    for p in constraint_params:
                        hyperparams.pop(p, None)
            
            # Then handle other parameters
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
            
            # Create group key using processed parameters
            group_key = tuple(hyperparams.get(param, None) for param in grouping_params)
            
            # Initialize group if not already present
            if group_key not in results:
                results[group_key] = {
                    'hyperparams': hyperparams,
                    'folds': []
                }
            
            # Find the best iteration for this specific file (not averaged across files)
            best_iteration = find_best_iteration(file)
            
            # Process file with its own best iteration
            file_hyperparams, metrics = process_file(file, best_iteration)
            results[group_key]['folds'].append(metrics)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    return results

def calculate_statistics(results):
    stats = {}
    metrics = ['score', 'f1', 'smoothness', 'avg_freeze', 'iteration']
    
    for group_key, group_data in results.items():
        folds = group_data['folds']
        hyperparams = group_data['hyperparams']
        
        group_stats = {
            'hyperparams': hyperparams,
            'num_folds': len(folds)
        }
        
        # Calculate statistics for each metric
        for metric in metrics:
            values = [fold[metric] for fold in folds]
            group_stats[f'{metric}_mean'] = np.mean(values)
            if metric != 'iteration':  # Only calculate std for non-iteration metrics
                group_stats[f'{metric}_std'] = np.std(values)
        
        # Calculate score/f1 ratio
        group_stats['score_over_f1'] = group_stats['score_mean'] / group_stats['f1_mean']
        
        # Calculate new composite metrics
        group_stats['score_f1_freeze'] = group_stats['score_over_f1'] * group_stats['avg_freeze_mean']
        group_stats['score_f1_freeze_smooth'] = group_stats['score_f1_freeze'] * group_stats['smoothness_mean']
        
        stats[group_key] = group_stats
    
    return stats

def save_results_to_csv(stats, output_file, grouping_params):
    rows = []
    for group_key, group_stats in stats.items():
        row = {}
        # Add hyperparameters
        for param, value in zip(grouping_params, group_key):
            row[param] = value
        
        # Add statistics
        for key, value in group_stats.items():
            if key != 'hyperparams':
                row[key] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    param_dir = 'intra_model'
    sub_dir = 'geometries'  
    data_set = 'BM'
    name = 'geos_all'
    smoothness_metric = 'smoothness'
    models = [
        {
            'name': name+"_"+data_set,
            'model_summaries_sub_dir': param_dir+'/'+sub_dir+'/',
            'grouping_params': ['init','geometry',"geo params"]
        },
    ]

    for model in models:
        process_model(model['name'], model['model_summaries_sub_dir'], model['grouping_params'], smoothness_metric)

if __name__ == "__main__":
    main()