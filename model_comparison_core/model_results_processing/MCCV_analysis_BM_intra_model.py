import os
import pandas as pd
import numpy as np
from scipy import stats
import glob
from datetime import datetime
import io
import re

def extract_model_attribute(filepath, attribute_row='init'):
    """Extract the model attribute from the first table in the CSV file"""
    try:
        # Read until we find the first --- ITERATION RESULTS --- separator
        with open(filepath, 'r') as file:
            parameter_table = []
            for line in file:
                if '--- ITERATION RESULTS ---' in line:
                    break
                parameter_table.append(line)
        
        # Convert parameter table to DataFrame
        parameter_df = pd.read_csv(io.StringIO(''.join(parameter_table)))
        
        # Find the attribute value
        attribute_value = parameter_df[parameter_df['Parameter'] == attribute_row]['Optimized Value'].iloc[0]
        return attribute_value
    except Exception as e:
        print(f"Error extracting attribute from {filepath}: {str(e)}")
        return None

def get_model_label(files, attribute_row='init'):
    """Get the model label from the first file in the list"""
    if not files:
        return "Unknown"
    
    # Use the first file to determine the model attribute
    first_file = files[0]
    attribute = extract_model_attribute(first_file, attribute_row)
    return attribute if attribute is not None else "Unknown"

def organize_files_by_model_and_split(source_folder, range_size=50, attribute_row='init'):
    """Organizes files by model (range) and splits within each model"""
    files_by_model = {}
    model_labels = {}
    
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    
    for file in csv_files:
        match = re.search(r'dict_(\d+)', file)
        if match:
            dict_num = int(match.group(1))
            model_num = dict_num // range_size
            split_num = dict_num % range_size
            
            if model_num not in files_by_model:
                files_by_model[model_num] = {}
            
            if split_num not in files_by_model[model_num]:
                files_by_model[model_num][split_num] = []
            
            filepath = os.path.join(source_folder, file)
            files_by_model[model_num][split_num].append(filepath)
            
            # Get model label if we haven't already
            if model_num not in model_labels:
                model_labels[model_num] = get_model_label(
                    files_by_model[model_num][split_num], 
                    attribute_row
                )
    
    return files_by_model, model_labels

def process_file(filepath):
    """Process a single file and return metrics"""
    try:
        with open(filepath, 'r') as file:
            content = file.read()

        parts = content.split('--- ITERATION RESULTS ---')
        if len(parts) < 2:
            print(f"Warning: File {filepath} does not contain expected format")
            return None

        second_table = pd.read_csv(io.StringIO(parts[1].strip()))
        
        # Find the maximum F1 score
        max_f1 = second_table['f1'].max()
        max_f1_rows = second_table[second_table['f1'] == max_f1]
        best_row = max_f1_rows.loc[max_f1_rows['score'].idxmin()]

        return {
            'score': float(best_row['score']),
            'f1': float(best_row['f1']),
            'msad': float(best_row['msad'])
        }
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
        return None

def process_model_split_results(files):
    """Process all files for a particular model and split"""
    results = []
    for filepath in files:
        result = process_file(filepath)
        if result is not None:
            results.append(result)
    
    if not results:
        return None
    
    # Average the results if there are multiple evaluations
    return {
        'score': np.mean([r['score'] for r in results]),
        'f1': np.mean([r['f1'] for r in results]),
        'msad': np.mean([r['msad'] for r in results])
    }

def perform_statistical_test(control_data, test_data, test_type='ttest'):
    """Perform statistical test on two datasets"""
    if len(control_data) == 0 or len(test_data) == 0:
        return np.nan, np.nan
    
    if test_type == 'ttest':
        statistic, p_value = stats.ttest_rel(control_data, test_data)
    elif test_type == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(control_data, test_data)
    else:
        raise ValueError("Invalid test type. Choose 'ttest' or 'wilcoxon'.")
    
    return statistic, p_value

def run_mccv_analysis(files_by_model, control_model_num, test_type='ttest'):
    """Run MCCV analysis comparing control model to each test model"""
    if control_model_num not in files_by_model:
        raise ValueError(f"Control model {control_model_num} not found in data")
    
    control_splits = files_by_model[control_model_num]
    results = {}
    
    # Process control model results
    control_results_by_split = {}
    for split_num, files in control_splits.items():
        result = process_model_split_results(files)
        if result is not None:
            control_results_by_split[split_num] = result
    
    # Compare with each other model
    for model_num, model_splits in files_by_model.items():
        if model_num == control_model_num:
            continue
            
        print(f"Comparing control model with model {model_num}")
        
        # Process test model results
        test_results_by_split = {}
        for split_num, files in model_splits.items():
            result = process_model_split_results(files)
            if result is not None:
                test_results_by_split[split_num] = result
        
        # Find common splits
        common_splits = set(control_results_by_split.keys()) & set(test_results_by_split.keys())
        
        if not common_splits:
            print(f"No common splits found between control and model {model_num}")
            continue
        
        # Prepare data for statistical tests
        metric_data = {
            metric: {
                'control': [],
                'test': []
            } for metric in ['score', 'f1', 'msad']
        }
        
        for split in common_splits:
            for metric in ['score', 'f1', 'msad']:
                metric_data[metric]['control'].append(control_results_by_split[split][metric])
                metric_data[metric]['test'].append(test_results_by_split[split][metric])
        
        # Calculate statistics for each metric
        model_results = {}
        for metric in ['score', 'f1', 'msad']:
            control_values = metric_data[metric]['control']
            test_values = metric_data[metric]['test']
            
            statistic, p_value = perform_statistical_test(control_values, test_values, test_type)
            
            model_results[metric] = {
                'control_mean': np.mean(control_values),
                'control_std': np.std(control_values),
                'test_mean': np.mean(test_values),
                'test_std': np.std(test_values),
                'difference': np.mean(test_values) - np.mean(control_values),
                'statistic': statistic,
                'p_value': p_value,
                'n_splits': len(common_splits)
            }
        
        results[f"model_{model_num}"] = model_results
    
    return results

def save_results_to_csv(results, output_dir, control_model_num, model_labels):
    """Save analysis results to separate CSV files for each metric"""
    if not results:
        print("No results to save")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    control_label = model_labels.get(control_model_num, f"Model {control_model_num}")

    # Create separate files for each metric
    for metric in ['score', 'f1', 'msad']:
        rows = []
        
        # First add the control group data
        control_data = next(iter(results.values()))[metric]  # Get control data from first comparison
        rows.append({
            'Model': control_label,
            'Mean': control_data['control_mean'],
            'Std Dev': control_data['control_std'],
            'Difference': '-',
            'Statistic': '-',
            'p-value': '-'
        })

        # Then add each test model's data
        for test_model, model_results in results.items():
            model_num = int(test_model.split('_')[1])
            model_label = model_labels.get(model_num, test_model)
            
            metric_results = model_results[metric]
            rows.append({
                'Model': model_label,
                'Mean': metric_results['test_mean'],
                'Std Dev': metric_results['test_std'],
                'Difference': metric_results['difference'],
                'Statistic': metric_results['statistic'],
                'p-value': metric_results['p_value']
            })

        df = pd.DataFrame(rows)
        file_name = f"mccv_analysis_{metric}_{timestamp}.csv"
        file_path = os.path.join(output_dir, file_name)
        df.to_csv(file_path, index=False, float_format='%.6f')
        print(f"Results for {metric} saved to {file_path}")

def main():
    # Configuration
    source_folder = r"C:\Users\Jesse\Documents\Python\HGP_concise\HGPLVM_output_repository\model_summaries\MCCV\GPDM_MCCV_BM_IC_testing\GPDMM_testing_inits_MCCV_f1_dist_msad_Bimanual 3D - Copy"
    output_dir = os.path.join(source_folder, "MCCV", "Bimanual 3D", "ttest")
    range_size = 50
    control_model_num = 17  # This would be the range number (e.g., 17 for dict_850 to dict_899)
    test_type = 'ttest'
    attribute_row = 'init'  # The row name containing the model attribute
    
    try:
        # Organize files by model and split
        print("Organizing files...")
        files_by_model, model_labels = organize_files_by_model_and_split(
            source_folder, 
            range_size, 
            attribute_row
        )
        
        if not files_by_model:
            raise ValueError("No files were organized. Check your source folder.")
        
        # Run analysis
        print("\nRunning MCCV analysis...")
        results = run_mccv_analysis(files_by_model, control_model_num, test_type=test_type)
        
        if results:
            # Save results
            save_results_to_csv(results, output_dir, control_model_num, model_labels)
            print("\nAnalysis complete!")
        else:
            print("No valid results to save")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()