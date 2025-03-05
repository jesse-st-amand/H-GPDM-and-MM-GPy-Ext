import os
import pandas as pd
import numpy as np
from scipy import stats
import glob
from datetime import datetime
import io


def find_best_iteration(filepath):
    """First pass to find best iteration based on validation data"""
    with open(filepath, 'r') as file:
        content = file.read()

    parts = content.split('--- ITERATION RESULTS ---')
    second_table = pd.read_csv(io.StringIO(parts[1].strip()))
    
    # Filter for validation rows
    validation_rows = second_table[second_table['label'] == 'validation']
    
    if validation_rows.empty:
        # If no validation data, fall back to all data
        print(f"Warning: No validation data found in {filepath}. Using all data.")
        validation_rows = second_table
    
    # Find the maximum F1 score
    max_f1 = validation_rows['f1'].max()
    
    # Filter rows with maximum F1 score
    max_f1_rows = validation_rows[validation_rows['f1'] == max_f1]
    
    # Among these rows, find the one with minimum score
    best_validation_row = max_f1_rows.loc[max_f1_rows['score'].idxmin()]
    
    return best_validation_row['iteration']


def process_file(filepath):
    """
    Process a file using a two-pass approach:
    1. Find the best iteration based on validation data
    2. Get test metrics at the best iteration
    """
    print(f"Processing file: {filepath}")

    # First pass: find the best iteration based on validation data
    best_iteration = find_best_iteration(filepath)
    print(f"Best iteration found: {best_iteration}")
    
    # Second pass: get test metrics at the best iteration
    with open(filepath, 'r') as file:
        content = file.read()

    parts = content.split('--- ITERATION RESULTS ---')
    second_table = pd.read_csv(io.StringIO(parts[1].strip()))
    
    # Filter for test rows
    test_rows = second_table[second_table['label'] == 'test'].copy()
    
    if test_rows.empty:
        # If no test data, fall back to all data
        print(f"Warning: No test data found in {filepath}. Using all data.")
        test_rows = second_table.copy()
    
    # Find the closest iteration to the best one
    test_rows.loc[:, 'iteration_diff'] = abs(test_rows['iteration'] - best_iteration)
    closest_test_row = test_rows.loc[test_rows['iteration_diff'].idxmin()]
    
    print(f"Using test metrics from iteration {closest_test_row['iteration']}")
    print(f"Selected test row:\n{closest_test_row[['iteration', 'score', 'f1', 'smoothness', 'avg_freeze']]}")

    result = {
        'score': closest_test_row['score'],
        'f1': closest_test_row['f1'],
        'smoothness': closest_test_row['smoothness'],
        'freeze': closest_test_row['avg_freeze']
    }

    print(f"Returned result: {result}")
    return result


def get_model_results(path):
    results = {}
    for file in glob.glob(os.path.join(path, '*.csv')):
        sim_num = int(file.split('_')[-1].split('.')[0])
        results[sim_num] = process_file(file)
    return results


def perform_statistical_test(control_data, test_data, test_type='ttest'):
    if test_type == 'ttest':
        statistic, p_value = stats.ttest_rel(control_data, test_data)
    elif test_type == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(control_data, test_data)
    else:
        raise ValueError("Invalid test type. Choose 'ttest' or 'wilcoxon'.")

    return statistic, p_value


def run_mccv_analysis(control_path, test_paths, test_type='ttest'):
    control_results = get_model_results(control_path)
    control_name = os.path.basename(control_path)

    results = {metric: {} for metric in ['score', 'f1', 'smoothness', 'freeze']}

    # Add control model to results
    for metric in ['score', 'f1', 'smoothness', 'freeze']:
        control_data = [control_results[sim][metric] for sim in control_results.keys()]
        results[metric][control_name] = {
            'mean': np.mean(control_data),
            'std': np.std(control_data)
        }

    # Process test models
    for test_path in test_paths:
        model_name = os.path.basename(test_path)
        test_results = get_model_results(test_path)

        common_sims = set(control_results.keys()) & set(test_results.keys())

        for metric in ['score', 'f1', 'smoothness', 'freeze']:
            control_data = [control_results[sim][metric] for sim in common_sims]
            test_data = [test_results[sim][metric] for sim in common_sims]

            statistic, p_value = perform_statistical_test(control_data, test_data, test_type)

            results[metric][model_name] = {
                'mean': np.mean(test_data),
                'std': np.std(test_data),
                'difference': np.mean(test_data) - np.mean(control_data),
                'statistic': statistic,
                'p_value': p_value
            }

    return results


def save_results_to_csv(results, output_dir):
    """
    Save the analysis results to an Excel file with formatting.
    Bold is applied to cells with p-values < 0.05.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Function to parse the model name into parameter dictionary
    def parse_model_params(model_name):
        # Start with an empty dictionary for the parameters
        params = {}
        param_order = []  # Track the order of parameters
        
        # Remove control suffix if present, before parsing parameters
        if model_name.endswith(';control'):
            model_name = model_name[:-8]  # Remove ';control' suffix
        
        # Return empty params if model name does not contain parameters
        if '-' not in model_name:
            return params, param_order
            
        # Split by semicolon to get parameter pairs
        param_pairs = model_name.split(';')
        
        for pair in param_pairs:
            if '-' in pair:
                # Split by hyphen to get parameter category and value
                parts = pair.split('-', 1)  # Split on first hyphen only
                if len(parts) == 2:
                    category, value = parts
                    
                    # Format the parameter category (replace _ with space, capitalize words)
                    # Using a standardized way to format parameter names to avoid duplication
                    formatted_category = ' '.join(word.capitalize() for word in category.split('_'))
                    
                    # Ensure consistent naming for Geo Params
                    if formatted_category.lower() == "geo params":
                        formatted_category = "Geo Params"
                    
                    # Format the parameter value if it's a string (contains alphabetic characters)
                    if any(c.isalpha() for c in value):
                        value = ' '.join(word.capitalize() for word in value.split('_'))
                    
                    params[formatted_category] = value
                    param_order.append(formatted_category)
        
        # Handle special cases for geometry parameters with implicit values
        geometry_param_mapping = {
            'Ellipse': '2',
            'Toroid': '3',
            'Linear': '3',
            'Klein Bottle': '3',
            'Mobius Strip': '2',
            'Spiral': '2',
            'Cocentric Circles': '2',
            'Torus': '3',
            'Sphere': '3',
        }
        
        # Find the geometry and geo params keys if they exist (case-insensitive)
        geometry_key = next((key for key in params.keys() if key.lower() == "geometry"), None)
        geo_params_key = next((key for key in params.keys() if key.lower() == "geo params"), None)
        
        # If there's a Geometry parameter and either:
        # 1. No Geo Params key exists, or
        # 2. Geo Params key exists but has a blank value
        if geometry_key:
            geo_value = params[geometry_key]
            
            # Check if geo params is missing or has a blank value
            missing_geo_params = (geo_params_key is None or 
                                  not params[geo_params_key] or 
                                  params[geo_params_key].strip() == 'Na' or
                                  params[geo_params_key].strip() == '')
            
            if missing_geo_params:
                # Look for a matching geometry to assign implicit value
                for geometry_name, implicit_value in geometry_param_mapping.items():
                    if geometry_name.lower() in geo_value.lower():
                        # If geo params key already exists, update its value
                        if geo_params_key:
                            params[geo_params_key] = implicit_value
                        else:
                            # Otherwise create a new entry with standard formatting
                            params["Geo Params"] = implicit_value
                            # Add Geo Params after Geometry in the parameter order
                            if geometry_key in param_order:
                                geo_idx = param_order.index(geometry_key)
                                param_order.insert(geo_idx + 1, "Geo Params")
                            else:
                                param_order.append("Geo Params")
                        break
        
        return params, param_order

    # Maps metrics to their new column names
    metric_column_names = {
        'f1': 'F1',
        'score': 'Distance',
        'freeze': 'Dampening',
        'smoothness': 'LDJ'
    }
    
    # Get all unique models across all metrics
    all_models = set()
    for metric in results.keys():
        all_models.update(results[metric].keys())
    
    # Extract parameters from all models once
    all_param_categories = set()
    model_params = {}
    param_orders = {}
    all_param_orders = []  # Collect all parameter orders to determine a consistent order
    
    for model in all_models:
        model_basename = os.path.basename(model)
        params, param_order = parse_model_params(model_basename)
        model_params[model] = params
        param_orders[model] = param_order
        all_param_categories.update(params.keys())
        all_param_orders.append(param_order)
    
    # Create a consistent order for all parameters based on their relative positions
    sorted_models = sorted(all_param_orders, key=len, reverse=True)
    reference_order = sorted_models[0] if sorted_models else []
    
    # Add any missing parameters that exist in other models
    for order in all_param_orders:
        for param in order:
            if param not in reference_order:
                reference_order.append(param)
    
    # Use this reference order for all parameter columns
    param_columns = reference_order
    
    # Prepare rows for the combined table
    combined_rows = []
    control_name = list(results[list(results.keys())[0]].keys())[0]  # Assume the first model in the first metric is the control
    
    # Process each model
    for model in all_models:
        is_control = model == control_name
        
        # Initialize row with model base and parameters
        row = {
            'Model Base': os.path.basename(model)
        }
        
        # Add parameter values
        for param in param_columns:
            row[param] = model_params[model].get(param, 'NA')
        
        # Add metric values
        for metric in ['f1', 'score', 'freeze', 'smoothness']:
            if metric in results and model in results[metric]:
                values = results[metric][model]
                row[metric_column_names[metric]] = values['mean']
                # Add standard deviation columns
                row[f"{metric_column_names[metric]} Std"] = values['std']
                
                # Add difference, statistic, p-value for non-control models
                if not is_control and 'difference' in values:
                    row[f"{metric_column_names[metric]} Diff"] = values['difference']
                    row[f"{metric_column_names[metric]} Stat"] = values['statistic']
                    row[f"{metric_column_names[metric]} p"] = values['p_value']
            else:
                # If this metric doesn't have data for this model, use NA
                row[metric_column_names[metric]] = 'NA'
                row[f"{metric_column_names[metric]} Std"] = 'NA'
                if not is_control:
                    row[f"{metric_column_names[metric]} Diff"] = 'NA'
                    row[f"{metric_column_names[metric]} Stat"] = 'NA'
                    row[f"{metric_column_names[metric]} p"] = 'NA'
        
        combined_rows.append(row)
    
    # Create DataFrame from combined rows
    df = pd.DataFrame(combined_rows)
    
    # Add the calculated column: Distance*Dampening/F1
    df['Distance*Dampening/F1'] = df.apply(
        lambda row: (
            float(row['Distance']) * float(row['Dampening']) / float(row['F1'])
            if (row['Distance'] != 'NA' and row['Dampening'] != 'NA' and row['F1'] != 'NA' and float(row['F1']) != 0)
            else 'NA'
        ),
        axis=1
    )
    
    # Sort the DataFrame by the calculated column (Distance*Dampening/F1)
    # First convert the column to numeric, with errors='coerce' to handle 'NA' values
    df['Sort_Value'] = pd.to_numeric(df['Distance*Dampening/F1'], errors='coerce')
    # Sort by the numeric column, with NA values at the end
    df = df.sort_values(by='Sort_Value', ascending=True, na_position='last')
    # Drop the temporary sorting column
    df = df.drop(columns=['Sort_Value'])
    
    # Define column order
    base_columns = ['Model Base'] + param_columns
    metric_columns = []
    for metric in ['F1', 'Distance', 'Dampening', 'LDJ']:
        metric_columns.append(metric)
        metric_columns.append(f"{metric} Std")
        # Only add difference columns for the standard deviation, statistics and p-values
        # if there are non-control models
        if len(all_models) > 1:
            metric_columns.append(f"{metric} Diff")
            metric_columns.append(f"{metric} Stat")
            metric_columns.append(f"{metric} p")
    
    # Add the calculated column at the end
    final_columns = base_columns + metric_columns + ['Distance*Dampening/F1']
    
    # Reorder columns and only include columns that exist
    existing_columns = [col for col in final_columns if col in df.columns]
    df = df[existing_columns]
    
    # Save to Excel file
    file_name = f"combined_mccv_analysis_{timestamp}.xlsx"
    file_path = os.path.join(output_dir, file_name)
    
    # Need to import these libraries for Excel formatting
    try:
        import openpyxl
        from openpyxl.styles import Font
        
        # Create Excel writer with openpyxl engine
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Results']
            
            # Format headers - make them bold with light gray background
            header_font = Font(bold=True)
            for col_idx, col_name in enumerate(df.columns):
                col_letter = openpyxl.utils.get_column_letter(col_idx + 1)
                cell = worksheet[f"{col_letter}1"]
                cell.font = header_font
            
            # Set column widths for better readability
            for col_idx, col_name in enumerate(df.columns):
                col_letter = openpyxl.utils.get_column_letter(col_idx + 1)
                
                # Make Model Base column wider
                if col_name == 'Model Base':
                    worksheet.column_dimensions[col_letter].width = 30
                # Make parameter columns medium width
                elif col_name in param_columns:
                    worksheet.column_dimensions[col_letter].width = 15
                # Make metric columns slightly narrower
                elif col_name in ['F1', 'Distance', 'Dampening', 'LDJ'] or ' p' in col_name:
                    worksheet.column_dimensions[col_letter].width = 12
                # Make the calculated column wider
                elif col_name == 'Distance*Dampening/F1':
                    worksheet.column_dimensions[col_letter].width = 22
            
            # Find columns to hide (Std, Diff, and Stat columns)
            columns_to_hide = []
            for col_idx, col_name in enumerate(df.columns):
                if any(suffix in col_name for suffix in [' Std', ' Diff', ' Stat']):
                    col_letter = openpyxl.utils.get_column_letter(col_idx + 1)  # +1 because Excel columns start at 1
                    columns_to_hide.append(col_letter)
            
            # Hide those columns
            for col_letter in columns_to_hide:
                col_dimension = worksheet.column_dimensions[col_letter]
                col_dimension.hidden = True
            
            # Find p-value columns
            p_value_columns = [col for col in df.columns if ' p' in col]
            
            # Apply bold formatting to cells with p < 0.05
            for col_idx, col_name in enumerate(df.columns):
                if col_name in p_value_columns:
                    col_letter = openpyxl.utils.get_column_letter(col_idx + 1)  # +1 because Excel columns start at 1
                    
                    # Go through each row
                    for row_idx, p_value in enumerate(df[col_name]):
                        # Check if p-value is less than 0.05 (and not NA)
                        if p_value != 'NA' and not isinstance(p_value, str):
                            try:
                                p_val_float = float(p_value)
                                if p_val_float < 0.05:
                                    # Bold the cell
                                    cell = worksheet[f"{col_letter}{row_idx + 2}"]  # +2 for header row and 1-indexing
                                    cell.font = Font(bold=True)
                                    
                                    # Also bold the corresponding metric's mean, std, and diff cells
                                    metric_prefix = col_name.split(' p')[0]
                                    related_cols = [metric_prefix, f"{metric_prefix} Std", f"{metric_prefix} Diff"]
                                    
                                    for rel_col in related_cols:
                                        if rel_col in df.columns:
                                            rel_col_idx = df.columns.get_loc(rel_col)
                                            rel_col_letter = openpyxl.utils.get_column_letter(rel_col_idx + 1)
                                            rel_cell = worksheet[f"{rel_col_letter}{row_idx + 2}"]
                                            rel_cell.font = Font(bold=True)
                            except (ValueError, TypeError):
                                continue
        
        print(f"Combined results saved to {file_path} with formatting")
        
    except ImportError:
        # Fallback to CSV if openpyxl is not available
        print("Warning: openpyxl not available. Saving as CSV without formatting.")
        csv_file_path = os.path.join(output_dir, f"combined_mccv_analysis_{timestamp}.csv")
        df.to_csv(csv_file_path, index=False)
        print(f"Combined results saved to {csv_file_path}")

# Debugging function to print raw data
def print_raw_data(results):
    for metric in results.keys():
        print(f"\nMetric: {metric}")
        for model, values in results[metric].items():
            print(f"Model: {model}")
            print(f"Mean: {values['mean']}")
            print(f"Std Dev: {values['std']}")
            if 'difference' in values:
                print(f"Difference: {values['difference']}")
            print("---")
