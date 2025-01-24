import os
import pandas as pd
import numpy as np
from scipy import stats
import glob
from datetime import datetime
import io


def process_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    parts = content.split('--- ITERATION RESULTS ---')
    second_table = pd.read_csv(io.StringIO(parts[1].strip()))

    print(f"Processing file: {filepath}")
    print(f"Raw data (first 5 rows):\n{second_table[['iteration', 'score', 'f1', 'smoothness', 'avg_freeze']].head()}")

    # Find the maximum F1 score
    max_f1 = second_table['f1'].max()

    # Filter rows with maximum F1 score
    max_f1_rows = second_table[second_table['f1'] == max_f1]

    # Among the rows with max F1, find the one with minimum score
    best_row = max_f1_rows.loc[max_f1_rows['score'].idxmin()]

    print(f"Best row:\n{best_row[['iteration', 'score', 'f1', 'smoothness', 'avg_freeze']]}")

    result = {
        'score': best_row['score'],
        'f1': best_row['f1'],
        'smoothness': best_row['smoothness'],
        'freeze': best_row['avg_freeze']
    }

    print(f"Returned result: {result}")

    return result


def get_model_results(path):
    results = {}
    for file in glob.glob(os.path.join(path, '*.csv')):
        sim_num = int(file.split('_')[-3].split('.')[0])
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
    control_name = os.path.basename(os.path.dirname(control_path))

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
        model_name = os.path.basename(os.path.dirname(test_path))
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for metric in results.keys():
        rows = []
        control_name = list(results[metric].keys())[0]  # Assume the first model is the control

        # Add control row
        rows.append({
            'Model': control_name,
            'Mean': results[metric][control_name]['mean'],
            'Std Dev': results[metric][control_name]['std'],
            'Difference': '-',
            'Statistic': '-',
            'p-value': '-'
        })

        # Add other models
        for model, values in results[metric].items():
            if model != control_name:
                rows.append({
                    'Model': model,
                    'Mean': values['mean'],
                    'Std Dev': values['std'],
                    'Difference': values['difference'],
                    'Statistic': values['statistic'],
                    'p-value': values['p_value']
                })

        df = pd.DataFrame(rows)

        file_name = f"mccv_analysis_{metric}_{timestamp}.csv"
        file_path = os.path.join(output_dir, file_name)

        df.to_csv(file_path, index=False)
        print(f"Results for {metric} saved to {file_path}")

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
