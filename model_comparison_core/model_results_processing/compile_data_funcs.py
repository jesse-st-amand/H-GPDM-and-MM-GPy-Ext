import os
import pandas as pd
import io

def try_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value

def safe_split_and_convert(composite, n):
    parts = composite.split('|||', n - 1)
    parts = parts + [''] * (n - len(parts))
    return [try_float(part) for part in parts]

def prepare_for_comparison(df, grouping_params):
    for param in grouping_params:
        if param in df.columns:
            df[param] = df[param].apply(try_float)
    return df

def debug_dataframe(df, name):
    print(f"\nDebugging {name}:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")

def process_file(filepath, parameters_to_include):
    print(f"Processing file: {os.path.basename(filepath)}")
    with open(filepath, 'r') as file:
        content = file.read()

    parts = content.split('--- ITERATION RESULTS ---')
    first_table = pd.read_csv(io.StringIO(parts[0]))
    second_table = pd.read_csv(io.StringIO(parts[1].strip()))

    # Find the maximum F1 score
    max_f1 = second_table['f1'].max()

    # Filter rows with maximum F1 score
    max_f1_rows = second_table[second_table['f1'] == max_f1]

    # Among the rows with max F1, find the one with minimum score
    best_row = max_f1_rows.loc[max_f1_rows['score'].idxmin()]

    best_score = best_row['score']
    best_iteration = best_row['iteration']
    best_f1 = best_row['f1']
    best_msad = best_row['msad']

    data = {}
    for param in parameters_to_include:
        if param == 'score':
            data[param] = best_score
        else:
            value = first_table.loc[first_table['Parameter'] == param, 'Optimized Value'].iloc[0] if param in first_table['Parameter'].values else 'none'
            data[param] = try_float(value)
        print(f"  {param}: {data[param]}")

    data['best_score_iteration'] = best_iteration
    data['best_f1'] = best_f1
    data['best_msad'] = best_msad
    data['Filename'] = os.path.basename(filepath)

    return data

def create_composite_group(row, grouping_params):
    return '|||'.join(str(try_float(row[param])) for param in grouping_params)

def process_data(full_path, parameters_to_include, grouping_params):
    all_data = []
    for filename in os.listdir(full_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(full_path, filename)
            try:
                data = process_file(filepath, parameters_to_include)
                all_data.append(data)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    result_df = pd.DataFrame(all_data)
    if result_df.empty:
        print("No data was successfully processed. Exiting.")
        return None

    result_df['composite_group'] = result_df.apply(lambda row: create_composite_group(row, grouping_params), axis=1)

    return result_df

def calculate_statistics(result_df, grouping_params):
    grouped_stats = result_df.groupby(grouping_params).agg({
        'score': ['mean', 'std', 'min', 'max', 'count'],
        'best_score_iteration': 'mean',
        'best_f1': ['mean', 'std', 'min', 'max'],
        'best_msad': ['mean', 'std', 'min', 'max']
    }).reset_index()

    grouped_stats.columns = [f'{col[0]}_{col[1]}' if isinstance(col, tuple) else col for col in grouped_stats.columns]

    for param in grouping_params:
        if param not in grouped_stats.columns:
            if f'{param}_' in grouped_stats.columns:
                grouped_stats = grouped_stats.rename(columns={f'{param}_': param})
            else:
                print(f"Warning: '{param}' column is missing from grouped_stats")

    grouped_stats = prepare_for_comparison(grouped_stats, grouping_params)

    grouped_stats = grouped_stats.rename(columns={
        'score_mean': 'Mean Best Score',
        'score_std': 'Std Dev Best Score',
        'score_min': 'Min of Best Scores',
        'score_max': 'Max of Best Scores',
        'score_count': 'Count',
        'best_score_iteration_mean': 'Avg Iteration of Best Score',
        'best_f1_mean': 'Mean Best F1 Score',
        'best_f1_std': 'Std Dev Best F1 Score',
        'best_f1_min': 'Min of Best F1 Scores',
        'best_f1_max': 'Max of Best F1 Scores',
        'best_msad_mean': 'Mean Best MSAD',
        'best_msad_std': 'Std Dev Best MSAD',
        'best_msad_min': 'Min of Best MSAD',
        'best_msad_max': 'Max of Best MSAD'
    })

    # Add the new column "score over f1"
    grouped_stats['score over f1'] = grouped_stats['Mean Best Score'] / grouped_stats['Mean Best F1 Score']

    # Reorder columns
    columns = grouping_params + [
        'Count', 'Avg Iteration of Best Score',
        'Mean Best F1 Score', 'Std Dev Best F1 Score', 'Min of Best F1 Scores', 'Max of Best F1 Scores',
        'Mean Best MSAD', 'Std Dev Best MSAD', 'Min of Best MSAD', 'Max of Best MSAD',
        'Mean Best Score', 'Std Dev Best Score', 'Min of Best Scores', 'Max of Best Scores',
        'score over f1'
    ]
    grouped_stats = grouped_stats[columns]

    return grouped_stats

def save_results(grouped_stats, output_file, grouping_params):
    with open(output_file, 'w', newline='') as f:
        f.write(f"Grouped by: {', '.join(grouping_params)}\n")
        f.write("Grouped Statistics:\n")
        grouped_stats.to_csv(f, index=False, float_format='%.6f')  # Format floating-point numbers to 6 decimal places

    print(f"\nAll results saved to '{output_file}'")

def main(full_path, parameters_to_include, grouping_params, output_file):
    result_df = process_data(full_path, parameters_to_include, grouping_params)
    if result_df is None:
        return

    grouped_stats = calculate_statistics(result_df, grouping_params)

    print("\nFinal Grouped Statistics:")
    print(grouped_stats)

    save_results(grouped_stats, output_file, grouping_params)