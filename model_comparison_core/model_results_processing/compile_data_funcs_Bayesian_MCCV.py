import os
import pandas as pd
import io
from statistics import mean


def try_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def process_file(filepath):
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
    best_f1 = best_row['f1']

    # Extract parameter values
    parameters = {}
    for _, row in first_table.iterrows():
        param = row['Parameter']
        value = try_float(row['Optimized Value'])
        parameters[param] = value

    return parameters, best_f1, best_score


def main(directory_path):
    all_best_f1 = []
    all_best_scores = []
    parameters = None

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory_path, filename)
            try:
                file_parameters, best_f1, best_score = process_file(filepath)
                all_best_f1.append(best_f1)
                all_best_scores.append(best_score)

                # Store parameters (assuming they're the same for all files)
                if parameters is None:
                    parameters = file_parameters
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    if not all_best_f1 or not all_best_scores:
        print("No data was successfully processed. Exiting.")
        return None, None, None

    avg_best_f1 = mean(all_best_f1)
    avg_best_score = mean(all_best_scores)

    return avg_best_f1, avg_best_score, parameters



