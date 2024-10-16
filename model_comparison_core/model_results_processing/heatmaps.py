import os
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def process_csv_files(directory):
    results = defaultdict(lambda: defaultdict(float))

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                csv_reader = csv.reader(file)
                iteration_results = False
                lowest_score_row = None
                lowest_score = float('inf')

                for row in csv_reader:
                    if row and row[0] == '--- ITERATION RESULTS ---':
                        iteration_results = True
                        continue

                    if iteration_results and len(row) >= 4:
                        try:
                            score = float(row[1])
                            if score < lowest_score:
                                lowest_score = score
                                lowest_score_row = row
                        except ValueError:
                            continue

                if lowest_score_row:
                    pred_classes = ast.literal_eval(lowest_score_row[3])
                    for pred, gt in zip(pred_classes['pred'], pred_classes['gt']):
                        results[gt][pred] += 1

    return results


def ensure_dir_exists(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def create_heatmap(results, save_dir, save_name):
    ensure_dir_exists(save_dir)

    classes = sorted(set(results.keys()) | set(k for v in results.values() for k in v.keys()))
    matrix = np.zeros((len(classes), len(classes)))

    for gt in classes:
        for pred in classes:
            matrix[classes.index(gt)][classes.index(pred)] = results[gt][pred]

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth Class')
    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name + '.png')
    plt.savefig(save_dir, dpi=300)
    plt.close()

    print(f"Heatmap has been saved as '{save_dir}'")




