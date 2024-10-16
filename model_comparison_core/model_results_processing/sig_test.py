import pandas as pd
import numpy as np
from scipy import stats

# Read the CSV file
input_file_path = r'C:/Users/Jesse/Documents/Python/GPy/HGPLVM_output_repository/model_comps/comp.csv'
df = pd.read_csv(input_file_path)

# Define the metrics we want to analyze and their corresponding column names
metrics = ['Mean Best F1 Score', 'Mean Best MSAD', 'Mean Best Score']
metric_columns = {
    'Mean Best F1 Score': {
        'std': 'Std Dev Best F1 Score',
        'min': 'Min of Best F1 Scores',
        'max': 'Max of Best F1 Scores'
    },
    'Mean Best MSAD': {
        'std': 'Std Dev Best MSAD',
        'min': 'Min of Best MSAD',
        'max': 'Max of Best MSAD'
    },
    'Mean Best Score': {
        'std': 'Std Dev Best Score',
        'min': 'Min of Best Scores',
        'max': 'Max of Best Scores'
    }
}

# Prepare DataFrames to store results
model_comparison_results = []
overall_statistics = []
model_rankings = []
performance_iterations_correlation = []

for metric in metrics:
    # Model Comparison
    for _, row in df.iterrows():
        model_comparison_results.append({
            'Metric': metric,
            'Model': row['model'],
            'Mean': row[metric],
            'Std Dev': row[metric_columns[metric]['std']],
            'Min': row[metric_columns[metric]['min']],
            'Max': row[metric_columns[metric]['max']],
            'Count (Folds)': row['Count']
        })

    # Overall Statistics
    overall_mean = df[metric].mean()
    overall_std = df[metric].std()
    cv = (overall_std / overall_mean) * 100
    overall_statistics.append({
        'Metric': metric,
        'Mean across all models': overall_mean,
        'Standard deviation across models': overall_std,
        'Coefficient of Variation (%)': cv
    })

    # Model Rankings
    rankings = df.sort_values(by=metric, ascending=False)
    for i, (_, row) in enumerate(rankings.iterrows(), 1):
        model_rankings.append({
            'Metric': metric,
            'Rank': i,
            'Model': row['model'],
            'Score': row[metric]
        })

    # Performance-Iterations Correlation
    correlation, p_value = stats.pearsonr(df['Avg Iteration of Best Score'], df[metric])
    performance_iterations_correlation.append({
        'Metric': metric,
        'Correlation': correlation,
        'P-value': p_value
    })

# Convert results to DataFrames
model_comparison_df = pd.DataFrame(model_comparison_results)
overall_statistics_df = pd.DataFrame(overall_statistics)
model_rankings_df = pd.DataFrame(model_rankings)
performance_iterations_correlation_df = pd.DataFrame(performance_iterations_correlation)

# Save results to CSV files
output_dir = r'C:/Users/Jesse/Documents/Python/GPy/HGPLVM_output_repository/analysis_results/'
model_comparison_df.to_csv(f'{output_dir}model_comparison_results.csv', index=False)
overall_statistics_df.to_csv(f'{output_dir}overall_statistics.csv', index=False)
model_rankings_df.to_csv(f'{output_dir}model_rankings.csv', index=False)
performance_iterations_correlation_df.to_csv(f'{output_dir}performance_iterations_correlation.csv', index=False)

print("Analysis results have been saved to CSV files in the 'analysis_results' directory.")

# Check if all models have the same number of folds
if df['Count'].nunique() == 1:
    print(f"All models used {df['Count'].iloc[0]} folds for cross-validation.")
else:
    print("Warning: Not all models used the same number of folds for cross-validation.")
    print(df[['model', 'Count']])