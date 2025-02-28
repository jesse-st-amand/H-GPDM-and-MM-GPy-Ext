from model_comparison_core.model_results_processing.compile_data_funcs_Bayesian_MCCV import main
import datetime
import os

def CRI(model_dict, csv_path='', **kwargs):
    # Define your parameters here
    out_path = os.path.join(*csv_path.split('/')[-3:])
    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.abspath(
        out_dir+'/output_repository/compiled_model_summaries/' + out_path )

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Output file name
    output_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S" + '.csv')
    output_file = os.path.join(output_path, 't' ,output_filename)

    # Ensure the path uses forward slashes
    output_file = output_file.replace('\\', '/')

    avg_best_f1, avg_best_score, parameters = main(csv_path)
    if avg_best_f1 >= 0:
        return avg_best_score/avg_best_f1
    else:
        return 1000000
