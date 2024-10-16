from compile_data_funcs import *
import datetime

if __name__ == "__main__":
    # Define your parameters here
    prefix = 'unpacked_'
    name = 'GPDM_params_Bayesian_BM_50'
    path = 'C:/Users/Jesse/Documents/Python/GPy/HGPLVM_output_repository/model_summaries/' + name + '/'
    # Parameters to include in the combined CSV
    grouping_params = ['input_dim']
    # Set up baseline
    baseline = (15,)

    local_vars = {}
    string = f'from model_comparison_core.model_parameters.{name} import *'
    exec(string, {}, local_vars)
    directory_name = prefix + local_vars['output_dir_name']
    full_path = path + directory_name + '/'
    # Output directory
    output_dir = directory_name
    output_base = os.path.abspath(
        'C:/Users/Jesse/Documents/Python/GPy/HGPLVM_output_repository/compiled_model_summaries/')
    output_path = os.path.join(output_base, output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Output file name
    output_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S" + '.csv')
    output_file = os.path.join(output_path, output_filename)

    # Ensure the path uses forward slashes
    output_file = output_file.replace('\\', '/')

    parameters_to_include = ['score'] + grouping_params

    main(full_path, parameters_to_include, grouping_params, baseline, output_file)