from compile_data_funcs import *
import datetime

if __name__ == "__main__":
    # Define your parameters here
    prefix = 'unpacked_'
    name = 'transformer_params_Bayesian_BM'
    path = 'C:/Users/Jesse/Documents/Python/GPy/HGPLVM_output_repository/model_summaries/' + name + '/'
    grouping_params = ['hidden_size_multiplier', 'num_layers', 'num_heads', 'dropout']
    baseline = (8, 2, 2, .1)

    local_vars = {}
    string = f'from model_comparison_core.model_parameters.{name} import *'
    exec(string, {}, local_vars)
    directory_name = prefix + local_vars['output_dir_name']
    full_path = path + directory_name + '/'

    output_dir = directory_name
    output_base = os.path.abspath('C:/Users/Jesse/Documents/Python/GPy/HGPLVM_output_repository/compiled_model_summaries/')
    output_path = os.path.join(output_base, output_dir)

    os.makedirs(output_path, exist_ok=True)

    output_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S" + '.csv')
    output_file = os.path.join(output_path, output_filename)
    output_file = output_file.replace('\\', '/')

    parameters_to_include = ['score'] + grouping_params

    main(full_path, parameters_to_include, grouping_params, baseline, output_file)