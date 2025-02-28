import os
import shutil
import datetime
from compile_data_funcs import main as compile_main
def unpack_directory(source_dir):
    print(f"Attempting to unpack directory: {source_dir}")
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"The source directory does not exist: {source_dir}")

    parent_dir = os.path.dirname(source_dir)
    source_dir_name = os.path.basename(source_dir)
    copy_dir_name = f"unpacked_{source_dir_name}"
    copy_dir = os.path.join(parent_dir, copy_dir_name)

    print(f"Creating copy directory: {copy_dir}")
    os.makedirs(copy_dir, exist_ok=True)

    print(f"Copying files from {source_dir} to {copy_dir}")
    shutil.copytree(source_dir, copy_dir, dirs_exist_ok=True)

    working_dir = copy_dir
    sub_dirs = [f.path for f in os.scandir(working_dir) if f.is_dir()]

    for dir_path in sub_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(working_dir, file)
                shutil.move(src_path, dest_path)
        shutil.rmtree(dir_path)

    print(f"Finished unpacking all folders to {working_dir}")
    print("Original subdirectories have been removed.")
    return copy_dir

def process_model(model_name, model_summaries_sub_dir, grouping_params,smoothness_metric):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    base_path = base_dir + '/output_repository/model_summaries/' + model_summaries_sub_dir

    print(f"Processing model: {model_name}")
    print(f"Importing parameters from: model_comparison_core.model_parameters.{model_name}")

    # Dynamically import parameters
    local_vars = {}
    string = f'from model_comparison_core.model_parameters.{model_summaries_sub_dir}.{model_name} import *'
    exec(string, {}, local_vars)

    # Get the directory name from local_vars
    output_dir_name = local_vars.get('output_dir_name')
    if not output_dir_name:
        raise ValueError("output_dir_name not found in imported parameters")

    print(f"Directory name from parameters: {output_dir_name}")

    # Construct the full path
    full_path = os.path.join(base_path, model_name, output_dir_name)
    print(f"Full path to unpack: {full_path}")

    try:
        # Unpack the directory
        unpacked_dir = unpack_directory(full_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the directory exists and the path is correct.")
        return

    # Output directory
    output_base = os.path.abspath(
        base_dir+r'\output_repository\compiled_model_summaries')
    output_path = os.path.join(output_base, os.path.basename(unpacked_dir))

    print(f"Creating output directory: {output_path}")
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Output file name
    output_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S" + '.csv')
    output_file = os.path.join(output_path, output_filename)

    # Ensure the path uses forward slashes
    output_file = output_file.replace('\\', '/')
    print(f"Output file: {output_file}")

    parameters_to_include = ['score'] + grouping_params

    # Run the main compilation function
    compile_main(unpacked_dir, parameters_to_include, grouping_params, output_file, smoothness_metric=smoothness_metric)

def main():
    model_summaries_sub_dir = 'MCCV_AISTATS_sparc'
    data_set = 'BM'
    smoothness_metric = 'smoothness'
    models = [
        {
            'name': 'GPDM_MCCV_50'+data_set,
            'model_summaries_sub_dir': model_summaries_sub_dir,
            'grouping_params': ['input_dim']
        },
        {
            'name': 'RNN_MCCV_'+data_set,
            'model_summaries_sub_dir': model_summaries_sub_dir,
            'grouping_params': ['hidden_size', 'num_layers']
        },
        {
            'name': 'transformer_MCCV_'+data_set,
            'model_summaries_sub_dir': model_summaries_sub_dir,
            'grouping_params': ['hidden_size_multiplier', 'num_layers', 'num_heads', 'dropout']
        },
        {
            'name': 'VAE_MCCV_'+data_set,
            'model_summaries_sub_dir': model_summaries_sub_dir,
            'grouping_params': ['hidden_size', 'latent_size']
        }
    ]

    for model in models:
        process_model(model['name'], model['model_summaries_sub_dir'], model['grouping_params'], smoothness_metric)

if __name__ == "__main__":
    main()