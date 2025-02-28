
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
import pickle
from joblib import Parallel, delayed
import numpy as np
import traceback
import shutil  # Add this import for directory operations
from model_comparison_core.comp_data_func import comp_func, data_func

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # Memory usage in MB

def create_directory(path):
    if path.split('/')[-1].startswith('temp_'):
        if os.path.exists(path):
            print(f"Temporary directory already exists. Removing: {path}")
            shutil.rmtree(path)
        os.makedirs(path)
        print(f"Temporary directory created: {path}")
    elif not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

def get_last_processed_index(dir_path):
    files = os.listdir(dir_path)
    dict_indices = [int(f.split('_dict_')[1].split('_')[0]) for f in files if '_dict_' in f]
    return max(dict_indices) if dict_indices else -1

def process_combined_dict_with_error_handling(data_func, comp_func, combined_dict, dir_path, seed, dict_index, space):
    try:
        from model_comparison_core.algorithm_sims import func_simulator

        data_dict = combined_dict[1]
        comp_dict = combined_dict[0]

        FS = func_simulator(data_func, comp_func, space=space, parallelize=False)

        results = FS.run_sims(data_dict, comp_dict, seed, dict_index=dict_index, dir_path=dir_path)

        return dict_index, results
    except Exception as e:
        error_message = f"Error processing dict_index {dict_index}: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return dict_index, {"error": error_message}

def run_batch(data_func, comp_func, num_cores, batch_start, batch_end, combined_dicts_file, dir_path, space_file):
    with open(combined_dicts_file, 'rb') as f:
        combined_dicts = pickle.load(f)
    with open(space_file, 'rb') as f:
        space = pickle.load(f)

    if batch_end - batch_start > 1 :
        results = Parallel(n_jobs=num_cores, verbose=10)(
            delayed(process_combined_dict_with_error_handling)(data_func, comp_func,
                combined_dicts['dicts'][i], dir_path, combined_dicts['seed'], i, space
            ) for i in range(batch_start, batch_end)
        )

        # Unpack the results
        processed_indices, processed_results = zip(*results)

        return processed_indices, processed_results
    else:
        return process_combined_dict_with_error_handling(data_func, comp_func, combined_dicts['dicts'][batch_start],
                              dir_path, combined_dicts['seed'], batch_start, space)

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)

    param_dir = 'inter_model'
    sub_dir = 'ICANN'  
    temp = False
    names = [
        'GPDM_geo_8_MCCV_BM',
    ]

    for name in names:
        local_vars = {}
        string = f'from model_comparison_core.model_parameters.{param_dir+'.'+sub_dir+'.'+name} import num_sims, combined_dicts, comp_tuples, data_tuples, space'
        exec(string, {}, local_vars)
        dir_path = parent_directory+"/output_repository/model_summaries/"+param_dir+"/"+sub_dir+"/"+name+"/"
        create_directory(dir_path)

        last_processed_index = -1 if temp else get_last_processed_index(dir_path)

        num_cores = max(1, psutil.cpu_count() - 6)
        batch_size = num_cores

        print(f"Starting processing with {num_cores} cores")
        print(f"Initial memory usage: {memory_usage():.2f} MB")

        combined_dicts_file = 'combined_dicts.pkl'
        space_file = 'space.pkl'

        for seed_dict in local_vars['combined_dicts']:
            with open(space_file, 'wb') as f:
                pickle.dump(local_vars['space'], f)
            with open(combined_dicts_file, 'wb') as f:
                pickle.dump(seed_dict, f)

            for batch_start in range(last_processed_index + 1, len(seed_dict['dicts']), batch_size):
                batch_end = min(batch_start + batch_size, len(seed_dict['dicts']))

                print(f"Processing batch {batch_start // batch_size + 1}")
                print(f"Memory usage before processing: {memory_usage():.2f} MB")

                processed_indices, results = run_batch(data_func, comp_func, num_cores, batch_start, batch_end, combined_dicts_file, dir_path, space_file)

                print(f"Finished batch. Processed indices: {processed_indices}")
                print(f"Memory usage after processing: {memory_usage():.2f} MB")
                print("---")

            for file_to_remove in [combined_dicts_file, space_file]:
                try:
                    if os.path.exists(file_to_remove):
                        os.remove(file_to_remove)
                        print(f"Successfully removed {file_to_remove}")
                    else:
                        print(f"File {file_to_remove} does not exist, skipping removal")
                except Exception as e:
                    print(f"Error removing {file_to_remove}: {e}")

            print(f"Finished processing {name}")
            print(f"Final memory usage: {memory_usage():.2f} MB")
        


