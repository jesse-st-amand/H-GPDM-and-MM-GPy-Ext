import matplotlib
matplotlib.use("TkAgg")
import os
import psutil
import pickle
from joblib import Parallel, delayed



def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # Memory usage in MB


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


def get_last_processed_index(dir_path):
    files = os.listdir(dir_path)
    dict_indices = [int(f.split('_dict_')[1].split('_')[0]) for f in files if '_dict_' in f]
    return max(dict_indices) if dict_indices else -1


def process_combined_dict(combined_dict, dir_path, seed, dict_index, space):
    from HGPLVM.algorithm_sims import func_simulator

    data_dict = combined_dict[1]
    comp_dict = combined_dict[0]

    if data_dict['constant_args_dict']['data_set_name'] == 'Bimanual 3D':
        from HGPLVM.GPDMM_comp_data_func import multi_GPDMM as comp_func, Bimanual3D_constructor as data_func
    elif data_dict['constant_args_dict']['data_set_name'] == 'Movements CMU':
        from HGPLVM.GPDMM_comp_data_func import multi_GPDMM as comp_func, \
            MovementsCMU_constructor as data_func
    else:
        raise ValueError(f"Unrecognized data set {data_dict['constant_args_dict']}")

    FS = func_simulator(data_func, comp_func, space=space, parallelize=False)

    results = FS.run_sims(data_dict, comp_dict, seed, dict_index=dict_index, dir_path=dir_path)

    return dict_index


def run_batch(batch_start, batch_end, combined_dicts_file, dir_path, space_file):
    with open(combined_dicts_file, 'rb') as f:
        combined_dicts = pickle.load(f)
    with open(space_file, 'rb') as f:
        space = pickle.load(f)

    num_cores = max(1, psutil.cpu_count() // 2)

    processed_indices = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(process_combined_dict)(
            combined_dicts['dicts'][i], dir_path, combined_dicts['seed'], i, space
        ) for i in range(batch_start, batch_end)
    )

    return processed_indices


if __name__ == "__main__":
    names = [
        'input_params_sparse_ICs_BCs'
    ]

    for name in names:
        local_vars = {}
        string = f'from HGPLVM.model_parameters.{name} import output_dir_name, num_sims, combined_dicts, comp_tuples, data_tuples, space'
        exec(string, {}, local_vars)
        dir_path = "C:/Users/Jesse/Documents/Python/GPy/HGPLVM_output_repository/model_summaries/" + local_vars['output_dir_name']
        create_directory(dir_path)

        last_processed_index = get_last_processed_index(dir_path)

        num_cores = max(1, psutil.cpu_count() // 2)
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

                processed_indices = run_batch(batch_start, batch_end, combined_dicts_file, dir_path, space_file)

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


    print("All processing completed")