from model_comparison_core.wrapper import TransformerModelWrapper, RNN_Wrapper, VAEModelWrapper
import copy
import random
import numpy as np
from HGPLVM.hgp import HGP
import matplotlib
matplotlib.use("TkAgg")
import os
import datetime
import csv
import json
import pickle
def create_embedded_dict_from_space_dict_colon_paths_with_mixed_values(space_dict):
    if space_dict is None:
        return {}
    list_of_keys = list(space_dict.keys())
    list_of_values = list(space_dict.values())
    root = {}
    for path, value in zip(list_of_keys, list_of_values):
        # Check if the path has an index (e.g., 'a:b-1')
        if '-' in path:
            path, index = path.split('-')
            index = int(index)  # Convert index to integer
        else:
            index = None

        keys = path.split(':')
        current_level = root
        for i, key in enumerate(keys[:-1]):  # Navigate through all but the last key
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]

        final_key = keys[-1]
        # Check if value is a string and contains a colon, then convert to tuple
        if isinstance(value, str) and ':' in value:
            value = tuple(value.split(':'))

        if index is not None:
            # Initialize the value container appropriately if it doesn't exist
            if final_key not in current_level:
                current_level[final_key] = () if isinstance(value, tuple) else []
            # Check the type of the existing container
            if isinstance(current_level[final_key], tuple) and isinstance(value, tuple):
                current_level[final_key] += value  # Concatenate tuples
            elif not isinstance(current_level[final_key], tuple):
                current_level[final_key].append(value)  # Append to list for non-tuples
        else:
            current_level[final_key] = value  # Assign the actual value if no index

    return root
def flatten_dict(d, parent_key='', sep=':'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}{'-'}{i}", sep='-').items())
                else:
                    items.append((f"{new_key}{'-'}{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)
def merge_dicts(d1, d2):
    """
    Merge two dictionaries recursively.
    For each matching key:
    - If both values are tuples, concatenate them.
    - Otherwise, replace the value from d1 with the value from d2.
    """
    for key, value in d2.items():
        if key in d1:
            if isinstance(value, dict) and isinstance(d1[key], dict):
                merge_dicts(d1[key], value)
            elif isinstance(value, tuple) and isinstance(d1[key], tuple):
                d1[key] = list(value)  # Concatenate tuples
            else:
                d1[key] = value  # Replace the value from d1 with d2
        else:
            d1[key] = value  # Add the new key from d2 to d1
    return d1

def save_optimization_result(result, comp_dict, data_dict, dir_path, dict_index, sim_index, space_dict=None):
    filename = generate_filename(dir_path, comp_dict, dict_index, sim_index)
    write_results_to_csv(filename, result, comp_dict, data_dict, space_dict)


def write_results_to_csv(filename, result, comp_dict, data_dict, space_dict=None):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write first table
        writer.writerow(['Parameter', 'Optimized Value'])

        exclude_keys = ['Y_indices', 'Y1 indices', 'Y2 indices', 'Y3 indices', 'num_inducing_list', 'arch_dict',
                        'prior_list-1', 'init_list', 'Y_uk_ID', 'pred_seq_len', 'Y_indices']
        end_path_key_value(space_dict, writer, exclude_keys)

        flat_comp_dict = flatten_dict(comp_dict)
        end_path_key_value(flat_comp_dict, writer, exclude_keys)

        writer.writerow(['actions', json.dumps(list(np.array(data_dict['actions'])[data_dict['actions_indices']]))])
        writer.writerow(['people', json.dumps(data_dict['people'])])

        # Add a separator between tables
        writer.writerow([])
        writer.writerow(['--- ITERATION RESULTS ---'])
        writer.writerow([])

        # Write second table
        writer.writerow(['iteration', 'score', "loss", 'predicted classes', 'f1', 'smoothness'])
        for i, (score, iter, loss, pc, f1, smoothness) in enumerate(zip(result['fun'], result['iter'], result['loss'],
                                                    result['pred_classes'], result['f1'], result['smoothness'])):
            writer.writerow([str(iter), score, loss, pc, f1, smoothness])
def end_path_key_value(path_dict, writer, exclude_keys = []):
    if path_dict is None:
        return
    for key, value in path_dict.items():
        last_key = key.split(':')[-1]
        last_key_numbered = key.split(':')[-1]
        if len(last_key_numbered.split('-')) > 1:
            last_key = last_key.split('-')[0]
        if last_key not in exclude_keys:
            writer.writerow([last_key_numbered, value])
def generate_filename(dir_path, comp_dict, dict_index, sim_index):
    return os.path.join(dir_path,
        f"{comp_dict['attr_dict']['model_type'].replace(' ', '_')}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_dict_{dict_index}_fold_{sim_index}.csv")

def comp_func(data_set_class,  model_dict, seed=0, dict_index=0, fold_num=1, save_path='', space_dict=None, return_model = False, **kwargs):
    np.random.seed(int(seed))
    random.seed(int(seed))
    new_dict = create_embedded_dict_from_space_dict_colon_paths_with_mixed_values(space_dict)
    new_model_dict = copy.deepcopy(merge_dicts(model_dict, new_dict))

    if new_model_dict['attr_dict']['model_type'].lower() == 'gpdm':
        new_model_dict['attr_dict']['pred_seq_len'] = int(
            new_model_dict['attr_dict']['seq_len'] * new_model_dict['attr_dict']['pred_seq_len_ratio'])
        model = HGP(new_model_dict, data_set_class=data_set_class)
        model.optimize(max_iters=new_model_dict['attr_dict']['max_iters'], optimizer=new_model_dict['arch_dict']['top_node_dict']['attr_dict']['opt'])
        model.get_attribute_dict()
        data_set_class.store_HGP_attributes(model)
        print(model.get_attribute_dict())
    elif new_model_dict['attr_dict']['model_type'].lower() == 'vae':
        model = VAEModelWrapper(new_model_dict, data_set_class=data_set_class)
        model.optimize(num_epochs=model_dict['attr_dict']['num_epochs'])
    elif new_model_dict['attr_dict']['model_type'].lower() == 'transformer':
        model = TransformerModelWrapper(new_model_dict, data_set_class=data_set_class)
        criterion_classification, criterion_generation, optimizer = model.setup()
        model.optimize(criterion_classification, criterion_generation, optimizer,
                       num_epochs=model_dict['attr_dict']['num_epochs'])
    elif new_model_dict['attr_dict']['model_type'].lower() == 'lstm':
        model = RNN_Wrapper(new_model_dict, data_set_class=data_set_class)
        criterion_classification, criterion_generation, optimizer = model.setup()
        model.optimize(criterion_classification, criterion_generation, optimizer,
                       num_epochs=model_dict['attr_dict']['num_epochs'])
    else:
        raise ValueError(f'Unknown model type {new_model_dict["attr_dict"]["model_type"]}')

    score = model.score()
    result = {
        'space': space_dict,
        'fun': model.arch.score_list,
        'iter': model.arch.iter_list,
        'loss': model.arch.loss_list,
        'pred_classes': model.arch.pred_classes,
        'f1': model.arch.f1_list,
        'smoothness': model.arch.smoothness_list
    }


    save_optimization_result(result, new_model_dict, data_set_class.__dict__, save_path, dict_index, fold_num,
                             space_dict)

    if return_model:
        return score, model
    else:
        return score


def data_func(data_set_class_dict, fold_num=0, seed=0, space_dict=None, **kwargs):
    return load_mccv_data(data_set_class_dict['data_set_name'], fold_num)

def load_mccv_data(data_set_name, fold_num):
    # Define the path where the pickled files are stored
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    load_path = parent_directory+f'/DSCs/data/MCCV/{data_set_name}/'

    # Initialize an empty list to store the loaded objects

    file_path = os.path.join(load_path, f'data_set_class_{fold_num}.pkl')

    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data_set_class = pickle.load(f)
        print(f"Loaded data_set_class for fold {fold_num} from {file_path}")
    else:
        print(f"Warning: File not found for fold {fold_num} at {file_path}")

    return data_set_class