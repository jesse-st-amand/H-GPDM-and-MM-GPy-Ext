import copy
import random
import numpy as np
from HGPLVM.hgp import HGP
from HGPLVM.data_set_classes.data_set_classes import Bimanual3D, MovementsCMU
import matplotlib

matplotlib.use("TkAgg")
import os
import datetime
import csv
import json
def create_embedded_dict_from_colon_paths_with_mixed_values(list_of_keys, list_of_values):
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

def save_optimization_result(result, comp_dict, data_dict, dir_path, dict_index, sim_index, space_dict):
    filename = generate_filename(dir_path, comp_dict, dict_index, sim_index)
    write_results_to_csv(filename, result, comp_dict, data_dict, space_dict)

def write_results_to_csv(filename, result, comp_dict, data_dict, space_dict):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Parameter', 'Bound', 'Optimized Value'])

        opts = result['x']

        end_path_key_value(space_dict, writer)

        '''for bound, d_name, opt in zip(space.bounds, space.dimension_names, opts):
            writer.writerow([d_name, str(bound), opt])'''


        # Flatten and write the comp_dict
        flat_comp_dict = flatten_dict(comp_dict)

        exclude_keys = ['Y_indices', 'Y1 indices', 'Y2 indices', 'Y3 indices', 'num_inducing_list', 'arch_dict',
                        'prior_list-1', 'init_list', 'Y_uk_ID', 'pred_seq_len', 'Y_indices']

        end_path_key_value(flat_comp_dict, writer, exclude_keys)


        writer.writerow(['actions', '', json.dumps(list(np.array(data_dict['actions'])[data_dict['actions_indices']]))])
        writer.writerow(['people', '', json.dumps(data_dict['people'])])

        # Add a separator between tables
        writer.writerow([])
        writer.writerow(['--- ITERATION RESULTS ---'])
        writer.writerow([])

        writer.writerow(['iteration', 'score', "obj func value"])
        for i, (score, iter, obj) in enumerate(zip(result['fun'], result['iter'], result['ObjFunVal'])):
            writer.writerow([str(iter), score, obj])
def end_path_key_value(path_dict, writer, exclude_keys = []):
    for key, value in path_dict.items():
        last_key = key.split(':')[-1]
        last_key_numbered = key.split(':')[-1]
        if len(last_key_numbered.split('-')) > 1:
            last_key = last_key.split('-')[0]
        if last_key not in exclude_keys:
            writer.writerow([last_key_numbered, '', value])
def generate_filename(dir_path, comp_dict, dict_index, sim_index):
    return os.path.join(dir_path,
        f"{comp_dict['arch_dict']['top_node_dict']['BC_dict']['type'].replace(' ', '_')}_"
        f"{comp_dict['arch_dict']['top_node_dict']['prior_dict']['name']}_"
        f"{comp_dict['arch_dict']['top_node_dict']['prior_dict']['dynamics_dict']['name']}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_dict_{dict_index}_sim_{sim_index}.csv")

def multi_GPDMM(data_set_class, space_dict, hgp_dict, opts, seed=0, dict_index=0, sim_index=0, dir_path='', **kwargs):
    np.random.seed(int(seed))
    random.seed(int(seed))
    print('seed: ' + str(int(seed)))
    new_dict = create_embedded_dict_from_colon_paths_with_mixed_values(list(space_dict.keys()), list(space_dict.values()))
    new_hgp_dict = copy.deepcopy(merge_dicts(hgp_dict, new_dict))
    new_hgp_dict['attr_dict']['pred_seq_len'] = int(new_hgp_dict['attr_dict']['seq_len']*new_hgp_dict['attr_dict']['pred_seq_len_ratio'])
    hgp = HGP(new_hgp_dict, data_set_class=data_set_class)
    hgp.optimize(max_iters=new_hgp_dict['attr_dict']['max_iters'], optimizer=opts[1])
    hgp.get_attribute_dict()

    data_set_class.store_HGP_attributes(hgp)
    print(hgp.get_attribute_dict())

    # Save the result after optimization
    hgp.arch.model.score_list.append(hgp.score())
    hgp.arch.model.iter_list.append(hgp.arch.model.learning_n)
    result = {
        'space': space_dict,
        'x': hgp.get_attribute_dict(),  # Assuming this contains the optimized parameters
        'fun': hgp.arch.score_list,
        'iter': hgp.arch.iter_list,
        'ObjFunVal': hgp.arch.loss_list
    }
    save_optimization_result(result, new_hgp_dict, data_set_class.__dict__, dir_path, dict_index, sim_index, space_dict)


    return hgp



def Bimanual3D_constructor(space_dict,  data_set_class_dict, seed=0,  **kwargs):
    np.random.seed(int(seed))
    random.seed(int(seed))
    print('seed: '+str(int(seed)))
    new_dict = create_embedded_dict_from_colon_paths_with_mixed_values(list(space_dict.keys()),
                                                                       list(space_dict.values()))
    new_data_set_class_dict = copy.deepcopy(merge_dicts(data_set_class_dict, new_dict))
    num_seqs_per_action = 10
    data_set_class = Bimanual3D(new_data_set_class_dict['seq_len'], num_seqs_per_action,
                                num_folds=new_data_set_class_dict['num_folds'],
                                fold_num=new_data_set_class_dict['fold_num'])
    data_set_class.get_data_set(new_data_set_class_dict,new_data_set_class_dict['seq_len'],
                                new_data_set_class_dict['actions'], new_data_set_class_dict['people'],
                                       new_data_set_class_dict['num_sequences_per_action_train'],
                                       new_data_set_class_dict['num_sequences_per_action_test'])
    return data_set_class

def MovementsCMU_constructor(space_dict,  data_set_class_dict, seed=0,  **kwargs):
    np.random.seed(int(seed))
    random.seed(int(seed))
    print('seed: '+str(int(seed)))
    new_dict = create_embedded_dict_from_colon_paths_with_mixed_values(list(space_dict.keys()),
                                                                       list(space_dict.values()))
    new_data_set_class_dict = copy.deepcopy(merge_dicts(data_set_class_dict, new_dict))
    num_seqs_per_action = 6
    data_set_class = MovementsCMU(new_data_set_class_dict['seq_len'], num_seqs_per_action,
                                  num_folds=new_data_set_class_dict['num_folds'],
                                  fold_num=new_data_set_class_dict['fold_num'])
    data_set_class.get_data_set(new_data_set_class_dict,new_data_set_class_dict['seq_len'],
                                new_data_set_class_dict['actions'], new_data_set_class_dict['people'],
                                       new_data_set_class_dict['num_sequences_per_action_train'],
                                       new_data_set_class_dict['num_sequences_per_action_test'])
    return data_set_class