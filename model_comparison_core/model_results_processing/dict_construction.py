import numpy as np
import copy
from itertools import product
import random
import time

# Function to remove space variables from dictionaries
def remove_space_vars_recursive(obj, space):
    if isinstance(obj, dict):
        return {k: remove_space_vars_recursive(v, space) for k, v in obj.items()
                if not any(k == param.name.split(':')[-1] for param in space)}
    elif isinstance(obj, list):
        return [remove_space_vars_recursive(item, space) for item in obj]
    else:
        return obj

def find_tuples_recursive(obj, path=[]):
    tuples = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            tuples.extend(find_tuples_recursive(v, path + [k]))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            tuples.extend(find_tuples_recursive(item, path + [i]))
    elif isinstance(obj, tuple):
        tuples.append((path, obj))
    return tuples

def set_value_at_path(obj, path, value):
    for key in path[:-1]:
        if isinstance(obj, list):
            while len(obj) <= key:
                obj.append({})
            obj = obj[key]
        else:
            if key not in obj:
                obj[key] = {}
            obj = obj[key]
    if isinstance(obj, list):
        while len(obj) <= path[-1]:
            obj.append(None)
    obj[path[-1]] = value

def path_exists(obj, path):
    current = obj
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return False
            current = current[key]
        elif isinstance(current, list):
            if not isinstance(key, int) or key >= len(current):
                return False
            current = current[key]
        else:
            return False
    return True

def create_combinations(comp_arg_dict, data_arg_dict):
    comp_tuples = find_tuples_recursive(comp_arg_dict)
    data_tuples = find_tuples_recursive(data_arg_dict)
    print('comp_tuples: ')
    for comp_tuple in comp_tuples:
        print(comp_tuple)
    print('data_tuples: ')
    for data_tuple in data_tuples:
        print(data_tuple)

    all_tuples = comp_tuples + data_tuples
    combinations = list(product(*[v for _, v in all_tuples]))

    result = []
    for combo in combinations:
        new_comp_dict = copy.deepcopy(comp_arg_dict)
        new_data_dict = copy.deepcopy(data_arg_dict)

        for (path, _), value in zip(all_tuples, combo):
            if path_exists(new_comp_dict, path):
                set_value_at_path(new_comp_dict, path, value)
            if path_exists(new_data_dict, path):
                set_value_at_path(new_data_dict, path, value)

        # Extract the seed from the comp_dict
        seed = new_comp_dict['seed']
        # Remove the seed from the comp_dict
        del new_comp_dict['seed']

        result.append((seed, new_comp_dict, new_data_dict))

    return result, comp_tuples, data_tuples

def generate_seeds(n, random = False):
    if random:
        base_seed = int(time.time() * 1000)
    else:
        base_seed = int(1000)
    seeds = []

    for i in range(n):
        new_seed = (base_seed + i * 1000003) % (2 ** 32 - 1)
        seeds.append(new_seed)

    return tuple(seeds)

def write_comp_dict(kwargs):
    if kwargs['model_type'].lower() == 'transformer':
        dict = {
            'attr_dict': {
                'model_type': kwargs['model_type'],  # or use model_type variable if you want to switch between LSTM and Transformer
                'num_epochs': kwargs['num_epochs']
            },
            'arch_dict': {
                'model_type': kwargs['model_type'],  # or use model_type variable
                'input_size': kwargs['D'],
                'num_classes': kwargs['num_classes'],
                'hidden_size_multiplier': kwargs['hidden_size_multiplier'],
                'num_layers': kwargs['num_layers'],
                'num_heads': kwargs['num_heads'],
                'dropout': kwargs['dropout'],
                'max_seq_length': kwargs['seq_len'],
                'sample_len': kwargs['sample_len'],
                'scoring_method': kwargs['scoring_method'],
                'score_rate': kwargs['score_rate']
            }
        }
    elif kwargs['model_type'].lower() == 'lstm':
        dict = {
            'attr_dict': {
                'model_type': kwargs['model_type'],
                'num_epochs': kwargs['num_epochs']
            },
            'arch_dict': {
                'model_type': kwargs['model_type'],
                'input_size': kwargs['D'],
                'num_classes': kwargs['num_classes'],
                'hidden_size': kwargs['hidden_size'],
                'num_layers': kwargs['num_layers'],
                'sample_len': kwargs['sample_len'],
                'scoring_method': kwargs['scoring_method'],
                'score_rate': kwargs['score_rate']
            }
        }
    elif kwargs['model_type'].lower() == 'vae':
        dict = {
            'attr_dict': {
                'model_type': kwargs['model_type'],
                'num_epochs': kwargs['num_epochs']
            },
            'arch_dict': {
                'model_type': kwargs['model_type'],
                'input_size': kwargs['D'],
                'num_classes': kwargs['num_classes'],
                'hidden_size': kwargs['hidden_size'],
                'latent_size': kwargs['latent_size'],
                'sample_len': kwargs['sample_len'],
                'scoring_method':kwargs['scoring_method'],
                'score_rate': kwargs['score_rate']
            }
        }
    elif kwargs['model_type'].lower() == 'gpdm':
        dynamics_dict = {'name': kwargs['prior_dynamics'], 'Y_indices': kwargs['Y_indices']}

        BC_dict = {'type': kwargs['bc_name'], 'geometry': kwargs['geometry'], 'num_epochs': kwargs['num_epochs'],
                   'Y1 indices': kwargs['Y_indices'][0], 'Y2 indices': kwargs['Y_indices'][1], 'Y3 indices': kwargs['Y_indices'][2],
                   'num_acts': len(kwargs['actions']), 'constraints': kwargs['BC_param_constraints'],
                   'ARD': kwargs['ARD'], 'mapping': kwargs['mapping']}
        prior_dict = {'name': kwargs['prior_name'], 'dynamics_dict': dynamics_dict, 'order': kwargs['order']}

        top_node_dict = {'attr_dict': {
            'input_dim': kwargs['input_dim'],
            'init': kwargs['init1'],
            'opt': kwargs['opt1'],
            'max_iters': kwargs['max_iters'],
            'GPNode_opt': kwargs['GPNode_opt'],
            'kernel': None,
            'num_inducing_latent': kwargs['num_inducing_latent'],
            'num_inducing_dynamics': kwargs['num_inducing_dynamics']},
            'BC_dict': BC_dict,
            'prior_dict': prior_dict}

        arch_dict = {
            'attr_dict': {'name': kwargs['arch_type'], 'Y_indices': kwargs['Y_indices'], 'pred_group': kwargs['pred_group'],
                          'init_t': kwargs['init_t'], 'sub_seq_len': kwargs['pred_seq_len'],
                          'scoring_method':kwargs['scoring_method'],'score_rate': kwargs['score_rate']
                          },
            'top_node_dict': top_node_dict
            }

        dict = {'attr_dict': {'model_type': kwargs['model_type'],'pred_group': kwargs['pred_group'], 'seq_len': kwargs['seq_len'],
                                  'pred_seq_len_ratio': kwargs['pred_seq_len_ratio'], 'max_iters': kwargs['max_iters']},
                    'arch_dict': arch_dict}
    else:
        raise ValueError('Unrecognized name.')
    return dict

def construct_dict(**kwargs):
    if kwargs['data_set_name'] == 'Bimanual 3D':
        d = 54
        D = 117
        kwargs['D'] = D
        Y3_indices = np.arange(0, D * d, 1)
        Y2_indices = np.arange(D, D, 1)
        Y1_indices = np.arange(0, D, 1)
    elif kwargs['data_set_name'] == 'Movements CMU':
        d = 20
        D = 77
        kwargs['D'] = D
        Y3_indices = np.arange(0, D - 2 * d, 1)
        Y2_indices = np.arange(D-d, D, 1)
        Y1_indices = np.arange(0, D-d, 1)
    elif kwargs['data_set_name'] == 'Bimanual 3D with EPs':
        d = 54
        D = 117
        kwargs['D'] = D
        extra_ds = 6
        Y3_indices = np.arange(0, D - 2 * d - extra_ds, 1)
        Y2_indices = np.arange(D, D + extra_ds - 3, 1)
        Y1_indices = np.arange(0, D + extra_ds - 3, 1)
    elif kwargs['data_set_name'] == 'Movements CMU with EPs':
        d = 20
        D = 77
        kwargs['D'] = D
        extra_ds = 15
        Y3_indices = np.arange(0, D - 2 * d - extra_ds, 1)
        Y2_indices = np.arange(D-d, D + extra_ds, 1)
        Y1_indices = np.arange(0, D-d, 1)
    ## ------ Dict construction
    kwargs['Y_indices'] = [Y1_indices,Y2_indices,Y3_indices]

    var_data_args_dict = {}

    comp_dict_temp = write_comp_dict(kwargs)


    # --- Data
    data_set_class_dict = {'data_set_name': kwargs['data_set_name'], 'actions': kwargs['actions'], 'people': kwargs['people'], 'seq_len': kwargs['seq_len'],
                           'num_folds': kwargs['num_folds'], 'fold_list': kwargs['fold_list'], 'fold_num': kwargs['fold_num'],
                           'num_sequences_per_action_train': kwargs['num_sequences_per_action_train'],
                           'num_sequences_per_action_test': kwargs['num_sequences_per_action_test'], 'X_init': None}

    const_data_args_dict = {'data_set_name': kwargs['data_set_name'], 'indices': kwargs['Y_indices'],
                            'seq_len': kwargs['seq_len'],
                            'actions': kwargs['actions'], 'people': kwargs['people'],
                            'num_sequences_per_action_train': kwargs['num_sequences_per_action_train'],
                            'num_sequences_per_action_test': kwargs['num_sequences_per_action_test']}
    var_data_args_dict = {'data_set_class_dict': [data_set_class_dict]}
    data_arg_dict = {'constant_args_dict' : const_data_args_dict, 'variable_args_dict':var_data_args_dict}
    # --- Comp
    const_comp_args_dict = {}
    var_comp_args_dict = {"model_dict": [comp_dict_temp]}
    comp_arg_dict = {'constant_args_dict' :const_comp_args_dict, 'variable_args_dict':var_comp_args_dict,'seed': generate_seeds(kwargs['num_sims'])}
    # ------ Dict construction END

    # Remove space variables from dictionaries
    if kwargs['space'] is not None:
        comp_arg_dict = remove_space_vars_recursive(comp_arg_dict, kwargs['space'])
        data_arg_dict = remove_space_vars_recursive(data_arg_dict, kwargs['space'])

    # Create combinations of dictionaries
    combined_dicts, comp_tuples, data_tuples = create_combinations(comp_arg_dict, data_arg_dict)

    # Group dictionaries by seed
    grouped_dicts = {}
    for seed, comp_dict, data_dict in combined_dicts:
        if seed not in grouped_dicts:
            grouped_dicts[seed] = []
        grouped_dicts[seed].append([comp_dict, data_dict])

    # Format the result as requested
    result = [{'seed': seed, 'dicts': dicts_list} for seed, dicts_list in grouped_dicts.items()]

    return result, comp_tuples, data_tuples