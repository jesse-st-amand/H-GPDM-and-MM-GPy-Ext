from .data_set_classes import Bimanual3D, MovementsCMU
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation
def data_func(data_set_class_dict):
    dataset_name = data_set_class_dict['data_set_name']
    seq_len = data_set_class_dict['seq_len']
    num_folds = data_set_class_dict['num_folds']
    fold_num = data_set_class_dict['fold_num']

    if dataset_name == 'Bimanual 3D':
        num_seqs_per_action = 10
        data_set_class = Bimanual3D(seq_len, num_seqs_per_action, num_folds, fold_num)
    elif dataset_name == 'Movements CMU':
        num_seqs_per_action = 6
        data_set_class = MovementsCMU(seq_len, num_seqs_per_action, num_folds, fold_num)
    else:
        raise ValueError("Invalid dataset name.")

    data_set_class.get_data_set(
        data_set_class_dict,
        seq_len,
        data_set_class_dict['actions'],
        data_set_class_dict['people'],
        data_set_class_dict['num_sequences_per_action_train'],
        data_set_class_dict['num_sequences_per_action_test']
    )
    return data_set_class
