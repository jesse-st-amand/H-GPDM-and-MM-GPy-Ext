import numpy as np
from skopt.space import Real, Integer, Categorical

def get_params(**kwargs):
    data_set_name = kwargs['data_set_name']
    if data_set_name == "Bimanual 3D":
        actions = [0, 1, 2, 3, 4]
        num_sequences_per_action_test = 4
        num_sequences_per_action_validation = 5
        pred_seq_len_ratio = .4
        people = [0]
    elif data_set_name == "Movements CMU":
        actions = [0, 1, 2, 3, 4, 5, 6, 7]
        num_sequences_per_action_test = 2
        num_sequences_per_action_validation = 3
        pred_seq_len_ratio = .15
        people = [0]
    else:
        raise ValueError("Dataset does not exist.")
    num_sequences_per_action_train = 1
    num_representative_seqs = 1
    seq_len = 100
    Y_uk_ID = 1
    init_t = 0
    sample_len = int(pred_seq_len_ratio * seq_len)
    # ------ Dynamics END
    ## ------ Simulations
    parallelize = False
    save = False
    save_name = ""
    # ------ Simulations END
    ## ------ Optimization
    # ------ Optimization END
    local_vars = locals()
    final_vars = set(local_vars.keys()) - {'kwargs'}
    final_vars_dict = {var: local_vars[var] for var in final_vars if not var.startswith('_')}
    final_vars_dict.update(kwargs)
    return final_vars_dict


