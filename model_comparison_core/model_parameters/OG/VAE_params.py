import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from model_comparison_core.model_results_processing.dict_construction import construct_dict

## BC geo universal
space = None

## ------ Simulations
output_dir_name = 'combined_comparison_VAE_hidden_latent'

data_set_name = 'Bimanual 3D'
#data_set_name = 'Movements CMU'

if data_set_name == "Bimanual 3D":
    actions = [0, 1, 2, 3, 4]
    num_sequences_per_action_test = 9
elif data_set_name == "Movements CMU":
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    num_sequences_per_action_test = 5
else:
    raise ValueError("Dataset does not exist.")
people = [0]
num_sequences_per_action_train = 1
seq_len = 100
num_folds = 5*(num_sequences_per_action_train+num_sequences_per_action_test)
fold_num = tuple(np.arange(0,num_folds,1))
num_sims = 1
scoring_method = 'joint_angles:ff'

model_type = 'VAE'
num_epochs = 1500

num_classes = int(len(actions))
hidden_size = (600,800,100)
latent_size = (4, 8, 16)
pred_seq_len_ratio = .5
sample_len = int(seq_len*pred_seq_len_ratio)
## ------ Simulations END

initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}

combined_dicts, comp_tuples, data_tuples = construct_dict(**input_vars_dict)
