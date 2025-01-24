import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from model_comparison_core.model_results_processing.dict_construction import construct_dict
from skopt.space import Real, Integer, Categorical
import os

# Get the directory containing the script
param_dir = os.path.dirname(os.path.abspath(__file__))

space = None
## ------ Simulations


data_set_name = 'Bimanual 3D'

if data_set_name == "Bimanual 3D":
    actions = [0, 1, 2, 3, 4]
    num_sequences_per_action_test = 1
elif data_set_name == "Movements CMU":
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    num_sequences_per_action_test = 1
else:
    raise ValueError("Dataset does not exist.")
num_sequences_per_action_train = 1


##
hidden_size = 692
latent_size = 5
output_dir_name = 'temp_singular_f1_dist_ldj_'+data_set_name+'_VAE'
#space.append(Integer(400,700,name='arch_dict:hidden_size'))
#space.append(Integer(5,7,name='arch_dict:latent_size'))

num_epochs = 30
num_folds = 1
fold_start = 0
fold_end = 1
fold_list = list(np.arange(fold_start,fold_end,1))
fold_num = tuple(fold_list)
scoring_method = 'f1_dist_ldj:ff'
score_rate = int(num_epochs*.1)
##


people = [0]
seq_len = 100
model_type = 'VAE'
num_sims = 1

num_classes = int(len(actions))

pred_seq_len_ratio = .4
sample_len = int(seq_len*pred_seq_len_ratio)
## ------ Simulations END

initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}

combined_dicts, comp_tuples, data_tuples = construct_dict(**input_vars_dict)
