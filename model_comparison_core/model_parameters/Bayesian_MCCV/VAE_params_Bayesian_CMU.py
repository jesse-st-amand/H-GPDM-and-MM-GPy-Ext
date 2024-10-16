import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from model_comparison_core.model_results_processing.dict_construction import construct_dict
from skopt.space import Real, Integer, Categorical

space = []
## ------ Simulations


data_set_name = 'Movements CMU'

if data_set_name == "Bimanual 3D":
    actions = [0, 1, 2, 3, 4]
    num_sequences_per_action_test = 9
elif data_set_name == "Movements CMU":
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    num_sequences_per_action_test = 5
else:
    raise ValueError("Dataset does not exist.")
num_sequences_per_action_train = 1


##
output_dir_name = 'DOF_f1_dist_msad_'+data_set_name+'_Bayesian_VAE'
space.append(Integer(600,800,name='arch_dict:hidden_size'))
space.append(Integer(2,16,name='arch_dict:latent_size'))

num_epochs = 150
num_folds = 2*(num_sequences_per_action_train+num_sequences_per_action_test)
fold_start = 0
fold_end = 8
fold_list = list(np.arange(fold_start,fold_end,1))
fold_num = 0
scoring_method = 'f1_dist_msad:ff'
##


people = [0]
seq_len = 100
model_type = 'VAE'
num_sims = 1

num_classes = int(len(actions))
hidden_size = 600#(600,800,100)
latent_size = 4#(4, 8, 16)
pred_seq_len_ratio = .25
sample_len = int(seq_len*pred_seq_len_ratio)
## ------ Simulations END

initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}

combined_dicts, comp_tuples, data_tuples = construct_dict(**input_vars_dict)
