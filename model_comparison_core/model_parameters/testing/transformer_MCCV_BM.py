import numpy as np
import matplotlib;
matplotlib.use("TkAgg")
from model_comparison_core.model_results_processing.dict_construction import construct_dict
from skopt.space import Real, Integer, Categorical
import os

# Get the directory containing the script
param_dir = os.path.dirname(os.path.abspath(__file__))

space = None
data_set_name = 'Bimanual 3D'
##
hidden_size_multiplier = 35
num_layers = 2
num_heads = 13
dropout = .3
num_epochs = 2
num_folds = 1
fold_start = 0
fold_end = 1
fold_list = list(np.arange(fold_start,fold_end,1))
fold_num = tuple(fold_list)
scoring_method = 'f1_frechet_ldj:ff'
score_rate = 0
init_t = 0
##

num_sims = 1
people = [0]
#num_representative_seqs = 1
seq_len = 100
model_type = 'Transformer'
# ------ Simulations END
from model_comparison_core.model_parameters.more_inter_model_params import get_params
initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}
all_vars_dict = get_params(**input_vars_dict)
all_vars_dict.update(input_vars_dict)
combined_dicts, comp_tuples, data_tuples = construct_dict(**all_vars_dict)
pass



##

