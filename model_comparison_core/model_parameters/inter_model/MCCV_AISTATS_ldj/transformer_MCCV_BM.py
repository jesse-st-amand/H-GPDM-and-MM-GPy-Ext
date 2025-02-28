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
output_dir_name = 'MCCV_f1_dist_ldj_'+data_set_name+'_Bayesian_Transformer'
#space.append(Integer(25,45,name='arch_dict:hidden_size_multiplier'))
#space.append(Integer(2,4,name='arch_dict:num_layers'))
#space.append(Integer(2,18,name='arch_dict:num_heads'))

num_epochs = 100
num_folds = 50
fold_start = 0
fold_end = 50
fold_list = list(np.arange(fold_start,fold_end,1))
fold_num = tuple(fold_list)
scoring_method = 'f1_dist_ldj:ff'
score_rate = int(num_epochs*.1)
init_t = 0
##

num_sims = 1
people = [0]
#num_representative_seqs = 1
seq_len = 100
model_type = 'Transformer'

num_classes = int(len(actions))

pred_seq_len_ratio = .4
sample_len = int(seq_len*pred_seq_len_ratio)
# ------ Simulations END
initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}

combined_dicts, comp_tuples, data_tuples = construct_dict(**input_vars_dict)



##

