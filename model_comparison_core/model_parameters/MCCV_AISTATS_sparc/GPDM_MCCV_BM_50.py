import numpy as np
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
from model_comparison_core.model_parameters.more_GPDM_params import get_params
from model_comparison_core.model_results_processing.dict_construction import construct_dict
import os

# Get the directory containing the script
param_dir = os.path.dirname(os.path.abspath(__file__))



model_type = 'gpdm'

### space
space = None
## universal




## quick ref
data_set_name = 'Bimanual 3D'

if data_set_name == "Bimanual 3D":
    actions = [0, 1, 2, 3, 4]
    num_sequences_per_action_test = 9
elif data_set_name == "Movements CMU":
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    num_sequences_per_action_test = 5
else:
    raise ValueError("Dataset does not exist.")
num_sequences_per_action_train = 1
num_representative_seqs = 1
seq_len = 100

##
people = [0]
input_dim = 24
output_dir_name = 'MCCV_f1_dist_ldj_'+data_set_name+'_GPDMM_Bayesian_50'
max_iters = 100
num_folds = 50
fold_start = 0
fold_end = 50
fold_list = list(np.arange(fold_start,fold_end,1))
fold_num = tuple(fold_list)
ind_percentages = np.array([.5])
num_inducing_latent = 0#tuple(int(seq_len*num_sequences_per_action_train*len(actions)*len(people))*ind_percentages)
num_inducing_dynamics = tuple(int(seq_len*num_sequences_per_action_train)*ind_percentages)
scoring_method = 'f1_dist_ldj:ff'
score_rate = int(max_iters*.1)
##



init1 = 'PCA'#('fourier_basis','random','PCA','kernel pca:rbf','random_sine_waves','random projections')
bc_name = 'map geo'#, 'multi w GP geo')
geometry = 'toroid'#, 'ellipse', 'toroid')
order = 2#,2)
BC_param_constraints = ['variance','A','R','n','r']
mapping = 'GPLVM'
pred_group = 0



prior_name = 'GPDMM'
prior_dynamics = 'ff'
num_sims = 1

pred_seq_len_ratio = .4




notes = ''





initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}
all_vars_dict = get_params(**input_vars_dict)
all_vars_dict.update(input_vars_dict)
combined_dicts, comp_tuples, data_tuples = construct_dict(**all_vars_dict)
