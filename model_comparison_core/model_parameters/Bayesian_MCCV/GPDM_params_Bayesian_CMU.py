import numpy as np
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
from model_comparison_core.model_parameters.more_GPDM_params import get_params
from model_comparison_core.model_results_processing.dict_construction import construct_dict



model_type = 'gpdm'

### space
space = []#None
## universal




## quick ref
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
num_representative_seqs = 1


##
output_dir_name = 'f1_dist_msad_'+data_set_name+'_GPDMM_Bayesian'
space.append(Integer(8,20,name='arch_dict:top_node_dict:attr_dict:input_dim'))
max_iters = 300
num_folds = 2*(num_sequences_per_action_train+num_sequences_per_action_test)
fold_start = 0
fold_end = 8
fold_list = list(np.arange(fold_start,fold_end,1))
fold_num = 0
ind_percentages = np.array([0])
num_inducing_latent = 0#tuple(int(seq_len*num_sequences_per_action_train*len(actions)*len(people))*ind_percentages)
num_inducing_dynamics = 0#tuple(int(seq_len*num_sequences_per_action_train)*ind_percentages)
scoring_method = 'f1_dist_msad:ff'
##


input_dim = 15#(15,16)#, 21, 24, 27)#11#tuple(np.arange(5,20))
init1 = 'PCA'#('fourier_basis','random','PCA','kernel pca:rbf','random_sine_waves','random projections')
bc_name = 'map geo'#, 'multi w GP geo')
geometry = 'toroid'#, 'ellipse', 'toroid')
order = 2#,2)
BC_param_constraints = ['variance','A','R','n','r']
mapping = 'GPLVM'
pred_group = 0
seq_len = 100
people = [0]

prior_name = 'GPDMM'
prior_dynamics = 'ff'
num_sims = 1

pred_seq_len_ratio = .25


notes = ''





initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}
all_vars_dict = get_params(**input_vars_dict)
all_vars_dict.update(input_vars_dict)
combined_dicts, comp_tuples, data_tuples = construct_dict(**all_vars_dict)
