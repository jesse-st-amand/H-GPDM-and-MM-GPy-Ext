import numpy as np
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
from HGPLVM.model_results_processing.model_parameters.all_params import get_params
from HGPLVM.model_results_processing.model_parameters.dict_construction import construct_dict

#data_set_name = 'Bimanual 3D'
data_set_name = 'Movements CMU'



### space
space = []
## universal
#space.append(Integer(4,14,name='input_dims_list-0'))
space.append(Categorical([int(1)],name='arch_dict:top_node_dict:prior_dict:order'))
space.append(Categorical([1],name='num_sequences_per_action_train'))
#space.append(Categorical([100],name='seq_len'))
#space.append(Categorical([True,False],name='BC_dict:ARD'))
## BC geo universal
#space.append(Categorical(['ellipse','toroid','sine','linear','spiral','mobius strip','klein bottle','fourier'],name='BC_dict:geometry'))
## GP BC


## quick ref
input_dim1 = 11
init1 = ('umap:euclidean','random projections', 'FFT_2D', 'FFT_3D')
bc_name = 'gp map'
geometry = 'none'
BC_param_constraints = ['variance']
mapping = 'GP'
pred_group = 0
seq_len = 100
actions = [0,1,2,3,4,5,6,7,8]
num_sequences_per_action_train = 1
num_representative_seqs = 1
num_sequences_per_action_test = 5
prior_name = 'GPDMM'
prior_dynamics = 'ff'
num_sims = 15
num_folds = num_sequences_per_action_train+num_sequences_per_action_test
fold_num = tuple(np.arange(0,num_folds,1))
pred_seq_len_ratio = .3

comparison_name = 'DR2_initial_conditions'
score_type = 'score_PVNPP' #pos_vel_norm_pred_prob
output_dir_name = comparison_name + '_' + score_type + '_' +'dim_'+str(input_dim1)+'_BC_'+bc_name.replace(' ', '_')+'_prior_'+prior_name+ '_'+ prior_dynamics +'_len_'+str(seq_len)
notes = ''

initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}
all_vars_dict = get_params(**input_vars_dict)
all_vars_dict.update(input_vars_dict)
combined_dicts, comp_tuples, data_tuples = construct_dict(**all_vars_dict)