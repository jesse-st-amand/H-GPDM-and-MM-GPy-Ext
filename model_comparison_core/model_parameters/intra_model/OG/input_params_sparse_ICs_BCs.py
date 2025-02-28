import numpy as np
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
from HGPLVM.model_parameters.all_params import get_params
from HGPLVM.model_results_processing.dict_construction import construct_dict

#data_set_name = 'Bimanual 3D'
data_set_name = 'Movements CMU'



### space
space = None# []
## universal
#space.append(Integer(4,14,name='input_dims_list-0'))
#space.append(Categorical([1],name='num_sequences_per_action_train'))
#space.append(Categorical([100],name='seq_len'))
#space.append(Categorical([True,False],name='BC_dict:ARD'))
## BC geo universal
#space.append(Categorical(['ellipse','toroid','sine','linear','spiral','mobius strip','klein bottle','fourier'],name='BC_dict:geometry'))
## GP BC


## quick ref
input_dim1 = 15#, 21, 24, 27)#11#tuple(np.arange(5,20))
init1 = 'PCA'#('fourier_basis','random','PCA','kernel pca:rbf','random_sine_waves','random projections')
bc_name = ('map geo')#, 'multi w GP geo')
geometry = ('none')#, 'ellipse', 'toroid')
order = 2#,2)


BC_param_constraints = ['variance','A','R','n','r']
mapping = 'GPLVM'
pred_group = 0
seq_len = 100
actions = [0,1]#,2,3,4,5,6,7,8]
people = [0]
num_sequences_per_action_train = 1
num_representative_seqs = 1
num_sequences_per_action_test = 1
prior_name = 'GPDMM'
prior_dynamics = 'ff'
num_sims = 1
num_folds = 1#1*(num_sequences_per_action_train+num_sequences_per_action_test)
fold_num = tuple(np.arange(0,num_folds,1))
pred_seq_len_ratio = .3
max_iters = 100

ind_percentages = np.array([0])#,.25,.5,.75
num_inducing = tuple(int(seq_len*num_sequences_per_action_train*len(actions)*len(people))*ind_percentages)

comparison_name = 'temp_sparse_ICs_BCs'
score_type = 'score_PVNPP' #pos_vel_norm_pred_prob
output_dir_name = comparison_name + '_' + score_type + '_prior_'+prior_name+ '_'+ prior_dynamics +'_len_'+str(seq_len)
notes = ''

initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}
all_vars_dict = get_params(**input_vars_dict)
all_vars_dict.update(input_vars_dict)
combined_dicts, comp_tuples, data_tuples = construct_dict(**all_vars_dict)