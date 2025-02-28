import numpy as np
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
from model_comparison_core.model_parameters.more_GPDM_params import get_params
from model_comparison_core.model_results_processing.dict_construction import construct_dict
import os

# Get the directory containing the script
param_dir = os.path.dirname(os.path.abspath(__file__))

model_type = 'X1_H1_Y1'
space = None
data_set_name = 'Bimanual 3D'
##
people = [0]
input_dim = 9

max_iters = 100
num_folds = 1
fold_start = 0
fold_end = 1
fold_list = list(np.arange(fold_start,fold_end,1))
fold_num = 1
ind_percentages_latent = np.array([0])
ind_percentages_dynamics = np.array([0])
scoring_method = 'f1_frechet_ldj:ff'
score_rate = 10
##



init1 = 'kernel pca:rbf'#('fourier_basis','random','PCA','kernel pca:rbf','random_sine_waves','random projections')
bc_name = 'map geo'#, 'multi w GP geo')
geometry = 'fourier_basis'#, 'ellipse', 'toroid')
geo_params = 5
order = 1
BC_param_constraints = ['variance','A']#,'alpha','f','phi']
mapping = 'GPLVM'
pred_group = 0

init2 = 'kernel pca:rbf'
input_dim2 = 20
bc_name2 = 'none'
geometry2 = 'none'
geo_params2 = 0
BC_param_constraints2 = ['']
mapping2 = 'GPLVM'



prior_name = 'GPDMM'
prior_dynamics = 'ff'
num_sims = 1






notes = ''





initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}
all_vars_dict = get_params(**input_vars_dict)
all_vars_dict.update(input_vars_dict)
combined_dicts, comp_tuples, data_tuples = construct_dict(**all_vars_dict)
i0 = 0
