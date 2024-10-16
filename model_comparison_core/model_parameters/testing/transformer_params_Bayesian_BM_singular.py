import numpy as np
import matplotlib;
matplotlib.use("TkAgg")
from model_comparison_core.model_results_processing.dict_construction import construct_dict
from skopt.space import Real, Integer, Categorical





space = None#[]
# ------ Architecture END
## ------ Simulations
data_set_name = 'Bimanual 3D'
if data_set_name == "Bimanual 3D":
    actions = [0, 1, 2, 3, 4]
    num_sequences_per_action_test = 1
elif data_set_name == "Movements CMU":
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    num_sequences_per_action_test = 5
else:
    raise ValueError("Dataset does not exist.")
num_sequences_per_action_train = 1


##
output_dir_name = 'temp_'+data_set_name+'_Bayesian_Transformer'
#space.append(Integer(28,40,name='arch_dict:hidden_size_multiplier'))
#space.append(Integer(3,4,name='arch_dict:num_layers'))
#space.append(Integer(6,20,name='arch_dict:num_heads'))

num_epochs = 2
num_folds = 2*(num_sequences_per_action_train+num_sequences_per_action_test)
fold_start = 0
fold_end = 8
fold_list = list(np.arange(fold_start,fold_end,1))
fold_num = 0
num_sims = 1
scoring_method = 'f1_dist_msad:ff'
##


people = [0]
#num_representative_seqs = 1
seq_len = 100
model_type = 'Transformer'

num_classes = int(len(actions))
hidden_size_multiplier = 31#(8, 16, 32) ### hidden_size = num_heads*hidden_size_multiplier
num_layers = 2#(2, 3, 4)
num_heads = 17#(2, 4, 8)
dropout = .3#(.1,.2,.3)
pred_seq_len_ratio = .5
sample_len = int(seq_len*pred_seq_len_ratio)
# ------ Simulations END
initial_vars = set(globals().keys())
input_vars_dict = {var: globals().get(var) for var in initial_vars if not var.startswith('_')}

combined_dicts, comp_tuples, data_tuples = construct_dict(**input_vars_dict)



##

