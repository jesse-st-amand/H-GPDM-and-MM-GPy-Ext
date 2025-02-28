import numpy as np
from skopt.space import Real, Integer, Categorical

def get_params(**kwargs):
    data_set_name = kwargs['data_set_name']
    if data_set_name == "Bimanual 3D":
        actions = [0, 1, 2, 3, 4]
        num_sequences_per_action_test = 4
        num_sequences_per_action_validation = 5
        pred_seq_len_ratio = .4
        people = [0]
    elif data_set_name == "Movements CMU":
        actions = [0, 1, 2, 3, 4, 5, 6, 7]
        num_sequences_per_action_test = 2
        num_sequences_per_action_validation = 3
        pred_seq_len_ratio = .15
        people = [0]
    else:
        raise ValueError("Dataset does not exist.")
    num_sequences_per_action_train = 1
    num_representative_seqs = 1
    seq_len = 100
    ## ------ GPLVM dimensions
    #input_dim2 = 10
    #input_dim3 = 10
    active_dims = "all"
    num_inducing_latent = tuple(int(seq_len*num_sequences_per_action_train*len(actions)*len(people))*kwargs['ind_percentages_latent'])
    num_inducing_dynamics = tuple(int(seq_len*num_sequences_per_action_train)*kwargs['ind_percentages_dynamics'])


    # ------ GPLVM dimensions END
    ## ------ Initializations
    '''
    init = ''
    'none','random','random projections','isomap:5','pca','umap:cosine','kernel pca:rbf'
    '''
    #init2 = 'kernel pca:rbf'
    #init3 = 'kernel pca:rbf'
    X_init = None
    # ------ Initializations END
    ## ------ BCs
    '''
    bc_name = ''
    None,'tester','linear','linear geo','kernel','kernel geo','mlp','mlp geo','GP','sparse GP','GP geo','w GP geo',
    'multi w GP','multi w GP geo','sparse GP geo', 'gp map', 'gp map geo' (sparse not working w/o pred_var stuff)
    
    geometry = ''
    'ellipse','toroid'
    
    BC_param_constraints = ['']
    'A','variance','lengthscale','R','n','r'
    '''
    ARD=False
    num_epochs = 100
    # ------ BCs END
    ## ------ Dynamics
    '''
    prior_name = ''
    'GPDMM'
    
    prior_dynamics = ''
    'ff','fb'
    '''
    Y_uk_ID = 1
    init_t = 0
    pred_seq_len = int(pred_seq_len_ratio * seq_len)
    # ------ Dynamics END
    ## ------ Simulations
    parallelize = False
    save = False
    save_name = ""
    # ------ Simulations END
    ## ------ Optimization
    opt1 = 'lbfgsb'
    opt2 = 'lbfgsb'
    GPNode_opt = False
    GPNode_opt = False
    # ------ Optimization END
    local_vars = locals()
    final_vars = set(local_vars.keys()) - {'kwargs'}
    final_vars_dict = {var: local_vars[var] for var in final_vars if not var.startswith('_')}
    final_vars_dict.update(kwargs)
    return final_vars_dict


