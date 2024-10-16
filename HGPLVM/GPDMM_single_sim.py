import numpy as np
import random
#import matplotlib;
#matplotlib.use("TkAgg")
import matplotlib.animation as animation
from model_comparison_core.algorithm_sims import func_simulator
from model_comparison_core.comp_data_func import comp_func, data_func

data_set_name = 'Bimanual 3D'
#data_set_name = 'Movements CMU'





## ------ GPLVM dimensions
input_dim1 = 28
input_dim2 = 10
input_dim3 = 10
active_dims = "all"
# ------ GPLVM dimensions END
## ------ Initializations
'''
init = ''
'none','random','random projections','isomap:15','pca','umap:cosine','kernel pca:rbf'
'''
init1 = 'pca'
init2 = 'kernel pca:rbf'
init3 = 'kernel pca:rbf'
X_init = None
# ------ Initializations END
## ------ BCs
'''
bc_name = ''
None,'gp map geo','tester','linear','linear geo','kernel','kernel geo','mlp','mlp geo','GP','sparse GP','GP geo','w GP geo',
'multi w GP','multi w GP geo','sparse GP geo', 'gp map', 'gp map geo' (sparse not working w/o pred_var stuff)

geometry = ''
'ellipse','toroid'

BC_param_constraints = ['']
'A','variance','lengthscale','R','n','r'
'''
bc_name = 'map geo'
geometry = 'toroid'
geo_params = 3

BC_param_constraints = ['variance','A','R','n','r']
mapping = 'GPLVM'
pred_group = 0

ARD=False
num_epochs = 100
hidden_dim = 'input_dim1'
# ------ BCs END
## ------ Architecture
'''
arch_type = ''
'X1_Y1','X1_H2_Y2'
'''
arch_type = 'X1_Y1'
seq_len = 100
pred_seq_len_ratio = .5

# ------ Architecture END
## ------ Dynamics
'''
prior_name = ''
'GPDMM'

prior_dynamics = ''
'ff','fb'
'''
prior_name = 'GPDMM'
prior_dynamics = 'ff'
order = 2
Y_uk_ID = 1


init_t = 0
pred_seq_len = int(pred_seq_len_ratio * seq_len)
# ------ Dynamics END
## ------ Simulations
actions = [0,1]#,2,3,4]
people = [0]
num_sequences_per_action_train = 1
num_representative_seqs = num_sequences_per_action_train
num_sequences_per_action_test = 1
seed = 100000
parallelize = False
save = False
save_name = ""
num_folds = 1
fold_num = 0
scoring_method = 'f1_dist_msad:ff'
#num_inducing = int(seq_len*num_sequences_per_action_train*len(actions)*len(people)*0.5)
num_inducing = None
# ------ Simulations END
## ------ Optimization
opt1 = 'lbfgsb'
opt2 = 'lbfgsb'
#GPNode_opt = True
GPNode_opt = False
#opt1='scg'
#opt2='scg'
max_iters = 1
# ------ Optimization END
## ------ Dict construction
if data_set_name == 'Bimanual 3D':
    d = 54
    D = 117
    Y3_indices = np.arange(0, D * d, 1)
    Y2_indices = np.arange(D, D, 1)
    Y1_indices = np.arange(0, D, 1)
elif data_set_name == 'Movements CMU':
    d = 20
    D = 77
    Y3_indices = np.arange(0, D - 2 * d, 1)
    Y2_indices = np.arange(D - d, D, 1)
    Y1_indices = np.arange(0, D - d, 1)
Y_indices = [Y1_indices,Y2_indices]
data_set_class_dict = {'actions': actions, 'people': people, 'seq_len': seq_len,'num_folds':num_folds,'fold_num':fold_num,
                        'num_sequences_per_action_test': num_sequences_per_action_test,'X_init':None}
dynamics_dict = {'name':prior_dynamics,'Y_indices':Y_indices}


BC_dict = {'type': bc_name, 'geometry': geometry, 'hidden_dim':hidden_dim,'num_epochs':num_epochs,
           'Y1 indices':Y1_indices,'Y2 indices':Y2_indices,'Y3 indices':Y3_indices,'geo params':geo_params,
           'num_acts':len(actions), 'constraints': BC_param_constraints, 'ARD': ARD, 'mapping':mapping}
prior_dict = {'name':prior_name,'dynamics_dict':dynamics_dict,'order':order}

top_node_dict = {'attr_dict':{
                 'input_dim':input_dim1,
                 'init':init1,
                 'opt':opt1,
                 'max_iters':max_iters,
                 'GPNode_opt': GPNode_opt,
                 'kernel':None,
                 'num_inducing':num_inducing},
                 'BC_dict':BC_dict,
                 'prior_dict':prior_dict}

arch_dict = {'attr_dict':{'name':arch_type,'Y_indices':Y_indices,'pred_group':pred_group,
                          'init_t':init_t,'seq_len':pred_seq_len, 'scoring_method':scoring_method},
             'top_node_dict':top_node_dict
             }

hgp_dict = {'attr_dict':{'pred_group':pred_group, 'seq_len': seq_len,'pred_seq_len_ratio':pred_seq_len_ratio, 'max_iters':max_iters},
            'arch_dict':arch_dict}
# --- Data
const_data_args_dict = {'data_set_name': data_set_name, 'indices': Y_indices, 'seq_len': seq_len, 'actions': actions, 'people': people,
                        'num_sequences_per_action_train': num_sequences_per_action_train,
                        'num_sequences_per_action_test': num_sequences_per_action_test,
                        'data_set_class_dict': data_set_class_dict}
var_data_args_dict = {}
data_arg_dict = {'constant_args_dict' : const_data_args_dict, 'variable_args_dict':var_data_args_dict}
# --- Comp
const_comp_args_dict = {"opts":[opt1,opt2],"GPNode_opt":GPNode_opt}
var_comp_args_dict = {"hgp_dict":[hgp_dict]}#['None','None','None'],
comp_arg_dict = {'constant_args_dict' : const_comp_args_dict, 'variable_args_dict':var_comp_args_dict}
# ------ Dict construction END
## ------ Optimize
FS = func_simulator(data_func,comp_func,parallelize=parallelize)
results = FS.run_sims(data_arg_dict, comp_arg_dict, seed)
print(BC_param_constraints)
# ------ Optimize END
'''## ------ Save
if save:
    filename = 'saved_models/'+save_name+'.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    # Deserialize with dill
    with open(filename, 'rb') as f:
        pickle.results = load(f)
# ------ Save END'''
## ------ Test and plot
hgp = results[0]
data_mgmt = hgp.arch.data_set_class

if prior_dict['name'] is not None:
    hgp.score()
if num_sequences_per_action_test < 3:
    if prior_dict['name'] is None:
        X_preds = hgp.infer_X([np.vstack(data_mgmt.Y_test_list)],pred_group = pred_group)

        ax1, fig1 = data_mgmt.plot_gs(data_mgmt.sub_num_actions, data_mgmt.num_test_seqs_per_subj_per_act,
                                      data_mgmt.num_train_seqs_per_subj_per_act, X_latent=hgp.arch.model.top_node.X,
                                      X_pred=X_preds,
                                      x_dim=0, y_dim=1, z_dim=2)
    else:

        '''(Y_preds_train, X_preds_train, X_inferences_train, error_train, dtw_obj_train, traj_preds_train,
         IFs_train) = hgp.predict(data_mgmt.Y_train_list, 0, seq_len=pred_seq_len,pred_group=pred_group)'''

        Y_preds, X_preds, X_inferences, metrics_dict, traj_preds, IFs = hgp.predict(data_mgmt.Y_test_list, test = True, Y_uk_ID = Y_uk_ID, init_t = 0,
                                                                                      seq_len=pred_seq_len, pred_group=pred_group)

        if input_dim1 >= 3:
            '''ax1, fig1 = data_mgmt.plot_gs(data_mgmt.sub_num_actions, data_mgmt.num_test_seqs_per_subj_per_act,
                                                  data_mgmt.num_train_seqs_per_subj_per_act,X_test=X_inferences_train, X_latent=None, X_pred=X_preds_train, x_dim=0,
                                                  y_dim=1, z_dim=2)'''

            ax1, fig1 = data_mgmt.plot_gs(data_mgmt.sub_num_actions, data_mgmt.num_test_seqs_per_subj_per_act,
                              data_mgmt.num_train_seqs_per_subj_per_act, X_latent=hgp.arch.model.top_node.X, X_pred=X_preds,
                                          x_dim=0, y_dim=1, z_dim=2)

            '''ax1b, fig1b = data_mgmt.plot_gs(data_mgmt.sub_num_actions, data_mgmt.num_test_seqs_per_subj_per_act,
                                          data_mgmt.num_train_seqs_per_subj_per_act, X_test=X_inferences,
                                          X_latent=hgp.arch.model.top_node.X, X_pred=None, x_dim=0,
                                          y_dim=1, z_dim=2)'''

            '''ax1c, fig1c = data_mgmt.plot_gs(data_mgmt.sub_num_actions, data_mgmt.num_test_seqs_per_subj_per_act,
                                            data_mgmt.num_train_seqs_per_subj_per_act, X_test=X_inferences, X_pred=X_preds, x_dim=0,
                                            y_dim=1, z_dim=2)'''
        if input_dim1 >= 6:
            ax2, fig2 = data_mgmt.plot_gs(data_mgmt.sub_num_actions, data_mgmt.num_test_seqs_per_subj_per_act,
                                          data_mgmt.num_train_seqs_per_subj_per_act, X_latent=hgp.arch.model.top_node.X,
                                          X_pred=X_preds,
                                          x_dim=3, y_dim=4, z_dim=5)

        if input_dim1 >= 9:
            ax3, fig3 = data_mgmt.plot_gs(data_mgmt.sub_num_actions, data_mgmt.num_test_seqs_per_subj_per_act,
                                          data_mgmt.num_train_seqs_per_subj_per_act, X_latent=hgp.arch.model.top_node.X,
                                          X_pred=X_preds,
                                          x_dim=6, y_dim=7, z_dim=8)

        if prior_dict['name'] is not None:
            traj_preds_mat = []
            for traj in traj_preds:
                traj_preds_mat.append(traj)
            traj_preds_mat = np.around(np.vstack(traj_preds_mat), decimals=3)
            print('Trajectory Confusion Matrix')
            print(traj_preds_mat)

            '''for key in metrics_dict.keys():
                if key in ['frechet_distance', 'dtws']:
                    print(key)
                    for key2 in metrics_dict[key].keys():
                        print(key2)
                        for seq in metrics_dict[key][key2]:
                            print(seq)'''


            anis = []
            for i,IFs_i in enumerate(IFs):
                fig, animate = IFs_i.plot_animation_all_figures()
                anis.append(animation.FuncAnimation(fig, animate, seq_len, interval=100))
                '''if save:
                    save_path = save_name + "_" + actions[n_] + "_" + str(i) + ".gif"
                    if not os.path.isfile(file_path):
                        writervideo = animation.PillowWriter(fps=35)
                        anis[j].save(save_path, writer=writervideo)'''