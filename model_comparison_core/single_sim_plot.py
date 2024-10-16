
import os
import psutil
import pickle
from joblib import Parallel, delayed
import numpy as np
import traceback
import shutil  # Add this import for directory operations
from model_comparison_core.comp_data_func import comp_func, data_func
from model_comparison_core.algorithm_sims import func_simulator




if __name__ == "__main__":
    param_dir = 'testing'
    names = [
        'GPDM_params_Bayesian_BM_50_singular',

    ]

    for name in names:
        local_vars = {}
        string = f'from model_comparison_core.model_parameters.{param_dir+'.'+name} import output_dir_name, num_sims, combined_dicts, comp_tuples, data_tuples, space'
        exec(string, {}, local_vars)
        combined_dict = local_vars["combined_dicts"]
        data_dict = combined_dict[0]["dicts"][0][1]
        comp_dict = combined_dict[0]["dicts"][0][0]

        FS = func_simulator(data_func, comp_func, space=None, parallelize=False)

        results = FS.run_sims(data_dict, comp_dict, combined_dict[0]["seed"],
                              dict_index=data_dict["variable_args_dict"]['data_set_class_dict'][0]["fold_num"],
                              dir_path="", return_model = True)
        model = results[0][1]
        """if len(local_vars['combined_dicts'][0]['dicts']) == 1:
            if local_vars['combined_dicts'][0]['dicts'][0][1]['constant_args_dict'][
                'num_sequences_per_action_test'] < 3:

                traj_preds_mat = []
                
                for traj in model.arch.data_set_class.results_dict['pred_traj_lists']:
                    traj_preds_mat.append(traj)
                traj_preds_mat = np.around(np.vstack(traj_preds_mat), decimals=3)
                print('Trajectory Confusion Matrix')
                print(traj_preds_mat)
                import matplotlib
                matplotlib.use("TkAgg")
                import matplotlib.animation as animation
                IFs = model.IF_setup()
                anis = []
                for i, IFs_i in enumerate(IFs[:5]):
                    fig, animate = IFs_i.plot_animation_all_figures()
                    anis.append(animation.FuncAnimation(fig, animate, model.num_tps, interval=100))
                matplotlib.pyplot.show()"""

ax1, fig1 = model.arch.data_set_class.plot_gs(model.arch.data_set_class.sub_num_actions, model.arch.data_set_class.num_test_seqs_per_subj_per_act,
                              model.arch.data_set_class.num_train_seqs_per_subj_per_act, X_latent=model.arch.model.top_node.X,
                                          x_dim=0, y_dim=1, z_dim=2)

ax1, fig1 = model.arch.data_set_class.plot_gs(model.arch.data_set_class.sub_num_actions, model.arch.data_set_class.num_test_seqs_per_subj_per_act,
                              model.arch.data_set_class.num_train_seqs_per_subj_per_act, X_latent=model.arch.model.top_node.X,
                                          x_dim=3, y_dim=4, z_dim=5)