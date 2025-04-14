import os
import psutil
import pickle
from joblib import Parallel, delayed
import numpy as np
import traceback
import shutil  # Add this import for directory operations
from model_comparison_core.comp_data_func import comp_func, data_func
from model_comparison_core.algorithm_sims import func_simulator

def plot_model(model,indices):
    traj_preds_mat = []
    for traj in model.arch.data_set_class.results_dict['pred_traj_lists_test']:
        traj_preds_mat.append(traj)
    traj_preds_mat = np.around(np.vstack(traj_preds_mat), decimals=3)
    print('Trajectory Confusion Matrix')
    print(traj_preds_mat)
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.animation as animation
    IFs = model.IF_setup()
    anis = []
    for i, IFs_i in enumerate(IFs):
        if i in indices:
            fig, animate = IFs_i.plot_animation_all_figures()
            anis.append(animation.FuncAnimation(fig, animate, model.num_tps, interval=100))
    matplotlib.pyplot.show()

def plot_dynamic_latent_space(model, data_list=None, dims=[0, 1, 2], num_trajectories=None):
    """
    Plot the dynamic latent space trajectories
    
    Parameters:
    -----------
    model : model object
        The trained model
    data_list : list, optional
        List of data sequences to encode. If None, uses training data.
    dims : list, optional
        Which dimensions to plot (default: first 3)
    num_trajectories : int, optional
        Maximum number of trajectories to plot (None = all)
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Get dynamic latent trajectories
    trajectories, labels = model.arch.get_dynamic_latent_space(data_list)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot subset of trajectories if specified
    if num_trajectories is not None:
        indices = np.random.choice(len(trajectories), size=min(num_trajectories, len(trajectories)), replace=False)
        trajectories = [trajectories[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    # Plot each trajectory
    for traj, label in zip(trajectories, labels):
        # Select dimensions to visualize
        x = traj[:, dims[0]]
        y = traj[:, dims[1]]
        z = traj[:, dims[2]]
        
        # Color based on class
        color = plt.cm.tab10(label)
        
        # Plot trajectory
        ax.plot(x, y, z, '-', color=color, alpha=0.7, linewidth=1)
        
        # Mark start point
        ax.plot([x[0]], [y[0]], [z[0]], 'o', color=color, markersize=5)
    
    # Set labels
    ax.set_xlabel(f'Dimension {dims[0]}')
    ax.set_ylabel(f'Dimension {dims[1]}')
    ax.set_zlabel(f'Dimension {dims[2]}')
    
    plt.title('Dynamic Latent Space Trajectories')
    plt.tight_layout()
    return fig, ax

def format_dynamic_latent_space(trajectories, labels, num_tps):
    """
    Format dynamic latent trajectories into a single matrix compatible with plot_gs
    
    Parameters:
    -----------
    trajectories : list of numpy.ndarray
        List of trajectory arrays, each with shape (time_points, latent_dims)
    labels : list
        List of action labels for each trajectory
    num_tps : int
        Number of time points per sequence
    
    Returns:
    --------
    X_latent : numpy.ndarray
        Formatted latent space matrix
    """
    # Count unique action labels
    unique_labels = np.unique(labels)
    num_actions = len(unique_labels)
    
    # Count trajectories per action
    traj_per_action = {}
    for label in unique_labels:
        traj_per_action[label] = sum(1 for l in labels if l == label)
    
    # Find max trajectories per action to determine matrix size
    max_traj_per_action = max(traj_per_action.values())
    
    # Get latent dimension size
    latent_dims = trajectories[0].shape[1]
    
    # Create a combined latent space matrix
    # The structure is: [action1_seq1, action1_seq2, ..., action2_seq1, ...]
    all_trajectories = []
    for action in unique_labels:
        # Get all trajectories for this action
        action_trajectories = [traj for traj, label in zip(trajectories, labels) if label == action]
        
        # Ensure all trajectories have the same length (num_tps)
        for i, traj in enumerate(action_trajectories):
            if traj.shape[0] < num_tps:
                # Pad with zeros if needed
                padded_traj = np.zeros((num_tps, traj.shape[1]))
                padded_traj[:traj.shape[0], :] = traj
                action_trajectories[i] = padded_traj
            elif traj.shape[0] > num_tps:
                # Truncate if too long
                action_trajectories[i] = traj[:num_tps, :]
        
        # Add to the list
        all_trajectories.extend(action_trajectories)
    
    # Stack all trajectories into a single matrix
    X_latent = np.vstack(all_trajectories)
    
    return X_latent, max_traj_per_action

def create_directory(path):
    if path.split('/')[-1].startswith('temp_'):
        if os.path.exists(path):
            print(f"Temporary directory already exists. Removing: {path}")
            shutil.rmtree(path)
        os.makedirs(path)
        print(f"Temporary directory created: {path}")
    elif not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    param_dir = 'testing'
    plot_animation = False
    plot_latent = False
    plot_dynamics = True  # New flag for plotting dynamic latent space
    names = [
        #'VAE_MCCV_BM',
        #'VAE_MCCV_CMU',
        'RNN_MCCV_BM',
        #'RNN_MCCV_CMU',
        #'transformer_MCCV_BM',
        #'transformer_MCCV_CMU',
        

    ]

    for name in names:
        local_vars = {}
        string = f'from model_comparison_core.model_parameters.{param_dir+'.'+name} import  num_sims, combined_dicts, comp_tuples, data_tuples, space'
        exec(string, {}, local_vars)
        combined_dict = local_vars["combined_dicts"]
        data_dict = combined_dict[0]["dicts"][0][1]
        comp_dict = combined_dict[0]["dicts"][0][0]
        dir_path = parent_directory+"/output_repository/model_summaries/"+param_dir+"/"+name+"/"
        create_directory(dir_path)
        FS = func_simulator(data_func, comp_func, space=None, parallelize=False)

        results = FS.run_sims(data_dict, comp_dict, combined_dict[0]["seed"],
                              dict_index=data_dict["variable_args_dict"]['data_set_class_dict'][0]["fold_num"],
                              dir_path=dir_path, return_model = True)
        model = results[0][1]
        
        if plot_animation == True:
            plot_model(model,[0,5,10,15,20])


        if plot_latent == True:
            # Check if the model is a GPLVM-based model with top_node.X
            if hasattr(model.arch, 'model') and hasattr(model.arch.model, 'top_node'):
                # GPLVM-based model
                ax1, fig1 = model.arch.data_set_class.plot_gs(model.arch.data_set_class.sub_num_actions, 
                                model.arch.data_set_class.num_test_seqs_per_subj_per_act,
                                model.arch.data_set_class.num_train_seqs_per_subj_per_act, 
                                X_latent=model.arch.model.top_node.X,
                                x_dim=0, y_dim=1, z_dim=2)
                
                ax2, fig2 = model.arch.data_set_class.plot_gs(model.arch.data_set_class.sub_num_actions, 
                                model.arch.data_set_class.num_test_seqs_per_subj_per_act,
                                model.arch.data_set_class.num_train_seqs_per_subj_per_act, 
                                X_latent=model.arch.model.top_node.X,
                                x_dim=8, y_dim=9, z_dim=10)
            elif hasattr(model.arch, 'get_latent_space'):
                # Neural network-based model (VAE, LSTM, Transformer)
                latent_space = model.arch.get_latent_space()
                print(f"Extracted latent space with shape: {latent_space.shape}")
                
                # Plot first 3 dimensions
                ax1, fig1 = model.arch.data_set_class.plot_gs(model.arch.data_set_class.sub_num_actions, 
                                model.arch.data_set_class.num_test_seqs_per_subj_per_act,
                                model.arch.data_set_class.num_train_seqs_per_subj_per_act, 
                                X_latent=latent_space,
                                x_dim=0, y_dim=1, z_dim=2)
                
                # If latent space has enough dimensions, plot another view
                if latent_space.shape[1] > 5:
                    ax2, fig2 = model.arch.data_set_class.plot_gs(model.arch.data_set_class.sub_num_actions, 
                                    model.arch.data_set_class.num_test_seqs_per_subj_per_act,
                                    model.arch.data_set_class.num_train_seqs_per_subj_per_act, 
                                    X_latent=latent_space,
                                    x_dim=3, y_dim=4, z_dim=5)
            else:
                print("Cannot plot latent space: Model architecture doesn't have a recognized latent space structure")
        
        # Plot dynamic latent space if the model has the get_dynamic_latent_space method
        if plot_dynamics == True and hasattr(model.arch, 'get_dynamic_latent_space'):
            print("Plotting dynamic latent space trajectories...")
            
            # Get dynamic latent trajectories
            trajectories, labels = model.arch.get_dynamic_latent_space()
            
            # Format for plot_gs
            print(f"Got {len(trajectories)} trajectories with shape {trajectories[0].shape}")
            X_dynamic_latent, num_latent_seqs = format_dynamic_latent_space(trajectories, labels, model.arch.data_set_class.num_tps)
            print(f"Formatted latent space with shape {X_dynamic_latent.shape}, {num_latent_seqs} sequences per action")
            
            # Plot first set of dimensions
            ax1, fig1 = model.arch.data_set_class.plot_gs(
                model.arch.data_set_class.sub_num_actions,
                model.arch.data_set_class.num_test_seqs_per_subj_per_act,
                num_latent_seqs=num_latent_seqs, 
                X_latent=X_dynamic_latent,
                x_dim=0, y_dim=1, z_dim=2
            )
            
            # Check if the latent space is large enough for a second view
            if X_dynamic_latent.shape[1] > 5:
                ax2, fig2 = model.arch.data_set_class.plot_gs(
                    model.arch.data_set_class.sub_num_actions,
                    model.arch.data_set_class.num_test_seqs_per_subj_per_act,
                    num_latent_seqs=num_latent_seqs, 
                    X_latent=X_dynamic_latent,
                    x_dim=3, y_dim=4, z_dim=5
                )
            
            # Save the figures
            fig1.savefig(os.path.join(dir_path, 'dynamic_latent_space_dims012.png'))
            if X_dynamic_latent.shape[1] > 5:
                fig2.savefig(os.path.join(dir_path, 'dynamic_latent_space_dims345.png'))
            
            # Show the figures
            import matplotlib.pyplot as plt
            plt.show()