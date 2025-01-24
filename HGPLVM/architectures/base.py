import numpy as np
from HGPLVM.GPLVM_node import GPLVM_node as GPNode

class Architecture_Base():
    def __init__(self, arch_dict = {}, **kwargs):
        if 'attr_dict' in arch_dict:
            self.attr_dict = arch_dict['attr_dict']
        else:
            self.attr_dict = {}

    def get_results(self, **kwargs):
        return self.data_set_class.get_results(['Y_pred_denorm_list', 'X_preds', 'X_inferences',
                                                        'pred_trajs', 'pred_traj_lists', 'Y_train_denorm_list',
                                                        'Y_test_denorm_list'],
                                                       self.dynamics.predict_Ys_Xs, self.data_set_class.Y_test_list, init_t=self.attr_dict['init_t'], 
                                                       seq_len=self.attr_dict['sub_seq_len'],pred_group=self.attr_dict['pred_group'],
                                                       **kwargs)
    
    def get_scores(self, true_sequences, pred_sequences, **kwargs):
        return self.data_set_class.score(self.attr_dict['sub_seq_len'],
                                              self.attr_dict['scoring_method'], 
                                              true_sequences, pred_sequences,
                                              self.data_set_class.action_IDs_test,
                                              self.data_set_class.results_dict['pred_trajs'])
    
    def get_class_preds(self):
        return {'pred':self.data_set_class.results_dict['pred_trajs'],
                                     'gt':self.data_set_class.action_IDs_test}
    
    def store_data(self, iteration, score_rate, log_likelihood):
        if score_rate == 0:
            pass
        elif iteration % score_rate == 1 or iteration == 1:
            self.score(iteration, log_likelihood)



    
class HGP_Architecture_Base(Architecture_Base):
    def set_kernels(self,kernel_list):
        self.kernels = kernel_list

    def set_backconstraints(self,BC_dict):
        self.BC_dict = BC_dict

    def set_priors(self,prior_dict):
        self.priors = prior_dict
        if self.priors is not None:
            for prior in self.priors:
                if self.priors[prior]['dynamics_dict'] is not None:
                    self.set_dynamics(self.priors[prior]['dynamics_dict']['name'],self.priors[prior]['dynamics_dict'])

    def set_dynamics(self, dynamics_dict):
        if dynamics_dict['name'] is None:
            return
        elif dynamics_dict['name'].lower() == 'ff':
            from HGPLVM.architectures.dynamics import FeedforwardDynamics as dyn
        elif dynamics_dict['name'].lower() == 'fb':
            from HGPLVM.architectures.dynamics import FeedbackDynamics as dyn
        else:
            raise NotImplementedError('Unknown dynamics')

        print('Setting dynamics: ' + dynamics_dict['name'])
        self.dynamics = dyn(self, attr_dict=dynamics_dict)

    def set_initializations(self,init_list,max_iters=100,GPNode_opt=False):
        '''

        :param init_list:
        :param max_iters:
        :param GPNode_opt: optimizes each node in the model individually as an initialization before running the HGP optimization
        :return:
        '''
        self.inits = init_list
        self.max_iters = max_iters
        self.GPNode_opt = GPNode_opt

    def set_optimizers(self,opt_list):
        self.opts = opt_list

    def set_input_dimensions(self,input_dims_list):
        self.in_dims = input_dims_list

    def set_num_inducing_points(self,num_inducing_list):
        self.num_inducing = num_inducing_list

    def set_dimensional_slices(self,dim_splits):
        self.dim_splits = dim_splits




class HGP_arch(HGP_Architecture_Base):
    def __init__(self,HGP, data_set_class, arch_dict={}, **kwargs):
        super(HGP_arch,self).__init__(arch_dict=arch_dict)
        self.HGP = HGP
        self.model = None
        self.dynamics = None
        self.set_up(data_set_class)


    def infer_X(self,Y_test_list,**kwargs):
        return self.model.get_top_X(Y_test_list,**kwargs)

    def GPNode_init_opt(self,GPNode,optimizer,max_iters,GPNode_opt):
        if GPNode_opt:
            GPNode.optimize(optimizer=optimizer, max_iters=max_iters)
            print('Node ' + str(GPNode.node_nID) + ' optimized')

    def data_dist_apprx(self, Y, Y_new, X):
        """
        Approximates the latent projection of Y_new via the nearest points in Y and associated X
        :param Y:
        :param Y_new:
        :param X:
        :return:
        """
        dist = -2. * Y_new.dot(Y.T) + np.square(Y_new).sum(axis=1)[:, None] + np.square(Y).sum(axis=1)[None, :]
        idx = dist.argmin(axis=1)
        return X[idx].copy()

    def denorm_trajs(self, Y_list, action_ID_list):
        """
        denormalize trajectories
        :param Y_list: List of trajectories
        :param action_ID_list: List integers corresponding to each action
        :return:
        """
        Y_denorm_list = []
        for Y, action_ID in zip(Y_list,action_ID_list):
            Y_denorm_list.append(self.data_set_class.denormalize(Y, action_ID))
        return Y_denorm_list

    def set_up(self, data_set_class):
        self.data_set_class = data_set_class
        self.set_Y(data_set_class)

    def predict(self, Y, test=False, **kwargs):
        if self.dynamics is None:
            raise ValueError("Dynamical modeling not set.")

        if not isinstance(Y, list):
            Y = [Y]

        if test:
            return self.dynamics.predict_and_test(Y, **kwargs)
        else:
            return self.dynamics.predict_Ys_Xs(Y, **kwargs)


    def set_Y(self):
        raise NotImplementedError

    def merge_dicts(self, d1, d2):
        """
        Merge two dictionaries recursively.
        For each matching key:
        - If both values are tuples, concatenate them.
        - Otherwise, replace the value from d1 with the value from d2.
        """
        for key, value in d2.items():
            if key in d1:
                if isinstance(value, dict) and isinstance(d1[key], dict):
                    self.merge_dicts(d1[key], value)
                elif isinstance(value, tuple) and isinstance(d1[key], tuple):
                    d1[key] = list(value)  # Concatenate tuples
                else:
                    d1[key] = value  # Replace the value from d1 with d2
            else:
                d1[key] = value  # Add the new key from d2 to d1
        return d1






































"""class X1_H2_Y2_feedback(X1_H2_Y2):
    def __init__(self, HGP, data_set_class, base_arch=None):
        self.N, self.D = self.Y.shape
        super().__init__(HGP, data_set_class, base_arch)

    def predict(self, Y_k_0, Y_k, Y_pred_idx=1, **kwargs):
        ts = self.model.top_node.prior.gpdm_timesteps
        tps = self.HGP.num_tps
        Y = np.zeros([tps,self.D])
        N_0, D_k = Y_k_0.shape

        X_0 = self.infer_X([Y_k_0], **kwargs)
        pred_traj, pred_traj_list = self.model.top_node.prior.predict_best_z(X_0)

        Y_uk_0 = self.model.get_bottom_Ys(X_0)[Y_pred_idx]
        Y[:N_0, :] = np.hstack(Y_k_0,Y_uk_0)

        X_0_len = X_0.shape[0]
        X = np.zeros([tps, self.model.top_node.D])
        X[:X_0_len, :] = X_0
        pred_tps = np.arange(X_0_len,tps,1)-ts

        for i in pred_tps:
            Y_k_t = Y_k[i-ts+1:i+1,:]
            Y_k_t1, Y_pred_t1, X_t1 = self.pred_t1(Y_k_t, **kwargs)
            X[i + 1, :] = X_t1
            Y_t1 = np.hstack([Y_k_t1, Y_pred_t1])
            Y[i + 1, :] = Y_t1

        return Y, X

    def pred_t1(self,Y_k_t, **kwargs):
        X_t = self.infer_X([Y_k_t], **kwargs)
        X_t1 = self.mn_predict(X_t, traj)
        Y_pred_t1 = self.model.get_bottom_Ys(X_pred[i + ts, :])[Y_pred_idx]
        return Y_k_t1, Y_pred_t1, X_t1

    def predict_list(self, Y_test_list, init_t, seq_len=10,**kwargs):
        Y_preds = []
        X_preds = []
        X_inferences = []
        pred_trajs = []
        pred_traj_lists = []

        self.set_up_Ys(Y_test_list)

        for Y_k in Y_test_list:
            Y_k_0 = self.set_up_Ys(Y_k)[0]
            X_inferences.append(self.infer_X([Y_k], **kwargs))
            Y_pred, X_pred = predict(Y_k_0, Y_k, **kwargs)
            Y_preds.append(Y_pred)
            X_preds.append(X_pred)
            pred_trajs.append(pred_traj)
            pred_traj_lists.append(pred_traj_list)

        return Y_preds, X_preds, X_inferences, pred_trajs, pred_traj_lists"""










































