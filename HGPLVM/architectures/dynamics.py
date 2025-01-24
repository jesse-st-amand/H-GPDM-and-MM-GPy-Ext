import numpy as np
from dtw import *
from HGPLVM.architectures.sequence_metrics import euclidean_distance
from HGPLVM.architectures.sequence_metrics import cosine_similarity_metric
from HGPLVM.architectures.sequence_metrics import pearson_correlation
from HGPLVM.architectures.sequence_metrics import procrustes_analysis
from HGPLVM.architectures.sequence_metrics import frechet_distance
import copy


class DynamicsBase:

    def __init__(self, arch, attr_dict):
        self.arch = arch
        self.attr_dict = attr_dict
        self.HMMs = None
        self.name = 'dynamic'

    def predict_and_test(self, Y_test_list, init_t=0, seq_len=10,**kwargs):
        Y_pred_denorm_list, X_preds, X_inferences, pred_trajs, pred_traj_lists = self.arch.data_set_class.get_results(
                                    ['Y_pred_denorm_list', 'X_preds', 'X_inferences', 'pred_trajs', 'pred_traj_lists'],
                                    self.predict_Ys_Xs, Y_test_list, init_t=init_t, seq_len=seq_len, **kwargs)


        IFs = self.IF_setup()
        score_value = self.score(Y_test_list, init_t=init_t, seq_len=seq_len, **kwargs)

        if not X_inferences:
            pass
        else:
            X_inferences = np.vstack(X_inferences)

        return np.vstack(Y_pred_denorm_list), np.vstack(X_preds), X_inferences, score_value, pred_traj_lists, IFs


    '''def score(self, Y_test_list, init_t=0, seq_len=10, **kwargs):
        # Clear any existing results
        self.arch.data_set_class.results_dict = {}

        results = self.arch.data_set_class.get_results(['Y_pred_denorm_list', 'X_preds', 'X_inferences',
                                                        'pred_trajs', 'pred_traj_lists', 'Y_train_denorm_list',
                                                        'Y_test_denorm_list'],
                                                       self.predict_Ys_Xs, Y_test_list, init_t=init_t, 
                                                       seq_len=self.arch.attr_dict['sub_seq_len'],
                                                       **kwargs)

        # Update state variables
        self.Y_test_CCs = self.arch.data_set_class.Y_test_CCs
        self.Y_pred_CCs = self.arch.data_set_class.Y_pos_list_to_stick_dicts_CCs(results[0])

        true_sequences = self.arch.data_set_class.CC_dict_list_to_CC_array_list_min_PV(self.Y_test_CCs)
        pred_sequences = self.arch.data_set_class.CC_dict_list_to_CC_array_list_min_PV(self.Y_pred_CCs)

        scores = self.arch.data_set_class.score(self.arch.attr_dict['sub_seq_len'],
                                              self.arch.attr_dict['scoring_method'], 
                                              true_sequences, pred_sequences,
                                              self.arch.data_set_class.action_IDs_test,
                                              self.arch.data_set_class.results_dict['pred_trajs'])

        self.arch.pred_classes.append({'pred':self.arch.data_set_class.results_dict['pred_trajs'],
                                     'gt':self.arch.data_set_class.action_IDs_test})
        self.arch.f1_list.append(scores['f1'])
        self.arch.score_list.append(scores['avg_norm_distance'])
        self.arch.smoothness_list.append(scores['avg_norm_smoothness'])
        self.arch.freeze_list.append(scores['avg_freeze'])
        self.arch.iter_list.append(self.arch.model.learning_n)
        self.arch.loss_list.append(self.arch.model.ObjFunVal)
        
        print('')
        print('SCORES: ')
        print('avg_norm_distance: ')
        print(scores['avg_norm_distance'])
        print('f1: ')
        print(scores['f1'])
        print('avg_norm_smoothness: ')
        print(scores['avg_norm_smoothness'])
        print('avg_freeze: ')
        print(scores['avg_freeze'])

        if scores['f1'] == 0:
            f1_score = 0.01
        else:
            f1_score = scores['f1']
        return scores['avg_norm_distance']'''

    def CCs_setup(self,Y_prediction_list):
        self.Y_test_CCs = self.arch.data_set_class.Y_test_CCs
        self.Y_train_CCs = self.arch.data_set_class.Y_train_CCs
        self.Y_pred_CCs = self.arch.data_set_class.Y_pos_list_to_stick_dicts_CCs(Y_prediction_list)

    def IF_setup(self):
        IFs = []
        for i,(Y_p_SD, Y_k_SD) in enumerate(zip(self.Y_pred_CCs, self.Y_test_CCs)):
            IFs.append(self.arch.data_set_class.IFs_func([Y_k_SD,Y_p_SD]))
        return IFs


    def mn_predict(self,Xstar,traj):
        return self.arch.model.top_node.prior.mn_predict(Xstar,traj)

    def velocity(self, Y):
        return Y[1:,:] - Y[:-1]




class FeedforwardDynamics(DynamicsBase):

    def __init__(self, arch, attr_dict):
        super().__init__(arch, attr_dict)
        self.name = 'dynamic:ff'

    def predict_Ys_Xs(self, Y_test_list, init_t, seq_len=10,**kwargs):
        Y_preds = []
        X_preds = []
        X_inferences = []
        pred_trajs = []
        pred_traj_lists = []

        for Y in Y_test_list:
            #X_inferences.append(self.arch.infer_X([Y], pred_group=0))
            Y_pred, X_pred, pred_traj, pred_traj_list = self.predict_Y_from_Y0(Y[init_t:init_t + seq_len, :], **kwargs)
            Y_preds.append(Y_pred)
            #Y_preds.append(Y)
            X_preds.append(X_pred)
            pred_trajs.append(pred_traj)
            pred_traj_lists.append(pred_traj_list)

        Y_train_denorm_list = self.arch.denorm_trajs(self.arch.data_set_class.Y_train_list, self.arch.data_set_class.action_IDs_train)
        Y_test_denorm_list = self.arch.denorm_trajs(self.arch.data_set_class.Y_test_list,
                                                     self.arch.data_set_class.action_IDs_test)
        Y_pred_denorm_list = self.arch.denorm_trajs(Y_preds, pred_trajs)
        self.CCs_setup(Y_pred_denorm_list)
        return Y_pred_denorm_list, X_preds, X_inferences, pred_trajs, pred_traj_lists, Y_train_denorm_list, Y_test_denorm_list

    def predict_Ys(self, Y0_list, **kwargs):
        Y_preds = []
        for Y0 in Y0_list:
            Y_pred, X_pred, pred_traj, pred_traj_list = self.predict_Y_from_Y0(Y0, **kwargs)
            Y_preds.append(Y_pred)
        return Y_preds

    def predict_Y_from_Y0(self, Y0, **kwargs):
        X0 = self.arch.infer_X([Y0], **kwargs)
        Y_pred, X_pred, pred_traj, pred_traj_list = self.predict_trajectory(X0)
        return Y_pred, X_pred, pred_traj, pred_traj_list

    def predict_trajectory(self, X0):
        pred_traj, pred_traj_list = self.arch.model.top_node.prior.predict_best_z(X0)
        X_pred = self.pred_X_from_X0(X0, pred_traj)
        Y_pred = np.hstack(self.arch.model.get_bottom_Ys(X_pred))
        return Y_pred, X_pred, pred_traj, pred_traj_list

    def pred_X_from_X0(self, X_0, traj):
        ts = self.arch.model.top_node.prior.gpdm_timesteps
        tps = self.arch.HGP.num_tps
        X_0_len = X_0.shape[0]
        X_pred = np.zeros([tps, self.arch.model.top_node.D])
        X_pred[:X_0_len, :] = X_0
        pred_tps = np.arange(X_0_len,tps,1)-ts
        for i in pred_tps:
            X_pred[i + ts, :] = self.mn_predict(X_pred[i:i + ts, :].reshape([ts, -1]), traj)
        return X_pred



    def set_Y(self, data_set_class):
        self.Y = data_set_class.Y_train
        self.N, self.D = self.Y.shape




class FeedbackDynamics(DynamicsBase):

    def __init__(self, arch, attr_dict):
        super().__init__(arch, attr_dict)
        self.name = 'dynamic:fb'
        self.Y_indices_list = attr_dict['Y_indices']
        self.Y_k_ID = None
        self.Y_uk_ID = None

    def pred_Y_uk_given_Y_k_0(self, Y_k_0, Y_k, init_t, seq_len, **kwargs):
        N_0, D_k = Y_k_0.shape


        ts = self.arch.model.top_node.prior.gpdm_timesteps
        tps = Y_k.shape[0]
        pred_tps = np.arange(init_t + seq_len, tps, 1) - ts

        X_0 = self.arch.infer_X([Y_k_0], **kwargs)
        X = np.zeros([tps, self.arch.model.top_node.D])
        X[:N_0, :] = X_0

        pred_traj, pred_traj_list = self.arch.model.top_node.prior.predict_best_z(X_0)


        Y_uk_0 = self.get_Y_uk(np.hstack(self.arch.model.get_bottom_Ys(X_0)))
        D_uk = Y_uk_0.shape[1]
        Y_uk = np.zeros([tps, D_uk])
        Y_uk[:N_0,:] = Y_uk_0

        for i in pred_tps:
            Y_k_t = Y_k[i - ts + 1:i + 1, :]
            Y_uk_t1, X_t1 = self.pred_Y_uk_t1_given_Y_k_t(Y_k_t, 0, **kwargs)
            X[i + 1, :] = X_t1
            Y_uk[i + 1, :] = Y_uk_t1

        return Y_uk, X, pred_traj, pred_traj_list

    def pred_Y_uk_t1_given_Y_k_t(self, Y_k_t, traj, **kwargs):
        X_t = self.arch.infer_X([Y_k_t], **kwargs)
        X_t1 = self.mn_predict(X_t, traj)
        Y_uk_t1 = self.get_Y_uk(np.hstack(self.arch.model.get_bottom_Ys(X_t1)))
        return Y_uk_t1, X_t1

    def predict_Ys(self, Y0_list, **kwargs):
        raise NotImplementedError

    def predict_Ys_Xs(self, Y_test_list, init_t, pred_group, seq_len=10, **kwargs):
        self.set_k_and_uk(pred_group)
        Y_preds = []
        X_preds = []
        X_inferences = []
        pred_trajs = []
        pred_traj_lists = []

        for Y in Y_test_list:
            Y_k = self.get_Y_k(Y)
            Y_k_0 = Y_k[init_t:init_t+seq_len,:]
            X_inferences.append(self.arch.infer_X([Y], pred_group=0))#**kwargs))
            Y_uk, X_pred, pred_traj, pred_traj_list = self.pred_Y_uk_given_Y_k_0(Y_k_0, Y_k, init_t, seq_len, pred_group=pred_group, **kwargs)
            Y_k_uk = self.combine_matrices(Y_k,Y_uk,Y)
            Y_preds.append(Y_k_uk)
            #Y_preds.append(Y)
            X_preds.append(X_pred)
            pred_trajs.append(pred_traj)
            pred_traj_lists.append(pred_traj_list)

        Y_train_denorm_list = self.arch.denorm_trajs(self.arch.data_set_class.Y_train_list,
                                                     self.arch.data_set_class.action_IDs_train)
        Y_test_denorm_list = self.arch.denorm_trajs(self.arch.data_set_class.Y_test_list,
                                                    self.arch.data_set_class.action_IDs_test)
        Y_pred_denorm_list = self.arch.denorm_trajs(Y_preds, self.arch.data_set_class.action_IDs_test)
        self.CCs_setup(Y_pred_denorm_list)
        return Y_pred_denorm_list, X_preds, X_inferences, pred_trajs, pred_traj_lists, Y_train_denorm_list, Y_test_denorm_list


    def set_k_and_uk(self, pred_group):
        if pred_group == 1:
            self.Y_uk_ID = int(1)
        elif pred_group == 2:
            self.Y_uk_ID = int(0)
        else:
            raise ValueError("Pred group must be 1 or 2.")

        # only written for binary condition
        if self.Y_uk_ID == 1:
            self.Y_k_ID = 0
        elif self.Y_uk_ID == 0:
            self.Y_k_ID = 1
        else:
            raise ValueError("Binary condition not met.")

    def get_Y_uk(self, Y):
        return self.get_Y_sub(Y, self.Y_indices_list[self.Y_uk_ID])

    def get_Y_k(self, Y):
        return self.get_Y_sub(Y, self.Y_indices_list[self.Y_k_ID])

    def get_Y_sub(self, Y, Y_sub_indices):
        Y_sub = Y[:,Y_sub_indices]
        if Y_sub.ndim == 1:
            # Add a new axis
            Y_sub = Y_sub[:, np.newaxis]
        return Y_sub

    def combine_matrices(self, Y1, Y2, Y):
        # Initialize the combined matrix with zeros
        combined_matrix = np.zeros(Y.shape)

        # Place matrix1 at the specified indices
        combined_matrix[:, self.Y_indices_list[self.Y_k_ID]] = Y1

        # Place matrix2 at the specified indices
        combined_matrix[:, self.Y_indices_list[self.Y_uk_ID]] = Y2

        return combined_matrix


