import numpy as np
import copy
from itertools import combinations
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from glob import glob
from . import DSC_tools
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from .evaluation_metrics import score_f1_dist_smoothness
class DataSetClassBase:
    def __init__(self, name):
        self.name = name
        self.results_dict = {}

    def custom_combinations(self, input_list, n):
        N = len(input_list)
        if n > N:
            return "Invalid input: n must be less than or equal to N"

        result = []
        for combo in combinations(input_list, n):
            chosen = list(combo)
            remaining = [item for item in input_list if item not in chosen]
            result.append([chosen, remaining])

        return result

    @staticmethod
    def ensure_dimensions(arr, n):
        """
        Ensure that the array has n dimensions by adding singleton dimensions
        to the end of the array.
        """
        while arr.ndim < n:
            arr = arr[..., np.newaxis]
        return arr

    def get_data_set(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def store_results(self, func, labels=None, *args, **kwargs):
        """
        Executes the input function with the provided arguments and stores the result in results_dict.
        If the function returns a tuple, each element is labeled with provided labels or default labels.

        Parameters:
        func (function): The function to be executed.
        labels (list, optional): A list of labels for the function's return values. Default labels are used if not provided.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

        Returns:
        None
        """
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            if labels is None:
                labels = [f"result_{i}" for i in range(len(result))]
            elif len(labels) != len(result):
                raise ValueError("The number of labels must match the number of returned values.")

            for i, value in enumerate(result):
                self.results_dict[labels[i]] = value
        else:
            self.results_dict[func.__name__] = result

    def get_results(self, labels, func=None, *args, **kwargs):
        if self.check_results(labels):
            results_list = []
            for label in labels:
                results_list.append(self.results_dict[label])
            return results_list
        elif func is not None:
            self.store_results(func, labels, *args, **kwargs)
            return self.get_results(labels, func, *args, **kwargs)
        else:
            raise ValueError("Dictionary does not have the expected keys.")

    def check_results(self, labels):
        """
        Checks if all the provided labels exist in the results_dict.

        Parameters:
        labels (list or str): A list of labels or a single label to check in results_dict.

        Returns:
        bool: True if all labels are present, False otherwise.
        """
        if isinstance(labels, str):
            labels = [labels]
        return all(label in self.results_dict for label in labels)

class DataSetClassSequencesBase(DataSetClassBase):
    def __init__(self, name, seq_len):
        self.num_tps = seq_len
        super().__init__(name)

class DataSetClassMovementBase(DataSetClassSequencesBase):
    def __init__(self, name, seq_len):
        super().__init__(name, seq_len)
        self.mn_start_pos = None
        self.cartesian_D = None
        self.full_num_actions = None
        self.full_num_subjects = None
        self.X_init = None
    def initialize_dataset(self, fold_num, num_folds, full_num_seqs_per_subj_per_act, graphic, IFs_func, actions, subjects, score_EPs_ff, score_EPs_fb):
        self.fold_num = fold_num
        self.num_folds = num_folds
        self.full_num_seqs_per_subj_per_act = full_num_seqs_per_subj_per_act
        self.graphic = graphic
        self.IFs_func = IFs_func
        self.actions = actions
        self.subjects = subjects
        self.score_EPs_ff = score_EPs_ff
        self.score_EPs_fb = score_EPs_fb

    def get_data_set(
            self,
            attr_dict,
            seq_len,
            actions,
            people,
            num_sequences_per_action_train,
            num_sequences_per_action_test,
            name=None,
    ):
        """
        Loads and processes the dataset, integrating both Bimanual3D and MovementsCMU.

        Parameters:
        - attr_dict: dictionary of attributes
        - seq_len: sequence length
        - actions: list of action indices
        - people: list of people indices
        - num_sequences_per_action_train: number of sequences per action for training
        - num_sequences_per_action_test: number of sequences per action for testing
        - name: dataset name (e.g., 'Bimanual 3D' or 'Movements CMU')
        """

        # Ensure that the dataset name is set
        if name is None:
            name = self.name

        # Update actions and people indices
        self.actions_indices = np.array(actions, dtype=int)
        self.people = people

        # Load sequences
        Y_pos_list, _ = self.sequence_actions()
        self.Y_pos_list = Y_pos_list
        Y_pos_radians_cyclic = self.list_to_array_3D(Y_pos_list)
        self.Y_pos_radians_cyclic = Y_pos_radians_cyclic
        # Initialize starting positions (mn_start_pos)
        if name == "Bimanual 3D":
            self.mn_start_pos = np.tile(Y_pos_radians_cyclic[0, :, 0, 0, 0][:, np.newaxis], len(self.actions))
        elif name == "Movements CMU":
            self.mn_start_pos = Y_pos_radians_cyclic[0,:,0,:,0]
        else:
            raise ValueError("Dataset does not exist.")
        # Compute velocities and positions
        self.Y_vel = self.pos_list_to_vel_array_3D(Y_pos_list)
        self.Y_pos = self.vel_array_3D_to_continuous_pos_array_3D(self.Y_vel)

        # Handle indices if provided
        if 'indices' in attr_dict.keys():
            self.indices_list = attr_dict['indices']

        # Determine the number of validation sequences per action
        if 'X_init' in attr_dict.keys():
            if attr_dict['X_init'] is not None:
                num_sequences_per_action_validation = num_sequences_per_action_train - 1
                num_sequences_per_action_train = 1
            else:
                num_sequences_per_action_validation = (
                        self.full_num_seqs_per_subj_per_act
                        - num_sequences_per_action_train
                        - num_sequences_per_action_test
                )
        else:
            num_sequences_per_action_validation = (
                    self.full_num_seqs_per_subj_per_act
                    - num_sequences_per_action_train
                    - num_sequences_per_action_test
            )

        # Get training, testing, and validation datasets
        (
            Y_train,
            Y_test,
            Y_validation,
            Y_train_list,
            Y_test_list,
            Y_validation_list,
        ) = self.get_NxD_subsets_pos(
            seq_len,
            actions,
            people,
            num_sequences_per_action_train,
            num_sequences_per_action_test,
            num_sequences_per_action_validation,
        )

        # Convert positions to stick figure representations
        self.Y_train_CCs = self.Y_pos_list_to_stick_dicts_CCs(
            self.denorm_trajs(Y_train_list, self.action_IDs_train)
        )
        self.Y_test_CCs = self.Y_pos_list_to_stick_dicts_CCs(
            self.denorm_trajs(Y_test_list, self.action_IDs_test)
        )
        self.Y_validation_CCs = self.Y_pos_list_to_stick_dicts_CCs(
            self.denorm_trajs(Y_validation_list, self.action_IDs_validation)
        )
        # Handle datasets with or without EPs (End Points)
        if name == self.name:
            self.Y_train = Y_train
            self.Y_test = Y_test
            self.Y_validation = Y_validation
            self.Y_train_list = Y_train_list
            self.Y_test_list = Y_test_list
            self.Y_validation_list = Y_validation_list

        elif name == f"{self.name} with EPs":
            # Process training data
            Y_train_CC_list = []
            for Y_train_CC in self.Y_train_CCs:
                CC_short_list = [Y_train_CC[EP_key]['CC'] for EP_key in self.score_EPs_fb]
                Y_train_CC_list.append(np.hstack(CC_short_list))

            Y_train_CC_array = np.vstack(Y_train_CC_list)
            Y_train_CC_avg = np.mean(Y_train_CC_array, axis=0)
            Y_train_CC_std = np.std(Y_train_CC_array, axis=0)

            Y_train_list_EPs = []
            for Y_train_CC_temp, Y_train_JA in zip(Y_train_CC_list, Y_train_list):
                Y_train_CC_norm = (Y_train_CC_temp - Y_train_CC_avg) / Y_train_CC_std
                Y_train_list_EPs.append(np.hstack([Y_train_JA, Y_train_CC_norm]))

            # Process testing data
            Y_test_CC_list = []
            for Y_test_CC in self.Y_test_CCs:
                CC_short_list = [Y_test_CC[EP_key]['CC'] for EP_key in self.score_EPs_fb]
                Y_test_CC_list.append(np.hstack(CC_short_list))

            Y_test_list_EPs = []
            for Y_test_CC_temp, Y_test_JA in zip(Y_test_CC_list, Y_test_list):
                Y_test_CC_norm = (Y_test_CC_temp - Y_train_CC_avg) / Y_train_CC_std
                Y_test_list_EPs.append(np.hstack([Y_test_JA, Y_test_CC_norm]))

            Y_validation_CC_list = []
            for Y_validation_CC in self.Y_validation_CCs:
                CC_short_list = [Y_validation_CC[EP_key]['CC'] for EP_key in self.score_EPs_fb]
                Y_validation_CC_list.append(np.hstack(CC_short_list))

            Y_validation_list_EPs = []
            for Y_validation_CC_temp, Y_validation_JA in zip(Y_validation_CC_list, Y_validation_list):
                Y_validation_CC_norm = (Y_validation_CC_temp - Y_train_CC_avg) / Y_train_CC_std
                Y_validation_list_EPs.append(np.hstack([Y_validation_JA, Y_validation_CC_norm]))

            # Update averages and standard deviations
            Y_train_CC_avg = Y_train_CC_avg[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
            Y_train_CC_std = Y_train_CC_std[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
            self.avgs = np.append(self.avgs, Y_train_CC_avg, axis=1)
            self.stds = np.append(self.stds, Y_train_CC_std, axis=1)

            # Update datasets
            self.Y_train = np.vstack(Y_train_list_EPs)
            self.Y_test = np.vstack(Y_test_list_EPs)
            self.Y_validation = np.vstack(Y_validation_list_EPs)
            self.Y_train_list = Y_train_list_EPs
            self.Y_test_list = Y_test_list_EPs
            self.Y_validation_list = Y_validation_list_EPs

        else:
            raise ValueError(f"Invalid dataset name: {name}")

        # Handle initialization of X (if required)
        if 'X_init' in attr_dict.keys():
            if attr_dict['X_init'] == 'medoid':
                from sklearn_extra.cluster import KMedoids
                from sklearn.decomposition import KernelPCA

                transformer = KernelPCA(n_components=Y_validation.shape[1], kernel='rbf')
                X = transformer.fit_transform(Y_validation)

                X_NxD_list = self.split_NxD_matrix(X, self.num_tps)

                X_list = []
                Y_list = []
                X_list_temp = []
                Y_list_temp = []
                for i, (Y, X_seq) in enumerate(zip(Y_validation_list, X_NxD_list)):
                    X_list_temp.append(X_seq)
                    Y_list_temp.append(Y)
                    if (i + 1) % num_sequences_per_action_validation == 0:
                        X_list.append(X_list_temp)
                        Y_list.append(Y_list_temp)
                        X_list_temp = []
                        Y_list_temp = []

                rep_seqs = []
                rep_Ys = []
                for sequences in X_list:
                    sequences = np.array(sequences)
                    sequences_flattened = sequences.reshape((sequences.shape[0], -1))
                    kmedoids = KMedoids(n_clusters=1, metric="euclidean")
                    kmedoids.fit(sequences_flattened)
                    medoid_index = kmedoids.medoid_indices_[0]
                    rep_seqs.append(sequences[medoid_index])
                    rep_Ys.append(Y_list[i][medoid_index])

                self.X_init = np.vstack(rep_seqs)

        # Ensure X_init is defined
        if not hasattr(self, 'X_init'):
            self.X_init = None

        true_sequences = []
        pred_sequences_control = []
        for Yt, Yp in zip(self.Y_test_CCs, self.Y_train_CCs):
            Yt_key_list = []
            Yp_key_list = []
            for key in Yt.keys():
                if key == 'Hips':
                    Yt_key_list.append(Yt['Hips']['CC'])
                    Yp_key_list.append(Yp['Hips']['CC'])
                else:
                    Yt_key_list.append(Yt[key]['CC'] - Yt['Hips']['CC'])
                    Yp_key_list.append(Yp[key]['CC'] - Yp['Hips']['CC'])
            true_sequences.append(np.hstack(Yt_key_list))
            pred_sequences_control.append(np.hstack(Yp_key_list))


        return self.Y_vel, self.Y_pos, self.mn_start_pos, self.X_init

    def CC_2D_list_to_stick_dict_list(self, stick_mat_list):
        return [self.graphic.CC_2D_array_to_stick_dict(stick_mat) for stick_mat in stick_mat_list]

    def CC_dict_list_to_CC_array_list_min_PV(self, CC_dict_list):
        CC_array_list = []
        for Y in CC_dict_list:
            Y_key_list = []
            for key in Y.keys():
                if key == 'Hips':
                    Y_key_list.append(Y['Hips']['CC'])
                else:
                    Y_key_list.append(Y[key]['CC'] - Y['Hips']['CC'])
            CC_array_list.append(np.hstack(Y_key_list))
        return CC_array_list

    def _sequence_actions(self, npy_dir, file_prefix, num_tps):
        num_seqs_per_set = self.full_num_seqs_per_subj_per_act
        data_array = []
        for subject in self.subjects:
            file_pattern = f"{file_prefix}{subject}*{num_tps}.npy"
            subject_data = []
            for action in self.actions:
                action_data = []
                action_dir = os.path.join(npy_dir, action.lower())
                full_pattern = os.path.join(action_dir, file_pattern)
                abs_pattern = os.path.abspath(full_pattern)
                #print(f"Looking for files with pattern: {abs_pattern}")
                seq_files = glob(abs_pattern)
                if not seq_files:
                    print(f"No files found for pattern: {abs_pattern}")
                for seq_file in seq_files:
                    if len(action_data) >= num_seqs_per_set:
                        break
                    #print(f"Loading file: {seq_file}")
                    stick_mat = np.load(seq_file)
                    action_data.append(stick_mat)
                subject_data.append(action_data)
            data_array.append(subject_data)
        if not data_array or not data_array[0] or not data_array[0][0]:
            raise ValueError("No data loaded. Please check your file paths and patterns.")
        # Set the number of subjects and actions
        self.full_num_subjects = len(data_array)
        self.full_num_actions = len(data_array[0])
        self.num_dims = data_array[0][0][0].shape[1]

        return data_array, None

    def pos2vel_NxD(self, test_seq):
        test_v = test_seq[1:, :] - test_seq[:-1, :]
        # test_v[:, :self.cartesian_D] = np.zeros(test_v[:, :self.cartesian_D].shape)
        disc_ind = np.where(np.abs(test_v) > np.pi)

        add = 0
        for x, y, tv_i in zip(disc_ind[0], disc_ind[1], test_v[disc_ind]):
            if tv_i > 0:
                add = -2 * np.pi
            else:
                add = 2 * np.pi
            test_v[x, y] = tv_i + add

        return np.vstack([np.zeros([1, test_v.shape[1]]), test_v])

    def vel2pos_NxD(self, vel_NxD, action_ID=0):
        num_seqs = int(vel_NxD.shape[0] / self.num_tps)
        pos_NxD = np.zeros(vel_NxD.shape)
        for seq_i in range(num_seqs):
            for tp_j in range(self.num_tps - 1):
                if tp_j == 0:
                    pos_NxD[seq_i * self.num_tps, :] = self.mn_start_pos[:, action_ID]
                pos_NxD[seq_i * self.num_tps + tp_j + 1, :] = pos_NxD[seq_i * self.num_tps + tp_j, :] + vel_NxD[
                                                                                                        seq_i * self.num_tps + tp_j + 1,
                                                                                                        :]
        return pos_NxD


    def get_EP_labels(self, type):
        if type == 'ff':
            return self.score_EPs_ff
        elif type == 'fb':
            return self.score_EPs_fb
        else:
            raise ValueError(f'Type {type} not supported')

    def get_EP_score_list(self, type, Y_CCs):
        Y_list = []
        for i, (CC_i) in enumerate(Y_CCs):
            Y_list.append(self.get_EPs(CC_i, type))
        return Y_list

    def get_EPs(self, Y_SD, type):
        EP_list = []
        for EP_name in self.get_EP_labels(type):
            if EP_name == 'Hips':
                EP_list.append(Y_SD[EP_name]['CC'])
            else:
                EP_list.append(Y_SD[EP_name]['CC'] - Y_SD['Hips']['CC'])
        return EP_list

    def score(self, sample_len, type, Y1, Y2, Y1_ID, Y2_ID):
        if ':' in type:
            type, subtype = type.split(':')
        else:
            subtype = None
        if type == 'dynamic':
            return self.score_dynamic(sample_len, subtype, Y1, Y2, Y1_ID, Y2_ID)
        elif type == 'f1_dist_msad':
            return self.score_f1_dist_msad(sample_len, Y1, Y2, Y1_ID, Y2_ID)
        elif type == 'f1_frechet_ldj':
            return score_f1_dist_smoothness(sample_len, Y1, Y2, Y1_ID, Y2_ID, distance_metric='frechet', smoothness_metric='ldj')
        elif type == 'f1_dtw_mse_ldj':
            return score_f1_dist_smoothness(sample_len, Y1, Y2, Y1_ID, Y2_ID, distance_metric='dtw_mse', smoothness_metric='ldj')
        elif type == 'f1_frechet_sparc':
            return score_f1_dist_smoothness(sample_len, Y1, Y2, Y1_ID, Y2_ID, distance_metric='frechet', smoothness_metric='sparc')
        elif type == 'static':
            # self.score_static()
            raise NotImplementedError
        else:
            raise ValueError(f'Type {type} not supported')

    
    
    def get_joint_angles(self, Y_SD):
        """
        Extract joint angles from Y_SD.
        Assuming Y_SD is a dictionary where each key is a joint name,
        and each value is a dictionary with 'CC' key containing joint angles.
        """
        joint_angles = []
        for joint_name in Y_SD.keys():
            if 'CC' in Y_SD[joint_name]:
                joint_angles.append(Y_SD[joint_name]['CC'])
        # Stack joint angles along the second axis (assuming time is along axis 0)
        joint_angles_array = np.hstack(joint_angles)
        return joint_angles_array

    def compute_dtw_distance(self, x_g, x_p):
        from fastdtw import fastdtw
        # Ensure the input arrays are 2D
        x_g_2d = x_g.reshape(x_g.shape[0], -1)
        x_p_2d = x_p.reshape(x_p.shape[0], -1)

        # Compute DTW distance
        distance, _ = fastdtw(x_g_2d, x_p_2d, dist=euclidean)

        return distance

    def compute_confusion_matrix(self, action_IDs_g, action_IDs_p):
        cm = confusion_matrix(action_IDs_g, action_IDs_p)
        # Optionally, plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Compute accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"Classification accuracy: {accuracy * 100:.2f}%")
        return cm, accuracy

    def score_variance(self):
        return

    def index_subset(self, len_seq, actions, people, num_train_seqs_per_subj_per_act,
                     num_test_seqs_per_subj_per_act, num_validation_seqs_per_subj_per_act=0, num_simulations=1):
        """
        Splits data for given actions and sequences_per_action.
        """

        self.sub_num_actions = len(actions)
        self.sub_num_subjects = len(people)
        self.num_train_seqs_per_subj_per_act = num_train_seqs_per_subj_per_act
        self.num_test_seqs_per_subj_per_act = num_test_seqs_per_subj_per_act
        self.num_validation_seqs_per_subj_per_act = num_validation_seqs_per_subj_per_act
        self.sub_tot_num_train_seqs = self.num_train_seqs_per_subj_per_act * self.sub_num_actions * self.sub_num_subjects
        self.sub_tot_num_test_seqs = self.num_test_seqs_per_subj_per_act * self.sub_num_actions * self.sub_num_subjects
        self.num_simulations = num_simulations

        seq_indices = np.arange(0, self.full_num_seqs_per_subj_per_act, 1)

        actions_dict_train = {action: None for action in actions}
        actions_dict_test = {action: None for action in actions}
        actions_dict_validation = {action: None for action in actions}

        people_dict_train = {person: copy.deepcopy(actions_dict_train) for person in people}
        people_dict_test = {person: copy.deepcopy(actions_dict_test) for person in people}
        people_dict_validation = {person: copy.deepcopy(actions_dict_validation) for person in people}

        training_sims = {}
        validation_sims = {}
        testing_sims = {}
        for sim_num in range(self.num_folds):
            training_sims[sim_num] = copy.deepcopy(people_dict_train)
            testing_sims[sim_num] = copy.deepcopy(people_dict_test)
            validation_sims[sim_num] = copy.deepcopy(people_dict_validation)

        if num_train_seqs_per_subj_per_act == 1:
            combo_list = DSC_tools.DSC_format_list(len(actions), len(seq_indices), self.num_folds)
        else:
            print(
                'WARNING: Folds randomized. Maximization of differences between subsequent folds only occurs when num_train_seqs_per_subj_per_act == 1')
            combo_list = DSC_tools.DSC_simple(seq_indices, self.num_train_seqs_per_subj_per_act, self.num_folds,
                                              len(actions))

        len(combo_list) / len(seq_indices)
        for person in people:
            for sim_num, combo in enumerate(combo_list[:self.num_folds]):
                for act_num, indices in enumerate(combo):
                    sel_seq_indices_train = indices[0]
                    sel_seq_indices_test = indices[1][:num_test_seqs_per_subj_per_act]
                    sel_seq_indices_validation = indices[1][num_test_seqs_per_subj_per_act:]
                    training_sims[sim_num][person][act_num] = sel_seq_indices_train
                    testing_sims[sim_num][person][act_num] = sel_seq_indices_test
                    validation_sims[sim_num][person][act_num] = sel_seq_indices_validation
        if self.fold_num == 'all':
            return {'train': training_sims, 'test': testing_sims, 'validation': validation_sims}
        else:
            return {'train': {0: training_sims[self.fold_num]}, 'test': {0: testing_sims[self.fold_num]},
                    'validation': {0: validation_sims[self.fold_num]}}

    def indices2data(self, full_data_set, indices, num_seqs):
        num_sims = len(indices.keys())
        len_seqs = full_data_set.shape[0]
        num_dims = full_data_set.shape[1]
        num_actions = self.sub_num_actions
        num_people = self.sub_num_subjects
        data_array = np.zeros([num_sims,len_seqs, num_dims, num_seqs, num_actions, num_people])
        seq_eps = [((i + 1) * self.num_tps) - 1 for i in range(num_seqs * num_actions * num_people)]
        action_num = 0
        action_IDs_per_sim = []
        data_NxD_per_sim = []
        for sim_h, sim_ID in enumerate(indices.keys()):
            data_NxD = []
            action_IDs = []
            for person_i, person_ID in enumerate(indices[sim_h]):
                for action_j, action_ID in enumerate(indices[sim_h][person_ID]):
                    for seq_k, seq_ID in enumerate(indices[sim_h][person_ID][action_ID]):
                        data_array[sim_h,:, :, seq_k, action_j, person_i] = full_data_set[:, :, seq_ID, action_ID,
                                                                                     person_ID]
                        data_NxD.append(full_data_set[:, :, seq_ID, action_ID, person_ID])
                        action_IDs.append(action_ID)
            if data_NxD == []:
                data_NxD_array = np.array([])
            else:
                data_NxD_array = np.vstack(data_NxD)
            data_NxD_per_sim.append(data_NxD_array)
            action_IDs_per_sim.append(action_IDs)

        if num_sims == 1:
            data_array = data_array[0,:,:,:,:,:]
        data_NxD_array = np.vstack(data_NxD_per_sim)


        return data_array, data_NxD_array, seq_eps, action_IDs

    def get_NxD_subsets_pos(self, seq_len, actions, people, num_sequences_per_action_train,
                            num_sequences_per_action_test, num_sequences_per_action_validation):
        actions, people = self.get_acts_ppl(actions, people)
        self.indices = self.index_subset(seq_len, actions, people, num_sequences_per_action_train,
                                    num_sequences_per_action_test,
                                    num_validation_seqs_per_subj_per_act=num_sequences_per_action_validation)
        #print(indices)

        data_subset_train, _, self.seq_eps_train, self.action_IDs_train = self.indices2data(self.Y_pos,
                                                                                            self.indices['train'],
                                                                                            self.num_train_seqs_per_subj_per_act)
        data_subset_test, _, self.seq_eps_test, self.action_IDs_test = self.indices2data(self.Y_pos, self.indices['test'],
                                                                                         self.num_test_seqs_per_subj_per_act)
        data_subset_validation, _, self.seq_eps_validation, self.action_IDs_validation = self.indices2data(self.Y_pos,
                                                                                            self.indices['validation'],
                                                                                            num_sequences_per_action_validation)

        #print(np.sum(self.Y_pos.flatten()))
        self.controls = data_subset_train[:, :, :1, :, :]
        data_subset_norm_train, self.avgs, self.stds = self.normalize_per_dim(data_subset_train)
        #print('avgs: ', np.sum(self.avgs[:5].flatten()))
        #print('stds: ', np.sum(self.stds[:5].flatten()))
        data_subset_norm_test = self.normalize_per_dim(data_subset_test, self.avgs, self.stds)
        data_subset_norm_validation = self.normalize_per_dim(data_subset_validation, self.avgs, self.stds)
        self.mn_start_pos_norm = self.mn_start_pos - self.avgs
        self.mn_start_pos_norm /= self.stds
        Y_train_list = self.format_seq_list(data_subset_norm_train)
        Y_test_list = self.format_seq_list(data_subset_norm_test)
        Y_validation_list = self.format_seq_list(data_subset_norm_validation)
        Y_train = np.vstack(Y_train_list)
        Y_test = np.vstack(Y_test_list)
        if Y_validation_list == []:
            Y_validation = np.array([])
        else:
            Y_validation = np.vstack(Y_validation_list)
        return Y_train, Y_test, Y_validation, Y_train_list, Y_test_list, Y_validation_list

    def get_acts_ppl(self, actions, people):
        if isinstance(actions, list):
            actions_out = actions
        else:
            all_acts = np.arange(0, self.full_num_actions, 1)
            actions_out = np.random.choice(all_acts, actions, replace=False)
        if isinstance(people, list):
            people_out = people
        else:
            all_ppl = np.arange(0, self.full_num_subjects, 1)
            people_out = np.random.choice(all_ppl, people, replace=False)
        return actions_out, people_out

    def format_NxD(self, X):
        num_tps, num_dims, num_seqs, num_acts, num_ppl = X.shape
        NxD_list = []
        for act_i in range(num_acts):
            for prsn_j in range(num_ppl):
                for seq_k in range(num_seqs):
                    NxD_list.append(X[:, :, seq_k, act_i, prsn_j])

        return np.vstack(NxD_list)

    def format_seq_list(self, X):
        num_tps, num_dims, num_seqs, num_acts, num_ppl = X.shape
        NxD_list = []
        for act_i in range(num_acts):
            for prsn_j in range(num_ppl):
                for seq_k in range(num_seqs):
                    NxD_list.append(X[:, :, seq_k, act_i, prsn_j])

        return NxD_list

    def reformat_5D(self, X, num_tps, num_dims, num_seqs, num_acts, num_ppl):
        X_new = np.zeros([num_tps, num_dims, num_seqs, num_acts, num_ppl])
        for act_i in range(num_acts):
            for prsn_j in range(num_ppl):
                for seq_k in range(num_seqs):
                    beg_i = num_tps * num_seqs * num_ppl * act_i + num_tps * num_seqs * prsn_j + num_tps * seq_k
                    end_i = num_tps * num_seqs * num_ppl * act_i + num_tps * num_seqs * prsn_j + num_tps * seq_k + num_tps
                    X_new[:, :, seq_k, act_i, prsn_j] = X[beg_i:end_i, :]
        return X_new

    def get_NxD_subsets_vel(self, seq_len, actions, people, num_sequences_per_action_train,
                            num_sequences_per_action_test):
        actions, people = self.get_acts_ppl(actions, people)
        indices = self.index_subset(seq_len, actions, people, num_sequences_per_action_train,
                                    num_sequences_per_action_test)
        data_subset_train, _, self.seq_eps_train, self.action_IDs_train = self.indices2data(self.Y_vel, indices['train'],
                                                                                            self.sub_tot_num_train_seqs)
        data_subset_test, _, self.seq_eps_test, self.action_IDs_test = self.indices2data(self.Y_vel, indices['test'],
                                                                                         self.sub_tot_num_test_seqs)
        self.controls = data_subset_train[:, :, :1, :, :]
        data_subset_norm_train, self.avgs, self.stds = self.normalize_per_dim(data_subset_train)
        data_subset_norm_test = self.normalize_per_dim(data_subset_test, self.avgs, self.stds)
        self.Y_train = self.format_NxD(data_subset_norm_train)
        self.Y_test = self.format_NxD(data_subset_norm_test)
        return self.Y_train, self.Y_test

    def pos_list_to_vel_array_3D(self, Y):
        data_array = np.zeros(
            [self.num_tps, self.num_dims, self.full_num_seqs_per_subj_per_act, self.full_num_actions,
             self.full_num_subjects])
        for i, subject in enumerate(Y):
            for j, action in enumerate(subject):
                for k, seq in enumerate(action):
                    data_array[:, :, k, j, i] = self.pos2vel_NxD(seq)
        return data_array

    def list_to_array_3D(self, Y):
        data_array = np.zeros(
            [self.num_tps, self.num_dims, self.full_num_seqs_per_subj_per_act, self.full_num_actions,
             self.full_num_subjects])
        for i, subject in enumerate(Y):
            for j, action in enumerate(subject):
                for k, seq in enumerate(action):
                    data_array[:, :, k, j, i] = seq
        return data_array

    def denorm_trajs(self, Y_list, action_ID_list):
        """
        denormalize trajectories
        :param Y_list: List of trajectories
        :param action_ID_list: List integers corresponding to each action
        :return:
        """
        Y_denorm_list = []
        for Y, action_ID in zip(Y_list, action_ID_list):
            Y_denorm_list.append(self.denormalize(Y, action_ID))
        return Y_denorm_list

    def split_NxD_matrix(self, NxD, L):
        # Get the number of rows (N) and columns (D) of the original matrix
        N, D = NxD.shape

        # Calculate the number of sequences
        num_sequences = N // L

        # Initialize the list to hold the LxD matrices
        sequences = []

        # Loop through the NxD and extract LxD matrices
        for i in range(num_sequences):
            start_index = i * L
            end_index = start_index + L
            sequence = NxD[start_index:end_index, :]
            sequences.append(sequence)

        return sequences

    def Y_pos_list_to_stick_dicts_CCs(self, Y_pos_list):
        stick_dicts_CCs_list = []
        for Y_pos in Y_pos_list:
            stick_dicts_CCs_list.append(self.graphic.HGPLVM_angles_PV_to_stick_dicts_CCs(Y_pos))
        return stick_dicts_CCs_list

    def denormalize_by_action(self, Y, action_num):
        return (Y * self.stds[:, :, :, action_num, :].flatten().reshape(1, -1)) + self.avgs[:, :, :, action_num,
                                                                                  :].flatten().reshape(1, -1)

    def denormalize(self, Y, *args):
        return (Y * self.stds[:, :, :, :, :].flatten().reshape(1, -1)) + self.avgs[:, :, :, :, :].flatten().reshape(1,
                                                                                                                    -1)

    def normalize_per_dim(self, X, avgs=None, stds=None):
        X_new = X.copy()
        #print(np.sum(X_new[-5:]))
        if avgs is None or stds is None:
            avgs = np.round(np.mean(X_new, (0, 2, 3, 4), keepdims=True), 3)
            stds = np.round(np.std(X_new, (0, 2, 3, 4), keepdims=True), 3)
            stds[stds == 0] = 1
            # Create a normalized matrix
            X_new = (X_new - avgs) / stds
            #print(np.sum(X_new[-5:]))
            return X_new, avgs, stds
        else:
            X_new -= avgs
            X_new /= stds
            return X_new

    def plot_gs(self, num_actions, num_test_seqs_per_subj_per_act, num_latent_seqs=0, X_test=None, X_latent=None,
                X_pred=None, x_dim=0, y_dim=1, z_dim='time'):
        plt.figure()
        ax = plt.axes(projection='3d')
        colors = ['r', 'g', 'c', 'm', 'y', 'k', 'b', 'w','r', 'g', 'c', 'm', 'y', 'k', 'b', 'w']
        test_seq_num = 0
        for action_num in range(num_actions):
            color_val = colors[action_num]
            for test_seq_set_num in range(num_test_seqs_per_subj_per_act):
                if X_test is not None:
                    g1 = X_test[
                         (test_seq_set_num + action_num * num_test_seqs_per_subj_per_act) * self.num_tps:(1 + test_seq_set_num + action_num * num_test_seqs_per_subj_per_act) * self.num_tps,
                         :]
                    if z_dim == 'time':
                        zdata = np.linspace(1, self.num_tps, self.num_tps)[:, None].flatten()
                    else:
                        zdata = g1[:, z_dim]
                    if y_dim == 'time':
                        ydata = np.linspace(1, self.num_tps, self.num_tps)[:, None].flatten()
                    else:
                        ydata = g1[:, y_dim]
                    if x_dim == 'time':
                        xdata = np.linspace(1, self.num_tps, self.num_tps)[:, None].flatten()
                    else:
                        xdata = g1[:, x_dim]

                    ax.scatter3D(xdata, ydata, zdata, c=color_val)
            # test_seq_num += 1

                if X_pred is not None:
                    X_pred_1seq = X_pred[
                                  (test_seq_set_num + action_num * num_test_seqs_per_subj_per_act) * self.num_tps:(1 + test_seq_set_num + action_num * num_test_seqs_per_subj_per_act) * self.num_tps,
                                  :]
                    if x_dim == 'time':
                        xpred = np.linspace(1, self.num_tps, self.num_tps)[:, None].flatten()
                    else:
                        xpred = X_pred_1seq[:, x_dim]
                    if y_dim == 'time':
                        ypred = np.linspace(1, self.num_tps, self.num_tps)[:, None].flatten()
                    else:
                        ypred = X_pred_1seq[:, y_dim]
                    if z_dim == 'time':
                        zpred = np.linspace(1, self.num_tps, self.num_tps)[:, None].flatten()
                    else:
                        zpred = X_pred_1seq[:, z_dim]
                    ax.plot3D(xpred, ypred, zpred, c=color_val)

            if X_latent is not None:
                for latent_seq_num in range(num_latent_seqs):
                    X_ltnt_tmp = X_latent[
                                 (latent_seq_num + action_num * num_latent_seqs) * self.num_tps:
                                 (1 + latent_seq_num + action_num * num_latent_seqs) * self.num_tps, :]
                    if x_dim == 'time':
                        x_ltnt = np.linspace(1, self.num_tps, self.num_tps)[:, None].flatten()
                    else:
                        x_ltnt = X_ltnt_tmp[:, x_dim]
                    if y_dim == 'time':
                        y_ltnt = np.linspace(1, self.num_tps, self.num_tps)[:, None].flatten()
                    else:
                        y_ltnt = X_ltnt_tmp[:, y_dim]
                    if z_dim == 'time':
                        z_ltnt = np.linspace(1, self.num_tps, self.num_tps)[:, None].flatten()
                    else:
                        z_ltnt = X_ltnt_tmp[:, z_dim]

                    ax.scatter3D(x_ltnt, y_ltnt, z_ltnt, c=color_val, marker='s')

        plt.show()
        return ax, ax.figure

    def store_HGP_attributes(self, hgp):
        # self.hgp = hgp
        self.hgp_attr_dict = hgp.get_attribute_dict()

    def vel_array_3D_to_continuous_pos_array_3D(self, Y):
        num_tps, num_dims, num_seqs, num_acts, num_subs = Y.shape
        data_array = np.zeros(Y.shape)
        for i in range(num_subs):
            for j in range(num_acts):
                for k in range(num_seqs):
                    data_array[:, :, k, j, i] = self.vel2pos_NxD(Y[:, :, k, j, i], j)
        return data_array
