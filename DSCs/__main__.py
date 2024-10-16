from DSCs.data_func import data_func
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation

def add_jitter(sequence, jitter_strength=0.05):
    """
    Add random jitter to a sequence to simulate jerky movement.

    :param sequence: numpy array of shape (100, D)
    :param jitter_strength: float, controls the intensity of the jitter
    :return: numpy array of shape (100, D) with added jitter
    """
    jitter = np.random.normal(0, jitter_strength, sequence.shape)
    return sequence + jitter


def create_jerky_dataset(data_list, jitter_strength=0.05):
    """
    Create a jerky version of the entire dataset.

    :param data_list: list of numpy arrays, each of shape (100, D)
    :param jitter_strength: float, controls the intensity of the jitter
    :return: list of numpy arrays, each of shape (100, D) with added jitter
    """
    return [add_jitter(sequence, jitter_strength) for sequence in data_list]


def freeze_sequence(sequence, freeze_point):
    """
    Freeze a sequence at a given point.

    :param sequence: numpy array of shape (100, D)
    :param freeze_point: int, the point at which to freeze the sequence
    :return: numpy array of shape (100, D) with frozen values after freeze_point
    """
    frozen_sequence = sequence.copy()
    frozen_sequence[freeze_point:] = sequence[freeze_point]
    return frozen_sequence


def create_frozen_dataset(data_list, min_freeze_point=10):
    """
    Create a dataset where each sequence freezes at a random point after sample_len.

    :param data_list: list of numpy arrays, each of shape (100, D)
    :param sample_len: int, minimum point after which freezing can occur
    :param min_freeze_point: int, optional minimum freeze point (defaults to sample_len if not provided)
    :return: list of numpy arrays, each of shape (100, D) with frozen sequences
    """
    frozen_data_list = []
    for sequence in data_list:
        freeze_point = np.random.randint(min_freeze_point, 100)
        frozen_data_list.append(freeze_sequence(sequence, freeze_point))

    return frozen_data_list


def shuffle_dataset_and_labels(data_list, labels):
    """
    Shuffle the dataset and labels in the same order.

    :param data_list: list of numpy arrays, each of shape (100, D)
    :param labels: list or array of labels
    :return: tuple of (shuffled data list, shuffled labels, shuffle indices)
    """
    indices = np.arange(len(data_list))
    np.random.shuffle(indices)

    shuffled_data = [data_list[i] for i in indices]
    shuffled_labels = [labels[i] for i in indices]

    return shuffled_data, shuffled_labels, indices

def print_dicts(scores_list):
    for scores in scores_list:
        for key in scores.keys():
            print(key, scores[key])

if __name__ == "__main__":
    # Main configuration
    data_set_name = 'Bimanual 3D'
    #data_set_name = 'Movements CMU'
    print('WARNING: called main.')
    if data_set_name == "Bimanual 3D":
        actions = [0,1,2,3,4]
        num_sequences_per_action_train = 1
        num_sequences_per_action_test = 1
        people = [0]
    elif data_set_name == "Movements CMU":
        actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        num_sequences_per_action_train = 1
        num_sequences_per_action_test = 1
        people = [0]
    else:
        raise ValueError("Dataset does not exist.")

    seq_len = 100


    num_folds = 5 * (num_sequences_per_action_train + num_sequences_per_action_test)
    fold_num = 4

    data_set_class_dict = {
        'data_set_name': data_set_name,
        'actions': actions,
        'people': people,
        'seq_len': seq_len,
        'num_folds': num_folds,
        'fold_num': fold_num,
        'num_sequences_per_action_train': num_sequences_per_action_train,
        'num_sequences_per_action_test': num_sequences_per_action_test,
        'X_init': None
    }

    data_set_class = data_func(data_set_class_dict)


    def IF_func(pred_sequences):
        return [data_set_class.IFs_func([Y_k_SD,Y_p_SD]) for Y_k_SD,Y_p_SD in zip(data_set_class.Y_test_CCs,pred_sequences)]

    output = 'none'
    #output = 'score'
    units = 'CCs'
    act_ID = 0
    sample_len = 10
    true_labels = data_set_class.action_IDs_test
    pred_labels = data_set_class.action_IDs_train

    pred_joint_angles_control = data_set_class.denorm_trajs(data_set_class.Y_train_list, pred_labels)
    true_joint_angles = data_set_class.denorm_trajs(data_set_class.Y_test_list, true_labels)
    if units == 'joint angles':
        pred_sequences_control = pred_joint_angles_control
        true_sequences = true_joint_angles
    elif units == 'CCs':
        #pred_CCs = data_set_class.Y_pos_list_to_stick_dicts_CCs(pred_joint_angles_control)
        #true_CCs = data_set_class.Y_pos_list_to_stick_dicts_CCs(true_joint_angles)
        pred_CCs = data_set_class.Y_pos_list_to_stick_dicts_CCs(true_joint_angles)

        #pred_CCs = data_set_class.Y_train_CCs
        true_CCs = data_set_class.Y_test_CCs


        true_sequences = data_set_class.CC_dict_list_to_CC_array_list_min_PV(true_CCs)
        pred_sequences_control = data_set_class.CC_dict_list_to_CC_array_list_min_PV(pred_CCs)

        '''for Yt, Yp in zip(true_CCs, pred_CCs):
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
            pred_sequences_control.append(np.hstack(Yp_key_list))'''
    elif units == 'CCs EPs':
        #pred_sequences_control = [mat[:,np.array([15,16,69,70],dtype=int)] for mat in data_set_class.graphic.SAPV_list_2_SAPCC_list(pred_joint_angles_control)]
        #true_sequences = [mat[:,np.array([15,16,69,70],dtype=int)] for mat in data_set_class.graphic.SAPV_list_2_SAPCC_list(true_joint_angles)]
        true_sequences = [np.hstack([Y['LeftHand']['CC'],Y['RightHand']['CC']]) for Y in data_set_class.Y_test_CCs]
        pred_sequences_control = [np.hstack([Y['LeftHand']['CC'],Y['RightHand']['CC']]) for Y in data_set_class.Y_train_CCs]
    else:
        raise ValueError("Unit type does not exist.")


    pred_sequences_control = pred_sequences_control
    pred_sequences_jitter = create_jerky_dataset(pred_sequences_control, jitter_strength=0.15)
    pred_sequences_frozen = create_jerky_dataset(
                            create_frozen_dataset(pred_sequences_control, min_freeze_point=sample_len)
                            , jitter_strength=0.15)
    # Create shuffled dataset
    pred_sequences_shuffled, shuffled_labels, shuffle_indices = shuffle_dataset_and_labels(pred_sequences_control,
                                                                                           pred_labels)
    if output == 'score':
        scores_list = []
        for ps, labels in [(pred_sequences_control, pred_labels),
                           (pred_sequences_jitter, pred_labels),
                           (pred_sequences_frozen, pred_labels),
                           (pred_sequences_shuffled, shuffled_labels),
                           (pred_sequences_shuffled, pred_labels)
                           ]:
            scores = data_set_class.score_joint_angles(sample_len, true_sequences, ps, true_labels, labels,
                                                       distance_metric='frechet')
            scores_list.append(scores)

        print_dicts(scores_list)
    elif output == 'animate':
        if units == 'joint angles':
            pred_CCs = data_set_class.Y_pos_list_to_stick_dicts_CCs(pred_sequences_control)
        elif units == 'CCs':
            pred_CCs = data_set_class.CC_2D_list_to_stick_dict_list(pred_sequences_control)
        else:
            raise ValueError("Unit type does not exist.")
        IFs = IF_func(pred_CCs)
        anis = []
        for i, IFs_i in enumerate(IFs):
            fig, animate = IFs_i.plot_animation_all_figures()
            anis.append(animation.FuncAnimation(fig, animate, seq_len, interval=100))
        matplotlib.pyplot.show()



