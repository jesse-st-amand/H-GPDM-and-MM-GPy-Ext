import numpy as np
from sklearn.metrics import f1_score
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
import os
import sys
from sparc import sparc, log_dimensionless_jerk
from DSCs.evaluation_metrics import score_f1_dist_smoothness, calculate_smoothness_metric, calculate_smoothness_singular
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import pickle


def IFs_setup(prediction, ground_truth, data_set_class):
    IFs = []
    for i, (Y_p_SD, Y_k_SD) in enumerate(zip(prediction, ground_truth)):
        IFs.append(data_set_class.IFs_func([Y_k_SD, Y_p_SD]))
    return IFs
def load_mccv_data(data_set_name, fold_num):
    # Define the path where the pickled files are stored
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    load_path = parent_directory + f'/DSCs/data/MCCV/{data_set_name}/'

    # Initialize an empty list to store the loaded objects

    file_path = os.path.join(load_path, f'data_set_class_{fold_num}.pkl')

    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data_set_class = pickle.load(f)
        print(f"Loaded data_set_class for fold {fold_num} from {file_path}")
    else:
        print(f"Warning: File not found for fold {fold_num} at {file_path}")

    return data_set_class


def test_score_joint_angles_real_data(data_set_name):
    np.random.seed(42)  # For reproducibility

    def warp_sequence(seq, warp_factor=0.2):
        t = np.linspace(0, 1, len(seq))
        warped_t = t + warp_factor * np.sin(2 * np.pi * t)
        warped_t = np.clip(warped_t, 0, 1)
        interp_func = interp1d(t, seq, axis=0, kind='cubic')
        return interp_func(warped_t)

    def add_noise(seq, noise_level=0.05):
        return seq + np.random.normal(0, noise_level, seq.shape)

    def freeze_sequence(seq, freeze_point=0.6):
        freeze_index = int(freeze_point * len(seq))
        frozen_part = seq[freeze_index:freeze_index + 1].repeat(len(seq) - freeze_index, axis=0)
        return np.concatenate([seq[:freeze_index], frozen_part])

    # Generate ground truth sequences
    num_sequences = 12
    sequence_length = 100
    num_features = 3


    # true_sequences = load_mccv_data(data_set_name, num_sequences).Y_test_list

    DSC = load_mccv_data(data_set_name, 0)
    true_sequences = DSC.Y_test_CCs[:4]
    Y_arr_list = DSC.CC_dict_list_to_CC_array_list_min_PV(true_sequences)
    
    # Create test cases with different modifications
    test_cases = [
        ("Slight noise", add_noise(Y_arr_list[0], 0.1)),
        ("Heavy noise", add_noise(Y_arr_list[1], 1.0)),
        ("Frozen", freeze_sequence(Y_arr_list[2],.2)),
        ("Frozen with noise", add_noise(freeze_sequence(Y_arr_list[3],.2), 0.1))
    ]

    # Labels for testing (4 sequences, 3 classes)
    true_labels = np.random.randint(0, 3, num_sequences)  # 3 classes
    pred_labels = true_labels.copy()  # Perfect classification for testing metrics

    # Test each case
    sample_len = 0
    smoothness_metric = 'ldj'

    print("\nTesting different sequence modifications:")
    
    import warnings

# Method 1: Using context manager (preferred for specific code blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for i, (case_name, modified_seq) in enumerate(test_cases):
            print(f"\nResults for {case_name}:")
            result = score_f1_dist_smoothness(
                sample_len, 
                [Y_arr_list[i]], 
                [modified_seq], 
                [true_labels[i]], 
                [pred_labels[i]],
                smoothness_metric=smoothness_metric
            )

    # Convert sequences for visualization
    Y_modified_list = [case[1] for case in test_cases]
    Y_stick_dict_list1 = DSC.CC_2D_list_to_stick_dict_list(Y_arr_list)
    Y_stick_dict_list2 = DSC.CC_2D_list_to_stick_dict_list(Y_modified_list)
    IFs = IFs_setup(Y_stick_dict_list2, Y_stick_dict_list1, DSC)

    return IFs

# Run the test
if __name__ == "__main__":
    data_set_name = 'Bimanual 3D'
    #data_set_name = 'Movements CMU'
    IFs = test_score_joint_angles_real_data(data_set_name)
    
    # Visualization code
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.animation as animation

    anis = []
    for i, IFs_i in enumerate(IFs):
        fig, animate = IFs_i.plot_animation_all_figures()
        anis.append(animation.FuncAnimation(fig, animate, 100, interval=100))
    matplotlib.pyplot.show()