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
import contextlib


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


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


def test_score_joint_angles_real_data(data_set_name, cases_to_animate=None):
    """
    Test scoring metrics on joint angle data with various modifications.
    
    Args:
        data_set_name: Name of the dataset to use
        cases_to_animate: List of test case names to animate. If None, no animations are created.
                         Example: ['Slight noise', 'Frozen']
    """
    with suppress_stdout():
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
    


    # true_sequences = load_mccv_data(data_set_name, num_sequences).Y_test_list

    DSC = load_mccv_data(data_set_name, 0)
    true_sequences = DSC.Y_test_CCs[:10]
    Y_arr_list = DSC.CC_dict_list_to_CC_array_list_min_PV(true_sequences)
    
    # Create test cases with different modifications
    def apply_modification_to_all(Y_arr_list, modification_func, **kwargs):
        return [modification_func(seq, **kwargs) for seq in Y_arr_list]

    test_cases = [
        ("Slight noise", apply_modification_to_all(Y_arr_list, add_noise, noise_level=0.01)),
        ("Heavy noise", apply_modification_to_all(Y_arr_list, add_noise, noise_level=1.0)),
        ("Frozen", apply_modification_to_all(Y_arr_list, freeze_sequence, freeze_point=0.8)),
        ("Frozen with noise", apply_modification_to_all(Y_arr_list, lambda x: add_noise(freeze_sequence(x, 0.8), 0.1))),
        ("Warped 0.2", apply_modification_to_all(Y_arr_list, warp_sequence, warp_factor=0.2)),
        ("Warped 0.3", apply_modification_to_all(Y_arr_list, warp_sequence, warp_factor=0.3)),
        ("Warped 0.4", apply_modification_to_all(Y_arr_list, warp_sequence, warp_factor=0.4))
    ]

    # Labels for testing
    true_labels = np.array([0, 0, 0, 0, 0, 1, 1,1,1,1])  
    pred_labels = np.array([0, 0, 0, 0, 0, 1, 0,0,1,1])   # Perfect classification for testing metrics

    # Test each case
    sample_len = 0 # init index
    smoothness_metric = 'ldj'
    distance_metric = 'frechet'

    print("\nTesting different sequence modifications:")
    
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for case_name, modified_sequences in test_cases:
            print(f"\nResults for {case_name}:")
            result = score_f1_dist_smoothness(
                sample_len, 
                Y_arr_list,  # Original sequences
                modified_sequences,  # Modified sequences
                true_labels, 
                pred_labels,
                smoothness_metric=smoothness_metric,
                distance_metric=distance_metric
            )

    # Create animations only for selected test cases
    if cases_to_animate is not None:
        IFs_selected = []
        for case_name, modified_sequences in test_cases:
            if case_name in cases_to_animate:
                print(f"\nCreating animation for: {case_name}")
                Y_stick_dict_list1 = DSC.CC_2D_list_to_stick_dict_list(Y_arr_list)
                Y_stick_dict_list2 = DSC.CC_2D_list_to_stick_dict_list(modified_sequences)
                IFs = IFs_setup(Y_stick_dict_list2, Y_stick_dict_list1, DSC)
                IFs_selected.extend(IFs)
        return IFs_selected
    
    return None

# Run the test
if __name__ == "__main__":
    data_set_name = 'Bimanual 3D'
    #data_set_name = 'Movements CMU'
    
    # Specify which test cases to animate
    cases_to_animate = []  # Example: animate only these cases
    IFs = test_score_joint_angles_real_data(data_set_name, cases_to_animate)
    
    # Visualization code - only runs if cases were selected for animation
    if IFs is not None:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.animation as animation

        anis = []
        for i, IFs_i in enumerate(IFs):
            fig, animate = IFs_i.plot_animation_all_figures()
            anis.append(animation.FuncAnimation(fig, animate, 100, interval=100))
        matplotlib.pyplot.show()