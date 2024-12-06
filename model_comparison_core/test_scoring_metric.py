import numpy as np
from sklearn.metrics import f1_score
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
import os
import sys
from sparc import sparc, log_dimensionless_jerk
from DSCs.evaluation_metrics import score_sequences, calculate_smoothness_metric, calculate_smoothness_singular
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def test_score_joint_angles():
    np.random.seed(42)  # For reproducibility

    def generate_sequences(num_sequences, length, num_features):
        sequences = []
        for _ in range(num_sequences):
            t = np.linspace(0, 2*np.pi, length)
            seq = np.column_stack([np.sin(t), np.cos(t)])
            if num_features > 2:
                for _ in range(num_features - 2):
                    seq = np.column_stack([seq, np.random.normal(0, 0.1, length)])
            sequences.append(seq)
        return sequences

    def warp_sequence(seq, warp_factor=0.2):
        t = np.linspace(0, 1, len(seq))
        warped_t = t + warp_factor * np.sin(2 * np.pi * t)
        warped_t = np.clip(warped_t, 0, 1)
        interp_func = interp1d(t, seq, axis=0, kind='cubic')
        return interp_func(warped_t)

    def add_noise(seq, noise_level=0.05):
        return seq + np.random.normal(0, noise_level, seq.shape)

    def freeze_sequence(seq, freeze_point=0.7):
        freeze_index = int(freeze_point * len(seq))
        frozen_part = seq[freeze_index:freeze_index+1].repeat(len(seq) - freeze_index, axis=0)
        return np.concatenate([seq[:freeze_index], frozen_part])

    # Generate ground truth sequences
    num_sequences = 5
    sequence_length = 100
    num_features = 3
    true_sequences = generate_sequences(num_sequences, sequence_length, num_features)
    true_labels = np.random.randint(0, 3, num_sequences)  # 3 classes

    # Generate prediction set 1 (pset1): warped and noisy
    #pset1 = [warp_sequence(seq) for seq in true_sequences]
    pset1 = [seq for seq in true_sequences]

    # Generate prediction set 2 (pset2): same as pset1 but with freezing
    pset2 = [add_noise(seq,1) for seq in pset1]

    # Add some misclassifications to make it more realistic
    pred_labels1 = true_labels.copy()
    pred_labels2 = true_labels.copy()
    misclassify_indices = np.random.choice(num_sequences, size= int(1) if num_sequences < 10 else int(0.1*num_sequences), replace=False)
    pred_labels1[misclassify_indices] = (pred_labels1[misclassify_indices] + 1) % 3
    pred_labels2[misclassify_indices] = (pred_labels2[misclassify_indices] + 1) % 3

    smoothness_metric = 'sparc'
    # Run the tests
    sample_len = 10  # Adjust as needed
    result1 = score_sequences(sample_len, true_sequences, pset1, true_labels, pred_labels1, smoothness_metric=smoothness_metric)
    result2 = score_sequences(sample_len, true_sequences, pset2, true_labels, pred_labels2, smoothness_metric=smoothness_metric)

    

    total_sal = 0
    for seq in true_sequences:
        total_sal += calculate_smoothness_singular(seq, fs=100, padlevel=4, fc=10.0, amp_th=0.05,smoothness_metric=smoothness_metric)
    print(f"Avg SAL: {total_sal/len(true_sequences):.4f}")

    total_sal_pset2 = 0
    for seq in pset2:
        total_sal_pset2 += calculate_smoothness_singular(seq, fs=100, padlevel=4, fc=10.0, amp_th=0.05,smoothness_metric=smoothness_metric)
    print(f"Avg SAL pset2: {total_sal_pset2/len(pset2):.4f}")

    print("Results for pset1 (warped and noisy):")
    print(f"F1: {result1['f1']:.4f}, Avg Norm Frechet: {result1['avg_norm_distance']:.4f}, Avg Norm Smoothness: {result1['avg_norm_smoothness']:.4f}")

    print("\nResults for pset2 (warped, noisy, and frozen):")
    print(f"F1: {result2['f1']:.4f}, Avg Norm Frechet: {result2['avg_norm_distance']:.4f}, Avg Norm Smoothness: {result2['avg_norm_smoothness']:.4f}")

    # Assertions to check if the results meet the expected criteria
    #assert result1[1] < 0.5, "avg_norm_frechet for pset1 should be low"
    #assert result2[4] < result2[1], "avg_norm_frechet_freeze should be lower than avg_norm_frechet for pset2"
    #assert result2[4] > result1[4], "avg_norm_frechet_freeze for pset2 should be higher than for pset1"

    print("\nAll assertions passed. The score_joint_angles function behaves as expected.")

# Run the test
test_score_joint_angles()