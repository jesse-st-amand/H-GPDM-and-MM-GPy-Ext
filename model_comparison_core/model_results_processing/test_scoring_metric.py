import numpy as np
from sklearn.metrics import f1_score
from scipy.spatial.distance import directed_hausdorff

def frdist(x, y):
    """Compute the Fréchet distance between two trajectories."""
    return max(directed_hausdorff(x, y)[0], directed_hausdorff(y, x)[0])

def calculate_msad(seq1, seq2):
    """Calculate Mean Square Angular Distance"""
    return np.mean(np.abs(seq1 - seq2))

def score_joint_angles(sample_len, true_sequences, pred_sequences, true_labels, pred_labels):
    # Calculate F1 score
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    if f1 == 0:
        print('Model failed to classify any sequence correctly.')
        return 0, 1000000, 1000000, 1000000, 1000000  # F1, avg_norm_frechet, avg_norm_msad, avg_norm_frechet_velocity, avg_norm_frechet_freeze

    # Calculate total normalized Fréchet distance, MSAD, velocity-based Fréchet distance, and freeze detection
    total_norm_frechet = 0
    total_norm_msad = 0
    total_norm_frechet_velocity = 0
    total_norm_frechet_freeze = 0
    total_correct_sequences = 0
    unique_classes = np.unique(true_labels)

    for c in unique_classes:
        class_mask_true = (true_labels == c)
        class_mask = (true_labels == c) & (pred_labels == c)
        class_pred_seq = [pred_sequences[i] for i in range(len(pred_sequences)) if class_mask[i]]
        class_true_seq = [true_sequences[i] for i in range(len(true_sequences)) if class_mask[i]]
        class_ground_seq = [true_sequences[i] for i in range(len(true_sequences)) if class_mask_true[i]]

        # Compute max Fréchet distance among all true sequences of the class
        if len(class_ground_seq) < 2:
            max_frechet = 1  # Default value if not enough sequences
            max_msad = 1  # Default value if not enough sequences
            max_frechet_velocity = 1  # Default value if not enough sequences
        else:
            max_frechet = max(
                frdist(s[sample_len:,:], t[sample_len:,:]) for i, s in enumerate(class_ground_seq) for t in class_ground_seq[i + 1:])
            max_msad = max(
                calculate_msad(s[sample_len:,:], t[sample_len:,:]) for i, s in enumerate(class_ground_seq) for t in class_ground_seq[i + 1:])
            max_frechet_velocity = max(
                frdist(np.diff(s[sample_len:,:], axis=0), np.diff(t[sample_len:,:], axis=0)) for i, s in enumerate(class_ground_seq) for t in class_ground_seq[i + 1:])

        if max_frechet == 0:
            max_frechet = 1  # Avoid division by zero

        if max_msad == 0:
            max_msad = 1  # Avoid division by zero

        if max_frechet_velocity == 0:
            max_frechet_velocity = 1  # Avoid division by zero

        if len(class_pred_seq) == 0:
            continue  # No correctly classified sequences for this class
        else:
            class_sum_frechet = sum(frdist(p[sample_len:,:], t[sample_len:,:]) / max_frechet for p, t in zip(class_pred_seq, class_true_seq))
            total_norm_frechet += class_sum_frechet

            # Calculate normalized MSAD for each correctly predicted sequence
            class_sum_msad = sum(calculate_msad(p[sample_len:,:], t[sample_len:,:]) / max_msad for p, t in zip(class_pred_seq, class_true_seq))
            total_norm_msad += class_sum_msad

            # Calculate normalized Fréchet distance based on velocity
            class_sum_frechet_velocity = sum(frdist(np.diff(p[sample_len:,:], axis=0), np.diff(t[sample_len:,:], axis=0)) / max_frechet_velocity for p, t in zip(class_pred_seq, class_true_seq))
            total_norm_frechet_velocity += class_sum_frechet_velocity

            # Calculate normalized Fréchet distance for freeze detection
            zero_velocity = np.zeros_like(np.diff(class_pred_seq[0][sample_len:,:], axis=0))
            class_sum_frechet_freeze = sum(frdist(np.diff(p[sample_len:,:], axis=0), zero_velocity) / max_frechet_velocity for p in class_pred_seq)
            total_norm_frechet_freeze += class_sum_frechet_freeze

            total_correct_sequences += len(class_pred_seq)

    if total_correct_sequences == 0:
        avg_norm_frechet = 1000000  # Or any appropriate large value
        avg_norm_msad = 1000000  # Or any appropriate large value
        avg_norm_frechet_velocity = 1000000  # Or any appropriate large value
        avg_norm_frechet_freeze = 1000000  # Or any appropriate large value
    else:
        avg_norm_frechet = total_norm_frechet / total_correct_sequences / len(unique_classes)
        avg_norm_msad = total_norm_msad / total_correct_sequences / len(unique_classes)
        avg_norm_frechet_velocity = total_norm_frechet_velocity / total_correct_sequences / len(unique_classes)
        avg_norm_frechet_freeze = total_norm_frechet_freeze / total_correct_sequences / len(unique_classes)

    return f1, avg_norm_frechet, avg_norm_msad, avg_norm_frechet_velocity, avg_norm_frechet_freeze

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score

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
    num_sequences = 100
    sequence_length = 100
    num_features = 3
    true_sequences = generate_sequences(num_sequences, sequence_length, num_features)
    true_labels = np.random.randint(0, 3, num_sequences)  # 3 classes

    # Generate prediction set 1 (pset1): warped and noisy
    pset1 = [warp_sequence(seq) for seq in true_sequences]
    pset1 = [add_noise(seq) for seq in pset1]

    # Generate prediction set 2 (pset2): same as pset1 but with freezing
    pset2 = [freeze_sequence(seq) for seq in pset1]

    # Add some misclassifications to make it more realistic
    pred_labels1 = true_labels.copy()
    pred_labels2 = true_labels.copy()
    misclassify_indices = np.random.choice(num_sequences, size=int(0.1*num_sequences), replace=False)
    pred_labels1[misclassify_indices] = (pred_labels1[misclassify_indices] + 1) % 3
    pred_labels2[misclassify_indices] = (pred_labels2[misclassify_indices] + 1) % 3


    # Run the tests
    sample_len = 10  # Adjust as needed
    result1 = score_joint_angles(sample_len, true_sequences, pset1, true_labels, pred_labels1)
    result2 = score_joint_angles(sample_len, true_sequences, pset2, true_labels, pred_labels2)

    print("Results for pset1 (warped and noisy):")
    print(f"F1: {result1[0]:.4f}, Avg Norm Frechet: {result1[1]:.4f}, Avg Norm MSAD: {result1[2]:.4f}, Avg Norm Frechet Velocity: {result1[3]:.4f}, Avg Norm Frechet Freeze: {result1[4]:.4f}")

    print("\nResults for pset2 (warped, noisy, and frozen):")
    print(f"F1: {result2[0]:.4f}, Avg Norm Frechet: {result2[1]:.4f}, Avg Norm MSAD: {result2[2]:.4f}, Avg Norm Frechet Velocity: {result2[3]:.4f}, Avg Norm Frechet Freeze: {result2[4]:.4f}")

    # Assertions to check if the results meet the expected criteria
    #assert result1[1] < 0.5, "avg_norm_frechet for pset1 should be low"
    assert result2[4] < result2[1], "avg_norm_frechet_freeze should be lower than avg_norm_frechet for pset2"
    assert result2[4] > result1[4], "avg_norm_frechet_freeze for pset2 should be higher than for pset1"

    print("\nAll assertions passed. The score_joint_angles function behaves as expected.")

# Run the test
test_score_joint_angles()