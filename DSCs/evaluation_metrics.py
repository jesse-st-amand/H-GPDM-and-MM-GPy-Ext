import numpy as np
from sklearn.metrics import f1_score
from frechetdist import frdist
from scipy.spatial.distance import euclidean
from sparc import sparc, log_dimensionless_jerk

def calculate_distance(s1, s2, distance_metric='frechet'):
    if distance_metric == 'dtw':
        from fastdtw import fastdtw
        distance, _ = fastdtw(s1, s2, dist=euclidean)
        return distance
    elif distance_metric == 'frechet':
        return frdist(s1, s2)
    elif distance_metric == 'dtw_mse':
        from fastdtw import fastdtw
        # Apply fastDTW to align sequences, then calculate MSE
        _, path = fastdtw(s1, s2, dist=euclidean)
        s1_aligned = np.array([s1[p[0]] for p in path])
        s2_aligned = np.array([s2[p[1]] for p in path])
        mse = np.mean(np.sum((s1_aligned - s2_aligned) ** 2, axis=1))
        return mse
    elif distance_metric == 'mse':
        # Ensure sequences are the same length
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        mse = np.mean(np.sum((s1 - s2) ** 2, axis=1))
        return mse
    else:
        raise ValueError("Invalid distance metric. Choose 'dtw', 'frechet', 'dtw_mse', 'mse', or 'dtw_frechet'.")


def calculate_freeze_metric(prediction, ground_truth):
    """
    Calculate a metric that detects sequence freezing by comparing net movement over time windows.
    Random noise should average out in the net displacement calculation.
    
    Args:
        prediction, ground_truth: Input sequences (n_samples, n_dimensions) - position data
        
    Returns:
        float: Freeze metric score. Higher values indicate more freezing relative to ground truth.
    """
    # Calculate net displacement over sliding windows
    window_size = 15  # Number of frames to consider for net movement
    
    def get_net_velocities(sequence):
        n_samples = len(sequence)
        net_velocities = []
        
        '''for i in range(0, n_samples - window_size):
            # Sum up all the small movements within the window
            movements = sequence[i+1:i+window_size+1] - sequence[i:i+window_size]
            # Get the total displacement magnitude over the window
            total_movement = np.linalg.norm(np.sum(movements, axis=0))
            net_velocities.append(total_movement)'''

        for i in range(0, n_samples - window_size):
            # Sum up all the small movements within the window
            movements = sequence[i+window_size] - sequence[i]
            # Get the total displacement magnitude over the window
            total_movement = np.linalg.norm(movements)
            net_velocities.append(total_movement)
            
        return np.array(net_velocities)
    
    # Get net velocities for both sequences
    vp = get_net_velocities(prediction)
    vg = get_net_velocities(ground_truth)
    
    # Calculate mean movement
    vp_mean = np.mean(vp)
    vg_mean = np.mean(vg)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    # Calculate ratio of mean movements
    freeze_score = (vg_mean + eps)/(vp_mean + eps)
    
    return freeze_score

def score_f1_dist_smoothness(sample_len, true_sequences, pred_sequences, true_labels, pred_labels,
                           distance_metric='frechet', smoothness_metric='ldj'):
        
        # Calculate F1 score
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        if f1 == 0:
            print('Model failed to classify any sequence correctly.')
            return {'f1': 0, 'avg_norm_distance': 1000000, 'avg_smoothness': 1000000, 'avg_freeze': 1000000}

        # Initialize variables for metrics
        total_norm_distance = 0
        total_smoothness = 0  # Changed from total_norm_smoothness
        total_freeze = 0
        total_correct_sequences = 0

        unique_classes = np.unique(true_labels)
        num_classes = len(unique_classes)
        for c in unique_classes:
            class_mask = (true_labels == c) & (pred_labels == c)
            class_mask_true = (true_labels == c)

            class_pred_seq = [pred_sequences[i] for i in range(len(pred_sequences)) if class_mask[i]]
            class_true_seq = [true_sequences[i] for i in range(len(true_sequences)) if class_mask[i]]
            class_ground_seq = [true_sequences[i] for i in range(len(true_sequences)) if class_mask_true[i]]
            if len(class_pred_seq) == 0:
                continue  # No correctly classified sequences for this class

            # Compute max distance among all true sequences of the class
            if len(class_ground_seq) < 2:
                max_distance = 1  # Default value if not enough sequences
            else:
                max_distance = max(calculate_distance(s[sample_len:, :], t[sample_len:, :], distance_metric)
                                   for i, s in enumerate(class_ground_seq) for t in class_ground_seq[i + 1:])

            # Calculate normalized distance and raw smoothness
            class_sum_distance = sum(calculate_distance(p[sample_len:, :], t[sample_len:, :], distance_metric) / max_distance
                                     for p, t in zip(class_pred_seq, class_true_seq))
            class_sum_smoothness = sum(calculate_smoothness_metric(p[sample_len:, :], t[sample_len:, :], smoothness_metric)
                                 for p, t in zip(class_pred_seq, class_true_seq))
            
            # Calculate freeze metric
            class_sum_freeze = sum(calculate_freeze_metric(p[sample_len:, :], t[sample_len:, :])
                                 for p, t in zip(class_pred_seq, class_true_seq))

            total_norm_distance += class_sum_distance
            total_smoothness += class_sum_smoothness  
            total_freeze += class_sum_freeze
            total_correct_sequences += len(class_pred_seq)

        if total_correct_sequences == 0:
            return {'f1': f1, 'avg_norm_distance': 1000000, 'avg_smoothness': 1000000, 'avg_freeze': 1000000}

        avg_norm_distance = total_norm_distance / total_correct_sequences / num_classes
        avg_smoothness = total_smoothness / total_correct_sequences  # Changed from avg_norm_smoothness
        avg_freeze = total_freeze / total_correct_sequences  # No need to normalize by num_classes

        print('\nMETRIC DETAILS:')
        print(f'F1 Score: {f1:.4f}')
        print(f'Average Normalized Distance: {avg_norm_distance:.4f}')
        print(f'Average Smoothness: {avg_smoothness:.4f}')  # Changed from Average Normalized Smoothness
        print(f'Average Freeze Score: {avg_freeze:.4f}')

        return {
            'f1': f1, 
            'avg_norm_distance': avg_norm_distance, 
            'avg_smoothness': avg_smoothness,  
            'avg_freeze': avg_freeze
        }


def calculate_smoothness_metric(prediction, ground_truth, smoothness_metric='sparc', fs=100, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calculate smoothness as a ratio between prediction and ground truth.
    Higher values indicate the prediction is less smooth compared to ground truth.
    
    Args:
        prediction, ground_truth: Input sequences (n_samples, n_dimensions) - position data
        smoothness_metric: 'sparc' or 'ldj'
        fs: sampling frequency in Hz (default 100)
    """
    fs = 256
    # Convert position to velocity using central difference
    def pos_to_vel(pos, fs):
        vel = np.gradient(pos, axis=0)
        return vel

    # Convert both sequences to velocity
    v_pred = pos_to_vel(prediction, fs)
    v_gt = pos_to_vel(ground_truth, fs)

    # Handle each dimension separately and take mean
    n_dims = prediction.shape[1]
    smoothness_ratios = []

    for d in range(n_dims):
        if smoothness_metric == 'sparc':
            score_pred = sparc(v_pred[:, d], fs=fs, padlevel=padlevel, fc=fc, amp_th=amp_th)[0]
            score_gt = sparc(v_gt[:, d], fs=fs, padlevel=padlevel, fc=fc, amp_th=amp_th)[0]
        elif smoothness_metric == 'ldj':
            score_pred = log_dimensionless_jerk(v_pred[:, d], fs=fs)
            score_gt = log_dimensionless_jerk(v_gt[:, d], fs=fs)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        # Calculate ratio (similar to freeze metric)
        if smoothness_metric == 'ldj':
            # For LDJ, more negative means smoother
            # We use exp() to convert to positive numbers where larger means less smooth
            ratio = (score_pred + eps) / (score_gt + eps)
        else:
            # For SPARC, closer to 0 means smoother
            # We use -score to make larger values mean less smooth
            ratio = (-score_gt + eps) / (-score_pred + eps)
            
        smoothness_ratios.append(ratio)

    smoothness_ratios_sans_nan = [x for x in smoothness_ratios if not np.isnan(x)]
    return np.mean(smoothness_ratios_sans_nan)

def calculate_smoothness_singular(s1, smoothness_metric='sparc', fs=100, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calculate smoothness difference between two sequences.
    
    Args:
        s1, s2: Input sequences (n_samples, n_dimensions)
        smoothness_metric: 'sparc' or 'ldj'
        fs: sampling frequency in Hz (default 100)
    """
    # Handle each dimension separately and take mean
    n_dims = s1.shape[1]
    smoothness_diffs = []
    
    for d in range(n_dims):
        if smoothness_metric == 'sparc':
            # Extract just the first element if sparc returns a tuple
            score = sparc(s1[:, d], fs=fs, padlevel=padlevel, fc=fc, amp_th=amp_th)[0]
        elif smoothness_metric == 'ldj':
            score = log_dimensionless_jerk(s1[:, d], fs=fs)
        smoothness_diffs.append(score)
    
    return np.mean(smoothness_diffs)

