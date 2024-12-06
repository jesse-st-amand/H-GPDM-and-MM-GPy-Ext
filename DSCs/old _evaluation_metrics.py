def score_dynamic(self, sample_len, type, Y_ground_t_CCs, Y_pred_CCs, action_IDs_g, action_IDs_p):
        EP_g_list = self.get_EP_score_list(type, Y_ground_t_CCs)
        EP_p_list = self.get_EP_score_list(type, Y_pred_CCs)

        prediction_scores = np.zeros((len(EP_p_list), len(EP_g_list)))
        ground_scores = np.zeros((len(EP_g_list), len(EP_g_list)))

        for i, (EP_p, ID_p, ID_p_g, pred_traj_prob_list) in enumerate(
                zip(EP_p_list, action_IDs_p, action_IDs_g, self.results_dict['pred_traj_lists'])):
            EP_p = np.array(EP_p)
            for j, (EP_g, ID_g) in enumerate(zip(EP_g_list, action_IDs_g)):
                EP_g = np.array(EP_g)
                prediction_scores[i, j] = self.score_value_dynamic(EP_g, EP_p, ID_g, ID_p, ID_p_g,
                                                                   pred_traj_prob_list)

        for i, (EP_g_i, ID_g_i) in enumerate(zip(EP_g_list, action_IDs_g)):
            EP_g_i = np.array(EP_g_i)
            for j, (EP_g_j, ID_g_j) in enumerate(zip(EP_g_list, action_IDs_g)):
                EP_g_j = np.array(EP_g_j)
                ground_scores[i, j] = self.score_value_dynamic(EP_g_i, EP_g_j, ID_p_g=ID_g_i, ID_g=ID_g_j)
        n = int(ground_scores.shape[1] / len(self.actions_indices))
        sq_diag = np.vstack(DSC_tools.get_square_diagonal(ground_scores, n, len(self.actions_indices)))
        pred = np.sum(prediction_scores ** 2, axis=1)

        if len(ground_scores) == 1:
            var_len = len(ground_scores[0, :])
        else:
            var_len = (len(ground_scores[0, :]) - 1)
        var = np.sum(sq_diag ** 2, axis=1) / var_len
        scores_per_act = []
        for i, is_zero in enumerate(var == 0):
            if is_zero:
                scores_per_act.append(np.sqrt(pred[i]))
            else:
                scores_per_act.append(pred[i] / var[i])
        scores_per_act = np.array(scores_per_act)
        score = np.sum(scores_per_act)

        print(f"Score: {score}")

        if np.isnan(score) or np.isinf(score):
            return 1000000
        else:
            return score


def score_value_dynamic(self, x_g, x_p, ID_g=0, ID_p=None, ID_p_g=0, pred_probs=[]):
    if np.array_equal(x_g, x_p):
        return 0

    # Ensure the input arrays are 2D
    x_g_2d = x_g.reshape(x_g.shape[0], -1)
    x_p_2d = x_p.reshape(x_p.shape[0], -1)

    # Calculate DTW distance for positions
    dist_pos, _ = fastdtw(x_g_2d, x_p_2d, dist=euclidean)

    # Calculate velocities
    vel_g = np.diff(x_g, axis=0)
    vel_p = np.diff(x_p, axis=0)

    # Reshape velocities to 2D
    vel_g_2d = vel_g.reshape(vel_g.shape[0], -1)
    vel_p_2d = vel_p.reshape(vel_p.shape[0], -1)

    # Calculate DTW distance for velocities
    dist_vel, _ = fastdtw(vel_g_2d, vel_p_2d, dist=euclidean)

    # Combine distances
    dist = dist_pos + dist_vel

    if ID_p is not None:
        if ID_p_g == ID_p and ID_p_g != ID_g:
            return 0
        elif (ID_p_g == ID_p and ID_p_g == ID_g) or (ID_p_g != ID_p and ID_p_g == ID_g):
            return dist
        elif ID_p_g != ID_p and ID_p_g != ID_g and ID_p == ID_g:
            return dist  
        else:
            return 0
    else:
        if ID_g == ID_p_g:
            return dist
        else:
            return 0

def score_f1_dist_msad(self, sample_len, true_sequences, pred_sequences, true_labels, pred_labels,
                        distance_metric='frechet'):
    def calculate_distance(s1, s2):
        if distance_metric == 'dtw':
            distance, _ = fastdtw(s1, s2, dist=euclidean)
            return distance
        elif distance_metric == 'frechet':
            return frdist(s1, s2)
        elif distance_metric == 'dtw_mse':
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
            raise ValueError("Invalid distance metric. Choose 'dtw', 'frechet', 'dtw_mse', or 'mse'.")

    # Calculate F1 score
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    if f1 == 0:
        print('Model failed to classify any sequence correctly.')
        return {'f1': 0, 'avg_norm_distance': 1000000, 'avg_norm_msad': 1000000}

    # Initialize variables for metrics
    total_norm_distance = 0
    total_norm_msad = 0
    total_correct_sequences = 0

    unique_classes = np.unique(true_labels)
    num_classes = len(unique_classes)
    for c in unique_classes:
        class_mask = (true_labels == c) & (pred_labels == c)
        class_pred_seq = [pred_sequences[i] for i in range(len(pred_sequences)) if class_mask[i]]
        class_true_seq = [true_sequences[i] for i in range(len(true_sequences)) if class_mask[i]]
        class_mask_true = (true_labels == c)
        class_ground_seq = [true_sequences[i] for i in range(len(true_sequences)) if class_mask_true[i]]
        if len(class_pred_seq) == 0:
            continue  # No correctly classified sequences for this class

        # Compute max distance among all true sequences of the class
        if len(class_ground_seq) < 2:
            max_distance = max_msad = 1  # Default values if not enough sequences
        else:
            max_distance = max(calculate_distance(s[sample_len:, :], t[sample_len:, :])
                                for i, s in enumerate(class_ground_seq) for t in class_ground_seq[i + 1:])
            max_msad = max(self.calculate_msad(s[sample_len:, :], t[sample_len:, :])
                            for i, s in enumerate(class_ground_seq) for t in class_ground_seq[i + 1:])


        class_sum_distance = sum(calculate_distance(p[sample_len:, :], t[sample_len:, :]) / max_distance
                                    for p, t in zip(class_pred_seq, class_true_seq))
        total_norm_distance += class_sum_distance

        class_sum_msad = sum(self.calculate_msad(p[sample_len:, :], t[sample_len:, :]) / max_msad
                                for p, t in zip(class_pred_seq, class_true_seq))
        total_norm_msad += class_sum_msad

        total_correct_sequences += len(class_pred_seq)

    if total_correct_sequences == 0:
        return {'f1': f1, 'avg_norm_distance': 1000000, 'avg_norm_msad': 1000000}

    avg_norm_distance = total_norm_distance / total_correct_sequences / num_classes
    avg_norm_msad = total_norm_msad / total_correct_sequences / num_classes

    return {'f1': f1, 'avg_norm_distance': avg_norm_distance, 'avg_norm_msad': avg_norm_msad}

def calculate_msad(self, pred_sequence, true_sequence):
    # Calculate acceleration for both sequences
    pred_acc = np.diff(pred_sequence, n=2, axis=0)
    true_acc = np.diff(true_sequence, n=2, axis=0)

    # Calculate Mean Squared Acceleration Difference
    msad = np.mean(np.square(pred_acc - true_acc))

    return msad