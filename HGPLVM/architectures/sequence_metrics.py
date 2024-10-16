import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial import procrustes


def euclidean_distance(seq1, seq2):
    return np.sqrt(np.sum((np.array(seq1) - np.array(seq2))**2, axis=1)).sum()


def pearson_correlation(seq1, seq2):
    return pearsonr(np.ravel(seq1), np.ravel(seq2))[0]

def cosine_similarity_metric(seq1, seq2):
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    return cosine_similarity(seq1.reshape(1, -1), seq2.reshape(1, -1))[0][0]

def procrustes_analysis(seq1, seq2):
    mtx1, mtx2, disparity = procrustes(np.array(seq1), np.array(seq2))
    return disparity

# This is a placeholder. HMM needs to be specifically trained for your data.
def hidden_markov_model(seq1, seq2, model):
    log_likelihood_seq1 = model.score(seq1)
    log_likelihood_seq2 = model.score(seq2)
    return abs(log_likelihood_seq1 - log_likelihood_seq2)


def edit_distance_real(seq1, seq2, eps=1.0):
    m, n = len(seq1), len(seq2)
    # Create a 2D array to store results of subproblems
    dp = np.zeros((m + 1, n + 1))

    # Fill dp[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # If first sequence is empty
            elif j == 0:
                dp[i][j] = i  # If second sequence is empty
            elif np.linalg.norm(np.array(seq1[i - 1]) - np.array(seq2[j - 1])) <= eps:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # Remove
                                   dp[i][j - 1])  # Insert
    return dp[m][n]


def longest_common_subsequence(seq1, seq2, eps=1.0):
    m, n = len(seq1), len(seq2)
    L = np.zeros((m + 1, n + 1))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if np.linalg.norm(np.array(seq1[i - 1]) - np.array(seq2[j - 1])) < eps:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L[m][n]

def calculate_frechet_distance(P, Q, i, j, dp):
    if dp[i][j] > -1:
        return dp[i][j]
    if i == 0 and j == 0:
        dp[i][j] = np.linalg.norm(np.array(P[0]) - np.array(Q[0]))
    elif i > 0 and j == 0:
        dp[i][j] = max(calculate_frechet_distance(P, Q, i-1, 0, dp), np.linalg.norm(np.array(P[i]) - np.array(Q[0])))
    elif i == 0 and j > 0:
        dp[i][j] = max(calculate_frechet_distance(P, Q, 0, j-1, dp), np.linalg.norm(np.array(P[0]) - np.array(Q[j])))
    elif i > 0 and j > 0:
        dp[i][j] = max(min(calculate_frechet_distance(P, Q, i-1, j, dp),
                           calculate_frechet_distance(P, Q, i-1, j-1, dp),
                           calculate_frechet_distance(P, Q, i, j-1, dp)),
                       np.linalg.norm(np.array(P[i]) - np.array(Q[j])))
    else:
        dp[i][j] = float('inf')
    return dp[i][j]

def frechet_distance(P, Q):
    m, n = len(P), len(Q)
    dp = np.full((m, n), -1.0)
    return calculate_frechet_distance(P, Q, m-1, n-1, dp)



