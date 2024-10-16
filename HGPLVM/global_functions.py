import numpy as np
from GPy.util.linalg import jitchol, symmetrify, dpotri
import time
import sys

def mat2vec(X,D_t):
    X_new = []
    for vec in X:
        X_new.append(vec.reshape(1, -1))
    return X.flatten()#np.hstack(X_new)

def vec2mat(X,D,N_t):
    X_new = []
    D_t = int(D/N_t)
    for n in range(N_t):
        vec = X[0, D_t * n:D_t * (n + 1)]
        X_new.append(vec.reshape(1, -1))
    #print('done')
    return np.vstack(X_new)

def vectorize(X,N_t=1):
    '''
    Matrix must either be NxD with rows representing vectors or 1xN*D
    :param X:
    :param D_t:
    '''
    N,D = X.shape
    if N==N_t:
        raise ValueError('Target dimension is the same as the input dimension.')
    if N_t == 1:
        return X.flatten().reshape(1,-1)#mat2vec(X,N_t)
    elif N == 1:
        return vec2mat(X,D,N_t).T
    else:
        raise ValueError('Chosen dimensions unsuitable for this function')

def inverse(A):
    L = jitchol(A)
    Ai, _ = dpotri(L, lower=1)
    symmetrify(Ai)
    return Ai

def pseudo_inv(X1):
    X1_X1_inv = inverse(X1.T @ X1)
    return (X1_X1_inv @ X1.T)

class race():
    def __init__(self, name,count=0, finish_condition=None, terminate=False):
        self.name = name
        self.laps = []
        self.start = time.time()
        self.fc = finish_condition
        self.term = terminate
        self.count = count

    def lap(self, count):
        self.count = count
        print(self.time_elapsed())
        if self.fc is None:
            self.laps.append(self.time_elapsed())
        elif self.count < self.fc:
            self.laps.append(self.time_elapsed())
        elif self.count == self.fc:
            self.finish()
    def time_elapsed(self):
        return time.time() - self.start

    def finish(self):
        if self.term:
            print(self.laps)
            print('Race '+self.name+' finished. Program terminated by request.')
            raise NotImplementedError
        else:
            print('Race '+self.name+' finished.')
class bit_racer():
    def __init__(self):
        self.universal_time = time.time()
        self.races = {}
    def global_cp(self,stopwatch): # cp - checkpoint
        print(time.time()-self.universal_time)
    def race_cp(self,name,text=None,count=0,finish_condition=None, terminate=False):
        if text is not None:
            print(text)
        if not name in self.races.keys():
            print('0')
            self.races[name] = race(name,count,finish_condition=finish_condition,terminate=terminate)
        else:
            self.races[name].lap(count)


import numpy as np


def block_shift_operator(N, T):
    """
    Create a block shift operator matrix.

    Parameters:
    N (int): Number of sequences
    T (int): Length of each sequence

    Returns:
    numpy.ndarray: Block shift operator matrix of shape (N*T, N*T)
    """
    # Create the basic shift operator for a single sequence
    S = np.eye(T, k=1)

    # Create the block diagonal matrix
    S_b = np.kron(np.eye(N), S)

    return S_b


'''# Example usage
N = 3  # Number of sequences
T = 4  # Length of each sequence

S_b = block_shift_operator(N, T)
print(S_b)

# Test the operator
X = np.arange(N * T).reshape(N * T, 1)
print("\nOriginal X:")
print(X)

print("\nAfter left multiplication (S_b @ X):")
print(S_b @ X)

print("\nAfter right multiplication (X.T @ S_b.T):")
print((X.T @ S_b.T).T)'''