# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import sys
from .sparse_gp_regression import SparseGPRegression
from GPy.core import Param
import numpy as np
def fourier_basis(N, D, per_seq = False):
    """
        Generate a Fourier basis set.

        Parameters:
        - D (int): Total number of waves (including the constant vector if odd).
        - N (int): Number of points to generate for each wave.

        Returns:
        - fourier_basis (numpy array): An array where each column is a basis function.
        """
    # Time array from 0 to 2pi
    t = np.linspace(0, 2 * np.pi, N)

    # Initialize the basis array
    if D % 2 == 1:
        # Include the constant vector if D is odd
        fourier_basis = np.ones((N, D))
        num_harmonics = (D - 1) // 2
    else:
        # Exclude the constant vector if D is even
        fourier_basis = np.zeros((N, D))
        num_harmonics = D // 2

    # Fill the basis with sine and cosine functions
    for i in range(1, num_harmonics + 1):
        print(i)
        sine_index = 2 * i - 2 if D % 2 == 0 else 2 * i - 1
        cosine_index = 2 * i - 1 if D % 2 == 0 else 2 * i
        fourier_basis[:, sine_index] = np.sin(i * t)  # Sine functions
        fourier_basis[:, cosine_index] = np.cos(i * t)  # Cosine functions

    return fourier_basis

class SparseGPLVM(SparseGPRegression):
    """
    Sparse Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, X=None, kernel=None, Z=None, init='PCA', num_inducing=10):
        if X is None:
            from GPy.util.initialization import initialize_latent
            X, fracs = initialize_latent(init, input_dim, Y)
        X = Param('latent space', X)
        if Z is None:
            i = np.arange(0, Y.shape[0], int(Y.shape[0] / num_inducing))
            Z = X.view(np.ndarray)[i].copy()
        SparseGPRegression.__init__(self, X, Y, Z=Z, kernel=kernel, num_inducing=num_inducing)
        self.link_parameter(self.X, 0)

    def parameters_changed(self):
        super(SparseGPLVM, self).parameters_changed()
        self.X.gradient = self.kern.gradients_X_diag(self.grad_dict['dL_dKdiag'], self.X)
        self.X.gradient += self.kern.gradients_X(self.grad_dict['dL_dKnm'], self.X, self.Z)

    def plot_latent(self, labels=None, which_indices=None,
                resolution=50, ax=None, marker='o', s=40,
                fignum=None, plot_inducing=True, legend=True,
                plot_limits=None,
                aspect='auto', updates=False, predict_kwargs={}, imshow_kwargs={}):
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_latent(self, labels, which_indices,
                resolution, ax, marker, s,
                fignum, plot_inducing, legend,
                plot_limits, aspect, updates, predict_kwargs, imshow_kwargs)

    def get_pred_var(self):
        return self.Z

    def update_pred_var(self,f):
        self.Z = Param('inducing_inputs', f)

    def objective_function(self):
        obj = super().objective_function()
        print(obj)
        return obj

