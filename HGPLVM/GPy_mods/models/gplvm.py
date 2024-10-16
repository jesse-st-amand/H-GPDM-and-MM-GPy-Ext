# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from GPy import kern
from GPy.core import GP, Param
from GPy.likelihoods import Gaussian
from ..inference.latent_function_inference import exact_gaussian_inference

class GPLVM(GP):
    """
    Gaussian Process Latent Variable Model


    """
    def __init__(self, Y, input_dim, init='PCA', X=None, kernel=None, likelihood=None, name="gplvm",variance=1):

        """
        :param Y: observed data
        :type Y: np.ndarray
        :param input_dim: latent dimensionality
        :type input_dim: int
        :param init: initialisation method for the latent space
        :type init: 'PCA'|'random'
        """
        if X is None:
            from GPy.util.initialization import initialize_latent
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)
        if kernel is None:
            kernel = kern.RBF(input_dim, lengthscale=fracs, ARD=input_dim > 1) + kern.Bias(input_dim, np.exp(-2))
        if likelihood is None:
            likelihood = Gaussian(variance=variance)

        super(GPLVM, self).__init__(X, Y, kernel, likelihood,inference_method=exact_gaussian_inference.ExactGaussianInference(), name='GPLVM')

        self.X = Param('latent_mean', X)
        self.link_parameter(self.X, index=0)

    def parameters_changed(self):
        super(GPLVM, self).parameters_changed()
        #self.parameters[3]
        self.X.gradient = self.kern.gradients_X(self.grad_dict['dL_dK'], self.X, None)

    def objective_function(self):
        """
        The objective function for the given algorithm.

        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the objective function here.

        For probabilistic models this is the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your objective to minimize here!
        """
        obj_fun = -float(self.log_likelihood()) - self.log_prior()
        print(obj_fun)
        return obj_fun

    '''def get_pred_var(self):
        return self.X

    def update_pred_var(self,f):
        self.X = Param('latent_mean', f)'''