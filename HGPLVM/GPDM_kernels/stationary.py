# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import integrate
from HGPLVM.GPDM_kernels.kern_2nd_order import Kern
from GPy.core.parameterization import Param
from GPy.util.linalg import tdot
from GPy import util
from GPy.util.config import config  # for assesing whether to use cython
from paramz.caching import Cache_this
from paramz.transformations import Logexp

from GPy.kern.src import stationary_cython

'''try:
    from GPy.kern.src import stationary_cython

    use_stationary_cython = config.getboolean('cython', 'working')
except ImportError:
    print('warning in stationary: failed to import cython module: falling back to numpy')'''
use_stationary_cython = False


class Stationary(Kern):
    """
    Stationary kernels (covariance functions).

    Stationary covariance fucntion depend only on r, where r is defined as

    .. math::
        r(x, x') = \\sqrt{ \\sum_{q=1}^Q (x_q - x'_q)^2 }

    The covariance function k(x, x' can then be written k(r).

    In this implementation, r is scaled by the lengthscales parameter(s):

    .. math::

        r(x, x') = \\sqrt{ \\sum_{q=1}^Q \\frac{(x_q - x'_q)^2}{\ell_q^2} }.

    By default, there's only one lengthscale: seaprate lengthscales for each
    dimension can be enables by setting ARD=True.

    To implement a stationary covariance function using this class, one need
    only define the covariance function k(r), and it derivative.

    ```
    def K_of_r(self, r):
        return foo
    def dK_dr(self, r):
        return bar
    ```

    The lengthscale(s) and variance parameters are added to the structure automatically.

    Thanks to @strongh:
    In Stationary, a covariance function is defined in GPy as stationary when it depends only on the l2-norm |x_1 - x_2 |.
    However this is the typical definition of isotropy, while stationarity is usually a bit more relaxed.
    The more common version of stationarity is that the covariance is a function of x_1 - x_2 (See e.g. R&W first paragraph of section 4.1).
    """

    def __init__(self, input_dim, seq_eps, variance, lengthscale, ARD, active_dims, name, useGPU=False):
        super(Stationary, self).__init__(input_dim, seq_eps, active_dims, name, useGPU=useGPU)
        self.ARD = ARD
        if not ARD:
            if lengthscale is None:
                lengthscale = np.ones(1)
            else:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only 1 lengthscale needed for non-ARD kernel"
        else:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size in [1, input_dim], "Bad number of lengthscales"
                if lengthscale.size != input_dim:
                    lengthscale = np.ones(input_dim) * lengthscale
            else:
                lengthscale = np.ones(self.input_dim)
        self.lengthscale = Param('lengthscale', lengthscale, Logexp())
        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size == 1
        self.link_parameters(self.variance, self.lengthscale)

    def _save_to_input_dict(self):
        input_dict = super(Stationary, self)._save_to_input_dict()
        input_dict["variance"] = self.variance.values.tolist()
        input_dict["lengthscale"] = self.lengthscale.values.tolist()
        input_dict["ARD"] = self.ARD
        return input_dict

    def K_of_r(self, r):
        raise NotImplementedError("implement the covariance function as a fn of r to use this class")

    def dK_dr(self, r):
        raise NotImplementedError("implement derivative of the covariance function wrt r to use this class")

    @Cache_this(limit=3, ignore_args=())
    def dK2_drdr(self, r):
        raise NotImplementedError("implement second derivative of covariance wrt r to use this method")

    @Cache_this(limit=3, ignore_args=())
    def dK2_drdr_diag(self):
        "Second order derivative of K in r_{i,i}. The diagonal entries are always zero, so we do not give it here."
        raise NotImplementedError("implement second derivative of covariance wrt r_diag to use this method")

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, Xt1,  X2=None, X2t1=None):
        """
        Kernel function applied on inputs X and X2.
        In the stationary case there is an inner function depending on the
        distances from X to X2, called r.

        K(X, Xt1, X2) = K_of_r((X-X2)**2)
        """
        r = self._scaled_dist(X, Xt1, X2)
        return self.K_of_r(r)

    @Cache_this(limit=3, ignore_args=())
    def dK_dr_via_X(self, X, Xt1, X2):
        """
        compute the derivative of K wrt X going through X
        """
        # a convenience function, so we can cache dK_dr
        return self.dK_dr(self._scaled_dist(X, Xt1, X2))

    @Cache_this(limit=3, ignore_args=())
    def dK2_drdr_via_X(self, X, Xt1, X2):
        # a convenience function, so we can cache dK_dr
        return self.dK2_drdr(self._scaled_dist(X, Xt1, X2))


    @Cache_this(limit=3, ignore_args=())
    def _scaled_dist(self, X, Xt1, X2=None, X2t1=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

        Note that if thre is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards

        """
        if self.ARD:
            if X2 is not None:
                X2 = X2 / self.lengthscale
            return self._unscaled_dist(X / self.lengthscale, X2)
        else:
            return self._unscaled_dist(X, X2) / self.lengthscale

    def Kdiag(self, X, X2=[None, None]):
        ret = np.empty((X[0].shape[0],), dtype=np.float64)
        ret[:] = self.variance
        return ret

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X[0].shape)

    def reset_gradients(self):
        self.variance.gradient = 0.
        if not self.ARD:
            self.lengthscale.gradient = 0.
        else:
            self.lengthscale.gradient = np.zeros(self.input_dim)

    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Given the derivative of the objective with respect to the diagonal of
        the covariance matrix, compute the derivative wrt the parameters of
        this kernel and stor in the <parameter>.gradient field.

        See also update_gradients_full
        """
        self.variance.gradient = np.sum(dL_dKdiag)
        self.lengthscale.gradient = 0.
        self.lengthscale_t1.gradient = 0.

    def update_gradients_full(self, dL_dK, X, Xt1, X2=None, X2t1=None, reset=True):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        raise NotImplementedError

    def update_gradients_direct(self, dL_dVar, dL_dLen):
        """
        Specially intended for the Grid regression case.
        Given the computed log likelihood derivates, update the corresponding
        kernel and likelihood gradients.
        Useful for when gradients have been computed a priori.
        """
        self.variance.gradient = dL_dVar
        self.lengthscale.gradient = dL_dLen

    def _inv_dist(self, X, Xt1, X2=None, X2t1=None):
        """
        Compute the elementwise inverse of the distance matrix, expecpt on the
        diagonal, where we return zero (the distance on the diagonal is zero).
        This term appears in derviatives.
        """
        dist = self._scaled_dist(X, Xt1, X2).copy()
        return 1. / np.where(dist != 0., dist, np.inf)

    def _lengthscale_grads_pure(self, tmp, X, X2):
        return -np.array([np.sum(tmp * np.square(X[:, q:q + 1] - X2[:, q:q + 1].T)) for q in
                          range(self.input_dim)]) / self.lengthscale ** 3

    def _lengthscale_grads_cython(self, tmp, X, X2):
        N, M = tmp.shape
        Q = self.input_dim
        X, X2 = np.ascontiguousarray(X), np.ascontiguousarray(X2)
        grads = np.zeros(self.input_dim)
        stationary_cython.lengthscale_grads(N, M, Q, tmp, X, X2, grads)
        return -grads / self.lengthscale ** 3

