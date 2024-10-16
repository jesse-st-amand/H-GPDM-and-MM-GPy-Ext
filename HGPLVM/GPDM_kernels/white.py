import numpy as np
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this
from HGPLVM.GPDM_kernels.static import Static

class White(Static):
    def __init__(self, input_dim, seq_eps, variance=1., active_dims=None, name='white'):
        super(White, self).__init__(input_dim, seq_eps, variance, active_dims, name)

    def to_dict(self):
        input_dict = super(White, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.White"
        return input_dict

    def K(self, X, X2=[None, None]):
        Xt, Xt1 = X
        X2t, X2t1 = X2
        if X2t is None and X2t1 is None:
            return np.eye(Xt.shape[0]) * self.variance
        else:
            return np.zeros((Xt.shape[0], X2t.shape[0]))

    def psi2(self, Z, variational_posterior):
        return np.zeros((Z.shape[0], Z.shape[0]), dtype=np.float64)

    def psi2n(self, Z, variational_posterior):
        return np.zeros((1, Z.shape[0], Z.shape[0]), dtype=np.float64)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            self.variance.gradient = np.trace(dL_dK)
        else:
            self.variance.gradient = 0.

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = dL_dKdiag.sum()

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        self.variance.gradient = dL_dpsi0.sum()

