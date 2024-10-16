import numpy as np
from GPy.core.parameterization import Parameterized
from GPy.core.parameterization.priors import Prior
from GPy.util.linalg import jitchol, symmetrify, dpotri
from GPy.util.linalg import pdinv, dpotrs
from paramz.domains import _REAL
import GPy.kern as kern
import GPy

class GPDM_prior_base():
    def __init__(self):
        self.gradients = np.array([])

    def update_gradients(self,X):
        pass

    def lnpdf(self, xi):
        raise NotImplementedError

    def pdf(self, xi): # Calculated via logscale to circumvent some numerical errors
        return np.exp(self.lnpdf(xi))

    def dL_dX(self, xi):
        raise NotImplementedError

    def update_parameters(self, X0s, X):
        raise NotImplementedError

    def inverse(self,A):
        L = jitchol(A)
        Ai, _ = dpotri(L, lower=1)
        symmetrify(Ai)
        return Ai

class GPDM_IG_prior(GPDM_prior_base): # IG - Isotropic Gaussian
    def __init__(self, sigma, D):
        super(GPDM_IG_prior, self).__init__()
        self.sigma = sigma
        self.sigma_I = sigma * np.identity(D)
        self.sigma_inv = pdinv(self.sigma_I)[0]
        x0_logdet = np.linalg.det(self.sigma_inv)
        self.x0_const = -.5  * (D*np.log(2 * np.pi) + x0_logdet)

    def lnpdf(self, xi):
        if isinstance(xi,list):
            lnpdf_x = 0
            for x in xi:
                lnpdf_x += - .5 * x @ self.sigma_inv @ x.T + self.x0_const
        else:
            lnpdf_x =  - .5 * (xi @ self.sigma_inv @ xi.T)[0][0] + self.x0_const
        return lnpdf_x


    def dL_dX(self, xi):
        return -xi / self.sigma

    def update_parameters(self, X):
        pass

class GPDM_kernel_prior(Parameterized,GPDM_prior_base):  # IG - Isotropic Gaussian
    def __init__(self, X, D, kern=None):
        super(GPDM_kernel_prior, self).__init__()
        self.D = D
        if kern is None:
            self.kern =  GPy.kern.Linear(self.D, ARD=True)  + GPy.kern.RBF(self.D, 1, ARD=True)
        else:
            self.kern = kern
        #self.link_parameter(self.kern)
        #self.update_parameters(X)

    def lnpdf(self, xi):
        L_KX = jitchol(self.K_X)
        logdet_K_X = 2. * np.sum(np.diag(L_KX))

        X_N1 = np.concatenate([xi.reshape(1, -1), self.X])
        K_XN1 = self.kern.K(X_N1)

        L_KXN1 = jitchol(K_XN1)
        logdet_K_XN1 = 2. * np.sum(np.diag(L_KXN1))

        K_XN1_inv = self.inverse(K_XN1)

        return .5 * (logdet_K_X - logdet_K_XN1) + np.trace(X_N1.T @ K_XN1_inv @ X_N1 - self.X.T @ self.K_X_inv @ self.X)

    def update_parameters(self, X):
        self.X = X

        self.K_X = self.kern.K(self.X)
        self.K_X_inv = self.inverse(self.K_X)
        self.KXXK_prior = .5 * self.K_X_inv @ self.X @ self.X.T @ self.K_X_inv

        """X_N1 = np.concatenate([np.vstack(X0s), self.X])
        K_XN1 = self.kern.K(X_N1)

        K_XN1_inv = self.inverse(K_XN1)

        self.KXXK_N1 = .5 * K_XN1_inv @ X_N1 @ X_N1.T @ K_XN1_inv

        self.KXXK_prior = .5 * self.K_X_inv @ self.X @ self.X.T @ self.K_X_inv

        self.dL_dK = (-.5 * self.D * self.K_X_inv + .5 * self.D * K_XN1_inv + self.KXXK_prior - self.KXXK_N1)"""

        self.dL_dK = (-.5 * self.D * self.K_X_inv + self.KXXK_prior)

        self.kern.update_gradients_full(self.dL_dK, self.X)

        self.gradients = np.array([])#self.kern.gradient#np.zeros(self.kern.gradient.shape)#self.kern.gradient

    def dL_dX(self, xi):
        return 0#np.array([])#self.kern.gradients_X(self.dL_dK, self.X, None)

    def update_gradients(self,X):


        #self.KXXK_prior = .5 * self.K_X_inv @ self.X @ self.X.T @ self.K_X_inv
        #self.dL_dK = (-.5 * self.D * self.K_X_inv + self.KXXK_prior)
        #self.kern.update_gradients_full(self.dL_dK, self.X)
        self.gradients = np.array([])#self.kern.gradient

    """def update_gradients(self,X):
        self.KXXK_prior = .5 * self.K_X_inv @ self.X @ self.X.T @ self.K_X_inv
        self.dL_dK = (-.5 * self.D * self.K_X_inv + self.KXXK_prior)
        self.kern.update_gradients_full(self.dL_dK, self.X)
        self.gradients = self.kern.gradient"""

