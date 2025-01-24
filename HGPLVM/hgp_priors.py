import logging
from GPy.core import Param
from GPy.core.parameterization import Parameterized
from GPy.core.parameterization.priors import Prior
from GPy.util.linalg import jitchol, symmetrify, dpotri
from GPy.util.linalg import pdinv, dpotrs
from paramz.domains import _REAL
from HGPLVM.GPDM_kernels.rbf_acc import RBF_Acc
from HGPLVM.GPDM_kernels.linear_acc import Linear_Acc
from HGPLVM.GPDM_kernels.white import White
from HGPLVM.GPDM_priors import GPDM_IG_prior
import GPy
from GPy.util.linalg import jitchol, tdot, dtrtrs, dpotri, pdinv
from GPy.util import diag
import numpy as np
log_2_pi = np.log(2*np.pi)
from scipy.special import logsumexp
logger = logging.getLogger("GP")
from functools import wraps




class MultivariateGaussian(Prior):
    """
    Implementation of the multivariate Gaussian probability function, coupled with random variables.

    :param mu: mean (N-dimensional array)
    :param var: covariance matrix (NxN)

    """
    domain = _REAL
    _instances = []

    def __init__(self, mu, var):
        self.mu = np.array(mu) # added
        self.var = np.array(var)
        assert len(self.var.shape) == 2, 'Covariance must be a matrix'
        assert self.var.shape[0] == self.var.shape[1], \
            'Covariance must be a square matrix'
        self.N, self.D = self.mu.shape
        self.inv, self.LW, self.hld, self.logdet = pdinv(self.var)
        self.constant = -0.5 * self.D * (self.N * np.log(2 * np.pi) + self.logdet)

    def __str__(self):
        return 'MultiN(' + str(self.mu[0,0]) + ', ' + str(self.var[0,0]) + ')'

    def expand(self, x):
        return x.reshape([self.N,self.D])

    def summary(self):
        raise NotImplementedError

    def pdf(self, x):
        x = self.expand(np.array(x))
        return np.exp(self.lnpdf(x)).flatten()

    def lnpdf(self, x):
        x = self.expand(np.array(x))
        d = x - self.mu
        alpha, _ = dpotrs(self.LW, d, lower=1)
        return (self.constant - 0.5 * np.sum(alpha * d)).flatten()

    def lnpdf_grad(self, x):
        x = self.expand(np.array(x))
        d = x - self.mu
        return (- np.dot(self.inv, d)).flatten()

    def rvs(self, n):
        return np.random.multivariate_normal(self.mu, self.var, n)

    def plot(self):
        import sys

        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from GPy.plotting.matplot_dep import priors_plots

        priors_plots.multivariate_plot(self)

def inverse(A):
    L = jitchol(A)
    Ai, _ = dpotri(L, lower=1)
    symmetrify(Ai)
    return Ai

class GPDM_base(Prior, Parameterized):
    """
    Gaussian Process Dynamical Model

    :param X: sequence data (NxD)
    :param kern: kernel
    :param seq_eps: List of int elements indicating the end points of each data sequence. Note: sequences are stacked
    end-to-end along dimension N

    Adapted from Wang et al. 2006 Gaussian Process Dynamical Models for Human Motion
    """
    domain = _REAL
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, X, x0_prior, seq_eps, sigma, GPNode=None, name='GPDM_prior'): # X should be a latent dimension Param
        super(GPDM_base, self).__init__(name=name)
        #self.GPNode = GPNode
        self.name = name
        self.sigma = sigma
        self.seq_eps = seq_eps
        self.gradients = None
        self.seq_x0s = self.get_seq_x0s()
        self.num_seqs = len(self.seq_x0s)
        self.seq_len = self.seq_eps[0]-self.seq_x0s[0]+1
        self.N, self.D = X.shape
        self.X = X
        self.Xout = None
        self.Xin = None
        self.Xout_0 = None
        self.K_Xin = None
        self.M_inv = None
        self.S_inv = None
        self.sliced_dims = None
        self.dims = None
        self.M_inv_slice = None
        self.S_inv_slice = None
        self.Xin_slice = None
        self.Xout_slice = None
        self.X0_slice = None
        self.dL_dthetaL = None
        self.kern = None
        self.x0_prior = x0_prior

    def block_shift_operator(self, seq_len, num_seqs, num_pos=1):
        """
        Create a block shift operator matrix.

        Parameters:
        num_pos (int): Number of positions to shift (default is 1)

        Returns:
        numpy.ndarray: Block shift operator matrix
        """
        S = np.eye(seq_len, k=num_pos)
        S_b = np.kron(np.eye(num_seqs), S)
        return S_b

    def link_kern(self):
        logger.info("adding kernel as parameter")
        self.link_parameter(self.kern)
        if hasattr(self.x0_prior, 'parameters'):
            self.link_params()
        #self.update_parameters(self.X)

    def link_params(self):
        for p in self.x0_prior.parameters:
            self.link_parameter(p)

    def get_seq_x0s(self):
        if len(self.seq_eps) == 1:
            return np.array([0])
        else:
            return np.concatenate([np.array([0]),(np.array(self.seq_eps) + 1)[:-1]])

    def update_parameters(self, X):
        self.X = X
        # Create subsets of X to make the first and last time points independent
        self.Xin, self.Xout, self.Xout_0 = self.partition_X_grads(X)
        if hasattr(self.x0_prior, 'parameters'):
            self.x0_prior.update_parameters(self.Xout_0)
        self.K_Xin = self.kern_K(self.Xin)
        self.LL_n_grads()

    def expand(self, X):
        return X.reshape([self.N, self.D])

    def pdf(self, X):
        X = self.expand(np.array(X))
        return np.exp(self.lnpdf(X)).flatten()

    def lnpdf(self, X):
        return self.LL.flatten()

    def lnpdf_grad(self, X):
        return self.dL_dX.flatten()

    def var_predict(self, Xstar_in):
        k_x_xstar = self.kern_K(self.Xin, Xstar_in)
        k_star_in = self.kern_K(Xstar_in)
        return k_star_in - k_x_xstar.T @ self.S_inv @ k_x_xstar

    def mn_predict(self, Xstar_in, *args):
        k_x_xstar = self.kern_K(self.Xin, Xstar_in)
        alpha = self.M_inv @ self.Xout
        return k_x_xstar.T @ alpha


    def var_predict_slice(self, Xstar_in):
        k_x_xstar = self.kern_K(self.Xin_kern_slice, Xstar_in)
        k_star_in = self.kern_K(Xstar_in)
        return k_star_in - k_x_xstar.T @ self.S_inv_slice @ k_x_xstar

    def mn_predict_slice(self, Xstar_in):
        k_x_xstar = self.kern_K(self.Xin_kern_slice, Xstar_in)
        alpha = self.M_inv_slice @ self.Xout_slice
        return k_x_xstar.T @ alpha

    def pred_lnpdf(self, X):
        X = self.expand(np.array(X))
        return self.LL.flatten()

    def pred_pdf_slice(self, Ustar):
        Xstar_in, Xstar_out, Xstar0s = self.partition_X_preds(Ustar)
        if self.sliced_dims is not None:
            Xstar_in = np.hstack([Xstar_in,np.zeros([Xstar_in.shape[0],len(self.sliced_dims)])])

        mn = self.mn_predict_slice(Xstar_in)
        k_star = self.var_predict_slice(Xstar_in)

        k_star_inv = pdinv(k_star)[0]
        Z = Xstar_out - mn
        ln_px0s = 0#self.x0_prior.lnpdf(Xstar0s).flatten()
        ln_pi_const = -.5 * self.D * (Xstar_out.shape[0]) * np.log(2 * np.pi)
        ln_det = -.5 * self.D * np.log(np.linalg.det(k_star))
        tr_ZKZ = -.5 * np.trace(Z.T @ k_star_inv @ Z)
        return np.exp(ln_pi_const + ln_det + tr_ZKZ)

    def pred_pdf(self, Ustar):
        M, _ = Ustar.shape
        Xstar_in, Xstar_out, X0s = self.partition_X_preds(Ustar)
        mn = self.mn_predict(Xstar_in)
        k_star = self.var_predict(Xstar_in)
        k_star_inv = pdinv(k_star)[0]
        Z = Xstar_out - mn
        ln_px0s = 0#self.x0_prior.lnpdf(X0s)
        ln_pi_det = -.5 * self.D * np.log((2*np.pi)**(M-1)*np.linalg.det(k_star))
        tr_ZKZ = -.5 * np.trace(Z.T @ k_star_inv @ Z)
        return np.exp(ln_px0s + ln_pi_det + tr_ZKZ)

    def ln_pred_pdf(self, Ustar):
        M, _ = Ustar.shape
        Xstar_in, Xstar_out, X0s = self.partition_X_preds(Ustar)
        mn = self.mn_predict(Xstar_in)
        k_star = self.var_predict(Xstar_in)
        k_star_inv = pdinv(k_star)[0]
        Z = Xstar_out - mn
        ln_px0s = 0#self.x0_prior.lnpdf(X0s)
        ln_pi_det = -.5 * self.D * np.log((2*np.pi)**(M-1)*np.linalg.det(k_star))
        tr_ZKZ = -.5 * np.trace(Z.T @ k_star_inv @ Z)
        return ln_px0s + ln_pi_det + tr_ZKZ

    def LL_comparison(self, X1, X2):
        LL1 = self.ln_pred_pdf(X1)
        LL2 = self.ln_pred_pdf(X2)
        return np.abs(LL1 - LL2)

    def lnpdf_comparison(self, X):
        return self.ln_pred_pdf(X)

    def predict_best_z(self, X_0):
        return 0, [1,0]

    def update_gradients(self):
        raise NotImplementedError

    def LL_n_grads(self):
        raise NotImplementedError

    def partition_X_grads(self):
        raise NotImplementedError

    def partition_X_preds(self):
        raise NotImplementedError

    def partition_dL_Xs(self):
        raise NotImplementedError

class GPDM_1st_order(GPDM_base):
    """
    Gaussian Process Dynamical Model

    :param X: sequence data (NxD)
    :param kern: kernel
    :param seq_eps: List of int elements indicating the end points of each data sequence. Note: sequences are stacked
    end-to-end along dimension N

    Adapted from Wang et al. 2006 Gaussian Process Dynamical Models for Human Motion
    """
    domain = _REAL
    _instances = []

    def __init__(self, X, gpdm_var, x0_prior, seq_eps, sigma, GPNode=None, name='GPDM_prior'): # X should be a latent dimension Param
        super(GPDM_1st_order, self).__init__(X, x0_prior, seq_eps, sigma, GPNode=GPNode, name='GPDM_prior')
        self.GPDM_order = 1
        self.gpdm_timesteps = 1
        #self.kern = GPy.kern.White(self.D, variance=gpdm_var) + GPy.kern.Linear(self.D, variances=np.ones(self.GPNode.input_dim) * 1, ARD=True) + GPy.kern.RBF(self.D, variance=1, lengthscale=1, ARD=True)#
        self.kern = GPy.kern.White(self.D, variance=gpdm_var) + GPy.kern.Linear(self.D, variances=1, ARD=False) + GPy.kern.RBF(self.D, variance=1, lengthscale=1, ARD=False)
        self.link_kern()

    @staticmethod
    def kern_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result

        return wrapper

    @kern_wrapper
    def kern_K(self, X1, X2=None):
        return self.kern.K(X1, X2)

    @kern_wrapper
    def kern_Kdiag(self, X1, X2=[None, None]):
        return self.kern.Kdiag(X1)#, X2)

    @kern_wrapper
    def kern_X_grads(self, dL_dX, X1, X2=None):
        return self.kern.gradients_X(dL_dX, X1, X2)

    @kern_wrapper
    def kern_X_grads_diag(self, dL_dKdiag, X1):
        return self.kern.gradients_X_diag(dL_dKdiag, X1)

    @kern_wrapper
    def update_kern_param_grads(self, dL_dX, X1, X2=None):
        return self.kern.update_gradients_full(dL_dX, X1, X2)

    @kern_wrapper
    def update_kern_param_grads_diag(self, dL_dX, X1):
        return self.kern.update_gradients_diag(dL_dX, X1)

    def partition_X_preds(self, X):
        if X.shape[0] > 1:
            X0 = X[0,:].reshape(1,-1)
            Xt = X[:-1, :]
            Xt1 = X[1:, :]
        else:
            X0 = None
            Xt = X[0, :].reshape(1, -1)
            Xt1 = None
        return Xt, Xt1, X0

    def partition_X_grads(self,X):
        x0_list = []
        Xout_list = []
        Xin_list = []
        for x0_i, ep_i in zip(self.seq_x0s, self.seq_eps):
            x0_list.append(X[x0_i, :].reshape(1,-1))
            Xout_list.append(X[x0_i + 1:ep_i + 1, :])
            Xin_list.append(X[x0_i:ep_i, :])
        return np.concatenate(Xin_list), np.concatenate(Xout_list), x0_list

    def group_dL_Xs(self,X,dL_Xin_partition,dL_Xout_partition):
        dL_dXin = np.zeros([self.N, self.D])
        dL_dXout = np.zeros([self.N, self.D])
        dL_dX0 = np.zeros([self.N, self.D])
        for i, (x0_i, ep_i) in enumerate(zip(self.seq_x0s, self.seq_eps)):
            dL_dXin[x0_i:ep_i, :] = dL_Xin_partition[(x0_i-i):(ep_i-i), :]
            dL_dXout[x0_i + 1:ep_i + 1, :] = dL_Xout_partition[x0_i-i:ep_i-i, :]
            dL_dX0[x0_i, :] = self.x0_prior.dL_dX(X[x0_i, :])
        return dL_dX0 + dL_dXin + dL_dXout

    def partition_Zs(self, Z):
        return Z



class GPDM_2nd_order(GPDM_base):
    """
    Second Order Gaussian Process Dynamical Model

    :param X: sequence data (NxD)
    :param kern: kernel
    :param seq_eps: List of int elements indicating the end points of each data sequence. Note: sequences are stacked
    end-to-end along dimension N

    Adapted from Wang et al. 2006 Gaussian Process Dynamical Models for Human Motion
    """

    def __init__(self, X, gpdm_var, x0_prior, seq_eps, sigma, GPNode=None, name='GPDM2_prior'): # X should be a latent dimension Param
        super(GPDM_2nd_order, self).__init__(X, x0_prior, seq_eps, sigma, GPNode=GPNode, name='GPDM_prior')
        self.GPDM_order = 2
        self.gpdm_timesteps = 2
        self.kern = White(self.D, seq_eps, variance=gpdm_var) + Linear_Acc(self.D,seq_eps,variances1=1,variances2=1) + RBF_Acc(self.D, seq_eps, 1, ARD=False)#
        self.link_kern()

    def __str__(self):
        return 'GPDM2'

    @staticmethod
    def kern_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result

        return wrapper

    @kern_wrapper
    def kern_K(self, X1, X2=[None, None]):
        return self.kern.K(X1, X2)

    @kern_wrapper
    def kern_Kdiag(self, X1, X2=[None, None]):
        return self.kern.Kdiag(X1)

    @kern_wrapper
    def kern_X_grads(self, dL_dK, X1, X2=[None, None]):
        dL_dKin = self.partition_dL_dK_grads(dL_dK)
        return self.kern.gradients_X(dL_dKin, X1, X2)

    @kern_wrapper
    def kern_X_grads_diag(self, dL_dKdiag, X1):
        return self.kern.gradients_X_diag(dL_dKdiag, X1)

    @kern_wrapper
    def update_kern_param_grads(self, dL_dK, X1, X2=[None, None]):
        return self.kern.update_gradients_full(dL_dK, X1, X2)

    @kern_wrapper
    def update_kern_param_grads_diag(self, dL_dX, X1):
        return self.kern.update_gradients_diag(dL_dX, X1)

    def partition_X_preds(self, X):
        if X.shape[0] > 2:
            X0 = [X[0,:].reshape(1,-1),X[1,:].reshape(1,-1)]
            Xt = X[:-2,:]
            Xt1 = X[1:-1,:]
            Xt2 = X[2:, :]
        else:
            X0 = None
            Xt = X[0, :].reshape(1,-1)
            Xt1 = X[1, :].reshape(1,-1)
            Xt2 = None
        return [Xt,Xt1], Xt2, X0

    def partition_X_grads(self,X):
        x0_list = []
        Xt_list = []
        Xt1_list = []
        Xt2_list = []
        for x0_i, ep_i in zip(self.seq_x0s, self.seq_eps):
            x0_list.append([X[x0_i, :].reshape(1,-1),X[x0_i+1, :].reshape(1,-1)])
            Xt_list.append(X[x0_i:ep_i - 1, :])
            Xt1_list.append(X[x0_i + 1:ep_i, :])
            Xt2_list.append(X[x0_i + 2:ep_i + 1, :])
        return [np.concatenate(Xt_list),np.concatenate(Xt1_list)], np.concatenate(Xt2_list), x0_list

    def partition_dL_dK_grads(self,dL_dK):
        N,D = dL_dK.shape
        if N == self.N-self.num_seqs*2 and D == self.N-self.num_seqs*2:
            S = self.block_shift_operator(self.seq_len-2, self.num_seqs, 1)
            dL_dKt1 = S@dL_dK
        elif N == self.N_z or D == self.N_z:
            dL_dKt1 = dL_dK.copy()

        return dL_dK, dL_dKt1

    def group_dL_Xs(self,X,dL_Xin_partition,dL_Xout_partition):
        dL_dXout = np.zeros([self.N, self.D])
        dL_dX0 = np.zeros([self.N, self.D])
        dL_dXin = np.zeros([self.N, self.D])
        for i, (x0_i, ep_i) in enumerate(zip(self.seq_x0s, self.seq_eps)):
            dL_dXin[x0_i:ep_i-1, :] = dL_Xin_partition[(x0_i - 2*i):ep_i - 2*(i+1) + 1, :]
            dL_dXout[x0_i + 2:ep_i + 1, :] = dL_Xout_partition[(x0_i - 2*i):ep_i - 2*(i+1) + 1, :]
            dL_dX0[x0_i, :] = self.x0_prior.dL_dX(X[x0_i, :])
            dL_dX0[x0_i+1, :] = self.x0_prior.dL_dX(X[x0_i+1, :])
        return dL_dX0 + dL_dXin + dL_dXout


    def partition_Zs(self, Z):
        return [Z, Z]


def Full_GPDM_N_Order(GPDM_N_Order):
    class full_GPDM(GPDM_N_Order):
        """
        Gaussian Process Dynamical Model

        :param X: sequence data (NxD)
        :param kern: kernel
        :param seq_eps: List of int elements indicating the end points of each data sequence. Note: sequences are stacked
        end-to-end along dimension N

        Adapted from Wang et al. 2006 Gaussian Process Dynamical Models for Human Motion
        """
        def __init__(self, X, gpdm_var, x0_prior, seq_eps, sigma, GPNode=None,name='GPDM_prior'):  # X should be a latent dimension Param
            super().__init__(X, gpdm_var, x0_prior, seq_eps, sigma, GPNode=GPNode, name='GPDM_prior')
            self.gpdm_var = gpdm_var
            self.sigma = sigma
            self.name = name
            self.GPDM_class = "full"
            self.LL = 0
            self.dL_dX = np.zeros(X.shape)

        def __reduce__(self):
            return (GPDM_N_Order_func,
                    (self.GPDM_order,Full_GPDM_N_Order, self.X, self.gpdm_var, self.x0_prior, self.seq_eps, self.sigma, None, self.name))


        def LL_n_grads(self):
            self.LL_calc()
            self.dL_dX_calc()
            self.update_kern_param_grads(self.dL_dK, self.Xin)
            self.gradients = np.concatenate([self.kern.gradient, self.x0_prior.gradients])

        def LL_calc(self):
            ln_px0s = 0
            for x0 in self.Xout_0:
                ln_px0s += self.x0_prior.lnpdf(x0)
            self.M_inv = inverse(self.K_Xin)
            self.S_inv = self.M_inv.copy()
            L2 = jitchol(self.K_Xin)
            K_logdet = 2. * np.sum(np.log(np.diag(L2)))
            C_K_logdet = -.5 * self.D * ((self.N - 1) * np.log(2 * np.pi) + K_logdet)
            TrKXX = -.5 * np.trace(self.M_inv @ self.Xout @ self.Xout.T)
            self.LL = ln_px0s + C_K_logdet + TrKXX

        def dL_dX_calc(self):
            self.KXXK = .5 * self.M_inv @ self.Xout @ self.Xout.T @ self.M_inv
            self.dL_dK =  -.5 * self.D * self.M_inv + self.KXXK # for Xin
            temp_dL_dXin = self.kern_X_grads(self.dL_dK, self.Xin)
            temp_dL_dXout = -self.M_inv @ self.Xout
            self.dL_dX = self.group_dL_Xs(self.X, temp_dL_dXin, temp_dL_dXout)  # computes dL_dX0

        def S_and_M_inv_slice(self, dims, sliced_dims):
            self.dims = dims
            self.sliced_dims = sliced_dims
            self.Xin_kern_slice = self.Xin.copy() #sliced dims are replaced with zeros for use in kernels
            if sliced_dims is not None:
                self.Xin_kern_slice[:,sliced_dims] = 0
            self.Xin_slice = self.Xin[:,dims].copy()
            self.Xout_slice = self.Xout[:,dims].copy()
            #self.X0s_slice = self.Xout_0[:,dims]
            self.M_inv_slice = self.S_inv_slice = inverse(self.kern_K(self.Xin_kern_slice))


    return full_GPDM

def Sparse_GPDM_N_Order(GPDM_N_Order):
    class sparse_GPDM(GPDM_N_Order):
        """
        Gaussian Process Dynamical Model

        :param X: sequence data (NxD)
        :param kern: kernel
        :param seq_eps: List of int elements indicating the end points of each data sequence. Note: sequences are stacked
        end-to-end along dimension N

        Adapted from Wang et al. 2006 Gaussian Process Dynamical Models for Human Motion
        """


        def __init__(self, num_inducing, X, gpdm_var, x0_prior, seq_eps, sigma, Z=None, GPNode=None,
                     name='GPDM_prior'):  # X should be a latent dimension Param
            super(sparse_GPDM, self).__init__(X, gpdm_var, x0_prior, seq_eps, sigma, GPNode, name)
            self.num_inducing = num_inducing
            if Z is None:
                i = np.linspace(0,X.shape[0]-1,self.num_inducing,dtype=int)
                self.Z = X.view(np.ndarray)[i].copy()
            else:
                raise NotImplementedError("Handling of pre-initialized Z must be implemented")
            self.GPDM_class = "sparse"
            self.setup_Z()
            self.N_z, self.D_z = self.Z.shape
            self.gpdm_var = gpdm_var




        def __reduce__(self):
            return (GPDM_N_Order_func,
                    (self.GPDM_order,Sparse_GPDM_N_Order, self.X, self.num_inducing, self.gpdm_var, self.x0_prior, self.seq_eps, self.sigma, None, self.name))

        def setup_Z(self):
            self.Z = Param('inducing inputs', self.Z)
            logger.info("Adding Z as parameter")
            self.link_parameter(self.Z, index=0)

        def var_predict(self, Xstar_in):
            Zin = self.partition_Zs(self.Z)
            k_z_xstar = self.kern_K(Zin, Xstar_in)
            k_star_in = self.kern_K(Xstar_in)
            return k_star_in - k_z_xstar.T @ self.S_inv @ k_z_xstar

        def mn_predict(self, Xstar_in):
            Zin = self.partition_Zs(self.Z)
            k_z_xstar = self.kern_K(Zin, Xstar_in)
            #return k_z_xstar.T @ self.M_inv @ self.Knm.T @ self.Xout
            return k_z_xstar.T @ self.wv

        def LL_n_grads(self):
            self.dL_dX_calc()
            Zin = self.partition_Zs(self.Z)
            self.update_kern_param_grads_diag(self.grad_dict['dL_dKdiag'], self.Xin)
            kerngrad = self.kern.gradient.copy()
            self.update_kern_param_grads(self.grad_dict['dL_dKnm'], self.Xin, Zin)#########
            kerngrad += self.kern.gradient
            self.update_kern_param_grads(self.grad_dict['dL_dKmm'], Zin)############
            self.kern.gradient += kerngrad
            # gradients wrt Z
            self.Z.gradient = self.kern_X_grads(self.grad_dict['dL_dKmm'], Zin)
            self.Z.gradient += self.kern_X_grads(self.grad_dict['dL_dKnm'].T, Zin, self.Xin)

        def dL_dX_calc(self):
            self.dL_dK = self.dL_dK_calc()
            Zin = self.partition_Zs(self.Z)
            temp_dL_dXin = self.kern_X_grads_diag(self.grad_dict['dL_dKdiag'], self.Xin)
            temp_dL_dXin += self.kern_X_grads(self.grad_dict['dL_dKnm'], self.Xin, Zin)
            temp_dL_dXout = self.grad_dict['dL_dY']

            self.dL_dX = self.group_dL_Xs(self.X, temp_dL_dXin, temp_dL_dXout)

        def S_and_M_inv(self):
            raise NotImplementedError

        def dL_dK_calc(self):
            self.const_jitter = 1e-6
            Y = self.Xout
            X = self.Xin
            Z = self.Z
            num_inducing, _ = Z.shape
            num_data, output_dim = Y.shape

            # make sure the noise is not hetero
            sigma_n = 1
            Zin = self.partition_Zs(Z)
            Kmm = self.kern_K(Zin)
            Knn = self.kern_Kdiag(X)
            self.Knm = self.kern_K(X, Zin)
            U = self.Knm

            # factor Kmm
            diag.add(Kmm, self.const_jitter)
            Kmmi, L, Li, _ = pdinv(Kmm)

            # compute beta_star, the effective noise precision
            LiUT = np.dot(Li, U.T)
            sigma_star = Knn + sigma_n - np.sum(np.square(LiUT), 0)
            beta_star = 1. / sigma_star

            # Compute and factor A
            A = tdot(LiUT * np.sqrt(beta_star)) + np.eye(num_inducing)
            LA = jitchol(A)

            # back substutue to get b, P, v
            URiy = np.dot(U.T * beta_star, Y)
            tmp, _ = dtrtrs(L, URiy, lower=1)
            b, _ = dtrtrs(LA, tmp, lower=1)
            tmp, _ = dtrtrs(LA, b, lower=1, trans=1)
            v, _ = dtrtrs(L, tmp, lower=1, trans=1)
            tmp, _ = dtrtrs(LA, Li, lower=1, trans=0)
            P = tdot(tmp.T)

            # compute log marginal
            log_marginal = -0.5 * num_data * output_dim * np.log(2 * np.pi) + \
                           -np.sum(np.log(np.diag(LA))) * output_dim + \
                           0.5 * output_dim * np.sum(np.log(beta_star)) + \
                           -0.5 * np.sum(np.square(Y.T * np.sqrt(beta_star))) + \
                           0.5 * np.sum(np.square(b))

            for x0 in self.Xout_0:
                log_marginal += self.x0_prior.lnpdf(x0)

            self.LL = log_marginal

            # compute dL_dR
            Uv = np.dot(U, v)
            dL_dR = 0.5 * (np.sum(U * np.dot(U, P), 1) - 1. / beta_star + np.sum(np.square(Y), 1) - 2. * np.sum(Uv * Y,
                                                                                                                1) + np.sum(
                np.square(Uv), 1)) * beta_star ** 2

            # Compute dL_dKmm
            # vvT_P = tdot(v.reshape(-1,1)) + P # original GPy code
            vvT_P = tdot(v) + P
            dL_dK = 0.5 * (Kmmi - vvT_P)
            KiU = np.dot(Kmmi, U.T)
            dL_dK += np.dot(KiU * dL_dR, KiU.T)

            # Compute dL_dU
            # vY = np.dot(v.reshape(-1,1),Y.T) # original GPy code
            vY = np.dot(v, Y.T)
            dL_dU = vY - np.dot(vvT_P, U.T)
            dL_dU *= beta_star
            dL_dU -= 2. * KiU * dL_dR

            #dL_dthetaL = likelihood.exact_inference_gradients(dL_dR)
            #'dL_dthetaL': dL_dthetaL,

            dL_dY = (1 / sigma_n) * np.diag(beta_star) @ (-Y + U @ P @ U.T @ np.diag(beta_star) @ Y)

            self.wv = v
            self.S_inv = Kmmi - P
            self.M_inv = P
            self.grad_dict = {'dL_dKmm': dL_dK, 'dL_dKdiag': dL_dR, 'dL_dKnm': dL_dU.T,
                              'dL_dY': dL_dY}



    return sparse_GPDM
def GPDM_N_Order_func(order, GPDM_func, *args, **kwargs):
    if order == 1:
        GPDM_N_Order = GPDM_1st_order
    elif order == 2:
        GPDM_N_Order = GPDM_2nd_order
    else:
        raise ValueError('GPDM can only be 1st or 2nd order.')
    GPDM_class = GPDM_func(GPDM_N_Order)
    return GPDM_class(*args, **kwargs)

def GPDM(order, num_inducing, *args, **kwargs):
    if num_inducing is None or num_inducing == 0:
        num_inducing = None
        GPDM_func = Full_GPDM_N_Order
        return GPDM_N_Order_func(order, GPDM_func, *args, **kwargs)
    elif num_inducing > 0 and isinstance(num_inducing, int):
        GPDM_func = Sparse_GPDM_N_Order
        return GPDM_N_Order_func(order, GPDM_func, num_inducing, *args, **kwargs)
    else:
        raise ValueError('Model must either not have inducing variables, or include an integer number greater than 1.')


class GPDMM(Prior,Parameterized):
    """
    Gaussian Process Dynamical Mixture Model:
    Represents a mixture of GPDMs wherein the mixture parameter is distributed in equal proportions over the GPDMs
    X: Latent space of the GPDM, NxD, where n of N represents a time point in 1 or more sequences stacked end-to-end
    GPDM: Selection of GPDM i.e. 1st or 2nd order.
    """
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, X, num_inducing, num_seqs, gpdm_var, seq_eps, sigma, GPDM_order=1, GPNode=None, name='GPDMM'):  # X should be a latent dimension Param
        self.gradients = None
        self.num_seqs = num_seqs
        self.N, self.D = X.shape
        self.X = X
        self.gpdm_timesteps = GPDM_order
        self.GPDMs = []
        super().__init__(name=name)

        self.pzs = []

        num_seqs = len(seq_eps)
        k_set_size = int(np.size(seq_eps) / self.num_seqs)
        k_set_eps = np.arange(k_set_size, num_seqs + k_set_size, k_set_size, dtype=int)
        self.seq_eps_k_list = []
        self.seq_x0s_k_list = []
        if self.num_seqs == 1:
            seq_x0s = np.array([0])
        else:
            seq_x0s = np.concatenate([np.array([0]), (np.array(seq_eps) + 1)[:-1]])


        for k_ep in k_set_eps:
            self.seq_eps_k_list.append(np.array(seq_eps)[(k_ep - k_set_size):k_ep])
            self.seq_x0s_k_list.append(np.array(seq_x0s)[(k_ep - k_set_size):k_ep])

        X_list = self.partition_latent_space(self.X)
        if num_inducing is not None:
            num_inducing_per_gpdm = int(num_inducing)

        else:
            num_inducing_per_gpdm = None
            #num_inducing_per_gpdm = 25

        self.num_GPDMs = 0
        GPDMs_temp = []
        for X_GPDM,seq_eps,seq_x0s in zip(X_list,self.seq_eps_k_list,self.seq_x0s_k_list):
            seq_eps_GPDM = seq_eps-seq_x0s[0]
            self.pzs.append(X_GPDM.shape[0] / (self.N))
            x0_prior = GPDM_IG_prior(sigma, self.D)
            GPDM_model = GPDM(GPDM_order,num_inducing_per_gpdm,X_GPDM, gpdm_var, x0_prior, seq_eps_GPDM, sigma)
            self.GPDMs.append(GPDM_model)
            self.num_GPDMs +=1

        logger.info("adding GPDMM parameters")
        self.link_params()
        self.update_parameters(self.X)

    def link_params(self):
        for GPDM in self.GPDMs:
            for p in range(len(GPDM.parameters)):
                self.link_parameter(GPDM.parameters[0])

    def expand(self, X):
        return X.reshape([self.N, self.D])

    def partition_latent_space(self,X):
        X = self.expand(X)
        X_list = []
        for seq_eps, seq_x0s in zip(self.seq_eps_k_list, self.seq_x0s_k_list):
            X_list.append(X[seq_x0s[0]:(seq_eps[-1]+1),:])
        return X_list

    def update_parameters(self, X):
        X_list = self.partition_latent_space(X)
        grads = []
        for X_GPDM,GPDM in zip(X_list,self.GPDMs):
            GPDM.update_parameters(X_GPDM)
            grads.append(GPDM.gradients)

    def lnpdf_grad(self, X):
        lnpdf_grads = []
        X_list = self.partition_latent_space(X)
        for X_GPDM, GPDM in zip(X_list, self.GPDMs):
            lnpdf_grads.append(GPDM.lnpdf_grad(X_GPDM.flatten()))
        return np.concatenate(lnpdf_grads)

    def pdf(self, X):
        pdfs = []
        X_list = self.partition_latent_space(X)
        for X_GPDM, GPDM in zip(X_list, self.GPDMs):
            pdfs.append(GPDM.pdf(X_GPDM.flatten()))
        return np.sum(pdfs)

    def lnpdf(self, X):
        lnpdfs = []
        X_list = self.partition_latent_space(X)
        for X_GPDM, GPDM in zip(X_list, self.GPDMs):
            lnpdfs.append(GPDM.lnpdf(X_GPDM.flatten()))
        return np.sum(lnpdfs)

    def lnpdf_comparison(self, X):
        lnpdfs = []
        for GPDM in self.GPDMs:
            lnpdfs.append(GPDM.ln_pred_pdf(X))
        return np.hstack(lnpdfs)

    def predict_trajectory(self, Xstar_i, traj_j, num_tps, init_t=0):
        GPDM = self.GPDMs[traj_j]
        return GPDM.predict_trajectory(Xstar_i,num_tps,init_t=init_t)

    def mn_predict(self, Xstar, traj):
        GPDM = self.GPDMs[traj]
        Xstar_in, Xstar_out, X0 = GPDM.partition_X_preds(Xstar)
        return GPDM.mn_predict(Xstar_in)

    def px0_given_z(self,x0,z):
        #return self.GPDMs[z].pred_pdf(x0)
        return self.GPDMs[z].pred_pdf(x0)

    def ln_px0_given_z(self,x0,z):
        #return self.GPDMs[z].pred_pdf(x0)
        return self.GPDMs[z].ln_pred_pdf(x0)

    def px0_given_z_slice(self,x0,z):
        #return self.GPDMs[z].pred_pdf(x0)
        return self.GPDMs[z].pred_pdf_slice(x0)

    def ln_px0_given_z_slice(self,x0,z):
        #return self.GPDMs[z].pred_pdf(x0)
        return self.GPDMs[z].ln_pred_pdf_slice(x0)

    def pzk_given_x0(self, x0, z_k):
        z_set = set(np.arange(0, self.num_seqs, 1).flatten())
        z_set.remove(z_k)

        log_probs = np.array([self.ln_pz(z_j) + self.ln_px0_given_z(x0, z_j) for z_j in z_set])
        log_probs = np.append(log_probs, self.ln_pz(z_k) + self.ln_px0_given_z(x0, z_k))

        # Use logsumexp for numerical stability
        log_M = logsumexp(log_probs)

        return np.exp(log_probs[-1] - log_M)

    def pzk_given_x0_slice(self,x0,z_k):
        Z = 0#0.00001
        for z_j in range(self.num_seqs):
            Z += self.pz(z_j)*self.px0_given_z_slice(x0,z_j)
        return self.pz(z_k)*self.px0_given_z_slice(x0,z_k) / Z

    def ln_pzk_given_x0_slice(self,x0,z_k):
        z_set = set(np.arange(0, self.num_seqs, 1).flatten())
        z_set.remove(z_k)
        M = 1
        for z_j in z_set:
            M += np.exp(
                self.ln_pz(z_j) + self.ln_px0_given_z(x0, z_j) - (self.ln_pz(z_k) + self.ln_px0_given_z(x0, z_k)))
        return 1 / M

    def pz(self,z):
        return self.pzs[z]

    def ln_pz(self,z):
        return np.log(self.pzs[z])

    def predict_best_z(self,x0):
        pzk_given_x0_list = []
        for z_k in range(self.num_seqs):
            pzk_given_x0_list.append(self.pzk_given_x0(x0, z_k))
        #print(pzk_given_x0_list)
        return int(np.argmax(pzk_given_x0_list)),pzk_given_x0_list

    def predict_best_z_slice(self,x0):
        pzk_given_x0_list = []
        for z_k in range(self.num_seqs):
            pzk_given_x0_list.append(self.pzk_given_x0_slice(x0, z_k))
        print(pzk_given_x0_list)
        return np.argmax(pzk_given_x0_list)

    def S_and_M_inv_slice_calc(self,dims, sliced_dims):
        for z_k in range(self.num_seqs):
            self.GPDMs[z_k].S_and_M_inv_slice(dims, sliced_dims)



