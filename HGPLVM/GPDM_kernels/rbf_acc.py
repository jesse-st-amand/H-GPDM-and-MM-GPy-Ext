# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from HGPLVM.GPDM_kernels.stationary import Stationary
from GPy.kern.src.psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from GPy.core import Param
from paramz.caching import Cache_this
from paramz.transformations import Logexp
from GPy.kern.src.grid_kerns import GridRBF
from GPy.util.linalg import tdot
from GPy import util
from GPy.util.config import config # for assesing whether to use cython
from scipy import sparse

'''try:
    from . import stationary_cython
    use_stationary_cython = config.getboolean('cython', 'working')
except ImportError:
    print('warning in stationary: failed to import cython module: falling back to numpy')'''
use_stationary_cython = False


class RBF_Acc(Stationary):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    """
    _support_GPU = True
    def __init__(self, input_dim, seq_eps, variance=1., lengthscale=None, lengthscale_t1=None, ARD=False, active_dims=None, name='rbf', useGPU=False, inv_l=False):
        super(RBF_Acc, self).__init__(input_dim, seq_eps, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)
        if self.useGPU:
            self.psicomp = PSICOMP_RBF_GPU()
        else:
            self.psicomp = PSICOMP_RBF()
        self.use_invLengthscale = inv_l
        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)
            assert lengthscale.size == 1, "Only 1 lengthscale needed for non-ARD kernel"
        self.lengthscale = Param('lengthscale', lengthscale, Logexp())
        if lengthscale_t1 is None:
            lengthscale_t1 = np.ones(1)
        else:
            lengthscale_t1 = np.asarray(lengthscale_t1)
            assert lengthscale_t1.size == 1, "Only 1 lengthscale needed for non-ARD kernel"
        self.lengthscale_t1 = Param('lengthscale', lengthscale_t1, Logexp())



    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(RBF_Acc, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.RBF"
        input_dict["inv_l"] = self.use_invLengthscale
        if input_dict["inv_l"] == True:
            input_dict["lengthscale"] = np.sqrt(1 / float(self.inv_l))
        return input_dict

    #@Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=[None, None]):
        """
        Kernel function applied on inputs X and X2.
        In the stationary case there is an inner function depending on the
        distances from X to X2, called r.

        K(X, X2) = K_of_r((X-X2)**2)
        """
        Xt, Xt1 = X
        X2t, X2t1 = X2
        r1, r2 = self._scaled_dist(Xt, Xt1, X2t, X2t1)
        return self.K_of_r(r1, r2)

    def K_of_r(self, r1, r2):
        return self.variance * np.exp(-0.5 * r1**2 - 0.5 * r2**2)

    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """

        return self._unscaled_dist_singular(X, X2)


    def _unscaled_dist_singular(self, X, X2=None):
        if X2 is None:
            Xsq = np.sum(np.square(X), 1)
            r2 = -2. * tdot(X) + (Xsq[:, None] + Xsq[None, :])
            util.diag.view(r2)[:, ] = 0.  # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            X1sq = np.sum(np.square(X), 1)
            X2sq = np.sum(np.square(X2), 1)
            r2 = -2. * np.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)


    #@Cache_this(limit=3, ignore_args=())
    def _scaled_dist(self, Xt, Xt1, X2t = None, X2t1 = None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

        Note that if thre is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards

        """
        if X2t is not None and X2t1 is not None:
            X2t = X2t * self.lengthscale
            X2t1 = X2t1 * self.lengthscale_t1

        r_Xt = self._unscaled_dist(Xt * self.lengthscale, X2t)
        r_Xt1 = self._unscaled_dist(Xt1 * self.lengthscale_t1, X2t1)
        return r_Xt, r_Xt1

    def __getstate__(self):
        dc = super(RBF_Acc, self).__getstate__()
        if self.useGPU:
            dc['psicomp'] = PSICOMP_RBF()
            dc['useGPU'] = False
        return dc

    def __setstate__(self, state):
        self.use_invLengthscale = False
        return super(RBF_Acc, self).__setstate__(state)

    def parameters_changed(self):
        if self.use_invLengthscale: self.lengthscale[:] = 1./np.sqrt(self.inv_l+1e-200)
        super(RBF_Acc,self).parameters_changed()

    def update_gradients_full(self, dL_dK, X, X2=[None, None]):
        if self.use_invLengthscale:
            raise NotImplementedError

        Xt, Xt1 = X
        X2t, X2t1 = X2
        Kx = self.K(X, X2)
        r1, r2 = self._scaled_dist(Xt, Xt1, X2t, X2t1)

        self.variance.gradient = np.sum(Kx * dL_dK) / self.variance

        self.lengthscale.gradient = np.sum(-.5*r1 * Kx * dL_dK) * self.lengthscale
        self.lengthscale_t1.gradient = np.sum(-.5*r2 * Kx * dL_dK) * self.lengthscale_t1


    def gradients_X(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt K (dL_dK), compute the derivative wrt X
        """
        if X2 is None:
            return self._gradients_X_X2_pure(dL_dK, X, X)
        else:
            return self._gradients_X_X2_pure(dL_dK, X, X2)

    '''def _gradients_X_pure(self, dL_dK, X):

        dL_dX = np.zeros_like(X[0])
        for q in range(dL_dX.shape[1]):
            dK_dX = self.grad_x(X, X, q)
            dL_dX[:, q] = 2 * np.sum(dL_dK * dK_dX, axis=1) - np.diag(dL_dK) * np.diag(dK_dX)
        return dL_dX'''

    def _gradients_X_X2_pure(self, dL_dK, X, X2):
        N_x, D_x = np.hstack(X).shape
        if X2[0] is None:
            X2 = X
        N_x2, _ = np.hstack(X2).shape
        dL_dK1, _ = dL_dK
        dL_dX_list = []
        for q in range(D_x):
            if N_x == N_x2:
                dK_dX = self.grad_x(X, X2, q)
                dL_dX_list.append(2 * np.sum(dL_dK1 * dK_dX, axis=1) - np.diag(dL_dK1) * np.diag(dK_dX))
            elif N_x > N_x2:
                dK_dX = self.grad_x(X2, X, q)
                dL_dX_list.append(2 * np.sum(dL_dK1 * dK_dX.T, axis=1))
            elif N_x < N_x2:
                dK_dX = self.grad_x(X2, X, q)
                dL_dX_list.append(2 * np.sum(dL_dK1.T * dK_dX, axis=0))

        dL_dX_arr = np.vstack(dL_dX_list).T

        dL_dX = dL_dX_arr[:,:D_x//2] + dL_dX_arr[:,D_x//2:]

        return dL_dX

    def grad_x(self, X1, X2, q):
        X1_stack = np.hstack(X1)
        X2_stack = np.hstack(X2)
        num_data1, num_data2 = X1_stack.shape[0], X2_stack.shape[0]
        K1 = np.tile(X1_stack[:, q:q + 1], (1, num_data2))
        K2 = np.tile(X2_stack[:, q:q + 1].T, (num_data1, 1))
        K = self.K(X1, X2)

        if q < X1_stack.shape[1] // 2:
            return -self.lengthscale * (K1 - K2) * K
        else:
            return -self.lengthscale_t1 * (K1 - K2) * K

    '''def _gradients_X_pure(self, dL_dK, X):
        Xt, Xt1 = X
        Nt, D = Xt.shape
        dL_dX = np.zeros_like(Xt)
        for q in range(Xt.shape[1]):
            dK_dX = self._grad_x(Xt, Xt1, q)
            dL_dX[:, q] = 2 * np.sum(dL_dK * dK_dX, axis=1) - np.diag(dL_dK) * np.diag(dK_dX)
        return dL_dX'''

    '''def _gradients_X_pure(self, dL_dK, X):
        Xt, Xt1 = X
        dL_dX = np.zeros_like(Xt)
        for q in range(Xt.shape[1]):
            dK_dXt = self._grad_x(self.lengthscale,X, X, q)
            #dK_dXt1 = self._grad_x(self.lengthscale_t1,Xt1, Xt1, q)
            dL_dX[:, q] = 2 * np.sum(dL_dK * dK_dX, axis=1) - np.diag(dL_dK) * np.diag(dK_dX)
            #               + 2 * np.sum(dL_dK * dK_dXt1, axis=1) - np.diag(dL_dK) * np.diag(dK_dXt1))
        return dL_dX

    def _gradients_X_X2_pure(self, dL_dK, X, X2):
        Xt, Xt1 = X
        X2t, X2t1 = X2
        dL_dX = np.zeros_like(Xt)
        for q in range(Xt.shape[1]):
            dK_dXt = self._grad_x(self.lengthscale, Xt, X2t, q)
            dK_dXt1 = self._grad_x(self.lengthscale_t1, Xt1, X2t1, q)
            dL_dX[:, q] = (2 * np.sum(dL_dK * dK_dXt, axis=1) - np.diag(dL_dK) * np.diag(dK_dXt)
                           + 2 * np.sum(dL_dK * dK_dXt1, axis=1) - np.diag(dL_dK) * np.diag(dK_dXt1))
        return dL_dX

    def _grad_x(self, ls, X1: np.ndarray, X2: np.ndarray, q: int):
        Xt, Xt1 = X
        X2t, X2t1 = X2
        num_data, num_data2 = X1.shape[0], X2.shape[0]
        K1 = np.tile(X1[:, q][:, None], (1, num_data2))
        K2 = np.tile(X2[:, q][None, :], (num_data, 1))
        return -ls * (K1 - K2) * self.K([X1, X2])'''

    def _inv_dist(self, X, X2=[None, None]):
        """
        Compute the elementwise inverse of the distance matrix, except on the
        diagonal, where we return zero (the distance on the diagonal is zero).
        This term appears in derviatives.
        """
        Xt, Xt1 = X
        X2t, X2t1 = X2
        r1, r2 = self._scaled_dist(Xt, Xt1, X2t, X2t1)
        return 1./np.where(r1 != 0., r1, np.inf), 1./np.where(r2 != 0., r2, np.inf)

    '''def _gradients_X_pure(self, dL_dK, X):
            Xt, Xt1 = X
            Nt, D = Xt.shape
            Kx = self.K([Xt,Xt1])
            dK_dX = np.zeros([Nt*Nt,Nt*D])
            dL_dK_flat = np.zeros([1,dL_dK.flatten().shape[0]])
            for i in range(Nt):
                t = i #+ 1
                for j in range(Nt):
                    tau = j #+ 1
                    #dL_dK_flat[0,(i * Nt) + j] = dK_dX[t, tau]
                    dL_dK_flat[0,(i * Nt) + j] = dL_dK[i, j]
                    if t-tau == 1:
                        n = t-2
                        if n >= 0 and  n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j] * self.lengthscale * (
                                        Xt1[t - 2, :] - Xt[t - 2, :])
                        n = t-1
                        if n >= 0 and  n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j]*(self.lengthscale_t1*(Xt1[t-1,:]-Xt[t-1,:])-self.lengthscale*(Xt1[t-2,:]-Xt[t-2,:]))
                        n = t
                        if n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j]*(-self.lengthscale_t1*(Xt1[t-1,:]-Xt[t-1,:]))
                    elif t-tau == 0:
                        continue
                    elif t-tau == -1:
                        n = t - 1
                        if n >= 0 and  n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j] * (-self.lengthscale * (Xt[t-1, :] - Xt1[t-1, :]))
                        n = t
                        if n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j] * (
                                    -self.lengthscale_t1 * (Xt[t, :] - Xt1[t, :]) + self.lengthscale * (
                                        Xt[t-1, :] - Xt1[t-1, :]))
                        n = t + 1
                        if n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j] * (self.lengthscale_t1 * (Xt[t, :] - Xt1[t, :]))
                    else:
                        n = tau - 1
                        if n >= 0 and  n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j] * (
                                        self.lengthscale * (Xt[t-1, :] - Xt[tau-1, :]))
                        n = t - 1
                        if n >= 0 and  n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j] * (
                                        -self.lengthscale * (Xt[t-1, :] - Xt[tau-1, :]))
                        n = tau
                        if n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j] * (self.lengthscale_t1 * (Xt1[t-1, :] - Xt1[tau-1, :]))
                        n = t
                        if n < Nt:
                            dK_dX[(i * Nt) + j, (n * D) :(n*D+D)] = Kx[i, j] * (-self.lengthscale_t1 * (Xt1[t-1, :] - Xt1[tau-1, :]))

            dL_dX_flat = dL_dK_flat @ dK_dX
            dL_dXin = np.zeros(Xt.shape)

            for n in range(Nt):
                dL_dXin[int(n), :] = dL_dX_flat[0,int(n * D) :int(n*D+D)]

            return dL_dXin'''

    """def _gradients_X_pure(self, dL_dK, X):
        Xt, Xt1 = X
        Nt, D = Xt.shape
        '''print(f"Xt shape: {Xt.shape}")
        print(f"Xt1 shape: {Xt1.shape}")'''
        Kx = self.K([Xt, Xt1])

        # Precompute differences
        diff_t = Xt1 - Xt
        diff_t_minus_1 = np.roll(diff_t, 1, axis=0)
        diff_t_minus_1[0] = 0  # Handle boundary condition

        # Create lists for sparse matrix construction
        rows, cols, data = [], [], []

        max_col_index = 0  # To keep track of the maximum column index
        min_col_index = Nt * D  # To keep track of the minimum column index

        for i in range(Nt):
            for j in range(max(0, i - 2), min(Nt, i + 3)):  # Only compute for nearby time steps
                if i == j:
                    continue

                if i - j == 1:
                    for d in range(D):
                        rows.extend([i * Nt + j] * 3)
                        new_cols = [
                            max(0, min(j * D + d, Nt * D - 1)),
                            max(0, min((j + 1) * D + d, Nt * D - 1)),
                            max(0, min((j + 2) * D + d, Nt * D - 1))
                        ]
                        cols.extend(new_cols)
                        max_col_index = max(max_col_index, max(new_cols))
                        min_col_index = min(min_col_index, min(new_cols))
                        data.extend([
                            Kx[i, j] * self.lengthscale * diff_t_minus_1[j, d],
                            Kx[i, j] * (self.lengthscale_t1 * diff_t[j, d] - self.lengthscale * diff_t_minus_1[j, d]),
                            Kx[i, j] * (-self.lengthscale_t1 * diff_t[j, d])
                        ])
                elif i - j == -1:
                    for d in range(D):
                        rows.extend([i * Nt + j] * 3)
                        new_cols = [
                            max(0, min(i * D + d, Nt * D - 1)),
                            max(0, min((i + 1) * D + d, Nt * D - 1)),
                            max(0, min((i + 2) * D + d, Nt * D - 1))
                        ]
                        cols.extend(new_cols)
                        max_col_index = max(max_col_index, max(new_cols))
                        min_col_index = min(min_col_index, min(new_cols))
                        data.extend([
                            Kx[i, j] * (-self.lengthscale * diff_t[i, d]),
                            Kx[i, j] * (-self.lengthscale_t1 * diff_t[min(i + 1, Nt - 1), d] + self.lengthscale *
                                        diff_t[i, d]),
                            Kx[i, j] * (self.lengthscale_t1 * diff_t[min(i + 1, Nt - 1), d])
                        ])
                else:
                    for d in range(D):
                        rows.extend([i * Nt + j] * 4)
                        new_cols = [
                            max(0, min(j * D + d, Nt * D - 1)),
                            max(0, min((i - 1) * D + d, Nt * D - 1)),
                            max(0, min(j * D + d, Nt * D - 1)),
                            max(0, min(i * D + d, Nt * D - 1))
                        ]
                        cols.extend(new_cols)
                        max_col_index = max(max_col_index, max(new_cols))
                        min_col_index = min(min_col_index, min(new_cols))
                        data.extend([
                            Kx[i, j] * self.lengthscale * (Xt[max(i - 1, 0), d] - Xt[j, d]),
                            Kx[i, j] * (-self.lengthscale * (Xt[max(i - 1, 0), d] - Xt[j, d])),
                            Kx[i, j] * (self.lengthscale_t1 * (Xt1[i, d] - Xt1[j, d])),
                            Kx[i, j] * (-self.lengthscale_t1 * (Xt1[i, d] - Xt1[j, d]))
                        ])

        # Convert lists to 1D numpy arrays
        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        data = np.array(data, dtype=float).flatten()

        # Debug information
        '''print(f"Shapes: rows={rows.shape}, cols={cols.shape}, data={data.shape}")
        print(f"Nt={Nt}, D={D}")
        print(f"Min column index: {min_col_index}")
        print(f"Max column index: {max_col_index}")
        print(f"Expected max column index: {Nt * D - 1}")'''

        dK_dX = sparse.coo_matrix((data, (rows, cols)), shape=(Nt * Nt, Nt * D))

        dL_dX_flat = dL_dK.flatten() @ dK_dX

        return dL_dX_flat.reshape(Xt.shape)"""

    '''def _gradients_X_X2_pure(self, dL_dK, X, X2):
        Xt, Xt1 = X
        X2t, X2t1 = X2
        Nt, D = Xt.shape
        N2t, D = X2t.shape
        Kx = self.K([Xt, Xt1],[X2t, X2t1])
        dK_dX = np.zeros([Nt * N2t, Nt * D])
        dL_dK_flat = np.zeros([1, dL_dK.flatten().shape[0]])
        for i in range(Nt):
            t = i  # + 1
            for j in range(N2t):
                tau = j  # + 1

                n = t - 1
                if n >= 0 and n < Nt:
                    dK_dX[(i * N2t) + j, (n * D):(n * D + D)] = Kx[i, j] * (
                            -self.lengthscale * (Xt[t - 1, :] - Xt[tau - 1, :]))

                n = t
                if n < Nt:
                    dK_dX[(i * N2t) + j, (n * D):(n * D + D)] = Kx[i, j] * (
                                -self.lengthscale_t1 * (Xt1[t - 1, :] - Xt1[tau - 1, :]))
        dL_dX_flat = dL_dK_flat @ dK_dX
        dL_dX = np.zeros(Xt.shape)

        for n in range(Nt):
            dL_dX[int(n), :] = dL_dX_flat[0, int(n * D):int(n * D + D)]

        return dL_dX'''
