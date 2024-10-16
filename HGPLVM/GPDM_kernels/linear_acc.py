# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from HGPLVM.GPDM_kernels.kern_2nd_order import Kern
from GPy.util.linalg import tdot
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this
from GPy.kern.src.psi_comp import PSICOMP_Linear

class Linear_Acc(Kern):
    """
    Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^{\\text{input_dim}} \sigma^2_i x_iy_i

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: array or list of the appropriate size (or float if there
                     is only one variance parameter)
    :param ARD: Auto Relevance Determination. If False, the kernel has only one
                variance parameter \sigma^2, otherwise there is one variance
                parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    """

    def __init__(self, input_dim, seq_eps, variances1=None, variances2=None, ARD=False, active_dims=None, name='linear'):
        super(Linear_Acc, self).__init__(input_dim, seq_eps, active_dims, name)
        self.ARD = ARD
        if not ARD:
            if variances1 is not None:
                variances1 = np.asarray(variances1)
                assert variances1.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances1 = np.ones(1)

            if variances2 is not None:
                variances2 = np.asarray(variances2)
                assert variances2.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances2 = np.ones(1)
        else:
            raise NotImplementedError
            if variances1 is not None:
                variances1 = np.asarray(variances1)
                assert variances1.size == self.input_dim, "bad number of variances, need one ARD variance per input_dim"
            else:
                variances1 = np.ones(self.input_dim)

            if variances2 is not None:
                variances2 = np.asarray(variances2)
                assert variances2.size == self.input_dim, "bad number of variances, need one ARD variance per input_dim"
            else:
                variances2 = np.ones(self.input_dim)

        self.variances1 = Param('variances1', variances1, Logexp())
        self.link_parameter(self.variances1)

        self.variances2 = Param('variances2', variances2, Logexp())
        self.link_parameter(self.variances2)

        self.psicomp = PSICOMP_Linear()

    def to_dict(self):
        input_dict = super(Linear_Acc, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.Linear"
        input_dict["variances"] = self.variances.values.tolist()
        input_dict["ARD"] = self.ARD
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return Linear_Acc(**input_dict)

    @Cache_this(limit=3)
    def K(self, X, X2=[None,None]):
        Xt, Xt1 = X
        X2t, X2t1 = X2
        if self.ARD:
            raise NotImplementedError("ARD kernel is not yet implemented.")
        else:
            return self._dot_product(Xt, X2t) * self.variances1 + self._dot_product(Xt1, X2t1) * self.variances2

    @Cache_this(limit=3, ignore_args=(0,))
    def _dot_product(self, X, X2=None):
        if X2 is None:
            return tdot(X)
        else:
            return np.dot(X, X2.T)

    def update_gradients_full(self, dL_dK, X, X2=[None,None]):
        Xt, Xt1 = X
        X2t, X2t1 = X2

        if X2t is None:
            #self.variances1.gradient = np.trace(dL_dK.T @ Xt @ Xt.T)
            #self.variances2.gradient = np.trace(dL_dK.T @ Xt1 @ Xt1.T)
            self.variances1.gradient = np.sum(Xt @ Xt.T * dL_dK)
            self.variances2.gradient = np.sum(Xt1 @ Xt1.T * dL_dK)
        else:
            #self.variances1.gradient = np.trace(dL_dK.T @ Xt @ X2t.T)
            #self.variances2.gradient = np.trace(dL_dK.T @ Xt1 @ X2t1.T)
            self.variances1.gradient = np.sum(Xt @ X2t.T * dL_dK)
            self.variances2.gradient = np.sum(Xt1 @ X2t1.T * dL_dK)



    def gradients_X(self, dL_dK, X, X2=[None,None]):
        Xt, Xt1 = X
        X2t, X2t1 = X2

        dL_dKt1, dL_dKt2 = dL_dK
        if X2t is None:
            return 2*self.variances1*dL_dKt1@Xt1 + 2*self.variances2*dL_dKt2@Xt
        else:
            return self.variances1 * dL_dKt1 @ X2t1 + self.variances2 * dL_dKt2 @ X2t
            #return (self.variances2 * X2t1 + self.variances1 * X2t)

    '''def gradients_X(self, dL_dK, X, X2=None):
        Xt, Xt1 = X
        if X2 is None:
            dK_dXt = 2 * (self.variances2 * Xt1 + self.variances1 * Xt)
        else:
            X2t, X2t1 = X2
            dK_dXt = (self.variances2 * X2t1 + self.variances1 * X2t)
        return dL_dK @ dK_dXt'''

    '''def update_gradients_diag(self, dL_dKdiag, X):
        Xt, Xt1 = X
        self.kappa.gradient = np.einsum('ij,i->j', np.square(X), dL_dKdiag)
        self.W.gradient = 2.*np.einsum('ij,ik,jl,i->kl', X, X, self.W, dL_dKdiag)'''

    def update_gradients_diag(self, dL_dKdiag, X):
        Xt, Xt1 = X
        tmp_t = np.diag(dL_dKdiag)@ Xt ** 2
        tmp_t1 = np.diag(dL_dKdiag)@ Xt1 ** 2
        if self.ARD:
            self.variances1.gradient = tmp_t.sum(0)
            self.variances2.gradient = tmp_t1.sum(0)
        else:
            self.variances1.gradient = np.atleast_1d(tmp_t.sum())
            self.variances2.gradient = np.atleast_1d(tmp_t1.sum())

    def gradients_X_diag(self, dL_dKdiag, X):
        Xt, Xt1 = X
        return 2.*self.variances1*np.diag(dL_dKdiag)@Xt + 2.*self.variances2*np.diag(dL_dKdiag)@Xt1

    def Kdiag(self, X, X2=[None, None]):
        Xt, Xt1 = X
        X2t, X2t1 = X2
        return np.sum(self.variances2 * np.square(Xt1), -1) + np.sum(self.variances1 * np.square(Xt), -1)