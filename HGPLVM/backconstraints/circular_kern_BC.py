import numpy as np
import GPy
from GPy.core import Param
from HGPLVM.backconstraints.kern_BC_base import Kernel_BC
from HGPLVM.global_functions import inverse
from HGPLVM.backconstraints.backconstraints_base import BC_Base
from HGPLVM.backconstraints.geometries import geometry_base

class Urtasun_Kernel_BC(BC_Base):
    """
    Back-constraint that enforces points to lie on a circular manifold using RBF kernels
    Based on Urtasun 2007 and Taubert's work
    """
    def __init__(self, GPNode, output_dim, param_dict, name='Circular BC'):

        super().__init__(GPNode, output_dim, param_dict, name=name)
        
    def initialize_Y(self):
        super().initialize_Y()
        # Initialize phases based on sequence length
        phis, _, self.phases = self.map(self.Y)
        self.cos_phi = phis[0]
        self.sin_phi = phis[1]
        self.kern = GPy.kern.RBF(self.cos_phi.shape[1], variance=1, lengthscale=1, ARD=False)
        self.K_cos = self.kern.K(self.cos_phi)
        self.K_sin = self.kern.K(self.sin_phi)
        # Initialize mapping coefficients
        c_dim = self.output_dim // 2
        if self.output_dim % 2 == 0:
            self.c1 = Param('c1', np.random.randn(self.Y.shape[0],c_dim))
            self.c2 = Param('c2', np.random.randn(self.Y.shape[0],c_dim))
        else:
            self.c1 = Param('c1', np.random.randn(self.Y.shape[0],c_dim+1))
            self.c2 = Param('c2', np.random.randn(self.Y.shape[0],c_dim))




    def map(self, Y = None, phases = None):
        seq_len = self.GPNode.seq_eps[0] - self.GPNode.seq_x0s[0] + 1
        if phases is None:
            Y_KE = 1 / np.sum((.5 * Y ** 2), 1)
            if Y.shape[0] < seq_len:
                ratio = 2 * np.pi / np.sum(Y_KE)
                phases = np.cumsum(Y_KE * ratio)
            else:
                seq_x0s = np.arange(0, Y.shape[0], seq_len)
                seq_eps = np.arange(seq_len - 1, Y.shape[0], seq_len)
                phases_list = []
                for i, (x0_i, ep_i) in enumerate(zip(seq_x0s, seq_eps)):
                    ratio = 2 * np.pi / np.sum(Y_KE[x0_i:ep_i + 1])
                    phases = np.cumsum(Y_KE[x0_i:ep_i + 1] * ratio)
                    phases_list.append(phases)
                phases = np.hstack(phases_list).reshape([-1, 1])

        geometry = self.parametric_curve(phases)
        return geometry, Y, phases
        
    def f(self):
        # Map to 2D circular manifold
        x1 = self.K_cos @ self.c1
        x2 = self.K_sin @ self.c2
        return np.column_stack([x1, x2])

    def BC_f_new(self, Y_new, pred_group=0, **kwargs):
        if pred_group == 0:
            # For new points, compute their phase based on sequence position
            phis_new, _, self.phases = self.map(Y_new)
            cos_phi_new = self.kern.K(phis_new[0])
            sin_phi_new = self.kern.K(phis_new[1])
            K_cos_new = self.kern.K(cos_phi_new, self.cos_phi)
            K_sin_new = self.kern.K(sin_phi_new, self.sin_phi)
            
            # Map to circular manifold
            x1_new = K_cos_new @ self.c1
            x2_new = K_sin_new @ self.c2
            
            return np.column_stack([x1_new, x2_new])
        else:
            raise ValueError('Circular BC has no pred var groups 1 and 2.')

    def initialize_A(self):
        # Initialize coefficients for circular mapping
        pass

    def update_gradients(self, dL_dX):
        # Update gradients for mapping coefficients
        self.c1.gradient = self.K_cos.T @ dL_dX[:, :self.c1.shape[1]]
        self.c2.gradient = self.K_sin.T @ dL_dX[:, self.c1.shape[1]:]
        


        return dL_dX

    def link_params(self):
        self.link_parameter(self.c1)
        self.link_parameter(self.c2)
        

    def parametric_curve(self, phases):
        cos_phi = np.cos(phases)
        sin_phi = np.sin(phases)
        return cos_phi, sin_phi