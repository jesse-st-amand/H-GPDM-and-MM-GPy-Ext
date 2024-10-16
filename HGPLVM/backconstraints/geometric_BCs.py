import logging

import numpy as np
import GPy
from GPy.core import Param
from HGPLVM.backconstraints.GP_mapping_no_constraints import No_BC

from HGPLVM.global_functions import inverse
from HGPLVM.backconstraints.GP_BC_base import GP_BC, Multi_N_Weighted_GP_BC
from HGPLVM.backconstraints.GP_BC_base import sparse_GP_BC
from HGPLVM.backconstraints.GP_BC_base import weighted_GP_BC
from HGPLVM.backconstraints.kern_BC_base import Kernel_BC
from HGPLVM.backconstraints.lin_BC_base import Linear_BC, Linear_X_BC
from HGPLVM.backconstraints.mlp_BC import MLP_BC
from HGPLVM.backconstraints.GP_BC_base import Multi_Weighted_GP_BC
from HGPLVM.backconstraints.backconstraints_base import BC_Base
class Geometric_BC_base():
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='base', *args,**kwargs):
        if geometry_name.lower() == 'none':
            from HGPLVM.backconstraints.geometries import none
            self.geometry = none(self)
        elif geometry_name.lower() == 'ellipse':
            from HGPLVM.backconstraints.geometries import ellipse
            self.geometry = ellipse(self)
        elif geometry_name.lower() == 'toroid':
            from HGPLVM.backconstraints.geometries import toroid
            self.geometry = toroid(self)
        elif geometry_name.lower() == 'linear':
            from HGPLVM.backconstraints.geometries import linear
            self.geometry = linear(self)
        elif geometry_name.lower() == 'sine':
            from HGPLVM.backconstraints.geometries import sine
            self.geometry = sine(self)
        elif geometry_name.lower() == 'sphere':
            from HGPLVM.backconstraints.geometries import sphere
            self.geometry = sphere(self)
        elif geometry_name.lower() == 'spiral':
            from HGPLVM.backconstraints.geometries import spiral
            self.geometry = spiral(self)
        elif geometry_name.lower() == 'torus':
            from HGPLVM.backconstraints.geometries import torus
            self.geometry = torus(self)
        elif geometry_name.lower() == 'mobius strip':
            from HGPLVM.backconstraints.geometries import mobius_strip
            self.geometry = mobius_strip(self)
        elif geometry_name.lower() == 'klein bottle':
            from HGPLVM.backconstraints.geometries import klein_bottle
            self.geometry = klein_bottle(self)
        elif geometry_name.lower() == 'ep hands':
            from HGPLVM.backconstraints.geometries import EP_hands_geometry
            self.geometry = EP_hands_geometry(self)
        elif geometry_name.lower() == 'ep hands toroid':
            from HGPLVM.backconstraints.geometries import EP_hands_toroid_geometry
            self.geometry = EP_hands_toroid_geometry(self)

        elif geometry_name.lower() == 'fourier':
            from HGPLVM.backconstraints.geometries import fourier
            self.geometry = fourier(self, param_dict['geo params'])
        elif geometry_name.lower() == 'wavelet_basis':
            from HGPLVM.backconstraints.geometries import wavelet_basis
            self.geometry = wavelet_basis(self)
        elif geometry_name.lower() == 'legendre_basis':
            from HGPLVM.backconstraints.geometries import legendre_basis
            self.geometry = legendre_basis(self)
        elif geometry_name.lower() == 'hermite_basis':
            from HGPLVM.backconstraints.geometries import hermite_basis
            self.geometry = hermite_basis(self)
        elif geometry_name.lower() == 'laguerre_basis':
            from HGPLVM.backconstraints.geometries import laguerre_basis
            self.geometry = laguerre_basis(self)
        elif geometry_name.lower() == 'chebyshev_basis':
            from HGPLVM.backconstraints.geometries import chebyshev_basis
            self.geometry = chebyshev_basis(self)
        elif geometry_name.lower() == 'zernike_basis':
            from HGPLVM.backconstraints.geometries import zernike_basis
            self.geometry = zernike_basis(self)
        elif geometry_name.lower() == 'spherical_harmonics_basis':
            from HGPLVM.backconstraints.geometries import spherical_harmonics_basis
            self.geometry = spherical_harmonics_basis(self)
        elif geometry_name.lower() == 'haar_basis':
            from HGPLVM.backconstraints.geometries import haar_basis
            self.geometry = haar_basis(self)
        elif geometry_name.lower() == 'walsh_basis':
            from HGPLVM.backconstraints.geometries import walsh_basis
            self.geometry = walsh_basis(self)
        else:
            raise ValueError(geometry_name + ' is not a valid geometry.')
        self.geometry_name = geometry_name
        super().__init__(GPNode, output_dim, param_dict,*args,**kwargs)

    def initialize_Y(self):
        self.Y_geos, Y_out, self.Y_phases = self.geometry.map(self.geometry.get_initial_params(),Y = self.GPNode.Y)
        super().initialize_Y()

    def initialize_X(self, X_init = None):
        input_dim = self.GPNode.input_dim - self.Y_geos.shape[1]
        if X_init is None:
            if input_dim > 0:
                init_X = self.GPNode.init_embedding(embed_type=self.GPNode.attr_dict['init'],Y=self.Y,input_dim = input_dim)
                self.X = np.hstack([self.Y_geos,init_X])
            else:
                self.X = self.Y_geos
        else:
            self.X = np.hstack([self.Y_geos, X_init[:,:-self.Y_geos.shape[1]]])

    def get_X(self):
        X_geos, _, _ = self.geometry.map(self.geometry.get_params(),phases = self.Y_phases)
        return np.hstack([X_geos, self.X[:, X_geos.shape[1]:]])

    def link_params(self):
        self.geometry.link_params()
        if self.geometry.parameters is not None:
            for i in range(len(self.geometry.parameters)):
                self.link_parameter(self.geometry.parameters[0], index=0)
        super().link_params()

    def constrain_params(self):
        for con in self.param_dict['constraints']:
            for param in self.parameters:
                if con == param.name:
                    param.constrain_fixed()
        super().constrain_params()


    def update_gradients(self, dL_dX):
        dL_dXin = super().update_gradients(dL_dX)
        '''if self.GPNode.prior is not None:
            self.geometry.update_gradients(self.GPNode.prior.lnpdf_grad(self.X).reshape(self.X.shape), self.Y_phases)
        else:
            self.geometry.update_gradients(dL_dXin, self.Y_phases)'''
        self.geometry.update_gradients(dL_dXin, self.Y_phases)

'''class No_BC(Geometric_BC_base,none):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='No BC'):
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)'''

class GP_Geo_BC(Geometric_BC_base,GP_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='GP Geo BC'):
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

class sparse_GP_Geo_BC(Geometric_BC_base,sparse_GP_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='sparse GP Geo BC'):
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

class weighted_GP_Geo_BC(Geometric_BC_base,weighted_GP_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='weighted GP Geo BC'):
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

class Multi_Weighted_GP_Geo_BC(Geometric_BC_base,Multi_Weighted_GP_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='multi weighted GP Geo BC'):
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

class Multi_N_Weighted_GP_Geo_BC(Geometric_BC_base,Multi_N_Weighted_GP_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='multi weighted GP Geo BC'):
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

class No_BC_geo(Geometric_BC_base,No_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='No BC geo'):
        self.param_dict = param_dict
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

    def constrain_params(self):
        No_BC.constrain_params(self)

    def link_params(self):
        No_BC.link_params(self)

    def update_gradients(self, dL_dX):
        No_BC.update_gradients(self, dL_dX)

class No_BC_geo_learn(Geometric_BC_base,No_BC):

    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='No BC geo'):
        self.param_dict = param_dict
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)



class Linear_Geo_BC(Geometric_BC_base,Linear_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='Linear Geo BC'):
        self.param_dict = param_dict
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

    def constrain_params(self):
        Linear_BC.constrain_params(self)

    def link_params(self):
        Linear_BC.link_params(self)

    def update_gradients(self, dL_dX):
        Linear_BC.update_gradients(self, dL_dX)

class Linear_X_Geo_BC(Geometric_BC_base,Linear_X_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='Linear Geo BC'):
        self.param_dict = param_dict
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

    def constrain_params(self):
        Linear_X_BC.constrain_params(self)

    def link_params(self):
        Linear_X_BC.link_params(self)

    def update_gradients(self, dL_dX):
        Linear_X_BC.update_gradients(self, dL_dX)

class Kernel_Geo_BC(Geometric_BC_base,Kernel_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='Kernel Geo BC'):
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

    def constrain_params(self):
        Kernel_BC.constrain_params(self)

    def link_params(self):
        Kernel_BC.link_params(self)

    def update_gradients(self, dL_dX):
        Kernel_BC.update_gradients(self, dL_dX)

class MLP_Geo_BC(Geometric_BC_base,MLP_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='MLP Geo BC'):
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)

    def get_X(self, X):
        X_geos, _, _ = self.geometry.map(self.geometry.get_params(),phases = self.Y_phases)
        return MLP_BC.get_X(self,np.hstack([X_geos, X[:, X_geos.shape[1]:]]))

    def constrain_params(self):
        MLP_BC.constrain_params(self)

    def link_params(self):
        MLP_BC.link_params(self)

    def update_gradients(self, dL_dX):
        MLP_BC.update_gradients(self, dL_dX)


'''class GP_Geo_BC(GP_BC):
    """
    """
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='GP Geo BC'):
        if geometry_name.lower() == 'ellipse':
            from HGPLVM.backconstraints.geometries import ellipse
            self.geometry = ellipse(self)
        elif geometry_name.lower() == 'toroid':
            from HGPLVM.backconstraints.geometries import toroid
            self.geometry = toroid(self)
        else:
            raise ValueError(geometry_name + ' is not a valid geometry.')
        self.geometry_name = geometry_name
        self.Geo_param_dict = {}
        super().__init__(GPNode, output_dim, param_dict, name='GP Geo BC')

    def initialize_X(self):
        input_dim = self.GPNode.input_dim - self.Y_geos.shape[1]
        if input_dim > 0:
            init_X = self.GPNode.init_embedding(embed_type=self.param_dict['embed_type'],Y=self.Y,input_dim = input_dim)
            self.X = np.hstack([self.Y_geos,init_X])
        else:
            self.X = self.Y_geos

    def get_X(self):
        X_geos, _, _ = self.geometry.map(self.geometry.get_params(),phases = self.Y_phases)
        return np.hstack([X_geos, self.X[:, X_geos.shape[1]:]])

    def link_params(self):
        self.geometry.link_params()
        for i in range(len(self.geometry.parameters)):
            self.link_parameter(self.geometry.parameters[0], index=0)
        super().link_params()

    def constrain_params(self):
        for con in self.param_dict['constraints']:
            for param in self.parameters:
                if con == param.name:
                    self.constrain_parameter(param)
        super().constrain_params()

    def initialize_Y(self):
        self.Y_geos, Y_out, self.Y_phases = self.geometry.map(self.geometry.get_initial_params(),Y = self.GPNode.Y)
        super().initialize_Y()

    def update_gradients(self, dL_dX):
        dL_dXin = super().update_gradients(dL_dX)
        self.geometry.update_gradients(dL_dXin, self.Y_phases)'''


"""class Multi_Weighted_GP_Geo_BC(Geometric_BC_base,Multi_Weighted_GP_BC):
    def __init__(self, geometry_name, GPNode, output_dim, param_dict, name='multi weighted GP Geo BC'):
        super().__init__(geometry_name, GPNode, output_dim, param_dict, name=name)
    def get_X(self, X_p_X_l):
        X_p = X_p_X_l[:, 0].reshape(-1,1)
        X_l = X_p_X_l[:, 1:]
        X_geos, _, _ = self.geometry.map(self.geometry.get_params(), phases=X_p)
        return np.hstack([X_geos, X_l])

    def f_phases(self):
        return self.kern.K(self.Y) @ self.K_YY_inv @ self.X_phases_X_l

    def f_phases_new(self, Y_new, pred_group=0, **kwargs):
        f_p = self.f_phases()

        if pred_group == 1:
            K_K_inv = self.kern1.K(Y_new, self.Y1) @ self.K_YY_inv1
        elif pred_group == 2:
            K_K_inv = self.kern2.K(Y_new, self.Y2) @ self.K_YY_inv2
        else:
            K_K_inv = self.kern_0.K(Y_new, self.Y) @ self.K_YY_inv

        return K_K_inv @ f_p
    def f(self):
        X_p_X_l = self.f_phases()
        return self.get_X(X_p_X_l)

    def f_new(self, Y_new, pred_group=0, **kwargs):
        X_p_X_l = self.f_phases_new(Y_new, pred_group=pred_group)
        return self.get_X(X_p_X_l)

    def construct_mapping(self):
        X_l = self.X[:,3:]
        self.X_phases_X_l = np.hstack([self.Y_phases, X_l])
        kern = GPy.kern.RBF(self.input_dim, ARD=False)
        self.GP = GPy.models.GPRegression(self.Y, self.X_phases_X_l, kern)
        self.GP.optimize(messages=True, max_f_eval=1000)
        self.kern = GPy.kern.RBF(self.input_dim, ARD=self.param_dict['ARD'],
                                 variance=kern.variance.values,
                                 lengthscale=kern.lengthscale.values)#np.ones(self.input_dim)*
        self.kern_0 = self.kern.copy()
        self.K_YY = self.kern.K(self.Y)
        self.K_YY_inv = inverse(self.K_YY+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)

        if self.param_dict['ARD']:
            ls1 = self.kern_0.lengthscale.values[self.param_dict['Y1 indices']]
            ls2 = self.kern_0.lengthscale.values[self.param_dict['Y2 indices']]
        else:
            ls1 = self.kern_0.lengthscale.values
            ls2 = self.kern_0.lengthscale.values


        self.kern1 = GPy.kern.RBF(self.Y1.shape[1], variance=self.kern_0.variance.values, lengthscale=ls1,ARD=self.param_dict['ARD'],active_dims=self.param_dict['Y1 indices'])
        self.K_YY1 = self.kern1.K(self.Y)
        self.K_YY_inv1 = inverse(self.K_YY1+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)

        self.kern2 = GPy.kern.RBF(self.Y2.shape[1], variance=self.kern_0.variance.values, lengthscale=ls2,ARD=self.param_dict['ARD'],active_dims=self.param_dict['Y2 indices'])
        self.K_YY2 = self.kern2.K(self.Y)
        self.K_YY_inv2 = inverse(self.K_YY2+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)
    
    def update_gradients(self, dL_dX):
        dL_dK = dL_dX @ (self.K_YY_inv @ self.X).T
        #dL_dK = dL_dX @ self.X.T @ self.K_YY_inv
        self.kern.update_gradients_full(dL_dK, self.Y)
        dL_dY = self.kern.gradients_X(dL_dK, self.Y)
        dL_dXout = (self.kern.K(self.Y) @ self.K_YY_inv).T @ dL_dX
        return dL_dXout"""

"""def update_gradients(self, dL_dX):
        dL_dX_geo = dL_dX[:,:3]
        dL_dX_lat = dL_dX[:, 3:]

        dL_dK = dL_dX_lat @ (self.K_YY_inv @ self.X[:, 3:]).T
        self.kern.update_gradients_full(dL_dK, self.Y)
        dL_dY = self.kern.gradients_X(dL_dK, self.Y)
        dL_dXout = (self.kern.K(self.Y) @ self.K_YY_inv).T @ dL_dX_lat
        #self.A.gradient = np.sum((self.kern.K(self.Y) @ self.K_YY_inv @ self.X).T @ dL_dX, 0)

        # print(self.kern.lengthscale)
        self.geometry.update_gradients(dL_dX_geo, self.Y_phases)

    def get_X_geo(self,Y_phases):
            X_geos, _, _ = self.geometry.map(self.geometry.get_params(),phases = Y_phases)
            return X_geos

    def get_Xs(self, Y_phases):
        X_geos = self.get_X_geo(Y_phases)
        return X_geos, self.X[:, X_geos.shape[1]:]

    def get_Y_phases(self, Y_new, pred_group=0):
        f_geo = self.kern_geo.K(self.Y) @ self.K_YY_inv_geo @ self.Y_phases

        if pred_group == 1:
            K_K_inv_geo = self.kern1_geo.K(Y_new, self.Y1) @ self.K_YY_inv1_geo
        elif pred_group == 2:
            K_K_inv_geo = self.kern2_geo.K(Y_new, self.Y2) @ self.K_YY_inv2_geo
        else:
            K_K_inv_geo = self.kern_0_geo.K(Y_new, self.Y) @ self.K_YY_inv_geo
        return K_K_inv_geo @ f_geo

    def f(self):
        X_geos, X = self.get_Xs(self.Y_phases)
        self.X = np.hstack([X_geos, X])
        X_new =self.kern.K(self.Y) @ self.K_YY_inv @ X
        return np.hstack([X_geos, X_new])

    def f_new(self, Y_new, pred_group=0, **kwargs):
        Y_phases_new = self.get_Y_phases(Y_new)
        X_geos, X = self.get_Xs(Y_phases_new)
        f = self.kern.K(self.Y) @ self.K_YY_inv @ X


        if pred_group == 1:
            K_K_inv = self.kern1.K(Y_new, self.Y1) @ self.K_YY_inv1
        elif pred_group == 2:
            K_K_inv = self.kern2.K(Y_new, self.Y2) @ self.K_YY_inv2
        else:
            K_K_inv = self.kern_0.K(Y_new, self.Y) @ self.K_YY_inv

        f_geo = self.kern_geo.K(self.Y) @ self.K_YY_inv_geo @ X_geos

        if pred_group == 1:
            K_K_inv_geo = self.kern1_geo.K(Y_new, self.Y1) @ self.K_YY_inv1_geo
        elif pred_group == 2:
            K_K_inv_geo = self.kern2_geo.K(Y_new, self.Y2) @ self.K_YY_inv2_geo
        else:
            K_K_inv_geo = self.kern_0_geo.K(Y_new, self.Y) @ self.K_YY_inv_geo

        return np.hstack([X_geos,K_K_inv @ f])


    def construct_mapping(self):
        X_geos, X = self.get_Xs(self.Y_phases)
        #self.kern = GPy.kern.RBF(self.input_dim, ARD=self.param_dict['ARD'])
        X_dim = X.shape[1]
        kern = GPy.kern.RBF(self.input_dim, ARD=False)
        self.GP = GPy.models.GPRegression(self.Y, X, kern)
        self.GP.optimize(messages=True, max_f_eval=1000)
        self.kern = GPy.kern.RBF(self.input_dim, ARD=self.param_dict['ARD'],
                                 variance=kern.variance.values,
                                 lengthscale=kern.lengthscale.values)#np.ones(self.input_dim)*
        self.kern_0 = self.kern.copy()
        self.K_YY = self.kern.K(self.Y)
        self.K_YY_inv = inverse(self.K_YY+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)

        if self.param_dict['ARD']:
            ls1 = self.kern_0.lengthscale.values[self.param_dict['Y1 indices']]
            ls2 = self.kern_0.lengthscale.values[self.param_dict['Y2 indices']]
        else:
            ls1 = self.kern_0.lengthscale.values
            ls2 = self.kern_0.lengthscale.values


        self.kern1 = GPy.kern.RBF(self.Y1.shape[1], variance=self.kern_0.variance.values, lengthscale=ls1,ARD=self.param_dict['ARD'],active_dims=self.param_dict['Y1 indices'])
        self.K_YY1 = self.kern1.K(self.Y)
        self.K_YY_inv1 = inverse(self.K_YY1+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)

        self.kern2 = GPy.kern.RBF(self.Y2.shape[1], variance=self.kern_0.variance.values, lengthscale=ls2,ARD=self.param_dict['ARD'],active_dims=self.param_dict['Y2 indices'])
        self.K_YY2 = self.kern2.K(self.Y)
        self.K_YY_inv2 = inverse(self.K_YY2+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)


        # ---------- X_geos

        kern_geo = GPy.kern.RBF(self.input_dim, ARD=False)
        self.GP_geo = GPy.models.GPRegression(self.Y, self.Y_phases, kern_geo)
        self.GP_geo.optimize(messages=True, max_f_eval=1000)
        self.kern_geo = GPy.kern.RBF(self.input_dim, ARD=self.param_dict['ARD'],
                                 variance=kern_geo.variance.values,
                                 lengthscale=kern_geo.lengthscale.values)  # np.ones(self.input_dim)*
        self.kern_0_geo = self.kern_geo.copy()
        self.K_YY_geo = self.kern_geo.K(self.Y)
        self.K_YY_inv_geo = inverse(self.K_YY_geo + np.eye(self.K_YY_geo.shape[0]) * self.GP_geo.Gaussian_noise.variance.values)

        if self.param_dict['ARD']:
            ls1_geo = self.kern_0_geo.lengthscale.values[self.param_dict['Y1 indices']]
            ls2_geo = self.kern_0_geo.lengthscale.values[self.param_dict['Y2 indices']]
        else:
            ls1_geo = self.kern_0_geo.lengthscale.values
            ls2_geo = self.kern_0_geo.lengthscale.values

        self.kern1_geo = GPy.kern.RBF(self.Y1.shape[1], variance=self.kern_0_geo.variance.values, lengthscale=ls1_geo,
                                  ARD=self.param_dict['ARD'], active_dims=self.param_dict['Y1 indices'])

        self.K_YY1_geo = self.kern1_geo.K(self.Y)
        self.K_YY_inv1_geo = inverse(self.K_YY1_geo + np.eye(self.K_YY_geo.shape[0]) * self.GP_geo.Gaussian_noise.variance.values)

        self.kern2_geo = GPy.kern.RBF(self.Y2.shape[1], variance=self.kern_0_geo.variance.values, lengthscale=ls2_geo,
                                  ARD=self.param_dict['ARD'], active_dims=self.param_dict['Y2 indices'])
        self.K_YY2_geo = self.kern2_geo.K(self.Y)
        self.K_YY_inv2_geo = inverse(self.K_YY2_geo + np.eye(self.K_YY_geo.shape[0]) * self.GP_geo.Gaussian_noise.variance.values)'''
"""