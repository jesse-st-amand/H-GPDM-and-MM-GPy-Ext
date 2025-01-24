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
        self.geometry.update_gradients(dL_dXin, self.Y_phases)



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