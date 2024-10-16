from HGPLVM.backconstraints.backconstraints_base import BC_Base
import GPy
from HGPLVM.global_functions import inverse
from GPy.core import Param
import numpy as np
import copy
from GPy.core import Param

class No_BC(BC_Base):
    """
    """
    def __init__(self, GPNode, output_dim, param_dict, name='No_BC'):
        super().__init__(GPNode, output_dim, param_dict, name)
        self.first_call = True


    def initialize_Y(self):
        self.Y = self.GPNode.Y
        self.Y1 = self.Y[:, self.param_dict['Y1 indices']]
        self.Y2 = self.Y[:, self.param_dict['Y2 indices']]

    def get_X(self):
        return self.X


    def f(self):
        return self.GPNode.X.values.copy()#.copy()#.values#.copy()#.copy().values.copy()

    def BC_f_new(self, Y_new, pred_group=0, **kwargs):
        if self.param_dict['mapping'] == 'BC':
            raise ValueError('No BC has no BC mapping. Using GP Map or GPLVM map')

        return self.f_new(Y_new, pred_group=pred_group, **kwargs)


    def update_gradients(self, dL_dX):
        return dL_dX

    def construct_mapping(self):
        pass

    def link_params(self):
        #self.link_parameter(Param('X', self.X.copy()))
        pass

    def set_X(self, X, HGP, GPNode):
        if HGP is not None:
            HGP.unlink_parameter(X)
        X = Param('latent mean',
                  self.f())
        if GPNode.prior is not None:
            X.set_prior(GPNode.prior)
        if HGP is not None:
            HGP.link_parameter(X,0)
        return X