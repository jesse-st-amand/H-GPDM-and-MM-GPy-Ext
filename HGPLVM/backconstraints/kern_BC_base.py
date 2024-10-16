from HGPLVM.backconstraints.backconstraints_base import BC_Base
import GPy
from HGPLVM.global_functions import inverse
from GPy.core import Param
import numpy as np

class Kernel_BC(BC_Base):
    """
    """
    def __init__(self, GPNode, output_dim, param_dict, name='Kernel_BC'):
        super(Kernel_BC, self).__init__(GPNode, output_dim, param_dict, name)
        self.first_call = True


    def initialize_Y(self):
        self.Y = self.GPNode.Y
        self.Y1 = self.Y[:, self.param_dict['Y1 indices']]
        self.Y2 = self.Y[:, self.param_dict['Y2 indices']]

    def get_X(self):
        return self.X

    def get_Y_new(self,Y_in):
        return Y_in

    def f(self):
        return self.K_Y @ self.A

    def BC_f_new(self, Y_new, pred_group=0, **kwargs):
        if pred_group == 0:
            K_Y_new_old = self.kern.K(Y_new, self.Y)
            return K_Y_new_old @ self.A
        else:
            raise ValueError('Kern BC has no pred var groups 1 and 2. Use GP map.')



    def initialize_A(self):
        self.A = Param('A', inverse(self.K_Y) @ self.X)
        #self.A.fix()


    def update_gradients(self, dL_dX):
        #print(self.GPNode.HGPLVM.A.gradient[:3,:3])
        #print(self.GPNode.HGPLVM.A.values[:3, :3])
        self.A.gradient = self.K_Y @ dL_dX
        return dL_dX

    def construct_mapping(self):
        #self.Y = self.Y
        kern = GPy.kern.RBF(self.Y.shape[1], ARD=False)
        self.GP = GPy.models.GPRegression(self.Y, self.X, kern)
        self.GP.optimize(messages=True, max_f_eval=1000)
        self.kern = self.GP.kern
        self.K_Y = self.kern.K(self.Y)


    def link_params(self):
        self.link_parameter(self.kern)
        self.link_parameter(self.A)

'''class Kernel_X_BC(BC_Base):
    """
    """
    def __init__(self, GPNode, output_dim, param_dict, name='Kernel_BC'):
        super(Kernel_BC, self).__init__(GPNode, output_dim, param_dict, name)
        self.first_call = True


    def initialize_Y(self):
        self.Y = self.GPNode.Y
        self.Y1 = self.Y[:, self.param_dict['Y1 indices']]
        self.Y2 = self.Y[:, self.param_dict['Y2 indices']]

    def get_X(self):
        return self.X

    def get_Y_new(self,Y_in):
        return Y_in

    def f(self):
        return self.K_Y @ self.A

    def BC_f_new(self, Y_new, pred_group=0, **kwargs):
        if pred_group == 0:
            K_Y_new_old = self.kern.K(Y_new, self.Y)
            return K_Y_new_old @ self.A
        else:
            raise ValueError('Kern BC has no pred var groups 1 and 2. Use GP map.')



    def initialize_A(self):
        self.A = Param('A', inverse(self.K_Y) @ self.X)


    def update_gradients(self, dL_dX):

        self.A.gradient = self.K_Y @ dL_dX
        return dL_dX

    def construct_mapping(self):
        self.Y = self.GPNode.Y
        kern = GPy.kern.RBF(self.input_dim, ARD=False)
        self.GP = GPy.models.GPRegression(self.Y, self.X, kern)
        self.GP.optimize(messages=True, max_f_eval=1000)
        self.kern = self.GP.kern
        self.K_Y = self.kern.K(self.Y)


    def link_params(self):
        self.link_parameter(self.kern)
        self.link_parameter(self.A)'''