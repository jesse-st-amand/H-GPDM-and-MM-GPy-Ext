from HGPLVM.backconstraints.backconstraints_base import BC_Base
from HGPLVM.global_functions import inverse
import numpy as np
from GPy.core import Param
import GPy
class Linear_BC(BC_Base):
    """
    """

    def __init__(self, GPNode, output_dim, param_dict, name='linear geometric base'):
        super().__init__(GPNode, output_dim, param_dict, name)
        self.first_call = True

    def initialize_Y(self):
        self.Y = self.GPNode.Y
        self.Y1 = self.Y[:, self.param_dict['Y1 indices']]
        self.Y2 = self.Y[:, self.param_dict['Y2 indices']]

    def get_X(self):
        return self.X

    def get_Y_new(self,Y_in):
        Y = Y_in
        return Y

    def BC_f_new(self, Y_new, pred_group=0, **kwargs):
        if pred_group == 0:
            return Y_new @ self.A
        else:
            raise ValueError('Lin BC has no pred var groups 1 and 2. Use GP map.')


    def initialize_A(self):
        kern = GPy.kern.Linear(self.Y.shape[0], ARD=False) + GPy.kern.White(self.Y.shape[0],variance=1000)
        A = inverse(kern.K(self.Y.T)) @ self.Y.T @ self.get_X() #moore-penrose inverse with white noise for compensation of dimensions constant over time
        self.A = Param('A', A)


    def f(self):
        return self.Y @ self.A


    def update_gradients(self, dL_dX):
        self.A.gradient = (dL_dX.T @ self.Y).T
        return dL_dX

    def construct_mapping(self):
        pass


class Linear_X_BC(BC_Base):
    """
    """

    def __init__(self, GPNode, output_dim, param_dict, name='linear x geometric base'):
        super().__init__(GPNode, output_dim, param_dict, name)
        self.first_call = True

    def initialize_Y(self):
        self.Y = self.GPNode.Y
        self.Y1 = self.Y[:, self.param_dict['Y1 indices']]
        self.Y2 = self.Y[:, self.param_dict['Y2 indices']]

    def get_X(self):
        return self.X

    def get_Y_new(self,Y_in):
        Y = Y_in
        return Y

    def BC_f_new(self, Y_new, pred_group=0, **kwargs):
        raise ValueError('Lin x BC has no predictor. Use GP map.')


    def initialize_A(self):
        A = np.eye(self.X.shape[1]) #moore-penrose inverse with white noise for compensation of dimensions constant over time
        self.A = Param('A', A)


    def f(self):
        return self.X @ self.A


    def update_gradients(self, dL_dX):
        self.A.gradient = (dL_dX.T @ self.X).T
        return dL_dX

    def construct_mapping(self):
        pass