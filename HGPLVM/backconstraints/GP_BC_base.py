from HGPLVM.backconstraints.backconstraints_base import BC_Base
import GPy
from HGPLVM.global_functions import inverse
import numpy as np
from GPy.core import Param
class GP_BC(BC_Base):
    """
    """

    def __init__(self, GPNode, output_dim, param_dict, name='GP BC'):
        super(GP_BC, self).__init__(GPNode, output_dim, param_dict, name)

    def get_X(self):
        return self.X

    def get_Y_new(self,Y_in):
        return Y_in

    def f(self):
        self.X = self.get_X()
        return self.kern.K(self.Y) @ self.K_YY_inv @ self.X

    '''def f_new(self, Y_in,**kwargs):
        Y_new = self.get_Y_new(Y_in)
        #return self.kern.K(Y_new,self.Y) @self.K_YY_inv@self.X
        f = self.f()
        return self.kern_0.K(Y_new, self.Y) @ self.K_YY_inv @ f'''

    def initialize_A(self):
        pass

    def update_gradients(self, dL_dX):
        dL_dK = dL_dX @ (self.K_YY_inv @ self.X).T
        #dL_dK = dL_dX @ self.X.T @ self.K_YY_inv
        self.kern.update_gradients_full(dL_dK, self.Y)
        dL_dY = self.kern.gradients_X(dL_dK, self.Y)
        dL_dXout = (self.kern.K(self.Y) @ self.K_YY_inv).T @ dL_dX
        return dL_dXout

    def set_kern(self,ARD=False):
        return GPy.kern.RBF(self.input_dim, ARD=ARD)

    def construct_mapping(self):
        self.kern = self.set_kern(self.param_dict['ARD']) #+ GPy.kern.Linear(self.input_dim+2, ARD=False)#+ GPy.kern.Bias(self.input_dim+2, np.exp(-2))
        GP = GPy.models.GPRegression(self.Y, self.X, self.kern)
        GP.optimize(messages=True, max_f_eval=1000)
        self.kern_0 = self.kern.copy()
        self.K_YY = self.kern.K(self.Y)
        self.K_YY_inv = inverse(self.K_YY+ np.eye(self.K_YY.shape[0])*GP.Gaussian_noise.variance.values)
        #print((self.kern.K(self.Y) @ self.K_YY_inv)[:5, 0])
        ### if optimized kernel is different from BC mapping kernel
        #self.kern = GPy.kern.RBF(new_input_dim, kern.variance,lengthscale=kern.lengthscale, ARD=False)#
        ###

    def link_params(self):
        self.link_parameter(self.kern)

    def constrain_params(self):
        for con in self.param_dict['constraints']:
            if con.lower() == 'variance':
                self.kern.variance.constrain_fixed()
            if con.lower() == 'lengthscale':
                self.kern.lengthscale.constrain_fixed()


class sparse_GP_BC(GP_BC):

    def __init__(self, GPNode, output_dim, param_dict, name='GP BC'):
        super().__init__(GPNode, output_dim, param_dict, name)
        self.initialize_Z()

    def f(self):
        self.X = self.get_X()
        return self.kern.K(self.Y_sub,self.Y) @ self.K_YY_inv @ self.X

    def construct_mapping(self):
        i = np.arange(0, self.GPNode.N, int(self.GPNode.N / self.GPNode.num_inducing))
        i = np.arange(0, self.GPNode.num_inducing, int(1))
        self.Y_sub = self.Y.view(np.ndarray)[i].copy()#np.eye(self.GPNode.num_inducing,self.Y.shape[1])#
        super().construct_mapping()

    def update_gradients(self, dL_dZ):
        dL_dK = dL_dZ @ (self.K_YY_inv @ self.X).T
        self.kern.update_gradients_full(dL_dK, self.Y_sub, self.Y)
        dL_dY = self.kern.gradients_X(dL_dK, self.Y_sub, self.Y)
        dL_dXout = (self.kern.K(self.Y_sub, self.Y) @ self.K_YY_inv).T@dL_dZ
        return dL_dXout

    def initialize_Z(self):
        self.Z = self.f()
        self.GPNode.set_Z(self.Z)

class weighted_GP_BC(GP_BC):
    """
    """

    def __init__(self, GPNode, output_dim, param_dict, name='w GP BC'):
        super().__init__(GPNode, output_dim, param_dict, name)

    def f(self):
        self.X = self.get_X()
        #print(self.X[:3,:3])
        return self.kern.K(self.Y) @ self.K_YY_inv @ self.X @ np.diag(self.A)

    '''def f_new(self, Y_in):
        Y_new = self.get_Y_new(Y_in)
        return self.kern.K(Y_new, self.Y) @ self.K_YY_inv @ self.X @ np.diag(self.A)'''

    def update_gradients(self, dL_dX):
        '''if self.GPNode.prior is not None:
            dL_dX = self.GPNode.prior.lnpdf_grad(self.X).reshape(self.X.shape)
        else:
            dL_dX_prior = dL_dX'''

        dL_dK = dL_dX @ (self.K_YY_inv @ self.X @ np.diag(self.A)).T
        self.kern.update_gradients_full(dL_dK, self.Y)
        dL_dY = self.kern.gradients_X(dL_dK, self.Y)
        dL_dXout = (self.kern.K(self.Y) @ self.K_YY_inv).T @ dL_dX @ np.diag(self.A).T
        self.A.gradient = np.sum((self.kern.K(self.Y) @ self.K_YY_inv @ self.X).T @ dL_dX,0)

        #print(self.kern.lengthscale)
        return dL_dXout

    def set_kern(self, ARD=False):
        return GPy.kern.RBF(self.input_dim, ARD=ARD)

    def construct_mapping(self):
        self.kern = self.set_kern(self.param_dict['ARD'])
        self.GP = GPy.models.GPRegression(self.Y, self.X, self.kern)
        self.GP.optimize(messages=True, max_f_eval=1000)
        self.K_YY = self.kern.K(self.Y)
        self.K_YY_inv = inverse(self.K_YY+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)

        ### if optimized kernel is different from BC mapping kernel
        # self.kern = GPy.kern.RBF(new_input_dim, kern.variance,lengthscale=kern.lengthscale, ARD=False)#
        ###

    def link_params(self):
        self.link_parameter(self.A)
        super().link_params()


    def initialize_A(self):
        self.A = Param('A', np.ones(self.output_dim))
        pass
    def constrain_params(self):
        for con in self.param_dict['constraints']:
            if con == 'A':
                self.A.constrain_fixed()
        super().constrain_params()

class Multi_Weighted_GP_BC(weighted_GP_BC):

    def __init__(self, GPNode, output_dim, param_dict, name='multi w GP BC'):
        super().__init__(GPNode, output_dim, param_dict, name)



    '''def f_new(self, Y_new, pred_group=0, **kwargs):

        f = self.f()
        self.f_new_called = True
        if pred_group == 1:
            K_K_inv = self.kern1.K(Y_new, self.Y1) @ self.K_YY_inv1
        elif pred_group == 2:
            K_K_inv = self.kern2.K(Y_new, self.Y2) @ self.K_YY_inv2
        else:
            K_K_inv = self.kern_0.K(Y_new, self.Y) @ self.K_YY_inv

        return K_K_inv @ f'''

    '''def f_new(self, Y_new, pred_group=0, **kwargs):
        if pred_group == 0:
            f = self.f()
            K_K_inv = self.kern_0.K(Y_new, self.Y) @ self.K_YY_inv
            return K_K_inv @ f
        elif pred_group == 1:
            if self.first_call:
                kern = GPy.kern.RBF(self.Y1.shape[1], ARD=False)
                self.GP1 = GPy.models.GPRegression(self.Y1, self.GPNode.X.values, kern)
                self.GP1.optimize(messages=True, max_f_eval=1000)
                self.first_call = False
            return self.GP1.predict_noiseless(Y_new)[0]'''

    def BC_f_new(self, Y_new, pred_group=0, **kwargs):

        f = self.f()
        self.f_new_called = True
        if pred_group == 1:
            K_K_inv = self.kern1.K(Y_new, self.Y1) @ self.K_YY_inv1
        elif pred_group == 2:
            K_K_inv = self.kern2.K(Y_new, self.Y2) @ self.K_YY_inv2
        else:
            K_K_inv = self.kern_0.K(Y_new, self.Y) @ self.K_YY_inv

        return K_K_inv @ f


    def construct_mapping(self):
        #self.kern = GPy.kern.RBF(self.input_dim, ARD=self.param_dict['ARD'])
        kern = GPy.kern.RBF(self.input_dim, ARD=False)
        self.GP = GPy.models.GPRegression(self.Y, self.X, kern, noise_var=1)
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
        """self.kern1 = GPy.kern.RBF(self.Y1.shape[1], ARD=False)
        self.GP1 = GPy.models.GPRegression(self.Y1, self.X, self.kern1)
        self.GP1.optimize(messages=True, max_f_eval=1000)"""
        self.K_YY1 = self.kern1.K(self.Y)
        self.K_YY_inv1 = inverse(self.K_YY1+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)

        self.kern2 = GPy.kern.RBF(self.Y2.shape[1], variance=self.kern_0.variance.values, lengthscale=ls2,ARD=self.param_dict['ARD'],active_dims=self.param_dict['Y2 indices'])
        self.K_YY2 = self.kern2.K(self.Y)
        self.K_YY_inv2 = inverse(self.K_YY2+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)



class N_weighted_GP_BC(GP_BC):
    """
    """

    def __init__(self, GPNode, output_dim, param_dict, name='w GP BC'):
        super().__init__(GPNode, output_dim, param_dict, name)

    def f(self):
        self.X = self.get_X()
        return np.diag(self.A) @ self.kern.K(self.Y) @ self.K_YY_inv @ self.X

    def f_new(self, Y_in):
        Y_new = self.get_Y_new(Y_in)
        return np.diag(self.A) @ self.kern.K(Y_new, self.Y) @ self.K_YY_inv @ self.X

    def update_gradients(self, dL_dX):
        dL_dK = np.diag(self.A).T @ dL_dX @ (self.K_YY_inv @ self.X).T
        self.kern.update_gradients_full(dL_dK, self.Y)
        dL_dY = self.kern.gradients_X(dL_dK, self.Y)
        dL_dXout = (np.diag(self.A) @ self.kern.K(self.Y) @ self.K_YY_inv).T @ dL_dX
        self.A.gradient = np.sum(dL_dX @ (self.kern.K(self.Y) @ self.K_YY_inv @ self.X).T,axis=1)

        #print(self.kern.lengthscale)
        return dL_dXout

    def set_kern(self, ARD=False):
        return GPy.kern.RBF(self.input_dim, ARD=ARD)

    def construct_mapping(self):
        self.kern = self.set_kern(self.param_dict['ARD'])
        self.GP = GPy.models.GPRegression(self.Y, self.X, self.kern)
        self.GP.optimize(messages=True, max_f_eval=1000)
        self.K_YY = self.kern.K(self.Y)
        self.K_YY_inv = inverse(self.K_YY+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)

        ### if optimized kernel is different from BC mapping kernel
        # self.kern = GPy.kern.RBF(new_input_dim, kern.variance,lengthscale=kern.lengthscale, ARD=False)#
        ###

    def link_params(self):
        self.link_parameter(self.A)
        super().link_params()


    def initialize_A(self):
        self.A = Param('A', np.ones(self.Y.shape[0]))
        pass
    def constrain_params(self):
        for con in self.param_dict['constraints']:
            if con == 'A':
                self.A.constrain_fixed()
        super().constrain_params()

class Multi_N_Weighted_GP_BC(N_weighted_GP_BC):

    def __init__(self, GPNode, output_dim, param_dict, name='multi w GP BC'):
        self.first_call = True
        super().__init__(GPNode, output_dim, param_dict, name)

    def initialize_Y(self):
        if 'Y' in self.param_dict:
            self.Y = self.param_dict['Y']
        else:
            self.Y = self.GPNode.Y

        self.Y1 = self.Y[:, self.param_dict['Y1 indices']]
        self.Y2 = self.Y[:, self.param_dict['Y2 indices']]

    def f_new(self, Y_new, pred_group=0, **kwargs):
        if pred_group == 0:
            f = self.f()
            K_K_inv = self.kern_0.K(Y_new, self.Y) @ self.K_YY_inv
            return K_K_inv @ f
        elif pred_group == 1:
            if self.first_call:
                kern = GPy.kern.RBF(self.Y1.shape[1],ARD=False)
                self.GP1 = GPy.models.GPRegression(self.Y1, self.GPNode.X.values, kern)
                self.GP1.optimize(messages=True, max_f_eval=1000)
                self.first_call = False
            return self.GP1.predict_noiseless(Y_new)[0]

    """def f_new(self, Y_new, pred_group=0, **kwargs):

        f = self.f()
        self.f_new_called = True
        if pred_group == 1:
            K_K_inv = self.kern1.K(Y_new, self.Y1) @ self.K_YY_inv1
        elif pred_group == 2:
            K_K_inv = self.kern2.K(Y_new, self.Y2) @ self.K_YY_inv2
        else:
            K_K_inv = self.kern_0.K(Y_new, self.Y) @ self.K_YY_inv

        return K_K_inv @ f"""


    """def f_new(self, Y, pred_group = 0, **kwargs):
        if self.first_call:
            kern = GPy.kern.RBF(self.Y.shape[1], ARD=False)
            self.GP0 = GPy.models.GPRegression(self.Y, self.GPNode.X.values, kern)
            self.GP0.optimize(messages=True, max_f_eval=1000)

            kern = GPy.kern.RBF(self.Y1.shape[1],ARD=False)
            self.GP1 = GPy.models.GPRegression(self.Y1, self.GPNode.X.values, kern)
            self.GP1.optimize(messages=True, max_f_eval=1000)

            '''kern = GPy.kern.RBF(self.Y2.shape[1], ARD=False)
            self.GP2 = GPy.models.GPRegression(self.Y2, self.GPNode.X.values, kern)
            self.GP2.optimize(messages=True, max_f_eval=1000)
            '''
            self.first_call = False
        if pred_group == 1:
            return self.GP1.predict_noiseless(Y)[0]
        elif pred_group == 2:
            #return self.GP2.predict_noiseless(Y)[0]
            raise NotImplementedError('Remove comments')
        else:
            return self.GP0.predict_noiseless(Y)[0]"""

    def construct_mapping(self):
        #self.kern = GPy.kern.RBF(self.input_dim, ARD=self.param_dict['ARD'])
        kern = GPy.kern.RBF(self.input_dim, ARD=False)
        self.GP = GPy.models.GPRegression(self.Y, self.X, kern, noise_var=1)
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
        """self.kern1 = GPy.kern.RBF(self.Y1.shape[1], ARD=False)
        self.GP1 = GPy.models.GPRegression(self.Y1, self.X, self.kern1)
        self.GP1.optimize(messages=True, max_f_eval=1000)"""
        self.K_YY1 = self.kern1.K(self.Y)
        self.K_YY_inv1 = inverse(self.K_YY1+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)

        self.kern2 = GPy.kern.RBF(self.Y2.shape[1], variance=self.kern_0.variance.values, lengthscale=ls2,ARD=self.param_dict['ARD'],active_dims=self.param_dict['Y2 indices'])
        self.K_YY2 = self.kern2.K(self.Y)
        self.K_YY_inv2 = inverse(self.K_YY2+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)




class kernel_GP_BC(GP_BC):

    def __init__(self, GPNode, output_dim, param_dict, name='multi w GP BC'):
        super().__init__(GPNode, output_dim, param_dict, name)



    def BC_f_new(self, Y_new, pred_group=0, **kwargs):

        f = self.f()
        self.f_new_called = True
        if pred_group == 1:
            K_K_inv = self.kern1.K(Y_new, self.Y1) @ self.K_YY_inv1
        elif pred_group == 2:
            K_K_inv = self.kern2.K(Y_new, self.Y2) @ self.K_YY_inv2
        else:
            K_K_inv = self.kern_0.K(Y_new, self.Y) @ self.K_YY_inv

        return K_K_inv @ f


    def construct_mapping(self):
        kern = GPy.kern.RBF(self.input_dim, ARD=False)
        self.GP = GPy.models.GPRegression(self.Y, self.X, kern, noise_var=1)
        self.GP.optimize(messages=True, max_f_eval=1000)
        self.kern = GPy.kern.RBF(self.input_dim, ARD=self.param_dict['ARD'],
                                 variance=kern.variance.values,
                                 lengthscale=kern.lengthscale.values)#np.ones(self.input_dim)*
        self.kern_0 = self.kern.copy()
        self.K_YY = self.kern.K(self.Y)
        self.K_YY_inv = inverse(self.K_YY+ np.eye(self.K_YY.shape[0])*self.GP.Gaussian_noise.variance.values)

        kern_X = GPy.kern.RBF(self.X.shape[1], ARD=False)
        self.GP = GPy.models.GPRegression(self.X, self.Y, kern_X, noise_var=1)
        self.GP.optimize(messages=True, max_f_eval=1000)
        self.kern = GPy.kern.RBF(self.input_dim, ARD=self.param_dict['ARD'],
                                 variance=kern.variance.values,
                                 lengthscale=kern.lengthscale.values)  # np.ones(self.input_dim)*
        self.kern_0 = self.kern.copy()
        self.K_YY = self.kern.K(self.Y)
        self.K_YY_inv = inverse(self.K_YY + np.eye(self.K_YY.shape[0]) * self.GP.Gaussian_noise.variance.values)

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

    def f(self):
        self.X = self.get_X()
        return self.kern.K(self.Y) @ self.K_YY_inv @ self.X @ np.diag(self.A)


    def update_gradients(self, dL_dX):
        dL_dK = dL_dX @ (self.K_YY_inv @ self.X @ np.diag(self.A)).T
        self.kern.update_gradients_full(dL_dK, self.Y)
        dL_dY = self.kern.gradients_X(dL_dK, self.Y)
        dL_dXout = (self.kern.K(self.Y) @ self.K_YY_inv).T @ dL_dX @ np.diag(self.A).T
        self.A.gradient = np.sum((self.kern.K(self.Y) @ self.K_YY_inv @ self.X).T @ dL_dX,0)

        #print(self.kern.lengthscale)
        return dL_dXout

    def set_kern(self, ARD=False):
        return GPy.kern.RBF(self.input_dim, ARD=ARD)

    def link_params(self):
        self.link_parameter(self.A)
        super().link_params()


    def initialize_A(self):
        self.A = Param('A', np.ones(self.output_dim))

    def constrain_params(self):
        for con in self.param_dict['constraints']:
            if con == 'A':
                self.A.constrain_fixed()
        super().constrain_params()