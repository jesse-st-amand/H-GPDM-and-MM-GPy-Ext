import logging
from GPy.core.mapping import Mapping
import GPy
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from GPy.core import Param


logger = logging.getLogger("GP")
class BC_Base(Mapping):
    """
    """
    def __init__(self, GPNode, output_dim, param_dict, name=''):
        self.GPNode = GPNode
        self.param_dict = param_dict
        self.output_dim = output_dim
        self.initialize_Y()
        self.N, self.input_dim = self.Y.shape
        self.initialize_X(self.param_dict['X_init'] )
        super(BC_Base, self).__init__(input_dim=self.input_dim, output_dim=output_dim, name=name)
        self.construct_mapping()
        self.initialize_A()
        self.link_params()
        self.constrain_params()
        self.first_calls = [True,True,True]
        self.GPs = [None, None, None]
        self.models = [None, None, None]
    def initialize_Y(self):
        if 'Y' in self.param_dict:
            self.Y = self.param_dict['Y']
        else:
            self.Y = self.GPNode.Y

        self.Y1 = self.Y[:, self.param_dict['Y1 indices']]
        self.Y2 = self.Y[:, self.param_dict['Y2 indices']]

    def initialize_X(self, X_init = None):
        if X_init is None:
            self.X = self.GPNode.init_embedding(embed_type=self.GPNode.attr_dict['init'], Y=self.Y)
        else:
            self.X = X_init


    def f_new(self, Y_new, pred_group, **kwargs):
        # Validate and select the appropriate data based on the pred_group
        X_group = self.GPNode.X.values
        if pred_group == 0:
            Y_group = self.Y
        elif pred_group == 1:
            Y_group = self.Y1
        elif pred_group == 2:
            Y_group = self.Y2
        else:
            raise ValueError("Invalid prediction group.")

        # Determine which model to use based on the mapping type
        if self.param_dict['mapping'] == 'GP':
            # Initialize GP model if first call
            if self.first_calls[pred_group]:
                kern = GPy.kern.RBF(Y_group.shape[1], ARD=False)
                self.models[pred_group] = GPy.models.GPRegression(Y_group, X_group, kern)
                self.models[pred_group].optimize(messages=True, max_f_eval=1000)
                self.first_calls[pred_group] = False
            # Predict from the GP model
            return self.models[pred_group].predict_noiseless(Y_new)[0]

        elif self.param_dict['mapping'] == 'RF':
            # Initialize RF model if first call
            if self.first_calls[pred_group]:
                self.models[pred_group] = RandomForestRegressor(n_estimators=100, random_state=42)
                self.models[pred_group].fit(Y_group, X_group)
                self.first_calls[pred_group] = False
            # Predict from the RF model
            return self.models[pred_group].predict(Y_new)  # Ensure Y_new is properly shaped for prediction

        elif self.param_dict['mapping'] == 'BC':
            return self.BC_f_new(Y_new, pred_group, **kwargs)
        elif self.param_dict['mapping'] == 'GPLVM':
            X, modX = self.GPNode.infer_newX(Y_new, optimize=True)
            return X.values
        else:
            raise ValueError('Wrong mapping type.')

    def initialize_A(self):
        pass

    def construct_mapping(self):
        pass

    def link_params(self):
        pass

    def constrain_params(self):
        pass

    def set_X(self, X, HGP, GPNode):
        X_grads = GPNode.X.gradient
        if HGP is not None:
            HGP.unlink_parameter(X)
        X = Param('latent mean',
                       self.f())
        X.gradient = X_grads
        if GPNode.prior is not None:
            X.set_prior(GPNode.prior)
        if HGP is not None:
            HGP.link_parameter(X,0)
            HGP.set_updates(on=False)
            X.fix()
            HGP.set_updates(on=True)
        return X

