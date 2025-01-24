import numpy as np
from GPy import kern
from HGPLVM.hgp_model import HGPLVM
from HGPLVM.GPLVM_node import GPLVM_node as GPNode
from HGPLVM.architectures.base import HGP_arch
from dtw import *

class X1_H2_Y2(HGP_arch):
    def __init__(self, HGP, data_set_class, base_arch=None):
        self.data_set_class = data_set_class
        super(X1_H2_Y2,self).__init__(HGP, data_set_class, base_arch)

    def set_up(self,data_set_class):
        super().set_up(data_set_class)
        if self.priors == None or self.in_dims == None:
            raise ValueError ('Priors and input dimensions must be set before building this architecture.')
        if self.kernels == None:
            k0 = kern.RBF(self.in_dims[0], 1, ARD=False) + kern.Linear(self.in_dims[0],ARD=False) + kern.Bias(self.in_dims[0], np.exp(-2))
            k1 = kern.RBF(self.in_dims[1], 1, ARD=False) + kern.Linear(self.in_dims[1], ARD=False) + kern.Bias(self.in_dims[1], np.exp(-2))
            k2 = kern.RBF(self.in_dims[2], 1, ARD=False) + kern.Linear(self.in_dims[2], ARD=False) + kern.Bias(self.in_dims[2], np.exp(-2))
            self.set_kernels([k0,k1,k2])
        if self.inits == None:
            self.set_initializations(['kernel pca:rbf','kernel pca:rbf','kernel pca:rbf'])
        if self.opts == None:
            self.set_optimizers(['lbfgsb','lbfgsb'])



        X1_Y1 = GPNode(self.Ys[0], self.in_dims[1], num_inducing=self.num_inducing[1], kernel=self.kernels[1],
                       seq_eps=self.data_set_class.seq_eps_train)
        #X1_Y1.set_backconstraint(BC_dict=self.BC_dict)
        X1_Y1.initialize_X(self.inits[1],Y=self.Ys[0])
        self.GPNode_init_opt(X1_Y1, self.opts[1], self.max_iters)


        X2_Y2 = GPNode(self.Ys[1], self.in_dims[2],num_inducing=self.num_inducing[2], kernel=self.kernels[2],
                       seq_eps=self.data_set_class.seq_eps_train)
        X2_Y2.initialize_X(self.inits[2],Y=self.Ys[1])
        #X2_Y2.set_backconstraint(BC_dict=self.BC_dict)
        self.GPNode_init_opt(X2_Y2, self.opts[1], self.max_iters)


        X1X2 = np.hstack([X1_Y1.X.values,X2_Y2.X.values])
        top_node = GPNode(X1X2, self.in_dims[0],num_inducing=self.num_inducing[0], kernel=self.kernels[0],
                          seq_eps=self.data_set_class.seq_eps_train)
        top_node.SetChild(0, X1_Y1)
        top_node.SetChild(1, X2_Y2)
        top_node.set_backconstraint(BC_dict=self.BC_dict)
        top_node.initialize_X(self.inits[0],Y=X1X2)
        top_node.set_prior(prior_dict=self.priors[0], num_seqs=self.data_set_class.sub_num_actions)



        self.GPNode_init_opt(top_node, self.opts[0], self.max_iters)
        self.model = HGPLVM(top_node, self)


    def set_Y(self, data_set_class):
        self.Y = data_set_class.Y_train
        self.BC_dict['Y'] = self.Y
        self.N, self.D = self.Y.shape
        self.indices_list = data_set_class.indices_list
        Ys = self.split_Y(self.Y)
        self.Ys = Ys

    def split_Y(self, Y):
        Ys = []
        for inds in self.indices_list:
            Ys.append(Y[:, inds])
        return Ys


class X1_H1_Y1(HGP_arch):
    def __init__(self,HGP, data_set_class, base_arch=None):
        super(X1_H1_Y1, self).__init__(HGP, base_arch)
        self.set_up(data_set_class)

    def set_up(self,data_set_class):
        self.data_set_class = data_set_class
        Y = self.data_set_class.Y_train
        if self.priors == None or self.in_dims == None:
            raise ValueError ('Priors and input dimensions must be set before building this architecture.')
        if self.kernels == None:
            k1 = kern.RBF(self.in_dims[1], 1, ARD=False) + kern.Linear(self.in_dims[1], ARD=False) + kern.Bias(self.in_dims[1], np.exp(-2))
            k0 = kern.RBF(self.in_dims[0], 1, ARD=False) + kern.Linear(self.in_dims[0], ARD=False) + kern.Bias(self.in_dims[0], np.exp(-2))
            self.set_kernels([k0, k1])
        if self.inits == None:
            self.set_initializations(['random', 'random'])
        if self.opts == None:
            self.set_optimizers(['lbfgsb', 'lbfgsb'])
        if self.BCs == None:
            self.set_backconstraints(['none','none'])

        X_Y = GPNode(self.data_mgmt.Y_train_list[0], self.in_dims[1],num_inducing=self.num_inducing[1], kernel=self.kernels[1], max_iters=self.max_iters, seq_eps=self.data_mgmt.seq_eps_train, optimizer=self.opts[1])
        X_Y.initialize_X(self.inits[1])
        X_Y.set_backconstraint(self.data_mgmt.Y_train_list[0], name=self.BCs[1])
        X_Y.set_prior(prior_dict=self.priors[1], num_seqs=self.data_mgmt.num_seqs)
        self.GPNode_init_opt(X_Y,self.opts[1],self.max_iters)

        top_node = GPNode(X_Y.X.values, self.in_dims[0],num_inducing=self.num_inducing[0], kernel=self.kernels[0], max_iters=self.max_iters, seq_eps=self.data_mgmt.seq_eps_train, optimizer=self.opts[0])
        top_node.SetChild(0, X_Y)
        top_node.initialize_X(self.inits[0])
        top_node.set_backconstraint(X_Y.X.values, name=self.BCs[0])
        top_node.set_prior(prior_dict=self.priors[0], num_seqs=self.data_mgmt.num_seqs)
        self.GPNode_init_opt(top_node, self.opts[0], self.max_iters)

        self.model = HGPLVM(top_node, self)