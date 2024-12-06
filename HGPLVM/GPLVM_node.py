'''
Imports for initialization methods and backconstraints are performed on call within the class for speed
'''
from GPy.core import Param
from HGPLVM.NodeStruct import NodeStruct
import HGPLVM.hgp_model as hgplvm
from paramz import ObsAr
import numpy as np
from HGPLVM.hgp_priors import GPDMM, GPDM#,GPDM_1st_order,GPDM_base,full_GPDM
from HGPLVM.GPDM_priors import GPDM_IG_prior
import random
from GPy.models import GPRegression
from GPy import kern
from GPy.util.linalg import pdinv
from GPy.util import diag


def print_current_seeds():
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    print(f"Current NumPy seed: {numpy_state[1][0]}")
    print(f"Current Python random seed: {python_state[1]}")

def GPLVM_node(Y, attr_dict, **kwargs):
    if attr_dict['num_inducing_latent'] is None or attr_dict['num_inducing_latent'] == 0:
        num_inducing = None
        from HGPLVM.GPy_mods.models.gplvm import GPLVM as GPLVM_class
    elif attr_dict['num_inducing_latent'] > 0: #and isinstance(num_inducing,int):
        from HGPLVM.GPy_mods.models.sparse_gplvm import SparseGPLVM as GPLVM_class
    else:
        raise ValueError('Model must either not have inducing variables, or include an integer number greater than 1.')

    class GPLVMN(GPLVM_class, NodeStruct):
        """
        Establishes the GP node structure. Use with HGPLVM class for optimization
        """

        def __init__(self, Y, attr_dict, input_dim=None, num_inducing=None, X=None, kernel=None, name="gplvm", variance=1,
                     seq_eps=None, num_types_seqs=None,prior=None,backconstraint=None,**kwargs):
            self.attr_dict = attr_dict
            self.HGPLVM = None
            self.node_nID = None
            self.node_lID = None
            self.K_inv = None
            self.W = None
            self.prior = None#prior
            self.gplvm = None
            self.backconstraint = None#backconstraint
            self.GP_Y_X = None
            self.grad_dict = None
            self.gpdm_var = 1
            self.p_indices = []
            self.p_array_indices = []
            self.seq_eps = seq_eps
            self.num_seqs = len(self.seq_eps)
            self.num_types_seqs = num_types_seqs
            self.variance = variance
            if X is None:
                self.X = self.init_embedding('random', Y, input_dim=input_dim)

            if len(self.seq_eps) == 1:
                self.seq_x0s = np.array([0])
            else:
                self.seq_x0s = np.concatenate([np.array([0]), (np.array(self.seq_eps) + 1)[:-1]])
            self.N, self.D = Y.shape[0], input_dim

            if num_inducing is None or num_inducing == 0:
                self.num_inducing = None
                super(GPLVMN, self).__init__(Y, input_dim, X = self.X, kernel=kernel)
            else:
                self.num_inducing = int(num_inducing)
                i = np.arange(0, self.N, int(self.N / self.num_inducing))
                self.Z = self.X.view(np.ndarray)[i].copy()
                super(GPLVMN, self).__init__(Y, input_dim,  X = self.X, Z=self.Z, kernel=kernel, num_inducing=self.num_inducing)

        def init_embedding(self, embed_type, Y, input_dim = None):
            print_current_seeds()
            if input_dim is None:
                input_dim = self.input_dim
            if len(embed_type.split(':')) > 1:
                embed_type, param = embed_type.split(':')
            else:
                param = ''
            if embed_type.lower() == 'None'.lower():
                return
            elif embed_type.lower() == 'BC'.lower():
                return self.backconstraint.f()
            elif embed_type.lower() == 'random projections':
                from HGPLVM.initializers import random_projections
                X = random_projections(Y,input_dim)
            elif embed_type.lower() == 'random'.lower():
                local_random = np.random.RandomState()
                print('Initiallizing with local random state.')
                X = np.asfortranarray(local_random.normal(0, 1, (Y.shape[0], input_dim)))
            elif embed_type.lower() == 'tsne'.lower():
                from sklearn.manifold import TSNE
                # not working
                X = TSNE(n_components=input_dim, learning_rate='auto', init='pca').fit_transform(Y)
            elif embed_type.lower() == 'ica'.lower():
                from sklearn.decomposition import FastICA
                embedding = FastICA(n_components=input_dim)
                X = embedding.fit_transform(Y)
            elif embed_type.lower() == 'mds'.lower():
                from sklearn.manifold import MDS
                embedding = MDS(n_components=input_dim)
                X = embedding.fit_transform(Y)
            elif embed_type.lower() == 'PCA'.lower():
                from sklearn.decomposition import PCA
                transformer = PCA(n_components=input_dim)
                X = transformer.fit_transform(Y)
            elif embed_type.lower() == 'PCA_GPy'.lower():
                from GPy.util.pca import PCA as PCA_GPy
                X = np.asfortranarray(np.random.normal(0, 1, (Y.shape[0], input_dim)))
                p = PCA_GPy(Y)
                PC = p.project(Y, min(input_dim, Y.shape[1]))
                X[:PC.shape[0], :PC.shape[1]] = PC
            elif embed_type.lower() == 'kernel PCA'.lower():
                from sklearn.decomposition import KernelPCA
                transformer = KernelPCA(n_components=input_dim, kernel=param)
                X = transformer.fit_transform(Y)
            elif embed_type.lower() == 'kernel PCA with EPs'.lower():
                from sklearn.decomposition import KernelPCA
                transformer = KernelPCA(n_components=input_dim-6, kernel=param)
                X = transformer.fit_transform(Y)
                X = np.append(X, Y[:, 117:], axis=1)
            elif embed_type.lower() == 'umap':
                import umap.umap_ as umap
                transformer = umap.UMAP(n_neighbors=10, n_components=input_dim, metric=param)
                X = transformer.fit_transform(Y)
            elif embed_type.lower() == 'isomap'.lower():
                from sklearn.manifold import Isomap
                embedding = Isomap(n_components=input_dim)#n_neighbors=int(param),
                X = embedding.fit_transform(Y)
            elif embed_type.lower() == 'LLE'.lower():  # ‘hessian’, ‘modified’, ‘ltsa’
                from sklearn.manifold import LocallyLinearEmbedding
                embedding = LocallyLinearEmbedding(n_neighbors=param, n_components=input_dim, method=param)
                X = embedding.fit_transform(Y)
            elif embed_type.lower() == 'FFT_2D'.lower():
                from HGPLVM.initializers import FFT_2D
                X = FFT_2D(Y, input_dim, self.num_seqs)
            elif embed_type.lower() == 'FFT_3D'.lower():
                from HGPLVM.initializers import FFT_3D
                X = FFT_3D(Y, input_dim, self.num_seqs)
            elif embed_type.lower() == 'lines'.lower():
                from HGPLVM.initializers import lines
                X = lines(Y.shape[0], input_dim)
            elif embed_type.lower() == 'random_sine_waves'.lower():
                from HGPLVM.initializers import random_sine_waves
                X = random_sine_waves(Y.shape[0], input_dim)
            elif embed_type.lower() == 'sine_waves'.lower():
                from HGPLVM.initializers import sine_waves
                X = sine_waves(Y.shape[0], input_dim)
            elif embed_type.lower() == 'fourier_basis'.lower():
                from HGPLVM.initializers import fourier_basis
                X = fourier_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            elif embed_type.lower() == 'wavelet_basis'.lower():
                from HGPLVM.initializers import wavelet_basis
                X = wavelet_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            elif embed_type.lower() == 'legendre_basis'.lower():
                from HGPLVM.initializers import legendre_basis
                X = legendre_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            elif embed_type.lower() == 'hermite_basis'.lower():
                from HGPLVM.initializers import hermite_basis
                X = hermite_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            elif embed_type.lower() == 'laguerre_basis'.lower():
                from HGPLVM.initializers import laguerre_basis
                X = laguerre_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            elif embed_type.lower() == 'chebyshev_basis'.lower():
                from HGPLVM.initializers import chebyshev_basis
                X = chebyshev_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            elif embed_type.lower() == 'zernike_basis'.lower():
                from HGPLVM.initializers import zernike_basis
                X = zernike_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            elif embed_type.lower() == 'spherical_harmonics_basis'.lower():
                from HGPLVM.initializers import spherical_harmonics_basis
                X = spherical_harmonics_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            elif embed_type.lower() == 'haar_basis'.lower():
                from HGPLVM.initializers import haar_basis
                X = haar_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            elif embed_type.lower() == 'walsh_basis'.lower():
                from HGPLVM.initializers import walsh_basis
                X = walsh_basis(Y.shape[0], input_dim, per_seq = param, num_seqs = self.num_seqs)
            else:
                print('Initialization method not recognized. Defaulted to PCA')
                self.init_embedding(Y=Y, input_dim=input_dim)
            return X

        def initialize_X(self, embed_type='PCA', X=None, Y=None, max_iters=500):
            if self.backconstraint is not None:
                X = self.backconstraint.X
            else:
                if Y is None and X is None:
                    raise ValueError('Either X or Y must have a declared value in order to initialize X')
                elif Y is None and X is not None:
                    pass
                else:
                    X = self.init_embedding(embed_type, Y, self.input_dim)

            self.unlink_parameter(self.X)
            self.X = Param('latent mean', X)
            self.link_parameter(self.X,0)
            self.initialize_Z()

        def initialize_Z(self):
            if self.num_inducing is not None:
                i = np.arange(0, self.N, int(np.round(self.N/self.num_inducing)))
                Z = self.X.view(np.ndarray)[i].copy()

                self.unlink_parameter(self.Z)
                self.Z = Param('inducing inputs', Z)
                self.link_parameter(self.Z,1)

        def set_backconstraint(self, BC_dict=None):
            if BC_dict is None:
                return
            if BC_dict['type'] is None or BC_dict['type'].lower() == 'None'.lower():
                return
            elif BC_dict['type'].lower() == 'map':
                from HGPLVM.backconstraints.GP_mapping_no_constraints import No_BC
                self.backconstraint = No_BC(self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'map geo':
                from HGPLVM.backconstraints.geometric_BCs import No_BC_geo
                self.backconstraint = No_BC_geo(BC_dict['geometry'], self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'map geo learn':
                from HGPLVM.backconstraints.geometric_BCs import No_BC_geo_learn
                self.backconstraint = No_BC_geo_learn(BC_dict['geometry'], self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'linear':
                from HGPLVM.backconstraints.lin_BC_base import Linear_BC
                self.backconstraint = Linear_BC(self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'linear x':
                from HGPLVM.backconstraints.lin_BC_base import Linear_X_BC
                self.backconstraint = Linear_X_BC(self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'linear geo':
                from HGPLVM.backconstraints.geometric_BCs import Linear_Geo_BC
                self.backconstraint = Linear_Geo_BC(BC_dict['geometry'],self, self.D,param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'linear x geo':
                from HGPLVM.backconstraints.geometric_BCs import Linear_X_Geo_BC
                self.backconstraint = Linear_X_Geo_BC(BC_dict['geometry'],self, self.D,param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'kernel':
                from HGPLVM.backconstraints.kern_BC_base import Kernel_BC
                self.backconstraint = Kernel_BC( self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'kernel geo':
                from HGPLVM.backconstraints.geometric_BCs import Kernel_Geo_BC
                self.backconstraint = Kernel_Geo_BC(BC_dict['geometry'], self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'mlp':
                from HGPLVM.backconstraints.mlp_BC import MLP_BC
                self.backconstraint = MLP_BC(self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'mlp geo':
                from HGPLVM.backconstraints.geometric_BCs import MLP_Geo_BC
                self.backconstraint = MLP_Geo_BC(BC_dict['geometry'], self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'gp':
                from HGPLVM.backconstraints.GP_BC_base import GP_BC
                self.backconstraint = GP_BC(self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'sparse gp':
                from HGPLVM.backconstraints.GP_BC_base import sparse_GP_BC
                self.backconstraint = sparse_GP_BC(self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'w gp':
                from HGPLVM.backconstraints.GP_BC_base import weighted_GP_BC
                self.backconstraint = weighted_GP_BC(self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'multi w gp':
                from HGPLVM.backconstraints.GP_BC_base import Multi_Weighted_GP_BC
                self.backconstraint = Multi_Weighted_GP_BC(self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'gp geo':
                from HGPLVM.backconstraints.geometric_BCs import GP_Geo_BC
                self.backconstraint = GP_Geo_BC(BC_dict['geometry'],self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'sparse gp geo':
                from HGPLVM.backconstraints.geometric_BCs import sparse_GP_Geo_BC
                self.backconstraint = sparse_GP_Geo_BC(BC_dict['geometry'],self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'w gp geo':
                from HGPLVM.backconstraints.geometric_BCs import weighted_GP_Geo_BC
                self.backconstraint = weighted_GP_Geo_BC(BC_dict['geometry'],self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'multi w gp geo':
                from HGPLVM.backconstraints.geometric_BCs import Multi_Weighted_GP_Geo_BC
                self.backconstraint = Multi_Weighted_GP_Geo_BC(BC_dict['geometry'],self, self.D, param_dict=BC_dict)
            elif BC_dict['type'].lower() == 'multi n w gp geo':
                from HGPLVM.backconstraints.geometric_BCs import Multi_N_Weighted_GP_Geo_BC
                self.backconstraint = Multi_N_Weighted_GP_Geo_BC(BC_dict['geometry'],self, self.D, param_dict=BC_dict)
            else:
                print('Backconstraint not recognized. None set.')
            BC_par_len = len(self.backconstraint.parameters)
            for i in range(BC_par_len):
                self.link_parameter(self.backconstraint.parameters[0])
            pass

        def set_prior(self, prior_dict, num_seqs=1, warning=True, **kwargs):
            N, D = self.X.shape
            if prior_dict["name"] is None or prior_dict["name"].lower() == "none":
                return
            elif prior_dict["name"].lower() == "gpdm1":
                x0_prior = GPDM_IG_prior(1, self.D)
                prior = GPDM(prior_dict["order"], prior_dict["num_inducing_dynamics"], self.X.values, self.gpdm_var, x0_prior, self.seq_eps, 1)
                self.X.set_prior(prior)
            elif prior_dict["name"].lower() == "gpdmm":
                prior = GPDMM(self.X.values, self.attr_dict["num_inducing_dynamics"], num_seqs, self.gpdm_var, self.seq_eps, 1,
                                     GPDM_order = prior_dict["order"],
                                     GPNode = self)
                self.X.set_prior(prior)
            else:
                print('No prior of the name: ' + prior_dict["name"])
                raise NameError
            self.prior = prior
            self.prior.name = prior_dict["name"]
            if hasattr(self.prior, 'parameters'):
                prior_par_len = len(self.prior.parameters)
                for i in range(prior_par_len):
                    self.link_parameter(self.prior.parameters[0])

        def set_HGPLVM(self, HGPLVM_in):
            self.HGPLVM = HGPLVM_in

        def set_nID(self, i):
            # set numeric ID
            self.node_nID = i

        def set_lID(self, i):
            # set numeric ID
            self.node_lID = self.mParent.node_lID.copy()
            self.node_lID.append(i)

        def create_HGPLVM(self):
            if self.GetParent() is None:
                return hgplvm.HGPLVM(self)
            else:
                print('Can only be called on most senior parent node')

        def set_Y(self, Y):
            # un-normalized
            self.Y = ObsAr(Y)
            self.Y_normalized = Y

        def set_Y_to_X_GP(self):
            kern = kern.RBF(self.Y.shape[1], 1, ARD=True) + kern.Linear(self.D, ARD=True) + kern.Bias(self.D, np.exp(-2))
            self.GP_Y_X = GPRegression(self.Y,self.X.values, kern)
            self.GP_Y_X.optimize(messages=False, max_f_eval=1000)

        def GP_infer_newX(self,Y_new):
            return self.GP_Y_X.predict_noiseless(Y_new)

        def objective_function(self):
            """
            The objective function for the given algorithm.

            This function is the true objective, which wants to be minimized.
            Note that all parameters are already set and in place, so you just need
            to return the objective function here.

            For probabilistic models this is the negative log_likelihood
            (including the MAP prior), so we return it here. If your model is not
            probabilistic, just return your objective to minimize here!
            """
            # If no prior is declared, log_prior is 0
            if self.prior is not None:
                log_prior = self.prior.lnpdf(self.X)
            else:
                log_prior = 0

            #print(-float(self.log_likelihood()) - log_prior)
            return -float(self.log_likelihood()) - log_prior

        def objective_function_gradients(self):
            """
            Updates the gradients of the latent space, kernel parameters, and likelihood function.
            The gradients are concatenated together as input for the HGPLVM parameter space
            Note: Mean function not implemented
            :return:
            a concatenated vector containing all of the gradients
            """
            return -(self._log_likelihood_gradients() + self._log_prior_gradients())

        def update_parameters(self):
            self.X.gradient += self.child_node_grad()
            prior_grads = np.zeros(self.X.shape).flatten()
            if self.prior is not None:
                self.prior.update_parameters(self.X)
                prior_grads = self.prior.lnpdf_grad(self.X) #retrieve stored parameter
            if self.backconstraint is not None:
                self.backconstraint.update_gradients(self.X.gradient + prior_grads.reshape(self.X.shape))

        def child_node_grad(self):
            if self.mParent is None:
                return 0
            else:
                return -self.mParent.grad_dict['K_inv']@self.X

        def parameters_changed(self):

            if self.backconstraint is not None:
                self.X = self.backconstraint.set_X(self.X, self.HGPLVM, self)

            super(GPLVMN, self).parameters_changed()
            self.update_parameters()

        def get_error(self, Y1, Y2=None):
            if Y2 is None:
                newX1, modX1 = self.infer_newX(Y1)
                muY1, varY1 = self.predict(newX1)
                err = (1 / Y1.shape[0]) * np.sum((Y1 - muY1) ** 2) / (np.mean(np.var(Y1)))
                return newX1, muY1
            else:
                err = (1 / Y1.shape[0]) * np.sum((Y1 - Y2) ** 2) / (np.mean(np.var(Y1)))

        def optimize(self, optimizer=None, start=None, **kwargs):
            try:
                ret = super(GPLVMN, self).optimize(optimizer, start, **kwargs)
            except KeyboardInterrupt:
                print("KeyboardInterrupt caught, calling on_optimization_end() to round things up")
                self.inference_method.on_optimization_end()
                raise
            return ret
    return GPLVMN(Y, attr_dict, **kwargs)



