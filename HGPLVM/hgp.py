from HGPLVM.architectures.factory_functions import hgp_arch_factory, rnn_arch_factory

class Model_Wrapper():
    def __init__(self, model_dict, data_set_class = None):
        self.attr_dict = model_dict['attr_dict']
        self.arch_dict = model_dict['arch_dict']

        if self.arch_dict is not None:
            self.set_data_set_class(data_set_class)
            self.build_architecture(data_set_class = data_set_class)

    def set_architecture(self, name):
        self.arch_dict['name'] = name

    def set_data_set_class(self, data_set_class):
        self.num_tps = data_set_class.num_tps
        self.avgs = data_set_class.avgs
        self.stds = data_set_class.stds
        self.sparse = False

    def set_dynamics(self, dynamics_type, **kwargs):
        self.arch_dict['dynamics_type'] = dynamics_type

    def set_optimizers(self, opt_list):
        self.arch_dict['opt_list'] = opt_list

    def predict(self, Y, test=False, **kwargs):
        return self.arch.predict(Y, test, **kwargs)

    def score(self, **kwargs):
        return self.arch.score(**kwargs)


class HGP(Model_Wrapper):

    def build_architecture(self,data_set_class):
        if self.arch_dict is None:
            raise ValueError('No architecture set.')
        self.arch = hgp_arch_factory(self, data_set_class, self.arch_dict)

    def optimize(self, *args, **kwargs):
        self.arch.model.optimize(*args, **kwargs)

    def get_attribute_dict(self):
        return self.arch.model.get_attribute_dict()

    def set_kernels(self, kernel_list):
        self.arch_dict['kernel_list'] = kernel_list

    def set_backconstraints(self, BC_dicts):
        self.arch_dict['BC_dicts'] = BC_dicts

    def set_priors(self, prior_dict):
        self.arch_dict['prior_dict'] = prior_dict

    def set_initializations(self, init_list, max_iters=100, GPNode_opt=False):
        self.arch_dict['init_list'] = init_list

    def set_optimizers(self, opt_list):
        self.arch_dict['opt_list'] = opt_list

    def set_input_dimensions(self, input_dims_list):
        self.arch_dict['input_dims_list']  = input_dims_list

    def set_num_inducing_points(self, num_inducing_list):
        #self.sparse = True
        self.arch_dict['num_inducing_list'] = num_inducing_list

    def opt_back_projection(self):
        self.arch.model.opt_back_projection()

    def infer_X(self, Y_test_list, **kwargs):
        return self.arch.infer_X(Y_test_list, **kwargs)

    def reconstruct_input(self, Ys, print_error=True):
        return self.arch.model.reconstruct_input(Ys, print_error=True)

    def IF_setup(self):
        return self.arch.IF_setup()


