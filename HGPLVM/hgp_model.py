import numpy as np
from GPy.core.model import Model
import logging
from paramz.core.parameter_core import adjust_name_for_printing
from collections import OrderedDict
from HGPLVM.global_functions import bit_racer as br


logger = logging.getLogger("GP")



class HGPLVM(Model):
    """
    Hierarchical Gaussian Process Latent Variable Model
    """
    def __init__(self, GPNode, arch = None, inference_method=None, name='HGPLVM'):
        self.timer = br()
        super(HGPLVM, self).__init__(name)
        self.nodes = []
        self.ObjFunVal = 0
        self.ObjFunGradVal = []
        self.num_nodes = 0
        self.num_linked_params = 0
        self.num_total_params = 0
        self.top_node = None
        self.bottom_nodes = []
        self.seq_eps = None
        self.set_top_node(GPNode)
        self.link_nodes(self.top_node)
        self.find_set_bottom_nodes(self.top_node)
        self.grad_dict = {}
        self.attr_dict = None
        self.concatenate_parameters(self.top_node)
        self.arch = arch
        self.learning_n = 0
        self.score_list = []
        self.iter_list = []
        self.ObjFunVal_list = []

    def link_nodes(self, GPNode):
        """
        Starting from the lead node, links each node to the HGPLVM model
        and creates an ID number for that node relative to the HGPLVM hierarchy.
        This function should only be called once on initialization.
        """
        for i, child in enumerate(GPNode.mChild):
            child.set_lID(i)
            self.link_nodes(child)
        GPNode.set_HGPLVM(self)
        GPNode.set_nID(self.num_nodes)
        self.nodes.append(GPNode)
        self.num_nodes += 1

    def concatenate_parameters(self, GPNode):
        for child in GPNode.mChild:
            self.concatenate_parameters(child)

        GPNode_par_len = len(GPNode.parameters)
        for i in range(GPNode_par_len):
            p = GPNode.parameters[0]
            self.link_parameter(p)
            GPNode.p_indices.append(self.num_linked_params)
            GPNode.p_array_indices.append(self.num_total_params)
            self.num_linked_params += 1
            self.num_total_params += p.size
            GPNode.p_array_indices.append(self.num_total_params)

    def set_top_node(self, GPNode):
        if GPNode.GetParent() is None:
            self.top_node = GPNode
            self.top_node.node_lID = []
        else:
            self.set_top_node(GPNode.GetParent())

    def get_objective_function_value(self):
        return self.ObjFunVal

    def get_objective_function_gradient_value(self):
        return self.ObjFunGradVal

    def get_top_node(self):
        return self.top_node

    def objective_function(self):
        """
        The objective function for the given algorithm.

        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the objective function here.

        For probabilistic models this is the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your objective to minimize here!

        For use in an HGP, call most start on most senior parent node
        """
        self.ObjFunVal = 0
        self.objective_function_node_pass(self.top_node)
        print(['obj fun: '+str(self.ObjFunVal)])
        return self.ObjFunVal
    def objective_function_node_pass(self,GPNode):
        """
        Passes the objective function call to node children and returns the single-node objective function
        """
        for child in GPNode.mChild:
            self.objective_function_node_pass(child)
        self.ObjFunVal += GPNode.objective_function()

    def objective_function_gradients(self):
        '''if self.learning_n < 30:
            return -(self._log_likelihood_gradients()+ self._log_prior_gradients())
        else:
            return -(self._log_prior_gradients())'''
        return -(self._log_likelihood_gradients()+ self._log_prior_gradients())

    def parameters_changed(self):
        #self.update_model(updates=False)
        self.update_hierarchy(self.top_node)
        #print(self.parameters)
        #print(self.top_node.Y[0,:])



    def update_hierarchy(self, GPNode):
        #GPNode.kernel_inversion()
        Ys = []
        children = False
        for child in GPNode.mChild:
            children = True
            self.update_hierarchy(child)
            Ys.append(child.X.values)
        if children:
            print('WARNING: Hierarchical model may need revision. Look at the ordering of the parameters')
            GPNode.set_Y(np.hstack(Ys))
        #self.grad_dict[str(GPNode.node_nID)] = GPNode.gradient
        GPNode.parameters_changed()

        self.arch.store_data(self.learning_n, self.arch.attr_dict['score_rate'], self.ObjFunVal)
        self.learning_n += 1
        #self.grad_dict[str(GPNode.node_nID)] = GPNode.gradient

    def get_attribute_dict(self):
        if self.attr_dict is None:
            self.attr_dict = {}
            self.write_attr_to_dict()
        return self.attr_dict

    def write_attr_to_dict(self):
        self.write_attr_recursive(self.top_node)

    def write_attr_recursive(self,GPNode):
        self.attr_dict[GPNode.node_nID]={}
        self.attr_dict[GPNode.node_nID]['latent dimensionality'] = GPNode.input_dim
        #self.attr_dict[GPNode.node_nID]['initializations']={}
        #self.attr_dict[GPNode.node_nID]['initializations']['init1']=GPNode.init
        if GPNode.backconstraint is not None:
            self.attr_dict[GPNode.node_nID]['BC']=GPNode.backconstraint.name
        else:
            self.attr_dict[GPNode.node_nID]['BC'] = 'none'
        if GPNode.prior is not None:
            self.attr_dict[GPNode.node_nID]['prior'] = GPNode.prior.name
        else:
            self.attr_dict[GPNode.node_nID]['prior'] = 'none'
        self.attr_dict[GPNode.node_nID]['gpdm_var'] = GPNode.gpdm_var
        for child in GPNode.mChild:
            self.write_attr_recursive(child)

    def find_set_bottom_nodes(self, GPNode):
        children = False
        for child in GPNode.mChild:
            children = True
            self.find_set_bottom_nodes(child)
        if not children:
            return self.bottom_nodes.append(GPNode)

    def infer_top_X(self, GPNode, Y_dict):
        Ys = []
        children = False
        for child in GPNode.mChild:
            children = True
            Y = self.infer_top_X(child, Y_dict)
            if len(GPNode.mChild) > 1:
                Ys.append(Y)
        if not children:
            X, modX = GPNode.infer_newX(Y_dict[str(GPNode.node_nID)],optimize=True)
            return X.values
        elif len(GPNode.mChild) > 1:
            X, modX = GPNode.infer_newX(np.hstack(Ys))
            return X.values
        else:
            X, modX = GPNode.infer_newX(Y)
            return X.values

    def BC_mapping_top_X(self, GPNode, Y_dict,**kwargs):
        Ys = []
        children = False
        for child in GPNode.mChild:
            children = True
            Y = self.infer_top_X(child, Y_dict)
            if len(GPNode.mChild) > 1:
                Ys.append(Y)
        if not children:
            X = GPNode.backconstraint.f_new(Y_dict[str(GPNode.node_nID)],**kwargs)
            return X
        elif len(GPNode.mChild) > 1:
            X = GPNode.backconstraint.f_new(np.hstack(Ys),**kwargs)
            return X
        else:
            X = GPNode.backconstraint.f_new(Y,**kwargs)
            return X
        #X = self.top_node.backconstraint.f_new(Y, **kwargs)

    def GP_infer_top_X(self, GPNode, Y_dict):
        Ys = []
        children = False
        for child in GPNode.mChild:
            children = True
            Y = self.infer_top_X(child, Y_dict)
            if len(GPNode.mChild) > 1:
                Ys.append(Y)
        if not children:
            X, modX = GPNode.GP_infer_newX(Y_dict[str(GPNode.node_nID)])
            return X
        elif len(GPNode.mChild) > 1:
            X, modX = GPNode.GP_infer_newX(np.hstack(Ys))
            return X
        else:
            X, modX = GPNode.GP_infer_newX(Y)
            return X

    def predict_bottom_Y(self, GPNode, X):
        Ys = []
        Y, varY = GPNode.predict(X)
        children = False
        for i,child in enumerate(GPNode.mChild):
            if child is not None:
                children = True
                if len(GPNode.mChild) > 1:
                    Ynew = self.predict_bottom_Y(child, Y[:,(i*child.D):((i+1)*child.D)])
                else:
                    Ynew = self.predict_bottom_Y(child, Y)
                Ys.append(Ynew)
        if children:
            return Ys
        elif GPNode.mParent:
            return Y
        else:
            return [Y]

    def define_input(self,GPNodes,Ys):
        Y_dict = {}
        for GPNode,Y in zip(GPNodes,Ys):
            Y_dict[str(GPNode.node_nID)] = Y
        return Y_dict

    def get_bottom_Ys(self,X):
        return self.predict_bottom_Y(self.top_node, X)

    def get_top_X(self,Ys,**kwargs):
        if self.top_node.backconstraint:
            return self.get_top_X_BC(Ys,**kwargs)
            #return self.top_node.backconstraint.f_new(Ys[kwargs['pred_group']],**kwargs)
        elif self.top_node.GP_Y_X:
            return self.get_top_X_GP(Ys)
        else:
            return self.get_top_X_GPLVM(Ys)




    def get_top_X_GPLVM(self,Ys):
        Y_dict = self.define_input(self.bottom_nodes, Ys)
        return self.infer_top_X(self.top_node, Y_dict)

    def get_top_X_BC(self,Ys,**kwargs):
        Y_dict = self.define_input(self.bottom_nodes, Ys)
        return self.BC_mapping_top_X(self.top_node, Y_dict,**kwargs)

    def get_top_X_GP(self,Ys):
        Y_dict = self.define_input(self.bottom_nodes, Ys)
        return self.GP_infer_top_X(self.top_node, Y_dict)

    def reconstruct_input_BC(self,Ys,print_error=True):
        '''
        Automatically calculates the whole-network reconstruction.
        Note: The order of Y matrices in 'Ys' should be smallest to largest node_nID
        :param Ys: List of Y matrices defining the observation variables
        :return:
        '''
        X = self.get_top_X_BC(Ys)
        Ys_recon = self.predict_bottom_Y(self.top_node, X)
        '''if print_error is True:
            for Y1, Y2, GPNode in zip(Ys,Ys_recon,self.bottom_nodes):
                self.get_single_node_error(Y1,Y2, GPNode)'''
        return Ys_recon, X

    def reconstruct_input(self,Ys,print_error=True):
        '''
        Automatically calculates the whole-network reconstruction.
        Note: The order of Y matrices in 'Ys' should be smallest to largest node_nID
        :param Ys: List of Y matrices defining the observation variables
        :return:
        '''
        X = self.get_top_X(Ys)
        Ys_recon = self.predict_bottom_Y(self.top_node, X)
        '''if print_error is True:
            for Y1, Y2, GPNode in zip(Ys,Ys_recon,self.bottom_nodes):
                self.get_single_node_error(Y1,Y2, GPNode)'''
        return Ys_recon, X

    def get_single_node_error(self, Y1, Y2, GPNode=None):
        if GPNode is None:
            err = (1 / Y1.shape[0]) * np.sum((Y1 - Y2) ** 2) / (np.mean(np.var(Y1)))
            #print('Prediction error: ' + str(err))
            return err
        else:
            err = (1 / Y1.shape[0]) * np.sum((Y1 - Y2) ** 2) / (np.mean(np.var(Y1)))
            #print('Node ' + str(GPNode.node_nID) + ' reconstruction error: ' + str(err))

    def opt_back_projection(self):
        self.opt_back_projection_iter(self.top_node)

    def opt_back_projection_iter(self,GPNode):
        GPNode.set_Y_to_X_GP()
        for i, child in enumerate(GPNode.mChild):
            self.opt_back_projection_iter(child)


    def optimize(self, optimizer=None, start=None, **kwargs):
        self._IN_OPTIMIZATION_ = True

        #self.inference_method.on_optimization_start()
        try:
            ret = super(HGPLVM, self).optimize(optimizer, start, **kwargs)
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught, calling on_optimization_end() to round things up")
            self.inference_method.on_optimization_end()
            raise

        self._IN_OPTIMIZATION_ = False
        return ret

    def __str__(self, VT100=True):
        model_details = [['Name', self.name],
                         ['Objective', '{}'.format(float(self.objective_function()))],
                         ["Number of Parameters", '{}'.format(self.size)],
                         ["Number of Optimization Parameters", '{}'.format(self._size_transformed())],
                         ["Updates", '{}'.format(self._update_on)],
                         ]
        max_len = max(map(len, model_details))
        to_print = [""] + ["{0:{l}} : {1}".format(name, detail, l=max_len) for name, detail in model_details] + ["Parameters:"]

        header = True
        VT100 = True
        name = adjust_name_for_printing(self.name) + "."
        names = self.parameter_names(adjust_for_printing=True)
        desc = self._description_str
        iops = OrderedDict()
        for opname in self._index_operations:
            if opname != 'priors':
                iops[opname] = self.get_property_string(opname)

        format_spec = '  |  '.join(self._format_spec(name, names, desc, iops, VT100))

        to_print = []

        if header:
            to_print.append(format_spec.format(name=name, desc='value', **dict((name, name) for name in iops)))

        for i in range(len(names)):
            to_print.append(format_spec.format(name=names[i], desc=desc[i], **dict((name, iops[name][i]) for name in iops)))

        return "\n".join(to_print)
