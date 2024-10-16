
def hgp_arch_factory(HGP, data_set_class, arch_dict):
    """if name.lower() == 'base'.lower():
        from HGPLVM.architectures.base import HGP_Architecture_Base
        return HGP_Architecture_Base()"""
    if arch_dict['attr_dict']['name'].lower() == 'X1_Y1'.lower():
        from model_comparison_core.architectures.structures import X1_Y1 as arch
    elif arch_dict['attr_dict']['name'].lower() == 'X1_H2_Y2'.lower():
        from model_comparison_core.architectures.structures import X1_H2_Y2 as arch
    elif arch_dict['attr_dict']['name'].lower() == 'X1_H1_Y1'.lower():
        from model_comparison_core.architectures.structures import X1_H1_Y1 as arch
    else:
        raise NotImplementedError('Requested architecture not implemented.')
    return arch(HGP, data_set_class, arch_dict = arch_dict)

def rnn_arch_factory(RNN_wrapper, data_set_class, arch_dict):
    if arch_dict['model_type'].lower() == 'LSTM'.lower():
        from model_comparison_core.architectures.structures import SequenceClassifier as arch
    else:
        raise NotImplementedError('Requested architecture not implemented.')
    return arch(RNN_wrapper, data_set_class, **arch_dict)

'''
def dynamical_architecture_factory(arch_class, dyn_class):
    class DynamicalArchitecture(arch_class, dyn_class):
        def __init__(self, HGP, data_set_class, base_arch=None):
            super().__init__(HGP, data_set_class, base_arch)

    return DynamicalArchitecture


def arch_factory(arch_name, dynamics_name=None, *args, **kwargs):
    if arch_name.lower() == 'base'.lower():
        from HGPLVM.architectures.base import HGP_Architecture_Base
        return HGP_Architecture_Base()
    elif arch_name.lower() == 'X1_Y1'.lower():
        from HGPLVM.architectures.structures import X1_Y1 as arch_class
    elif arch_name.lower() == 'X1_H2_Y2'.lower():
        from HGPLVM.architectures.structures import X1_H2_Y2 as arch_class
    elif arch_name.lower() == 'X1_H1_Y1'.lower():
        from HGPLVM.architectures.structures import X1_H1_Y1 as arch_class
    else:
        raise NotImplementedError('Requested architecture not implemented.')

    if dynamics_name is None:
        arch = arch_class
    else:
        if dynamics_name.lower() == 'ff'.lower():
            from HGPLVM.architectures.dynamics import FeedforwardDynamics as dyn_class
        elif dynamics_name.lower() == 'fb'.lower():
            from HGPLVM.architectures.dynamics import FeedbackDynamics as dyn_class
        else:
            raise NotImplementedError('Requested dynamics not implemented.')
        arch = dynamical_architecture_factory(arch_class, dyn_class)

    return arch(*args, **kwargs)
'''