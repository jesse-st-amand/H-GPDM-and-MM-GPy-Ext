from HGPLVM.hgp import HGP
import gc

def multi_GPDMM(data_set_class, hgp_dict, opts, seed=None, **kwargs):
    print('seed: ' + str(int(seed)))

    hgp = HGP(hgp_dict, data_set_class=data_set_class)
    hgp.optimize(max_iters=hgp_dict['attr_dict']['max_iters'], optimizer=opts[1])
    hgp.get_attribute_dict()

    data_set_class.store_HGP_attributes(hgp)
    print(hgp.get_attribute_dict())

    gc.collect()
    return hgp
