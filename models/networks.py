import torch.nn as nn
from models.XYScanNet import XYScanNet
from models.XYScanNetP import XYScanNetP

def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'XYScanNet':
        model_g = XYScanNet()
    elif generator_name == 'XYScanNetP':
        model_g = XYScanNetP()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
