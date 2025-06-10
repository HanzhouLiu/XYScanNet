import torch.nn as nn
from models.Stripformer import Stripformer
#from models.StripMamba_v20 import StripMamba_v0  
from models.StripMamba_v1 import StripMamba_v1  # Inter (score) + Inter (score)
from models.StripMamba_v2 import StripMamba_v2  # Inter (score) + Intra (not score)
from models.StripMamba_v3 import StripMamba_v3  # Intra (not score) + Intra (not score)
from models.StripMamba_v4 import StripMamba_v4  # Inter (score) + Intra (score)
from models.StripMamba_v5 import StripMamba_v5  # Intra (score) + Intra (score)
from models.StripMamba_v6 import StripMamba_v6  # Dual branch, parallel
from models.StripMamba_v7 import StripMamba_v7  # Transformer like
from models.StripMamba_v8 import StripMamba_v8  # Baseline: Stripformer
from models.StripMamba_v9 import StripMamba_v9  # Baseline: Stripformer
from models.StripMamba_v10 import StripMamba_v10  # Baseline: Stripformer
from models.StripMamba_v11 import StripMamba_v11  # Baseline: Stripformer
from models.StripMamba_v12 import StripMamba_v12  # Baseline: Stripformer
from models.StripMamba_v13 import StripMamba_v13  # Baseline: Stripformer
from models.StripMamba_v14 import StripMamba_v14  # Baseline: Stripformer
from models.StripMamba_v15 import StripMamba_v15  # Baseline: Stripformer
from models.StripMamba_v16 import StripMamba_v16  # Baseline: Stripformer
from models.StripMamba_v17 import StripMamba_v17  # Baseline: Stripformer
from models.StripMamba_v18 import StripMamba_v18  # Baseline: Stripformer
from models.StripMamba_v19 import StripMamba_v19  # Baseline: Stripformer
from models.StripMamba_v20 import StripMamba_v20  # Baseline: Stripformer
from models.StripMamba_v21 import StripMamba_v21  # Baseline: Stripformer
from models.StripMamba_v22 import StripMamba_v22  # Baseline: Stripformer
from models.StripMamba_v23 import StripMamba_v23  # Baseline: Stripformer
from models.StripMamba_v24 import StripMamba_v24  # Baseline: Stripformer
from models.StripMamba_v25 import StripMamba_v25  # Baseline: Stripformer
from models.StripMamba_v26 import StripMamba_v26  # Baseline: Stripformer
from models.StripMamba_v27 import StripMamba_v27  # Baseline: Stripformer
from models.StripMamba_v28 import StripMamba_v28  # Baseline: Stripformer
from models.StripMamba_v29 import StripMamba_v29  # Baseline: Stripformer
from models.StripMamba_v30 import StripMamba_v30  # Baseline: Stripformer
from models.StripMamba_v31 import StripMamba_v31  # Baseline: Stripformer
from models.StripMamba_v32 import StripMamba_v32  # Baseline: Stripformer
from models.StripMamba_v33 import StripMamba_v33  # Baseline: Stripformer
from models.StripMamba_v34 import StripMamba_v34  # Baseline: Stripformer
from models.StripMamba_v35 import StripMamba_v35  # Baseline: Stripformer
from models.StripMamba_v36 import StripMamba_v36  # Baseline: Stripformer
from models.StripMamba_v37 import StripMamba_v37  # Baseline: Stripformer
from models.StripMamba_v38 import StripMamba_v38  # Baseline: Stripformer
from models.StripMamba_v39 import StripMamba_v39  # Baseline: Stripformer
from models.StripMamba_v40 import StripMamba_v40  # Baseline: Stripformer
from models.StripMamba_v41 import StripMamba_v41  # Baseline: Stripformer
from models.StripMamba_v42 import StripMamba_v42  # Baseline: Stripformer
from models.StripMamba_v46 import StripMamba_v46  # Baseline: Stripformer
from models.StripMamba_v47 import StripMamba_v47  # Baseline: Stripformer
from models.StripMamba_v48 import StripMamba_v48  # Baseline: Stripformer

from models.StripMamba import StripMamba  # Baseline: Stripformer
from models.StripMambaPlus import StripMambaPlus  # Baseline: Stripformer
from models.FFTTest import FFTTest
from models.FFTformer import fftformer
from models.StripScanNet import StripScanNet
from models.XYScanNet import XYScanNet
from models.XYScanNetP import XYScanNetP

def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'Stripformer':
        model_g = Stripformer()
    elif generator_name == 'StripMamba_v1':
        model_g = StripMamba_v1()
    elif generator_name == 'StripMamba_v2':
        model_g = StripMamba_v2()
    elif generator_name == 'StripMamba_v3':
        model_g = StripMamba_v3()
    elif generator_name == 'StripMamba_v4':
        model_g = StripMamba_v4()
    elif generator_name == 'StripMamba_v5':
        model_g = StripMamba_v5()
    elif generator_name == 'StripMamba_v6':
        model_g = StripMamba_v6()
    elif generator_name == 'StripMamba_v7':
        model_g = StripMamba_v7()
    elif generator_name == 'StripMamba_v8':
        model_g = StripMamba_v8()
    elif generator_name == 'StripMamba_v9':
        model_g = StripMamba_v9()
    elif generator_name == 'StripMamba_v10':
        model_g = StripMamba_v10()
    elif generator_name == 'StripMamba_v11':
        model_g = StripMamba_v11()
    elif generator_name == 'StripMamba_v12':
        model_g = StripMamba_v12()
    elif generator_name == 'StripMamba_v13':
        model_g = StripMamba_v13()
    elif generator_name == 'StripMamba_v14':
        model_g = StripMamba_v14()
    elif generator_name == 'StripMamba_v15':
        model_g = StripMamba_v15()
    elif generator_name == 'StripMamba_v16':
        model_g = StripMamba_v16()
    elif generator_name == 'StripMamba_v17':
        model_g = StripMamba_v17()
    elif generator_name == 'StripMamba_v18':
        model_g = StripMamba_v18()
    elif generator_name == 'StripMamba_v19':
        model_g = StripMamba_v19()
    elif generator_name == 'StripMamba_v20':
        model_g = StripMamba_v20()
    elif generator_name == 'StripMamba_v21':
        model_g = StripMamba_v21()
    elif generator_name == 'StripMamba_v22':
        model_g = StripMamba_v22()
    elif generator_name == 'StripMamba_v23':
        model_g = StripMamba_v23()
    elif generator_name == 'StripMamba_v24':
        model_g = StripMamba_v24()
    elif generator_name == 'StripMamba_v25':
        model_g = StripMamba_v25()
    elif generator_name == 'StripMamba_v26':
        model_g = StripMamba_v26()
    elif generator_name == 'StripMamba_v27':
        model_g = StripMamba_v27()
    elif generator_name == 'StripMamba_v28':
        model_g = StripMamba_v28()
    elif generator_name == 'StripMamba_v29':
        model_g = StripMamba_v29()
    elif generator_name == 'StripMamba_v30':
        model_g = StripMamba_v30()
    elif generator_name == 'StripMamba_v31':
        model_g = StripMamba_v31()
    elif generator_name == 'StripMamba_v32':
        model_g = StripMamba_v32()
    elif generator_name == 'StripMamba_v33':
        model_g = StripMamba_v33()
    elif generator_name == 'StripMamba_v34':
        model_g = StripMamba_v34()
    elif generator_name == 'StripMamba_v35':
        model_g = StripMamba_v35()
    elif generator_name == 'StripMamba_v36':
        model_g = StripMamba_v36()
    elif generator_name == 'StripMamba_v37':
        model_g = StripMamba_v37()
    elif generator_name == 'StripMamba_v38':
        model_g = StripMamba_v38()
    elif generator_name == 'StripMamba_v39':
        model_g = StripMamba_v39()
    elif generator_name == 'StripMamba_v40':
        model_g = StripMamba_v40()
    elif generator_name == 'StripMamba_v41':
        model_g = StripMamba_v41()
    elif generator_name == 'StripMamba_v42':
        model_g = StripMamba_v42()
    elif generator_name == 'StripMamba_v46':
        model_g = StripMamba_v46()
    elif generator_name == 'StripMamba_v47':
        model_g = StripMamba_v47()
    elif generator_name == 'StripMamba_v48':
        model_g = StripMamba_v48()
    elif generator_name == 'StripMamba':
        model_g = StripMamba()
    elif generator_name == 'StripMambaPlus':
        model_g = StripMambaPlus()
    elif generator_name == 'FFTTest':
        model_g = FFTTest()
    elif generator_name == 'FFTformer':
        model_g = fftformer()
    elif generator_name == 'StripScanNet':
        model_g = StripScanNet()
    elif generator_name == 'XYScanNet':
        model_g = XYScanNet()
    elif generator_name == 'XYScanNetP':
        model_g = XYScanNetP()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
