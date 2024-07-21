import flax.linen as nn
import nir
import numpy as np


def nir_conv(mdl,params):
    return nir.Conv2d(
            weight = np.array(params['params']['kernel']),
            stride = mdl.stride,
            padding = mdl.padding,
            dilation = mdl.dilation,
            groups = 1,
            bias = np.array(params['params']['bias'])
        )
