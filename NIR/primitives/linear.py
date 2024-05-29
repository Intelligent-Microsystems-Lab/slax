from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import nir


def nir_dense(mdl,params):
    if mdl.use_bias == True:
        return nir.Affine(
            weight = np.array(params['params']['kernel']),
            bias = np.array(params['params']['bias']),
        )
    else:
        return nir.Linear(
            weight = np.array(params['params']['kernel'])
        )