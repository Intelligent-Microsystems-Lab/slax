"""
Slax docs
"""

#from . import _nir
from .model.neuron import LIF
from . import train
from . import model
from . import eval

from .train.CustomLearning.custom_learning import OTPE, rtrl
from .train.CustomLearning.diag_rtrl import diag_rtrl
from .train.CustomLearning.OTTT import OTTT
from .eval.randman_dataset import randman
from .eval.gen_ll import gen_loss_landscape
from .model import surrogate
from .model.surrogate import *
from .eval.utils import *
from .train.utils import *
from .model.utils import *
from .nir.utils import from_nir, to_nir