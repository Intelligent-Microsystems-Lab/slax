#from . import _nir
from .models.neurons import LIF
from . import train
from . import models
from . import eval
#from . import train_2 as nir

from .train.custom_learning import diag_rtrl, OTTT, OTPE, rtrl
from .eval.randman_dataset import randman
from .eval.gen_ll import gen_loss_landscape
from .models import surrogates
from .models.surrogates import *
from .eval.utils import *
from .train.utils import *
from .models.utils import *
from .nir.utils import from_nir, to_nir