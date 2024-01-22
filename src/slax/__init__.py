from . import neurons
from .neurons import LIF
from . import custom_learning
from .custom_learning import DenseOSTL
from .randman_dataset import randman
from .gen_ll import gen_loss_landscape
from . import surrogates
from .surrogates import fast_sigmoid
from . import utils
from .utils import train_online, train_online_deferred, train_offline, FPTT, compare_grads, recurrent

def main():
    print('slax')
if __name__ == '__main__':
    main()