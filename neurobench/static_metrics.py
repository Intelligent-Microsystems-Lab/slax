import torch
import snntorch as snn
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves
import jax.numpy as jnp
# static metrics, only require model

def parameter_count(model,params):
    """ Number of parameters in the model.

    Args:
        model: A NeuroBenchModel.
    Returns:
        int: Number of parameters.
    """
    return ravel_pytree(params)[0].size#sum(p.numel() for p in model.__net__().parameters())

def footprint(model,params):
    """ Memory footprint of the model.

    Args:
        model: A NeuroBenchModel.
    Returns:
        float: Model size in bytes.
    """
    # Count the number of parameters and multiply by the size of each parameter in bytes
    # param_size = 0
    # for param in model.__net__().parameters():
    #     param_size += param.numel() * param.element_size()

    param_size = 0
    for param in tree_leaves(params):
        param_size += param.nbytes

    # Count the number of buffers and multiply by the size of each buffer in bytes
    buffer_size = 0
    # for buffer in model.__net__().buffers():
    #     buffer_size += buffer.numel() * buffer.element_size()

    # Return the sum of the parameters and buffers
    return param_size + buffer_size

def connection_sparsity(model,params):
    """ Sparsity of model connections between layers. Based on number of zeros 
    in supported layers, other layers are not taken into account in the computation:
    Supported layers:
    Linear
    Conv1d, Conv2d, Conv3d
    RNN, RNNBase, RNNCell
    LSTM, LSTMBase, LSTMCell
    GRU, GRUBase, GRUCell

    Args:
        model: A NeuroBenchModel.
    Returns:
        float: Connection sparsity, rounded to 3 decimals.
    """

    def get_nr_zeros_weights(module):
        """ Get the number of zeros in a module's weights.

        Args:
            module: A torch.nn.Module.
        Returns:
            int: Number of zeros in the module's weights.
        """
    #     children = list(module.children())
    #     regular_layers = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    #     recurr_layers  = (torch.nn.RNNBase)
    #     recurr_cells   = (torch.nn.RNNCellBase)
    #     if len(children) == 0: # it is a leaf
    #         # print(module)
    #         if isinstance(module, regular_layers):
    #             count_zeros = torch.sum(module.weight == 0)
    #             count_weights = module.weight.numel()
    #             return count_zeros, count_weights
            
    #         elif isinstance(module, recurr_layers):
    #             attribute_names = []
    #             for i in range(module.num_layers): 
    #                 param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
    #                 if module.bias:
    #                     param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
    #                 if module.proj_size > 0: # it is lstm
    #                     param_names += ['weight_hr_l{}{}']

    #                 attribute_names += [x.format(i, '') for x in param_names]
    #                 if module.bidirectional:
    #                     suffix = '_reverse'
    #                     attribute_names += [x.format(i, suffix) for x in param_names]

    #             count_zeros = 0
    #             count_weights = 0
    #             for attr in attribute_names:
    #                 attr_val = getattr(module, attr)
    #                 count_zeros += torch.sum(attr_val == 0)
    #                 count_weights += attr_val.numel() 

    #             return count_zeros, count_weights
            
    #         elif isinstance(module, recurr_cells):
    #             attribute_names = ['weight_ih', 'weight_hh']
    #             if module.bias:
    #                 attribute_names += ['bias_ih', 'bias_hh']
   
    #             count_zeros = 0
    #             count_weights = 0
    #             for attr in attribute_names:
    #                 attr_val = getattr(module, attr)
    #                 count_zeros += torch.sum(attr_val == 0)
    #                 count_weights += attr_val.numel() 

    #             return count_zeros, count_weights

    #         elif isinstance(module, snn.SpikingNeuron):
    #             return 0, 0 # it is a neuromorphic neuron layer
    #         else:
    #             # print('Module type: ', module, 'not found.')
    #             return 0,0
        
    #     else:
    #         count_zeros = 0
    #         count_weights = 0
    #         for child in children:
    #             child_zeros, child_weights = get_nr_zeros_weights(child)
    #             count_zeros += child_zeros
    #             count_weights += child_weights
    #         return count_zeros, count_weights
    # # Pull the layers from the model's network
    # layers = model.__net__().children()
    # # For each layer, count where the weights are zero
    # count_zeros = 0
    # count_weights = 0
    # for module in layers:
    #     zeros, weights = get_nr_zeros_weights(module)
    #     count_zeros += zeros
    #     count_weights += weights

    # Return the ratio of zeros to weights, rounded to 4 decimals
    p = ravel_pytree(params)[0]
    return round(1-jnp.count_nonzero(p).tolist()/p.size,4)#round((count_zeros / count_weights).item(), 4)