from jax.tree_util import tree_leaves, tree_structure, tree_unflatten, tree_map
import optax
from jax.flatten_util import ravel_pytree

def layerwise_cosine_similarity(pytree_0,pytree_1):
    '''
    Computes the cosine similarity of each item between two pytrees with the same structure.

    Args:
        pytree_0: The first pytree with the same structure as pytree_1
        pytree_1: The second pytree with the same structure as pytree_0
    
    Returns:
        A pytree with the structure as the inputs. Each item contains the scalar cosine similarity value.
    '''
    return tree_map(lambda x,y: optax.cosine_similarity(x.flatten(),y.flatten()),pytree_0,pytree_1)

def global_cosine_similarity(pytree_0,pytree_1):
    '''
    Computes the cosine similarity of all elements between two pytrees. 

    Args:
        pytree_0: The first pytree
        pytree_1: The second pytree
    
    Returns:
        A pytree with the structure as the inputs. Each item contains the scalar cosine similarity value.
    '''
    return optax.cosine_similarity(ravel_pytree(pytree_0)[0],ravel_pytree(pytree_1)[0])

def compare_grads(train_func,reference_params,reference_grad,train_func_args,comparison_func=layerwise_cosine_similarity):
    '''
    Performs a comparison function on a given reference pytree of gradients and a calculated pytree of gradients, using
    a given training function and its arguments.

    Args:
        train_func: The returned function from calling `train_online_deffered` or a similar function with the same inputs
        and outputs
        reference_params: A pytree of the reference parameters
        reference_grad: A pytree of the reference gradients
        train_func_args: A tuple of the arguments for `train_func` (params,carry,batch,opt_state)
        comparison_func: A function that takes in two pytrees and performs some comparison operation. Defaults to `layerwise_cosine_similarity'
    
    Returns:
        The output of comparison_func
    '''

    params,batch,opt_state = train_func_args
    params['params'] = tree_unflatten(tree_structure(params['params']),tree_leaves(reference_params['params']))
    _,_,_,_,new_grad = train_func(params,batch,opt_state,return_grad=True)
    reference_grad = tree_unflatten(tree_structure(new_grad),tree_leaves(reference_grad))
    return comparison_func(reference_grad,new_grad)