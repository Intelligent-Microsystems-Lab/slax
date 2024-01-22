import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Sequence
from jax.tree_util import Partial, tree_leaves, tree_structure, tree_unflatten, tree_map
from .neurons import LIF, LTC
from jax.lax import stop_gradient as stop_grad
import optax
from jax.flatten_util import ravel_pytree

class train_online_deferred(nn.Module):
    '''
    A helper tool for easily implementing an online training loop where parameter updates are saved till the end.

    Args:
    snnModel: An initializes Flax module
    loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
    either an array to averaged or a scalar loss value.
    optimizer: An initialized optax optimizer
    '''
    snnModel: Callable
    loss_fn: Callable
    optimizer: Callable

    def __call__(self,params,carry,batch,opt_state):
        '''
        Args:
        params: Variable collection for snnModel
        carry: Carry/state for snnModel
        batch: Tuple of the input and labels
        opt_state: Variable collection of the optimizer
        '''
        def loss_fn(params,state,batch):
            state,s = self.snnModel.apply(params,state,batch[0])
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s,state)

        def one_step(carry,batch):
            params,state,opt_state,grad = carry
            g,(loss,s,state) = jax.jacrev(loss_fn,has_aux=True)(params,state,batch)
            grad = tree_map(lambda x,y: x+y,g,grad)
            return (params,state,opt_state,grad), (s,loss)
        
        def loop(p,state,batch,opt_state):
            grad = tree_map(jnp.zeros_like,p)
            (_,state,opt_state,grad),(s,loss) = jax.lax.scan(one_step,(p,state,opt_state,grad),batch)
            grad = tree_map(lambda x: x/batch[0].shape[1],grad)
            updates, opt_state = self.optimizer.update(grad,opt_state)
            params = optax.apply_updates(p, updates)
            return params,opt_state,s,loss,grad,state
        
        return loop(params,carry,batch,opt_state)
    
class train_online(nn.Module):
    '''
    A helper tool for easily implementing an online training loop where parameter updates are applied at each time-step.

    Args:
    snnModel: An initializes Flax module
    loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
    either an array to averaged or a scalar loss value.
    optimizer: An initialized optax optimizer
    '''
    snnModel: Callable
    loss_fn: Callable
    optimizer: Callable

    def __call__(self,params,carry,batch,opt_state):
        '''
        Args:
        params: Variable collection for snnModel
        carry: Carry/state for snnModel
        batch: Tuple of the input and labels
        opt_state: Variable collection of the optimizer
        '''

        def loss_fn(params,state,batch):
            state,s = self.snnModel.apply(params,state,batch[0])
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s,state)

        def one_step(carry,batch):
            params,state,opt_state = carry
            grad,(loss,s,state) = jax.jacrev(loss_fn,has_aux=True)(params,state,batch)
            updates, opt_state = self.optimizer.update(grad,opt_state)
            params = optax.apply_updates(params, updates)
            return (params,state,opt_state), (s,loss)
        
        def loop(p,state,batch,opt_state):
            (params,state,opt_state),(s,loss) = jax.lax.scan(one_step,(p,state,opt_state),batch)
            return params,opt_state,s,loss,state
        
        return loop(params,carry,batch,opt_state)
    

class FPTT(nn.Module):
    '''
    An implementation of Forward Propagation Through Time as an online training loop, which can be used the same way as `train_online`.

    Kag, A. &amp; Saligrama, V.. (2021). Training Recurrent Neural Networks via Forward Propagation Through Time. <i>Proceedings of the 38th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 139:5189-5200 Available from https://proceedings.mlr.press/v139/kag21a.html.

    Args:
    snnModel: An initializes Flax module
    loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
    either an array to averaged or a scalar loss value.
    optimizer: An initialized optax optimizer
    alpha: A hyperparameter for FPTT that modulates the influence of the risk function
    '''
    snnModel: Callable
    loss_fn: Callable
    optimizer: Callable
    alpha: float = 0.5

    def __call__(self,params,carry,batch,opt_state):
        '''
        Args:
        params: Variable collection for snnModel
        carry: Carry/state for snnModel
        batch: Tuple of the input and labels
        opt_state: Variable collection of the optimizer
        '''
        def loss_fn(params,state,batch,lam,W_bar):
            state, s = self.snnModel.apply(params,state,batch[0])
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            combine = lambda x,y,z: jnp.sum(jnp.square(x - y - (0.5/self.alpha)*z))
            
            loss = loss + (0.5*self.alpha)*jnp.sum(jnp.array(tree_leaves(tree_map(combine,{'params':params['params']},W_bar,lam))))
            #jnp.sum(ravel_pytree(tree_map(combine,{'params':params['params']},W_bar,lam))[0])
            return loss,(loss,s,state)

        def one_step(carry,batch):
            params,state,opt_state,lam,W_bar = carry
            grad,(loss,s,state) = jax.jacrev(loss_fn,has_aux=True)(params,state,batch,lam,W_bar)
            updates, opt_state = self.optimizer.update(grad,opt_state)
            params = optax.apply_updates(params, updates)
            lam = tree_map(lambda x,y,z: z - self.alpha*(x-y),params,W_bar,lam) # check if this should be updated params in one_step or original params from __call__
            W_bar = tree_map(lambda x,y,z: 0.5*(x+y) - (0.5/self.alpha)*z,W_bar,params,lam)
            return (params,state,opt_state,lam,W_bar), (s,loss)
        
        def loop(p,state,batch,opt_state):
            lam = tree_map(jnp.zeros_like,p)
            W_bar = p
            (p_update,state,opt_state,lam,W_bar),(s,loss) = jax.lax.scan(one_step,(p,state,opt_state,lam,W_bar),batch)
            return p_update,opt_state,s,loss,state
        
        return loop(params,carry,batch,opt_state)
    


class train_offline(nn.Module):
    '''
    A helper tool for easily implementing an offline training loop. It's implementation is similar to `train_online`.

    Args:
    snnModel: An initializes Flax module
    loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
    either an array to averaged or a scalar loss value.
    optimizer: An initialized optax optimizer
    '''
    snnModel: Callable
    loss_fn: Callable
    optimizer: Callable

    def __call__(self,params,carry,batch,opt_state):
        '''
        Args:
        params: Variable collection for snnModel
        carry: Carry/state for snnModel
        batch: Tuple of the input and labels
        opt_state: Variable collection of the optimizer
        '''
        def loss_fn(params,state,batch):
            p_apply = Partial(self.snnModel.apply,params)
            state, s = jax.lax.scan(p_apply,state,batch[0])
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s,state)
            
        def loop(p,state,batch,opt_state):
            grad,(loss,s,state) = jax.jacrev(loss_fn,has_aux=True)(p,state,batch)
            updates, opt_state = self.optimizer.update(grad,opt_state)
            params = optax.apply_updates(p, updates)
            return params,opt_state,s,loss,grad,state
        
        return loop(params,carry,batch,opt_state)
    
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

    params,carry,batch,opt_state = train_func_args
    reference_params = tree_unflatten(tree_structure(params),tree_leaves(reference_params))
    _,_,_,_,new_grad,_ = train_func(reference_params,carry,batch,opt_state)
    reference_grad = tree_unflatten(tree_structure(new_grad),tree_leaves(reference_params))
    return comparison_func(reference_grad,new_grad)

def recurrent(chain,model_carry):
    def execute(carry,x):
        counter = 0
        if model_carry == None:
            for mdl in chain:
                if 'v_threshold' in mdl.__annotations__.keys():
                    carry[counter],x = mdl(carry[counter],x)
                    counter += 1
                else:
                    x = mdl(x)
            carry[counter-1]['rec'] = jnp.zeros_like(x)
        else:
            x = jnp.concatenate([x,carry[len(chain)-1]['rec']])
            for mdl in chain:
                if 'v_threshold' in mdl.__annotations__.keys():
                    carry[counter],x = mdl(carry[counter],x)
                    counter += 1
                else:
                    x = mdl(x)
        return carry,x
    return execute