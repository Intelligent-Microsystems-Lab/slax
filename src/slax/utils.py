import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Sequence
from jax.tree_util import Partial, tree_leaves, tree_structure, tree_unflatten, tree_map
from .neurons import LIF, LTC
from jax.lax import stop_gradient as stop_grad
#from utils import train_online, FPTT, train_offline, train_online_deferred
import optax
from flax.core.frozen_dict import unfreeze, freeze
from jax.flatten_util import ravel_pytree
from .surrogates import fast_sigmoid
#import randman_dataset as rd

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

