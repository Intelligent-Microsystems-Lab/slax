import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable
from jax.tree_util import tree_leaves, tree_map
import optax
from ..models.utils import reinit_model, SNNCell

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

    def __call__(self,params,batch,opt_state,return_grad=False):
        '''
        Args:
        params: Variable collection for snnModel
        carry: Carry/state for snnModel
        batch: Tuple of the input and labels
        opt_state: Variable collection of the optimizer
        '''

        def loss_fn(params,state,batch):
            params = {'params':params,'carry':state}
            s,state_upd = self.snnModel.apply(params,batch[0],mutable='carry')
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s,state_upd)

        def one_step(carry,batch):
            params,opt_state,grad = carry
            g,(loss,s,state_upd) = jax.jacrev(loss_fn,has_aux=True)(params['params'],params['carry'],batch)
            grad = tree_map(lambda x,y: x+y,g,grad)
            params.update(state_upd)
            return (params,opt_state,grad), (s,loss)
        
        def loop(p,batch,opt_state):
            grad = tree_map(jnp.zeros_like,p['params'])
            (p,opt_state,grad),(s,loss) = jax.lax.scan(one_step,(p,opt_state,grad),batch,unroll=jnp.iinfo(jnp.uint32).max)
            grad = tree_map(lambda x: x/batch[0].shape[1],grad)
            updates, opt_state = self.optimizer.update(grad,opt_state)
            p['params'] = optax.apply_updates(p['params'], updates)
            p['carry'] = tree_map(jnp.zeros_like,p['carry'])
            if return_grad:
                return p,opt_state,s,loss,grad
            else:
                return p,opt_state,s,loss
        
        return loop(params,batch,opt_state)
    
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

    def __call__(self,params,batch,opt_state):
        '''
        Args:
        params: Variable collection for snnModel
        carry: Carry/state for snnModel
        batch: Tuple of the input and labels
        opt_state: Variable collection of the optimizer
        '''

        def loss_fn(params,state,batch):
            params = {'params':params,'carry':state}
            s,state_upd = self.snnModel.apply(params,batch[0],mutable='carry')
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s,state_upd)

        def one_step(carry,batch):
            params,opt_state = carry
            grad,(loss,s,state_upd) = jax.jacrev(loss_fn,has_aux=True)(params['params'],params['carry'],batch)
            updates, opt_state = self.optimizer.update(grad,opt_state)
            params['params'] = optax.apply_updates(params['params'], updates)
            params.update(state_upd)
            return (params,opt_state), (s,loss)
        
        def loop(p,batch,opt_state):
            (p,opt_state),(s,loss) = jax.lax.scan(one_step,(p,opt_state),batch,unroll=jnp.iinfo(jnp.uint32).max)
            p['carry'] = tree_map(jnp.zeros_like,p['carry'])
            return p,opt_state,s,loss
        
        return loop(params,batch,opt_state)
    

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

    def __call__(self,params,batch,opt_state,unroll=jnp.iinfo(jnp.uint32).max):
        '''
        Args:
        params: Variable collection for snnModel
        carry: Carry/state for snnModel
        batch: Tuple of the input and labels
        opt_state: Variable collection of the optimizer
        '''
        
        def loss_fn(params,state,batch,lam,W_bar):
            p = {'params':params,'carry':state}
            s,state_upd = self.snnModel.apply(p,batch[0],mutable='carry')
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            #combine = lambda x,y,z: jnp.sum(jnp.square(x - y - (0.5/self.alpha)*z))
            combine = lambda x,y,z: x - y - (0.5/self.alpha)*z
            
            #loss = loss + (0.5*self.alpha)*jnp.sum(jnp.array(tree_leaves(tree_map(combine,params,W_bar,lam))))
            loss = loss + (0.5*self.alpha)*optax.tree_utils.tree_l2_norm(tree_map(combine,params,W_bar,lam),squared=True)
            #jnp.sum(ravel_pytree(tree_map(combine,{'params':params['params']},W_bar,lam))[0])
            return loss,(loss,s,state_upd)

        def one_step(carry,batch):
            params,opt_state,lam,W_bar = carry
            grad,(loss,s,state_upd) = jax.jacrev(loss_fn,has_aux=True)(params['params'],params['carry'],batch,lam,W_bar)
            updates, opt_state = self.optimizer.update(grad,opt_state)
            params['params'] = optax.apply_updates(params['params'], updates)
            lam = tree_map(lambda x,y,z: z - self.alpha*(x-y),params['params'],W_bar,lam) # check if this should be updated params in one_step or original params from __call__
            W_bar = tree_map(lambda x,y,z: 0.5*(x+y) - (0.5/self.alpha)*z,W_bar,params['params'],lam)
            params.update(state_upd)
            return (params,opt_state,lam,W_bar), (s,loss)
        
        def loop(p,batch,opt_state):
            lam = tree_map(jnp.zeros_like,p['params'])
            W_bar = p['params']
            (p_update,opt_state,lam,W_bar),(s,loss) = jax.lax.scan(one_step,(p,opt_state,lam,W_bar),batch,unroll=unroll)
            p_update['carry'] = tree_map(jnp.zeros_like,p_update['carry'])
            return p_update,opt_state,s,loss
        
        return loop(params,batch,opt_state)
    


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

    def __call__(self,params,batch,opt_state,return_grad=False,unroll=jnp.iinfo(jnp.uint32).max):
        '''
        Args:
        params: Variable collection for snnModel
        carry: Carry/state for snnModel
        batch: Tuple of the input and labels
        opt_state: Variable collection of the optimizer
        '''
        def apply(p,xs):
            s,upd = self.snnModel.apply(p,xs,mutable='carry')
            p.update(upd)
            return p,s
        def loss_fn(params,state,batch):
            p = {'params':params,'carry':state}
            p,s = jax.lax.scan(apply,p,batch[0],unroll=unroll)
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s,p)
            
        def loop(p,batch,opt_state):
            grad,(loss,s,p) = jax.jacrev(loss_fn,has_aux=True)(p['params'],p['carry'],batch)
            updates, opt_state = self.optimizer.update(grad,opt_state)
            p['params'] = optax.apply_updates(p['params'], updates)
            p['carry'] = tree_map(jnp.zeros_like,p['carry'])
            if return_grad:
                return p,opt_state,s,loss,grad
            else:
                return p,opt_state,s,loss
        
        return loop(params,batch,opt_state)
    

class flax_wrapper(SNNCell):
    '''
    A helper tool that takes a Flax RNN (which handles the carry state explicitly) and returns a module with the same function but hides the state.
    This makes it compatible with Slax utilities.

    Args:
    mdl: The Flax RNN to convert.
    '''
    mdl: Callable
    @nn.compact
    def __call__(self,carry,x=None):
        m = reinit_model(self.mdl)
        if x == None:
            x = carry
            carry = self.variable('carry','flax',m.initialize_carry,jax.random.PRNGKey(0),x.shape)
            c = carry.value
            hidden_carry = True
        else:
            c = carry['flax']
            hidden_carry = False

        c,x = m(c,x)

        if hidden_carry:
            carry.value = c
            return x
        else:
            carry['flax'] = x
            return carry, x