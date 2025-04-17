import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx
import flax
from typing import Callable
from jax.tree_util import tree_leaves, tree_map
import optax
from slax.model.utils import reinit_model, compat_scan, RNN, scan


class train_online_deferred(nnx.Module):
    '''
    A helper tool for easily implementing an online training loop where parameter updates are saved till the end.

    Args:
    snnModel: An initializes Flax module
    loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
    either an array to averaged or a scalar loss value.
    optimizer: An initialized optax optimizer
    '''
    def __init__(self,snnModel,loss_fn,optimizer,unroll=False):
        self.snnModel = snnModel
        self.loss_fn = loss_fn
        self.optimizer = nnx.Optimizer(snnModel,optimizer)
        self.unroll = unroll
        self.loss = None
        self.grad = None
        self.output = None

    @nnx.jit
    def __call__(self,batch):
        '''
        Args:
        batch: Tuple of the input and labels
        '''

        graph, param, state = nnx.split(self.snnModel,nnx.Param,...)

        def loss_fn(mdl,batch):
            s = mdl(batch[0])
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s)

        def one_step(carry,batch):
            st,grad = carry
            model = nnx.merge(graph,param,st)
            g,(loss,s) = nnx.grad(loss_fn,has_aux=True)(model,batch)
            _,_,st =  nnx.split(model,nnx.Param,...)
            grad = tree_map(lambda x,y: x+y,g,grad)
            return (st,grad), (s,loss)

        def loop(p,batch):
            grad = tree_map(jnp.zeros_like,p)
            (st,grad),(s,loss) = scan(one_step,(state,grad),batch,unroll=self.unroll)
            grad = tree_map(lambda x: x/batch[0].shape[0],grad)
            return s, loss, grad

        s, loss, grad = loop(param,batch)

        self.optimizer.update(grad)
        self.grad = nnx.Variable(grad)
        self.loss = nnx.Variable(loss)
        self.output = nnx.Variable(s)
    
class train_online(nnx.Module):
    '''
    A helper tool for easily implementing an online training loop where parameter updates are applied at each time-step.

    Args:
    snnModel: An initializes Flax module
    loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
    either an array to averaged or a scalar loss value.
    optimizer: An initialized optax optimizer
    '''
    def __init__(self,snnModel,loss_fn,optimizer,unroll=False,reset_state=True,loss_aux=False):
        self.snnModel = snnModel
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.opt_state = optimizer.init(nnx.split(snnModel,nnx.Param,...)[1])
        self.unroll = unroll
        self.loss = None
        self.grad = None
        self.output = None
        self.reset_state = reset_state
        self.loss_aux = loss_aux
    @nnx.jit
    def __call__(self,batch):
        '''
        Args:
        batch: Tuple of the input and labels
        '''
        graph, param, state = nnx.split(self.snnModel,nnx.Param,...)
        def loss_fn(mdl,batch):
            s = mdl(batch[0])
            if self.loss_aux:
                loss = self.loss_fn(s,batch[1])
                return loss[0],(loss,s)
            else:
                loss = self.loss_fn(s,batch[1])
            return loss,(loss,s)

        def one_step(carry,batch):
            st,p,opt_state = carry
            model = nnx.merge(graph,p,st)
            g,(loss,s) = nnx.grad(loss_fn,has_aux=True)(model,batch)
            _,p,st =  nnx.split(model,nnx.Param,...)
            updates, opt_state = self.optimizer.update(g,opt_state,p)
            p = optax.apply_updates(p, updates)

            return (st,p,opt_state), (s,loss)
        
        def loop(p,batch,opt_state):
            (st,param,opt_state),(s,loss) = scan(one_step,(state,p,opt_state),batch,unroll=self.unroll)
            return param,opt_state,s,loss,st
        
        p,o,s,l,state = loop(param,batch,self.opt_state)
        nnx.update(self.snnModel,p)
        if not self.reset_state:
            nnx.update(self.snnModel,state)
        self.opt_state = nnx.Variable(o)
        self.loss = nnx.Variable(l)
        self.output = nnx.Variable(s)

class train_online(nnx.Module):
    '''
    A helper tool for easily implementing an online training loop where parameter updates are applied at each time-step.

    Args:
    snnModel: An initializes Flax module
    loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
    either an array to averaged or a scalar loss value.
    optimizer: An initialized optax optimizer
    '''
    def __init__(self,snnModel,loss_fn,optimizer,unroll=False,reset_state=True,loss_aux=False):
        self.snnModel = snnModel
        self.loss_fn = loss_fn
        self.optimizer = nnx.Optimizer(snnModel,optimizer)
        self.unroll = unroll
        self.loss = None
        self.grad = None
        self.output = None
        self.reset_state = reset_state
        self.loss_aux = loss_aux
    @nnx.jit
    def __call__(self,batch):
        '''
        Args:
        batch: Tuple of the input and labels
        '''
        graph, param, state = nnx.split(self.snnModel,nnx.Param,...)
        opt_graph,opt_st = nnx.split(self.optimizer)
        def loss_fn(mdl,batch):
            s = mdl(batch[0])
            if self.loss_aux:
                loss = self.loss_fn(s,batch[1])
                return jnp.mean(loss[0]),(loss,s)
            else:
                loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s)

        def one_step(carry,batch):
            opt_s = carry
            optim = nnx.merge(opt_graph,opt_s)
            g,(loss,s) = nnx.grad(loss_fn,has_aux=True)(optim.model,batch)
            optim.update(g)
            _,opt_s =  nnx.split(optim)
            return opt_s, (s,loss)
        
        def loop(batch,opt_s):
            opt_s,(s,loss) = scan(one_step,opt_s,batch,unroll=self.unroll)
            return opt_s,s,loss
        
        o,s,l = loop(batch,opt_st)
        nnx.update(self.optimizer,o)
        if self.reset_state:
            nnx.update(self.snnModel,state)
        self.loss = nnx.Variable(l)
        self.output = nnx.Variable(s)

# class FPTT(nnx.Module):
#     '''
#     A helper tool for easily implementing an online training loop where parameter updates are applied at each time-step.

#     Args:
#     snnModel: An initializes Flax module
#     loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
#     either an array to averaged or a scalar loss value.
#     optimizer: An initialized optax optimizer
#     '''
#     def __init__(self,snnModel,loss_fn,optimizer,alpha=0.5,unroll=False):
#         self.snnModel = snnModel
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer
#         self.opt_state = optimizer.init(nnx.split(snnModel,nnx.Param,...)[1])
#         self.alpha = alpha
#         self.unroll = unroll
#         self.loss = None
#         self.grad = None
#         self.output = None
#     @nnx.jit
#     def __call__(self,batch):
#         '''
#         Args:
#         batch: Tuple of the input and labels
#         '''
#         graph, param, state = nnx.split(self.snnModel,nnx.Param,...)
#         def loss_fn(mdl,batch,lam,W_bar):
#             params = nnx.split(mdl,nnx.Param,...)[1]
#             s = mdl(batch[0])
#             loss = jnp.mean(self.loss_fn(s,batch[1]))
#             combine = lambda x,y,z: x - y - (0.5/self.alpha)*z
#             loss += (0.5*self.alpha)*optax.tree_utils.tree_l2_norm(tree_map(combine,params,W_bar,lam),squared=True)
#             return loss,(loss,s)

#         def one_step(carry,batch):
#             st,p,opt_state,lam,W_bar = carry
#             model = nnx.merge(graph,p,st)
#             g,(loss,s) = nnx.grad(loss_fn,has_aux=True)(model,batch,lam,W_bar)
#             _,p,st =  nnx.split(model,nnx.Param,...)
#             updates, opt_state = self.optimizer.update(g,opt_state,p)
#             p = optax.apply_updates(p, updates)
#             lam = tree_map(lambda x,y,z: z - self.alpha*(x-y),p,W_bar,lam) # check if this should be updated params in one_step or original params from __call__
#             W_bar = tree_map(lambda x,y,z: 0.5*(x+y) - (0.5/self.alpha)*z,W_bar,p,lam)

#             return (st,p,opt_state,lam,W_bar), (s,loss)
        
#         def loop(p,batch,opt_state):
#             lam = tree_map(jnp.zeros_like,p)
#             W_bar = p
#             (_,param,opt_state,_,_),(s,loss) = compat_scan(one_step,(state,p,opt_state,lam,W_bar),batch,unroll=self.unroll)
#             return param,opt_state,s,loss
        
#         p,o,s,l = loop(param,batch,self.opt_state)
    
#         nnx.update(self.snnModel,p)
#         self.opt_state = o
#         self.loss = l
#         self.output = s

class FPTT(nnx.Module):
    '''
    A helper tool for easily implementing an online training loop where parameter updates are applied at each time-step.

    Args:
    snnModel: An initializes Flax module
    loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
    either an array to averaged or a scalar loss value.
    optimizer: An initialized optax optimizer
    '''
    def __init__(self,snnModel,loss_fn,optimizer,alpha=0.5,unroll=False,reset_state=True):
        self.snnModel = snnModel
        self.loss_fn = loss_fn
        self.optimizer = self.optimizer = nnx.Optimizer(snnModel,optimizer)
        self.alpha = alpha
        self.unroll = unroll
        self.loss = None
        self.grad = None
        self.output = None
        self.reset_state = reset_state
    @nnx.jit
    def __call__(self,batch):
        '''
        Args:
        batch: Tuple of the input and labels
        '''
        graph, param, state = nnx.split(self.snnModel,nnx.Param,...)
        opt_graph,opt_st = nnx.split(self.optimizer)
        def loss_fn(mdl,batch,lam,W_bar):
            params = nnx.split(mdl,nnx.Param,...)[1]
            s = mdl(batch[0])
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            combine = lambda x,y,z: x - y - (0.5/self.alpha)*z
            loss += (0.5*self.alpha)*optax.tree_utils.tree_l2_norm(tree_map(combine,params,W_bar,lam),squared=True)
            return loss,(loss,s)

        def one_step(carry,batch):
            opt_s,lam,W_bar = carry
            optim = nnx.merge(opt_graph,opt_s)
            g,(loss,s) = nnx.grad(loss_fn,has_aux=True)(optim.model,batch,lam,W_bar)
            _,p,_ =  nnx.split(optim.model,nnx.Param,...)
            optim.update(g)
            _,opt_s =  nnx.split(optim)
            lam = tree_map(lambda x,y,z: z - self.alpha*(x-y),p,W_bar,lam) # check if this should be updated params in one_step or original params from __call__
            W_bar = tree_map(lambda x,y,z: 0.5*(x+y) - (0.5/self.alpha)*z,W_bar,p,lam)

            return (opt_s,lam,W_bar), (s,loss)
        
        def loop(p,batch,opt_s):
            lam = tree_map(jnp.zeros_like,p)
            W_bar = p
            (opt_s,_,_),(s,loss) = scan(one_step,(opt_s,lam,W_bar),batch,unroll=self.unroll)
            return opt_s,s,loss


        o,s,l = loop(param,batch,opt_st)
        nnx.update(self.optimizer,o)
        if self.reset_state:
            nnx.update(self.snnModel,state)
        self.loss = nnx.Variable(l)
        self.output = nnx.Variable(s)

class train_offline(nnx.Module):
    '''
    A helper tool for easily implementing an online training loop where parameter updates are saved till the end.

    Args:
    snnModel: An initializes Flax module
    loss_fn: A loss function that takes the model output (excluding carry) and the batch labels as arguments and returns
    either an array to averaged or a scalar loss value.
    optimizer: An initialized optax optimizer
    '''
    def __init__(self,snnModel,loss_fn,optimizer,unroll=False,scan=True):
        self.snnModel = snnModel
        self.loss_fn = loss_fn
        self.optimizer = nnx.Optimizer(snnModel,optimizer)
        self.unroll = unroll
        self.scan = scan
        self.loss = None
        self.grad = None
        self.output = None

    @nnx.jit
    def __call__(self,batch):
        '''
        Args:
        batch: Tuple of the input and labels
        '''

        graph, param, state = nnx.split(self.snnModel,nnx.Param,...)

        def scan_loss_fn(mdl,batch):
            s = RNN(mdl,unroll=self.unroll)(batch[0])
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s)

        def loss_fn(mdl,batch):
            s = mdl(batch[0])
            loss = jnp.mean(self.loss_fn(s,batch[1]))
            return loss,(loss,s)

        def one_step(st,batch):
            model = nnx.merge(graph,param,st)
            if self.scan:
                g,(loss,s) = nnx.grad(scan_loss_fn,has_aux=True)(model,batch)
            else:
                g,(loss,s) = nnx.grad(loss_fn,has_aux=True)(model,batch)
            _,_,st =  nnx.split(model,nnx.Param,...)
            return g,s,loss

        def loop(batch):
            grad,s,loss = one_step(state,batch)
            return s, loss, grad

        s, loss, grad = loop(batch)

        self.optimizer.update(grad)
        self.grad = nnx.Variable(grad)
        self.loss = nnx.Variable(loss)
        self.output = nnx.Variable(s)