import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx
from typing import Any, Sequence
from jax.tree_util import Partial, tree_leaves, tree_map
from jax.lax import stop_gradient as stop_grad
from jax.flatten_util import ravel_pytree
from slax.model.utils import reinit_model, connect, RNN
from slax.train.helpers import output, sum_output, diag, diag_rtrl_update, rtrl_update
from functools import partial

def forw(graph,params,state,x,a_hat,leak):
    model = nnx.merge(graph,params,state)
    out = model(x)
    a_hat += jnp.zeros_like(x)
    return out, nnx.split(model,nnx.Param,...)[2], a_hat

@jax.custom_vjp
def forward(graph,params,state,x,a_hat,leak):
    model = nnx.merge(graph,params,state)
    out = model(x)
    a_hat += jnp.zeros_like(x)
    return out, nnx.split(model,nnx.Param,...)[2], a_hat

def f_fwd(graph,params,state,x,a_hat,leak):
    output,f_vjp = jax.vjp(forw,graph,params,state,x,a_hat,leak)
    leaves,tree = jax.tree.flatten(f_vjp)
    a_hat = leak*a_hat + x
    leaves[-1] = a_hat
    f_vjp = jax.tree.unflatten(tree,leaves)
    return (output[0], output[1], a_hat), f_vjp

def f_bwd(res,g):
    f_vjp = res
    return f_vjp(g)

forward.defvjp(f_fwd,f_bwd)

class OTTT(nnx.Module):
    '''
    An implementation of Online Training Through Time (OTTT) from Xiao, M., Meng, Q., Zhang, Z., He, D., & Lin, Z. (2022). Online training through time for spiking neural networks. Advances in neural information processing systems, 35, 20717-20730.

    Args:
    chain: A sequence of initialized Flax modules. This is somewhat similar to nn.Sequential, but the last module must be the
    module to take a carry variable.

    leak: A value (between 0 and 1) that is the proportion of the state that remains after one time-step
    '''
    def __init__(self, mdl, leak = nnx.sigmoid(2.)):
        self.mdl = mdl
        self.leak = nnx.Variable(leak)
        self.a_hat = nnx.Variable(jnp.zeros(1))

    def __call__(self,x):
        g,p,s = nnx.split(self.mdl,nnx.Param,...)
        y,s,self.a_hat.value = forward(g,p,s,x,self.a_hat.value,self.leak.value)
        nnx.update(self.mdl,s)

        return y