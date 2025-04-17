import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax import nnx
from typing import Any, Sequence
from jax.tree_util import Partial, tree_leaves, tree_map
from jax.lax import stop_gradient as stop_grad
from jax.flatten_util import ravel_pytree
from slax.model.utils import reinit_model, connect, RNN
from slax.train.helpers import output, sum_output, diag, rtrl_update, diag_rtrl_update
from functools import partial

def map_stack(x,len_state):
    p_sum = lambda x,y: jnp.sum(x,axis=list(range(-1,-y -1,-1)))
    out = jnp.stack(jax.tree.leaves(jax.tree.map(p_sum,x,len_state)))
    return out

def map_stack2(x,len_state):
    p_sum = lambda y: jnp.sum(x,axis=list(range(-1,-y -1,-1)))
    out = jnp.stack(jax.tree.leaves(jax.tree.map(p_sum,len_state)))
    return out

def sum_last(x):
    return jnp.stack(jax.tree.leaves(jax.tree.map(partial(jnp.sum,axis=-1),(x,))))

def sum_output(graph,param,state,x):
    len_state = jax.tree.map(lambda x: len(x.shape),state)
    def forward(x):
        model = nnx.merge(graph,param,state)
        out = model(x)
        return out, nnx.split(model,nnx.Param,...)[2]
    (out,f_vjp,state) = jax.vjp(forward,x,has_aux=True)
    return (sum_last(out),sum_last(state),map_stack2(out,len_state),map_stack(state,len_state)),(out,state,f_vjp)

def d_sum(a,b):
    l = len(a.shape)
    l1 = list(range(l))
    l2 = [0]+list(range(2,l-1))+[Ellipsis]+[l-1]
    l3 = list(range(1,l-1))+[Ellipsis]+[l-1]
    return jnp.einsum(a,list(range(l)),b,[0]+list(range(2,l-1))+[Ellipsis]+[l-1],list(range(1,l-1))+[Ellipsis]+[l-1])

def d2_sum(a,b):
    l = len(a.shape)
    out = jnp.einsum(a,list(range(l)),b,list(range(l-1))+[Ellipsis]+[l-1],[Ellipsis]+[l-1])
    return out

def diag_rtrl_update(x,y,z):
    b = jax.tree.map(lambda r,t: r+0*t,y,z)
    out = jax.tree.map(lambda inp, ds: d_sum(x,inp) + ds,b,z)
    return out

def calc_grad(x,y):
    out = jax.tree.map(lambda inp: d2_sum(x,inp),y)
    return out

def calc_E(state,params):
    shape = jnp.stack(jax.tree.leaves(state)).shape[:-1]
    E = jax.tree.map(lambda x: jnp.zeros(shape + x.shape),params)
    return E


class diag_rtrl(nnx.Module):
    def __init__(self, mdl, diagonal_sum=True):
        self.mdl = mdl
        graph,param,state = nnx.split(mdl,nnx.Param,...)
        self.E = nnx.Variable(jax.tree.map(lambda x: jnp.array([0.],dtype=x.dtype),param))
        self.diagonal_sum = diagonal_sum
    @nnx.jit
    def __call__(self,x):
        @jax.custom_vjp
        def exec_model(graph,param,state,E,x):
            model = nnx.merge(graph,param,state)
            out = model(x)
            state = nnx.split(model,nnx.Param,...)[2]
            E = calc_E(state,param)
            return out, state, E
        def exec(graph,param,state,E,x):
            if self.diagonal_sum:
                ((ds_dp,_),(du_dp,_),(_,ds_du),(_,du_du)),(out,state,f_vjp) = jax.jacrev(sum_output,argnums=[1,2],has_aux=True)(graph,param,state,x)
            else:
                grads,(out,state,f_vjp) = jax.jacrev(output,argnums=[1,2],has_aux=True)(graph,param,state,x)
                (ds_grad),(du_dp,du_du) = jax.tree.map(diag,grads)
                ds_dp,ds_du = jax.tree.map(lambda x: x[0],ds_grad)
            
            ds_du, du_du = jnp.stack(jax.tree.leaves(ds_du)), jnp.stack(jax.tree.leaves(du_du))
        
            ds_dtheta = diag_rtrl_update(ds_du,E,ds_dp)
            E = diag_rtrl_update(du_du,E,du_dp)
            return (out,state,E), (ds_dtheta,f_vjp)

        def exec_bwd(res,g):
            ds_dtheta,f_vjp = res
            gr = jnp.stack(jax.tree.leaves((g[0],)))
            grad = calc_grad(gr,ds_dtheta)
            passback = jax.tree.leaves(f_vjp(g[0]))[0]

            return (None,grad,None,None,passback)
        
        exec_model.defvjp(exec, exec_bwd)
        graph, param, state = nnx.split(self.mdl,nnx.Param,...)
        out,state,E = exec_model(graph,param,state,self.E.value,x)
        self.E.value = E
        nnx.update(self.mdl,state)
        return out