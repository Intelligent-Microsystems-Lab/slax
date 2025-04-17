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


class diag_rtrl(nnx.Module):
    def __init__(self, mdl, batch_sz, diagonal_sum=True):
        self.mdl = mdl
        graph,param,state = nnx.split(mdl,nnx.Param,...)
        E = [jax.tree.map(jnp.zeros_like,param)]*len(jax.tree.leaves(state))
        self.E = nnx.Variable(jax.tree.map(lambda *args: jnp.stack(batch_sz*[jnp.stack(args)]),*E))
        self.diagonal_sum = diagonal_sum
    @nnx.jit
    def __call__(self,x):
        @jax.custom_vjp
        def exec_model(graph,param,state,E,x):
            model = nnx.merge(graph,param,state)
            out = model(x)
            return out, nnx.split(model,nnx.Param,...)[2], E
        
        def exec(graph,param,state,E,x):
            if self.diagonal_sum:
                ((ds_dp,ds_du),(du_dp,du_du)),(out,state,f_vjp) = jax.jacrev(sum_output,argnums=[1,2],has_aux=True)(graph,param,state,x)
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
            grad = jax.tree.map(lambda x: g[0]*x,ds_dtheta)
            passback = jax.tree.leaves(f_vjp(g[0]))[0]
            return (None,grad,None,None,passback)
        
        exec_model.defvjp(exec, exec_bwd)
        graph, param, state = nnx.split(self.mdl,nnx.Param,...)
        vp_exec = partial(jax.vmap(partial(exec_model,graph,param)),state,self.E.value)
        if len(x.shape)<2:
            x = x.reshape(1,-1)
        out,state,E = vp_exec(x)
        self.E.value = E
        nnx.update(self.mdl,state)
        return out
    
class OTPE(nnx.Module):
    def __init__(self, mdl, batch_sz, input_sz, leak, diagonal_sum=True):
        self.mdl = mdl
        graph,param,state = nnx.split(mdl,nnx.Param,...)
        E = [jax.tree.map(jnp.zeros_like,param)]*len(jax.tree.leaves(state))
        self.E = nnx.Variable(jax.tree.map(lambda *args: jnp.stack(batch_sz*[jnp.stack(args)]),*E))
        self.R_hat = nnx.Variable(jax.tree.map(lambda *args: jnp.stack(batch_sz*[jnp.stack(args)]),*E))
        self.g_bar = nnx.Variable(jnp.zeros(input_sz))
        self.r = nnx.Variable(jnp.zeros(1))
        self.leak = leak
        self.diagonal_sum = diagonal_sum

    @nnx.jit
    def __call__(self,x):
        @jax.custom_vjp
        def exec_model(graph,param,state,E,x):
            model = nnx.merge(graph,param,state)
            out = model(x)
            return out, nnx.split(model,nnx.Param,...)[2], E
        
        def exec(graph,param,state,E,x):
            if self.diagonal_sum:
                ((ds_dp,ds_du),(du_dp,du_du)),(out,state,f_vjp) = jax.jacrev(sum_output,argnums=[1,2],has_aux=True)(graph,param,state,x)
            else:
                grads,(out,state,f_vjp) = jax.jacrev(output,argnums=[1,2],has_aux=True)(graph,param,state,x)
                (ds_grad),(du_dp,du_du) = jax.tree.map(diag,grads)
                ds_dp,ds_du = jax.tree.map(lambda x: x[0],ds_grad)
            ds_du, du_du = jnp.stack(jax.tree.leaves(ds_du)), jnp.stack(jax.tree.leaves(du_du))
            ds_dtheta = diag_rtrl_update(ds_du,E,ds_dp)
            E = diag_rtrl_update(du_du,E,du_dp)
            R_hat = jax.tree_map(lambda x,y: self.leak*x + y, R_hat, ds_dtheta)
            ratio = self.leak*r
            r = ratio + 1
            ratio = (ratio/r)
            g_bar = ratio*g_bar + (1-ratio)*ds_du/self.leak

            return (out,state,E,R_hat,g_bar,r), (ds_dtheta,f_vjp)

        def exec_bwd(res,g):
            ds_dtheta,f_vjp = res
            grad = jax.tree.map(lambda x: g[0]*x,ds_dtheta)
            passback = jax.tree.leaves(f_vjp(g[0]))[0]
            return (None,grad,None,None,passback)
        
        exec_model.defvjp(exec, exec_bwd)
        graph, param, state = nnx.split(self.mdl,nnx.Param,...)
        vp_exec = partial(jax.vmap(partial(exec_model,graph,param)),state,self.E.value)
        if len(x.shape)<2:
            x = x.reshape(1,-1)
        out,state,E,R_hat,g_bar,r = vp_exec(x)
        self.E.value = E
        self.R_hat.value = R_hat
        self.g_bar.value = g_bar
        self.E.value = E
        self.r.value = r
        nnx.update(self.mdl,state)
        return out
    
class OTTT(nnx.Module):
    '''
    An implementation of Online Training Through Time (OTTT) from Xiao, M., Meng, Q., Zhang, Z., He, D., & Lin, Z. (2022). Online training through time for spiking neural networks. Advances in neural information processing systems, 35, 20717-20730.

    Args:
    chain: A sequence of initialized Flax modules. This is somewhat similar to nn.Sequential, but the last module must be the
    module to take a carry variable.

    leak: A value (between 0 and 1) that is the proportion of the state that remains after one time-step
    '''
    def __init__(self, mdl, input_sz, leak):
        self.mdl = mdl
        self.leak = nnx.Variable(leak)
        self.a_hat = nnx.Variable(jnp.zeros(input_sz))

    @nnx.jit
    def __call__(self,x):
        @jax.custom_vjp
        def exec_model(graph,param,leak,state,a_hat,x):
            model = nnx.merge(graph,param,state)
            out = model(x)
            return out, nnx.split(model,nnx.Param,...)[2], a_hat
        
        def exec(graph,param,leak,state,a_hat,x):
            _,(out,state,f_vjp) = output(graph,param,state,x)
            a_hat = leak*a_hat + x
            return (out,state,a_hat), (a_hat,f_vjp)
        
        def apply_a_hat(g,a_hat,params):
            def use_a_hat(x):
                if len(x.shape)>1:
                    return jnp.outer(g,a_hat)
                else:
                    return g
            return jax.tree.map(use_a_hat,params)
    
        def exec_bwd(res,g):
            a_hat,f_vjp = res
            params = nnx.split(self.mdl,nnx.Param,...)[1]
            grad = apply_a_hat(g[0],a_hat,params)
            passback = jax.tree.leaves(f_vjp(g[0]))[0]
            return (None,grad,None,None,None,passback)
        
        exec_model.defvjp(exec, exec_bwd)
        graph, param, state = nnx.split(self.mdl,nnx.Param,...)
        vp_exec = partial(jax.vmap(partial(exec_model,graph,param,self.leak.value)),state,self.a_hat.value)
        if len(x.shape)<2:
            x = x.reshape(1,-1)
        out,state,a_hat = vp_exec(x)
        self.a_hat.value = a_hat
        nnx.update(self.mdl,state)
        return out

class rtrl(nnx.Module):
    def __init__(self, mdl, input_sz, diagonal_sum=False):
        '''
        An implementation of Real Time Recurrent Learing (RTRL) from Williams, R. J., & Zipser, D. (1989). A learning algorithm for continually running fully recurrent neural networks. Neural computation, 1(2), 270-280.

        Args:
        chain: A sequence of initialized Flax modules. This is somewhat similar to nn.Sequential, but the last module must be the
        module to take a carry variable.
        '''
        self.mdl = mdl
        graph,param,state = nnx.split(mdl,nnx.Param,...)
        E = [jax.tree.map(jnp.zeros_like,param)]*len(jax.tree.leaves(state))
        self.E = nnx.Variable(jax.tree.map(lambda *args: jnp.zeros(input_sz + jnp.stack(args).shape),*E))
        self.diagonal_sum = diagonal_sum
    @nnx.jit
    def __call__(self,x):
        @jax.custom_vjp
        def exec_model(graph,param,state,E,x):
            model = nnx.merge(graph,param,state)
            out = model(x)
            return out, nnx.split(model,nnx.Param,...)[2], E
        
        def exec(graph,param,state,E,x):
            grads,(out,state,f_vjp) = jax.jacrev(output,argnums=[1,2],has_aux=True)(graph,param,state,x)
            (ds_dp,ds_du),(du_dp,du_du) = grads
            ds_du, du_du = jnp.stack(jax.tree.leaves(ds_du)), jnp.stack(jax.tree.leaves(du_du))
            ds_dtheta = rtrl_update(ds_du,E,ds_dp)
            E = rtrl_update(du_du,E,du_dp)
            return (out,state,E), (ds_dtheta,f_vjp)

        def exec_bwd(res,g):
            ds_dtheta,f_vjp = res
            grad = jax.tree.map(lambda x: g[0].dot(x),ds_dtheta)
            passback = jax.tree.leaves(f_vjp(g[0]))[0]
            return (None,grad,None,None,passback)
        
        exec_model.defvjp(exec, exec_bwd)
        graph, param, state = nnx.split(self.mdl,nnx.Param,...)
        vp_exec = partial(jax.vmap(partial(exec_model,graph,param)),state,self.E.value)
        if len(x.shape)<2:
            x = x.reshape(1,-1)
        out,state,E = vp_exec(x)
        self.E.value = E
        nnx.update(self.mdl,state)
        return out