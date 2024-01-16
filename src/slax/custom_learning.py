import jax
import flax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Sequence
from .surrogates import fast_sigmoid
from jax.tree_util import Partial, tree_leaves, tree_map, tree_unflatten, tree_flatten
from .neurons import LIF
from jax.lax import stop_gradient as stop_grad
from flax.core.frozen_dict import unfreeze, freeze
from jax.flatten_util import ravel_pytree
    

def reinit_model(mdl):
    d = {}
    for kw in mdl.__annotations__.keys():
        if kw != 'parent':
            d[kw] = mdl.__getattr__(kw)
    return mdl.__class__(**d)

class DenseOSTL(nn.Module):
    '''
    An efficient implementation of Online Spatio-Temporal Learning (OSTL) from Bohnstingl, T., Woźniak, S., Pantazi, A., & Eleftheriou, E. (2022). Online spatio-temporal learning in deep neural networks. IEEE Transactions on Neural Networks and Learning Systems.

    This implementation only works for dense layers (not convolutional layers).

    Args:
    chain: A sequence of initialized Flax modules. This is somewhat similar to nn.Sequential, but the last module must be the
    module to take a carry variable.
    '''
    chain: Sequence

    @nn.compact
    def __call__(self,carry,x):
        model = self.chain
        class chain(nn.Module):
            @nn.compact
            def __call__(self,carry,x):
                for m in model[:-1]:
                    x=reinit_model(m)(x)
                x = reinit_model(model[-1])(carry,x)
                
                return x

        def f(chain,carry,x):
            carry['chain'],x = chain(carry['chain'],x)
            return carry,x
        
        def summed_output(p,chain,carry,x):
            carry,spikes = chain.apply(p,carry,x)
            return (jnp.sum(spikes),jnp.sum(carry['Vmem']),spikes),(carry,spikes)

        def f_fwd(chain,carry,x):
            ((ds_dp,ds_du,_),(du_dp,du_du,_),(_,_,k)),(carry['chain'],spikes) = jax.jacrev(summed_output,has_aux=True,argnums=(0,2,3))(chain.variables,chain,carry['chain'],x)
            
            
            ds_du = ds_du['Vmem']
            ds_dparams = ds_dp

            du_du = du_du['Vmem']
            du_dparams = du_dp

            ds_dtheta = jax.tree_map(lambda x,y: ds_du*x + y,carry['E'], ds_dparams)
            carry['E'] = jax.tree_map(lambda x,y: du_du*x + y, carry['E'], du_dparams)

            return (carry,spikes),(ds_dtheta,k)
        
        def f_bwd(res,g):
            ds_dtheta,k = res
            grads = jax.tree_map(lambda x: g[1]*x,ds_dtheta)
            return {'params': grads['params']},None,g[1].dot(k)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        m = chain()
        wait=0
        if carry==None:
            wait = 1
            carry = {'chain': None}
        if len(x.shape)>1:
            pf = Partial(f_custom,m)
            carry,spikes = jax.vmap(pf)(carry,x)
        else:
            carry,spikes = f_custom(m,carry,x)
        if wait==1:
            if len(x.shape)>1:
                sz = lambda var: jnp.stack([jnp.zeros_like(var)]*x.shape[0],axis=0)
            else:
                sz = jnp.zeros_like
            carry['E'] = jax.tree_map(sz,m.variables)
        return carry,spikes
    

class DenseOTPE(nn.Module):
    chain: Sequence
    leak: float

    @nn.compact
    def __call__(self,carry,x):
        model = self.chain
        class chain(nn.Module):
            @nn.compact
            def __call__(self,carry,x):
                for m in model:
                    x=m()(carry,x)
                return x

        def f(chain,carry,x):
            carry['chain'],x = chain(carry['chain'],x)
            return carry,x
        
        def summed_output(p,chain,carry,x):
            carry,spikes = chain.apply(p,carry,x)
            return (jnp.sum(spikes),jnp.sum(carry['Vmem']),spikes),(carry,spikes)

        def f_fwd(chain,carry,x):
            ((ds_dp,ds_du,_),(du_dp,du_du,_),(_,_,k)),(carry['chain'],spikes) = jax.jacrev(summed_output,has_aux=True,argnums=(0,2,3))(chain.variables,chain,carry['chain'],x)
            
            
            ds_du = ds_du['Vmem']
            ds_dparams = ds_dp

            du_du = du_du['Vmem']
            du_dparams = du_dp

            ds_dtheta = jax.tree_map(lambda x,y: ds_du*x + y,carry['E'], ds_dparams)
            carry['E'] = jax.tree_map(lambda x,y: du_du*x + y, carry['E'], du_dparams)
            carry['R_hat'] = jax.tree_map(lambda x,y: ds_du*x + y,carry['R_hat'],ds_dtheta)

            ratio = self.leak*carry['ratio']
            carry['ratio'] = ratio + 1
            ratio = (ratio/carry['ratio'])

            carry['passback'] = ratio*carry['passback'] + k

            return (carry,spikes),(carry['R_hat'],carry['passback'])
        
        def f_bwd(res,g):
            R_hat,passback = res
            grads = jax.tree_map(lambda x: g[1]*x,R_hat)
            return {'params': freeze(grads['params'])},None,g[1].dot(passback)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        m = chain()
        wait=0
        if carry==None:
            wait = 1
            carry = {'chain': None}
        if len(x.shape)>1:
            pf = Partial(f_custom,m)
            carry,spikes = jax.vmap(pf)(carry,x)
        else:
            carry,spikes = f_custom(m,carry,x)
        if wait==1:
            if len(x.shape)>1:
                sz = lambda var: jnp.stack([jnp.zeros_like(var)]*x.shape[0],axis=0)
            else:
                sz = jnp.zeros_like
            carry['E'] = jax.tree_map(sz,m.variables)
            carry['R_hat'] = jax.tree_map(sz,m.variables)
            carry['ratio'] = jnp.sum(jnp.zeros_like(x),axis=-1)
            carry['passback'] = jnp.stack([jnp.zeros_like(spikes)]*x.shape[-1],axis=-1)
        return carry,spikes
    

class OTTT(nn.Module):
    chain: Any
    leak: float

    @nn.compact
    def __call__(self,carry,x):
        model = self.chain
        class chain(nn.Module):
            @nn.compact
            def __call__(self,carry,x):
                for m in model:
                    x=m()(carry,x)
                return x

        def f(chain,carry,x):
            carry['chain'],x = chain(carry['chain'],x)
            return carry,x

        def summed_output(p,chain,carry,x):
            carry,spikes = chain.apply(p,carry,x)
            return (jnp.sum(spikes),spikes),(carry,spikes)
        
        def fast_update(g,a_hat,params):
            if g.size==params.size:
                return g.reshape(params.shape)
            else:
                return jnp.outer(a_hat.flatten(),g.flatten())

        def f_fwd(chain,carry,x):
            ((ds_du_prev,_),(_,k)),(carry['chain'],spikes) = jax.jacrev(summed_output,has_aux=True,argnums=(2,3))(chain.variables,chain,carry['chain'],x)
            carry['a_hat'] = carry['a_hat']*self.leak + x
            p = jax.lax.stop_gradient({'params':chain.variables['params']})
            ds_du = ravel_pytree(ds_du_prev)[0]/self.leak

            return (carry,spikes),(carry['a_hat'],k,ds_du,p)
        
        def f_bwd(res,g):
            a_hat,k,ds_du,p = res
            p_update = Partial(fast_update,g[1]*ds_du,a_hat)
            grads = tree_map(p_update,p)#jax.tree_map(lambda x: g[1]*x,ds_dtheta)
            return {'params': freeze(grads['params'])},None,g[1].dot(k)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

        m = chain()
        wait=0
        if carry==None:
            wait = 1
            carry = {'chain': None}
        if len(x.shape)>1:
            pf = Partial(f_custom,m)
            carry,spikes = jax.vmap(pf)(carry,x)
        else:
            carry,spikes = f_custom(m,carry,x)
        if wait==1:
            if len(x.shape)>1:
                sz = lambda var: jnp.stack([jnp.zeros_like(var)]*x.shape[0],axis=0)
            else:
                sz = jnp.zeros_like
            carry['a_hat'] = jnp.zeros_like(x)
        return carry,spikes



# class RTRL(nn.Module):
#     model: Callable

#     @nn.compact
#     def __call__(self,x):
#         model = self.model

#         # class chain(nn.Module):
#         #     @nn.compact
#         #     def __call__(self,x):
#         #         return model()(x)

#         def f(chain,x):
#             out = chain(x)
#             return out
        
#         def output(params,state,chain,x,f_p,f_s):
#             params = f_p(params)
#             state = f_s(state)
#             p = {'params':params,'state':state}
#             out = chain.apply(p,x,mutable='state')
#             return (out[0],ravel_pytree(out[1])[0]),(out[0],out[1])

#         def f_fwd(chain,x):
#             #print(chain.variables['state'])
#             #chain.variables['state'].update(stop_grad(chain.variables['state']))
#             #var, rf = ravel_pytree(chain.variables)
#             params, f_p = ravel_pytree(chain.variables['params'])
#             state, f_s = ravel_pytree(chain.variables['state'])
#             ((ds_dparams,ds_du,k),(du_dparams,du_du,_)),(spikes,updates) = jax.jacfwd(output,has_aux=True,argnums=(0,1,3))(params,state,chain,x,f_p,f_s)
#             chain.variables['state'].update(updates['state'])

#             #sz = ravel_pytree(chain.variables['params'])[0].size

#             #ds_du = rf(ds_dp)#[:,sz:]
#             #ds_dparams = ds_dp[:,:sz]

#             #du_du = du_dp[:,sz:]
#             #du_dparams = du_dp[:,:sz]
#             leak = nn.sigmoid(2.)

#             #du_du = spikes*leak
            
#             E = self.variable('state','E')
#             #print(jax.tree_map(lambda x: x.shape,ds_dparams))
#             ds_dtheta = ds_du.dot(E.value) + ds_dparams
#             E.value = du_du.dot(E.value) + du_dparams#/leak
#             #E.value = (du_du*E.value.T).T + du_dparams

#             #ds_dtheta = jax.tree_map(lambda x,y: ds_du*x + y, stop_grad(E.value), ds_dparams)
#             #E.value = jax.tree_map(lambda x,y: du_du*x + y, stop_grad(E.value), du_dparams)

#             return spikes,(ds_dtheta,k)
        
#         def f_bwd(res,g):
#             ds_dtheta,k= res
#             roll = ravel_pytree(m.variables['params'])[1]
#             grads = roll(g.dot(ds_dtheta))
#             #grads = jax.tree_map(lambda x: g*x,ds_dtheta)
#             return {'params': grads},g.dot(k)

#         f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

#         m = model()
#         if len(x.shape) > 1:
#             out = nn.vmap(f_custom)(m,x)
#         else:
#             out = f_custom(m,x)
#         if self.is_initializing():
#             E = self.variable('state','E', jnp.zeros, (ravel_pytree(m.variables['state'])[0].size,ravel_pytree(m.variables['params'])[0].size))
#         return out
    

# class RTRL(nn.Module):
#     chain: Sequence

#     @nn.compact
#     def __call__(self,carry,x):
#         model = self.chain
#         class chain(nn.Module):
#             @nn.compact
#             def __call__(self,carry,x):
#                 for m in model:
#                     x=m()(carry,x)
#                 return x

#         def f(chain,carry,x):
#             carry['chain'],x = chain(carry['chain'],x)
#             return carry,x
        
#         def output(un_p,un_c,p,chain,carry,x):
#             carry,spikes = chain.apply(p,carry,x)
#             return (spikes,carry['Vmem'],spikes),(carry,spikes)

#         def f_fwd(chain,carry,x):
#             p, un_p = ravel_pytree(chain.variables)
#             c, un_c = ravel_pytree(carry['chain'])
#             ((ds_dp,ds_du,_),(du_dp,du_du,_),(_,_,k)),(carry['chain'],spikes) = jax.jacrev(summed_output,has_aux=True,argnums=(0,2,3))(chain.variables,chain,carry['chain'],x)
            
            
#             ds_du = ds_du['Vmem']
#             ds_dparams = ds_dp

#             du_du = du_du['Vmem']
#             du_dparams = du_dp
# #ds_du.dot(E.value) + ds_dparams
#             ds_dtheta = jax.tree_map(lambda x,y: ds_du.dot(x) + y,carry['E'], ds_dparams)
#             carry['E'] = jax.tree_map(lambda x,y: du_du.dot(x) + y, carry['E'], du_dparams)

#             return (carry,spikes),(ds_dtheta,k)
        
#         def f_bwd(res,g):
#             ds_dtheta,k = res
#             grads = jax.tree_map(lambda x: g[1]*x,ds_dtheta)
#             return {'params': freeze(grads['params'])},None,g[1].dot(k)

#         f_custom = nn.custom_vjp(f,f_fwd,f_bwd)

#         m = chain()
#         wait=0
#         if carry==None:
#             wait = 1
#             carry = {'chain': None}
#         if len(x.shape)>1:
#             pf = Partial(f_custom,m)
#             carry,spikes = jax.vmap(pf)(carry,x)
#         else:
#             carry,spikes = f_custom(m,carry,x)
#         if wait==1:
#             if len(x.shape)>1:
#                 sz = lambda var: jnp.stack([jnp.stack([jnp.zeros_like(var)]*x.shape[1],axis=0)]*x.shape[0],axis=0)
#             else:
#                 sz = jnp.zeros_like
#             carry['E'] = jnp.zeros((spikes.size,ravel_pytree(m.variables).size))
#         return carry,spikes