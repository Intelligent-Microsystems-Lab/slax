import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence
from jax.tree_util import Partial, tree_leaves, tree_map
from jax.lax import stop_gradient as stop_grad
from jax.flatten_util import ravel_pytree
from ..models.utils import reinit_model

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
    def __call__(self,x):
        mdl = self.chain
        class model(nn.Module):
            @nn.compact
            def __call__(self,x):
                for m in mdl[:-1]:
                    x=reinit_model(m)(x)
                x = reinit_model(mdl[-1])(x)

                if self.is_initializing():
                    init = Partial(jax.tree_map,jnp.zeros_like)
                    E = self.variable('carry','E',init,{'params':self.variables['params']})
                
                return x

        def f(chain,x):
            x = chain(x)
            return x
        
        def summed_output(chain,p,carry,x):
            p.update(carry)
            spikes,carry = chain.apply(p,x,mutable='carry')
            return (jnp.sum(spikes),jnp.sum(tree_leaves(carry)[0]),spikes),(carry,spikes)

        def f_fwd(chain,x):
            variables = stop_grad(chain.variables)
            p = {'params': variables['params']}
            carry = {'carry': variables['carry']}
            E = variables['carry']['E']
            del carry['carry']['E']

            ((ds_dp,ds_du,_),(du_dp,du_du,_),(_,_,k)),(carry,spikes) = jax.jacrev(summed_output,has_aux=True,argnums=(1,2,3))(chain,p,carry,x)

            ds_du = tree_leaves(ds_du)[0]
            ds_dparams = ds_dp

            du_du = tree_leaves(du_du)[0]
            du_dparams = du_dp

            ds_dtheta = jax.tree_map(lambda x,y: ds_du*x + y, E, ds_dparams)
            E = jax.tree_map(lambda x,y: du_du*x + y, E, du_dparams)

            carry['carry']['E'] = E
            chain.variables.update(carry)

            return spikes,(ds_dtheta,k)
        
        def f_bwd(res,g):
            ds_dtheta,k = res
            grads = jax.tree_map(lambda x: g*x,ds_dtheta)
            return {'params': grads['params']},g.dot(k)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)
        chain = model()

        if len(x.shape)>1:
            spikes = nn.vmap(f_custom,variable_axes={'params':None,'carry':0},split_rngs={'params':False})(chain,x)
        else:
            spikes = f_custom(chain,x)
            
        return spikes
    

class DenseOTPE(nn.Module):
    '''
    An efficient implementation of Online Spatio-Temporal Learning (OSTL) from Bohnstingl, T., Woźniak, S., Pantazi, A., & Eleftheriou, E. (2022). Online spatio-temporal learning in deep neural networks. IEEE Transactions on Neural Networks and Learning Systems.

    This implementation only works for dense layers (not convolutional layers).

    Args:
    chain: A sequence of initialized Flax modules. This is somewhat similar to nn.Sequential, but the last module must be the
    module to take a carry variable.
    '''
    chain: Sequence
    leak: float

    @nn.compact
    def __call__(self,x):
        mdl = self.chain
        class model(nn.Module):
            @nn.compact
            def __call__(self,x):
                for m in mdl[:-1]:
                    x=reinit_model(m)(x)
                x = reinit_model(mdl[-1])(x)

                if self.is_initializing():
                    init = Partial(jax.tree_map,jnp.zeros_like)
                    E = self.variable('carry','E',init,{'params':self.variables['params']})
                    R_hat = self.variable('carry','R_hat',init,{'params':self.variables['params']})
                    g_bar = self.variable('carry','E',init,{'params':self.variables['params']})
                    ratio = self.variable('carry','E',jnp.zeros,1)
                
                return x

        def f(chain,x):
            x = chain(x)
            return x
        
        def summed_output(chain,p,carry,x):
            p.update(carry)
            spikes,carry = chain.apply(p,x,mutable='carry')
            return (jnp.sum(spikes),jnp.sum(tree_leaves(carry)[0]),spikes),(carry,spikes)

        def f_fwd(chain,x):
            variables = stop_grad(chain.variables)
            p = {'params': variables['params']}
            carry = {'carry': variables['carry']}
            E = variables['carry']['E']
            R_hat = variables['carry']['R_hat']
            g_bar = variables['carry']['g_bar']
            r = variables['carry']['ratio']
            del carry['carry']['E']
            del carry['carry']['R_hat']
            del carry['carry']['g_bar']
            del carry['carry']['ratio']

            ((ds_dp,ds_du,_),(du_dp,du_du,_),(_,_,k)),(carry,spikes) = jax.jacrev(summed_output,has_aux=True,argnums=(1,2,3))(chain,p,carry,x)

            ds_du = tree_leaves(ds_du)[0]
            ds_dparams = ds_dp

            du_du = tree_leaves(du_du)[0]
            du_dparams = du_dp

            ds_dtheta = jax.tree_map(lambda x,y: ds_du*x + y, E, ds_dparams)
            E = jax.tree_map(lambda x,y: du_du*x + y, E, du_dparams)

            ds_dtheta = jax.tree_map(lambda x,y: ds_du*x + y, E, ds_dparams)
            E = jax.tree_map(lambda x,y: du_du*x + y, E, du_dparams)
            R_hat = jax.tree_map(lambda x,y: ds_du*x + y, R_hat, ds_dtheta)

            ratio = self.leak*r
            r = ratio + 1
            ratio = (ratio/r)

            g_bar = ratio*g_bar + (1-ratio)*ds_du

            carry['carry']['E'] = E
            carry['carry']['R_hat'] = R_hat
            carry['carry']['g_bar'] = g_bar
            carry['carry']['ratio'] = r

            chain.variables.update(carry)

            return spikes,(R_hat,k)
        
        def f_bwd(res,g):
            R_hat,k = res
            grads = jax.tree_map(lambda x: g*x,R_hat)
            return {'params': grads['params']},g.dot(k)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)
        chain = model()

        if len(x.shape)>1:
            spikes = nn.vmap(f_custom,variable_axes={'params':None,'carry':0},split_rngs={'params':False})(chain,x)
        else:
            spikes = f_custom(chain,x)
            
        return spikes

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
            carry['a_hat'] = jnp.zeros_like(x)
        return carry,spikes

class RTRL(nn.Module):
    '''
    An efficient implementation of Online Spatio-Temporal Learning (OSTL) from Bohnstingl, T., Woźniak, S., Pantazi, A., & Eleftheriou, E. (2022). Online spatio-temporal learning in deep neural networks. IEEE Transactions on Neural Networks and Learning Systems.

    This implementation only works for dense layers (not convolutional layers).

    Args:
    chain: A sequence of initialized Flax modules. This is somewhat similar to nn.Sequential, but the last module must be the
    module to take a carry variable.
    '''
    chain: Sequence

    @nn.compact
    def __call__(self,x):
        mdl = self.chain
        class model(nn.Module):
            @nn.compact
            def __call__(self,x):
                for m in mdl[:-1]:
                    x=reinit_model(m)(x)
                x = reinit_model(mdl[-1])(x)

                if self.is_initializing():
                    init = Partial(jax.tree_map,jnp.zeros_like)
                    E = self.variable('carry','E',init,{'params':self.variables['params']})
                
                return x

        def f(chain,x):
            x = chain(x)
            return x
        
        def output(chain,p,carry,x):
            p.update(carry)
            spikes,carry = chain.apply(p,x,mutable='carry')
            return (spikes,tree_leaves(carry)[0]),(carry,spikes)

        def f_fwd(chain,x):
            variables = stop_grad(chain.variables)
            p = {'params': variables['params']}
            carry = {'carry': variables['carry']}
            E = variables['carry']['E']
            del carry['carry']['E']

            ((ds_dp,ds_du,k),(du_dp,du_du,_)),(carry,spikes) = jax.jacrev(output,has_aux=True,argnums=(1,2,3))(chain,p,carry,x)

            ds_du = tree_leaves(ds_du)[0]
            ds_dparams = ds_dp

            du_du = tree_leaves(du_du)[0]
            du_dparams = du_dp

            ds_dtheta = jax.tree_map(lambda x,y: ds_du.dot(x) + y, E, ds_dparams)
            E = jax.tree_map(lambda x,y: du_du.dot(x) + y, E, du_dparams)

            carry['carry']['E'] = E
            chain.variables.update(carry)

            return spikes,(ds_dtheta,k)
        
        def f_bwd(res,g):
            ds_dtheta,k = res
            grads = jax.tree_map(lambda x: g*x,ds_dtheta)
            return {'params': grads['params']},g.dot(k)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)
        chain = model()

        if len(x.shape)>1:
            spikes = nn.vmap(f_custom,variable_axes={'params':None,'carry':0},split_rngs={'params':False})(chain,x)
        else:
            spikes = f_custom(chain,x)
            
        return spikes