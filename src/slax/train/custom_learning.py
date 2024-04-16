import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence
from jax.tree_util import Partial, tree_leaves, tree_map
from jax.lax import stop_gradient as stop_grad
from jax.flatten_util import ravel_pytree
from ..models.utils import reinit_model, SNNCell, connect


class DenseOSTL(SNNCell):
    '''
    An implementation of Online Spatio-Temporal Learning (OSTL) from Bohnstingl, T., Woźniak, S., Pantazi, A., & Eleftheriou, E. (2022). Online spatio-temporal learning in deep neural networks. IEEE Transactions on Neural Networks and Learning Systems.

    This implementation only works for dense layers (not convolutional layers) and only one carry state (e.g. will not work as expected with LSTMs).

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
                # for m in mdl[:-1]:
                #     x=reinit_model(m)(x)
                # x = reinit_model(mdl[-1])(x)
                m = reinit_model(connect(mdl))
                x = m(x)
                #print(m)

                if self.is_initializing():
                    init = Partial(jax.tree_map,jnp.zeros_like)
                    E = self.variable('carry','E',init,{'params':self.variables['params']})
                
                return x

        def f(chain,x):
            x = chain(x)
            return x
        
        def summed_output_p(chain,x):
            spikes = chain(x)
            return (spikes,jnp.sum(tree_leaves(chain.variables['carry'])[0]))
        
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
            del chain.variables['carry']['E']

            ((ds_dp,ds_du,_),(du_dp,du_du,_),(_,_,_)),(carry,spikes) = jax.jacrev(summed_output,has_aux=True,argnums=(1,2,3))(chain,p,carry,x)
            (_,_), bwd = nn.vjp(summed_output_p,chain,x,vjp_variables=['params','carry'])

            ds_du = tree_leaves(ds_du)[0]
            ds_dparams = ds_dp

            du_du = tree_leaves(du_du)[0]
            du_dparams = du_dp

            ds_dtheta = jax.tree_map(lambda x,y: ds_du*x + y, E, ds_dparams)
            E = jax.tree_map(lambda x,y: du_du*x + y, E, du_dparams)

            carry['carry']['E'] = E
            chain.variables.update(carry)

            return spikes,(ds_dtheta,stop_grad(bwd))
        
        def f_bwd(res,g):
            ds_dtheta,bwd = res
            grads = jax.tree_map(lambda x: g*x,ds_dtheta)
            return {'params': grads['params']},bwd((g,0.))[1]

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)
        chain = model()

        if len(x.shape)>1:
            spikes = nn.vmap(f_custom,variable_axes={'params':None,'carry':0},split_rngs={'params':False})(chain,x)
        else:
            spikes = f_custom(chain,x)
            
        return spikes
    
class DenseOTPE(SNNCell):
    '''
    An implementation of Online Training with Post-synaptic Estimates (OTPE) from Summe, T., Schaefer, C. J., & Joshi, S. (2023). Estimating Post-Synaptic Effects for Online Training of Feed-Forward SNNs. arXiv preprint arXiv:2311.16151.

    This implementation only works for dense layers (not convolutional layers) and only one carry state.

    Args:
    chain: A sequence of initialized Flax modules. This is somewhat similar to nn.Sequential, but the last module must be the
    module to take a carry variable.

    leak: A value (between 0 and 1) that is the proportion of the state that remains after one time-step
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
                    g_bar = self.variable('carry','g_bar',jnp.zeros_like,x)
                    ratio = self.variable('carry','ratio',jnp.zeros,1)
                
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
            R_hat = jax.tree_map(lambda x,y: self.leak*x + y, R_hat, ds_dtheta)

            ratio = self.leak*r
            r = ratio + 1
            ratio = (ratio/r)

            g_bar = ratio*g_bar + (1-ratio)*ds_du/self.leak

            carry['carry']['E'] = E
            carry['carry']['R_hat'] = R_hat
            carry['carry']['g_bar'] = g_bar
            carry['carry']['ratio'] = r

            chain.variables.update(carry)

            return spikes,(R_hat,k,g_bar)
        
        def f_bwd(res,g):
            R_hat,k,g_bar = res
            grads = jax.tree_map(lambda x: g*x,R_hat)
            return {'params': grads['params']},(g*g_bar).dot(k)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)
        chain = model()

        if len(x.shape)>1:
            spikes = nn.vmap(f_custom,variable_axes={'params':None,'carry':0},split_rngs={'params':False})(chain,x)
        else:
            spikes = f_custom(chain,x)
            
        return spikes

class OTTT(nn.Module):
    '''
    An implementation of Online Training Through Time (OTTT) from Xiao, M., Meng, Q., Zhang, Z., He, D., & Lin, Z. (2022). Online training through time for spiking neural networks. Advances in neural information processing systems, 35, 20717-20730.

    Args:
    chain: A sequence of initialized Flax modules. This is somewhat similar to nn.Sequential, but the last module must be the
    module to take a carry variable.

    leak: A value (between 0 and 1) that is the proportion of the state that remains after one time-step
    '''
    chain: Any
    leak: float

    @nn.compact
    def __call__(self,x):
        mdl = self.chain
        class model(nn.Module):
            @nn.compact
            def __call__(self,x):
                in_size = x.shape
                for m in mdl[:-1]:
                    x=reinit_model(m)(x)
                x = reinit_model(mdl[-1])(x)

                if self.is_initializing():
                    a_hat = self.variable('carry','a_hat',jnp.zeros,in_size)
                
                return x

        def f(chain,x):
            x = chain(x)
            return x
        
        def summed_output(chain,p,carry,x):
            p.update(carry)
            spikes,carry = chain.apply(p,x,mutable='carry')
            return (jnp.sum(spikes),spikes),(carry,spikes)
        
        def summed_output_p(chain,x):
            spikes = chain(x)
            return (spikes,jnp.sum(tree_leaves(chain.variables['carry'])[0]))
        
        def fast_update(g,a_hat,params):
            if g.size==params.size:
                return g.reshape(params.shape)
            else:
                return jnp.outer(a_hat.flatten(),g.flatten())

        def f_fwd(chain,x):
            variables = stop_grad(chain.variables)
            p = {'params': variables['params']}
            carry = {'carry': variables['carry']}
            a_hat = variables['carry']['a_hat']
            del carry['carry']['a_hat']

            ((ds_du_prev,_),(_,_)),(carry,spikes) = jax.jacrev(summed_output,has_aux=True,argnums=(2,3))(chain,p,carry,x)
            (_,_), bwd = nn.vjp(summed_output_p,chain,x,vjp_variables=['params','carry'])

            a_hat = a_hat*self.leak + x
            p = jax.lax.stop_gradient({'params':chain.variables['params']})
            ds_du = ravel_pytree(ds_du_prev)[0]/self.leak
            carry['carry']['a_hat'] = a_hat
            chain.variables.update(carry)

            return spikes,(a_hat,stop_grad(bwd),ds_du)
        
        def f_bwd(res,g):
            a_hat,bwd,ds_du = res
            p_update = Partial(fast_update,g*ds_du,a_hat)
            grads = tree_map(p_update,chain.variables['params'])
            return {'params': grads},bwd((g,0.))[1]


        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)
        chain = model()

        if len(x.shape)>1:
            spikes = nn.vmap(f_custom,variable_axes={'params':None,'carry':0},split_rngs={'params':False})(chain,x)
        else:
            spikes = f_custom(chain,x)
        return spikes

class RTRL(nn.Module):
    '''
    An implementation of Real Time Recurrent Learing (RTRL) from Williams, R. J., & Zipser, D. (1989). A learning algorithm for continually running fully recurrent neural networks. Neural computation, 1(2), 270-280.

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
                    #init = Partial(jax.tree_map,lambda vars: jnp.stack([jnp.zeros_like(vars)]*x.size,axis=0))
                    E = self.variable('carry','E',jnp.zeros,(ravel_pytree(self.variables['carry'])[0].size,ravel_pytree(self.variables['params'])[0].size))
                
                return x

        def f(chain,x):
            x = chain(x)
            return x
        
        def output(chain,p,carry,x,unravel_p,unravel_carry):
            p = unravel_p(p)
            carry = unravel_carry(carry)
            p.update(carry)
            spikes,carry = chain.apply(p,x,mutable='carry')
            return (spikes,ravel_pytree(carry)[0]),(carry,spikes)

        def f_fwd(chain,x):
            variables = stop_grad(chain.variables)
            p,unravel_p = ravel_pytree({'params': variables['params']})
            
            E = variables['carry']['E']
            carry = {'carry': variables['carry']}
            del carry['carry']['E']
            carry,unravel_carry = ravel_pytree(carry)

            ((ds_dp,ds_du,k),(du_dp,du_du,_)),(carry,spikes) = jax.jacrev(output,has_aux=True,argnums=(1,2,3))(chain,p,carry,x,unravel_p,unravel_carry)

            ds_dparams = ds_dp

            du_dparams = du_dp
            ds_dtheta = jax.tree_map(lambda x,y: ds_du.dot(x) + y, E, ds_dparams)
            E = jax.tree_map(lambda x,y: du_du.dot(x) + y, E, du_dparams)

            carry['carry']['E'] = E
            chain.variables.update(carry)

            return spikes,(ds_dtheta,k)
        
        def f_bwd(res,g):
            ds_dtheta,k = res
            grads = unravel_params(g.dot(ds_dtheta))
            return {'params': grads},g.dot(k)

        f_custom = nn.custom_vjp(f,f_fwd,f_bwd)
        chain = model()

        if not self.is_initializing():
            unravel_params = ravel_pytree(chain.variables['params'])[1]


        if len(x.shape)>1:
            spikes = nn.vmap(f_custom,variable_axes={'params':None,'carry':0},split_rngs={'params':False})(chain,x)
        else:
            spikes = f_custom(chain,x)
            
        return spikes