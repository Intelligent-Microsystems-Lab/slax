import jax
import jax.numpy as jnp
from flax import nnx



def output(graph,param,state,x):
    def forward(x):
        model = nnx.merge(graph,param,state)
        out = model(x)
        return out, nnx.split(model,nnx.Param,...)[2]
    out,f_vjp,state = jax.vjp(forward,x,has_aux=True)
    return (jnp.expand_dims(out,0),jnp.stack(jax.tree.leaves(state))),(out,state,f_vjp)

def sum_output(graph,param,state,x):
    def forward(x):
        model = nnx.merge(graph,param,state)
        out = model(x)
        return out, nnx.split(model,nnx.Param,...)[2]
    (out,f_vjp,state) = jax.vjp(forward,x,has_aux=True)
    return (jnp.sum(out),jnp.stack(jax.tree.map(jnp.sum,jax.tree.leaves(state)))),(out,state,f_vjp)
    
def diag_rtrl_update(a,b,c):
    def custom_mul(x,y):
        if len(y.shape) == 2:
            return jnp.einsum('...ij,ij->...j',x,y)
        else:
            return jnp.einsum('...ji,jki->...ki',x,y)
    return jax.tree.map(lambda x,y: custom_mul(a,x)+y,b,c)

def diag(x):
    d = jnp.diagonal(x,axis1=1,axis2=-1)
    d = jnp.squeeze(d)
    return d

def rtrl_update(a,b,c):
    return jax.tree.map(lambda x,y: a.dot(x)+y,b,c)