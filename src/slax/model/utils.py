from functools import partial
import jax.numpy as jnp
import jax
from flax import nnx
from typing import Any, Callable, Sequence, DefaultDict, Tuple
import flax
from itertools import chain
from typing import (
  Any,
  Callable,
  Dict,
  Iterable,
  Iterator,
  List,
  Literal,
  Mapping,
  Optional,
  Tuple,
  Type,
  TypeVar,
  Union,
  overload,
)


Dict = Any

class Neuron(nnx.Module):
    """SNN base class."""




def reinit_model(mdl,in_features):
    #fields = nnx.split(mdl)[0].nodedef.static_fields
    fields = nnx.split(mdl)[0].static_fields
    f = dict(fields)
    f['in_features'] = in_features
    return mdl.__class__(**f,rngs=nnx.Rngs(0))

class connect(nnx.Module):
    """Connects modules together while optionally specifying skip and recurrent connections.

    Args:
    chains: A list of modules
    cat: A dictionary for skip/recurrent connections where each key is a number corresponding to the list index and the values are what it additionally feed to.

    Returns:
    A module that sequentially connects the provides modules and adds any additionally specified connections.
    """

    def __init__(self,chains,cat=None):
        if cat == None:
            cat = flax.core.frozen_dict.FrozenDict({})
        else:
            cat = flax.core.frozen_dict.FrozenDict({str(k):cat[k] for k in cat.keys()})
        self.chain = tuple(chains)
        self.pair = cat
        #self.carry_init = nn.initializers.zeros_init()
        pair = cat
        counter = 0
        d = []
        u = set(chain.from_iterable(pair.values()))
        self.rec = {}
        self.u = u

        #flax.jax_utils.partial_eval_by_shape(d1,[(2,),])


        for mdl in self.chain:
            d.append({})
            if str(counter) in list(pair.keys()):
                for i in self.pair[str(counter)]:
                    d[counter][i] = reinit_model(mdl,self.chain[i].out_features)
                    
            if counter in u:
                self.rec[counter] = jnp.zeros(self.chain[counter].out_features)
            counter += 1

        self.d = d


    def __call__(self,x):
        
        u = self.u
        pair = self.pair
        counter = 0
        inp = x

        x = inp
        counter = 0
        for mdl in self.chain:
            x = mdl(x)
            if str(counter) in list(pair.keys()):
                for i in self.pair[str(counter)]:
                    x += self.d[counter][i](self.rec[i])
            if counter in u:
                self.rec[counter] = x
            counter += 1
            
        return x
    
def compat_scan(f,carry,xs,unroll=False,length=None):
    ind = jnp.zeros(1,jnp.uint32)
    def exec(c,inp):
        state,k = c
        if isinstance(xs,Iterable):
            vals = jax.tree.map(lambda x: x[k][0],xs)
            state,out = f(state,vals)
        else:
            state,out = f(state,xs[k][0])
        k += jnp.uint32(1)
        return (state,k),out
    (carry,ind), ys = jax.lax.scan(exec,(carry,ind),xs,unroll=unroll,length=length)
    return carry, ys

# class RNN(nnx.Module):
#     def __init__(self,mdl,unroll=False,length=None):
#         """Applies a provided model or module over a sequence.

#         Args:
#         model: The model to apply of the sequence
#         xs: Input data
#         unroll: The number of loop iterations to unroll. In general, a higher number reduces execution time at the cost of compilation time.
#         Length: The number of iterations if it cannot be inferred

#         Returns:
#         A model/module that takes in data with the time dimension, if xs is not given. If xs is given, returns the stacked output.
#         """
#         graph,param,var = nnx.split(mdl,nnx.Param,...)
#         self.mdl = mdl
#         self.var = var
#         self.unroll = unroll
#         self.length = length
#         def forward(state,inp):
#             model = nnx.merge(graph,param,state)
#             out = model(inp)
#             state = nnx.split(model,nnx.Param,...)[2]
#             return state, out
#         self.forward = forward
        
#     def __call__(self,x):
#         return jax.lax.scan(self.forward,self.var,x,unroll=self.unroll,length=self.length)[1]

def forward(graph,param,state,inp):
    model = nnx.merge(graph,param,state)
    out = model(inp)
    state = nnx.split(model,nnx.Param,...)[2]
    return state, out

class RNN(nnx.Module):
    def __init__(self,mdl,unroll=jnp.iinfo(jnp.uint32).max,broadcast_state=True):
        """Applies a provided model or module over a sequence.

        Args:
        model: The model to apply of the sequence
        xs: Input data
        unroll: The number of loop iterations to unroll. In general, a higher number reduces execution time at the cost of compilation time.
        broadcast_state: If true, executes the first time step outside of the loop

        Returns:
        A model/module that loops over the first axis of its input
        """
        #graph,param,var = nnx.split(mdl,nnx.Param,...)
        self.mdl = mdl
        #self.var = var
        self.unroll = unroll
        self.broadcast_state = broadcast_state

        self.first = lambda x: x[0]
        self.rest = lambda x: x[1:]
    
    @partial(jax.jit,static_argnums=0)
    def __call__(self,x):
        graph,param,var = nnx.split(self.mdl,nnx.Param,...)
        p_forward = partial(forward,graph,param)
        if self.broadcast_state:
            var,y_1 = p_forward(var,jax.tree.map(self.first,(x,))[0])
            ys = jax.lax.scan(p_forward,var,jax.tree.map(self.rest,(x,))[0],unroll=self.unroll)[1]
            return jnp.concat([jnp.expand_dims(y_1,axis=0),ys])
        else:
            return jax.lax.scan(p_forward,var,x,unroll=self.unroll)[1]

# class RNN(nnx.Module):
#     def __init__(self,mdl,unroll=jnp.iinfo(jnp.uint32).max,broadcast_state=True):
#         """Applies a provided model or module over a sequence.

#         Args:
#         model: The model to apply of the sequence
#         xs: Input data
#         unroll: The number of loop iterations to unroll. In general, a higher number reduces execution time at the cost of compilation time.
#         broadcast_state: If true, executes the first time step outside of the loop

#         Returns:
#         A model/module that loops over the first axis of its input
#         """
#         graph,param,var = nnx.split(mdl,nnx.Param,...)
#         self.mdl = mdl
#         self.var = var
#         self.unroll = unroll
#         self.broadcast_state = broadcast_state

#         self.first = lambda x: x[0]
#         self.rest = lambda x: x[1:]

#         def forward(state,inp):
#             model = nnx.merge(graph,param,state)
#             out = model(inp)
#             state = nnx.split(model,nnx.Param,...)[2]
#             return state, out
#         self.forward = forward
    
#     @partial(jax.jit,static_argnums=0)
#     def __call__(self,x):
#         if self.broadcast_state:
#             var,y_1 = self.forward(self.var,jax.tree.map(self.first,(x,))[0])
#             ys = jax.lax.scan(self.forward,var,jax.tree.map(self.rest,(x,))[0],unroll=self.unroll)[1]
#             return jnp.concat([jnp.expand_dims(y_1,axis=0),ys])
#         else:
#             return jax.lax.scan(self.forward,self.var,x,unroll=self.unroll)[1]
        
@partial(jax.jit,static_argnums=(0,3))
def scan(f, init, xs, unroll):
    first = lambda x: x[0]
    rest = lambda x: x[1:]
    var, y_1 = f(init,jax.tree.map(first,(xs,))[0])
    var, ys = jax.lax.scan(f,var,jax.tree.map(rest,(xs,))[0],unroll=unroll)
    return var, jax.tree.map(lambda x,y: jnp.concat([jnp.expand_dims(x,axis=0),y]),(y_1,),(ys,))[0]