from flax.linen.recurrent import Carry, PRNGKey
import jax.numpy as jnp
import jax
from jax.experimental import sparse as sp
import flax.linen as nn
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
from flax.linen.module import (
  RNGSequences,
  FrozenVariableDict,
  VariableDict,
)
from flax.core.scope import (
  CollectionFilter,
  DenyList,
  Variable,
  union_filters,
)

Dict = Any

class SNNCellBase(nn.RNNCellBase):
    """RNN cell base class."""
    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Carry:
        """Initialize the RNN cell carry.

        Args:
        rng: random number generator passed to the init_fn.
        input_shape: a tuple providing the shape of the input to the cell.

        Returns:
        An initialized carry for the given RNN cell.
        """
        @jax.jit
        def get_carry():
            p_init = jax.tree_util.Partial(self.lazy_init,rng)
            #vars = flax.jax_utils.partial_eval_by_shape(p_init,[((input_shape),input_dtype),])
            vars = flax.jax_utils.partial_eval_by_shape(p_init,[(input_shape),])
            p_carry = lambda x: self.carry_init(rng,x.shape,x.dtype)
            return jax.tree_util.tree_map(p_carry,vars['carry'])
        return get_carry()
    
    @property
    def carry_init(self) -> Callable:
        """Initializer for the carry state"""
        return nn.initializers.zeros_init()
    
    @property
    def num_feature_axes(self) -> int:
        return 1
    
    @property
    def has_time(self) -> int: #could also call it feed-forward
        return False
    
    def apply(
        self,
        variables: VariableDict,
        *args,
        rngs: Optional[RNGSequences] = None,
        method: Union[Callable[..., Any], str, None] = None,
        mutable: CollectionFilter = DenyList('params'),
        capture_intermediates: Union[bool, Callable[['Module', str], bool]] = False,
        **kwargs,
        ):
        
        nn.Module._module_checks(self)

        if isinstance(method, str):
            attribute_name = method
            method = getattr(self, attribute_name)
            if not callable(method):
                class_name = type(self).__name__
                raise TypeError(
                    f"'{class_name}.{attribute_name}' must be a callable, got"
                    f' {type(method)}.'
                )
        # if the `method` string is a submodule, we create a lambda function
        # that calls the submodule, forwarding all arguments.
            if isinstance(method, nn.Module):
                method = lambda self, *args, **kwargs: getattr(self, attribute_name)(
                *args, **kwargs
                )
        elif method is None:
            method = self.__call__
        method = nn.module._get_unbound_fn(method)
        out = nn.apply(
            method,
            self,
            mutable=mutable,
            capture_intermediates=capture_intermediates,
            )(variables, *args, **kwargs, rngs=rngs)
        if bool(out[1]):
            if self.has_time:
                return out[0]
            else:
                return out
        else: 
            return out[0]


class Neuron(SNNCellBase):
    None

class SNNCell(SNNCellBase):
    None


def reinit_model(mdl,name=None):
    d = {}
    for kw in mdl.__dataclass_fields__.keys():
        if kw != 'parent' and kw != 'name':
            d[kw] = mdl.__getattr__(kw)
    if name != None:
        d['name'] = name
    return mdl.__class__(**d)
    
def connect(chains,p=None,return_carry=True,return_initialized=True):
    """Connects modules together while optionally specifying skip and recurrent connections.

    Args:
    chains: A list of modules
    p: A dictionary for skip/recurrent connections where each key is a number corresponding to the list index and the values are what it additionally feed to.
    return_carry: Whether or not to return a carry state. When processing the time dimension inside the network, like with sl.RNN, set this to False.
    return_initialized: Whether or not the returned model is initialized.

    Returns:
    A module that sequentially connects the provides modules and adds any additionally specified connections.
    """
    
    #loop = has_time
    class connect(SNNCell):
        merge: str = 'cat'
        carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()
        has_time: bool = not return_carry # could also call it feed-forward

        #def setup(self):
            #self.has_time=has_time

        @nn.compact
        def __call__(self,carry,x=None):
            if x==None:
                x = carry
                ex_carry = False
            else:
                self.variables['carry'] = carry
                ex_carry = True
            
            if p == None:
                pair = {}
            else:
                pair = p
            u = set(chain.from_iterable(pair.values()))
            counter = 0
            inp = x
            ms = []
            vs = []

            if self.is_initializing():
                for i in range(len(chains)):
                    mdl = chains[i]
                    ms.append(reinit_model(mdl,name='chain_{}'.format(i)))
                    x = ms[-1](x)
                    if counter in u:
                        v = self.variable('carry','rec_{}'.format(counter),jnp.zeros,x.shape)
                        vs.append(v)
                    counter += 1
            x = inp
            counter = 0
            c2 = 0
            if self.merge == 'cat':
                if self.is_initializing():
                    for mdl in ms:
                        x = mdl(x)
                        if counter in pair.keys():
                            for i in pair[counter]:
                                #v = self.variable('carry','rec_{}'.format(i))
                                v = vs[c2]
                                x += reinit_model(mdl,name='chain_{}_rec_{}'.format(counter,i))(v.value)
                                c2 += 1
                        # if counter in u:
                        #     v = self.variable('carry','rec_{}'.format(counter))
                        #     v.value = x
                        counter += 1
                else:
                    for i in range(len(chains)):
                        mdl = chains[i]
                        x = reinit_model(mdl,name='chain_{}'.format(i))(x)
                        if counter in pair.keys():
                            for j in pair[counter]:
                                v = self.get_variable('carry','rec_{}'.format(j))
                                x += reinit_model(mdl,name='chain_{}_rec_{}'.format(counter,j))(v)
                        if counter in u:
                            self.put_variable('carry','rec_{}'.format(counter),x)
                            #v.value = x
                        counter += 1

            elif self.merge == 'add':
                for mdl in chains:
                    if counter in pair.keys():
                        for i in pair[counter]:
                            v = self.variable('carry','rec_{}'.format(i))
                            x += v.value
                        x = mdl(x)
                    if counter in u:
                        v = self.variable('carry','rec_{}'.format(counter))
                        v.value = x
                    counter += 1

            if ex_carry:
                carry = self.variables['carry']
                del self.variables['carry']
                return carry, x
            else:
                return x

    if return_initialized:
        return connect()
    else:
        return connect

###### Previously functional code below ##########

# class connect(SNNCell):
#     chain: Sequence
#     pair: Dict = None
#     merge: str = 'cat'
#     carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

#     @nn.compact
#     def __call__(self,x):
        
#         if self.pair == None:
#             pair = {}
#         else:
#             pair = self.pair
#         u = set(chain.from_iterable(pair.values()))
#         counter = 0
#         inp = x

#         if self.is_initializing():
#             for mdl in self.chain:
#                 x = mdl(x)
#                 if counter in u:
#                     v = self.variable('carry','rec_{}'.format(self.chain[counter].name),jnp.zeros,x.shape)
#                 counter += 1
#         x = inp
#         #counter = 0
#         if self.merge == 'cat':
#             for mdl in self.chain:
#                 x = mdl(x)
#                 if counter in pair.keys():
#                     for i in self.pair[counter]:
#                         v = self.variable('carry','rec_{}'.format(self.chain[i].name))
#                         x += reinit_model(mdl)(v.value)
#                 if counter in u:
#                     v = self.variable('carry','rec_{}'.format(self.chain[counter].name))
#                     v.value = x
#                 counter += 1

#         elif self.merge == 'add':
#             for mdl in self.chain:
#                 if counter in pair.keys():
#                     for i in self.pair[counter]:
#                         v = self.variable('carry','rec_{}'.format(self.chain[i].name))
#                         x += v.value
#                     x = mdl(x)
#                 if counter in u:
#                     v = self.variable('carry','rec_{}'.format(self.chain[counter].name))
#                     v.value = x
#                 counter += 1
            
#         return x

def RNN(model,unroll=jnp.iinfo(jnp.uint32).max):
    """Applies a provided model or module over a sequence.

    Args:
    model: The model to apply of the sequence
    unroll: The number of loop iterations to unroll. In general, a higher number reduces execution time at teh cost of compilation time.

    Returns:
    A model/module that takes in data with the time dimension.
    """
    class rnn(SNNCell):
        mdl: Callable = model
        @nn.compact
        def __call__(self, carry, x):
            return carry,self.mdl(x)
        @nn.nowrap
        def initialize_carry(self,rng, shape):
            return ()
        
    class rec(nn.Module):
        @nn.compact
        def __call__(self, xs):
            scan = nn.scan(
                rnn,
                variable_carry=["carry","batch_stats"],
                variable_broadcast="params",
                split_rngs={"params": False},
                unroll=unroll
            )

            if self.is_initializing():
                _,x = rnn(name='RNN_')((), xs[0])
                return jnp.stack([x]*xs.shape[0],0)
            else:
                _,x = scan(name='RNN_')((), xs)
                # del self.variables['carry']
                return x
            
    class output_model(nn.Module):
        @nn.compact
        def __call__(self,x):
            x = rec()(x)
            return x
    return output_model()



def pack_policy(prim,*_,**__):
    v = _
    print(prim)
    #print(v[0].dtype)
    #print(*_)
    if v[0].dtype == jnp.uint8:#should this be jnp.int32 or uint8?:
        #print('yay')
        return True
    else:
        return False
    
@jax.custom_vjp
def pack(x,axis=None):
    a = sp.empty(shape=x.shape,dtype=x.dtype)
    # if x.size > jnp.iinfo(jnp.uint16).max:
    #    a.data = jax.vmap(lambda x_in: jnp.packbits(jnp.bool_(x_in),axis=axis).view(x.dtype))(x)
    # else:
    a.data = jnp.packbits(jnp.bool_(x),axis=axis).view(x.dtype)
    return a
def pack_fwd(x,axis=None):
  return pack(x,axis=axis), ()

def pack_bwd(res, g):
  return (g.data.reshape(g.shape),None)

pack.defvjp(pack_fwd, pack_bwd)

@jax.custom_vjp
def unpack(arr,axis=None):
    # if arr.data.size > jnp.iinfo(jnp.uint16).max:
    #     x = jax.vmap(lambda x: jnp.array(jnp.unpackbits(x.view(jnp.uint8),axis=axis),arr.dtype).reshape(arr.shape[1:]))(arr.data.reshape(arr.shape[0],-1))
    # else:
    x = jnp.array(jnp.unpackbits(arr.data.view(jnp.uint8),axis=axis),arr.dtype).reshape(arr.shape)
    return x
def unpack_fwd(arr,axis=None):
    return unpack(arr,axis=axis),()

def unpack_bwd(res, g):
    return (g,None)

unpack.defvjp(unpack_fwd, unpack_bwd)


class LeakyIntegrator(nn.Module):
    leak: float

    @nn.compact
    def __call__(self,x):
        value = self.variable('carry','LeakyIntegrator',jnp.zeros,1)
        v = value.value

        v = v*self.leak + x

        return v

class OnlineLeakyIntegrator(nn.Module):
    leak: float
    n_steps: int

    @nn.compact
    def __call__(self,x):
        value = self.variable('carry','LeakyIntegrator',jnp.zeros,1)
        counter = self.variable('carry','counter',jnp.zeros,1)
        v = value.value
        c = counter.value
        c += 1
        #v += x * self.leak**(self.n_steps-c)

        return x * self.leak**(self.n_steps-c)#v