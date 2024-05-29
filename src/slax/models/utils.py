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

def connect(chains,cat=None,return_carry=True,return_initialized=True):
    """Connects modules together while optionally specifying skip and recurrent connections.

    Args:
    chains: A list of modules
    cat: A dictionary for skip/recurrent connections where each key is a number corresponding to the list index and the values are what it additionally feed to.
    return_carry: Whether or not to return a carry state. When processing the time dimension inside the network, like with sl.RNN, set this to False.
    return_initialized: Whether or not the returned model is initialized.

    Returns:
    A module that sequentially connects the provides modules and adds any additionally specified connections.
    """

    if cat == None:
        cat = flax.core.frozen_dict.FrozenDict({})
    else:
        cat = flax.core.frozen_dict.FrozenDict({str(k):cat[k] for k in cat.keys()})

    class _connect(SNNCell):
        chain: Sequence = tuple(chains)
        pair: Dict = cat
        carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

        @nn.compact
        def __call__(self,x):
            
            if self.pair == None:
                pair = {}
            else:
                pair = self.pair
            u = set(chain.from_iterable(pair.values()))
            counter = 0
            inp = x

            if self.is_initializing():
                for mdl in self.chain:
                    x = mdl(x)
                    if counter in u:
                        v = self.variable('carry','rec_{}'.format(self.chain[counter].name),jnp.zeros,x.shape)
                    counter += 1
            x = inp
            counter = 0
            #print(4 in list(pair.keys()))
            for mdl in self.chain:
                x = mdl(x)
                if str(counter) in list(pair.keys()):
                    for i in self.pair[str(counter)]:
                        #v = self.variable('carry','rec_{}'.format(self.chain[i].name))
                        v = self.get_variable('carry','rec_{}'.format(self.chain[i].name))
                        x += reinit_model(mdl,name='rec_{}_{}'.format(i,counter))(v)
                if counter in u:
                    #v = self.variable('carry','rec_{}'.format(self.chain[counter].name))
                    self.put_variable('carry','rec_{}'.format(self.chain[counter].name),x)
                    #v.value = x
                counter += 1
                
            return x
    if return_initialized:
        return _connect()
    else:
        return _connect

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