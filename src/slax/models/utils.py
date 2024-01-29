from flax.linen.recurrent import Carry, PRNGKey
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Sequence, Dict, Tuple
import flax
from itertools import chain

def reinit_model(mdl):
    d = {}
    for kw in mdl.__dataclass_fields__.keys():
        if kw != 'parent':
            d[kw] = mdl.__getattr__(kw)
    #d['name'] += add_name
    return mdl.__class__(**d)

class recurrent(nn.RNNCellBase):
    chain: Sequence
    num_feature_axes: int = 1
    @nn.compact
    def __call__(self,carry,x):
        c2 = 0
        counter = 0
        for mdl in self.chain:
            if 'num_feature_axes' in mdl.__annotations__.keys():
                carry[counter],x = mdl(carry[counter],x)
                counter += 1
            else:
                x = mdl(x)
            if c2==0:
                x += reinit_model(mdl)(carry[-1]['rec'])
                c2 += 1
        carry[-1]['rec'] = x
        return carry,x
    
    @nn.nowrap
    def initialize_carry(self,key,shape):
        carry = []
        for mdl in self.chain:
            if 'v_threshold' in mdl.__annotations__.keys():
                carry.append(mdl.initialize_carry(key,shape))
            else:
                shape = flax.jax_utils.partial_eval_by_shape(mdl,[(shape),]).shape
                # signal_dims = shape[-self.num_feature_axes : -1]
                # batch_dims = shape[: -self.num_feature_axes]
                # shape = batch_dims + signal_dims + (mdl.features,)
        carry[-1]['rec'] = jnp.zeros(shape)
        return carry

class connect(nn.RNNCellBase):
    chain: Sequence
    num_feature_axes: int = 1
    pair: Dict = None

    @nn.compact
    def __call__(self,x):
        if self.pair == None:
            pair = {}
        else:
            pair = self.pair
        u = set(chain.from_iterable(pair.values()))
        c2 = 0
        counter = 0
        inp = x

        if self.is_initializing():
            for mdl in self.chain:
                x = mdl(x)
                if c2 in u:
                    v = self.variable('carry','rec_{}'.format(self.chain[c2].name),jnp.zeros,x.shape)
                c2 += 1
        x = inp
        
        for mdl in self.chain:
            x = mdl(x)
            if c2 in pair.keys():
                for i in self.pair[c2]:
                    v = self.variable('carry','rec_{}'.format(self.chain[i].name))
                    x += reinit_model(mdl)(v.value)
            if c2 in u:
                v = self.variable('carry','rec_{}'.format(self.chain[c2].name))
                v.value = x
            c2 += 1
            
        return x

def RNN(model):
        class rnn(nn.RNNCellBase):
            mdl: Callable = model
            num_feature_axes: int = 1
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
                    unroll=jnp.iinfo(jnp.uint32).max
                )

                if self.is_initializing():
                    _,x = rnn(name='RNN_')((), xs[0])
                    return jnp.stack([x]*xs.shape[0],0)
                else:
                    _,x = scan(name='RNN_')((), xs)
                    return x
                
        class output_model(nn.Module):
            @nn.compact
            def __call__(self,x):
                x = rec()(x)
                return x
        return output_model()


class SNNCell(nn.RNNCellBase):
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
        vars = self.init(rng,jnp.zeros(input_shape))
        return vars['carry']
    
    @property
    def num_feature_axes(self) -> int:
        return 1