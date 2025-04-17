import unittest
import jax
import jax.numpy as jnp
import slax as sl
import optax
import flax.linen as nn
from jax.tree_util import tree_map, tree_leaves, tree_structure, tree_unflatten

class TestModelSurrogates(unittest.TestCase):

    def test_atan(self):
        key = jax.random.PRNGKey(0)

        p = sl.LIF(spike_fn=sl.atan()).init(key,jnp.zeros(10))
        f = lambda x: jnp.square(jnp.square(sl.LIF(spike_fn=sl.atan()).apply(p,x)[0])).sum()
        b1 = jnp.ceil(jnp.unique(jnp.diag(jax.jacfwd(jax.grad(f))(jnp.ones(10)*1.202))))[0].tolist() == -2

        self.assertTrue(b1)

    def test_fast_sigmoid(self):
        key = jax.random.PRNGKey(0)

        p = sl.LIF(spike_fn=sl.fast_sigmoid()).init(key,jnp.zeros(10))
        f = lambda x: jnp.square(jnp.square(sl.LIF(spike_fn=sl.fast_sigmoid()).apply(p,x)[0])).sum()
        b1 = jnp.ceil(jnp.unique(jnp.diag(jax.jacfwd(jax.grad(f))(jnp.ones(10)*1.105))))[0].tolist() == -4

        self.assertTrue(b1)

    def test_multi_gauss(self):
        key = jax.random.PRNGKey(0)

        p = sl.LIF(spike_fn=sl.multi_gauss()).init(key,jnp.zeros(10))
        f = lambda x: jnp.square(jnp.square(sl.LIF(spike_fn=sl.multi_gauss()).apply(p,x)[0])).sum()
        b1 = jnp.ceil(jnp.unique(jnp.diag(jax.jacfwd(jax.grad(f))(jnp.ones(10)*1.22))))[0].tolist() == -2

        self.assertTrue(b1)