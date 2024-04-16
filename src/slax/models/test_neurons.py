import unittest
import jax
import jax.numpy as jnp
import slax as sl
import optax
import flax.linen as nn
from jax.tree_util import tree_map, tree_leaves, tree_structure, tree_unflatten

class TestModelNeurons(unittest.TestCase):

    def test_LIF(self):
        key = jax.random.PRNGKey(0)

        p = sl.LIF().init(key,jnp.zeros(10))

        out = sl.LIF().apply(p,jnp.ones(10)*2)
        b1 = jnp.prod(jax.flatten_util.ravel_pytree(out)[0]).tolist() == 1

        self.assertTrue(b1)

    def test_LTC(self):
        key = jax.random.PRNGKey(0)

        key = jax.random.PRNGKey(0)

        p = sl.LTC().init(key,jnp.zeros(10))

        out = sl.LTC().apply(p,jnp.ones(10)*2)
        b1 = jnp.sum(jax.flatten_util.ravel_pytree(out)[0]).tolist() > 18.06
        self.assertTrue(b1)