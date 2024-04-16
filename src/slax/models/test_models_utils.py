import unittest
import jax
import jax.numpy as jnp
import slax as sl
import optax
import flax.linen as nn
from jax.tree_util import tree_map, tree_leaves, tree_structure, tree_unflatten

class TestModelUtils(unittest.TestCase):

    def test_utils(self):
        key = jax.random.PRNGKey(0)
        x = jnp.ones((10,32,64))

        snn = sl.connect([nn.Dense(50),
                  sl.RNN(sl.LIF()),
                  nn.Dense(10),
                  sl.RNN(sl.LIF())],return_carry=False)
        bp_model = snn
        bp_params = bp_model.init(key,jnp.zeros_like(x))
        b1 = bp_model.apply(bp_params,x).shape == (10,32,10)

        self.assertTrue(b1)


