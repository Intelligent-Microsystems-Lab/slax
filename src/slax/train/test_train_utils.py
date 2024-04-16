import unittest
import jax
import jax.numpy as jnp
import slax as sl
import optax
import flax.linen as nn
from jax.tree_util import tree_map, tree_leaves, tree_structure, tree_unflatten

class TestTrainUtils(unittest.TestCase):

    def test_all(self):
        key = jax.random.PRNGKey(0)

        batch = (jnp.ones((20,32,64)),jnp.ones((20,32,10)))

        snn = sl.connect([nn.Dense(50),
                  sl.LIF(),
                  nn.Dense(10),
                  sl.LIF()])


        bp_model = snn
        bp_params = bp_model.init(key,jnp.zeros_like(batch[0]))

        optimizer = optax.adamax(0.01)
        bp_opt_state = optimizer.init(tree_map(jnp.float32,bp_params['params']))
        bp_train = sl.train_offline(bp_model,optax.softmax_cross_entropy,optimizer)
        fptt_train = sl.FPTT(bp_model,optax.softmax_cross_entropy,optimizer)
        bp_eval = sl.train_online_deferred(bp_model,optax.softmax_cross_entropy,optimizer)
        bp_online = sl.train_online(bp_model,optax.softmax_cross_entropy,optimizer)
        _,_,_,_ = bp_train(bp_params,batch,bp_opt_state)
        out_1 = fptt_train(bp_params,batch,bp_opt_state)
        out_2 = bp_online(bp_params,batch,bp_opt_state)
        out_3 = bp_eval(bp_params,batch,bp_opt_state)