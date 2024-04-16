import unittest
import jax
import jax.numpy as jnp
import slax as sl
import optax
import flax.linen as nn
from jax.tree_util import tree_map, tree_leaves, tree_structure, tree_unflatten

class TestEvalUtils(unittest.TestCase):

    def test_upper(self):
        key = jax.random.PRNGKey(0)

        batch = (jnp.ones((20, 32, 64)),jnp.ones((20, 32, 10)))

        snn = sl.connect([nn.Dense(50),
                  sl.LIF(),
                  nn.Dense(10),
                  sl.LIF()])
        OSTL_snn = sl.connect([sl.DenseOSTL([nn.Dense(50),
                  sl.LIF()]),
                  sl.DenseOSTL([nn.Dense(10),
                  sl.LIF()])])

        bp_model = snn
        bp_params = bp_model.init(key,jnp.zeros_like(batch[0][0]))

        ostl_model = OSTL_snn
        ostl_params = ostl_model.init(key,jnp.zeros_like(batch[0][0]))
        ostl_params['params'] = tree_unflatten(tree_structure(ostl_params['params']),tree_leaves(bp_params['params']))

        optimizer = optax.adamax(0.01)
        bp_opt_state = optimizer.init(tree_map(jnp.float32,bp_params['params']))
        ostl_opt_state = optimizer.init(tree_map(jnp.float32,ostl_params['params']))


        bp_train = sl.train_offline(bp_model,optax.softmax_cross_entropy,optimizer)
        ostl_eval = sl.train_online_deferred(ostl_model,optax.softmax_cross_entropy,optimizer)
        _,_,_,_,grad = bp_train(bp_params,batch,bp_opt_state,True,unroll=5)
        
        out = sl.compare_grads(ostl_eval,bp_params,grad,(ostl_params,batch,ostl_opt_state))
        print(out)
        b1 = (jnp.max(jnp.stack(tree_leaves(out))) > 0.9).tolist()#(jnp.prod(jnp.isclose(jnp.stack(tree_leaves(out)),jnp.array([0.8269028 , 0.7635841 , 0.9998024 , 0.20835316]))) == 1).tolist()

        out = sl.compare_grads(ostl_eval,bp_params,grad,(ostl_params,batch,ostl_opt_state),sl.global_cosine_similarity)
        b2 = (out > 0.9).tolist()#jnp.isclose(out,jnp.array(0.9996188879013062)).tolist()
        print(out)
        self.assertTrue(b1)

        self.assertTrue(b2)
        