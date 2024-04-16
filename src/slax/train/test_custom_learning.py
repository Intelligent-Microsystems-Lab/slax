import unittest
import jax
import jax.numpy as jnp
import slax as sl
import optax
import flax.linen as nn
from jax.tree_util import tree_map, tree_leaves, tree_structure, tree_unflatten

class TestModelSurrogates(unittest.TestCase):

    def test_ostl(self):
        input_sz = 64
        output_sz = 10
        seq_len = 20
        manifold_key = jax.random.PRNGKey(0)
        random_seed = manifold_key
        key = manifold_key
        batch_sz = 32
        dtype = jnp.float32

        batch = sl.randman(manifold_key,random_seed,nb_classes=output_sz,nb_units=input_sz,nb_steps=seq_len,nb_samples=100,batch_sz=batch_sz,dim_manifold=2,alpha=2.,dtype=dtype)

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

        b1 = tree_leaves(out)[-1] > 0.999 and tree_leaves(out)[-2] > 0.999

        self.assertTrue(b1)


    def test_ottt(self):
        input_sz = 64
        output_sz = 10
        seq_len = 20
        manifold_key = jax.random.PRNGKey(0)
        random_seed = manifold_key
        key = manifold_key
        batch_sz = 32
        dtype = jnp.float32

        batch = sl.randman(manifold_key,random_seed,nb_classes=output_sz,nb_units=input_sz,nb_steps=seq_len,nb_samples=100,batch_sz=batch_sz,dim_manifold=2,alpha=2.,dtype=dtype)

        snn = sl.connect([nn.Dense(50),
                  sl.LIF(),
                  nn.Dense(10),
                  sl.LIF()])
        
        OTTT_snn = sl.connect([sl.OTTT([nn.Dense(50),
                  sl.LIF()],nn.sigmoid(2.)),
                  sl.OTTT([nn.Dense(10),
                  sl.LIF()],nn.sigmoid(2.))])
        
        bp_model = snn
        bp_params = bp_model.init(key,jnp.zeros_like(batch[0][0]))

        ottt_model = OTTT_snn
        ottt_params = ottt_model.init(key,jnp.zeros_like(batch[0][0]))
        ottt_params['params'] = tree_unflatten(tree_structure(ottt_params['params']),tree_leaves(bp_params['params']))

        optimizer = optax.adamax(0.01)
        bp_opt_state = optimizer.init(tree_map(jnp.float32,bp_params['params']))
        ottt_opt_state = optimizer.init(tree_map(jnp.float32,ottt_params['params']))

        bp_train = sl.train_offline(bp_model,optax.softmax_cross_entropy,optimizer)
        ottt_eval = sl.train_online_deferred(ottt_model,optax.softmax_cross_entropy,optimizer)

        _,_,_,_,grad = bp_train(bp_params,batch,bp_opt_state,True,unroll=5)
        out = sl.compare_grads(ottt_eval,bp_params,grad,(ottt_params,batch,ottt_opt_state))

        b1 = tree_leaves(out)[-1] > 0.99 and tree_leaves(out)[-2] > 0.99

        self.assertTrue(b1)

    def test_otpe(self):
        input_sz = 64
        output_sz = 10
        seq_len = 20
        manifold_key = jax.random.PRNGKey(0)
        random_seed = manifold_key
        key = manifold_key
        batch_sz = 32
        dtype = jnp.float32

        batch = sl.randman(manifold_key,random_seed,nb_classes=output_sz,nb_units=input_sz,nb_steps=seq_len,nb_samples=100,batch_sz=batch_sz,dim_manifold=2,alpha=2.,dtype=dtype)

        snn = sl.connect([nn.Dense(50),
                  sl.LIF(),
                  nn.Dense(10),
                  sl.LIF()])
        
        OTPE_snn = sl.connect([sl.DenseOTPE([nn.Dense(50),
                  sl.LIF()],nn.sigmoid(2.)),
                  sl.DenseOSTL([nn.Dense(10),
                  sl.LIF()])])

        bp_model = snn
        bp_params = bp_model.init(key,jnp.zeros_like(batch[0][0]))

        otpe_model = OTPE_snn
        otpe_params = otpe_model.init(key,jnp.zeros_like(batch[0][0]))
        otpe_params['params'] = tree_unflatten(tree_structure(otpe_params['params']),tree_leaves(bp_params['params']))

        optimizer = optax.adamax(0.01)
        bp_opt_state = optimizer.init(tree_map(jnp.float32,bp_params['params']))
        otpe_opt_state = optimizer.init(tree_map(jnp.float32,otpe_params['params']))

        bp_train = sl.train_offline(bp_model,optax.softmax_cross_entropy,optimizer)
        otpe_eval = sl.train_online_deferred(otpe_model,optax.softmax_cross_entropy,optimizer)

        _,_,_,_,grad = bp_train(bp_params,batch,bp_opt_state,True,unroll=5)
        out = sl.compare_grads(otpe_eval,bp_params,grad,(otpe_params,batch,otpe_opt_state))

        b1 = tree_leaves(out)[-3] > 0.85 and tree_leaves(out)[-4] > 0.85

        self.assertTrue(b1)

    def test_rtrl(self):
        input_sz = 64
        output_sz = 10
        seq_len = 20
        manifold_key = jax.random.PRNGKey(0)
        random_seed = manifold_key
        key = manifold_key
        batch_sz = 32
        dtype = jnp.float32

        batch = sl.randman(manifold_key,random_seed,nb_classes=output_sz,nb_units=input_sz,nb_steps=seq_len,nb_samples=100,batch_sz=batch_sz,dim_manifold=2,alpha=2.,dtype=dtype)

        snn = sl.connect([nn.Dense(50),
                  sl.LIF(),
                  nn.Dense(10),
                  sl.LIF()])
        
        RTRL_snn = sl.connect([sl.RTRL([sl.connect([nn.Dense(50),
                  sl.LIF(),
                  nn.Dense(10),
                  sl.LIF()])])])
        
        bp_model = snn
        bp_params = bp_model.init(key,jnp.zeros_like(batch[0][0]))

        rtrl_model = RTRL_snn
        rtrl_params = rtrl_model.init(key,jnp.zeros_like(batch[0][0]))
        rtrl_params['params'] = tree_unflatten(tree_structure(rtrl_params['params']),tree_leaves(bp_params['params']))

        optimizer = optax.adamax(0.01)
        bp_opt_state = optimizer.init(tree_map(jnp.float32,bp_params['params']))
        rtrl_opt_state = optimizer.init(tree_map(jnp.float32,rtrl_params['params']))

        bp_train = sl.train_offline(bp_model,optax.softmax_cross_entropy,optimizer)
        rtrl_eval = sl.train_online_deferred(rtrl_model,optax.softmax_cross_entropy,optimizer)

        _,_,_,_,grad = bp_train(bp_params,batch,bp_opt_state,True,unroll=5)
        out = sl.compare_grads(rtrl_eval,bp_params,grad,(rtrl_params,batch,rtrl_opt_state))

        b1 = tree_leaves(out)[-3] > 0.999 and tree_leaves(out)[-4] > 0.999

        self.assertTrue(b1)