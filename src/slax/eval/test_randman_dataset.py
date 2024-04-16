import unittest
import jax
import jax.numpy as jnp
import slax as sl

class TestRandmanDataset(unittest.TestCase):

    def test_upper(self):
        input_sz = 64
        output_sz = 10
        seq_len = 20
        manifold_key = jax.random.PRNGKey(0)
        random_seed = manifold_key
        batch_sz = 32
        dtype = jnp.float32
        batch = sl.randman(manifold_key,random_seed,nb_classes=output_sz,nb_units=input_sz,nb_steps=seq_len,nb_samples=100,batch_sz=batch_sz,dim_manifold=2,alpha=2.,dtype=dtype)
        self.assertEqual(batch[0].shape, (20, 32, 64))
        self.assertEqual(batch[1].shape, (20, 32, 10))


# if __name__ == '__main__':
#     unittest.main()