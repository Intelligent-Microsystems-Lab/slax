# Adapted from https://github.com/fzenke/randman
# If using, please cite Zenke, F., and Vogels, T.P. (2021). The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks. Neural Computation 1–27.



# MIT License
# Copyright (c) <year> <copyright holders>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from .jax_randman import JaxRandman as Randman

def standardize(x,eps=1e-7):
    mi = x.min(0)
    ma = x.max(0)
    return (x-mi)/(ma-mi+eps)

@Partial(jax.jit,static_argnames=('nb_classes','nb_units','nb_steps','dim_manifold','nb_spikes','nb_samples','alpha','shuffle','time_encode','batch_sz','dtype'))
def randman(manifold_seed, random_seed, nb_classes=10, nb_units=100, nb_steps=100, dim_manifold=2, nb_spikes=1, nb_samples=1000, alpha=2.0, shuffle=True, time_encode=True, batch_sz=None, dtype=jnp.float32):
    """ Adapted from https://github.com/fzenke/randman.
    If using this, please cite Zenke, F., and Vogels, T.P. (2021). The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks. Neural Computation 1–27.
    
    Generates event-based generalized spiking randman classification dataset.
    In this dataset each unit fires a fixed number of spikes. If information is set to be encoded in time, then only
    nb_spikes occurs per neuron for each trial at a specific time, which precludes rate/count based learning. If encoded
    in the spike rate/count, then the number of spikes each neuron generates encodes information while spike times are random.
    Args: 
        manifold_seed: The JAX PRNG key that determines the random manifolds
        random_seed: The JAX PRNG random seed, which determines random elements such as sampling from the manifold.
        nb_classes: The number of classes to generate. Defaults to 10
        nb_units: The number of units to assume. Defaults to 100
        nb_steps: The number of time steps to assume. Defaults to 100
        dim_manifold: The dimensionality of the hypercube the random manifold is generated along. Defaults to 2
        nb_spikes: The number of spikes per unit. Defaults to 1
        nb_samples: Number of samples from each manifold per class. Defaults to 1_000
        alpha: Randman smoothness parameter. Defaults to 2.0
        shuffe: Whether to shuffle the dataset. Defaults to True
        time_encode: Whether to encode information in spike timeing (alternative being rate/count). Defaults to True
        batch_sz: The size of the batch dimension. Defaults to None
        dtype: The data type of the output spikes and labels. Defaults to float32
    """
  
    data = []
    labels = []
    targets = []

    uniform_key, shuffle_key, sample_key = jax.random.split(random_seed,3)
    
    max_value = jnp.iinfo(jnp.int32).max
    randman_seeds = jax.random.randint(manifold_seed, shape=(nb_classes,nb_spikes),maxval=max_value,minval=0)

    for k in range(nb_classes):
        x = jax.random.uniform(uniform_key,(nb_samples,dim_manifold))
        submans = [ Randman(nb_units, dim_manifold, alpha=alpha, seed=randman_seeds[k,i]) for i in range(nb_spikes) ]
        units = []
        times = []
        for i,rm in enumerate(submans):
            y = rm.eval_manifold(x)
            y = standardize(y)
            units.append(jnp.repeat(jnp.arange(nb_units).reshape(1,-1),nb_samples,axis=0))
            times.append(y)

        units = jnp.concatenate(units,axis=1)
        events = jnp.concatenate(times,axis=1)

        data.append(events)
        labels.append(k*jnp.ones(len(units)))
        targets.append(x)

    data = jnp.concatenate(data, axis=0)
    labels = jnp.array(jnp.concatenate(labels, axis=0), dtype=jnp.int32)
    targets = jnp.concatenate(targets, axis=0)

    idx = jnp.arange(len(data))
    if shuffle:
        idx = jax.random.permutation(shuffle_key,idx,independent=True)
    data = data[idx]
    labels = labels[idx]


    if time_encode:
        points = jnp.tile(jnp.int32(jnp.floor(data*nb_steps)),(nb_steps,1,1))
        vals = jnp.tile(jnp.arange(nb_steps),(nb_classes*nb_samples,nb_units,1)).transpose(2,0,1)
        data = jnp.where(vals==points,1,0)
        labels = jnp.tile(jax.nn.one_hot(labels,nb_classes),(nb_steps,1,1))


    else:
        points = jnp.tile(jnp.int32(jnp.floor(data*nb_steps)),(nb_steps,1,1))
        vals = jnp.tile(jnp.arange(nb_steps),(nb_classes*nb_samples,nb_units,1)).transpose(2,0,1)
        data = jnp.where(vals<=points,1,0)
        data = jax.random.permutation(sample_key,data,axis=0,independent=True)
        labels = jnp.tile(jax.nn.one_hot(labels,nb_classes),(nb_steps,1,1))

    if batch_sz == None:
        return dtype(data), dtype(labels)
    else:
        return dtype(data[:,:batch_sz,:]), dtype(labels[:,:batch_sz,:])