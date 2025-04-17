import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx
import numpy as np
import nir
from typing import Any, Callable
from slax.model.surrogate import fast_sigmoid, multi_gauss
from slax.model.utils import reinit_model, Neuron

class LIF(Neuron):
    '''
    A module for the Leaky Integrate-and-Fire neuron.
    
    Args:
    shape: The input shape as an integer or tuple
    init_tau: A float or array for the initial leak parameter, which is calculated as sigmoid(init_tau)
    spike_fn: The surrogate spike function, such as fast sigmoid, used in place Heaviside step function
    v_threshold: The membrane potential threshold for spiking. Defaults to 1.0
    v_reset: If the neuron uses a hard reset rather than subtraction-based reset after a spike, the membrane potential returns
    to this value. Defaults to 0.0
    subtraction_reset: Whether the neuron subtracts "1." from the membrane potential after a spike or resets to v_reset.
    Defaults to True
    trainable_tau: Whether the leak parameter is learnable parameter. Defaults to False
    carry_init: Initializer for the carry state
    dtype: Data type of the membrane potential. This only matters if you use "initialize_carry". Defaults to float32
    '''
    def __init__(self,size=1,init_tau=2.,spike_fn=fast_sigmoid(),v_threshold=1.0,v_reset=0.0,subtraction_reset=True,train_tau=False,
                 carry_init=jnp.zeros,stop_du_ds=False,output_Vmem=False,no_reset=False,dtype=jnp.float32):
        self.Vmem = nnx.Variable(carry_init(size))
        if train_tau:
            self.tau = nnx.Param(init_tau)
        else:
            self.tau = init_tau
        self.size = size
        self.out_features = size
        self.spike_fn = spike_fn
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.subtraction_reset = subtraction_reset
        self.train_tau = train_tau
        self.stop_du_ds = stop_du_ds
        self.output_Vmem = output_Vmem
        self.dtype = dtype
        self.no_reset = no_reset
        
    def __call__(self,x):
        if self.train_tau:
            self.Vmem.value = nn.sigmoid(self.tau.value)*self.Vmem.value + x
        else:
            self.Vmem.value = nn.sigmoid(self.tau)*self.Vmem.value + x
        spikes = self.spike_fn(self.Vmem.value-self.v_threshold)

        if not self.no_reset:
            if self.subtraction_reset:
                self.Vmem.value -= spikes*self.v_threshold
            else:
                self.Vmem.value = self.Vmem.value*(1-spikes) + self.v_reset*spikes

        if self.output_Vmem:
            return self.Vmem.value
        else:
            return spikes
        
    def output_nir(self):
        dt = 1e-4
        if self.train_tau:
            leak = np.array(nn.sigmoid(self.tau.value))*np.ones(self.size)
        else:
            leak = np.array(nn.sigmoid(self.tau))*np.ones(self.size)
        v_thresh = np.array(self.v_threshold)*np.ones(self.size)
        r = 1/(1-leak)
        tau_mem = dt*r
        v_leak = np.zeros(self.size)
        
        return nir.LIF(
            tau = tau_mem,
            r = r,
            v_leak = v_leak,
            v_threshold = v_thresh,
            )