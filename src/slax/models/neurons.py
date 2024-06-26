import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import nir
from typing import Any, Callable
from .surrogates import fast_sigmoid, multi_gauss
from .utils import SNNCell, reinit_model, Neuron

class LIF(Neuron):
    '''
    A module for the Leaky Integrate-and-Fire neuron.
    
    Args:
    init_tau: A float or array for the initial leak parameter, which is calculated as sigmoid(init_tau)
    spike_fn: The surrogate spike function, such as fast sigmoid, used in place Heaviside step function
    v_threshold: The membrane potential threshold for spiking. Defaults to 1.0
    v_reset: If the neuron uses a hard reset rather than subtraction-based reset after a spike, the membrane potential returns
    to this value. Defaults to 0.0
    subtraction_reset: Whether the neuron subtracts "1." from the membrane potential after a spike or resets to v_reset.
    Defaults to True
    trainable_tau: Whether the leak parameter is learnable parameter. Defaults to False
    dtype: Data type of the membrane potential. This only matters if you use "initialize_carry". Defaults to float32
    '''
    init_tau: float = 2.
    spike_fn: Callable = fast_sigmoid()
    v_threshold: float = 1.0
    v_reset: float = 0.0
    subtraction_reset: bool = True
    trainable_tau: bool = False
    dtype: Any = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()


    @nn.compact
    def __call__(self,carry,x=None):
        if x == None:
            x = carry
            Vmem = self.variable('carry','Vmem',self.carry_init,jax.random.PRNGKey(0),x.shape,self.dtype)
            vmem = Vmem.value
            hidden_carry = True
        else:
            vmem = carry['Vmem']
            hidden_carry = False

        if self.trainable_tau:
            tau = self.param('tau',lambda x: self.init_tau)
        else:
            tau = self.init_tau

        vmem = nn.sigmoid(tau)*vmem + x
        spikes = self.spike_fn(vmem-self.v_threshold)

        if self.subtraction_reset:
            vmem -= spikes*self.v_threshold
        else:
            vmem = vmem*(1-spikes) + self.v_reset*spikes

        if self.is_initializing() and x==None:
            vmem.value = self.carry_init(jax.random.PRNGKey(0),x.shape,self.dtype)

        if hidden_carry:
            Vmem.value = vmem
            return spikes
        else:
            carry['Vmem'] = vmem
            return carry, spikes
        
    def output_nir(self,params,input_sz):
        dt = 1e-4
        leak = np.array(nn.sigmoid(params['params']['tau']))*np.ones(input_sz)
        v_thresh = np.array(self.v_threshold)*np.ones(input_sz)
        print(leak)
        r = 1/(1-leak)
        print(r)
        tau_mem = dt*r
        v_leak = np.zeros(input_sz)
        
        return nir.LIF(
            tau = tau_mem,
            r = r,
            v_leak = v_leak,
            v_threshold = v_thresh,
            )

class LTC(Neuron):
    '''
    A module for the Liquid Time Constant neuron from Yin, B., Corradi, F., & Bohté, S. M. (2023). Accurate online training of dynamical spiking neural networks through forward propagation through time. Nature Machine Intelligence, 5(5), 518-527.
    
    Args:
    spike_fn: The surrogate spike function, such as fast sigmoid, used in place Heaviside step function. Defaults to `multi_gauss`
    v_reset: If the neuron uses a hard reset rather than subtraction-based reset after a spike, the membrane potential returns
    to this value. Defaults to 0.0
    is_recurrent: Whether or not to apply layer-wise recurrence. Defaults to True
    subtraction_reset: Whether the neuron subtracts "1." from the membrane potential after a spike or resets to v_reset.
    Defaults to False
    dtype: Data type of the membrane potential. This only matters if you use "initialize_carry". Defaults to float32
    '''
    connection_fn: Callable = None
    spike_fn: Callable = multi_gauss()
    v_reset: float = 0.0
    is_recurrent: bool = True
    subtraction_reset: bool = False
    dtype: Any = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, carry, x=None):

        if x == None:
            x = carry

            Vmem = self.variable('carry','Vmem',self.carry_init,jax.random.PRNGKey(0),x.shape,self.dtype)
            Spk_t = self.variable('carry','Spk_t',self.carry_init,jax.random.PRNGKey(0),x.shape,self.dtype)
            B_t = self.variable('carry','B_t',self.carry_init,jax.random.PRNGKey(0),x.shape,self.dtype)

            vmem = Vmem.value
            spk_t = Spk_t.value
            b_t = B_t.value
            hidden_carry = True
        else:
            vmem = carry['Vmem']
            spk_t = carry['Spk_t']
            b_t = carry['B_t']
            hidden_carry = False

        if self.connection_fn == None:
            cf = nn.Dense(x.shape[-1])
        else:
            cf = self.connection_fn

        if self.is_recurrent:
            tauM1 = nn.sigmoid(reinit_model(cf)(x) + reinit_model(cf)(spk_t) + reinit_model(cf)(vmem))
            tauAdp1 = nn.sigmoid(reinit_model(cf)(x) + reinit_model(cf)(spk_t) + reinit_model(cf)(b_t))
        else:
            tauM1 = nn.sigmoid(reinit_model(cf)(x) + reinit_model(cf)(vmem))
            tauAdp1 = nn.sigmoid(reinit_model(cf)(x) + reinit_model(cf)(b_t))

        rho = tauAdp1
        alpha = tauM1
        b_t = rho*b_t + (1-rho)*spk_t
        v_threshold = 0.1 + 1.8*b_t
        vmem = alpha*vmem + (1-alpha)*(-vmem+x)
        spikes = self.spike_fn(vmem - v_threshold)

        if self.subtraction_reset:
            vmem -= spikes*self.v_threshold
        else:
            vmem = vmem*(1-spikes) + self.v_reset*spikes

        if self.is_initializing() and x==None:
            vmem.value = self.carry_init(jax.random.PRNGKey(0),x.shape,self.dtype)

        if hidden_carry:
            Vmem.value = vmem
            Spk_t.value = spikes
            B_t.value = b_t
            return spikes
        else:
            carry['Vmem'] = vmem
            carry['Spk_t'] = spikes
            carry['B_t'] = b_t
            return carry, spikes