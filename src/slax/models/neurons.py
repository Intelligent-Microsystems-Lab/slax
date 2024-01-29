import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable
from .surrogates import fast_sigmoid, ActFun_adp
from .utils import SNNCell

class LIF(SNNCell):
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
            Vmem = self.variable('carry','Vmem',self.carry_init,jax.random.PRNGKey(0),x.shape)
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
            vmem *= (1-spikes)

        if self.is_initializing() and x==None:
            vmem.value = self.carry_init(jax.random.PRNGKey(0),x.shape)

        if hidden_carry:
            Vmem.value = vmem
            return spikes
        else:
            carry['Vmem'] = vmem
            return carry, spikes
        

def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=1):
    alpha = tau_m
    
    # ro = tau_adp

    # if isAdapt:
    #     beta = 1.8
    # else:
    #     beta = 0.

    # b = ro * b + (1 - ro) * spike
    # B = b_j0 + beta * b
    B = 1.


    d_mem = -mem + inputs
    mem = mem + d_mem*alpha
    inputs_ = mem - B
    mem_out = mem

    spike = ActFun_adp()(inputs_)#act_fun_adp(inputs_)  # act_fun : approximation firing function
    mem = (1-spike)*mem

    return mem, spike, B, b, mem_out


class LTC(SNNCell):
    init_tau: float = 2.
    spike_fn: Callable = ActFun_adp
    v_threshold: float = 1.0
    v_reset: float = 0.0
    subtraction_reset: bool = True
    trainable_tau: bool = False
    dtype: Any = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):

        mem_t = self.variable('state','mem_t',self.carry_init,jax.random.PRNGKey(0),x.shape)
        spk_t = self.variable('state','spk_t',self.carry_init,jax.random.PRNGKey(0),x.shape)
        b_t = self.variable('state','b_t',self.carry_init,jax.random.PRNGKey(0),x.shape)


        tauM1 = nn.sigmoid(nn.Dense(x.shape)(x+mem_t.value))
        tauAdp1 = nn.sigmoid(nn.Dense(x.shape)(x+b_t.value))
        

        mem_1,spk_1,_,b_1,mem_out = mem_update_adp(x, mem=mem_t.value,spike=spk_t.value,
                                        tau_adp=tauAdp1,tau_m=tauM1,b =b_t.value)

        mem_t.value = mem_1
        spk_t.value = spk_1
        b_t.value = b_1

        if self.output_mem:
            return mem_out
        else:
            return spk_1

    # def compute_output_size(self):
    #     return [self.hidden_size]