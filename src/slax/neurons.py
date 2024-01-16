import jax
import flax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Sequence
from .surrogates import fast_sigmoid, ActFun_adp
from jax.tree_util import Partial, tree_leaves, tree_map
from jax.lax import stop_gradient as stop_grad

class LIF(nn.Module):
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
    init_tau: float
    spike_fn: Callable = fast_sigmoid()
    v_threshold: float = 1.0
    v_reset: float = 0.0
    subtraction_reset: bool = True
    trainable_tau: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self,carry,x):
        if carry==None:
            carry = {'Vmem': jnp.zeros_like(x)}
        if self.trainable_tau:
            tau = self.param('tau',lambda x: self.init_tau)
        else:
            tau = self.init_tau
        carry['Vmem'] = nn.sigmoid(tau)*carry['Vmem'] + x
        spikes = self.spike_fn(carry['Vmem']-self.v_threshold)
        if self.subtraction_reset:
            carry['Vmem'] -= spikes*self.v_threshold
        else:
            carry['Vmem'] *= (1-spikes)
        return carry, spikes
    
    @nn.nowrap
    def initialize_carry(self,output_shape):
        return {'Vmem': jnp.zeros(output_shape,self.dtype)}



        



gamma = .5  # gradient scale
lens = 0.3

def gaussian(x, mu=0., sigma=.5):
    return jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / jnp.sqrt(2 * jnp.pi) / sigma


# class ActFun_adp(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):  # input = membrane potential- threshold
#         ctx.save_for_backward(input)
#         return input.gt(0).float()  # is firing ???

#     @staticmethod
#     def backward(ctx, grad_output):  # approximate the gradients
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         # temp = abs(input) < lens
#         scale = 6.0
#         hight = .15
#         # temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
#         temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
#                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
#                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
#         # temp =  gaussian(input, mu=0., sigma=lens)
#         return grad_input * temp.float() * gamma
        # return grad_input

#act_fun_adp = ActFun_adp.apply

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



class LTC(nn.Module):
    hidden_size: int
    is_rec: bool = True
    output_mem: bool = False

    @nn.compact
    def __call__(self, x_t):

        mem_t = self.variable('state','mem_t',jnp.zeros,self.hidden_size)
        spk_t = self.variable('state','spk_t',jnp.zeros,self.hidden_size)
        b_t = self.variable('state','b_t',jnp.zeros,self.hidden_size)

        if self.is_rec:
            dense_x = nn.Dense(self.hidden_size)(jnp.concatenate((x_t,spk_t.value),axis=-1))
        else:
            dense_x = nn.Dense(self.hidden_size)(x_t)
        tauM1 = nn.sigmoid(nn.Dense(self.hidden_size)(dense_x+mem_t.value))
        tauAdp1 = nn.sigmoid(nn.Dense(self.hidden_size)(dense_x+b_t.value))
        

        mem_1,spk_1,_,b_1,mem_out = mem_update_adp(dense_x, mem=mem_t.value,spike=spk_t.value,
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