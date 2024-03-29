import jax
import jax.numpy as jnp

def fast_sigmoid(slope=25):
    '''
    A function that returns the Heaviside step function with forward-mode autodiff compatible surrogate derivative for the fast-sigmoid
    function from Zenke, F., & Ganguli, S. (2018). Superspike: Supervised learning in multilayer spiking neural networks. Neural computation, 30(6), 1514-1541. 

    Args:
    slope: The sharpness factor of the fast sigmoid function
    '''
    @jax.custom_jvp
    def fs(x):
      # if not dtype float grad ops wont work
      return jnp.array(x >= 0.0, dtype=x.dtype)
    
    
    @fs.defjvp
    def fs_bwd(primal, tangent):
        x, = primal
        t, = tangent
        alpha = slope

        scale = 1 / (alpha * jnp.abs(x) + 1.) ** 2
        return (fs(x),scale*t)
    
    return fs

def atan(alpha=2.):
    @jax.custom_jvp
    def arctangent(x):
        # if not dtype float grad ops wont work
      return jnp.array(x >= 0.0, dtype=x.dtype)
    
    @arctangent.defjvp
    def backward(primal, tangent):
        x, = primal
        t, = tangent
        grad = alpha / 2 / (1 + (jnp.pi / 2 * alpha * x)**2) * t
        return arctangent(x),grad
    
    return arctangent

gamma = .5  # gradient scale
lens = 0.3

def gaussian(x, mu=0., sigma=.5):
    return jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / jnp.sqrt(2 * jnp.pi) / sigma
def ActFun_adp():
    @jax.custom_jvp
    def forward(x):
        return jnp.array(x >= 0.0, dtype=x.dtype)  # is firing ???

    @forward.defjvp
    def backward(primal, tangent):  # approximate the gradients
        input, = primal
        grad_input, = tangent

        scale = 6.0
        hight = .15

        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight

        return (forward(input),grad_input * temp * gamma)
    return forward