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
    '''
    A function that returns the Heaviside step function with forward-mode autodiff compatible surrogate derivative for the Arctangent
    function from Fang, W., Yu, Z., Chen, Y., Masquelier, T., Huang, T., & Tian, Y. (2021). Incorporating learnable membrane time constant to enhance learning of spiking neural networks. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 2661-2671). 

    Args:
    alpha: The sharpness factor of the fast sigmoid function
    '''
    @jax.custom_jvp
    def arctangent(x):
      return jnp.array(x >= 0.0, dtype=x.dtype)
    
    @arctangent.defjvp
    def backward(primal, tangent):
        x, = primal
        t, = tangent
        grad = alpha / 2 / (1 + (jnp.pi / 2 * alpha * x)**2) * t
        return arctangent(x),grad
    
    return arctangent


def multi_gauss(gamma = 0.5, lens = 0.3, scale = 6.0, height = 0.15):
    '''
    A function that returns the Heaviside step function with forward-mode autodiff compatible surrogate derivative for the Multi-Gaussian
    function from Yin, B., Corradi, F., & Bohté, S. M. (2021). Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks. Nature Machine Intelligence, 3(10), 905-913. 
    '''
    def gaussian(x, mu=0., sigma=.5):
        return jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / jnp.sqrt(2 * jnp.pi) / sigma

    @jax.custom_jvp
    def forward(x):
        return jnp.array(x >= 0.0, dtype=x.dtype)

    @forward.defjvp
    def backward(primal, tangent):
        input, = primal
        grad_input, = tangent

        temp = gaussian(input, mu=0., sigma=lens) * (1. + height) \
               - gaussian(input, mu=lens, sigma=scale * lens) * height \
               - gaussian(input, mu=-lens, sigma=scale * lens) * height

        return (forward(input),grad_input * temp * gamma)
    return forward