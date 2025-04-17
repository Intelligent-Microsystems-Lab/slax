import nir
import numpy as np
from flax import nnx
from functools import partial
import jax.numpy as jnp
from ..model.neuron import LIF


def nir_dense(mdl):
    if mdl.use_bias == True:
        return nir.Affine(
            weight = np.array(mdl.kernel.value),
            bias = np.array(mdl.bias.value),
        )
    else:
        return nir.Linear(
            weight = np.array(mdl.kernel.value)
        )
    
def from_nir_modules(node):
    if node.__class__ == nir.Affine:
        n = nnx.Linear(node.weight.shape[0],node.weight.shape[1],rngs=nnx.Rngs(0))
        n.bias.value = jnp.array(node.bias)
        n.kernel.value = jnp.array(node.weight)
    elif node.__class__ == nir.Linear:
        n = nnx.Linear(node.weight.shape[0],node.weight.shape[1],use_bias=False,rngs=nnx.Rngs(0))
        n.kernel.value = jnp.array(node.weight)
    elif node.__class__ == nir.LIF:
        leak = 1-(1/node.r)
        tau = np.log(leak / (1 - leak))
        n = LIF(node.tau.shape,init_tau=jnp.array(tau))
    return n

def nir_wrapper(name,graph):
    graph[-1].append(name)
    graph.append([])
    graph[-1].append(name)
    return graph

def wrap_nir(model):
    mdl = model.__deepcopy__()
    nir_nodes = {}
    nir_edges = [[]]
    for m in mdl.iter_modules():
        if m[1].__module__ == nnx.Linear.__module__:
            nir_nodes[m[0][0]]=nir_dense(m[1])
            setattr(mdl,m[0][0],partial(nir_wrapper,m[0][0]))
        elif 'output_nir' in dir(m[1]):
            nir_nodes[m[0][0]]=m[1].output_nir()
            setattr(mdl,m[0][0],partial(nir_wrapper,m[0][0]))
    nir_edges = mdl(nir_edges)
    return nir_nodes, nir_edges[1:-1]

def to_nir(model):
    nir_nodes, nir_edges = wrap_nir(model)
    return nir.NIRGraph(nir_nodes,nir_edges)

def from_nir(graph):
    class SNN(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            for key in graph.nodes.keys():
                node = graph.nodes[key]
                setattr(self,key,from_nir_modules(node))
        def __call__(self,x):
            for i in range(len(graph.edges)):
                x = getattr(self,graph.edges[i][0])(x)
            x = getattr(self,graph.edges[-1][1])(x)
            return x
    return SNN(rngs=nnx.Rngs(0))