from utils import timeit, benchmark_framework, log_result

import argparse
import numpy as np
import os


import jax
print(jax.numpy.ones(1).devices())

def rockpool_jax():
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"
    import jax
    import jax.numpy as jnp
    import slax as sl
    from rockpool.nn.modules import LIFJax, LinearJax
    from rockpool.nn.combinators import Sequential
    import rockpool


    key = jax.random.PRNGKey(0)

    benchmark_title = f"Rockpool_jax<br>v{rockpool.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        SNN = Sequential(
            LinearJax(shape=(n_neurons, n_neurons)),
            LIFJax(n_neurons,max_spikes_per_dt=1),
            LinearJax(shape=(n_neurons, n_neurons)),
            LIFJax(n_neurons,max_spikes_per_dt=1),
            LinearJax(shape=(n_neurons, n_neurons)),
            LIFJax(n_neurons,max_spikes_per_dt=1),
        )

        input_static = jax.random.normal(key,shape=(n_steps, batch_size, n_neurons), dtype=jnp.float32)

        params = SNN.parameters()

        @jax.jit
        def net_eval(params,events):
            mdl = SNN.reset_state()

            # - Apply the parameters
            mdl = mdl.set_attributes(params)

            # - Evolve the network
            output, _, _ = mdl(events, record=True)
            return output.sum()

        model = (net_eval,params)

        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        net_eval, params = model
        net_eval(params, input_static)
        bench_dict["output"] = input_static
        return bench_dict

    def backward_fn(bench_dict):
        input_static = bench_dict["input"]
        net_eval, params = bench_dict["model"]
        jax.grad(net_eval)(params, input_static)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def spyx_full():
    import spyx
    import spyx.nn as snn
    import jax
    import jax.numpy as jnp
    import jmp
    import haiku as hk

    # policy = jmp.get_policy("full")

    # hk.mixed_precision.set_policy(hk.Linear, policy)
    # hk.mixed_precision.set_policy(snn.LIF, policy)

    benchmark_title = f"Spyx full-precision v{spyx.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        def Model(x):
            x = hk.BatchApply(hk.Linear(n_neurons))(x)
            core = hk.DeepRNN([snn.LIF((n_neurons,),beta=0.5, activation=spyx.axn.Axon(spyx.axn.arctan())),])
            x, V = hk.dynamic_unroll(core, x, core.initial_state(x.shape[1]), time_major=True, unroll=jnp.iinfo(jnp.uint32).max)

            x = hk.BatchApply(hk.Linear(n_neurons))(x)
            core = hk.DeepRNN([snn.LIF((n_neurons,),beta=0.5, activation=spyx.axn.Axon(spyx.axn.arctan())),])
            x, V = hk.dynamic_unroll(core, x, core.initial_state(x.shape[1]), time_major=True, unroll=jnp.iinfo(jnp.uint32).max)

            x = hk.BatchApply(hk.Linear(n_neurons))(x)
            core = hk.DeepRNN([snn.LIF((n_neurons,),beta=0.5, activation=spyx.axn.Axon(spyx.axn.arctan())),])
            x, V = hk.dynamic_unroll(core, x, core.initial_state(x.shape[1]), time_major=True, unroll=jnp.iinfo(jnp.uint32).max)
            

            return x

        key = jax.random.PRNGKey(0)
        input_static = jax.random.normal(key,shape=(n_steps, batch_size, n_neurons), dtype=jnp.float32)

        
        # Since there's nothing stochastic about the network, we can avoid using an RNG as a param!
        SNN = hk.without_apply_rng(hk.transform(Model))
        params = SNN.init(rng=key, x=input_static)

        @jax.jit
        def net_eval(weights, events):
            readout = SNN.apply(weights, events)
            return readout.sum()

        model = (net_eval, params)

        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        net_eval, params = model
        net_eval(params, input_static)
        bench_dict["output"] = input_static
        return bench_dict

    def backward_fn(bench_dict):
        input_static = bench_dict["input"]
        net_eval, params = bench_dict["model"]
        jax.grad(net_eval)(params, input_static)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

def slax_full():
    import jax
    import jax.numpy as jnp
    import slax as sl
    import flax.linen as nn

    key = jax.random.PRNGKey(0)
    dtype = jnp.float16

    benchmark_title = f"slax full-precision v{'0.0.1'}"

    def new_policy(prim,*_,**__):
        v = _
        #print(v[0].dtype)
        #print(*_)
        if v[0].dtype == jnp.int32:
            #print('yay')
            return True
        else:
            return False
    np = jax.checkpoint_policies.save_from_both_policies(jax.checkpoint_policies.dots_saveable,new_policy)

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        class Model(nn.Module):
            @nn.compact
            def __call__(self,x):
                x = nn.Dense(n_neurons,param_dtype=dtype)(x)
                x = sl.RNN(sl.LIF(2.,spike_fn=sl.atan(),dtype=dtype),50)(x)
                x = nn.Dense(n_neurons,param_dtype=dtype)(x)
                x = sl.RNN(sl.LIF(2.,spike_fn=sl.atan(),dtype=dtype),50)(x)
                x = nn.Dense(n_neurons,param_dtype=dtype)(x)
                x = sl.RNN(sl.LIF(2.,spike_fn=sl.atan(),dtype=dtype),50)(x)

                return x

        input_static = jax.random.normal(key,shape=(n_steps, batch_size, n_neurons), dtype=dtype)

        # Since there's nothing stochastic about the network, we can avoid using an RNG as a param!
        SNN = Model()
        params = SNN.init(key, input_static)


        @jax.jit
        def net_eval(weights, events):
            readout = SNN.apply(weights, events,mutable=['carry'])
            traces, V_f = readout
            return traces.sum()
            #return readout[0].sum()

        model = (net_eval, params)

        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        net_eval, params = model
        net_eval(params, input_static)
        bench_dict["output"] = input_static
        return bench_dict

    def backward_fn(bench_dict):
        input_static = bench_dict["input"]
        net_eval, params = bench_dict["model"]
        jax.grad(net_eval)(params, input_static)


    #import pdb;pdb.set_trace()

    return prepare_fn, forward_fn, backward_fn, benchmark_title

def slax_flax():
    import jax
    import jax.numpy as jnp
    import slax as sl
    import flax.linen as nn

    key = jax.random.PRNGKey(0)
    dtype = jnp.float16

    benchmark_title = f"slax flax v{'0.0.1'}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        class Model(nn.Module):
            @nn.compact
            def __call__(self,x):
                x = nn.Dense(n_neurons,param_dtype=dtype)(x)
                x = nn.RNN(sl.LIF(2.,spike_fn=sl.atan(),dtype=dtype),time_major=True,unroll=500)(x)#jnp.iinfo(jnp.uint32).max)(x)
                x = nn.Dense(n_neurons,param_dtype=dtype)(x)
                x = nn.RNN(sl.LIF(2.,spike_fn=sl.atan(),dtype=dtype),time_major=True,unroll=500)(x)#jnp.iinfo(jnp.uint32).max)(x)
                x = nn.Dense(n_neurons,param_dtype=dtype)(x)
                x = nn.RNN(sl.LIF(2.,spike_fn=sl.atan(),dtype=dtype),time_major=True,unroll=500)(x)#jnp.iinfo(jnp.uint32).max)(x)

                return x

        input_static = jax.random.normal(key,shape=(n_steps, batch_size, n_neurons), dtype=dtype)

        # Since there's nothing stochastic about the network, we can avoid using an RNG as a param!
        SNN = sl.RNN(sl.get_connect([nn.Dense(n_neurons,param_dtype=dtype),#Model()
                              sl.LIF(2.,spike_fn=sl.atan(),dtype=dtype),
                              nn.Dense(n_neurons,param_dtype=dtype),
                              sl.LIF(2.,spike_fn=sl.atan(),dtype=dtype),
                              nn.Dense(n_neurons,param_dtype=dtype),
                              sl.LIF(2.,spike_fn=sl.atan(),dtype=dtype)]),unroll=500)
        #SNN = Model()
        params = SNN.init(key, input_static)

        @jax.jit
        def net_eval(weights, events):
            readout = SNN.apply(weights, events, mutable='carry')
            traces = readout[0]
            return traces.sum()
            #return readout[0].sum()

        model = (net_eval, params)

        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        net_eval, params = model
        net_eval(params, input_static)
        bench_dict["output"] = input_static
        return bench_dict

    def backward_fn(bench_dict):
        input_static = bench_dict["input"]
        net_eval, params = bench_dict["model"]
        jax.grad(net_eval)(params, input_static)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


benchmarks = {

    "spyx_full": spyx_full,
    "slax_full": slax_full,
    "rockpool_jax": rockpool_jax,
    "slax_flax": slax_flax,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=benchmarks.keys())
    parser.add_argument("batch_size")
    args = parser.parse_args()

    benchmark = benchmarks[args.benchmark]

    batch_size = int(args.batch_size)
    n_steps = 256
    n_layers = 3  # doesn't do anything at the moment
    device = "mps"

    for n_neurons in [512,
        1028,
        2048,
        4096,
        4096 + 2048,
        #8192,
        #16384,
    ]:  #  1024, 2048, 4096, 8192, 16384,
        prepare_fn, forward_fn, backward_fn, bench_desc = benchmark()
        print("Benchmarking", bench_desc, "with n_neurons =", n_neurons)
        forward_times, backward_times = benchmark_framework(
            prepare_fn=prepare_fn,
            forward_fn=forward_fn,
            backward_fn=backward_fn,
            benchmark_desc=bench_desc,
            n_neurons=n_neurons,
            n_layers=n_layers,
            n_steps=n_steps,
            batch_size=batch_size,
            device=device,
        )
          # Spyx uses grad which computes the forward and backward pass in one go, so we need to subtract here.
        backward_times = np.array(backward_times) - np.array(forward_times)

        log_result(
            bench_desc,
            n_neurons,
            np.array(forward_times).mean(),
            np.mean(np.array(forward_times)+np.array(backward_times)),
            np.std(np.array(forward_times)+np.array(backward_times))#memory,
        )