from utils import timeit, benchmark_framework, log_result

import argparse
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def rockpool_torch():
    import torch

    #torch.cuda.set_per_process_memory_fraction(0.7, device=None)
    from rockpool.nn.modules import LIFTorch, LinearTorch
    from rockpool.nn.combinators import Sequential
    import rockpool

    benchmark_title = f"Rockpool<br>v{rockpool.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        # model = Sequential(
        #     LinearTorch(shape=(n_neurons, n_neurons)),
        #     LIFTorch(n_neurons),
        # ).to(device)
        model = Sequential(
            LinearTorch(shape=(n_neurons, n_neurons)),
            LIFTorch(n_neurons,spiking_output=True,max_spikes_per_dt=True),
            LinearTorch(shape=(n_neurons, n_neurons)),
            LIFTorch(n_neurons,spiking_output=True,max_spikes_per_dt=True),
            LinearTorch(shape=(n_neurons, n_neurons)),
            LIFTorch(n_neurons,spiking_output=True,max_spikes_per_dt=True),
        ).to(device)
        input_static = torch.randn(batch_size, n_steps, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        output = model(input_static)[0]
        bench_dict["output"] = output
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

# def rockpool_jax():
#     import jax
#     import jax.numpy as jnp

#     #torch.cuda.set_per_process_memory_fraction(0.7, device=None)
#     from rockpool.nn.modules import LIFJax, LinearJax
#     from rockpool.nn.combinators import Sequential
#     import rockpool

#     benchmark_title = f"Rockpool_jax<br>v{rockpool.__version__}"

#     key = jax.random.PRNGKey(0)

#     def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
#         model = Sequential(
#             LinearJax(shape=(n_neurons, n_neurons)),
#             LIFJax(n_neurons,spiking_output=True,max_spikes_per_dt=True),
#             LinearJax(shape=(n_neurons, n_neurons)),
#             LIFJax(n_neurons,spiking_output=True,max_spikes_per_dt=True),
#             LinearJax(shape=(n_neurons, n_neurons)),
#             LIFJax(n_neurons,spiking_output=True,max_spikes_per_dt=True),
#         )
#         input_static = jax.random.normal(key,shape=(n_steps, batch_size, n_neurons), dtype=jnp.float32)
#         #model(input_static)
#         return dict(model=model, input=input_static, n_neurons=n_neurons)

#     def forward_fn(bench_dict):
#         model, input_static = bench_dict["model"], bench_dict["input"]
#         output = model(input_static)[0]
#         bench_dict["output"] = output
#         return bench_dict

#     def backward_fn(bench_dict):
#         output = bench_dict["output"]
#         loss = output.sum()
#         loss.backward(retain_graph=True)

#     return prepare_fn, forward_fn, backward_fn, benchmark_title

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
            LIFJax(n_neurons,spiking_output=True,max_spikes_per_dt=True),
            LinearJax(shape=(n_neurons, n_neurons)),
            LIFJax(n_neurons,spiking_output=True,max_spikes_per_dt=True),
            LinearJax(shape=(n_neurons, n_neurons)),
            LIFJax(n_neurons,spiking_output=True,max_spikes_per_dt=True),
        )

        input_static = jax.random.normal(key,shape=(n_steps, batch_size, n_neurons), dtype=jnp.float32)
        #input_static = jnp.ones(shape=(n_steps, batch_size, n_neurons), dtype=jnp.float32)

        
        # Since there's nothing stochastic about the network, we can avoid using an RNG as a param!
        #SNN = model#sl.RNN(sl.connect([nn.Dense(n_neurons),sl.LIF(2.),nn.Dense(n_neurons),sl.LIF(2.),nn.Dense(n_neurons),sl.LIF(2.),]))#Model()
        params = SNN.parameters()

        @jax.jit
        def net_eval(params,events):
            mdl = SNN.reset_state()

            # - Apply the parameters
            mdl = mdl.set_attributes(params)

            # - Evolve the network
            output, _, _ = mdl(events, record=True)
            #traces, V_f = readout
            #return traces.sum()
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


def rockpool_exodus():
    import torch

    #torch.cuda.set_per_process_memory_fraction(0.7, device=None)
    from rockpool.nn.modules import LIFExodus, LinearTorch
    from rockpool.nn.combinators import Sequential
    import rockpool

    benchmark_title = f"Rockpool EXODUS<br>v{rockpool.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        # model = Sequential(
        #     LinearTorch(shape=(n_neurons, n_neurons)),
        #     LIFExodus(n_neurons),
        # ).to(device)
        model = Sequential(
            LinearTorch(shape=(n_neurons, n_neurons)),
            LIFExodus(n_neurons),
            LinearTorch(shape=(n_neurons, n_neurons)),
            LIFExodus(n_neurons),
            LinearTorch(shape=(n_neurons, n_neurons)),
            LIFExodus(n_neurons),
        ).to(device)
        input_static = torch.randn(batch_size, n_steps, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        output = model(input_static)[0]
        bench_dict["output"] = output
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def sinabs():
    import torch
    from torch import nn

    #torch.cuda.set_per_process_memory_fraction(0.7, device=None)
    from sinabs.layers import LIF
    import sinabs

    benchmark_title = f"Sinabs<br>v{sinabs.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        model = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            LIF(tau_mem=torch.tensor(10.0)),
        ).to(device)
        input_static = torch.randn(batch_size, n_steps, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        sinabs.reset_states(model)
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def sinabs_exodus():
    import torch
    from torch import nn

    #torch.cuda.set_per_process_memory_fraction(0.7, device=None)
    from sinabs.exodus.layers import LIF
    import sinabs

    benchmark_title = f"Sinabs EXODUS<br>v{sinabs.exodus.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        # model = nn.Sequential(
        #     nn.Linear(n_neurons, n_neurons),
        #     LIF(tau_mem=torch.tensor(10.0)),
        # ).to(device)
        model = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            LIF(tau_mem=torch.tensor(10.0)),
            nn.Linear(n_neurons, n_neurons),
            LIF(tau_mem=torch.tensor(10.0)),
            nn.Linear(n_neurons, n_neurons),
            LIF(tau_mem=torch.tensor(10.0)),
        ).to(device)
        input_static = torch.randn(batch_size, n_steps, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        sinabs.reset_states(model)
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def norse():
    import torch
    from torch import nn

    #torch.cuda.set_per_process_memory_fraction(0.7, device=None)
    from norse.torch.module.lif import LIF
    from norse.torch import SequentialState
    import norse

    benchmark_title = f"Norse<br>v{norse.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        # model = SequentialState(
        #     nn.Linear(n_neurons, n_neurons),
        #     LIF(),
        # )
        model = SequentialState(
            nn.Linear(n_neurons, n_neurons),
            LIF(),
            nn.Linear(n_neurons, n_neurons),
            LIF(),
            nn.Linear(n_neurons, n_neurons),
            LIF(),
        )
        model = torch.compile(model, mode="max-autotune")
        model = model.to(device)
        input_static = torch.randn(n_steps, batch_size, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        # output.sum().backward() # JIT compile everything
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)[0]
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def snntorch():
    import torch
    from torch import nn

    #torch.cuda.set_per_process_memory_fraction(0.7, device=None)
    import snntorch

    benchmark_title = f"snnTorch<br>v{snntorch.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        class Model(nn.Module):
            def __init__(self, beta: float = 0.95):
                super().__init__()
                self.fc = nn.Linear(n_neurons, n_neurons)
                self.lif = snntorch.Leaky(beta=beta)
                self.mem = self.lif.init_leaky()

                self.fc2 = nn.Linear(n_neurons, n_neurons)
                self.lif2 = snntorch.Leaky(beta=beta)
                self.mem2 = self.lif.init_leaky()

                self.fc3 = nn.Linear(n_neurons, n_neurons)
                self.lif3 = snntorch.Leaky(beta=beta)
                self.mem3 = self.lif.init_leaky()

            def forward(self, x):
                output = []
                mem = self.mem
                mem2 = self.mem2
                mem3 = self.mem3
                for inp in x:
                    cur = self.fc(inp)
                    spk, mem = self.lif(cur, mem)
                    cur = self.fc2(spk)
                    spk, mem2 = self.lif2(cur, mem2)
                    cur = self.fc3(spk)
                    spk, mem3 = self.lif3(cur, mem3)
                    output.append(spk)
                return torch.stack(output)

        model = Model()
        #model = torch.compile(model, mode="max-autotune")
        model = model.to(device)
        input_static = torch.randn(n_steps, batch_size, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


# mix of https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/basic_concept.html#step-mode
# and https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/rsnn_sequential_fmnist.py
def spikingjelly():
    import torch
    from torch import nn

    #torch.cuda.set_per_process_memory_fraction(0.7, device=None)
    from spikingjelly.activation_based import neuron, surrogate, functional, layer

    benchmark_title = f"SpikingJelly PyTorch<br>v0.0.0.0.15"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        class Model(nn.Module):
            def __init__(self, tau=5.0):
                super().__init__()
                # self.model = nn.Sequential(
                #     layer.Linear(n_neurons, n_neurons),
                #     neuron.LIFNode(
                #         tau=tau, surrogate_function=surrogate.ATan(), step_mode="m"
                #     ),
                # )
                self.model = nn.Sequential(
                    layer.Linear(n_neurons, n_neurons),
                    neuron.LIFNode(
                        tau=tau, surrogate_function=surrogate.ATan(), step_mode="m"
                    ),
                    layer.Linear(n_neurons, n_neurons),
                    neuron.LIFNode(
                        tau=tau, surrogate_function=surrogate.ATan(), step_mode="m"
                    ),
                    layer.Linear(n_neurons, n_neurons),
                    neuron.LIFNode(
                        tau=tau, surrogate_function=surrogate.ATan(), step_mode="m"
                    ),
                )

            def forward(self, x):
                functional.reset_net(self.model)
                return self.model(x)

        model = Model().to(device)
        input_static = torch.randn(n_steps, batch_size, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def spikingjelly_cupy():
    import torch
    from torch import nn

    #torch.cuda.set_per_process_memory_fraction(0.7, device=None)
    from spikingjelly.activation_based import neuron, surrogate, functional, layer

    benchmark_title = f"SpikingJelly CuPy<br>v0.0.0.0.15"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        class Model(nn.Module):
            def __init__(self, tau=5.0):
                super().__init__()
                # self.model = nn.Sequential(
                #     layer.Linear(n_neurons, n_neurons),
                #     neuron.LIFNode(
                #         tau=tau, surrogate_function=surrogate.ATan(), step_mode="m"
                #     ),
                # )
                self.model = nn.Sequential(
                    layer.Linear(n_neurons, n_neurons),
                    neuron.LIFNode(
                        tau=tau, surrogate_function=surrogate.ATan(), step_mode="m"
                    ),
                    layer.Linear(n_neurons, n_neurons),
                    neuron.LIFNode(
                        tau=tau, surrogate_function=surrogate.ATan(), step_mode="m"
                    ),
                    layer.Linear(n_neurons, n_neurons),
                    neuron.LIFNode(
                        tau=tau, surrogate_function=surrogate.ATan(), step_mode="m"
                    ),
                )
                functional.set_backend(self.model, backend="cupy")

            def forward(self, x):
                functional.reset_net(self.model)
                return self.model(x)

        model = Model().to(device)
        input_static = torch.randn(n_steps, batch_size, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def lava():
    import torch

    torch.cuda.set_per_process_memory_fraction(0.7, device=None)
    import lava.lib.dl.slayer as slayer

    benchmark_title = f"Lava DL<br>v0.4.0.dev0"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        neuron_params = {
            "threshold": 0.1,
            "current_decay": 1,
            "voltage_decay": 0.1,
            "requires_grad": True,
        }
        # slayer.block automatically add quantization.
        # They can be disabled by setting pre_hook_fx=None
        model = slayer.block.cuba.Dense(
            neuron_params, n_neurons, n_neurons, pre_hook_fx=None
        ).to(device)
        input_static = torch.randn(batch_size, n_neurons, n_steps).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def spyx_full():
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"
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
            # print(x.shape)
            #x = x.transpose(1,0,2)
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

        #input_static = jnp.ones(shape=(n_steps, batch_size, n_neurons), dtype=jnp.uint8)
        key = jax.random.PRNGKey(0)
        input_static = jax.random.normal(key,shape=(n_steps, batch_size, n_neurons), dtype=jnp.float32)

        
        # Since there's nothing stochastic about the network, we can avoid using an RNG as a param!
        SNN = hk.without_apply_rng(hk.transform(Model))
        params = SNN.init(rng=key, x=input_static)

        @jax.jit
        def net_eval(weights, events):
            readout = SNN.apply(weights, events)
            #traces, V_f = readout
            #return traces.sum()
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


def spyx_half():
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"
    import spyx
    import spyx.nn as snn
    import jax
    import jax.numpy as jnp
    import jmp
    import haiku as hk

    policy = jmp.get_policy("half")

    hk.mixed_precision.set_policy(hk.Linear, policy)
    hk.mixed_precision.set_policy(snn.LIF, policy)

    benchmark_title = f"Spyx half-precision v{spyx.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        def Model(x):
            x = hk.BatchApply(hk.Linear(n_neurons, with_bias=False))(x)

            core = hk.DeepRNN(
                [
                    snn.LIF((n_neurons,), activation=spyx.axn.Axon(spyx.axn.arctan())),
                ]
            )

            # static unroll for maximum performance
            spikes, V = hk.dynamic_unroll(
                core, x, core.initial_state(x.shape[0]), time_major=False, unroll=5
            )

            return spikes, V

        input_static = jnp.ones(shape=(n_steps, batch_size, n_neurons), dtype=jnp.uint8)

        key = jax.random.PRNGKey(0)
        # Since there's nothing stochastic about the network, we can avoid using an RNG as a param!
        SNN = hk.without_apply_rng(hk.transform(Model))
        params = SNN.init(rng=key, x=input_static)

        @jax.jit
        def net_eval(weights, events):
            readout = SNN.apply(weights, events)
            traces, V_f = readout
            print(V_f)
            return traces.sum()

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
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"
    import jax
    import jax.numpy as jnp
    import slax as sl
    import flax.linen as nn

    key = jax.random.PRNGKey(0)

    benchmark_title = f"slax full-precision v{'0.0.1'}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        class Model(nn.Module):
            @nn.compact
            def __call__(self,x):
                x = nn.Dense(n_neurons)(x)
                x = sl.RNN(sl.LIF(2.,spike_fn=sl.atan()))(x)
                x = nn.Dense(n_neurons)(x)
                x = sl.RNN(sl.LIF(2.,spike_fn=sl.atan()))(x)
                x = nn.Dense(n_neurons)(x)
                x = sl.RNN(sl.LIF(2.,spike_fn=sl.atan()))(x)

                return x

        input_static = jax.random.normal(key,shape=(n_steps, batch_size, n_neurons), dtype=jnp.float32)
        #input_static = jnp.ones(shape=(n_steps, batch_size, n_neurons), dtype=jnp.float32)

        
        # Since there's nothing stochastic about the network, we can avoid using an RNG as a param!
        SNN = Model()#sl.RNN(sl.connect([nn.Dense(n_neurons),sl.LIF(2.),nn.Dense(n_neurons),sl.LIF(2.),nn.Dense(n_neurons),sl.LIF(2.),]))#Model()
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

    return prepare_fn, forward_fn, backward_fn, benchmark_title


benchmarks = {
    "rockpool_torch": rockpool_torch,
    "rockpool_exodus": rockpool_exodus,
    "sinabs": sinabs,
    "sinabs_exodus": sinabs_exodus,
    "norse": norse,
    "snntorch": snntorch,
    "spikingjelly": spikingjelly,
    "spikingjelly_cupy": spikingjelly_cupy,
    "lava": lava,
    "spyx_full": spyx_full,
    "spyx_half": spyx_half,
    "slax_full": slax_full,
    "rockpool_jax": rockpool_jax,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=benchmarks.keys())
    parser.add_argument("batch_size")
    args = parser.parse_args()

    benchmark = benchmarks[args.benchmark]

    batch_size = int(args.batch_size)
    n_steps = 500
    n_layers = 3  # doesn't do anything at the moment
    device = "mps"

    for n_neurons in [
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
        if (bench_desc[:4] == "slax") or (bench_desc[:4] == "spyx") or (bench_desc[:4] == "Spyx"):  # Spyx uses grad which computes the forward and backward pass in one go, so we need to subtract here.
            backward_times = (np.array(backward_times).mean() - np.array(forward_times).mean())
            memory = "nan"
        else:
            import torch
            memory = torch.cuda.max_memory_allocated()

        log_result(
            bench_desc,
            n_neurons,
            np.array(forward_times).mean(),
            np.array(backward_times).mean(),
            np.array(forward_times).mean()+np.array(backward_times).mean()#memory,
        )