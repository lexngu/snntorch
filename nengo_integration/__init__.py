"""
snntorch.nengo_integration
==========================

Wrap a trained ConvSNN model as a Nengo-compatible encoder or full-model node.

Quick start — flexible latent encoder
--------------------------------------
>>> from snntorch.nengo_integration import load_encoder, SNNTorchNode, build_nengo_network
>>> from dvsgest_conv_net import BEST_PARAMS
>>> import nengo
>>>
>>> # Stop at any named layer — see SNNEncoder docstring for the full table
>>> encoder = load_encoder(
...     weights_path="results/my_run/weights.pth",
...     params=BEST_PARAMS,
...     stop_at="fc1",        # raw fc1 pre-activation (default)
...     output_type="auto",   # membrane for LIF layers, activation otherwise
...     device="cpu",
... )
>>> print(encoder)
SNNEncoder(stop_at='fc1', output_type='auto', latent_dim=256, device=cpu)
>>>
>>> node = SNNTorchNode(encoder, input_shape=(2, 32, 32))
>>> net  = build_nengo_network(node)
>>>
>>> with nengo.Simulator(net, dt=0.001) as sim:
...     sim.run(0.1)
... latent_trace = sim.data[net.probe]  # (T, latent_dim)

Quick start — full model node (tooling exploration)
----------------------------------------------------
>>> from snntorch.nengo_integration import build_full_model_network
>>> from snntorch.nengo_integration.full_model_node import FullConvSNNNode
>>> from dvsgest_conv_net import ConvSNN, BEST_PARAMS
>>>
>>> model     = ConvSNN(**{k: BEST_PARAMS[k] for k in
...     ["beta","threshold","n_filters_1","n_filters_2","n_filters_3","fc_size","dropout"]})
>>> full_node = FullConvSNNNode(model, window_size=100)
>>> net       = build_full_model_network(full_node)
>>>
>>> with nengo.Simulator(net, dt=0.001) as sim:
...     sim.run(0.1)   # 100 steps = 1 full classification window
... pred_class = sim.data[net.probe][-1].argmax()

Common stop_at values for ConvSNN
----------------------------------
  "lif1"  after 1st conv block  →  32×16×16 = 8192-dim (flat)
  "lif2"  after 2nd conv block  →  64×8×8   = 4096-dim (flat)
  "lif3"  after 3rd conv block  →  128×4×4  = 2048-dim (flat)
  "fc1"   raw fc1 activation    →  fc_size  (256 with BEST_PARAMS)  ← default
  "lif4"  after fc1 LIF         →  fc_size
  "lif5"  full model output     →  N_CLASSES (11)
"""

from .encoder import SNNEncoder, load_encoder
from .node import SNNTorchNode
from .network import build_nengo_network, build_full_model_network
from .full_model_node import FullConvSNNNode

__all__ = [
    "SNNEncoder",
    "load_encoder",
    "SNNTorchNode",
    "build_nengo_network",
    "build_full_model_network",
    "FullConvSNNNode",
]
