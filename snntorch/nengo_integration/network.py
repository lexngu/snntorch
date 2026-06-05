"""
build_nengo_network        — latent encoder → Ensemble topology
build_full_model_network   — full ConvSNN → Node topology for testing

…

All components are stored as attributes on the returned ``nengo.Network``
so callers can access ``net.inp``, ``net.snn_node``, ``net.ens`` / ``net.out_node``,
``net.probe`` directly.

Usage — encoder network
-----------------------
>>> from snntorch.nengo_integration import load_encoder, SNNTorchNode
>>> from snntorch.nengo_integration import build_nengo_network
>>> from dvsgest_conv_net import BEST_PARAMS
>>> import nengo
>>>
>>> encoder = load_encoder("results/run/weights.pth", BEST_PARAMS)
>>> node    = SNNTorchNode(encoder)
>>> net     = build_nengo_network(node)
>>>
>>> with nengo.Simulator(net, dt=0.001) as sim:
...     sim.run(0.1)
... latent_trace = sim.data[net.probe]

Usage — full model network (for tooling exploration)
-----------------------------------------------------
>>> from snntorch.nengo_integration import build_full_model_network
>>> from snntorch.nengo_integration.full_model_node import FullConvSNNNode
>>> from dvsgest_conv_net import ConvSNN, BEST_PARAMS
>>>
>>> model    = ConvSNN(**{k: BEST_PARAMS[k] for k in
...     ["beta","threshold","n_filters_1","n_filters_2","n_filters_3","fc_size","dropout"]})
>>> full_node = FullConvSNNNode(model)
>>> net       = build_full_model_network(full_node)
>>>
>>> with nengo.Simulator(net, dt=0.001) as sim:
...     sim.run(0.1)   # 100 steps = 1 full classification window
... pred_class = sim.data[net.probe][-1].argmax()
"""

from __future__ import annotations

import nengo

from .node import SNNTorchNode
from .full_model_node import FullConvSNNNode


def build_nengo_network(
    node: SNNTorchNode,
    n_neurons: int | None = None,
    synapse: float = 0.005,
    dt: float = 0.001,
) -> nengo.Network:
    """Build a minimal Nengo network around an :class:`SNNTorchNode`.

    Parameters
    ----------
    node : SNNTorchNode
        Configured node (encoder already loaded).
    n_neurons : int or None
        Number of neurons in the latent-representing Ensemble.
        Defaults to ``latent_dim × 20`` — roughly 20 neurons per
        dimension, which gives adequate representation quality.
        Increase for higher fidelity; decrease to reduce compute.
    synapse : float
        Low-pass filter time constant (seconds) on the
        ``snn_node → ens`` connection and the probe.
        The default 5 ms smooths out spike-rate noise without
        adding significant lag.
    dt : float
        Simulation timestep (seconds).  Stored on ``net.dt`` for
        reference; pass the same value to ``nengo.Simulator``.

    Returns
    -------
    nengo.Network
        Network attributes:
        - ``net.inp``      — input Node (size = ``node.size_in``)
        - ``net.snn_node`` — SNNTorchNode wrapped as a nengo.Node
        - ``net.ens``      — Ensemble representing the latent vector
        - ``net.probe``    — Probe on the Ensemble output
        - ``net.dt``       — dt used during construction (for reference)
    """
    latent_dim = node.size_out
    if n_neurons is None:
        n_neurons = latent_dim * 20  # ~20 neurons/dimension

    with nengo.Network(label="SNN-Nengo") as net:
        net.inp = nengo.Node(size_in=node.size_in, label="input")

        net.snn_node = nengo.Node(
            node,
            size_in=node.size_in,
            size_out=node.size_out,
            label="snn_encoder",
        )

        net.ens = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=latent_dim,
            label="latent_ensemble",
        )

        nengo.Connection(net.inp, net.snn_node, synapse=None)
        nengo.Connection(net.snn_node, net.ens, synapse=synapse)
        net.probe = nengo.Probe(net.ens, synapse=synapse)
        net.dt = dt

    return net


def build_full_model_network(
    node: FullConvSNNNode,
    synapse: float = 0.005,
    dt: float = 0.001,
) -> nengo.Network:
    """Build a Nengo network wrapping the complete ConvSNN model.

    Intended for **initial Nengo tooling exploration**: the full SNN
    (including classification head) runs per timestep and the spike-count
    accumulator is directly observable via a Probe.  No neural Ensemble is
    added — the output is a plain Node, so decode weights play no role here.

    Parameters
    ----------
    node : FullConvSNNNode
        Full-model node to wrap.
    synapse : float
        Low-pass synapse on the probe (set to ``None`` for raw spike counts).
    dt : float
        Simulation timestep; pass the same value to ``nengo.Simulator``.

    Returns
    -------
    nengo.Network
        Network attributes:
        - ``net.inp``      — input Node (size = ``node.size_in``)
        - ``net.out_node`` — FullConvSNNNode wrapped as a nengo.Node
        - ``net.probe``    — Probe on the output Node
        - ``net.dt``       — dt reference
    """
    with nengo.Network(label="FullSNN-Nengo") as net:
        net.inp = nengo.Node(size_in=node.size_in, label="input")

        net.out_node = nengo.Node(
            node,
            size_in=node.size_in,
            size_out=node.size_out,
            label="full_snn",
        )

        nengo.Connection(net.inp, net.out_node, synapse=None)
        net.probe = nengo.Probe(net.out_node, synapse=synapse)
        net.dt = dt

    return net

