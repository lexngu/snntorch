"""
FullConvSNNNode — Nengo Node that runs the complete ConvSNN per timestep.

Purpose
-------
This node is intended for **initial Nengo tooling exploration**, not for
production inference.  It runs all model layers (including the classification
head ``fc2 → lif5``) at every Nengo timestep and accumulates the output
spike counts, mimicking the training-time evaluation loop.

Topology
--------
Each Nengo step advances the SNN by one event-frame and adds ``spk5``
(11-dim binary) to a running accumulator ``spk_acc``.  The node outputs
the **current running accumulator** so you can observe it evolve in real
time via a Probe.

After every ``window_size`` steps the accumulator resets automatically,
starting a fresh classification window.  The membrane states also reset at
that point and whenever the simulation restarts (``t`` goes backwards).

Classification
--------------
At any time, ``argmax(spk_acc)`` gives the predicted gesture class.
With ``window_size=100`` (the training default) the spike-count logits at
step 100 are directly comparable to what the trained model produces.

Example
-------
>>> node = FullConvSNNNode(model, window_size=100)
>>> net  = build_full_model_network(node)
>>> with nengo.Simulator(net, dt=0.001) as sim:
...     sim.run(0.1)   # 100 steps = one full classification window
... # net.probe.shape == (100, N_CLASSES)
... pred_class = sim.data[net.probe][-1].argmax()
"""

from __future__ import annotations

import sys
import os

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvsgest_conv_net import ConvSNN, N_CLASSES

try:
    import snntorch as snn
    _LIF_TYPE = snn.Leaky
except ImportError:
    _LIF_TYPE = type(None)


def _is_lif(module: nn.Module) -> bool:
    return isinstance(module, _LIF_TYPE)


class FullConvSNNNode:
    """Nengo-compatible callable that runs the full ConvSNN every step.

    Parameters
    ----------
    model : ConvSNN
        Instantiated ConvSNN (random init is fine for pipeline testing).
        Set to eval mode on construction.
    window_size : int
        Number of event frames per classification window (default: 100,
        matching ``N_TIME_BINS`` used during training).  After this many
        steps the spike accumulator and membrane states are reset.
    input_shape : tuple of int
        Spatial shape of one event frame, e.g. ``(2, 32, 32)``.
    device : str or torch.device
    """

    def __init__(
        self,
        model: ConvSNN,
        window_size: int = 100,
        input_shape: tuple[int, ...] = (2, 32, 32),
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.window_size = window_size
        self.input_shape = input_shape

        self.size_in: int = int(np.prod(input_shape))
        self.size_out: int = N_CLASSES

        self.model = model.to(self.device).eval()
        self._layers: list[tuple[str, nn.Module]] = list(
            model.named_children()
        )
        self._lif_names: list[str] = [
            n for n, m in self._layers if _is_lif(m)
        ]

        self._state: dict[str, torch.Tensor] = {}
        self._spk_acc: torch.Tensor | None = None
        self._step_count: int = 0
        self._last_t: float | None = None

        self._reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset membrane states and spike accumulator."""
        z = torch.zeros(1, device=self.device)
        self._state = {name: z for name in self._lif_names}
        self._spk_acc = torch.zeros(1, N_CLASSES, device=self.device)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Nengo callable interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """Advance the full SNN by one timestep.

        Parameters
        ----------
        t : float
            Current Nengo simulation time (seconds).
        x : numpy.ndarray, shape ``(size_in,)``
            Flat event frame.

        Returns
        -------
        numpy.ndarray, shape ``(N_CLASSES,)``
            Running spike-count accumulator.  ``argmax()`` → predicted class.
        """
        # Simulation restart detection
        if self._last_t is None or t < self._last_t:
            self._reset()
        self._last_t = t

        # Window boundary reset
        if self._step_count > 0 and self._step_count % self.window_size == 0:
            self._reset()

        # Reshape flat numpy input → tensor (1, C, H, W)
        frame = torch.tensor(x, dtype=torch.float32).reshape(
            1, *self.input_shape
        ).to(self.device)

        # Forward all layers
        act = frame
        last_spk = None
        for name, module in self._layers:
            if isinstance(module, nn.Linear) and act.dim() > 2:
                act = act.flatten(1)

            if _is_lif(module):
                spk, mem = module(act, self._state[name])
                self._state[name] = mem
                act = spk
                last_spk = spk
            else:
                act = module(act)

        # Accumulate output spikes (last LIF = lif5, shape (1, N_CLASSES))
        if last_spk is not None:
            self._spk_acc += last_spk

        self._step_count += 1
        return self._spk_acc.cpu().numpy().flatten()  # (N_CLASSES,)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FullConvSNNNode(size_in={self.size_in}, "
            f"size_out={self.size_out}, "
            f"window_size={self.window_size}, "
            f"device={self.device})"
        )
