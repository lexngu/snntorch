"""
SNNEncoder — flexible single-step latent extractor for ConvSNN.

Wraps a trained ConvSNN and advances the SNN by one event-frame at a
time, keeping all membrane states persistent between calls.

Extraction is controlled by two parameters:

``stop_at`` (str)
    Name of the layer (from ``model.named_children()``) at which to stop
    and extract the latent.  Examples for the default ConvSNN architecture:

    +-----------+----------------------------+------------+
    | stop_at   | Description                | latent_dim |
    +===========+============================+============+
    | ``"lif1"``| after 1st conv LIF         | 32×16×16   |
    | ``"lif2"``| after 2nd conv LIF         | 64×8×8     |
    | ``"lif3"``| after 3rd conv LIF (flat)  | 2048       |
    | ``"fc1"`` | raw fc1 pre-activation *(default)* | fc_size |
    | ``"lif4"``| after fc1 LIF              | fc_size    |
    | ``"lif5"``| after output LIF (full)    | N_CLASSES  |
    +-----------+----------------------------+------------+

``output_type`` (str)
    What to return from the stop layer:

    * ``"auto"`` *(default)* — ``"membrane"`` for ``snn.Leaky`` layers,
      ``"activation"`` for all other layers.
    * ``"membrane"`` — membrane potential (continuous float, LIF only).
    * ``"spikes"``   — binary spikes ({0,1}, LIF only).
    * ``"activation"`` — raw output of any layer (float).

Usage
-----
>>> encoder = SNNEncoder(model, stop_at="fc1")          # default: fc1 pre-activation
>>> encoder = SNNEncoder(model, stop_at="lif3")         # after 3rd conv block
>>> encoder = SNNEncoder(model, stop_at="lif4", output_type="membrane")
>>> encoder.reset_state()
>>> for frame in event_frames:        # frame: (1, 2, H, W) tensor
...     latent = encoder.step(frame)  # (1, latent_dim) tensor
"""

from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn

# Allow importing from the parent snntorch directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvsgest_conv_net import ConvSNN

try:
    import snntorch as snn
    _LIF_TYPE = snn.Leaky
except ImportError:
    # Fallback sentinel — won't match any module, so LIF detection is disabled
    _LIF_TYPE = type(None)

_VALID_OUTPUT_TYPES = ("auto", "membrane", "spikes", "activation")


def _is_lif(module: nn.Module) -> bool:
    return isinstance(module, _LIF_TYPE)


class SNNEncoder:
    """Generic single-step latent extractor for any n×CONV + FC-lif-FC-lif model.

    Iterates ``model.named_children()`` in order, applies each layer, and
    stops at ``stop_at``.  Membrane states for all ``snn.Leaky`` layers are
    tracked automatically — no manual state management required.

    Parameters
    ----------
    model : nn.Module
        The SNN model (e.g. ConvSNN).  Set to eval mode on construction.
    stop_at : str
        Name of the layer at which to stop and extract the latent.
        Must match a key in ``dict(model.named_children())``.
    output_type : {"auto", "membrane", "spikes", "activation"}
        What to return from the stop layer (see module docstring).
    device : str or torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        stop_at: str = "fc1",
        output_type: str = "auto",
        device: str | torch.device = "cpu",
    ) -> None:
        if output_type not in _VALID_OUTPUT_TYPES:
            raise ValueError(
                f"output_type must be one of {_VALID_OUTPUT_TYPES}, "
                f"got {output_type!r}"
            )

        self.device = torch.device(device)
        self.stop_at = stop_at
        self.output_type = output_type

        # Ordered list of (name, module) up to and including stop_at
        all_children = list(model.named_children())
        names = [n for n, _ in all_children]
        if stop_at not in names:
            raise ValueError(
                f"stop_at={stop_at!r} not found in model.named_children(). "
                f"Available: {names}"
            )
        stop_idx = names.index(stop_at)
        self._layers: list[tuple[str, nn.Module]] = all_children[: stop_idx + 1]

        # Identify stateful (snn.Leaky) layers among the active layers
        self._lif_names: list[str] = [
            n for n, m in self._layers if _is_lif(m)
        ]

        self.model = model.to(self.device).eval()
        self._state: dict[str, torch.Tensor] = {}
        self.reset_state()

        # Infer latent_dim by a single dry forward pass
        self.latent_dim: int = self._infer_latent_dim(model)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Zero all membrane states (call at the start of each new sequence)."""
        z = torch.zeros(1, device=self.device)
        self._state = {name: z for name in self._lif_names}

    # ------------------------------------------------------------------
    # Single-step inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, x_frame: torch.Tensor) -> torch.Tensor:
        """Advance the SNN by one event-frame timestep.

        Parameters
        ----------
        x_frame : torch.Tensor, shape ``(B, C, H, W)``
            One event frame (will be moved to ``self.device`` and cast to
            float32 if needed).

        Returns
        -------
        latent : torch.Tensor, shape ``(B, latent_dim)``
            The extracted representation at ``stop_at``.
        """
        x = x_frame.to(self.device).float()
        output = None

        for name, module in self._layers:
            # Auto-flatten when a Linear layer receives a multi-dim tensor
            if isinstance(module, nn.Linear) and x.dim() > 2:
                x = x.flatten(1)

            if _is_lif(module):
                spk, mem = module(x, self._state[name])
                self._state[name] = mem
                x = spk          # pass spikes to the next layer
                output = (spk, mem)
            else:
                x = module(x)
                output = x

        return self._extract(output, self._layers[-1][1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, raw_output, stop_module: nn.Module) -> torch.Tensor:
        """Apply output_type logic to the raw layer output."""
        effective = self.output_type

        if effective == "auto":
            effective = "membrane" if _is_lif(stop_module) else "activation"

        if effective == "activation":
            # raw_output may be a plain tensor (non-LIF) or (spk, mem) tuple
            if isinstance(raw_output, tuple):
                return raw_output[0]   # spk from a LIF treated as activation
            return raw_output

        if not _is_lif(stop_module):
            raise ValueError(
                f"output_type={self.output_type!r} requires a snn.Leaky stop layer, "
                f"but stop_at={self.stop_at!r} is {type(stop_module).__name__}."
            )
        spk, mem = raw_output
        return mem if effective == "membrane" else spk

    @torch.no_grad()
    def _infer_latent_dim(self, model: nn.Module) -> int:
        """Run one dummy forward to discover the latent dimension."""
        # Determine input spatial shape from model internals if possible
        try:
            first_conv = next(
                m for _, m in model.named_children()
                if isinstance(m, nn.Conv2d)
            )
            in_ch = first_conv.in_channels
            # Use 32×32 as a safe default; works for any square sensor
            dummy = torch.zeros(1, in_ch, 32, 32, device=self.device)
        except StopIteration:
            dummy = torch.zeros(1, 2, 32, 32, device=self.device)

        saved_state = {k: v.clone() for k, v in self._state.items()}
        latent = self.step(dummy)
        # Restore state so the dry run has no side effects
        self._state = saved_state
        return int(latent.shape[-1])

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SNNEncoder(stop_at={self.stop_at!r}, "
            f"output_type={self.output_type!r}, "
            f"latent_dim={self.latent_dim}, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_encoder(
    weights_path: str,
    params: dict,
    stop_at: str = "fc1",
    output_type: str = "auto",
    device: str | torch.device = "cpu",
) -> SNNEncoder:
    """Load a trained ConvSNN from disk and wrap it as an SNNEncoder.

    Parameters
    ----------
    weights_path : str
        Path to ``weights.pth`` saved by ``save_run()``.
    params : dict
        Hyperparameter dict used when training (e.g. ``BEST_PARAMS``).
        Must contain ``beta``, ``threshold``, ``n_filters_1``,
        ``n_filters_2``, ``n_filters_3``, ``fc_size``, ``dropout``.
    stop_at : str
        Layer name at which to extract the latent (default: ``"fc1"``).
    output_type : str
        Output extraction mode (default: ``"auto"``).
    device : str or torch.device

    Returns
    -------
    SNNEncoder
    """
    model = ConvSNN(
        beta        = params["beta"],
        threshold   = params["threshold"],
        n_filters_1 = params["n_filters_1"],
        n_filters_2 = params["n_filters_2"],
        n_filters_3 = params["n_filters_3"],
        fc_size     = params["fc_size"],
        dropout     = params.get("dropout", 0.4),
    )
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return SNNEncoder(model, stop_at=stop_at, output_type=output_type, device=device)
