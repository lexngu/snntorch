"""
SNNTorchNode — nengo.Node-compatible wrapper around SNNEncoder.

Each call to ``__call__(t, x)`` corresponds to one Nengo timestep.  The
node advances the SNN by exactly one event-frame and returns the latent
vector as a flat numpy array.

Hidden-state lifecycle
----------------------
* States are reset when ``t`` goes backwards (or on the very first call).
  This handles Nengo simulation restarts transparently.
* States are **not** reset mid-simulation; they accumulate across Nengo
  steps just as they would across timesteps inside the training loop.

Wiring example
--------------
>>> node = SNNTorchNode(encoder, input_shape=(2, 32, 32))
>>> with nengo.Network() as model:
...     snn = nengo.Node(node, size_in=node.size_in, size_out=node.size_out)
"""

from __future__ import annotations

import numpy as np
import torch

from .encoder import SNNEncoder


class SNNTorchNode:
    """Callable Nengo node wrapping an :class:`SNNEncoder`.

    Parameters
    ----------
    encoder : SNNEncoder
        Loaded and configured encoder (owns membrane states).
    input_shape : tuple of int
        Spatial shape of one event frame, e.g. ``(2, 32, 32)`` for
        2-polarity 32×32 DVS frames.  The node expects a *flat* input of
        size ``prod(input_shape)``.
    """

    def __init__(
        self,
        encoder: SNNEncoder,
        input_shape: tuple[int, ...] = (2, 32, 32),
    ) -> None:
        self.encoder = encoder
        self.input_shape = input_shape

        # nengo.Node reads these attributes to set connection dimensions
        self.size_in: int = int(np.prod(input_shape))
        self.size_out: int = encoder.latent_dim

        self._last_t: float | None = None

    # ------------------------------------------------------------------

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """Advance the SNN by one timestep and return the latent vector.

        Parameters
        ----------
        t : float
            Current Nengo simulation time in seconds.
        x : numpy.ndarray, shape ``(size_in,)``
            Flat event frame from Nengo.

        Returns
        -------
        numpy.ndarray, shape ``(size_out,)``
            Latent vector (membrane potential or spikes).
        """
        # Reset membrane states when time resets (new simulation run)
        if self._last_t is None or t < self._last_t:
            self.encoder.reset_state()
        self._last_t = t

        x_tensor = torch.tensor(
            x, dtype=torch.float32
        ).reshape(1, *self.input_shape)

        latent = self.encoder.step(x_tensor)  # (1, latent_dim)
        return latent.cpu().numpy().flatten()  # (latent_dim,)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SNNTorchNode(size_in={self.size_in}, size_out={self.size_out}, "
            f"stop_at={self.encoder.stop_at!r}, "
            f"output_type={self.encoder.output_type!r})"
        )
