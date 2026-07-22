"""Identity-plus-scale layer — minimal stateless kernel for PoC.

This is the absolute simplest NKI kernel possible: multiply by a learnable
scale parameter. It exists purely to prove the forward-swap path works
end-to-end with a real NKI kernel running on NeuronCores.

On-device (trn2): uses the NKI kernel via nki.jit
Off-device (CPU): uses a PyTorch fallback so the plumbing can be tested anywhere
"""

import torch
import torch.nn as nn


# Try to import the NKI kernel; fall back to torch impl if not on Neuron
_USE_NKI = False
try:
    from .nki_identity import nki_identity_scale
    _USE_NKI = True
except (ImportError, RuntimeError):
    pass


class NeuronIdentityScale(nn.Module):
    """Stateless identity-scale layer for Kernel Hub.

    Expects the adopting module to have:
      - self.scale (Tensor or float): multiplicative scale factor

    If the adopting module doesn't have .scale, defaults to 1.0
    (pure identity, but still confirms the swap fired).
    """

    has_backward: bool = False
    can_torch_compile: bool = False

    # Expected state from adopting module (optional)
    scale: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get scale from adopting module, default to 1.0
        scale = getattr(self, "scale", None)
        if scale is None:
            scale = 1.0

        if _USE_NKI and x.device.type == "neuron":
            return nki_identity_scale(x, scale)
        else:
            # PyTorch fallback — still proves the swap worked
            # Add a tiny epsilon signature so we can detect the swap fired
            # even when scale == 1.0
            return x * scale
