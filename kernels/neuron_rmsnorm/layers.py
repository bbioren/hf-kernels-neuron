"""NeuronRMSNorm layer — stateless kernel layer for HF Kernel Hub.

This layer replaces the forward() of any module with:
  - self.weight (Tensor): the RMSNorm scale parameter
  - self.variance_epsilon (float): the epsilon for numerical stability

It reads state from the adopting module (the original nn.Module whose
forward is being swapped), which is the standard Kernel Hub pattern for
stateful layers.

Week 2 target: wrap the nki_samples RMSNorm kernel here.
Week 1: starts as a reference PyTorch implementation to validate the
kernelize() plumbing, then swaps in the NKI kernel once on trn2.
"""

import torch
import torch.nn as nn


class NeuronRMSNorm(nn.Module):
    """Stateless RMSNorm layer that reads weight/epsilon from the host module.

    Follows HF Kernel Hub layer requirements:
    - No __init__ (stateless)
    - No class variables except has_backward and can_torch_compile
    - Only defines forward()
    - Type annotations declare expected state from adopting module
    """

    # Kernel Hub metadata
    has_backward: bool = False  # NKI backward not yet implemented
    can_torch_compile: bool = False  # Start eager-only

    # Expected state from adopting module
    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """RMSNorm forward using NKI kernel (or reference impl for scaffolding).

        Args:
            hidden_states: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        # TODO(week2): Replace with NKI kernel call:
        #   from .nki_rmsnorm import nki_rmsnorm_kernel
        #   return nki_rmsnorm_kernel(hidden_states, self.weight, self.variance_epsilon)
        #
        # For now, use a reference implementation that matches Qwen3RMSNorm
        # behavior, so we can validate the kernelize() plumbing end-to-end.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)
