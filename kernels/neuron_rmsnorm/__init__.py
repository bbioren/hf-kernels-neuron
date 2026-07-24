"""NKI RMSNorm kernel for Neuron (Trainium/Inferentia).

Ported from the NKI tutorials: awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/general/nki/tutorials/rmsnorm.html
Source: github.com/aws-neuron/nki-samples (rmsnorm_nki_kernels.py)

Single-file kernel following the transformers PR #46754 pattern:
  - NeuronRMSNorm class with forward()
  - `class layers:` namespace for the Hub loader
"""

import math

import torch
import torch.nn as nn

# Conditional NKI import — allows testing off-device
_HAS_NKI = False
try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    _HAS_NKI = True
except ImportError:
    pass


if _HAS_NKI:

    @nki.jit
    def _nki_rmsnorm_kernel(a_tensor, g_tensor, eps_value):
        """NKI RMSNorm kernel: out = (a / RMS(a)) * g

        Ported from nki_samples tutorial. Processes 128 rows at a time
        (partition dimension), applies RMSNorm along the free dimension,
        and multiplies by the weight vector g.

        Args:
            a_tensor: 2D input [rows, hidden_size]
            g_tensor: 1D weight [hidden_size]
            eps_value: epsilon for numerical stability
        """
        out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

        # Tile indices
        ix = nl.arange(128)[:, None]
        iw = nl.arange(1)[:, None]
        iy = nl.arange(a_tensor.shape[1])[None, :]

        num_rows = a_tensor.shape[0]

        # Load weight once (shared across all row tiles)
        g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

        # Process 128 rows per iteration
        for i in nl.affine_range(math.ceil(a_tensor.shape[0] / 128)):
            # Load input tile
            a_tile = nl.load(
                a_tensor[i * 128 + ix, iy], mask=(i * 128 + ix < num_rows)
            )

            # RMS computation: sqrt(mean(x^2) + eps)
            in_square = nl.square(a_tile)
            square_sum = nl.sum(in_square, axis=[1])
            mean = square_sum / a_tensor.shape[1]
            mean_plus_eps = nl.add(mean, eps_value)
            rms_reciprocal = nl.rsqrt(mean_plus_eps)

            # Normalize: x / RMS(x)
            out_tile = nl.multiply(a_tile, rms_reciprocal)

            # Multiply by weight (broadcast along partition axis)
            g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))
            out_tile[...] = nl.multiply(
                out_tile, g_bcast, mask=(i * 128 + ix < num_rows)
            )

            # Store result
            nl.store(
                out_tensor[i * 128 + ix, iy],
                value=out_tile,
                mask=(i * 128 + ix < num_rows),
            )

        return out_tensor


def _pytorch_rmsnorm(hidden_states, weight, eps):
    """Reference PyTorch RMSNorm — used as fallback off-device."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


class NeuronRMSNorm(nn.Module):
    """Stateless RMSNorm kernel layer for the HF Kernel Hub.

    Reads `self.weight` and `self.variance_epsilon` from the adopting module
    (the original Qwen3RMSNorm / LlamaRMSNorm whose forward() is being replaced).

    On Neuron hardware: calls the @nki.jit kernel
    Off Neuron: falls back to a PyTorch reference implementation
    """

    has_backward: bool = False
    can_torch_compile: bool = False

    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape

        # Flatten to 2D: [batch * seq_len, hidden_size]
        hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])

        if _HAS_NKI and hidden_states.device.type != "cpu":
            # Run NKI kernel on NeuronCores
            output_2d = _nki_rmsnorm_kernel(
                hidden_2d, self.weight, self.variance_epsilon
            )
        else:
            # PyTorch fallback (CPU or no NKI available)
            output_2d = _pytorch_rmsnorm(
                hidden_2d, self.weight, self.variance_epsilon
            )

        return output_2d.reshape(original_shape)


class layers:
    NeuronRMSNorm = NeuronRMSNorm
