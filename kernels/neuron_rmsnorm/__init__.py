# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from the NKI RMSNorm tutorial (aws-neuron/nki-samples), which is MIT-0.
# MIT-0 requires no attribution; provenance is recorded here as good practice and
# because the *reason* it is tutorial-derived is itself a finding: nki-library has
# no standalone RMSNorm, only rmsnorm_quant.py, which always quantises.
"""NKI RMSNorm kernel for Neuron (Trainium/Inferentia).

Ported from the NKI tutorials: awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/general/nki/tutorials/rmsnorm.html
Source: github.com/aws-neuron/nki-samples (rmsnorm_nki_kernels.py)

Single-file kernel following the transformers PR #46754 pattern:
  - NeuronRMSNorm class with forward()
  - `class layers:` namespace for the Hub loader
"""

import torch
import torch.nn as nn

# NKI 0.5.0 (top-level `nki`) is the going-forward surface. The older API bundled inside
# neuronx-cc (`neuronxcc.nki`) is kept only as a fallback for environments that lack the
# standalone package.
#
# This kernel was originally written against the older API using `nl.arange` index tensors
# plus `mask=` for ragged tails. Both are gone in 0.5.0: `nl.arange` was removed in favour of
# `nl.ds` slicing, and `nl.load`/`nl.store` no longer accept `mask`. See Finding #14.
_HAS_NKI = False
try:
    import nki
    import nki.language as nl

    _HAS_NKI = True
except ImportError:
    try:
        import neuronxcc.nki as nki
        import neuronxcc.nki.language as nl

        _HAS_NKI = True
    except ImportError:
        pass


# Partition-dimension tile height.
PARTITION_MAX = 128


if _HAS_NKI:

    @nki.jit
    def _nki_rmsnorm_kernel(a_tensor, g_tensor, eps_value):
        """RMSNorm: out = (a / RMS(a)) * g

        Tiles over rows in the partition dimension and normalizes along the free
        dimension. Because `nl.load` has no `mask` in NKI 0.5.0, the ragged tail is a
        separate static-size `nl.ds` tile rather than a masked full-height one.

        Args:
            a_tensor: 2D input [rows, hidden_size]
            g_tensor: 1D weight [hidden_size]
            eps_value: epsilon for numerical stability
        """
        out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

        num_rows, num_cols = a_tensor.shape

        # Load the weight once; reused across every row tile.
        g_tile = nl.load(g_tensor.reshape((1, num_cols)))

        num_full = num_rows // PARTITION_MAX
        rem = num_rows % PARTITION_MAX

        # NKI rejects inner function definitions inside a kernel, so the tile body is
        # inlined in both branches rather than factored out.
        # The reduction and reciprocal are computed in float32, for two reasons:
        #   1. Required. `nisa.tensor_scalar_arith` rejects a bf16 per-partition operand
        #      ("operand0 must be float32"), which is what the [rows, 1] reciprocal is.
        #   2. Correct. PyTorch's RMSNorm upcasts to float32 for the variance and casts
        #      back at the end, so this matches the reference more closely than computing
        #      the reduction in the input dtype did.
        for i in nl.affine_range(num_full):
            start = i * PARTITION_MAX
            a_tile = nl.load(a_tensor[nl.ds(start, PARTITION_MAX), :])
            mean_sq = nl.mean(nl.square(a_tile, dtype=nl.float32),
                              axis=[1], keepdims=True, dtype=nl.float32)
            rms_recip = nl.rsqrt(nl.add(mean_sq, eps_value))
            normed = nl.multiply(a_tile, rms_recip)
            g_b = nl.broadcast_to(g_tile, (PARTITION_MAX, num_cols))
            nl.store(
                out_tensor[nl.ds(start, PARTITION_MAX), :],
                value=nl.multiply(normed, g_b, dtype=a_tensor.dtype),
            )

        if rem > 0:
            start = num_full * PARTITION_MAX
            a_tile = nl.load(a_tensor[nl.ds(start, rem), :])
            mean_sq = nl.mean(nl.square(a_tile, dtype=nl.float32),
                              axis=[1], keepdims=True, dtype=nl.float32)
            rms_recip = nl.rsqrt(nl.add(mean_sq, eps_value))
            normed = nl.multiply(a_tile, rms_recip)
            g_b = nl.broadcast_to(g_tile, (rem, num_cols))
            nl.store(
                out_tensor[nl.ds(start, rem), :],
                value=nl.multiply(normed, g_b, dtype=a_tensor.dtype),
            )

        return out_tensor


def _pytorch_rmsnorm(hidden_states, weight, eps):
    """Reference PyTorch RMSNorm — used as fallback off-device."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


_warned: set[str] = set()


def _warn_once(reason: str) -> None:
    """Announce a fallback instead of taking it silently.

    This kernel is the reason Finding #8 exists: it fell back on CPU tensors with no
    signal of any kind, and an entire accuracy suite passed while never executing NKI.
    Correct-but-unaccelerated is the failure mode that costs the most time, precisely
    because nothing looks wrong. So say so, once, with the reason.
    """
    import warnings

    if reason not in _warned:
        _warned.add(reason)
        warnings.warn(
            f"neuron_rmsnorm: falling back to eager PyTorch RMSNorm ({reason}). "
            "The NKI kernel is NOT being used.",
            RuntimeWarning,
            stacklevel=3,
        )


def _nki_unsupported_reason(hidden_states: torch.Tensor):
    """Return None if the NKI kernel can run, else the reason it cannot."""
    if not _HAS_NKI:
        return "NKI unavailable"
    # @nki.jit hard-errors on CPU tensors, so this guard is mandatory.
    if hidden_states.device.type == "cpu":
        return "input on CPU; NKI requires XLA/Neuron tensors"
    if hidden_states.numel() == 0:
        return "empty tensor"
    return None


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

        reason = _nki_unsupported_reason(hidden_states)
        if reason is None:
            # Run NKI kernel on NeuronCores
            output_2d = _nki_rmsnorm_kernel(
                hidden_2d, self.weight, self.variance_epsilon
            )
        else:
            # PyTorch fallback — announced, never silent. See Finding #8.
            _warn_once(reason)
            output_2d = _pytorch_rmsnorm(
                hidden_2d, self.weight, self.variance_epsilon
            )

        return output_2d.reshape(original_shape)


class layers:
    NeuronRMSNorm = NeuronRMSNorm
