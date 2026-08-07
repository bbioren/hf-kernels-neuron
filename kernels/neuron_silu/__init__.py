# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Original work, not a port. NKI exposes `nl.silu` natively, and nki-library has no
# activations module, so there was nothing upstream to derive from.
"""NKI SiLU activation kernel for Neuron.

Swaps `transformers.activations.SiLUActivation`, which is decorated
`@use_kernel_forward_from_hub("SiLU")`. Stateless — no weights, forward takes a
single tensor.

NKI provides `nl.silu` natively, so unlike RMSNorm and RoPE there is nothing to
port: the kernel is a tiled load / activate / store. `nl.silu_dx` also exists if a
backward pass is ever wired up.

A caveat worth stating plainly, because it shapes the Week 4 recommendation:
standalone elementwise SiLU is **memory-bandwidth bound**. It reads N elements and
writes N elements to do about two FLOPs each. Replacing it with a NKI kernel does
not remove that traffic, and in eager mode it adds a separate kernel launch plus an
HBM round trip that a fused XLA lowering might have avoided. The profitable unit is
the *fused* gate/up/SiLU/down MLP, which is what nki-library's `core/mlp/` kernel
actually implements.

Is that fusion reachable through the Kernel Hub? Partly, and not the obvious way.
`_KERNEL_MAPPING` does contain a `"SwiGLUMLP"` entry (mapped to liger-kernels on
CUDA), but **no model file registers that name** — `grep` for
`use_kernel_forward_from_hub("SwiGLUMLP")` across transformers returns zero hits, and
`Qwen3MLP` carries no decorator at all. So the mapping entry is unreachable via the
decorator path.

Fused MLP replacement instead goes through the separate fusion API,
`register_kernel_replacements_and_fusions()` / `make_parent_class_for_kernel_fusion()`,
driven by `KernelConfig`: it swaps the first named child for the kernel and replaces
the siblings with `nn.Identity()`. That is the route a fused NKI MLP kernel would
have to take, and it is a different (more invasive) integration than the per-layer
forward swap this project has validated. Week 4/5 territory.

So this kernel exists to prove the activation interception path works and to complete
mechanism coverage — not because it is expected to be a speedup on its own. Do not
claim a win for it without measuring.
"""

import torch
import torch.nn as nn

# NKI 0.5.0 (top-level `nki`) is the going-forward surface; `neuronxcc.nki` is the older API
# bundled inside neuronx-cc, kept only as a fallback.
#
# This kernel originally used `nl.arange` index tensors plus `mask=` for ragged tails, which
# pinned it to the older API. Both are gone in 0.5.0 — `nl.arange` was removed in favour of
# `nl.ds`, and `nl.load`/`nl.store` no longer take `mask`. See Finding #14.
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


PARTITION_MAX = 128

# Conservative cap on the free-dimension width we load into SBUF in one go.
# 128 rows x 16384 cols x 4 bytes = 8 MiB, comfortably inside SBUF (28 MiB/core on
# Trn2) with room for the output tile. Wider inputs fall back rather than risk an
# allocation failure at trace time.
MAX_FREE_DIM = 16384


if _HAS_NKI:

    @nki.jit
    def _nki_silu_kernel(a_tensor):
        """out = silu(a) = a * sigmoid(a), tiled 128 rows at a time.

        `nl.silu` is a native NKI primitive, so there is no math to port here — the kernel
        is a tiled load / activate / store. The ragged tail is a separate static-size
        `nl.ds` tile because `nl.load` has no `mask` in NKI 0.5.0.

        Args:
            a_tensor: 2D input [rows, cols]
        """
        out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

        num_rows, _ = a_tensor.shape

        num_full = num_rows // PARTITION_MAX
        rem = num_rows % PARTITION_MAX

        for i in nl.affine_range(num_full):
            start = i * PARTITION_MAX
            tile = nl.load(a_tensor[nl.ds(start, PARTITION_MAX), :])
            nl.store(out_tensor[nl.ds(start, PARTITION_MAX), :], value=nl.silu(tile))

        if rem > 0:
            start = num_full * PARTITION_MAX
            tile = nl.load(a_tensor[nl.ds(start, rem), :])
            nl.store(out_tensor[nl.ds(start, rem), :], value=nl.silu(tile))

        return out_tensor


def _torch_silu(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation, identical to SiLUActivation.forward."""
    return nn.functional.silu(x)


_warned: set[str] = set()


def _warn_once(reason: str) -> None:
    """Never fall back silently — see Finding #8."""
    import warnings

    if reason not in _warned:
        _warned.add(reason)
        warnings.warn(
            f"neuron_silu: falling back to eager PyTorch SiLU ({reason}). "
            "The NKI kernel is NOT being used.",
            RuntimeWarning,
            stacklevel=3,
        )


def _nki_unsupported_reason(x: torch.Tensor):
    if not _HAS_NKI:
        return "NKI unavailable"
    if x.device.type == "cpu":
        return "input on CPU; NKI requires XLA/Neuron tensors"
    if x.numel() == 0:
        return "empty tensor"
    if x.shape[-1] > MAX_FREE_DIM:
        return f"last dim {x.shape[-1]} exceeds SBUF-safe width {MAX_FREE_DIM}"
    return None


class NeuronSiLU(nn.Module):
    """Stateless SiLU activation backed by NKI.

    Adopts the forward of `transformers.activations.SiLUActivation`, which holds no
    parameters, so nothing is read off `self`.
    """

    has_backward: bool = False
    can_torch_compile: bool = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        reason = _nki_unsupported_reason(input)
        if reason is not None:
            _warn_once(reason)
            return _torch_silu(input)

        original_shape = input.shape
        x2d = input.reshape(-1, original_shape[-1])
        out2d = _nki_silu_kernel(x2d)
        return out2d.reshape(original_shape)


class layers:
    NeuronSiLU = NeuronSiLU
