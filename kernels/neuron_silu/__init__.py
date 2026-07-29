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

import math

import torch
import torch.nn as nn

# IMPORT PATH MATTERS, AND THIS KERNEL REQUIRES `neuronxcc.nki`.
#
# Both `nki` and `neuronxcc.nki` import successfully but are NOT interchangeable at
# kernel-compile time, and neither is a superset. The top-level `nki` package fails to
# resolve `nl.arange` —
#     error: failed to resolve name 'nki.language.arange'
# — even though `hasattr(nl, "arange")` is True. This kernel uses the arange-based
# index-tensor idiom to mask partial tiles, so it needs `neuronxcc.nki`. Conversely our
# RoPE kernel needs top-level `nki`, because `neuronxcc.nki` treats shape values as
# symbolic scalars and rejects `//` on them. See docs/sticking-points.md.
_HAS_NKI = False
try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    _HAS_NKI = True
except ImportError:
    try:
        import nki
        import nki.language as nl

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
        """out = silu(a) = a * sigmoid(a), computed 128 rows at a time.

        Args:
            a_tensor: 2D input [rows, cols]
        """
        out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

        num_rows, num_cols = a_tensor.shape

        ix = nl.arange(PARTITION_MAX)[:, None]
        iy = nl.arange(num_cols)[None, :]

        for i in nl.affine_range(math.ceil(num_rows / PARTITION_MAX)):
            row_mask = i * PARTITION_MAX + ix < num_rows
            tile = nl.load(a_tensor[i * PARTITION_MAX + ix, iy], mask=row_mask)
            nl.store(
                out_tensor[i * PARTITION_MAX + ix, iy],
                value=nl.silu(tile),
                mask=row_mask,
            )

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
