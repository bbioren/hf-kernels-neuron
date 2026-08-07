# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE OF MODIFICATION (Apache-2.0 section 4b)
#   This file is a modified derivative work of:
#     aws-neuron/nki-library, src/nkilib_src/nkilib/core/embeddings/rope_hf.py
#     https://github.com/aws-neuron/nki-library  (Apache-2.0)
#   The notice above is reproduced from that file. Five adaptations were made to
#   fit the HuggingFace `kernels` per-layer forward-swap model; they are
#   enumerated in the module docstring below and analysed in
#   docs/nki-library-porting-analysis.md.
"""NKI RoPE (rotary position embedding) kernel for Neuron, HF-compatible.

PORTED FROM PRODUCTION nki-library, not from a tutorial.
Source: aws-neuron/nki-library, src/nkilib_src/nkilib/core/embeddings/rope_hf.py

This is the opposite situation from RMSNorm. For RMSNorm the production kernel was
unusable (fused with FP8 quantization, no standalone path) so we derived one from
the NKI tutorial. For RoPE there is *no* tutorial anywhere in nki-samples, and the
production kernel `rope_hf` is already written for HuggingFace tensor layout:
4D `[batch, heads, seq, head_dim]`, precomputed cos/sin, tuple return, `rotate_half`
convention. So this is a genuine port.

Adaptations made from the nki-library original (see docs/nki-library-porting-analysis.md):
  1. Inlined `div_ceil`; dropped `kernel_assert` and
     `get_verified_program_sharding_info` (Python-level validation in the wrapper
     instead, so we can fall back rather than crash).
  2. Dropped SPMD sharding — HF swaps per-layer and does not manage multi-core, so
     num_shards is fixed at 1. This is the single biggest semantic reduction.
  3. Converted destination-passing to internal allocation. The original takes
     preallocated `q_out`/`k_out`; HF's `apply_rotary_pos_emb` returns a tuple.
     Verified that multi-output `@nki.jit` works, so outputs are allocated in
     `nl.shared_hbm` and returned directly.
  4. Dropped the `rope_cache` (packed cos‖sin) branch — HF always passes cos/sin
     separately.
  5. Dropped the `backward=True` rotation. We expose `has_backward = False`, so
     carrying an unreachable branch would be dead code. nki-library retains it if
     a backward pass is ever wired up.

Function replacement, not layer replacement: `apply_rotary_pos_emb` is a free
function in transformers, decorated `@use_kernel_func_from_hub("rotary_pos_emb")`.
It is loaded via `LocalFuncRepository(repo_path, func_name=...)`, which looks the
name up at MODULE TOP LEVEL — deliberately *not* inside the `layers` namespace that
layer kernels use.
"""

import warnings

import torch

# IMPORT PATH MATTERS, AND THIS KERNEL REQUIRES THE TOP-LEVEL `nki`.
#
# Both `nki` and `neuronxcc.nki` import successfully, but they are NOT
# interchangeable at kernel-compile time, and neither is a superset:
#
#   * `neuronxcc.nki` supports `nl.arange` index tensors, but treats tensor shape
#     values as symbolic scalars, so `//` on them raises
#         NotImplementedError: math.trunc() is not supported for scalar
#     which breaks the `div_ceil` this kernel inherits from nki-library.
#   * top-level `nki` gives concrete ints for shapes (so `//` is fine) but fails to
#     resolve `nl.arange` at compile time:
#         error: failed to resolve name 'nki.language.arange'
#     even though `hasattr(nl, "arange")` is True.
#
# So each kernel is effectively pinned to whichever package its idiom requires. This
# one uses slicing + `//`, so it needs top-level `nki`. Our RMSNorm and SiLU kernels
# use arange index tensors, so they need `neuronxcc.nki`. Verified both ways by
# swapping the imports and re-running the suites. See docs/sticking-points.md.
_HAS_NKI = False
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    _HAS_NKI = True
except ImportError:
    try:
        import neuronxcc.nki as nki
        import neuronxcc.nki.isa as nisa
        import neuronxcc.nki.language as nl

        _HAS_NKI = True
    except ImportError:
        pass


# Number of 128-row sequence tiles coalesced into one SBUF buffer.
# Carried over from nki-library.
NUM_COALESCE_TILES = 8

# Partition-dimension max. RoPE requires seq_len to be a multiple of this.
PARTITION_MAX = 128


def _div_ceil(n: int, d: int) -> int:
    """Inlined from nkilib.core.utils.kernel_helpers."""
    return (n + d - 1) // d


if _HAS_NKI:

    def _apply_rope_single(x_tile, cos_tile, sin_tile):
        """Rotate one SBUF tile: y = [x1, x2] * [cos1, cos2] + [-x2, x1] * [sin1, sin2]

        Ported verbatim (forward branch) from nki-library `_apply_rope_single`.

        NKI has no concatenation primitive, so `rotate_half`'s
        `torch.cat((-x2, x1), dim=-1)` is expressed by pre-allocating a full-width
        destination and writing into disjoint halves, folding the negation into
        `op=nl.subtract` rather than spending an instruction on `nl.negative`.
        """
        seq_size, num_tiles, head_dim = x_tile.shape
        half = head_dim // 2

        result = nl.ndarray((seq_size, num_tiles, head_dim), dtype=x_tile.dtype, buffer=nl.sbuf)
        temp1 = nl.ndarray((seq_size, num_tiles, half), dtype=x_tile.dtype, buffer=nl.sbuf)
        temp2 = nl.ndarray((seq_size, num_tiles, half), dtype=x_tile.dtype, buffer=nl.sbuf)

        # result = x * cos
        nisa.tensor_tensor(dst=result, data1=x_tile, data2=cos_tile, op=nl.multiply)

        # result[:half] -= x[half:] * sin[:half]     <- supplies the "-x2"
        nisa.tensor_tensor(
            dst=temp1,
            data1=x_tile[:, :, half:],
            data2=sin_tile[:, :, :half],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=result[:, :, :half], data1=result[:, :, :half], data2=temp1, op=nl.subtract
        )

        # result[half:] += x[:half] * sin[half:]     <- supplies the "x1"
        nisa.tensor_tensor(
            dst=temp2,
            data1=x_tile[:, :, :half],
            data2=sin_tile[:, :, half:],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=result[:, :, half:], data1=result[:, :, half:], data2=temp2, op=nl.add
        )

        return result

    def _apply_rope_all_heads(x, x_out, cos_tile, sin_tile, batch_id, seq_start):
        """Apply the rotation to every head for one (batch, seq-tile) pair.

        cos/sin are loaded once per sequence tile and reused across all heads —
        this is the NKI equivalent of HF's `unsqueeze_dim=1` broadcast.
        """
        seq_tile_size, num_tiles, head_dim = cos_tile.shape
        num_heads = x.shape[1]
        span = num_tiles * seq_tile_size

        for head_id in range(num_heads):
            x_tile = nl.ndarray(
                (seq_tile_size, num_tiles, head_dim), dtype=x.dtype, buffer=nl.sbuf
            )
            x_src = (
                x[batch_id, head_id, seq_start : seq_start + span, :]
                .reshape_dim(0, (num_tiles, seq_tile_size))
                .permute((1, 0, 2))
            )
            nisa.dma_copy(x_tile, x_src)

            rotated = _apply_rope_single(x_tile, cos_tile, sin_tile)

            x_dst = (
                x_out[batch_id, head_id, seq_start : seq_start + span, :]
                .reshape_dim(0, (num_tiles, seq_tile_size))
                .permute((1, 0, 2))
            )
            nisa.dma_copy(x_dst, rotated)

    @nki.jit
    def _nki_rope_hf(q, k, cos, sin):
        """RoPE for HF layout. Returns (q_out, k_out).

        Args:
            q:   [batch, q_heads, seq_len, head_dim] @ HBM
            k:   [batch, k_heads, seq_len, head_dim] @ HBM  (k_heads may differ — GQA)
            cos: [batch, seq_len, head_dim] or [seq_len, head_dim] @ HBM
            sin: same shape as cos

        Requires seq_len % 128 == 0. Validated in the Python wrapper.
        """
        q_out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
        k_out = nl.ndarray(k.shape, dtype=k.dtype, buffer=nl.shared_hbm)

        batch_size, _, seq_len, head_dim = q.shape

        seq_tile_size = min(PARTITION_MAX, seq_len)
        num_seq_tiles = _div_ceil(seq_len, seq_tile_size)

        for batch_id in range(batch_size):
            for tile_idx in range(0, num_seq_tiles, NUM_COALESCE_TILES):
                num_tiles = min(NUM_COALESCE_TILES, num_seq_tiles - tile_idx)
                seq_start = tile_idx * seq_tile_size
                seq_end = seq_start + num_tiles * seq_tile_size

                cos_tile = nl.ndarray(
                    (seq_tile_size, num_tiles, head_dim), dtype=q.dtype, buffer=nl.sbuf
                )
                sin_tile = nl.ndarray(
                    (seq_tile_size, num_tiles, head_dim), dtype=q.dtype, buffer=nl.sbuf
                )

                if len(cos.shape) == 3:
                    cos_slice = cos[batch_id, seq_start:seq_end, :]
                    sin_slice = sin[batch_id, seq_start:seq_end, :]
                else:
                    cos_slice = cos[seq_start:seq_end, :]
                    sin_slice = sin[seq_start:seq_end, :]

                cos_src = cos_slice.reshape_dim(0, (num_tiles, seq_tile_size)).permute((1, 0, 2))
                sin_src = sin_slice.reshape_dim(0, (num_tiles, seq_tile_size)).permute((1, 0, 2))

                nisa.dma_copy(cos_tile, cos_src)
                nisa.dma_copy(sin_tile, sin_src)

                _apply_rope_all_heads(q, q_out, cos_tile, sin_tile, batch_id, seq_start)
                _apply_rope_all_heads(k, k_out, cos_tile, sin_tile, batch_id, seq_start)

        return q_out, k_out


# --------------------------------------------------------------------------
# PyTorch fallback — matches transformers' apply_rotary_pos_emb exactly
# --------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _torch_rope(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Reference implementation, identical to transformers' apply_rotary_pos_emb."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

_warned: set[str] = set()


def _warn_once(reason: str) -> None:
    """Surface a fallback instead of failing silently.

    Finding #8: a silent fallback is the worst failure mode on Neuron, because the
    numbers stay correct and the user concludes they have acceleration. Anything
    that declines the NKI path says so, once, with the reason.
    """
    if reason not in _warned:
        _warned.add(reason)
        warnings.warn(
            f"neuron_rope: falling back to eager PyTorch RoPE ({reason}). "
            "The NKI kernel is NOT being used.",
            RuntimeWarning,
            stacklevel=3,
        )


def _nki_unsupported_reason(q, k, cos, sin, unsqueeze_dim):
    """Return None if the NKI kernel can handle these inputs, else why not."""
    if not _HAS_NKI:
        return "NKI unavailable"

    # @nki.jit hard-errors on CPU tensors, so this guard is mandatory.
    if q.device.type == "cpu" or k.device.type == "cpu":
        return "inputs on CPU; NKI requires XLA/Neuron tensors"

    if q.ndim != 4 or k.ndim != 4:
        return f"kernel is 4D-only, got q.ndim={q.ndim}, k.ndim={k.ndim}"

    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim:
        return f"head_dim mismatch q={head_dim} k={k.shape[-1]}"
    if head_dim % 2 != 0:
        return f"head_dim must be even, got {head_dim}"

    seq_len = q.shape[2]
    if k.shape[2] != seq_len:
        return f"seq_len mismatch q={seq_len} k={k.shape[2]}"
    # nki-library asserts seq_len % (128 * LNC) == 0. We run single-core, so LNC=1.
    if seq_len % PARTITION_MAX != 0:
        return f"seq_len must be a multiple of {PARTITION_MAX}, got {seq_len}"

    if unsqueeze_dim != 1:
        return f"kernel assumes unsqueeze_dim=1 (broadcast over heads), got {unsqueeze_dim}"

    if cos is None or sin is None:
        return "cos/sin required"
    if cos.shape != sin.shape:
        return f"cos/sin shape mismatch {tuple(cos.shape)} vs {tuple(sin.shape)}"
    if cos.ndim not in (2, 3):
        return f"cos/sin must be 2D or 3D, got {cos.ndim}D"
    if cos.shape[-1] != head_dim:
        return f"cos last dim {cos.shape[-1]} != head_dim {head_dim}"
    if cos.ndim == 3 and cos.shape[0] != q.shape[0]:
        return f"cos batch {cos.shape[0]} != q batch {q.shape[0]}"
    if cos.shape[-2] < seq_len:
        return f"cos seq_len {cos.shape[-2]} < q seq_len {seq_len}"
    if q.dtype != k.dtype:
        return f"q/k dtype mismatch {q.dtype} vs {k.dtype}"

    return None


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """NKI-accelerated drop-in for transformers' `apply_rotary_pos_emb`.

    Signature matches transformers exactly, so it can be swapped in via
    `LocalFuncRepository(func_name="apply_rotary_pos_emb")` against the
    `"rotary_pos_emb"` kernel name.

    Falls back to eager PyTorch — with a warning — whenever the NKI kernel's
    constraints aren't met (non-multiple-of-128 seq_len being the common case).
    """
    reason = _nki_unsupported_reason(q, k, cos, sin, unsqueeze_dim)
    if reason is None:
        # cos/sin must share q's dtype; the kernel allocates SBUF tiles as q.dtype.
        if cos.dtype != q.dtype:
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
        return _nki_rope_hf(q, k, cos, sin)

    _warn_once(reason)
    return _torch_rope(q, k, cos, sin, unsqueeze_dim)


# Flags read by the kernels library off the FUNCTION object.
# `has_backward` defaults to True for function kernels (unlike layers, where it
# defaults to False), so it must be set explicitly. We do not wire autograd —
# nki-library has a backward rotation available, but it is not exposed here — so
# False is the honest value and keeps the kernel out of TRAINING mode.
apply_rotary_pos_emb.has_backward = False
apply_rotary_pos_emb.can_torch_compile = False
