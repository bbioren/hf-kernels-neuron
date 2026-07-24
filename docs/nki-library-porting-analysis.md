# nki-library → HF Kernel Hub: Porting Analysis

## Overview

The `aws-neuron/nki-library` is the production NKI kernel library. The goal of this PoC is to determine how to take these kernels and wrap them for the HuggingFace Kernel Hub so `use_kernels=True` gives users NKI-accelerated models automatically.

**Source:** https://github.com/aws-neuron/nki-library/tree/main/src/nkilib_src/nkilib/core

## Available Kernels in nki-library

| Kernel | Directory | Relevance to HF models |
|--------|-----------|----------------------|
| RMSNorm (+Quant) | `core/rmsnorm/` | RMSNorm in 87 model files (Qwen3, Llama, Mistral, etc.) |
| Attention | `core/attention/` | Flash-attention style, all transformer models |
| MLP | `core/mlp/` | Gate/Up/Down projections with SiLU |
| MoE | `core/moe/` | Mixture-of-experts routing |
| MoE Block | `core/moe_block/` | Full MoE transformer block |
| QKV | `core/qkv/` | Fused QKV projection |
| Embeddings | `core/embeddings/` | Token embeddings |
| Router TopK | `core/router_topk/` | MoE expert selection |
| Output Projection | `core/output_projection/` | Final linear projection |
| TopK | `core/topk/` | General top-k operation |
| Cumsum | `core/cumsum/` | Cumulative sum |
| Max | `core/max/` | Max reduction |

## Case Study: RMSNorm

### nki-library structure

```
core/rmsnorm/
├── __init__.py                    # Empty (just license)
├── rmsnorm_quant.py              # Main kernel: fused RMSNorm + FP8 Quant (42KB, ~800 lines)
├── rmsnorm_mx_prefill.py         # MX-format variant for prefill (51KB)
├── rmsnorm_quant_constants.py    # Constants/config dataclass (5.7KB)
├── rmsnorm_quant_tile_info.py    # Tile sizing logic (1.8KB)
├── rmsnorm_quant_torch.py        # PyTorch reference for testing (5.6KB)
└── rmsnorm_mx_prefill_torch.py   # PyTorch reference for MX variant (13.7KB)
```

### What makes porting hard

**1. Fusion — kernels do MORE than one op**

The nki-library RMSNorm kernel is actually `rmsnorm_quant` — it fuses:
- Optional pre-normalization (RMSNorm with a first gamma)
- Optional residual addition
- RMS normalization (the part we want)
- FP8 quantization (row or static)

For the HF Kernel Hub, we only need the RMSNorm part. We'd have to either:
- Extract just the `_rms_normalize_tile()` subroutine and wrap it standalone
- Or use the full kernel with `NormType.RMS_NORM` and `QuantizationType.NONE` (but NONE isn't even supported — the kernel always quantizes!)

**Finding:** The nki-library kernels are designed for the NxDI inference pipeline where RMSNorm is always followed by quantization. There is NO standalone RMSNorm kernel in nki-library.

**2. Internal dependencies**

The kernel imports from:
```python
from ..utils.common_types import DtypeMode, NormType, QuantizationType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import (
    get_program_sharding_info,
    get_verified_program_sharding_info,
    is_launched_as_spmd,
    is_rms_normalization,
)
from .rmsnorm_quant_constants import RMSNormQuantConstants, build_rms_norm_quant_constants
from .rmsnorm_quant_tile_info import RMSNormQuantTileInfo, build_rms_norm_quant_tile_info
```

HF Kernel Hub requires kernels to be **self-contained** — only stdlib, torch, and the kernel itself can be imported. You'd need to inline or vendor all these dependencies.

**3. Calling conventions don't match**

nki-library RMSNorm expects:
```python
rmsnorm_quant_kernel(
    hidden: nl.NkiTensor,          # [B, S, H] input
    ln_w: nl.NkiTensor,            # [H] weight
    kargs: RmsNormQuantKernelArgs,  # dataclass with eps, norm_type, quant_type, etc.
    input_dequant_scale: nl.NkiTensor = None,
    pre_norm_gamma: nl.NkiTensor = None,
    residual: nl.NkiTensor = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
)
```

HF Kernel Hub expects:
```python
class NeuronRMSNorm(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # reads self.weight, self.variance_epsilon from adopting module
        ...
```

The mismatch: nki-library passes weight and epsilon explicitly as arguments. HF Kernel Hub reads them from `self` (the adopting module). The kernel's `@nki.jit` function signature must be adapted.

**4. SPMD/multi-core assumptions**

The nki-library kernel supports SPMD launching (sharding across multiple NeuronCores). The HF Kernel Hub forward-swap is called per-layer and doesn't manage multi-core parallelism. You'd need to either:
- Strip the SPMD path and use single-core only
- Or add a multi-core dispatch layer in the HF wrapper

**5. No standalone pure-RMSNorm variant**

Looking at the full nki-library, there's:
- `rmsnorm_quant.py` — RMSNorm + FP8 quantization (always quantizes)
- `rmsnorm_mx_prefill.py` — RMSNorm + MX quantization for prefill

Neither provides a pure BF16→BF16 RMSNorm that matches what HF models need. The closest is `_rms_normalize_tile()` which is a subroutine, not a standalone kernel.

## Porting Strategy Options

### Option A: Extract subroutine, wrap standalone

Take `_rms_normalize_tile()` from `rmsnorm_quant.py`, inline its dependencies, and wrap it as a standalone `@nki.jit` kernel. This is essentially what the tutorial kernel does — it's the same math, just structured differently.

**Pros:** Clean, minimal, matches HF expectations
**Cons:** You're writing a new kernel, not really "porting" the production one. The production kernel's optimizations (SPMD, tiling constants, batch processing of 8 tiles) get lost.

### Option B: Use full kernel with quant disabled

Pass `NormType.RMS_NORM` + `QuantizationType.NONE` to `rmsnorm_quant_kernel`.

**Blocker:** `QuantizationType.NONE` isn't validated in the kernel — it expects ROW or STATIC. Would need kernel modification.

### Option C: Vendor the full kernel + dependencies

Copy `rmsnorm_quant.py` + all its dependencies into the HF kernel package. Configure it to do RMSNorm-only by adding a NONE quant path.

**Pros:** Uses the actual production kernel code
**Cons:** ~100KB of code for a simple norm layer. Breaks HF kernel requirements (no external imports beyond stdlib+torch). Would need significant refactoring.

### Option D: Use nki-library as a pip dependency

Add `nkilib` to `python-depends` in metadata.json. Call the kernel through its public API.

**Blocker:** `python-depends` only allows a curated set (currently `einops` and `nvidia-cutlass-dsl`). `nkilib` is not in the allowed list. Would need HF to add it.

## Recommendation

**Option A is the only viable path today.** Write standalone NKI kernels that match the HF layer interface, informed by the nki-library implementations but not directly importing them.

However, the PoC should highlight that **at scale**, Option D is what you'd want: make `nkilib` an allowed dependency, then the HF kernel just becomes a thin wrapper:

```python
class NeuronRMSNorm(nn.Module):
    def forward(self, hidden_states):
        return nkilib.rmsnorm(hidden_states, self.weight, eps=self.variance_epsilon)
```

This requires:
1. nkilib to expose a simple `rmsnorm(input, weight, eps)` function (it doesn't today)
2. HF to add `nkilib` to the allowed python-depends list
3. Or: a new pattern where the kernel's `metadata.json` can declare arbitrary pip deps for the neuron backend

## What This Means for "Automating at Scale"

The engineering team's question is: "can we mass-produce HF kernel wrappers for all nki-library kernels?"

**Answer: Not automatically today.** Each kernel requires:

1. **Defusion** — extracting the specific op from a fused kernel (most nki-library kernels fuse 2-3 ops together)
2. **Interface adaptation** — converting from explicit-argument calling to the stateless `self.weight` pattern
3. **Dependency inlining** — vendoring or removing internal utility imports
4. **SPMD stripping** — removing multi-core dispatch that doesn't fit the HF per-layer model

A realistic automation path would be:
- Add a **simple API layer** to nki-library (e.g. `nkilib.ops.rmsnorm(x, w, eps)`) that wraps the production kernels with sensible defaults
- Get `nkilib` onto the HF `python-depends` allowed list
- Then HF kernel wrappers become trivial one-liners

## Current Approach

For the PoC, we're using the **tutorial-derived kernel** (from the NKI docs) which implements pure RMSNorm without fusion. This is functionally correct and validates the full Kernel Hub integration path. We document that the production nki-library kernels would require the above porting work.

This is itself a finding: the gap between "NKI tutorial kernel" and "nki-library production kernel" is significant, and the PoC should recommend that nki-library add simple unfused entry points for the HF Kernel Hub use case.
