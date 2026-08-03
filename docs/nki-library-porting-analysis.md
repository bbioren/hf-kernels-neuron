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

---

# Week 3 Update — RoPE, MLP, and a strategy change

Three things changed the picture this week. Taken together they invert the Week 2
conclusion.

## 1. The "kernels are too fused to port" conclusion was over-generalized

Week 2 looked at exactly one kernel (RMSNorm), found it fused with FP8 quantization
with no unfused path, and concluded nki-library kernels are hard to port. That was
correct about RMSNorm and wrong as a general claim.

| Kernel | Standalone? | Fusion forced? | Interface match to HF | Verdict |
|--------|-------------|----------------|----------------------|---------|
| `rmsnorm/rmsnorm_quant.py` | no | **yes** — always quantizes; `QuantizationType.NONE` unsupported | poor (explicit args, kargs dataclass) | must reimplement |
| `embeddings/rope_hf.py` | **yes** | no | **already HF-shaped** | ported directly ✓ |
| `mlp/mlp.py` | **yes** | **no** — quant/norm both opt-in, default `NONE`/`NO_NORM` | good, except weight layout | feasible, blocked on weight lifecycle |

RMSNorm is the outlier, not the archetype. Any assessment of "can we port nki-library
at scale" that generalizes from a single kernel will be wrong in either direction.

## 2. Case study: RoPE — the port that worked

**Source:** `core/embeddings/rope_hf.py`. Note the directory name: `embeddings/` contains
*position* embeddings only — it is 100% RoPE. There is no token-embedding kernel anywhere
in nki-library.

```python
@nki.jit
def rope_hf(q, k, q_out, k_out, cos=None, sin=None,
            rope_cache=None, backward=False) -> Tuple[NkiTensor, NkiTensor]
```

**Why this one was easy:**
- 4D `[batch, heads, seq, head_dim]` — HF's layout exactly
- accepts **precomputed** cos/sin; no internal theta or `position_ids`
- returns a **tuple** `(q_out, k_out)`, like `apply_rotary_pos_emb`
- independent `q_heads` / `k_heads`, so GQA and MQA work unmodified
- `rotate_half` convention, matching Qwen3/Llama (not interleaved GPT-NeoX style)
- only **3 internal symbols** to inline (`kernel_assert`, `div_ceil`,
  `get_verified_program_sharding_info`), ~15 lines, and no `common_types` import
- a torch reference (`rope_hf_torch.py`) that is HF's `apply_rotary_pos_emb` verbatim

**There is no RoPE tutorial anywhere in nki-samples.** Zero rotary examples. So for RoPE
the production library was the *only* source and also the *better* one — the reverse of
RMSNorm.

**Adaptations we made:**

| Change | Why |
|--------|-----|
| Stripped SPMD sharding (`num_shards = 1`) | HF swaps per-layer and does not manage multi-core. Biggest semantic reduction. |
| Destination-passing → internal `nl.shared_hbm` allocation, return tuple | Matches HF's signature. Verified multi-output `@nki.jit` works first. |
| Dropped `rope_cache` branch | HF always passes cos/sin separately |
| Dropped `backward=True` branch | We expose `has_backward=False`; would be dead code |
| Asserts → Python-level guards | Lets us fall back gracefully instead of crashing |

**Result:** 20/20 accuracy cases + 6/6 guard cases pass on hardware.

**The inherited constraint is the real limitation:** `seq_len % (128 × LNC) == 0`. HF
passes arbitrary sequence lengths, so this is the single most likely reason a customer
silently gets no acceleration. A padding strategy (pad to a multiple of 128, slice the
result) would remove it and is worth a perf evaluation.

**NKI idiom worth recording — there is no concatenation primitive.** `rotate_half`'s
`torch.cat((-x2, x1), dim=-1)` is expressed by preallocating the full-width destination
and writing into disjoint halves, folding the negation into `op=nl.subtract` rather than
spending an instruction on `nl.negative`:

```python
nisa.tensor_tensor(dst=result, data1=x_tile, data2=cos_tile, op=nl.multiply)
nisa.tensor_tensor(dst=temp1, data1=x_tile[:, :, half:],
                   data2=sin_tile[:, :, :half], op=nl.multiply)
nisa.tensor_tensor(dst=result[:, :, :half], data1=result[:, :, :half],
                   data2=temp1, op=nl.subtract)          # <- supplies the "-x2"
```

## 3. Case study: MLP — feasible, but blocked on weight layout

**Source:** `core/mlp/mlp.py`, entry point `nkilib.core.mlp.mlp.mlp` (40 parameters).
This is the kernel that matters for performance, because a standalone elementwise SiLU is
memory-bandwidth bound while the fused unit is not.

```python
@nki.jit
def mlp(hidden_tensor, gate_proj_weights_tensor, up_proj_weights_tensor,
        down_proj_weights_tensor, normalization_weights_tensor=None, ...,
        activation_fn=ActFnType.SiLU, normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE, ...) -> list[NkiTensor]
```

**Good news, and it is substantial:**
- **Quantization is opt-in, not baked in** (`QuantizationType.NONE` is the default).
  Unlike RMSNorm, there is no forced-quant path. This is the cleanest kernel in
  nki-library for HF purposes.
- Fuses more than we need: optional residual add + optional RMSNorm/LayerNorm +
  gate/up projections + activation + down projection. Could cover
  `post_attention_layernorm` + `Qwen3MLP` + residual in one call.
- **Single-core works.** SPMD is optional (`program_ndim() in (0, 1)` allowed, explicit
  non-SPMD fallback in `mlp_cte_basic.py`). Not a blocker as feared.
- Qwen3-8B (H=4096, I=12288) and Qwen3-0.6B (H=1024, I=3072) satisfy every hard
  constraint on the BF16 non-quant path (`H % 128 == 0`, no divisibility requirement on
  `I` or `BxS`).
- A torch reference (`mlp_torch.py::mlp_torch_ref`) and an input generator
  (`test_mlp_common.py::build_fused_norm_mlp`) already exist.

**Two implementations behind one entry point:** `BxS <= 96` routes to TKG (decode),
`> 96` to CTE (prefill). Training at realistic sequence lengths is always CTE.

**Returns a `list`, always** — the wrapper must take `[0]`.

**The blocker: every HF MLP weight is transposed relative to what the kernel wants.**

| Tensor | HF `nn.Linear.weight` | NKI wants | Fix |
|--------|----------------------|-----------|-----|
| `gate_proj` | `[I, H]` | `[H, I]` | transpose |
| `up_proj` | `[I, H]` | `[H, I]` | transpose |
| `down_proj` | `[H, I]` | `[I, H]` | transpose |
| RMSNorm weight | `[H]` | `[1, H]` | `view(1, H)` — free |

No concatenation or swizzling needed on the BF16 path (gate and up stay separate), but
all three matrices need a **materialized** transpose — `.t()` is a free view in torch, but
the kernel DMAs from HBM assuming row-major, and non-contiguous tensor failures are a
known live issue on the Neuron beta, so a view is not safe to pass.

**And `kernelize()` gives nowhere clean to do it.** It rewrites `forward` pointers; it
never transforms parameters. Four options, each bad differently:

| Option | Cost |
|--------|------|
| Mutate parameters in place at kernelize time | `state_dict()` / `save_pretrained()` now emit transposed weights that won't load into a stock `Qwen3MLP`. Silent checkpoint corruption. |
| Hold a second transposed copy | ~2x MLP weight memory. MLP dominates Qwen3-8B, so close to 2x model memory — likely fatal on 96 GB at real batch sizes. |
| Transpose lazily on first forward, cache | Same memory cost, plus a first-step stall |
| Transpose per forward | Three weight-sized HBM round trips per layer per step — erases the entire speedup |

**This is arguably a more valuable finding than the kernel itself**, because it blocks
*every* fused-kernel port on Neuron, not just this one. It is a design decision the
kernels team should own, not something a PoC should quietly pick.

The fusion API does at least provide a home for the chosen strategy:
`make_parent_class_for_kernel_fusion` collapses sibling modules to `nn.Identity()`, so the
surviving `gate_proj` module is an obvious place to hold transposed weights.

### SPIKE RESULT (2026-07-29) — the kernel works, but only at toy sizes

The 1-2 day derisking spike recommended below was run (`scripts/spike_nkilib_mlp.py`). Two
outcomes, and the second one changes the plan.

**Positive: the production kernel is drivable directly from PyTorch/XLA with HF weights.**
No vendoring, no reimplementation — `from nkilib.core.mlp.mlp import mlp`, transpose the three
weights, take `[0]` off the returned list:

| Config | dtype | cos_sim vs Qwen3MLP | max_diff |
|--------|-------|---------------------|----------|
| H=1024, I=3072 | fp32 | 0.999989 | 1.5e-02 |
| H=1024, I=3072 | bf16 | 0.999995 | 9.6e-03 |
| H=4096, I=4096 | bf16 | 0.999979 | 1.3e-02 |

`max_diff ~1e-2` is expected for a fused three-matmul and consistent with nki-library's own
`rtol=2e-2` tolerance — but well outside the tight bar we hold RMSNorm and RoPE to. The
tolerance question flagged earlier is real, not hypothetical.

**Blocking: it cannot run single-core above `intermediate_size = 4096`.** Sharp boundary,
10 configs across three `hidden_size` values, passes iff `I <= 4096` (I=4096 passes, I=4224
fails). Not fixed by sequence length, `force_cte_mode`, or `mode=PREFILL`. Fails inside the
kernel's own tile arithmetic:

```
error: 'floordiv' does not allow division by zero
  kernel_helpers.py:104  <-  tile_info.py:37  <-  mlp_cte_tile_info.py:236
                             build_with_subtiling(bxs_dim_size, ..., bxs_dim_subtile_size)
```

Almost certainly the CTE sharding heuristic forcing `shard_on_inter = True` above I=4096 —
exactly our boundary — with no SPMD grid to shard across.

Every real model is on the wrong side: Qwen3-8B I=12288, Llama-3-8B and Mistral-7B I=14336.
See Finding #18. **A wrapper cannot work around this**, unlike the weight-layout question.

**This is the general SPMD concern, measured.** This document's Week 2 section listed "SPMD
multi-core assumptions don't fit the per-layer swap model" as a general worry. For RoPE,
stripping SPMD was clean and harmless (`num_shards = 1`). For the fused MLP it is not
optional — the tiling *requires* the multi-core path at any useful size. So
**SPMD-strippability is per-kernel, and for fused kernels it may not hold at all.** That is a
structural question about whether the Kernel Hub's one-layer-one-core model can host
nki-library's fused kernels, independent of dependency packaging or weight layout.

**Revised effort estimate:** the spike is done (worth it — it surfaced #18 before 2-3 weeks
went into the integration). Landing the fused MLP in HF is now **blocked**, not merely
expensive: Finding #18 must be resolved by nki-library first, then Finding #17 decided by the
HF kernels team. Do not start the fusion-API work until both.

**Original estimate, for the record:** validating standalone was a 1-2 day spike; getting it
into HF via the fusion API, correct *and* faster, was 2-3 weeks.

**Note on tolerance:** nki-library's own MLP tests use `rtol=2e-2` — two orders of
magnitude looser than the `cos_sim > 0.999` bar we hold RMSNorm/RoPE to. A fused
three-matmul kernel will not be bit-identical. Decide which bar applies before starting.

## 4. Strategy change: Option D is technically live today

`docs/nki-library-porting-analysis.md` originally listed Option D — "use nki-library as a
pip dependency" — as blocked, on the assumption that `nkilib` isn't available and isn't
allowed.

**Half of that is wrong. `nkilib` is already installed in the Neuron venv:**

```
/opt/aws_neuronx_venv_pytorch_2_9/lib/python3.12/site-packages/nkilib/
```

Every production kernel imports cleanly (verified, `scripts/probe_nkilib_bundled.py`):

| Module | Importable |
|--------|-----------|
| `nkilib.core.embeddings.rope_hf` | yes (`rope_hf` present) |
| `nkilib.core.mlp.mlp` | yes (`mlp` present, 40 params, signature matches `main`) |
| `nkilib.core.moe.moe_cte.moe_cte` | yes |
| `nkilib.core.router_topk.router_topk` | yes |
| `nkilib.core.rmsnorm.rmsnorm_quant` | yes |

**And the production kernel is directly callable from PyTorch/XLA, correctly.** Verified
(`scripts/experiment_nkilib_thin_wrapper.py`), calling installed `rope_hf` on Neuron:

| Strategy | Result |
|----------|--------|
| Pass preallocated `q_out`/`k_out`, read the **return value** | **q cos_sim 1.000001, k cos_sim 1.000000** |
| Pass preallocated outputs, read the **mutated arguments** | cos_sim **0.000000** — not mutated |

So the destination-passing convention is **vestigial across the XLA boundary**: the output
tensors must still be passed (as shape/dtype templates) but the results come back via the
return value, not by mutation. Worth documenting for anyone wrapping these kernels — the
nki-library integration tests use `must_alias_input`, which would mislead you into
strategy B and silently give zeros.

**What this means.** A thin-wrapper HF kernel is technically feasible *today*:

```python
class NeuronRoPE(nn.Module):
    def forward(self, q, k, cos, sin, unsqueeze_dim=1):
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        return rope_hf(q, k, q_out, k_out, cos=cos, sin=sin)
```

The remaining blocker is **policy, not code**: `python-depends` whitelists `nki` but not
`nkilib`, and the neuron table is unreachable anyway (Finding #12). A wrapper today would
have to under-declare its dependency and rely on `nkilib` happening to be preinstalled —
the same fragility that already applies to `nki`.

**Revised recommendation.** The Week 2 answer to "can we mass-produce HF wrappers?" was
"not automatically, each kernel needs defusion + interface adaptation + dependency
inlining + SPMD stripping." That is still true for the *self-contained* approach. But the
thin-wrapper approach removes all four of those steps at once, and it is now demonstrated
to work numerically. The ask changes from a large engineering program to two small
upstream changes:

1. Add `nkilib` to `kernels/python_depends.json` under the `neuron` backend — a four-line
   change, with `nki` already there as precedent and the exact JSON shape to copy.
2. Fix `_backend()` so the neuron table is actually consulted (Finding #12).

With those two, a per-kernel HF wrapper is a few dozen lines and mass porting becomes
mechanical. Without them, every kernel is a hand-port and the MLP's 7,250-line dependency
closure (≈480x RoPE's 15 lines) makes the larger kernels impractical.

**Our hand-ports remain the right choice for this PoC**: self-contained, no undeclarable
dependency, and they document the porting path and its friction. But they are not the
shape to build a program on.

## 5. MoE feasibility (rough, for Week 5)

`core/moe/moe_cte/moe_cte.py` exposes `@nki.jit moe_cte(...)`, a blockwise grouped-matmul
MoE. Weight layout is a **good** match for Llama4 and a poor one for Qwen3-MoE — the
opposite of the dense case:

- wants `gate_up_proj_weight [E, H, 2, I_TP]`, `down_proj_weight [E, I_TP, H]`
- transformers' `Llama4TextExperts` holds `gate_up_proj [E, H, 2I]`, `down_proj [E, I, H]`
  — a **free reshape, no transpose**
- `Qwen3MoeExperts` uses `[E, 2I, H]` / `[E, H, I]` via `F.linear`, so it *would* need
  transposes

The real work is metadata, not matmul: `moe_cte` requires caller-supplied
`token_position_to_id`, `block_to_expert`, `expert_affinities_masked`, and a `[T+1, H]`
hidden tensor with a padding row. Token sorting and block assignment live *outside* the
kernel, whereas the megablocks path HF wraps builds them internally. That gap is the port.

`core/router_topk/router_topk.py` maps cleanly onto `Qwen3MoeTopKRouter`
(x@w+bias → softmax/sigmoid → top-k → scatter affinities; constraints T≤2048, E≤512,
H%128==0, K≤8) but writes into caller-allocated mutable outputs rather than returning, so
it needs an allocation wrapper. It is also the **only** file under `src/` in the MLP/MoE
area that imports `neuronxcc.nki` (for `neuronxcc.nki.typing as nt`) — relevant given the
import-path split in Finding #14.

Naming caution: `"Llama4TextMoe"` is **commented out** in transformers' `_KERNEL_MAPPING`
("NOTE: No longer maintained"). The only live MoE layer name is `"MegaBlocksMoeMLP"`.

## 6. Documentation bugs found in nki-library

Worth reporting upstream:

1. **`rope_hf` is absent from the public API reference entirely** — the most HF-friendly
   kernel in the library is source-only and undiscoverable.
2. The API reference cites `nkilib.core.rope.RoPE`; the real path is
   `nkilib.core.embeddings.rope.RoPE`. `nkilib/core/rope.py` does not exist.
3. `mlp()`'s mode assert message says `"PREFILL (token gen) or DECODE (context
   encoding)"` — the parenthetical labels are swapped relative to actual behaviour.
   Anyone debugging from the error message will wire the mode backwards.
4. Nothing documents the single-core `intermediate_size <= 4096` limit for `mlp()`. The
   documented constraint is `H % 128 == 0`, which I=12288 satisfies, so the failure looks
   like a compiler bug rather than an unmet precondition. If the limit is intentional it
   should be a `kernel_assert` with a clear message; if not, it is a bug (Finding #18).
5. The integration tests use `"q_out.must_alias_input"` for destination-passing kernels,
   which suggests outputs are mutated in place. From PyTorch/XLA they are **not** — the
   results come back via the return value and the passed tensors stay untouched. Following
   the test idiom silently yields zeros (Finding #16).
