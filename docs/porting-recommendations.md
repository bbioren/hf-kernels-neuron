# Porting Recommendations for Engineering Team

How should the Neuron kernels team approach porting nki-library kernels to the HF Kernel Hub at scale? This doc accumulates recommendations as we learn from the PoC.

---

## Executive Summary

(To be written in Week 6 from accumulated findings below)

---

## Headline recommendation (revised Week 3)

**Do not build a hand-porting program. Fund two small upstream changes instead.**

Week 2's recommendation assumed each kernel needs defusion, interface adaptation,
dependency inlining, and SPMD stripping — a per-kernel engineering cost. Week 3 found that
`nkilib` is **already installed** in the Neuron venv and its production kernels are
**directly callable from PyTorch/XLA with correct results** (verified: installed `rope_hf`
gives `cos_sim 1.000001`). A thin-wrapper HF kernel is a few dozen lines with no vendoring.

The scale difference decides it: RoPE needed ~15 lines inlined; the MLP kernel's dependency
closure is **7,249 lines across 22 files** (≈480x). Hand-porting does not extend to the
kernels that matter for performance.

So the highest-value asks are:

| # | Ask | Owner | Size |
|---|-----|-------|------|
| 1 | Add `nkilib` to `kernels/python_depends.json` under the `neuron` backend | HF `kernels` | ~4 lines; `nki` is already there as precedent |
| 2 | Fix `_backend()` to report `neuron` (set `torch.neuron`) so the table is consulted | `torch_neuronx` | 1 attribute; also fixes build-variant resolution |
| 3 | Map `xla`→`neuron` in device resolution so `use_kernels=True` works | transformers | ~5 lines, 3 sites; **verified sufficient** |
| 4 | Decide the weight-layout contract for fused kernels (Finding #17) | HF `kernels` + nki-library | design decision, blocks all fused ports |

Items 1-3 are each smaller than a single kernel port, and together they unlock the whole
model zoo. Item 4 is the one genuine design question.

**Caveat to weigh:** a thin wrapper couples the HF kernel repo to `nkilib` *and*
`neuronx-cc` versions. nki-library's README warns GitHub `main` is not guaranteed
compatible with a given compiler. Vendored kernels don't have that coupling. This is a real
tradeoff, not a slam dunk — but version coupling is a more tractable problem than
maintaining hand-ports of 7,000-line kernels.

---

## Recommended Porting Strategy

### Short-term (PoC validated path)

Write standalone NKI kernels that match the HF layer interface, informed by nki-library implementations but not importing them. Each kernel is:

```
kernels/neuron_<name>/
├── __init__.py      # kernel class + layers namespace + @nki.jit function
└── metadata.json    # {"backend": {"type": "neuron"}}
```

**Effort per kernel:** ~1-2 days for simple ops (RMSNorm, SiLU), ~1 week for complex (attention, MoE)

### Medium-term (recommended investment)

1. **Add unfused entry points to nki-library:**
   - `nkilib.ops.rmsnorm(x, weight, eps) → Tensor`
   - `nkilib.ops.rope(q, k, cos, sin) → (Tensor, Tensor)`
   - `nkilib.ops.silu_and_mul(x) → Tensor`
   
   These wrap the production kernels with sensible defaults (no quant, single-core, BF16).

2. **Get `nkilib` on HF `python-depends` allowed list:**
   - Then HF kernel wrappers become trivial one-liners
   - Contact HF (danieldk) about adding to the curated dependency set
   - **The ask is now concrete:** `kernels/python_depends.json` already has a
     `neuron` backend section whitelisting `nki`, so the precedent and the exact
     JSON shape to copy live in the same file. Adding `nkilib` alongside it is a
     four-line change, not a design discussion. (Finding #12)

3. **Fix `_backend()` so it reports `neuron` on Neuron hosts — highest leverage.**
   - Root cause is a single check: `hasattr(torch, "neuron")` is False, even after
     `import torch_neuronx`. So `torch_neuronx` setting that attribute is the fix.
   - This one change unblocks **two** independent things:
     build-variant resolution (Finding #7) and `python-depends` validation
     (Finding #12). Today a Neuron kernel must declare `python-depends: []` while
     importing `nki`, because the neuron allowlist table is never consulted.
   - Note it does NOT fix device *routing* for `use_kernels=True` — that needs the
     separate transformers change in Finding #9. Two distinct fixes; don't conflate.

### Long-term (full automation)

Once unfused entry points exist, generating HF kernel wrappers is mechanical:

```python
# Auto-generated wrapper template
class Neuron{OpName}(nn.Module):
    has_backward = False
    can_torch_compile = False
    {state_annotations}
    
    def forward(self, {args}):
        return nkilib.ops.{op_name}({mapped_args})

class layers:
    Neuron{OpName} = Neuron{OpName}
```

A script could walk `nkilib.ops.*` and generate these + metadata.json for each.

---

## Kernel-by-Kernel Porting Notes

### RMSNorm

| Aspect | Finding |
|--------|---------|
| nki-library source | `core/rmsnorm/rmsnorm_quant.py` (fused with FP8 quant) |
| Standalone available? | No — must extract `_rms_normalize_tile()` subroutine |
| Dependencies to inline | `RmsNormQuantKernelArgs`, `RMSNormQuantConstants`, `kernel_helpers` |
| Interface adaptation | `(hidden, ln_w, kargs)` → `self.weight`, `self.variance_epsilon` |
| SPMD handling | Strip — HF calls per-layer, not multi-core |
| What we actually used | Tutorial kernel (equivalent math, clean implementation) |
| Accuracy | Bit-identical to Qwen3RMSNorm on all tested shapes |

### RoPE — the easy case, and the one that reverses the RMSNorm narrative

| Aspect | Finding |
|--------|---------|
| nki-library source | `core/embeddings/rope_hf.py` — **standalone and already HF-shaped** |
| Standalone available? | **Yes.** Two variants: `rope_hf` (HF layout) and `RoPE` (Neuron-native layout `[d_head, B, heads, S]`) |
| Tutorial available? | **No.** Zero rotary examples anywhere in nki-samples. nki-library is the only source. |
| Dependencies to inline | Only 3 symbols: `kernel_assert`, `div_ceil`, `get_verified_program_sharding_info`. No `common_types`. ~15 lines total. |
| Interface adaptation | Small. Signature already takes precomputed cos/sin and returns a tuple. Main change: destination-passing → internal allocation. |
| SPMD handling | Stripped (`num_shards = 1`). Biggest semantic reduction of the port. |
| What we actually used | **The production kernel**, ported. Not a reimplementation. |
| Accuracy | 20/20 cases, cos_sim ≥ 0.999951, bit-identical (elementwise op, so expected) |
| Key constraint inherited | `seq_len % (128 × LNC) == 0`, 4D only, `unsqueeze_dim=1` |

**This case inverts the RMSNorm conclusion, and that matters for the recommendation.**
For RMSNorm the production kernel was unusable (fused with FP8 quant, no unfused path)
so we wrote one from the tutorial. For RoPE the production kernel is the *better*
starting point and no tutorial exists at all. So "nki-library kernels are too fused to
port" is not a general truth — it is per-kernel, and the `embeddings/` module shows the
library already contains HF-friendly code.

**Porting friction actually encountered:**

1. **Destination-passing vs return-value.** `rope_hf(q, k, q_out, k_out, ...)` requires
   preallocated outputs; HF returns a tuple. Resolved by allocating in
   `nl.shared_hbm` inside the kernel and returning both — verified multi-output
   `@nki.jit` works. Adds no cost and removes the mismatch entirely.
2. **The `seq_len % 128` constraint is the real limitation.** HF passes arbitrary
   sequence lengths, so a Python-level guard plus eager fallback is mandatory. This
   is the single most likely reason a customer silently gets no acceleration.
3. **No concat primitive in NKI.** `rotate_half`'s `torch.cat((-x2, x1), -1)` becomes
   writes into disjoint slices of a preallocated destination, with the negation folded
   into `op=nl.subtract`. Worth documenting as the canonical NKI idiom — it is not
   obvious, and nki-library's own implementation is the best reference for it.
4. **Function vs layer asymmetry.** Layer repos resolve `kernel.layers.<name>`;
   func repos resolve `<name>` at module top level. A function kernel placed inside
   the `layers` namespace will not be found. Also `has_backward` defaults to **True**
   for functions and **False** for layers, so it must be set explicitly.
5. **Import path is a non-issue here.** Both `nki` and `neuronxcc.nki` resolve on the
   DLAMI, so nki-library source can be copied with its imports intact.

**Docs bugs found in nki-library's public API reference:**
- `rope_hf` is absent from the API reference entirely — source-only.
- The reference cites `nkilib.core.rope.RoPE`; the real path is
  `nkilib.core.embeddings.rope.RoPE`. `nkilib/core/rope.py` does not exist.

### SiLU

| Aspect | Finding |
|--------|---------|
| nki-library source | Not needed — `nl.silu` is a native NKI primitive (also `nl.silu_dx`, `nl.gelu`, `nl.gelu_apprx_tanh`, `nl.gelu_dx`, `nl.erf`) |
| HF integration type | `LayerRepository`, kernel name `"SiLU"` |
| Effort | Hours. Tiled load / activate / store. |
| Accuracy | 9/9 cases, cos_sim ≥ 0.999993, fp32 max_diff 2.1e-06 |
| Leverage | The decoration lives in `activations.py`, so **one** kernel covers every model using `ACT2FN["silu"]` |
| **Performance caveat** | **Standalone elementwise SiLU is memory-bandwidth bound.** It reads N and writes N to do ~2 FLOPs each. Replacing it does not remove that traffic, and in eager mode adds a kernel launch plus an HBM round trip. Do not claim a win without measuring. |
| Where the real win is | The **fused** gate/up/SiLU/down MLP — see the MLP row below and Finding #17 |

The same pattern applies to `GeLU`, `GeluTanh`, `NewGELU`, `FastGELU`, `QuickGELU`: each is
a single decoration in `activations.py`, each is hours of work now that the pattern exists,
and each carries the same memory-bound caveat.

### MLP (fused gate/up/SiLU/down) — the one that matters for performance

| Aspect | Finding |
|--------|---------|
| nki-library source | `core/mlp/mlp.py`, entry point `nkilib.core.mlp.mlp.mlp` (40 params) |
| Standalone? | **Yes**, and quantization/normalization are **opt-in** (default `NONE`/`NO_NORM`) — unlike RMSNorm, no forced-quant path |
| Fuses | optional residual add + optional RMSNorm/LayerNorm + gate/up + activation + down. Could cover `post_attention_layernorm` + `Qwen3MLP` + residual in one call. |
| SPMD required? | **No.** Single-core supported (`program_ndim() in (0,1)`, explicit non-SPMD fallback) |
| Constraints | `H % 128 == 0`; no divisibility requirement on `I` or `BxS` on the non-quant path. Qwen3-8B (H=4096, I=12288) and Qwen3-0.6B both satisfy all of them. |
| Dependency closure | **7,249 lines / 22 files** if vendored (≈480x RoPE). Impractical to hand-port — argues for the thin-wrapper strategy. |
| Torch reference | Yes: `mlp_torch.py::mlp_torch_ref`, plus an input generator in `test/.../test_mlp_common.py` |
| HF integration type | **Fusion API**, not a layer decorator — `register_kernel_replacements_and_fusions` / `make_parent_class_for_kernel_fusion`. Note `"SwiGLUMLP"` exists in `_KERNEL_MAPPING` but **no model registers it**, so the decorator path is dead. |
| **Blocker** | **All three weights are transposed vs. HF's `nn.Linear` layout, and `kernelize()` has no parameter-transformation hook.** See Finding #17 — every workaround is bad. This blocks all fused-kernel ports, not just this one. |
| Returns | a `list`, always — wrapper must take `[0]` |
| Tolerance note | nki-library's own MLP tests use `rtol=2e-2`, far looser than our `cos_sim > 0.999` bar. Decide which applies before starting. |
| Effort | 1-2 day spike to validate standalone against `mlp_torch_ref`; 2-3 weeks to land via the fusion API |

### MoE (Week 5 candidate)

| Aspect | Finding |
|--------|---------|
| Source | `core/moe/moe_cte/moe_cte.py` (blockwise grouped matmul), `core/router_topk/router_topk.py` |
| Weight layout | **Good** match for `Llama4TextExperts` (free reshape, no transpose). **Poor** for `Qwen3MoeExperts` (`[E,2I,H]`/`[E,H,I]` → needs transposes). Opposite of the dense case. |
| Real work | Metadata, not matmul. `moe_cte` requires caller-supplied `token_position_to_id`, `block_to_expert`, `expert_affinities_masked`, and a `[T+1, H]` hidden tensor with a padding row. Token sorting and block assignment live *outside* the kernel; the megablocks path HF wraps builds them internally. That gap is the port. |
| Router | `router_topk` maps cleanly onto `Qwen3MoeTopKRouter` (T≤2048, E≤512, H%128==0, K≤8) but writes into caller-allocated mutable outputs, so needs an allocation wrapper. It is also the only file in this area importing `neuronxcc.nki` (for `nki.typing`) — relevant given Finding #14. |
| Naming caution | `"Llama4TextMoe"` is **commented out** in `_KERNEL_MAPPING` ("no longer maintained"). The only live MoE layer name is `"MegaBlocksMoeMLP"`. |

### Attention

| Aspect | Finding |
|--------|---------|
| nki-library source | `core/attention/` |
| Notes | Complex — Flash Attention style, multiple variants. Stretch goal. |

---

## Blockers for Mass Porting

| Blocker | Severity | Resolution Path |
|---------|----------|-----------------|
| nki-library kernels are fused (multi-op) | High | Add unfused entry points |
| `nkilib` not in HF `python-depends` allowed list | High | Negotiate with HF team |
| Internal deps can't be imported in HF kernels | High | Inline or use unfused API |
| No backward kernels for training mode | Medium | Start inference-only, add later |
| Variant resolver doesn't detect Neuron on DLAMI | Medium | Fix `torch.neuron` attr |
| No kernel-builder for Neuron | Low-Medium | Flat upload works as workaround |

---

## Cost Estimate (revised with Week 3 actuals)

Measured, not estimated, for the three we did:

| Kernel | Approach | Actual effort | Dependency closure |
|--------|----------|--------------|--------------------|
| RMSNorm | reimplement from tutorial (production kernel unusable) | ~1 day | n/a — written fresh |
| RoPE | **port production kernel** | ~1 day incl. tests | 3 symbols, ~15 lines inlined |
| SiLU | native `nl.silu` | ~2 hours | none |

Forward projections:

| Complexity | Example | Hand-port (self-contained) | Thin wrapper over installed `nkilib` |
|-----------|---------|---------------------------|--------------------------------------|
| Native primitive exists | SiLU, GELU family | hours | hours |
| Standalone + HF-shaped | RoPE | ~1 day | ~2 hours |
| Standalone, wrong weight layout | **MLP** | **impractical** (7,249-line closure) | 1-2 day spike, then blocked on Finding #17 |
| Fused with forced quant | RMSNorm | must reimplement | n/a — no unfused entry point |
| Metadata-heavy | MoE | weeks (metadata construction is the work) | weeks — the wrapper doesn't help here |

The thin-wrapper column is the argument for fixing the dependency allowlist rather than
funding hand-ports. But note the last row: for MoE the bottleneck is constructing routing
metadata outside the kernel, which no dependency change addresses.

---

## Open Questions

Updated after Week 3. Answered ones kept for the record.

**Answered:**
- *Can we publish flat (no variant dirs) to the Hub and have it load correctly?*
  Loading works — the flat layout hits `get_local_kernel`'s fallback path. `digest` is
  optional; minimum repo is `__init__.py` + `metadata.json`. Upload itself untested (out of
  scope; no external actions).
- *Should we PR neuron entries to `_KERNEL_MAPPING`, or keep as KernelConfig?*
  PR them — but the mapping entries alone are insufficient. Device resolution has to be
  fixed first (Finding #9), or the entries are unreachable.
- *Is `nkilib` usable as a dependency?* Technically yes, today: it is already installed and
  its kernels are directly callable (Finding #16). Blocked only by the `python-depends`
  allowlist.

**Still open:**
- **What is the weight-layout contract for fused kernels?** (Finding #17) Does `kernels`
  want a `prepare_weights`-style hook called once at kernelize time, with a defined
  contract about whether `state_dict()` reflects original or kernel layout? Or should
  Neuron kernels accept HF-native layouts and transpose internally? This blocks every
  fused-kernel port and is the single most consequential unanswered question.
- **Will HF add `nkilib` to `python_depends.json`?** `nki` is already there for the neuron
  backend, so the precedent and JSON shape exist. Needs the conversation.
- **Which NKI import path is supported long-term**, `nki` or `neuronxcc.nki`? They have
  different capabilities and neither is a superset (Finding #14). Kernel authors currently
  need a compatibility table that does not exist.
- **How should version coupling be managed** if we adopt thin wrappers? The HF kernel repo
  would depend on `nkilib`, which is validated against a specific `neuronx-cc`. nki-library's
  README explicitly warns `main` may not match your compiler.
- **Is inference-only acceptable for beta?** All three kernels are `has_backward=False`.
  nki-library's `rope_hf` has a backward path and `nl.silu_dx` exists, so backward kernels
  are feasible but unbuilt.
- **Hub repo home:** `kernels-community/` vs `aws-neuron/`. Blocks publishing and fixes the
  `repo_id` in the proposed upstream diff.
- *Is there appetite for a `kernel-builder init --backends neuron` command?* Lower priority
  than originally thought — NKI kernels are pure Python and need no build step, so the flat
  layout suffices.
