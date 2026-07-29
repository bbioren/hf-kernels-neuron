# Porting Recommendations for Engineering Team

How should the Neuron kernels team approach porting nki-library kernels to the HF Kernel Hub at scale? This doc accumulates recommendations as we learn from the PoC.

---

## Executive Summary

(To be written in Week 6 from accumulated findings below)

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
| nki-library source | Likely in `core/mlp/` (fused with gate/up projections) |
| HF integration type | `LayerRepository` (layer replacement) |
| Notes | SiLU used in Qwen3/Llama/Mistral MLP gating |

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

## Cost Estimate (per kernel, current state)

| Complexity | Example | Time Estimate | Notes |
|-----------|---------|---------------|-------|
| Simple (unfused, matches HF interface) | RMSNorm, SiLU | 1-2 days | Write standalone from scratch |
| Medium (needs interface adaptation) | RoPE | 2-3 days | Function vs layer, arg mapping |
| Complex (fused, needs decomposition) | Attention, MLP | 1-2 weeks | Extract subkernels, test thoroughly |
| Very complex (multi-core, MoE) | MoE block | 2-4 weeks | SPMD, routing, expert dispatch |

---

## Open Questions

- Can we publish flat (no variant dirs) to the Hub and have it load correctly?
- Will HF add `nkilib` to python-depends, or do we need a separate mechanism?
- Should we PR neuron entries to `_KERNEL_MAPPING` in transformers, or keep as KernelConfig?
- Is there appetite for a `kernel-builder init --backends neuron` command?
