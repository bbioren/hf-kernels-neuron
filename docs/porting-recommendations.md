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

3. **Fix variant resolution for Neuron:**
   - `torch_neuronx` should set `torch.neuron = True` so auto-detection works
   - Or: `LocalLayerRepository.load()` should accept a `backend` override

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

### RoPE

| Aspect | Finding |
|--------|---------|
| nki-library source | TBD (Week 3) |
| HF integration type | `FuncRepository` (function replacement, not layer) |
| Notes | `apply_rotary_pos_emb` is a function decorated with `@use_kernel_forward_from_hub` |

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
