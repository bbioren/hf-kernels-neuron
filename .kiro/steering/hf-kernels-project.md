# HuggingFace Kernels on Neuron — Project Context

## What this project is

A 6-week internship PoC: take NKI kernels from `aws-neuron/nki-library` (the production kernel library), package them for the HuggingFace `kernels` library (Kernel Hub), and validate that a stock HuggingFace model (Qwen3 dense) runs end-to-end on Trainium with `use_kernels=True` swapping in NKI-accelerated layers. If the dense flow lands cleanly, extend to Qwen3-MoE (router, top_k, blockwise MoE MLP). Final deliverable is a PoC document for the kernels team that captures what worked, what the Kernel Hub integration required, measured MFU impact, and a recommendation on whether Neuron should invest in first-class HF Kernel Hub support.

Ben's first half (the attention tutorial) already ramped him on NKI, the Neuron profiler, and the Native PyTorch beta setup. Week 1 is HF Kernels architecture ramp-up, not Trainium ramp-up.

## Why this project

HuggingFace's `kernels` library is a runtime kernel replacement system that swaps `nn.Module.forward()` methods with optimized implementations pulled from the Hub. It is now merged to transformers mainline. Adding a `"neuron"` device path to the kernel mapping gives every HuggingFace model with RMSNorm (87 model files), rotary embeddings (66 model files), and standard activations access to NKI kernels automatically, in eager mode, with graceful fallback when no Neuron kernel exists.

This is the highest-leverage HF ecosystem integration point for Neuron: per-kernel work that scales to the entire model zoo rather than per-model work. An intern PoC is the right vehicle to prove the mechanism end-to-end and hand the kernels team a validated path.

## Key concepts

- **KERNEL_MAPPING**: dict of `(layer_class_name, device) → kernel_impl`. Adding `"neuron"` entries gives all HF models with RMSNorm/RoPE/SiLU access to NKI kernels automatically.
- **Neuron device path**: the routing branch in `kernels` lib that selects `_NeuronRepos` when on Neuron hardware. Already merged to transformers mainline.
- **kernelize() flow**: walks model tree, matches layer names against KERNEL_MAPPING for current device, hot-swaps `forward()` method pointers. Module weights stay in place.
- **LocalLayerRepository**: local on-disk kernel repo for development without Hub publishing. Requires `__init__.py` + `metadata.json`.
- **KernelConfig(use_local_kernel=True)**: transformers-side API for local kernels. Format: `{"RMSNorm": "path/to/kernel:ClassName"}`.
- **Stateless kernel**: pure `nn.Module` subclass that reads weights from the adopting module via `self`. No `__init__`, only `forward()`. Declares `has_backward` / `can_torch_compile`.
- **Single-file kernel pattern (PR #46754)**: kernel class + `class layers:` namespace in one `__init__.py` file. This is the correct authoring pattern for Python-only kernels (NKI).
- **nki-library**: `aws-neuron/nki-library` — the production NKI kernel library. Source of kernels to port. Kernels are fused, have internal deps, and use different calling conventions than HF expects.

## Source of NKI kernels

**Use `aws-neuron/nki-library` (production library), NOT `nki-samples` (tutorial code).**

The PoC's value is documenting how to port production kernels at scale. Key findings so far:
- nki-library kernels are fused (e.g. RMSNorm+Quant combined) — no standalone ops
- Internal dependencies (`common_types`, `kernel_helpers`, `kernel_assert`) not allowed by HF
- Calling conventions differ (explicit args vs `self.weight`)
- SPMD multi-core assumptions don't fit the per-layer swap model
- See `docs/nki-library-porting-analysis.md` for full analysis

## Kernel authoring pattern

Single-file, per PR #46754:
```python
# kernels/neuron_rmsnorm/__init__.py
class NeuronRMSNorm(nn.Module):
    has_backward = False
    can_torch_compile = False
    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states):
        # NKI kernel call here
        ...

class layers:
    NeuronRMSNorm = NeuronRMSNorm
```

Plus `metadata.json` with `{"backend": {"type": "neuron"}}`.

## Target model: Qwen3 dense

- **RMSNorm**: `Qwen3RMSNorm` — manual implementation, reads `self.weight` and `self.variance_epsilon`
- **RoPE**: `apply_rotary_pos_emb` — a function, needs `FuncRepository` not `LayerRepository`
- **SiLU**: used in MLP gating layer, decorator name `"SiLU"` in transformers

## Accuracy targets

- Cosine similarity > 0.999 against reference layer output
- For e2e: logits parity vs CPU/CUDA golden reference

## Week-by-week plan

### Week 1: HF Kernels architecture ramp-up and neuron-path verification ✓ DONE
- Verified `kernelize(device="neuron")` works on trn2
- Confirmed `LocalLayerRepository` loads local kernel packages
- Proved forward swap fires + fallback works
- Confirmed `KernelConfig(use_local_kernel=True)` accepts neuron mapping

### Week 2: RMSNorm NKI kernel, local validation on Qwen3 dense ✓ DONE
- Ported NKI RMSNorm kernel (tutorial-derived, production analysis documented)
- Accuracy: cosine sim = 1.000000 (bit-identical) on isolated layers
- Accuracy: cosine sim = 0.999954 through full 2-layer Qwen3 model forward
- Documented nki-library porting friction (fusion, deps, interface mismatch)

### Week 3: Package RMSNorm for Hub, add RoPE, register neuron entries
- Move from local loading to Hub-style packaged repo
- Add RoPE NKI kernel as a `FuncRepository` entry
- Add `"neuron"` entries to `_KERNEL_MAPPING` for RMSNorm and RoPE
- Confirm `use_kernels=True` alone triggers the swaps on Neuron
- Coordinate with Samir on Hub repo home (`kernels-community/` vs `aws-neuron/`)

### Week 4: Activations, full Qwen3 dense end-to-end, MFU measurement
- SiLU NKI activation kernel + registration
- Full Qwen3 dense with `use_kernels=True` selecting NKI RMSNorm + RoPE + SiLU
- Confirm correctness (logits parity)
- Measure MFU with and without `use_kernels=True`

### Week 5: Qwen3-MoE (stretch)
- Map Qwen3-MoE forward to Kernel Hub layer names
- Reuse RMSNorm/RoPE/SiLU from weeks 2-4
- At least one MoE-specific NKI kernel swapped and validated, or gap analysis

### Week 6: PoC document, review, and ship
- Kernel Hub mechanism and why forward-swap is the correct interception point
- Upstream state: neuron device path merged, kernel-builder gap
- What was validated: kernels, models, accuracy, MFU delta
- What is not done: backward kernels, torch.compile, Hub upload, MoE gaps
- Recommendation: is first-class HF Kernel Hub support worth engineering investment?

## Definition of done

**Floor (must hit):**
- `"neuron"` device support working locally with forward-swap proven on Trainium
- At least NKI RMSNorm and RoPE packaged and validated e2e on Qwen3 dense
- Measured MFU delta with denominator stated
- PoC document delivered to the kernels team

**Ceiling (stretch):**
- SiLU + MLP activation kernels added
- Hub publishing working for a Neuron kernel
- At least one Qwen3-MoE kernel swapped and validated

## Environment (verified 2026-07-22)

| Package | Version |
|---------|---------|
| kernels | 0.15.2 (PyPI) |
| transformers | 5.15.0.dev0 (commit bb3ffb97) |
| torch | 2.9.1+cu128 (Neuron DLAMI) |
| neuronx-cc | 2.26.6360.0+6f180f47 |
| Instance | trn2.3xlarge (4 NeuronCores, 96 GB HBM) |
| Neuron venv | `/opt/aws_neuronx_venv_pytorch_2_9` |

## Documentation sources

| Source | What it covers |
|--------|---------------|
| [kernels docs — Layers](https://github.com/huggingface/kernels/blob/main/docs/source/layers.md) | kernelize(), use_kernel_mapping, LocalLayerRepository |
| [kernels docs — Requirements](https://huggingface.co/docs/kernels/kernel-requirements) | metadata.json schema, backend types, build variants |
| [transformers PR #46754](https://github.com/huggingface/transformers/pull/46754/files) | "Writing kernels" doc — single-file pattern, KernelConfig |
| [NKI Tutorial — RMSNorm](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/general/nki/tutorials/rmsnorm.html) | Reference NKI kernel implementation |
| [nki-library GitHub](https://github.com/aws-neuron/nki-library) | Production kernel source (rmsnorm, attention, mlp, moe, etc.) |

## Coordination

- **Samir (arsamir)**: HF kernels team contact, Hub repo home decision
- **Pinak (panpinak)**: SA team reviewer
- **Hanbo Wang / Karthick Gopalswamy**: kernels team (PoC recipients)
- **Matt (mmcclean)**: final deliverable recipient

## Tracking Documents — UPDATE THESE THROUGHOUT

These docs accumulate findings that become the final PoC. Update them as you work, not just at the end.

| Document | Purpose | When to update |
|----------|---------|----------------|
| `docs/sticking-points.md` | Running log of things that blocked or slowed progress | Every time something takes >10 min to debug or is harder than expected |
| `docs/customer-experience.md` | What a customer would struggle with today | When you hit setup friction, unclear errors, missing docs, or workflow gaps |
| `docs/porting-recommendations.md` | How the engineering team should port kernels at scale | When you learn something about nki-library structure, HF requirements, or automation opportunities |
| `docs/poc-findings.md` | Technical findings with severity ratings | When you discover a gap, API issue, or architectural mismatch |
| `docs/nki-library-porting-analysis.md` | Deep analysis of nki-library kernel structure | When you investigate a new kernel from nki-library |
| `deliverables/week-N.md` | Weekly deliverable writeup | End of each week |

## What to Always Be Tracking

1. **Sticking points**: anything that took longer than expected, would trip up a customer, or reveals a systemic gap. Log it with time lost + who it affects.
2. **Customer experience**: imagine someone just did `pip install transformers` and wants NKI kernels. What's missing? What errors do they hit? What's underdocumented?
3. **Porting friction**: for each nki-library kernel you look at, note: is it fused? what deps does it pull? does the interface match HF? what would automation need?
4. **Recommendations**: concrete suggestions for the engineering team. Not just "this is hard" but "here's what to build/change to make it easy."
5. **Accuracy results**: always record cosine similarity, max abs diff, shapes tested, and whether NKI or fallback was used.
