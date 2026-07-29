# Customer Experience Report

What would a customer struggle with if they tried to use NKI kernels via the HF Kernel Hub today? This doc tracks friction from the perspective of someone who just wants `use_kernels=True` to work on Trainium.

---

## Setup Friction

| Issue | Severity | Notes |
|-------|----------|-------|
| Torch lives in `/opt/` venv, not system-wide | Medium | Must `source activate` the right venv. No `pip install torch` on DLAMI. |
| Ubuntu 24.04 blocks system pip (PEP 668) | Low | Venv required, but expected for modern Python |
| `torch_neuronx` import triggers runtime init | Medium | Fails if helper binaries not on PATH. Must use full DLAMI venv, not .pth hack. |
| ~15 min from fresh instance to working | Low | Acceptable for devs, but worse than GPU (just `pip install` and go) |
| No HF_TOKEN configured by default on DLAMI | Low | Rate-limited Hub access, warning messages |

## API / Integration Friction

| Issue | Severity | Notes |
|-------|----------|-------|
| `use_kernels=True` cannot select the neuron path at all | **Critical** | transformers' `kernelize(model, mode)` has no `device` arg; it reads `model.device.type`, which is `"cpu"` or `"xla"` on Neuron — never `"neuron"`. See Finding #9. |
| `"xla"` is not a supported device type in `kernels` | **Critical** | Kernelizing a model that has been moved to a Neuron device raises `Unsupported device type 'xla'`. So the correct way to run *breaks*, and the incorrect way (params on host) silently no-ops. |
| Customer must call the `kernels` library directly, bypassing transformers | High | `kernelize(model, device="neuron", mode=Mode.INFERENCE)` works, but it is not the documented transformers entry point and skips `KernelConfig` handling. |
| No way to ask "is my kernel actually active?" | High | Nothing reports which implementation is live. Combined with silent fallback, a customer cannot tell acceleration from no-op. |
| Function kernels swap process-globally | Medium | Kernelizing one model changes `apply_rotary_pos_emb` for every model in the process. Surprising for multi-model serving. |
| `@nki.jit` hard-errors on CPU tensors | Medium | Forces every kernel to carry a device guard, which is what creates the silent-fallback trap. |

## Silent Failure Modes (highest-risk category)

These are the issues where the customer gets **no error and no warning**, and would
reasonably believe things are working.

| Failure | What the customer sees | What's actually happening |
|---------|------------------------|---------------------------|
| Kernel falls back on host tensors | Correct numbers, no warning. Our own test even printed "Backend: NKI kernel" | Eager PyTorch. Zero NKI execution. Cost us a week of false confidence — see Finding #8. |
| `"neuron"` mapping ignored on a `cpu`-device model | `use_kernels=True` returns successfully | Mapping lookup misses; original forward retained |
| Accuracy test passes with `max_diff = 0.00e+00` | "Bit-identical, great" | Both sides ran the same PyTorch code. For a hardware kernel, a perfect match means the kernel didn't run. |

**The general lesson for the PoC:** on Neuron, the dangerous outcome is not a crash,
it's a no-op that looks like success. Any customer-facing story for NKI kernels on the
Hub needs an affirmative "this kernel is live on this layer" signal. Numerical
correctness alone cannot distinguish acceleration from fallback, because the fallback is
*also* numerically correct.

## Documentation Gaps

| Gap | Impact | Notes |
|-----|--------|-------|
| No "Hello World kernel" tutorial for authors | High | Had to piece together from PR, spec, and trial-and-error |
| `LocalLayerRepository` docs show removed `package_name` arg | Medium | TypeError on first try |
| `metadata.json` fields underdocumented for local dev | Medium | Required even for local testing, not obvious |
| No docs on Neuron-specific kernel authoring | High | Nothing tells you how to do this for Neuron specifically |
| `kernelize()` docstring omits `neuron` from supported devices | Medium | Lists "cuda", "mps", "npu", "rocm", "xpu". Both `neuron` and `cpu` are supported in code. A reader would conclude Neuron isn't. |
| No documented difference between `nki` and `neuronxcc.nki` | **High** | They have different capabilities and neither is a superset (Finding #14). Nothing says which is supported, or that `hasattr` lies about `nl.arange`. |
| `nki-library`'s `rope_hf` absent from the public API reference | Medium | The best HF-shaped kernel in the library is source-only. The reference also cites a non-existent import path (`nkilib.core.rope` vs real `nkilib.core.embeddings.rope`). |
| No guidance that layer vs function kernels resolve differently | Medium | Layer repos look in `kernel.layers.<name>`; func repos look at module top level. Getting it wrong yields a confusing "not found". |

## Runtime Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| `@nki.jit` requires XLA tensors, errors on CPU | Medium | `RuntimeError: Expected all tensors ... to be XLA tensors`. Forces a device guard in every kernel, which is what creates the silent-fallback trap. |
| A Neuron kernel cannot declare `python-depends: ["nki"]` | High | HF whitelists `nki` for the neuron backend, but `_backend()` reports cuda on the DLAMI so the entry is unreachable. Kernels must under-declare to load (Finding #12). |
| Kernel constraints silently disable acceleration | High | RoPE needs `seq_len % 128 == 0`; HF passes arbitrary lengths. Without an explicit warning the customer just gets eager speed. We added `warn_once`; upstream kernels generally don't. |
| Per-kernel NKI import path pinning | High | Some kernels only compile under `neuronxcc.nki`, others only under top-level `nki`. A multi-kernel repo needs both, discovered at compile time. |

## What a Customer Would Need to Do Today

1. Install `kernels` from PyPI (pinned to a minor: `>=0.15,<0.16`)
2. Install `transformers` from main (neuron path not in a tagged release yet)
3. Activate the DLAMI Neuron venv (`source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate`)
4. Clone the kernel repo locally (no Hub publishing yet)
5. **Bypass transformers entirely.** `use_kernels=True` cannot reach the neuron path
   (Finding #9), so call the `kernels` library directly with an explicit device:
   ```python
   from kernels import kernelize, Mode, use_kernel_mapping
   with use_kernel_mapping(mapping, inherit_mapping=False):
       kernelize(model, device="neuron", mode=Mode.INFERENCE)
   ```
6. **Manually attach function kernels.** For RoPE, replicate the `_hidden_kernels`
   attach/detach that transformers' wrapper does, or the function swap won't be found.
7. Move the model to the Neuron device *before* kernelizing, and know that leaving it on
   the host means the kernels silently don't run.
8. Verify the kernels actually ran — nothing reports it, and correct output does not
   imply acceleration.

Steps 5–8 are all consequences of gaps we found this week. None of them are documented
anywhere, and step 8 has no supported mechanism at all.

## What "Just Works" Would Look Like

```python
from transformers import AutoModelForCausalLM

# This is the dream. No config, no local files, no device override.
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    use_kernels=True,
    device_map="neuron",
)
# NKI kernels downloaded from Hub, swapped in, model runs accelerated.
```

## Gaps Between Current State and "Just Works"

Ordered by how much they block the "just works" experience.

| # | Gap | Owner | Effort | Notes |
|---|-----|-------|--------|-------|
| 1 | `use_kernels=True` can't select the neuron device | **transformers** | ~5 lines, 3 sites | Map `xla`→`neuron` when `xla_device_hw()=="NEURON"`. **Verified sufficient**: takes Qwen3 from 0→9 swapped layers. Sites: `hub_kernels.kernelize`, `kernel_config.infer_device`, `kernels._find_device`. |
| 2 | `_backend()` reports cuda on Neuron hosts | **torch_neuronx** | 1 attribute | Set `torch.neuron`. Unblocks build-variant resolution *and* `python-depends: ["nki"]` at once (Findings #7, #12). Does NOT fix #1. |
| 3 | No way to verify a kernel is live | HF `kernels` | small | Silent fallback + no reporting means acceleration is indistinguishable from a no-op. Biggest trust problem. |
| 4 | Neuron kernels not published on Hub | Neuron/HF | — | Flat layout works (no kernel-builder needed). Blocked on repo-home decision + gap 2 for honest deps. |
| 5 | `nkilib` not on the `python-depends` allowlist | HF `kernels` | 4 lines | Precedent and exact JSON shape already there for `nki`. Prerequisite for thin-wrapper porting. |
| 6 | `nki` vs `neuronxcc.nki` capability split | NKI team | needs a decision | Neither is a superset; kernels are pinned per-idiom (Finding #14). |
| 7 | `device_map="neuron"` doesn't exist | transformers | larger | Would make the "dream" snippet work as written. |
| 8 | transformers neuron path not in a tagged release | HF releases | — | Requires install from main. |

**The good news, and it is real:** gaps 1 and 2 are both small, well-understood, and
between them unblock most of the experience. The interception points already exist
upstream, Qwen3 already opts into them, and coverage is large (115 RMSNorm, 95 RoPE).
Nothing here requires architectural change — which is the central input to the
"is this worth investing in" question.
