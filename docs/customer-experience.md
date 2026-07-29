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

## Runtime Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| | | |

## What a Customer Would Need to Do Today

1. Install `kernels` from PyPI (pinned to a minor: `>=0.15,<0.16`)
2. Install `transformers` from main (neuron path not in a tagged release yet)
3. Activate the DLAMI Neuron venv (`source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate`)
4. Clone the kernel repo locally (no Hub publishing yet)
5. Use `KernelConfig(use_local_kernel=True)` with the path
6. Know to pass `device="neuron"` explicitly (auto-detection reports CUDA on DLAMI)

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

| Gap | Owner | Notes |
|-----|-------|-------|
| Neuron kernels not published on Hub | Neuron/HF | No kernel-builder variant, but flat upload might work |
| `device_map="neuron"` doesn't exist in transformers | Transformers team | Would need integration work |
| Auto-detection doesn't find Neuron (`torch.neuron` missing) | torch_neuronx team | `hasattr(torch, "neuron")` returns False |
| No `_KERNEL_MAPPING` neuron entries in transformers | This PoC (Week 3) | Need to PR or use KernelConfig |
| `transformers` neuron path not in tagged release | HF releases | Currently requires install from main |
