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
| | | |

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
