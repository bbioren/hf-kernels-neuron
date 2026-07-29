# HuggingFace Kernels on Neuron (NKI)

PoC project: package NKI kernels from `nki_samples` for the HuggingFace `kernels` library
and validate end-to-end on Qwen3 dense running on Trainium with `use_kernels=True`.

## Project Structure

```
kernels/                  # Local kernel repos (Hub-format layout)
  neuron_rmsnorm/         # NKI RMSNorm kernel      (layer swap,    Week 2)
  neuron_rope/            # NKI RoPE kernel         (function swap, Week 3)
  neuron_silu/            # NKI SiLU activation     (layer swap,    Week 3)
  neuron_identity/        # Minimal identity kernel for the Week 1 PoC
scripts/                  # Dev scripts, investigation probes, registration
tests/                    # Accuracy and integration tests (must run on trn2)
docs/                     # Findings, sticking points, porting analysis
deliverables/             # Weekly writeups
```

## Quick Start (on trn2)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install -r requirements.txt

# Per-kernel accuracy suites (RMSNorm, RoPE, SiLU)
make test-nki

# Qwen3 end-to-end kernel swap
make test-e2e

# Investigation probes (device path, NKI execution, API surface, packaging)
make probe

# Print the neuron kernel mapping + the proposed upstream diff
make registration
```

Developing locally? `make sync` rsyncs the tree to trn2 — tests must run there.

## Important: tests must run on Trainium

`@nki.jit` requires XLA tensors. A kernel handed CPU tensors takes its PyTorch
fallback, produces numerically *correct* output, and reports nothing. In Week 2 this
caused an entire test suite to pass while never executing a single NKI instruction.

So every accuracy test here calls `require_neuron()` (which refuses to report results
unless `xla_device_hw() == "NEURON"`) and asserts via a call counter that the NKI branch
ran and the fallback did not. **For a hardware kernel, a bit-identical result against a
PyTorch reference is evidence of failure, not success** — unless the op is elementwise.
See `docs/poc-findings.md` Finding #8.

## Current status

| Kernel | Interception point | Registrations upstream | Accuracy (on hardware) |
|--------|-------------------|------------------------|------------------------|
| RMSNorm | `RMSNorm` (layer) | 115 | 11/11 pass, NKI verified |
| RoPE | `rotary_pos_emb` (func) | 95 model files | 20/20 pass, NKI verified |
| SiLU | `SiLU` (layer) | 1 (covers all `ACT2FN["silu"]` users) | 9/9 pass, NKI verified |

End-to-end on Qwen3: all three execute (RMSNorm 9×, RoPE 2×, SiLU 2× per forward),
logits `cos_sim 1.000001` vs the unkernelized model.

**Known blocker:** `use_kernels=True` cannot reach the `"neuron"` device path
(Finding #9). Use the `kernels` library directly with `device="neuron"`, or see
`scripts/neuron_kernel_registration.py` for the verified minimal upstream fix.

## Versions Tested

| Package | Version | Notes |
|---------|---------|-------|
| `kernels` | 0.15.2 | from PyPI |
| `transformers` | 5.15.0.dev0 | from main, commit `bb3ffb97` |
| `torch` | 2.9.1+cu128 | from Neuron DLAMI |
| `torch_neuronx` | available | from Neuron DLAMI |
| `neuronx-cc` | 2.26.6360.0+6f180f47 | from Neuron DLAMI |
| Python | 3.12.3 | Ubuntu 24.04 |
| Instance | trn2.3xlarge | 1 device, 4 NeuronCores, 96 GB HBM |

## Links

- [HF Kernels docs](https://huggingface.co/docs/kernels/index)
- [HF Kernels GitHub](https://github.com/huggingface/kernels)
- [Project doc (internal)](https://quip-amazon.com/rabVAe6SgkUb)
- [transformers kernels integration PR](https://github.com/huggingface/transformers/pull/46754)
