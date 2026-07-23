# HuggingFace Kernels on Neuron (NKI)

PoC project: package NKI kernels from `nki_samples` for the HuggingFace `kernels` library
and validate end-to-end on Qwen3 dense running on Trainium with `use_kernels=True`.

## Project Structure

```
kernels/                  # Local kernel repos (Hub-format layout)
  neuron_rmsnorm/         # NKI RMSNorm kernel (Week 2)
  neuron_rope/            # NKI RoPE kernel (Week 3)
  neuron_silu/            # NKI SiLU activation kernel (Week 4)
  neuron_identity/        # Minimal identity kernel for Week 1 PoC
scripts/                  # Dev scripts (install, test, validate)
tests/                    # Accuracy and integration tests
docs/                     # PoC doc, notes, writeups
notebooks/                # Exploration notebooks
```

## Quick Start (on trn2)

```bash
# Install deps
pip install -r requirements.txt

# Verify neuron device path works
python scripts/verify_neuron_path.py

# Run identity kernel swap demo (Week 1)
python scripts/demo_identity_swap.py
```

## Week 1 Goal

Prove the `"neuron"` device path in the `kernels` library works on Trainium:
1. `kernelize()` accepts `device="neuron"` and selects `_NeuronRepos`
2. A minimal stateless NKI kernel loads via `LocalLayerRepository`
3. The forward swap fires (confirmed by numerical signature)
4. Fallback works when mapping is absent

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
