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
| `kernels` | TBD | pin after first trn2 run |
| `transformers` | TBD | pin after first trn2 run |
| `torch-neuronx` | TBD | SDK version on host |
| `neuronx-cc` | TBD | compiler version on host |

## Links

- [HF Kernels docs](https://huggingface.co/docs/kernels/index)
- [HF Kernels GitHub](https://github.com/huggingface/kernels)
- [Project doc (internal)](https://quip-amazon.com/rabVAe6SgkUb)
- [transformers kernels integration PR](https://github.com/huggingface/transformers/pull/46754)
