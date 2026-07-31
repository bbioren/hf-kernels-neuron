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

**PoC complete.** Final deliverable: [`deliverables/poc-document.md`](deliverables/poc-document.md).

| Kernel | Interception point | Registrations upstream | Accuracy (on hardware) |
|--------|-------------------|------------------------|------------------------|
| RMSNorm | `RMSNorm` (layer) | 115 | 11/11 pass, NKI verified |
| RoPE | `rotary_pos_emb` (func) | 95 model files | 20/20 + 6/6 guards, NKI verified |
| SiLU | `SiLU` (layer) | 1 (covers all `ACT2FN["silu"]` users) | 9/9 pass, NKI verified |

End-to-end: all three execute on **Qwen3 dense** (logits `cos_sim 1.000001`) and on
**Qwen3-MoE** with zero code changes (`cos_sim 1.000002`).

### The headline result

**The kernels are correct and 208x slower.** MFU 5.06% → 0.02%.

Every `@nki.jit` invocation from eager PyTorch/XLA costs **~53 ms of fixed overhead regardless of
problem size** — more than the entire 42 ms baseline forward pass. At 169 kernel calls per step
that dominates everything. It is an integration-model result, not a kernel-quality one: the
Kernel Hub wants many small invocations, NKI charges ~53 ms each, and nki-library's kernels are
built as a few large fused megakernels. See Finding #20.

The decisive follow-up — does graph mode amortize the cost? — **could not be answered here**,
because `torch.compile` doesn't work on this stack even for plain PyTorch (Finding #21). That is
the single most valuable remaining experiment.

### Known blockers

| Blocker | Effect |
|---|---|
| `use_kernels=True` can't reach `"neuron"` (#9) | silent no-op. Use `kernelize_for_neuron()`; a verified ~3-line upstream fix is in `scripts/neuron_kernel_registration.py` |
| ~53 ms per NKI invocation (#20) | eager per-layer swap is not performance-viable |
| `torch.compile` broken on this stack (#21) | blocks the decisive experiment |
| Fused MLP won't compile single-core above `intermediate_size` 4096 (#18) | excludes every real model |
| Qwen3-MoE needs `experts_implementation="batched_mm"` (#22) | undocumented; default fails with an unsupported `sort` HLO |

All findings with severity: [`docs/poc-findings.md`](docs/poc-findings.md).
Upstream asks with patches: [`docs/upstream-fixes.md`](docs/upstream-fixes.md).

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
