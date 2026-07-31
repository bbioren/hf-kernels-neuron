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

**Every `@nki.jit` invocation forks a subprocess. Caching it is one decorator and worth 102x.**

`nki/framework/compiled.py::_compile_opts()` calls `resolve_target()` on every invocation, which
falls through to `_detect_target()`, which runs `neuron-ls` to ask the hardware what it is — ~52 ms
per kernel call. It sits *outside* `_nki_compile_cache` because its result is part of the cache key,
so a cache **hit** still pays it in full. Finding #24.

| Qwen3-0.6B, 28 layers, forward only, 1 logical core | step time | MFU | penalty |
|---|---|---|---|
| baseline, seq 512 | 42.04 ms | 5.05% | — |
| kernelized, before the fix | 8753.65 ms | 0.02% | 208x |
| kernelized, after the fix | 141.43 ms | 1.50% | 3.36x |
| kernelized, after the fix, seq 2048 | 223.99 ms | 4.81% | **2.06x** |

Denominator: 632 TFLOPS/device TensorEngine ÷ 2 for LNC2 = 316 TFLOPS. Verified two ways
(`NEURON_PLATFORM_TARGET_OVERRIDE` and `lru_cache`), baseline re-run last as a control, cosine
similarity identical to six decimals across all variants. **Not Kernel Hub specific** — any eager
per-layer NKI use pays this today.

**But the kernels cannot win even with dispatch free**, and that is the real headline. Measured on
device with dispatch excluded (Finding #25):

| | device ms (N=28) | HBM traffic | marginal traffic/call | MBU |
|---|---|---|---|---|
| NKI SiLU | 0.607 | 188.7 MB | 6.29 MB = **1.00x** the unfused floor | 43.2% |
| torch SiLU | **0.224** | **6.3 MB** | **~0.00 MB** | 3.9% |
| NKI RMSNorm | 1.625 | 188.8 MB | 6.29 MB = **1.00x** the floor | 16.2% |
| torch RMSNorm | **0.637** | **6.4 MB** | **~0.00 MB** | 1.4% |

The kernels are *optimal* — marginal traffic is exactly one read in and one write out, the minimum
for an op that cannot fuse. Torch's traffic is independent of N, which is only possible if the chain
fused into one pass. So the 2.5–2.7x gap is entirely the **fusion barrier**: a NKI custom call is
opaque to the compiler, and each swap forces a HBM round-trip where the data previously stayed
resident across a fused region.

For memory-bound ops, fusion *is* the optimisation, so a NKI kernel is competing against not touching
memory at all. **Break-even is unreachable for these ops, not merely distant.** The uncomfortable
corollary: the ops the Kernel Hub is best at intercepting — RMSNorm (115 registrations), RoPE (95
model files), all of `ACT2FN` via one decoration — are precisely the ops that lose most from being
intercepted.

### Known blockers

| Blocker | Effect |
|---|---|
| **Compiler cannot fuse across a NKI custom call (#25)** | **2.5–2.7x on device for memory-bound ops, independent of dispatch cost and of kernel quality. The binding constraint — no plumbing work fixes it.** |
| **`_detect_target()` forks `neuron-ls` per invocation (#24)** | **~52 ms/call. One decorator fixes it; 102x verified. Highest value-to-effort item, and correct regardless of #25.** |
| `create_computation` rebuilt per invocation (#24) | ~0.59 ms/call residual. Attributed, not fixed. Demoted by #25 — closing it still leaves the device deficit. |
| `use_kernels=True` can't reach `"neuron"` (#9) | silent no-op. Use `kernelize_for_neuron()`; a verified ~3-line upstream fix is in `scripts/neuron_kernel_registration.py` |
| `torch_neuronx` op overrides aren't fake-tensor safe (#23) | breaks `torch.compile` on nearly any transformer (`Embedding`, `Softmax`, `CrossEntropyLoss`, …). `torch_xla.compile()` works around it. Unrelated to this integration. |
| Fused MLP won't compile single-core above `intermediate_size` 4096 (#18) | excludes every real model |
| Qwen3-MoE needs `experts_implementation="batched_mm"` (#22) | undocumented; default fails with an unsupported `sort` HLO |

Retracted, so none of it gets quoted. Earlier versions of this README said, in order: that the
slowdown was structural graph-transition cost; that `torch.compile` is broken on this stack; that the
decisive open question was whether graph mode amortises the cost; and that fixing the dispatch
residual was the difference between 3.4x slower and near parity. All four are wrong.
`torch.compile` works for ops `torch_neuronx` hasn't overridden (#23). Graph mode was never the
lever — 28 NKI calls already fuse into one HLO graph and one device execution and still cost 28x.
And the dispatch residual is no longer decisive, because #25 shows a 2.5–2.7x device deficit survives
closing it.

The one question that would change the conclusion: **can a NKI custom call participate in compiler
fusion?** If yes, #25 dissolves. If no, per-layer swapping of small memory-bound ops is closed on
merit.

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
