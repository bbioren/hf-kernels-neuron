# Week 4 Deliverable — MFU measurement

**Date:** 2026-07-31
**Hardware:** trn2.3xlarge, 1 Neuron device, LNC2 (4 physical cores → 2 logical), single logical core used
**Versions:** `kernels 0.15.2`, `transformers 5.15.0.dev0`, `torch 2.9.1+cu128`, `neuronx-cc 2.26.6360.0`, `nki 0.5.0`

---

## Headline

**The kernels make the model 208x slower.** MFU goes from 5.06% to 0.02%.

The cause is not kernel quality. Every `@nki.jit` invocation from eager PyTorch/XLA carries
~53 ms of fixed overhead, independent of problem size — larger than the entire 42 ms baseline
forward pass. At 169 kernel calls per step, that dominates everything else by two orders of
magnitude.

This is Finding #20, and it is the most consequential result of the project.

---

## The measurement

Qwen3-0.6B, **full 28 layers**, seq 512, batch 1, bf16, forward only, single logical core.

| Configuration | Step time | Throughput | Achieved | MFU (per core) | NKI calls/step |
|---|---|---|---|---|---|
| baseline, no kernels | **41.95 ms** | 12,204 tok/s | 15.98 TFLOPS | **5.06 %** | 0 |
| NKI SiLU only | 1,495.54 ms | 342 tok/s | 0.45 TFLOPS | 0.14 % | 28 |
| NKI RMSNorm + RoPE + SiLU | **8,753.65 ms** | 58 tok/s | 0.08 TFLOPS | **0.02 %** | 169 |

Steady state, not a compilation artifact: **zero compilations during the timed loop**, and step
time stable to within 0.2% (IQR 8746–8764 ms).

All kernels confirmed engaged via call counters — RMSNorm 113, RoPE 28, SiLU 28, zero
fallbacks. So this measures what it claims to.

### Denominator, stated explicitly

MFU numbers are meaningless without this, and Trn2 has two conventions plus an LNC subtlety.

| Quantity | Value |
|---|---|
| Trn2 peak bf16, TensorEngine only | 632 TFLOPS/device |
| Trn2 peak bf16, published (incl. Vector + Scalar engines) | 667 TFLOPS/device |
| Logical cores per device (LNC2) | 2 |
| Logical cores used by an eager per-layer swap | **1** |
| **Primary denominator used here** | **316 TFLOPS** (632 ÷ 2) |

An eager per-layer swap runs on one logical core, i.e. half the device. Quoting the per-device
632 for a single-core run would halve the reported MFU; quoting per-core as though it were a
device figure would double it. Both are printed by the script so either convention is
recoverable.

### FLOP count, auditable

670.42 GFLOP per forward step, computed explicitly rather than from a rule of thumb:

| Term | GFLOP | Share |
|------|-------|-------|
| MLP (gate + up + down) | 270.58 | 40.4 % |
| LM head | 159.32 | 23.8 % |
| QKV projections | 120.26 | 17.9 % |
| O projection | 60.13 | 9.0 % |
| attention QK^T | 30.06 | 4.5 % |
| attention AV | 30.06 | 4.5 % |

Attention terms are counted without a causal-mask discount, which is what an eager
implementation actually executes. Applying the usual ~2x discount would *raise* reported MFU;
we do not, and say so.

---

## Root cause: fixed cost per invocation

Rather than report "208x slower", the cost was attributed. Per-call added cost is **51.9 ms**
(SiLU only) and **51.6 ms** (all three) — uniform, which already suggests a fixed charge rather
than anything kernel-specific.

Confirmed by sweeping problem size for a single call (`scripts/experiment_nki_graph_break.py`):

| rows | tiles | NKI | torch `F.silu` |
|------|-------|-----|----------------|
| 128 | 1 | 54.57 ms | 0.250 ms |
| 256 | 2 | 53.38 ms | 0.250 ms |
| 512 | 4 | 52.72 ms | 0.269 ms |
| 1024 | 8 | 53.52 ms | 0.588 ms |
| 4096 | 32 | 53.86 ms | 0.304 ms |
| 14336 | 112 | 53.75 ms | 0.501 ms |

**52.7–54.6 ms across a 112x range in problem size.** Flat. And one call on 28x the data costs
1.02x one call on 1x the data.

Alternatives ruled out:

| Hypothesis | Test | Result |
|---|---|---|
| Interleaving forces graph breaks | 28 adjacent NKI calls vs 28 separated by torch ops | **equal** (51.85 vs 52.79 ms/call) — not the cause |
| Host-side dispatch | Finding #19 measured enqueue cost | 0.36 ms/call, **1/145** of the observed cost |
| Our kernels are bad | production nki-library `rope_hf` in the same run | **same ~52 ms** |
| Recompilation per step | compile events during timed loop | **zero** |
| Host-side sync artifact | variant A has a single `mark_step` for 28 calls | cost is **inside one graph execution** — device-side |

Reproduced four times across independent measurements: 51.9, 51.6, 51.85, 52.09 ms/call.
Stable to within 1%.

---

## What this means

**The arithmetic is brutal and simple.** At ~53 ms per invocation, one NKI call costs more than
the entire baseline forward pass (42 ms). So in eager mode on this stack, *any* per-layer NKI
swap loses, and swapping more layers loses harder. Even a perfectly fused one-call-per-layer
kernel would cost 28 × 53 ms = 1.5 s/step against a 42 ms baseline.

**It is an integration-model result, not a kernel result.** nki-library's kernels are designed
as large fused megakernels that amortize invocation cost across a whole transformer block. The
HF Kernel Hub's per-layer forward swap is the opposite shape — it maximizes invocation count.
Findings #17 (weight layout) and #18 (single-core width limits) found the same mismatch from
the weight and sharding directions; this is the same conclusion measured in time.

**The prediction I made in Week 3 was right about SiLU and wrong about why.** I predicted
RMSNorm and RoPE would help and standalone SiLU would not, on memory-bandwidth grounds. In
fact none of them help, and bandwidth has nothing to do with it — the cost is invocation
overhead that swamps both compute and memory traffic. Worth recording: the reasoning was
plausible and the conclusion was still wrong, which is why the measurement was necessary.

---

## What would change the answer

**Graph mode is now the decisive question**, and it could not be answered here.

If the ~53 ms is a per-invocation framework-boundary cost, compiling the model should amortize
it — the kernels become part of one graph entered once per step instead of 169 times. If it is
intrinsic to executing a NKI NEFF, compilation will not help.

`scripts/experiment_torch_compile_nki.py` tried to settle it and **could not**, because
`torch.compile` fails on this stack for **plain PyTorch** — `F.silu` with no NKI anywhere fails
identically across `openxla`, `inductor`, and `eager` backends in both bf16 and fp32
(`Dynamo failed to run FX node with fake tensors`). A NKI failure would be indistinguishable
from compilation being broken generally, so the experiment refuses to report a NKI result and
says so. See Finding #21.

Answering it needs either a stack where `torch.compile` works on Neuron (the Native PyTorch
beta's compile path) or direct guidance from the NKI / torch-neuronx teams on the supported way
to invoke a NKI kernel from a compiled graph.

**Second thing worth checking:** ~53 ms is large enough to look like a misconfiguration rather
than a design point. Worth asking whether it is expected on SDK 2.31 / NKI 0.5.0 via
torch-xla eager, before treating it as a fundamental property.

---

## Week 4 goals vs outcome

| Goal | Status |
|---|---|
| Measure MFU with and without kernels, denominator stated | **Done.** 5.06% → 0.02%, denominator explicit, FLOP count auditable |
| Report launch count alongside MFU (per Finding #19) | **Done**, and it turned out to be the whole story |
| Full-size model rather than the 2-layer stand-in | **Done** — Qwen3-0.6B at full 28 layers |
| Confirm the RoPE `seq_len % 128` guard doesn't silently disable the kernel | **Done** — seq 512, RoPE engaged 28/28, zero fallbacks |
| SiLU kernel + registration | Done in Week 3 |
| Test the prediction that SiLU wouldn't help | **Done** — and the prediction's *reasoning* was wrong; see above |

---

## What did not change

The correctness results stand independently and are unaffected:

- All three kernels numerically correct and execution-verified, with negative controls
- Qwen3 dense e2e: logits `cos_sim 1.000001`
- Qwen3-MoE: all three kernels transfer unchanged, logits `cos_sim 1.000002` (Week 5)
- The Kernel Hub interception mechanism works on Neuron: layer swap, function swap, graceful
  fallback, 115 + 95 upstream registration points

The PoC question was "should Neuron invest in first-class HF Kernel Hub support?" This
measurement does not answer no. It says the **eager per-layer path is not the one to invest
in**, and relocates the decision to the graph-mode question.
