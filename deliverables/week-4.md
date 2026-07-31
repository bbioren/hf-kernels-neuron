# Week 4 Deliverable — MFU measurement

**Date:** 2026-07-31
**Hardware:** trn2.3xlarge, 1 Neuron device, LNC2 (4 physical cores → 2 logical), single logical core used
**Versions:** `kernels 0.15.2`, `transformers 5.15.0.dev0`, `torch 2.9.1+cu128`, `neuronx-cc 2.26.6360.0`, `nki 0.5.0`

---

## Headline

> **SUPERSEDED IN PART — read this box first.** This deliverable was written before the ~53 ms was
> root-caused. Every measurement in it is correct and reproducible. The *attribution* was wrong: the
> cost is an uncached `neuron-ls` subprocess forked on every `@nki.jit` invocation, not a
> framework-boundary or NEFF-switching charge. One `lru_cache` removes 102x of it. The corrected
> headline is **3.4x slower at seq 512, 2.06x at seq 2048**, not 208x. Finding #24 has the full
> story; the "What would change the answer" section below is superseded outright. Kept as written
> because the reasoning trace is the useful part, and because the *ruling-out* work here is what
> eventually made the root cause findable.

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

### The same measurement after the root cause was fixed

`scripts/measure_mfu.py --fix-target-detection` caches `_detect_target`, which is the entire fix.
Same model, same kernels, same denominator:

| Configuration | Step time | MFU (per core) | NKI calls/step | vs baseline |
|---|---|---|---|---|
| baseline, seq 512 | 42.04 ms | 5.05 % | 0 | — |
| all three kernels, seq 512 | **141.43 ms** | **1.50 %** | 169 | 3.36x slower |
| baseline, seq 2048 | 108.76 ms | 9.90 % | 0 | — |
| all three kernels, seq 2048 | **223.99 ms** | **4.81 %** | 169 | **2.06x slower** |

The two sequence lengths are the amortisation test: call count is fixed by model depth, so more
sequence means more work per call. 2.59x more baseline work costs only 1.16x more per call, so the
residual overhead is near-fixed and the penalty nearly halves.

Residual added cost is 0.588 ms/call at seq 512 against **0.02 ms/call of device time** — the device
executes a 28-call NEFF in 0.609 ms at 43% memory-bandwidth utilisation and 95% engine active time.
The kernels were never the problem.

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

## What would change the answer — RESOLVED, and both guesses were wrong

> This section originally named graph mode as the decisive question and asked for a different stack.
> Kept in place because how it resolved is more instructive than the conclusion.

**Graph mode was not the decisive question, and `torch.compile` was not blocking it.**

Two corrections:

1. **`torch.compile` is not broken on this stack.** `add`, `mul` and `relu` all compile and run on
   XLA tensors. What fails is the set of ops `torch_neuronx` replaces with XLA user computations —
   `silu`, `gelu`, `Embedding`, `Softmax`, `CrossEntropyLoss`, `topk`, `argmax`, `Dropout` — because
   the dispatch predicate accepts a `FakeTensor` and then rejects it inside
   `_xla_user_computation`. Real upstream bug (Finding #23), but not a blocker here.
2. **torch-xla is already a graph runtime**, so the question was answerable without `torch.compile`
   at all. Counting device executions with torch-xla's own `ExecuteTime` metric: **28 NKI calls fuse
   into one HLO graph and one device execution** (196 nodes) and still cost 28x. Graph batching was
   never the lever.

That relocated the cost off the device, and the profile confirmed it: the NEFF containing all 28
calls executes in **0.609 ms** while wall time is 1459 ms, and 99.9% of that wall time is spent
before `mark_step`. cProfile then named the function — 51 of the 52 ms is `select.poll` waiting on
`subprocess.check_output` inside `_detect_target()`.

**The second thing this section flagged turned out to be right.** It said "~53 ms is large enough to
look like a misconfiguration rather than a design point. Worth asking whether it is expected." It
was not expected — it is a bug, and one decorator fixes it. That instinct should have been pursued
before four more framework-level experiments.

**What would change the answer now:** whether the residual ~0.59 ms/call in `create_computation` is
also cacheable. That is the difference between 3.4x slower and plausibly near parity, and it is a
scoping question for the NKI team rather than an experiment we can run.

---

## Week 4 goals vs outcome

| Goal | Status |
|---|---|
| Measure MFU with and without kernels, denominator stated | **Done, then re-done.** 5.06% → 0.02% as first measured; 5.05% → 1.50% after root-causing (→ 4.81% at seq 2048). Denominator explicit, FLOP count auditable. |
| Root-cause the regression rather than just reporting it | **Done** — an uncached `neuron-ls` subprocess per invocation, fix verified at 102x. Not in the original Week 4 goals; it is the most valuable output of the week. |
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
