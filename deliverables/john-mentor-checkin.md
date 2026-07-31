# Draft check-in for John (internship mentor) — Weeks 3-6

**Status: DRAFT, NOT SENT.** Review before sending.

**REWRITTEN TWICE.** Three earlier versions are wrong and should not be sent. (1) Asked John to decide
Week 4/5 scope — resolved by getting done. (2) Led with "the kernels are 208x slower for structural
reasons" and asked for a stack where `torch.compile` works — the conclusion was wrong and the ask is
withdrawn. (3) Led with the one-line NKI caching bug worth 102x and implied fixing dispatch was the
path to parity — the bug is real but Finding #25 shows dispatch is no longer the binding constraint.

Current version leads with both: the dispatch bug (fix it, 102x, one decorator) *and* the fusion
barrier (2.5-2.7x on device with dispatch excluded, which no plumbing work fixes). The top ask is now a
compiler question, not a dispatch one.

---

## The two things to read first

**One: every `@nki.jit` invocation forks a subprocess.** `nki/framework/compiled.py::_compile_opts()`
calls `resolve_target()` on every call, which falls through to `_detect_target()`, which runs
`neuron-ls` to ask the hardware what it is. **~52 ms per kernel call.** It sits outside
`_nki_compile_cache` because its result is part of the cache key, so a cache *hit* still pays it in
full. Caching it takes per-call cost from 51.74 ms to 0.49 ms with bit-identical accuracy, and
Qwen3-0.6B from **208x slower to 3.4x slower**. Not Kernel Hub specific — anything calling NKI kernels
per-layer from eager PyTorch is paying it right now. Fix it regardless of everything else.

**Two: even with dispatch removed entirely, the kernels lose on device by 2.5-2.7x.** Not because they
are bad — their marginal HBM traffic is *exactly* the theoretical floor for an op that cannot fuse.
Because a NKI custom call is opaque to the compiler, so nothing fuses across it, and for memory-bound
ops fusion is the whole optimisation. That makes break-even **unreachable** for RMSNorm / RoPE / SiLU
rather than distant, and it is the conclusion no plumbing work changes.

The recommendation therefore ends up: fix the dispatch bug, ask the compiler team whether NKI calls can
participate in fusion, and don't port small memory-bound ops at per-layer granularity.

I have now been wrong about this three times in a row with correct measurements every time, which is
itself the most useful thing I have to report. Detail below rather than a quiet doc edit.

---

## Short version (Slack)

> Hi John — significant update, and it reverses what I told you last time.
>
> **Found the cause of the 208x slowdown, and it's a one-line bug in NKI.** Every `@nki.jit` call
> runs `_compile_opts()` → `resolve_target()` → `_detect_target()`, which forks `neuron-ls` to
> detect the hardware. ~52 ms per call. It's outside the compile cache because its result is part
> of the cache *key*, so a cache hit still pays it.
>
> `lru_cache` on `_detect_target` takes it from 51.74 → 0.49 ms/call, **102x**, cos_sim identical to
> six decimals. At model level Qwen3-0.6B goes from 208x slower to **3.4x** slower (MFU 0.02% →
> 1.50%), and 2.06x at seq 2048. `NEURON_PLATFORM_TARGET_OVERRIDE` gets the same result without a
> code change.
>
> **This isn't a Kernel Hub problem.** Any per-layer NKI use from eager PyTorch pays it. Which is
> my first question: who owns `nki/compiler/target.py`, and should I write the CR or hand it over
> with the profile?
>
> **I was wrong twice and it's worth saying how.** I'd told you the ~53 ms was a graph-transition
> cost and that the decisive test was whether graph mode amortises it. Neither held up:
>
> - `torch.compile` is *not* broken on this stack — I'd concluded that from one error message.
>   `add`/`mul`/`relu` compile fine on XLA. What fails is the set of ops `torch_neuronx` overrides
>   with XLA user computations — `silu`, `gelu`, `Embedding`, `Softmax`, `CrossEntropyLoss`, `topk`,
>   `argmax`, `Dropout` — because the dispatch predicate accepts a `FakeTensor` and then rejects it.
>   That's a real bug hitting nearly every transformer, and it's a separate thing to file.
> - `torch.compile` was never the right instrument anyway. torch-xla is *already* a graph runtime.
>   I checked with torch-xla's execution counters: 28 NKI calls fuse into **one** HLO graph and
>   **one** device execution, and still cost 28x. Graph batching was never the lever. Withdrawing
>   that ask — sorry for sending you after a stack I didn't need.
>
> What actually found it was changing instrument instead of running another variant. The device
> profile says the NEFF with all 28 calls executes in **0.609 ms** at 43% memory bandwidth
> utilisation — so 99.96% of the wall time was never on the device. Then cProfile named the
> function in one run. My four earlier experiments all measured wall clock at the framework level
> and none of them *could* have seen this.
>
> **Then I found the thing that actually decides it, and it's worse.** I'd been reporting
> everything in wall-clock time, which conflates dispatch with device work. So I profiled device
> time only, comparing each NKI kernel against the torch op it replaces on identical work:
>
> ```
>                  device ms (28 calls)   HBM traffic   marginal traffic/call
>   NKI SiLU              0.607            188.7 MB     6.29 MB = 1.00x floor
>   torch SiLU            0.224              6.3 MB     ~0.00 MB
>   NKI RMSNorm           1.625            188.8 MB     6.29 MB = 1.00x floor
>   torch RMSNorm         0.637              6.4 MB     ~0.00 MB
> ```
>
> **NKI is 2.5-2.7x slower on device, with dispatch excluded entirely.** And it isn't the kernels —
> their marginal traffic is *exactly* the theoretical floor for an op that can't fuse (one read in,
> one write out, nothing spilled). Torch's traffic is independent of call count, which is only
> possible if the compiler fused the whole chain into one pass.
>
> So a NKI custom call is an **optimisation barrier**. The compiler can't fuse across it, and every
> swap forces a HBM round-trip where the data used to stay resident. For memory-bound ops fusion
> *is* the optimisation, so the kernel is competing against not touching memory at all.
>
> **That makes break-even unreachable for these ops, not just distant.** No amount of dispatch work
> or kernel-writing skill gets there. And the corollary is the sharpest thing in the PoC: the ops the
> Kernel Hub is *best* at intercepting — RMSNorm 115 registrations, RoPE 95 model files, every
> `ACT2FN` activation via one decoration — are precisely the ops that lose most from being
> intercepted. Reach and usefulness are inversely correlated, which is why this looked good for four
> weeks and then didn't.
>
> **The one question that would undo it:** can a NKI custom call participate in compiler fusion, or
> be made transparent to the fusion pass? If yes, this dissolves. If no, per-layer swapping of small
> memory-bound ops is closed on merit and the only viable shape is a kernel spanning a whole fused
> region — which is what nkilib already ships. I can't answer that from here; it's a compiler-team
> question and it's now my top ask.
>
> I also nearly filed a bug against my own kernels on the way to this. Dividing traffic by call count
> said they move 3.00x more data than necessary, which reads as a spilled fp32 intermediate — and the
> `nl.arange` migration had introduced exactly such a temporary, so I had a culprit ready. Hitting
> exactly 3.00x for both ops independently is what made me check. Traffic isn't linear in N, and once
> you solve for fixed vs marginal the kernels come out optimal. Lesson: vary N before dividing by N.
>
> **What landed:** Weeks 3-6 done. Three kernels validated on Qwen3 dense *and* MoE with zero
> changes for MoE (cos_sim 1.000001 / 1.000002). RoPE is a real port of nki-library's `rope_hf`.
> `use_kernels=True` can't reach Neuron (silent no-op) — verified a ~3-line transformers fix takes
> Qwen3 from 0 to 9 swapped layers. Qwen3-MoE won't run on Neuron at all by default because the
> experts path uses `torch.sort`/`histc` → unsupported HLO; fix is
> `experts_implementation="batched_mm"`, undocumented anywhere.
>
> **Questions, in order:** (1) **can a NKI custom call participate in compiler fusion?** Who do I ask.
> This decides whether the integration can ever be a perf win. (2) Who owns `target.py` and do I write
> the CR for the `lru_cache`? (3) Is `NEURON_PLATFORM_TARGET_OVERRIDE` customer-supported or
> internal-only — decides whether it can be documented as a workaround. (4) I'm now two layers below
> the Kernel Hub — NKI's dispatch path and the compiler's fusion behaviour. Still in scope, or hand
> off and go back to kernels?
>
> `deliverables/poc-document.md` has the full thing. Recommendation: fix the one-line dispatch bug
> regardless, ask the fusion question, and **don't** port small memory-bound ops at per-layer
> granularity — not "defer", they can't win.

---

## Longer version, if he wants the detail

### How the wrong conclusion survived four experiments

Worth spelling out, since it is the part I would most want feedback on.

The measurement was never wrong: ~52 ms per call, flat across a 112x sweep of problem size,
reproduced five times within 1%. A fixed per-call cost that ignores problem size is genuinely the
signature of graph-transition overhead, so that was a reasonable first hypothesis.

It then survived four tests. Interleaving NKI calls with torch ops: no change. 28x the data in one
call: 1.02x. Zero recompiles during the timed loop. Production `nkilib` kernels: same figure. Each
result raised my confidence in a wrong answer.

All four measured wall-clock time at the framework level, and none of them could see inside the
52 ms. No further variant of that instrument would have falsified it. What did:

| instrument | result | what it ruled out |
|---|---|---|
| torch-xla `ExecuteTime` counter | 28 NKI calls → **1** device execution | graph batching as the lever |
| neuron-explorer on that NEFF | `total_time` **0.609 ms**, 43% MBU, 95% active | every device-side explanation |
| wall-clock split | **99.9% before `mark_step`** | anything after dispatch |
| cProfile of one call | 51 of 52 ms in `select.poll` ← `subprocess.check_output` | everything else |

The device-vs-wall comparison is two numbers and their ratio was 2400x. It should have been the
first thing I measured, not the fifth. That is the lesson I am taking from this project.

A smaller related error: when I thought the cost was inside the NEFF, I wrote out three candidate
explanations ranked by plausibility. All three were device-side, because the framing had already
concluded the cost was in the execution. The real answer was not ranked low, it was absent.
Enumerating candidates inside one framing feels like rigour and isn't.

### The numbers as they now stand

Qwen3-0.6B, 28 layers, bf16, forward only, single logical core. Denominator 632 TFLOPS/device
TensorEngine ÷ 2 for LNC2 = 316 TFLOPS.

| configuration | step time | MFU | penalty |
|---|---|---|---|
| baseline, seq 512 | 42.04 ms | 5.05% | — |
| kernelized, seq 512, before fix | 8753.65 ms | 0.02% | 208x |
| kernelized, seq 512, after fix | 141.43 ms | 1.50% | 3.36x |
| baseline, seq 2048 | 108.76 ms | 9.90% | — |
| kernelized, seq 2048, after fix | 223.99 ms | 4.81% | **2.06x** |

169 NKI calls/step in all kernelized rows, zero fallbacks, IQRs non-overlapping.

The last two rows are the amortisation test: call count is fixed by depth, so more sequence means
more work per call. 2.59x more baseline work costs only 1.16x more per call, so the overhead is
near-fixed and the penalty nearly halves. Not purely fixed though — ~16% of it does scale.

### Status against your week-by-week plan

| Week | Plan | Outcome |
|------|------|---------|
| 1 | Neuron device path verification | done |
| 2 | RMSNorm kernel + validation | done, though the original accuracy numbers were invalid and were replaced |
| 3 | Hub packaging, RoPE, register neuron entries, `use_kernels=True` | RoPE done as a production port; entries done locally; packaging blocked on repo-home decision; `use_kernels=True` blocked upstream with a verified fix |
| 4 | SiLU, full Qwen3 e2e, MFU | done, then re-done after the root cause |
| 5 | Qwen3-MoE | done — all three kernels transfer unchanged, plus the `batched_mm` discovery |
| 6 | PoC doc + recommendation | drafted, rewritten after the root cause |

Coverage came out better than the guide estimated: 115 RMSNorm registrations vs 87, and 95 RoPE
model files vs 66.

### Corrections to my own work

Four now. Flagging them because they changed conclusions, not just numbers, and because the pattern
is the PoC's most transferable output.

1. **Week 2's accuracy results were measuring the PyTorch fallback.** `@nki.jit` needs XLA tensors;
   the tests built inputs with `torch.randn` on CPU, so every case took the fallback and compared it
   against a mathematically identical reference. `max_diff = 0.00e+00`. The perfection was the tell —
   a reduction kernel on hardware should differ by ~1e-4. Every test now asserts via a call counter
   that the NKI branch ran.
2. **My first benchmark timed an empty graph.** Reported 8-400x slower; I discarded outputs so XLA
   eliminated the computation. The tell was latency not varying with size. The script now refuses to
   report ratios unless latency demonstrably scales with problem size.
3. **Finding #14 was wrong** — I wrote up the two NKI import paths as having disjoint capabilities.
   They are two *versions*: `import nki` is 0.5.0, `from neuronxcc import nki` is older, and
   `nl.arange` was removed in 0.5.0 in favour of `nl.ds`. So it was our tech debt, not an upstream
   question. Since migrated, which improved fp32 accuracy ~50x as a side effect because the fix
   required computing the reduction in fp32.
4. **The 208x structural conclusion** — above.

Three of the four are the same failure: on a lazy-execution backend, a fallback is numerically
correct and an eliminated computation is fast, so measurements fail *silently*. The fourth is
different and worse — valid measurements, invalid conclusion, nothing broken to notice.

### Decisions I'm asking for

1. **Who owns `nki/compiler/target.py`, and do I write the CR?** Reproducer is
   `scripts/probe_target_override_fix.py`. The fix is one decorator. This is the highest-value item
   in the project and it is not Kernel Hub specific.
2. **Is `NEURON_PLATFORM_TARGET_OVERRIDE` supported for customers, or internal-only?** Decides
   whether it goes in documentation as a workaround or stays in my test harness. Related: on a host
   with no `neuron-ls`, `_detect_target()` silently returns `"trn3"` — compiling for the wrong
   generation rather than failing. That looks worth raising in the same conversation.
3. **Is the residual `create_computation` cost cacheable?** ~0.59 ms/call, every invocation rebuilds
   the XLA computation and its HLO protobufs. Same shape as bug #1, 100x smaller, but it is the
   difference between 3.4x slower and plausibly near parity. I did not attempt it — it sits inside
   `torch_xla`'s op-registry path and I would rather ask than guess.
4. **Scope.** I am now a layer below the Kernel Hub, inside NKI's dispatch path. In scope, or hand
   off and return to kernels?
5. **Does the rewritten recommendation land?** It now says invest, starting with two caching fixes,
   and defer per-layer kernel porting until the second one resolves. That is a materially different
   message from the version I described to you last time, so a sanity check before it reaches
   Hanbo/Karthick would help.
6. **Routing for the others**: the `torch_neuronx` fake-tensor bug (#23, reproducer included), the
   nki-library MLP divide-by-zero, and the `torch_neuronx` `torch.neuron` one-liner. File myself or
   through you?
7. **Samir**: do I reach out directly or would you rather introduce? Note the earlier draft to him
   implied HF's per-layer granularity might be structurally wrong for Neuron. That draft is now
   wrong and has been rewritten.

### What I'd do next if you said "keep going"

1. File the `_detect_target` fix — highest value, smallest change, benefits all eager NKI users.
2. Get an answer on the `create_computation` residual, since it gates whether per-layer swapping can
   reach parity.
3. Measure whether any of the three kernels actually beats the torch op it replaces. Every
   performance number in this project so far is about dispatch overhead; that is a different
   question and I have not answered it.
4. The MoE routing `sort`/`histc` kernel — small, unblocked by any of our findings, and the compiler
   error itself recommends NKI for it.

---

## Notes to self before sending

- Lead with the finding and the retraction, in that order. He needs the correction before it reaches
  the kernels team, not after.
- Do not soften "I was wrong twice." The reasoning trace is the useful part, and calibration is part
  of what is being assessed.
- The asks are decisions and ownership questions, not a task queue.
- Withdraw the `torch.compile` stack request explicitly. He may already have spent effort on it.
- Keep the caveat that the kernels are still a net loss. The temptation is to let 102x carry the
  message; 3.4x slower is the actual state.
