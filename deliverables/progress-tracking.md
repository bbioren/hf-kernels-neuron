*Synced from the project repo. Do not edit below "Progress Tracking:" by hand — it is overwritten
on each update. Everything above that heading is untouched by the sync.*
**Last synced:** 2026-08-06 · **Owner:** Ben Bioren

*A note on formatting: Quip's importer hardcodes table columns to 6em — about 11 characters — and
ignores every width hint the API can send (five encodings tested, all stripped). So anything with
prose in it is a list here rather than a table. Tables are kept only where the cells are short
enough to read at that width.*

## The five-line version

The mechanism **works on Trainium with no patches to HuggingFace's code** — the PoC's main question
is answered. Three NKI kernels swap into stock Qwen3 dense and Qwen3-MoE across two PyTorch stacks,
with correctness proven by execution counters rather than inferred from output. Two candidates now
beat the compiler on device. The upstream ask is **one item, down from three**, because two of the
three turned out to be my own mistake. Performance is the weak half: the ops the Hub intercepts most
widely are the ops with least to gain, and both winning candidates are blocked from real model shapes
by the same unanswered question.

## What I am working on right now

Re-validating everything on the **Native PyTorch** stack, since that is the one HuggingFace intends
and all earlier work was on torch-xla. That switch already invalidated two of three upstream asks, so
this is not bookkeeping. Next two tasks are both cheap and decisive — see Phase 1 below.

## Status against the six-week plan

- **Week 1 — Neuron device path verification, `LocalLayerRepository` demo.** Done.
- **Week 2 — RMSNorm kernel + Qwen3 validation.** Done, though the original accuracy numbers were
  invalid and had to be replaced: the kernel had never actually executed. The tests fed CPU tensors,
  every case silently took the PyTorch fallback, and it reported a flawless `max_diff = 0.00e+00`.
  Every test now asserts via a call counter that the NKI branch ran.
- **Week 3 — Hub packaging, RoPE, register neuron entries, `use_kernels=True`.** Done. RoPE is a
  **real port** of production `nkilib`. Neuron entries done locally. Hub packaging blocked on a
  repo-home decision. `use_kernels=True` **now works unpatched** on the native stack.
- **Week 4 — SiLU, full Qwen3 dense e2e, MFU.** Done, then re-done twice after root-causing the
  slowdown.
- **Week 5 — Qwen3-MoE.** Done. All three kernels transfer with **zero code changes**.
- **Week 6 — PoC doc + recommendation.** In progress; the document is live and current.

**Two corrections to the plan's own assumptions.** Upstream coverage is larger than estimated: **115**
RMSNorm registrations (plan said 87) and **95** RoPE model files (plan said 66), so the leverage
argument is stronger than written. And the kernels came from production `nkilib` rather than
`nki_samples` for RoPE — `nkilib` has no standalone RMSNorm (it always fuses quantisation) and no
activations module at all, so RMSNorm and SiLU are tutorial-derived. That sourcing constraint turned
out to be a finding in itself.

## Definition of done

**Floor — must hit:**

- `"neuron"` device support working, forward-swap proven on Trainium — **done**, and with *no patches*
  to HuggingFace's code on the native stack.
- NKI RMSNorm + RoPE packaged and validated e2e on Qwen3 dense with `use_kernels=True` — **done**,
  plus SiLU, plus Qwen3-MoE.
- Measured MFU delta with the denominator stated — **done**. 632 TFLOPS/device TensorEngine ÷ 2 for
  LNC2 = 316 TFLOPS per logical core.
- PoC document delivered to the kernels team — **not yet**. Written and current, pending review
  (decision 1).

**Ceiling — stretch:**

- SiLU / MLP activation kernels — **done** for SiLU. Fused RMSNorm+MLP tested: wins 1.76x at
  Qwen3-0.6B's shape, blocked above `intermediate_size = 4096`.
- Hub publishing for a Neuron kernel — **not done**. `kernel-builder` still has no Neuron target;
  blocked on decision 3.
- At least one Qwen3-MoE kernel swapped, or a gap analysis — **gap analysis done**. All three dense
  kernels transfer unchanged; the best MoE-specific target identified is the routing `sort`/`histc`
  step, which is what currently stops Qwen3-MoE running on Neuron at all.

## Performance

Qwen3-0.6B, 28 layers, bf16, forward only, single logical core, denominator 316 TFLOPS.

| seq | stack | baseline | kernelized | verdict |
|---|---|---|---|---|
| 512 | xla | 43.94 ms | 71.32 ms | 1.62x slower |
| 512 | native | 189.97 ms | 96.46 ms | 1.97x faster |
| 2048 | xla | 117.78 ms | 161.04 ms | 1.37x slower |
| 2048 | native | 340.74 ms | 251.86 ms | 1.35x faster |

**The native rows are a trap and should not be quoted alone.** The native baseline is 4.32x slower
than the XLA one, so the ratio flipped because the denominator got worse — not because anything got
faster. Native kernelized MFU (2.20%) is *below* XLA kernelized MFU (2.98%). Native fixed the
integration story, not performance.

**Two candidates do beat the compiler**, and both are shape *windows* rather than thresholds:

- **`nkilib` flash attention** — 1.48x faster at seq 2048 and **2.11x at 3072**; loses 2.01x at 512
  and 1.79x at 4096. Device time, torch-xla.
- **Fused RMSNorm+MLP** — **1.76x** at H=1024/I=3072; loses 1.45x at H=4096/I=4096. Wall clock,
  native, so provisional.

Also found and fixed: two framework dispatch bugs worth **322x** combined per kernel call
(52.25 → 0.162 ms). Both are the same bug twice — a cache exists and the surrounding code path
defeats it. Neither is a property of per-layer kernel dispatch on Neuron, which is how the slowdown
had been reported for weeks.

## Plan forward

Effort in working days. Targets are relative because the end date is not pinned — give me the
deadline and I will convert these to dates. Phases 1–3 are about **two weeks**; Phase 4 is another
week and carries real risk of returning a negative answer, which is still worth knowing.

**Why Phase 1 goes first even though it looks least productive:** both tasks test assumptions the
rest of the plan rests on. Learning in a day that the eager framing is wrong is much cheaper than
learning it in a fortnight.

**Phase 1 — settle the framing.** Owner: Ben.

- **`torch.compile` viability on native** — flip `can_torch_compile=True` on one kernel and see if it
  traces. *p0, 0.5–1d, Week 1.* Consulted: John. Highest information-per-day left: if it traces, the
  dispatch findings may dissolve and the fusion analysis needs redoing under a different compiler.
- **Device-time profiling on native** — does NEFF+NTFF capture work there? *p0, 1–1.5d, Week 1.*
  Gates all of Phase 2; every native number is wall clock until this works. Risk: may not be
  supported on a 0.1.0 stack.
- Measure the `_detect_target` fix's value on native. *p1, 0.5d, Week 1.* It was applied to both
  native runs, so its native contribution is unmeasured while I cite it as an upstream ask.

**Phase 2 — make the wins real.** Owner: Ben. All blocked on Phase 1.

- Re-measure flash attention on native, device time. *p0, 1d, Week 2.* The headline result is
  currently XLA-only.
- **Wire attention through transformers' attention interface.** *p0, 2–3d, Week 2.* Consulted: Samir.
  Biggest credibility gap — the best result currently bypasses the Kernel Hub entirely.
- Re-measure fused RMSNorm+MLP on device rather than wall clock. *p1, 0.5d, Week 2.*
- Find where XLA switches attention strategy — HLO dump either side of seq 4096. *p2, 1d, Week 2–3.*
  Consulted: compiler team. The attention window closes because the *compiler* improves at 4096, not
  because the kernel degrades, and nobody here knew that threshold existed.

**Phase 3 — upstream.** Owner: Ben.

- Mapping-entry PR, adding `"neuron"` to `_KERNEL_MAPPING`. *p0, 0.5–1d, Week 3.* **Blocked on
  decision 3.** Consulted: Samir.
- `_detect_target` CR or handoff package. *p0, 0.5d, Week 3.* **Blocked on decision 2.**
- Hub upload spike — variant dirs vs our flat layout. *p1, 1–2d, Week 3.* **Blocked on decision 3.**

**Phase 4 — SPMD.** Owner: Ben. **Blocked on decision 4.**

- **Can a per-layer swap express a multi-core launch?** Prototype. *p0, 2–3d, Week 4.* Consulted:
  kernels + nkilib. The gating question for the entire performance story.
- If yes, re-measure both winning candidates multi-core. *p0, 2d, Week 4.* Both were designed for
  that configuration; current numbers are their handicapped case.

**Phase 5 — stretch.**

- NKI kernel for the MoE routing `sort`/`histc`. *p2, 3–5d, Week 5.* Consulted: nkilib. Unblocks the
  default Qwen3-MoE path, the compiler error itself recommends NKI for it, and it is blocked by none
  of our findings.
- Backward kernels. *p2, 1–2 weeks.* Out of scope pending decision 6.

**Phase 6 — wrap.** Final PoC document, handoff, review pass. *p0, 1–2d, Week 5–6.* Consulted: John,
Pinak.

## Problems and risks

1. **Two stacks, and the numbers disagree in direction.** Any figure quoted without its stack
   misleads. Mitigated: every number now records its stack, and a warning sits above the headline in
   the results doc.
2. **No device-time profiling on native.** Can't attribute the 1.76x; all native numbers provisional.
   Phase 1 addresses it.
3. **Both winning candidates only win at shapes nobody deploys.** The wins are real and currently
   unusable. The `I <= 4096` boundary is unchanged across two compiler generations, which routes this
   to the SPMD question.
4. **The best result bypasses the Kernel Hub** — attention is called directly, so the strongest
   number is not yet a Kernel Hub result. Phase 2 addresses it.
5. **The project is eager-only, which is partly my assumption rather than a finding.** May have
   measured the configuration that matters least: compile is ~23% MFU against ~5% eager on Neuron's
   own figures. Phase 1 addresses it.
6. **Hub publishing never exercised.** Can't ship kernels even with the mapping entries. Phase 3.
7. **I have revised the headline five times.** Credibility risk if a sixth revision lands after it
   reaches the kernels team. Decision 1 exists for this. Each revision came from measuring something
   the previous version had assumed, and every measurement was individually correct — the pattern is
   documented rather than buried, and is arguably the most transferable output of the project.

## Decisions needed

1. **Review the recommendation before it goes to Hanbo / Karthick.** *p0. Owner: John + Pinak.*
   Blocks the final deliverable. My recommendation: do this before anything ships.
2. **Do I send a CR against a core SDK component, or hand it over with the reproducer?** *p0. Owner:
   John, consulted NKI team.* Blocks the 86x `_detect_target` fix. My recommendation: hand over with
   the reproducer and offer to write it. Candidate owners from the Neuron Expert Directory are
   `ggandii` (NKI APIs), `qieqingy` (NKI repo), with `zhehongb` as escalation.
3. **Publishing under `aws-neuron` on HF — our call or up a level, and who asks HF for the
   trusted-publisher flag?** *p0. Owner: Pinak, consulted Samir and John.* Blocks Hub upload and the
   mapping PR. My recommendation: `aws-neuron/` with `trust_remote_code=True` and a TODO, mirroring
   what transformers already ships for `Atlas-Inference`. The org already exists, and the trust gate
   is a settable flag rather than a hardcoded org.
4. **Who owns the multi-core SPMD question?** *p0. Owner: John, consulted kernels and nkilib.* Blocks
   the entire performance story. I cannot answer it alone.
5. **Am I still in scope?** I am two layers below the Kernel Hub now. *p1. Owner: John + Pinak.*
   Happy either way, but worth an explicit call.
6. **Does training matter for the beta?** All three kernels are inference-only. *p2. Owner: Pinak.*
   Backward kernels stay out of scope unless you say otherwise.

## Open questions for other teams

- **Can a NKI custom call participate in compiler fusion?** *Compiler team.* Decides whether small
  memory-bound ops can ever win — each swap currently *removes* a fusion the compiler was already
  doing.
- **Is the `intermediate_size > 4096` single-core limit intended?** *nkilib.* Unchanged across two
  compiler generations, so it looks like a design boundary rather than a bug. It excludes every
  deployed model.
- **Why is native eager 3–4x slower than the torch-xla graph path?** *Frameworks.* Largest single
  number in my tables and the first thing anyone will ask.
- **Should `torch_neuronx` set a `torch.neuron` attribute on the XLA stack?** *Frameworks.* Would fix
  dependency declaration for XLA users; native already does it.
- **`neuronx-cc` missing from `PATH` hangs forever with no error.** *Runtime.* Worst
  customer-experience issue found — cost about an hour and nearly had me replace a host driver to fix
  a `PATH` bug. The information for a perfect error message is already at the failure point.
- **`experts_implementation="batched_mm"` is needed for MoE on Trainium.** *Docs.* Undocumented, and
  without it Qwen3-MoE does not run at all.

## Changelog

**2026-08-06** — Set up automated sync from the project repo into this section. Established that Quip
ignores every table-width hint the API can send, so prose moved out of tables into lists.

**2026-08-05**

- Stood up the Native PyTorch stack. **Both integration gates I had reported were artifacts of running
  on torch-xla** — the proposed transformers device-resolution patch is withdrawn, and stock
  `use_kernels=True` works unpatched. Upstream ask drops from three items to one.
- Re-ran MFU on native: the headline *sign* flips to "1.97x faster", which is **not** a win, since the
  baseline is 4.32x worse. Documented as a trap rather than a result.
- Tested the fused RMSNorm+MLP Samir pointed out: **1.76x** at Qwen3-0.6B's MLP shape, 1.45x slower at
  H=4096/I=4096.
- Re-tested the `I > 4096` compile boundary on the new compiler: **unchanged**, so every deployed model
  is still excluded.
- Corrected my own repo-home recommendation. The Hub trust gate is a settable org flag, not a hardcoded
  `kernels-community` check, and a mapping entry can bypass it anyway — transformers already ships that
  pattern. `aws-neuron` already exists as an HF org.
- Named the eager-only framing as a limitation rather than a given.

**Earlier** — root-caused a 206x slowdown to two framework caching bugs (322x recovered), found the
first speedup (flash attention), and established why RMSNorm/RoPE/SiLU cannot win.

---

*Deeper reading, all in the project repo — ask me for access:* `deliverables/poc-document.md` (the live
technical document), `results/README.md` (every number with provenance), `docs/poc-findings.md` (full
findings log, 34 entries), `WORKLOG.md` (session history including the corrections).
