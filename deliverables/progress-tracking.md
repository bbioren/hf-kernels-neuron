*Synced from the project repo. Do not edit below this line by hand — it is overwritten on each
update. Everything above "Progress Tracking:" is untouched by the sync.*
**Last synced:** 2026-08-06 · **Owner:** Ben Bioren

## Status against the six-week plan

| Week | Planned | Outcome | Status |
|---|---|---|---|
| 1 | Neuron device path verification, `LocalLayerRepository` demo | Done | **Done** |
| 2 | RMSNorm kernel + Qwen3 validation | Done, though the original accuracy numbers were invalid and were replaced — the kernel had never executed | **Done** |
| 3 | Hub packaging, RoPE, register neuron entries, `use_kernels=True` | RoPE done as a **real port** of production `nkilib`. Neuron entries done locally. Hub packaging blocked on a repo-home decision. `use_kernels=True` **now works unpatched** | **Done** |
| 4 | SiLU, full Qwen3 dense e2e, MFU | Done, then re-done twice after root-causing the slowdown | **Done** |
| 5 | Qwen3-MoE | Done — all three kernels transfer with **zero code changes** | **Done** |
| 6 | PoC doc + recommendation | Live and current | **In progress** |

**Two corrections to the plan's own assumptions.** Upstream coverage is larger than estimated: **115**
RMSNorm registrations (plan said 87) and **95** RoPE model files (plan said 66). And the kernels came from
the production `nkilib`, not `nki_samples`, for RoPE — `nkilib` has no standalone RMSNorm (it always fuses
quantisation) and no activations module at all, so those two are tutorial-derived. That sourcing
constraint turned out to be a finding in itself.

## Definition of done

| Floor (must hit) | Status |
|---|---|
| `"neuron"` device support working, forward-swap proven on Trainium | **Done** — and with **no patches** to HuggingFace's code on the Native PyTorch stack |
| NKI RMSNorm + RoPE packaged and validated e2e on Qwen3 dense with `use_kernels=True` | **Done** — plus SiLU, plus Qwen3-MoE |
| Measured MFU delta with the denominator stated | **Done** — 632 TFLOPS/device TensorEngine ÷ 2 for LNC2 = 316 |
| PoC document delivered to the kernels team | **Not yet** — written and current, pending review (decision 1) |

| Ceiling (stretch) | Status |
|---|---|
| SiLU / MLP activation kernels | **Done** for SiLU. Fused RMSNorm+MLP tested — wins 1.76x at Qwen3-0.6B's shape, blocked above `I=4096` |
| Hub publishing working for a Neuron kernel | **Not done** — `kernel-builder` still has no Neuron target; blocked on decisions 3 |
| At least one Qwen3-MoE kernel swapped, or a gap analysis | **Gap analysis done.** All three dense kernels transfer unchanged; the best MoE-specific target identified is the routing `sort`/`histc` step |

## The five-line version

The mechanism **works on Trainium with no patches to HuggingFace's code** — the PoC's main question is
answered. Three NKI kernels swap into stock Qwen3 dense and Qwen3-MoE across two PyTorch stacks, with
correctness proven by execution counters rather than inferred from output. Two candidates now beat the
compiler on device. The upstream ask is **one item, down from three**, because two of the three turned out
to be my own mistake. Performance is the weak half: the ops the Hub intercepts most widely are the ops
with least to gain, and both winning candidates are blocked from real model shapes by the same unanswered
question.

## What I am working on right now

Re-validating everything on the **Native PyTorch** stack, since that is the one HuggingFace intends and
all earlier work was on torch-xla. That switch already invalidated two of three upstream asks, so this is
not bookkeeping. Next two tasks are both cheap and decisive — see Phase 1.

## Performance, stated carefully

Qwen3-0.6B, 28 layers, bf16, forward only, single logical core, denominator 316 TFLOPS.

| seq | stack | baseline | kernelized | verdict |
|---|---|---|---|---|
| 512 | torch-xla | 43.94 ms | 71.32 ms | 1.62x slower |
| 512 | native | 189.97 ms | 96.46 ms | *1.97x "faster"* |
| 2048 | torch-xla | 117.78 ms | 161.04 ms | 1.37x slower |
| 2048 | native | 340.74 ms | 251.86 ms | *1.35x "faster"* |

**The native rows are a trap and should not be quoted alone.** The native baseline is 4.32x slower than
the XLA one, so the ratio flipped because the denominator got worse. Native kernelized MFU (2.20%) is
*below* XLA kernelized MFU (2.98%). Nothing got faster. Native fixed the integration story, not
performance.

Two candidates do beat the compiler, both **shape windows** rather than thresholds:

| candidate | wins | loses | measured on |
|---|---|---|---|
| `nkilib` flash attention | 1.48x @ seq 2048, **2.11x @ 3072** | 2.01x @ 512, 1.79x @ 4096 | device time, torch-xla |
| fused RMSNorm+MLP | **1.76x** @ H=1024/I=3072 | 1.45x @ H=4096/I=4096 | wall clock, native |

Also found and fixed: two framework dispatch bugs worth **322x** combined per kernel call
(52.25 → 0.162 ms). Both are the same bug twice — a cache exists and the surrounding code path defeats it.
Neither is a property of per-layer kernel dispatch on Neuron, which is how the slowdown had been reported
for weeks.

## Plan forward

Effort in working days. Targets are relative because the intended end date is not pinned — give me the
deadline and I will convert these to dates.

| Phase | Milestone | Owner | Consulted | Priority | Effort | Target | Status |
|---|---|---|---|---|---|---|---|
| **1. Settle the framing** | `torch.compile` viability on native — flip `can_torch_compile=True` on one kernel, see if it traces | Ben | John | **p0** | 0.5–1 | Week 1 | Not started |
| 1 | Device-time profiling on native — does NEFF+NTFF capture work there? | Ben | — | **p0** | 1–1.5 | Week 1 | Not started |
| 1 | Measure the `_detect_target` fix's value on native | Ben | — | p1 | 0.5 | Week 1 | Not started |
| **2. Make the wins real** | Re-measure flash attention on native, device time | Ben | — | p0 | 1 | Week 2 | Blocked on 1 |
| 2 | Wire attention through transformers' attention interface | Ben | Samir | p0 | 2–3 | Week 2 | Blocked on 1 |
| 2 | Re-measure fused RMSNorm+MLP on device rather than wall clock | Ben | — | p1 | 0.5 | Week 2 | Blocked on 1 |
| 2 | Find where XLA switches attention strategy (HLO dump around seq 4096) | Ben | compiler | p2 | 1 | Week 2–3 | Not started |
| **3. Upstream** | Mapping-entry PR — add `"neuron"` to `_KERNEL_MAPPING` | Ben | Samir | p0 | 0.5–1 | Week 3 | **Blocked on decision 3** |
| 3 | `_detect_target` CR or handoff package | Ben | John, NKI team | p0 | 0.5 | Week 3 | **Blocked on decision 2** |
| 3 | Hub upload spike — variant dirs vs our flat layout | Ben | Samir | p1 | 1–2 | Week 3 | **Blocked on decision 3** |
| **4. SPMD** | Can a per-layer swap express a multi-core launch? Prototype | Ben | kernels + nkilib | **p0** | 2–3 | Week 4 | **Blocked on decision 4** |
| 4 | If yes: re-measure both winning candidates multi-core | Ben | — | p0 | 2 | Week 4 | Blocked |
| **5. Stretch** | NKI kernel for the MoE routing `sort`/`histc` | Ben | nkilib | p2 | 3–5 | Week 5 | Not started |
| 5 | Backward kernels | Ben | Pinak | p2 | 1–2 wks | — | Out of scope pending decision 6 |
| **6. Wrap** | Final PoC document, handoff, review pass | Ben | John, Pinak | p0 | 1–2 | Week 5–6 | In progress |

Phases 1–3 are about **two weeks**. Phase 4 is another week and carries real risk of returning a negative
answer, which is still worth knowing.

**Why Phase 1 goes first even though it looks least productive:** both tasks test assumptions the rest of
the plan rests on. Learning in a day that the eager framing is wrong is much cheaper than learning it in a
fortnight.

## Problems and risks

| # | Problem | Impact | Mitigation |
|---|---|---|---|
| 1 | Two stacks, and the numbers disagree in **direction** | any figure quoted without its stack misleads | every number now records its stack; a warning sits above the headline in the results doc |
| 2 | No device-time profiling on native | can't attribute the 1.76x; all native numbers provisional | Phase 1 |
| 3 | Both winning candidates only win at shapes nobody deploys | wins are real and currently unusable | `I<=4096` boundary unchanged across two compiler generations → routes to the SPMD question |
| 4 | Best result **bypasses the Kernel Hub** — attention is called directly | strongest number is not yet a Kernel Hub result | Phase 2 |
| 5 | Project is eager-only, which is partly my assumption not a finding | may have measured the configuration that matters least | Phase 1. Compile is ~23% MFU vs ~5% eager on Neuron's own figures |
| 6 | Hub publishing never exercised | can't ship kernels even with the mapping entries | Phase 3 |
| 7 | I have revised the headline **five times** | credibility risk if a sixth revision lands after it reaches the kernels team | decision 1 — review before it ships |

On #7: each revision came from measuring something the previous version had assumed, and every measurement
was individually correct. The pattern is documented rather than buried, and is arguably the most
transferable output of the project.

## Decisions needed

| # | Decision | Owner | Consulted | Priority | Blocks | My recommendation |
|---|---|---|---|---|---|---|
| 1 | **Review the recommendation before it goes to Hanbo / Karthick** | John, Pinak | — | **p0** | the final deliverable | do this before anything ships |
| 2 | Do I send a CR against a core SDK component, or hand it over with the reproducer? | John | NKI team | p0 | the 86x `_detect_target` fix | hand over with the reproducer, offer to write it. Candidate owners from the Neuron Expert Directory: `ggandii` (NKI APIs), `qieqingy` (NKI repo), `zhehongb` (escalation) |
| 3 | Publishing under `aws-neuron` on HF — our call or up a level? Who asks HF for the trusted-publisher flag? | Pinak | Samir, John | p0 | Hub upload + the mapping PR | `aws-neuron/` with `trust_remote_code=True` and a TODO, mirroring what transformers already ships for `Atlas-Inference`. The org already exists; the trust gate is a settable flag, not a hardcoded org |
| 4 | Who owns the multi-core SPMD question? | John | kernels, nkilib | p0 | the entire performance story | route out — I cannot answer it alone |
| 5 | Am I still in scope? I am two layers below the Kernel Hub now | John, Pinak | — | p1 | how I spend remaining time | happy either way, but worth an explicit call |
| 6 | Does training matter for the beta? All three kernels are inference-only | Pinak | — | p2 | whether backward kernels are in scope | out of scope unless you say otherwise |

## Open questions for other teams

| Question | Team | Why it matters |
|---|---|---|
| Can a NKI custom call participate in compiler fusion? | compiler | Decides whether small memory-bound ops can ever win. Each swap currently *removes* a fusion the compiler was already doing |
| Is the `intermediate_size > 4096` single-core limit intended? | nkilib | Unchanged across two compiler generations → looks like a design boundary, not a bug. Excludes every deployed model |
| Why is native eager 3–4x slower than the torch-xla graph path? | frameworks | Largest single number in the tables and the first thing anyone will ask |
| Should `torch_neuronx` set a `torch.neuron` attribute on the XLA stack? | frameworks | Would fix dependency declaration for XLA users. Native already does it |
| `neuronx-cc` missing from `PATH` **hangs forever** with no error | runtime | Worst customer-experience issue found. Cost ~1h and nearly had me replace a host driver to fix a `PATH` bug |
| `experts_implementation="batched_mm"` needed for MoE on Trainium | docs | Undocumented, and without it Qwen3-MoE does not run at all |

## Changelog

**2026-08-06** — Set up automated sync from the project repo into this section.

**2026-08-05**

- Stood up the Native PyTorch stack. **Both integration gates I had reported were artifacts of running on
  torch-xla** — the proposed transformers device-resolution patch is withdrawn. Stock `use_kernels=True`
  works unpatched. Upstream ask drops from 3 items to 1.
- Re-ran MFU on native: the headline *sign* flips to "1.97x faster", which is **not** a win — the baseline
  is 4.32x worse. Documented as a trap rather than a result.
- Tested the fused RMSNorm+MLP Samir pointed out: **1.76x** at Qwen3-0.6B's MLP shape, 1.45x slower at
  H=4096/I=4096.
- Re-tested the `I>4096` compile boundary on the new compiler: **unchanged**, so every deployed model is
  still excluded.
- Corrected my own repo-home recommendation. The Hub trust gate is a settable org flag, not a hardcoded
  `kernels-community` check, and a mapping entry can bypass it anyway — transformers already ships that
  pattern. `aws-neuron` already exists as an HF org.
- Named the eager-only framing as a limitation rather than a given.

**Earlier** — root-caused a 206x slowdown to two framework caching bugs (322x recovered), found the first
speedup (flash attention), and established why RMSNorm/RoPE/SiLU cannot win.

---

*Deeper reading, all in the project repo — ask me for access:* `deliverables/poc-document.md` (the live
technical document), `results/README.md` (every number with provenance), `docs/poc-findings.md` (full
findings log, 34 entries), `WORKLOG.md` (session history including the corrections).
