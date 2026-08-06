# HuggingFace Kernel Hub on Trainium — status, plan, and open questions

**Document purpose:** HuggingFace's `kernels` library is a runtime kernel-replacement system, now merged
to transformers mainline, that swaps `nn.Module.forward()` methods for optimized implementations pulled
from the Hub. Adding a `"neuron"` device path gives every HuggingFace model with RMSNorm (115 upstream
registrations), rotary embeddings (95 model files), and standard activations access to NKI kernels
automatically, with graceful fallback. This is per-kernel work that scales to the whole model zoo
rather than per-model work, which makes it the highest-leverage HF ecosystem integration point for
Neuron. This document is the standing record of where the PoC is, what is blocked, what needs deciding,
and the plan forward. It is updated as things change — see the Changelog for what moved since you last
read it.

**Owner:** Ben Bioren · **Reviewers:** Pinak Panigrahi (manager), John Gray (mentor) ·
**Recipients of the final PoC:** Hanbo Wang, Karthick Gopalswamy (kernels team) · **Updated:** 2026-08-05

**Tools in use:**

- NKI — 0.5.0 on torch-xla, 0.6.0b1 on native
- Native PyTorch (torch 2.11, torch-neuronx 0.1.0) **and** torch-xla (torch 2.9.1) — numbers are not
  interchangeable between them, see "Performance" below
- HuggingFace `kernels` 0.15.2 + transformers 5.15.0.dev0
- `nkilib` — the production kernel library, source of ported kernels
- trn2.3xlarge, 4 NeuronCores, LNC2, 96 GB HBM
- `neuron-explorer` for device-time profiling (torch-xla only so far)

**Scope:**

- Package NKI kernels for the HF Kernel Hub and validate a **stock** HuggingFace model running on
  Trainium with `use_kernels=True`
- Target: Qwen3 dense first, then Qwen3-MoE
- Deliverable is a PoC document plus a recommendation on whether Neuron should invest in first-class
  HF Kernel Hub support
- **Out of scope unless decided otherwise:** backward kernels, `torch.compile` (see risk 5 — this one
  should be challenged), multi-node

---

## Where we are, in five lines

The mechanism **works on Trainium with no patches to HuggingFace's code** — that was the PoC's main
question and it is answered. Three NKI kernels swap into stock Qwen3 dense and Qwen3-MoE across two
PyTorch stacks, with correctness proven by execution counters rather than inferred from output. Two
candidates now beat the compiler on device. The upstream ask is **one item, down from three**, because
two of the three turned out to be my own mistake. Performance is the weak half: the ops the Hub
intercepts most widely are the ops with least to gain, and both winning candidates are blocked from
real model shapes by the same unanswered question.

| | status |
|---|---|
| Mechanism works end to end | **done** — stock `use_kernels=True`, no patching |
| Kernels correct | **done** — 3 kernels, 2 model families, 2 stacks, 2 NKI versions |
| Speedup exists | **found** — 2 candidates, both shape-limited, both provisional |
| Performance competitive | **no** — reported honestly below |
| Upstream ask ready | **1 item**, blocked on a repo-home / access decision |
| PoC document | **live and current** |

---

## Ownership and deliverables

Effort is in working days. **Target** is relative because I do not know the intended end date — give me
the deadline and I will pin calendar dates. Everything in Phase 0 is complete.

| Phase | Milestone | Owner | Consulted | Priority | Effort | Target | Status | Links & notes |
|---|---|---|---|---|---|---|---|---|
| **0. Validate** | Verify the `"neuron"` device path and forward-swap mechanism | Ben | — | p0 | — | — | **Done** | Week 1 |
| 0 | NKI RMSNorm kernel, packaged + validated | Ben | — | p0 | — | — | **Done** | 11/11 cases, NKI execution asserted |
| 0 | NKI RoPE kernel — **real port** of `nkilib/core/embeddings/rope_hf.py` | Ben | — | p0 | — | — | **Done** | 20/20 + 6 guard cases |
| 0 | NKI SiLU kernel | Ben | — | p1 | — | — | **Done** | 9/9; `nl.silu` is native, nothing to port |
| 0 | Qwen3 dense end to end | Ben | — | p0 | — | — | **Done** | 9/2/2 swaps, logits cos_sim 1.000001 |
| 0 | Qwen3-MoE end to end, zero kernel changes | Ben | — | p1 | — | — | **Done** | cos_sim 1.000002. Needs `experts_implementation="batched_mm"` or MoE does not run on Neuron at all |
| 0 | MFU measurement, denominator stated | Ben | — | p0 | — | — | **Done** | 632 TFLOPS/device TensorEngine ÷ 2 for LNC2 = 316 |
| 0 | Root-cause the 206x slowdown | Ben | — | p0 | — | — | **Done** | Two framework caching bugs, 322x recovered |
| 0 | Find a speedup | Ben | Samir | p0 | — | — | **Done** | Flash attention 1.48x @ seq2048, 2.11x @ 3072 |
| 0 | Stand up the Native PyTorch stack | Ben | Samir | p0 | — | — | **Done** | Both integration gates were XLA artifacts |
| 0 | Fused RMSNorm+MLP (Samir's suggestion) | Ben | Samir | p1 | — | — | **Done** | 1.76x @ Qwen3-0.6B shape; blocked above `I=4096` |
| **1. Settle the framing** | **`torch.compile` viability on native** — flip `can_torch_compile=True` on one kernel, see if it traces | Ben | John | **p0** | 0.5–1 | Week 1 | **Not started** | Highest information-per-day left. If it traces, the dispatch findings may dissolve and the fusion analysis needs redoing under a different compiler |
| 1 | **Device-time profiling on native** — does NEFF+NTFF capture + `neuron-explorer` work there? | Ben | — | **p0** | 1–1.5 | Week 1 | **Not started** | Gates all of Phase 2. Every native number is wall clock until this works. Risk: may not be supported on a 0.1.0 stack |
| 1 | Measure Finding #24's value on native | Ben | — | p1 | 0.5 | Week 1 | **Not started** | Applied to both native runs, so its native contribution is currently unmeasured while I cite it as an upstream ask |
| **2. Make the wins real** | Re-measure flash attention on native, device time | Ben | — | p0 | 1 | Week 2 | Blocked on 1 | The project's headline result is currently XLA-only |
| 2 | **Wire attention through transformers' attention interface** | Ben | Samir | p0 | 2–3 | Week 2 | Blocked on 1 | Biggest credibility gap: the best result currently bypasses the Kernel Hub entirely |
| 2 | Re-measure fused RMSNorm+MLP on device rather than wall clock | Ben | — | p1 | 0.5 | Week 2 | Blocked on 1 | Makes 1.76x quotable |
| 2 | Find where XLA switches attention strategy (HLO dump either side of seq 4096) | Ben | compiler | p2 | 1 | Week 2–3 | Not started | The attention window closes because the *compiler* improves at 4096, not because the kernel degrades. Nobody here knew that threshold existed |
| **3. Upstream** | Mapping-entry PR — add `"neuron"` to `_KERNEL_MAPPING` | Ben | Samir | p0 | 0.5–1 | Week 3 | **Blocked on decision 3** | The one remaining upstream ask. Entry names a repo ID, so it needs the repo home settled |
| 3 | `_detect_target` CR or handoff package | Ben | John, NKI team | p0 | 0.5 | Week 3 | **Blocked on decision 2** | One decorator, 86x per call, still applies on native, benefits every eager NKI user |
| 3 | Hub upload spike — variant dirs vs our flat layout | Ben | Samir | p1 | 1–2 | Week 3 | **Blocked on decision 3** | `kernel-builder` has no Neuron target; our layout only loads via a fallback path |
| **4. SPMD** | **Can a per-layer swap express a multi-core launch?** Prototype | Ben | kernels + nkilib | **p0** | 2–3 | Week 4 | **Blocked on decision 4** | The gating question for the entire performance story. Real chance the answer is "not from here", which is still worth knowing |
| 4 | If yes: re-measure both winning candidates multi-core | Ben | — | p0 | 2 | Week 4 | Blocked | Both candidates were *designed* for this configuration; current numbers are their handicapped case |
| **5. Stretch** | NKI kernel for the MoE routing `sort`/`histc` | Ben | nkilib | p2 | 3–5 | Week 5 | Not started | Best-scoped new kernel work. Unblocks the default Qwen3-MoE path, which does not run on Neuron today. The compiler error itself recommends NKI for it. Blocked by none of our findings |
| 5 | Backward kernels | Ben | Pinak | p2 | 1–2 wks | — | **Out of scope** | Only if training matters — decision 6 |
| **6. Wrap** | Final PoC document, handoff, review pass | Ben | John, Pinak | p0 | 1–2 | Week 5–6 | In progress | `deliverables/poc-document.md` is live and current |

Phases 1–3 are about **two weeks**. Phase 4 is another week and carries genuine risk of returning a
negative answer.

**Why Phase 1 goes first even though it looks less productive:** both tasks test assumptions the rest of
the plan rests on. Learning in a day that the eager framing is wrong is much cheaper than learning it in
a fortnight.

---

## Status detail

### Validated

| what | evidence |
|---|---|
| 3 kernels in the Kernel Hub's single-file format | `kernels/neuron_{rmsnorm,rope,silu}/` |
| Stock `use_kernels=True` on native, **no patching** | shim asserted absent before the test runs |
| All 3 kernels on the native compiler + NKI 0.6.0b1 | cos_sim 0.999983 / 0.999980 / 1.000002 |
| Two framework dispatch bugs found, fixed, accuracy-neutral | 52.25 → 0.162 ms/call, **322x** |
| Kernels are provably optimal for unfusable ops | marginal HBM traffic = 1.00x the theoretical floor |
| The device gap is not a compiler-flag artifact | 5 flag settings; the quantity a flag would move is already at its minimum |

### Performance, stated carefully

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

The two winning candidates, both **shape windows** rather than thresholds:

| candidate | wins | loses | measured on |
|---|---|---|---|
| `nkilib` flash attention | 1.48x @ seq 2048, **2.11x @ 3072** | 2.01x @ 512, 1.79x @ 4096 | device time, torch-xla |
| fused RMSNorm+MLP | **1.76x** @ H=1024/I=3072 | 1.45x @ H=4096/I=4096 | wall clock, native |

---

## Problems and risks

| # | problem | impact | mitigation |
|---|---|---|---|
| 1 | Two stacks, and the numbers disagree in **direction** | any figure quoted without its stack misleads | every number records its stack; a warning sits above the headline in `results/README.md` |
| 2 | No device-time profiling on native | can't attribute the 1.76x; all native numbers provisional | Phase 1 |
| 3 | Both winning candidates only win at shapes nobody deploys | wins are real and currently unusable | `I<=4096` boundary unchanged across two compiler generations → routes to the SPMD question |
| 4 | Best result **bypasses the Kernel Hub** | strongest number isn't yet a Kernel Hub result | Phase 2 |
| 5 | Project is eager-only, which is partly my assumption not a finding | may have measured the configuration that matters least | Phase 1. Compile is ~23% MFU vs ~5% eager on Neuron's own figures |
| 6 | Hub publishing never exercised | can't ship kernels even with the mapping entries | Phase 3 |
| 7 | I have revised the headline **five times** | credibility risk if a sixth revision lands after it reaches the kernels team | decision 1 — review before it ships |

On #7: each revision came from measuring something the previous version had assumed, and every
measurement was individually correct. The pattern is documented rather than buried, and is arguably the
most transferable output of the project.

---

## Decisions needed

| # | decision | Owner | Consulted | Priority | Blocks | My recommendation |
|---|---|---|---|---|---|---|
| 1 | **Review the recommendation before it goes to Hanbo / Karthick** | John, Pinak | — | **p0** | the final deliverable | do this before anything ships |
| 2 | Do I send a CR against a core SDK component, or hand it over with the reproducer? | John | NKI team | p0 | the 86x `_detect_target` fix | hand over with reproducer, offer to write it. Candidate owners from the Neuron Expert Directory: `ggandii` (NKI APIs), `qieqingy` (NKI repo), `zhehongb` (escalation) |
| 3 | Publishing under `aws-neuron` on HF — our call or up a level? Who asks HF for the trusted-publisher flag? | Pinak | Samir, John | p0 | Hub upload + the mapping PR | `aws-neuron/` with `trust_remote_code=True` and a TODO, mirroring what transformers already ships for `Atlas-Inference` |
| 4 | Who owns the multi-core SPMD question? | John | kernels, nkilib | p0 | the entire performance story | route out — I cannot answer it alone |
| 5 | Am I still in scope? I am two layers below the Kernel Hub now | John, Pinak | — | p1 | how I spend remaining time | happy either way, but worth an explicit call |
| 6 | Does training matter for the beta? All 3 kernels are inference-only | Pinak | — | p2 | whether backward kernels are in scope | out of scope unless you say otherwise |

---

## Open questions for other teams

Not blocking me, but they need owners.

| question | Team | Notes |
|---|---|---|
| Can a NKI custom call participate in compiler fusion? | compiler | Decides whether small memory-bound ops can ever win. Each swap currently *removes* a fusion the compiler was already doing |
| Is the `intermediate_size > 4096` single-core limit intended? | nkilib | Unchanged across two compiler generations → looks like a design boundary, not a bug. Excludes every deployed model |
| Why is native eager 3–4x slower than the XLA graph path? | frameworks | Largest single number in my tables and the first thing anyone will ask |
| Should `torch_neuronx` set a `torch.neuron` attribute on the XLA stack? | frameworks | Would fix dependency declaration for XLA users. Native already does it |
| `neuronx-cc` missing from `PATH` **hangs forever** with no error | runtime | Worst customer-experience issue found. Cost ~1h and nearly had me replace a host driver to fix a `PATH` bug. The information for a perfect error message is already at the failure point |
| `experts_implementation="batched_mm"` needed for MoE on Trainium | docs | Undocumented, and without it Qwen3-MoE does not run at all |

---

## Changelog

**2026-08-05**
- Stood up the Native PyTorch stack. **Both integration gates I had reported were artifacts of running
  on torch-xla** — the proposed transformers device-resolution patch is withdrawn. Stock
  `use_kernels=True` works unpatched. Upstream ask drops from 3 items to 1.
- Re-ran MFU on native: the headline *sign* flips to "1.97x faster", which is **not** a win — the
  baseline is 4.32x worse. Documented as a trap rather than a result.
- Tested Samir's fused RMSNorm+MLP: **1.76x** at Qwen3-0.6B's MLP shape, 1.45x slower at H=4096/I=4096.
- Re-tested the `I>4096` compile boundary on the new compiler: **unchanged**, so every deployed model is
  still excluded.
- Corrected my own repo-home recommendation. The Hub trust gate is a settable org flag, not a hardcoded
  `kernels-community` check, and a mapping entry can bypass it anyway — transformers already ships that
  pattern. `aws-neuron` already exists as an HF org.
- Named the eager-only framing as a limitation rather than a given. Part of what kept me there was an
  XLA-stack blocker I never re-checked on native.
- Rewrote the PoC document as a living doc; merged the branch to `main`.

**Earlier** — root-caused a 206x slowdown to two framework caching bugs (322x recovered), found the first
speedup (flash attention), and established why RMSNorm/RoPE/SiLU cannot win. Full history in
[`WORKLOG.md`](../WORKLOG.md).

---

**Deeper reading:** [`deliverables/poc-document.md`](poc-document.md) is the live technical document.
[`results/README.md`](../results/README.md) has every number with provenance.
[`docs/poc-findings.md`](../docs/poc-findings.md) is the full findings log, 34 entries.
