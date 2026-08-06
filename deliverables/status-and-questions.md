# HF Kernel Hub on Trainium — status, questions, and plan

**For:** Pinak Panigrahi (manager), John Gray (mentor)
**Owner:** Ben Bioren
**Updated:** 2026-08-05
**Purpose:** standing status doc. Where the project is, what I'm working on, what's blocked, what I
need decided, and the plan forward with sizing. Updated as things change — see the Changelog at the
bottom for what moved since you last read it.

**Deeper reading, if you want it:**
[`deliverables/poc-document.md`](poc-document.md) is the live technical document (recommendation,
findings, what's not done). [`results/README.md`](../results/README.md) has every number with
provenance. [`docs/poc-findings.md`](../docs/poc-findings.md) is the full findings log, 34 entries.

---

## 1. Where we are, in five lines

The Kernel Hub mechanism **works on Trainium with no patches to HuggingFace's code** — that was the
PoC's main question and it's answered. Three NKI kernels (RMSNorm, RoPE, SiLU) swap into stock Qwen3
dense and Qwen3-MoE, on two different PyTorch stacks, with correctness proven by execution counters
rather than inferred from output. Two candidates now beat the compiler on device. The remaining
upstream ask is **one item**, down from three, because two of the three turned out to be my own
mistake. Performance is the weak half: the ops the Hub intercepts most widely are the ops with least
to gain, and both winning candidates are blocked from real model shapes by the same unanswered
question.

| | status |
|---|---|
| Mechanism works end to end | **done** — stock `use_kernels=True`, no patching |
| Kernels correct | **done** — 3 kernels, 2 model families, 2 stacks, 2 NKI versions |
| Speedup exists | **found** — 2 candidates, both shape-limited, both provisional |
| Performance competitive | **no** — and honestly reported below |
| Upstream ask ready | **1 item**, blocked on a repo-home/access decision |
| PoC document | **live and current** |

---

## 2. What I'm working on right now

Re-validating everything on the **Native PyTorch** stack, because that's the one HuggingFace intends
and all my earlier work was on torch-xla. That switch already invalidated two of my three upstream
asks, so the re-validation is not bookkeeping.

Immediately next, and both are cheap and decisive — see Phase 1 in the plan:

1. **Does `torch.compile` work with these kernels on native?** I have this project scoped to eager
   mode, and I'm no longer confident that was a real constraint rather than an assumption I inherited
   from the XLA stack. One experiment settles it.
2. **Can I profile device time on native?** Every native number I have is wall clock, which means I
   can't separate framework overhead from real device work — and that separation is what my strongest
   conclusions rest on.

---

## 3. Status detail

### Done and validated

| what | evidence |
|---|---|
| 3 NKI kernels packaged in the Kernel Hub's single-file format | `kernels/neuron_{rmsnorm,rope,silu}/` |
| RoPE is a **real port** of production `nkilib/core/embeddings/rope_hf.py` | 20/20 cases + 6 guard cases |
| Qwen3 dense end to end | 9 RMSNorm / 2 RoPE / 2 SiLU swapped, logits cos_sim 1.000001 |
| Qwen3-MoE end to end, **zero kernel code changes** | logits cos_sim 1.000002 |
| Stock `use_kernels=True` on native, **no patching** | shim asserted absent in the test |
| All 3 kernels on the native compiler + NKI 0.6.0b1 | cos_sim 0.999983 / 0.999980 / 1.000002 |
| Two framework dispatch bugs found, fixed, verified accuracy-neutral | 52.25 → 0.162 ms/call, **322x** |
| A speedup exists | flash attention 1.48x @ seq2048, 2.11x @ 3072 (device) |
| A second speedup | fused RMSNorm+MLP 1.76x @ Qwen3-0.6B MLP shape (wall clock) |

### Performance, stated carefully

Qwen3-0.6B, 28 layers, bf16, forward only, single logical core, denominator 316 TFLOPS.

| seq | stack | baseline | kernelized | verdict |
|---|---|---|---|---|
| 512 | torch-xla | 43.94 ms | 71.32 ms | 1.62x slower |
| 512 | native | 189.97 ms | 96.46 ms | *1.97x "faster"* |
| 2048 | torch-xla | 117.78 ms | 161.04 ms | 1.37x slower |
| 2048 | native | 340.74 ms | 251.86 ms | *1.35x "faster"* |

**The native rows are a trap and I don't want them quoted alone.** The native baseline is 4.32x
slower than the XLA one, so the ratio flipped because the denominator got worse. Native kernelized
MFU (2.20%) is *below* XLA kernelized MFU (2.98%). Nothing got faster. Native fixed the integration
story, not performance.

---

## 4. Problems and risks

| # | problem | impact | mitigation |
|---|---|---|---|
| 1 | Two stacks, and my numbers disagree in **direction** between them | any figure quoted without its stack is misleading | every number now records its stack; a warning sits above the headline in `results/README.md` |
| 2 | No device-time profiling on native | can't attribute the 1.76x fused-MLP win; all native numbers provisional | Phase 1, task A2 |
| 3 | Both winning candidates only win at shapes nobody deploys | the wins are real and currently unusable | the `I<=4096` boundary is unchanged on the new compiler — routes to the SPMD question |
| 4 | The best result **bypasses the Kernel Hub** — attention is called directly | strongest perf number isn't yet a Kernel Hub result | Phase 2, task B2 |
| 5 | Project is eager-only, which is partly my assumption not a finding | may have measured the configuration that matters least | Phase 1, task A1 |
| 6 | Hub publishing never exercised | can't ship kernels even with the mapping entries | Phase 3, task C3 |
| 7 | I've revised the headline **five times** | credibility risk if the sixth revision lands after it reaches the kernels team | asking for a review before it goes out (decision 1) |

On #7, for transparency: each revision came from measuring something the previous version had
assumed, and every measurement was individually correct. The pattern is documented rather than
buried — it's arguably the most transferable output of the project.

---

## 5. Decisions I need

Ordered by how much they block.

| # | decision | who | blocks | my recommendation |
|---|---|---|---|---|
| 1 | **Review the recommendation before it goes to Hanbo / Karthick** | John + Pinak | the final deliverable | do this before anything else ships |
| 2 | Do I send a CR against a core SDK component, or hand it over with the reproducer? | John | the 86x `_detect_target` fix | hand over with reproducer, offer to write it |
| 3 | Publishing under `aws-neuron` on HF — our call or up a level? Who asks HF for the trusted-publisher flag? | Pinak | Hub upload + the mapping-entry PR | `aws-neuron/` with `trust_remote_code=True` and a TODO, mirroring what transformers already does for `Atlas-Inference` |
| 4 | Who owns the multi-core SPMD question? | John | the entire performance story | route to kernels/nkilib — I can't answer it alone |
| 5 | Am I still in scope? I'm two layers below the Kernel Hub now | John + Pinak | how I spend the remaining time | happy either way, but worth an explicit call |
| 6 | Does training matter for the beta? All 3 kernels are inference-only | Pinak | whether backward kernels are in scope | out of scope unless you say otherwise |

---

## 6. Plan forward

Sized in working days. **Phases 1–3 are about two weeks**; Phase 4 is another week and carries real
risk of returning "not possible from here", which is itself a useful answer.

> I've written this relative to now rather than against calendar dates, because I don't know the
> intended end date. Tell me the deadline and I'll pin it.

### Phase 1 — settle the framing (2–3 days) · do this first

Two of my conclusions rest on assumptions I can cheaply test. If either is wrong, the rest of the
plan is pointed the wrong way, so this goes first.

| task | days | why it matters |
|---|---|---|
| **A1.** `torch.compile` viability: flip `can_torch_compile=True` on one kernel, see if it traces on native | 0.5–1 | Decides whether eager-only is a constraint or a habit. If it traces, my dispatch findings may largely dissolve and the fusion analysis needs re-testing under a different compiler. Highest information-per-day in the project. |
| **A2.** Device-time profiling on native: does NEFF+NTFF capture + `neuron-explorer` work there? | 1–1.5 | Without it every native number is wall clock and unattributable. Gates all of Phase 2. Risk: may not be supported on the beta stack. |
| **A3.** Finding #24's value on native: re-run MFU without the fix | 0.5 | I applied the fix to both native runs, so its native contribution is currently unmeasured while I'm citing it as an upstream ask. |

### Phase 2 — make the wins real (4–6 days) · needs A2

| task | days | why |
|---|---|---|
| **B1.** Re-measure flash attention on native, device time | 1 | The 1.48x / 2.11x figures are XLA-measured. This is the project's headline result. |
| **B2.** Wire attention through transformers' attention interface | 2–3 | Turns the best result into an actual Kernel Hub result instead of a direct call. Biggest credibility gap in the deliverable. |
| **B3.** Re-measure fused RMSNorm+MLP on device rather than wall clock | 0.5 | Makes the 1.76x quotable. |
| **B4.** Find where XLA switches attention strategy (HLO dump either side of seq 4096) | 1 | The attention win has an upper edge because the *compiler* improves at 4096, not because the kernel degrades. Nobody here knew that threshold existed. |

### Phase 3 — unblock the upstream asks (2–4 days) · gated on decisions 2 and 3

| task | days | blocked by |
|---|---|---|
| **C1.** Mapping-entry PR to transformers (`"neuron"` entries) | 0.5–1 | decision 3 (repo home) |
| **C2.** `_detect_target` CR or handoff package | 0.5 | decision 2 |
| **C3.** Hub upload spike — variant dirs vs our flat layout, `kernel-builder` has no Neuron target | 1–2 | decision 3 (org access) |

### Phase 4 — the SPMD question (3–5 days, high risk) · needs decision 4

This is the gating question for the whole performance story, and it may not be answerable by me.

| task | days | notes |
|---|---|---|
| **D1.** Can a per-layer swap express a multi-core launch? Prototype against one nkilib kernel | 2–3 | Genuine chance the answer is "not from here", which is still worth knowing definitively |
| **D2.** If D1 works: re-measure both winning candidates multi-core | 2 | Both were *designed* for this configuration, so it's their unhandicapped case |

### Stretch, if time allows

| task | days | notes |
|---|---|---|
| **E1.** NKI kernel for the MoE routing `sort`/`histc` | 3–5 | Best-scoped new kernel work identified. Unblocks the default Qwen3-MoE path on Neuron, which currently doesn't run at all. The compiler error itself recommends NKI for it. Blocked by none of our findings. |
| **E2.** Backward kernels | 1–2 weeks | Only if training matters — decision 6 |

### Wrap

| task | days |
|---|---|
| **F1.** Final PoC document, handoff to the kernels team, review pass | 1–2 |

---

## 7. Open questions for other teams

Not blocking my work, but they need owners.

| question | team | why it matters |
|---|---|---|
| Can a NKI custom call participate in compiler fusion? | compiler | Decides whether small memory-bound ops can ever win. Right now each swap *removes* a fusion the compiler was already doing. |
| Is the `intermediate_size > 4096` single-core limit intended? | nkilib | Unchanged across two compiler generations, which suggests a design boundary rather than a bug. Excludes every deployed model. |
| Why is native eager 3–4x slower than the XLA graph path? | frameworks | Largest single number in my tables and the first thing anyone will ask. |
| Should `torch_neuronx` set a `torch.neuron` attribute on the XLA stack? | frameworks | Would fix dependency declaration for XLA users. Native already does it. |
| `neuronx-cc` missing from `PATH` **hangs forever** with no error | runtime | Worst customer-experience issue I found. Cost me ~1h and nearly had me replace a host driver to fix a `PATH` bug. Information for a perfect error message is already at the failure point. |
| `experts_implementation="batched_mm"` needed for MoE on Trainium | docs | Undocumented, and without it Qwen3-MoE doesn't run at all |

---

## Changelog

**2026-08-05**
- Stood up the Native PyTorch stack. **Both integration gates I'd reported turned out to be
  artifacts of running on torch-xla** — the proposed transformers device-resolution patch is
  withdrawn. Stock `use_kernels=True` works unpatched. Upstream ask drops from 3 items to 1.
- Re-ran MFU on native: the headline *sign* flips to "1.97x faster", which is **not** a win — the
  baseline is 4.32x worse. Documented as a trap rather than a result.
- Tested Samir's fused RMSNorm+MLP: **1.76x faster** at Qwen3-0.6B's MLP shape, 1.45x slower at
  H=4096/I=4096. Second winning candidate, second shape window.
- Re-tested the `I>4096` compile boundary on the new compiler: **unchanged**, so every deployed model
  is still excluded.
- Corrected my own repo-home recommendation. The Hub trust gate is a settable org flag, not a
  hardcoded `kernels-community` check, and a mapping entry can bypass it anyway — transformers
  already ships that pattern. `aws-neuron` already exists as an HF org.
- Named the eager-only framing as a limitation rather than a given. Part of what kept me there was an
  XLA-stack blocker I never re-checked on native.
- Rewrote the PoC document as a living doc; merged the branch to `main`.

**Earlier** — root-caused a 206x slowdown to two framework caching bugs (322x recovered), found the
first speedup (flash attention), and established why RMSNorm/RoPE/SiLU cannot win. Full history in
[`WORKLOG.md`](../WORKLOG.md).
