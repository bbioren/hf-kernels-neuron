# HuggingFace Kernel Hub on Trainium — PoC findings and recommendation

**LIVING DOCUMENT.** This is the current state of the work, not a report written at the end. It is
updated as things change, including when a previous conclusion turns out to be wrong — which has
happened five times, and the corrections are kept visible rather than edited away. If a number here
disagrees with something said earlier in Slack or in `deliverables/week-*.md`, this file wins.

| | |
|---|---|
| **Last updated** | 2026-08-05 |
| **Hardware** | trn2.3xlarge, 1 device, 4 NeuronCores, LNC2, 96 GB HBM |
| **Stacks measured** | torch-xla (weeks 1–6) **and** Native PyTorch (current) — numbers are **not** interchangeable |
| **Raw artifacts** | [`results/`](../results/README.md), regenerate with `make results` |
| **Full findings log** | [`docs/poc-findings.md`](../docs/poc-findings.md) — 33 findings, this document summarises |
| **Code guide** | [`docs/CODE_GUIDE.md`](../docs/CODE_GUIDE.md) |

---

## Recommendation

**Yes, this is worth investing in — but point it at attention, not at normalisations and
activations, and answer one question first.**

Three parts, in order of how much they should drive the decision.

**1. The mechanism works, and it now costs HuggingFace exactly one small change.**

On the Native PyTorch stack, stock `use_kernels=True` swaps NKI kernels into a stock Qwen3 with **no
patching of any kind**: 9 RMSNorm, 2 RoPE, 2 SiLU, zero fallbacks, logits `cos_sim 1.000001`. The
two gates this project reported for weeks — `model.device.type` never being `"neuron"`, and
`hasattr(torch, "neuron")` being False — were both artifacts of our own use of torch-xla. They do not
exist on the stack HuggingFace intends.

What remains is adding `"neuron"` entries to transformers' `_KERNEL_MAPPING` and
`_FUNCTION_KERNEL_MAPPING`. `"rotary_pos_emb"` already has `cuda` / `rocm` / `xpu` siblings, so it is
a one-block addition, not an architectural change. Everything else in the chain already exists
upstream and is already wired into the models.

**2. The leverage argument holds, but reach and benefit are inversely correlated.**

Three kernels cover the highest-count interception points in transformers: 115 `RMSNorm`
registrations, 95 model files for `rotary_pos_emb`, and one activation decoration in
`activations.py` that covers every model using `ACT2FN["silu"]`. Per-kernel work genuinely scales
across the model zoo — demonstrated, not assumed: all three kernels transferred from Qwen3 dense to
Qwen3-MoE with **zero code changes**.

And those same ops are the ones with the least to gain. They are small, memory-bound, and already
fused by the compiler, so replacing one with an opaque custom call *removes* an optimisation that was
already happening. That is measured, not argued: on torch-xla our kernels move exactly the
theoretical minimum HBM traffic for an unfusable op, while torch's traffic is independent of chain
length — only possible if the whole chain collapsed into one pass.

So the uncomfortable shape of this PoC is that the mechanism's reach and its usefulness point in
opposite directions. That is a reason to aim it differently, not to abandon it.

**3. Two candidates do beat the compiler, and both are blocked by the same question.**

| candidate | result | where |
|---|---|---|
| `nkilib` flash attention (`attention_cte`) | **1.48x faster** at seq 2048, **2.11x** at 3072 | device time, torch-xla |
| `nkilib` fused RMSNorm+MLP (`NormType.RMS_NORM`) | **1.76x faster** at H=1024/I=3072 | wall clock, native |

Both win for the reason the analysis predicted: they replace a region the compiler does *not* already
fuse well, and they contain real arithmetic to restructure. Flash attention never materialises the
`[heads, S, S]` score matrix — a compiler fuses elementwise chains, it does not re-derive the
algorithm.

Both are also **shape windows, not thresholds.** Attention loses 2.01x at seq 512 and 1.79x at 4096.
The fused MLP loses 1.45x at H=4096/I=4096. Quoting either number without its shape is the single
most likely way this work gets misrepresented.

And both are `nkilib` kernels designed for a **multi-core SPMD launch**, being run single-core
because that is what a per-layer forward swap gives them. That is the gating question:

> **Can `kernelize()` express a multi-core launch?**

If yes, both candidates should improve and may extend to real model shapes. If no, per-layer swapping
tops out near parity for small ops and at toy shapes for fused ones. This question sits above weight
layout, above the compile boundary, and above every dispatch fix — those are all downstream of it.

### What would change this recommendation

- **If the SPMD launch cannot be expressed**, the honest conclusion narrows to: the Kernel Hub is a
  correct and cheap *compatibility* mechanism for Neuron, worth the one upstream entry, but not a
  performance story. Invest in the entries and stop there.
- **If the `I > 4096` single-core compile boundary is lifted**, the fused-MLP result extends to real
  models and the case gets substantially stronger on its own.
- **If native eager closes its gap to the XLA graph path** (see the performance section — it is
  currently 3–4x slower), every relative number here shifts.

---

## Where we are struggling right now

Kept current deliberately, because the honest state of a PoC includes what is not working.

**1. Two measurement stacks, and the numbers disagree in direction.** Everything through week 6 was
measured on torch-xla. The Native PyTorch drop is what HuggingFace intends, and on it the *sign* of
the headline flips: kernels go from 1.62x slower to 1.97x faster. That is not an improvement — the
native baseline is 4.32x slower than the XLA one, so the ratio moved because the denominator got
worse. Every performance claim now needs its stack attached, and the temptation to quote the
flattering one is a live risk rather than a hypothetical.

**2. No device-time profiling on native.** On torch-xla we can separate dispatch cost from device
work with `neuron-explorer` on a NEFF+NTFF pair, and that separation is what the strongest findings
rest on. It is not wired up for native. So the fused RMSNorm+MLP's 1.76x is wall clock, includes
dispatch, and cannot yet be attributed. It bounds the answer instead of settling it.

**3. The two winning candidates are at shapes nobody deploys.** The fused MLP wins at
`intermediate_size = 3072` (Qwen3-0.6B) and the single-core compile boundary is still exactly
`I <= 4096` — re-tested on the new compiler and unchanged. Qwen3-8B is 12288. Llama-3-8B and
Mistral-7B are 14336. So the result is real and currently unusable.

**4. Hub publishing is unproven and may not be reachable.** The loader expects
`build/<variant>/` directories; our flat Python-only layout works through a fallback path;
`kernel-builder` has no Neuron target at all. Nothing has been uploaded. Kernels also live in a
distinct `repo_type="kernel"` on the Hub, and `aws-neuron` currently has zero of them. The trust gate
is a solvable org flag rather than a blocker (see ask 2), but nothing about the *upload* path has been
exercised end to end.

**5. Attention wins by bypassing the thing this PoC is about.** The best result came from calling
`attention_cte` directly, not through the Kernel Hub. Wiring it through transformers' attention
interface is not done, so the strongest performance result is not yet a Kernel Hub result.

**6. Open questions we cannot answer from here.** Whether a NKI custom call can participate in
compiler fusion (compiler team). Whether the `I > 4096` boundary is intended (nkilib team). Whether
native has an equivalent of the torch-xla per-call lowering cost — unmeasured, and it should not be
assumed either way.

---

## What was built and validated

Three kernels, packaged in the Kernel Hub's single-file format, each `metadata.json` + `__init__.py`:

| kernel | interception point | upstream reach | source |
|---|---|---|---|
| `neuron_rmsnorm` | `RMSNorm` layer | 115 registrations | NKI tutorial, migrated to NKI 0.5.0 `nl.ds` |
| `neuron_rope` | `rotary_pos_emb` function | 95 model files | **real port of `nkilib/core/embeddings/rope_hf.py`** |
| `neuron_silu` | `SiLU` layer | every `ACT2FN["silu"]` model | `nl.silu` native |

Validated on both stacks. Every accuracy result asserts, via a call counter, that the NKI branch
actually executed — because the first version of this work reported a flawless `max_diff = 0.00e+00`
while the kernel had never run once.

| | torch-xla | Native PyTorch |
|---|---|---|
| Qwen3 dense e2e | 9 / 2 / 2 swaps, `cos_sim 1.000001` | 9 / 2 / 2 swaps, `cos_sim 1.000001` |
| Qwen3-MoE e2e | 9 / 2 / 2 swaps, `cos_sim 1.000002`, zero code changes | not re-run |
| isolated kernels | 11/11, 20/20 + 6 guards, 9/9 | RMSNorm 0.999983, SiLU 0.999980, RoPE q 1.000002 / k 1.000001 |
| `use_kernels=True` | blocked — needed a patch | **works unpatched** |
| NKI version | 0.5.0 | 0.6.0b1 |
| compiler | neuronx-cc 2.26.6360.0 | neuronx-cc 2.0.266551.0a0 |

Two incidental results worth keeping:

- **Qwen3-MoE does not run on Neuron at all by default.** The experts path uses `torch.sort` and
  `torch.histc`, which lower to an unsupported `sort` HLO. The fix is
  `experts_implementation="batched_mm"`, which is documented nowhere. Worth surfacing in Neuron's
  model-support docs regardless of what happens with the Kernel Hub.
- **`nkilib` is already installed in both venvs and its kernels are directly callable** from PyTorch,
  which makes thin wrappers feasible without vendoring. Hand-porting does not scale: RoPE needed ~15
  lines inlined, while the MLP kernel's dependency closure is 7,249 lines across 22 files.

---

## Performance: the honest picture

### The cross-stack table, which is the only correct way to read this

Qwen3-0.6B, 28 layers, bf16, forward only, single logical core. Denominator: 632 TFLOPS/device
(TensorEngine only, per the Cayman arch doc) ÷ 2 for LNC2 = **316 TFLOPS per logical core**. 169 NKI
launches per step, zero fallbacks. Both stacks have the dispatch fix applied.

| seq | stack | baseline | kernelized | verdict | baseline MFU | kernelized MFU |
|---|---|---|---|---|---|---|
| 512 | torch-xla | 43.94 ms | 71.32 ms | 1.62x slower | ~4.9% | 2.98% |
| 512 | native | 189.97 ms | 96.46 ms | *1.97x faster* | 1.12% | 2.20% |
| 2048 | torch-xla | 117.78 ms | 161.04 ms | 1.37x slower | ~9.9% | 6.69% |
| 2048 | native | 340.74 ms | 251.86 ms | *1.35x faster* | 3.16% | 4.28% |

**Read down the columns, not across the rows.** The native baseline is 4.32x slower than the XLA one
at seq 512. The native kernelized run is 1.35x slower than the XLA kernelized run. Native kernelized
MFU (2.20%) is *below* XLA kernelized MFU (2.98%). Nothing got faster on native; the baseline got
much slower, which flipped a ratio.

This matters beyond bookkeeping, because the mechanism is informative. On torch-xla the baseline
enjoys whole-forward graph fusion — that is why our kernels lose there, since an opaque custom call
blocks it. Native is eager, with no equivalent whole-graph pass, so there is far less fusion to lose
and the barrier stops costing much. **The fusion penalty only exists where there is fusion.** That
is coherent, and it does not make the kernels good.

### Dispatch: two framework bugs, 322x combined

Both on torch-xla, both diagnosed with verified accuracy-neutral fixes, neither shipped.

| | per call | cumulative |
|---|---|---|
| as found | 52.25 ms | — |
| after caching `_detect_target` | 0.605 ms | 86.3x |
| after registering the XLA computation once per cache key | **0.162 ms** | **322.5x** |
| device time floor | 0.0495 ms | |

Both are the same bug twice: **a cache exists and the surrounding code path defeats it.**

- `nki/compiler/target.py::_detect_target()` shells out to `neuron-ls` on every kernel invocation.
  It cannot be served by NKI's compile cache because its result is part of the cache *key*, so a
  cache hit still pays the full subprocess. One `lru_cache` fixes it. **This one still applies on
  native** — the subprocess is still forked there — and it is not Kernel-Hub specific: anyone
  invoking NKI kernels per-layer from eager PyTorch is paying it right now.
- `torch_xla`'s `Op` class already memoises the built computation, and its own docstring asks callers
  to register ops globally "to amortize the lowering cost". NKI applies `@xla_hlo_call` *inside*
  `__call__`, so a fresh `Op` with an empty memo is constructed per call. **This one does not port to
  native** — `torch_xla` is not importable there.

### The two winning candidates

**Flash attention** (`nkilib/core/attention/attention_cte`), device time, torch-xla, Qwen3-0.6B head
geometry, causal:

| seq | NKI ms/layer | torch ms/layer | verdict | NKI MB/layer | torch MB/layer |
|---|---|---|---|---|---|
| 512 | 0.2463 | 0.1225 | 2.01x slower | 8.39 | 3.16 |
| 1024 | 0.4939 | 0.4269 | 1.16x slower | 16.78 | 13.12 |
| **2048** | **1.1438** | 1.6902 | **1.48x FASTER** | 33.55 | 279.86 |
| **3072** | **1.8484** | 3.9062 | **2.11x FASTER** | 50.33 | 748.70 |
| 4096 | 2.8295 | 1.5784 | 1.79x slower | 67.11 | 395.05 |

NKI's traffic is exactly linear in sequence length — textbook flash attention. The window's *lower*
edge is the compiler already keeping the score matrix resident at short sequences. The *upper* edge
is more interesting: torch's traffic **drops 47%** from 3072 to 4096 while the score matrix grows
from 302 to 537 MB, so the compiler evidently switches attention strategy somewhere in between. The
kernel does not degrade — its numbers stay exactly on trend. Nobody on this project knew that
threshold existed, and finding it properly means dumping HLO either side of it.

**Fused RMSNorm+MLP** (`nkilib.core.mlp.mlp` with `normalization_type=NormType.RMS_NORM`), wall
clock, native. Samir Araujo pointed this out after reading the week 3–6 findings, and he was right
that we had missed it — we had always used the `NO_NORM` default and wrongly generalised "nkilib's
RMSNorm always fuses quantisation" from `core/rmsnorm/` to the whole library.

| shape | NKI fused | torch | verdict | cos_sim |
|---|---|---|---|---|
| H=1024 I=3072 (Qwen3-0.6B) | **0.6095 ms/block** | 1.0728 ms/block | **1.76x FASTER** | 0.999973 |
| H=4096 I=4096 | 2.5377 ms/block | 1.7459 ms/block | 1.45x slower | 0.999967 |

It replaces six torch ops with one call and absorbs an RMSNorm that would otherwise be its own
optimisation barrier. Provisional: wall clock, so it carries the degraded-native-baseline caveat, and
the NKI side pays one dispatch where torch pays six.

**The blocker is unchanged.** `intermediate_size > 4096` still fails to compile single-core, with the
same `'floordiv' does not allow division by zero` in the CTE tile arithmetic, on a compiler two
generations from the original measurement:

| H | I | result |
|---|---|---|
| 1024 | 3072 | PASS |
| 1024 | 4096 | PASS |
| 1024 | 5120 | FAIL |
| 4096 | 4096 | PASS |
| 4096 | 12288 | FAIL |

Two compiler generations agreeing on the exact threshold is strong evidence this is a **design
boundary** — the kernel is telling us single-core is not its execution model — rather than an
arithmetic bug to fix. Which routes straight back to the SPMD question.

---

## What we are asking for

Ordered by value per unit of effort.

**1. HuggingFace: add the `"neuron"` mapping entries.** The only remaining upstream blocker. One
block in `_KERNEL_MAPPING` and one in `_FUNCTION_KERNEL_MAPPING`. Blocked on the repo-home decision
below, because the entries name repo IDs.

**2. Publish under `aws-neuron/`, and ask HuggingFace for the `trustedKernelPublisher` flag in
parallel.** This replaces an earlier, worse-informed ask that framed this as a choice between
default-path trust and versioning control. It is neither a choice nor blocking.

The Hub trust gate is not a hardcoded `kernels-community` check —
`kernels/utils.py::_check_trust_remote_code` queries an org-level `trustedKernelPublisher` boolean via
the Hub API, and `kernels-community` simply has it set. Verified live: `kernels-community` reports
`trustedKernelPublisher: true` with 56 kernels, while `aws-neuron` already exists as a joint AWS/HF org
(31 models, 180 followers) with 0 kernels and no flag.

And an untrusted org does not block the default path anyway, because a mapping entry can declare its
own trust — which **upstream transformers already does**, for precisely this situation:

```python
LayerRepository(
    repo_id="Atlas-Inference/gdn",
    layer_name="Qwen3_5GatedDeltaNet",
    revision="ef12347fc77d6ddf1cb72c0bd0af1c7d6cc69172",
    # TODO: drop once Atlas-Inference is an allow-listed trusted publisher
    trust_remote_code=True,
)
```

So the recommendation is to mirror that exactly: ship the `"neuron"` entries against `aws-neuron/` with
`trust_remote_code=True` and the same kind of TODO, pin `revision=<commit sha>` rather than a mutable
`version=N` branch while the bypass is in place, and drop both once the flag is granted. Versioning
control comes free, since a version is just a `v<N>` branch and control is ordinary Hub write
permission.

Worth asking HF's criteria for the flag, since `numKernels: 0` may itself be the blocker — they may
reasonably want to see working published kernels first, which makes it a sequencing question. Note also
that `trust_remote_code=[...]` is *not* a per-repo allowlist: it is for signing identities, is
unimplemented, and warns then falls back to the publisher check.

**3. NKI: cache `_detect_target`.** One decorator, 86x per call, accuracy-neutral by measurement,
still applies on native, and benefits every eager NKI user rather than just this integration. This is
the best value-per-effort item in the project and it is correct regardless of everything else here.
Reproducer: `scripts/probe_target_override_fix.py`.

**4. NKI: register the XLA computation once per compile-cache key.** A further 3.7x on torch-xla.
Reproducer: `scripts/probe_op_registry_cache.py`, which asserts five structural landmarks in the
installed source and refuses to patch an unrecognised version.

**5. nkilib / kernels teams jointly: can a per-layer swap launch multi-core SPMD?** The gating
question for the whole performance story. Both winning candidates were built for a multi-core grid
and are being handicapped by the integration model.

**6. nkilib: document the `I <= 4096` single-core constraint, and give it a real diagnostic.**
Currently a divide-by-zero deep in tile arithmetic on a legal, documented-as-supported input.
Re-filed with two-compiler evidence.

**7. Neuron: raise an error when `neuronx-cc` is not on `PATH`.** On the native stack the first
operation needing a compile **hangs forever** with no diagnostic — the runtime forks, `execve`s
`neuronx-cc` by bare name, gets `ENOENT` on all seven `PATH` entries, and the child blocks before it
can report anything. It looks exactly like a driver or version problem, and the natural next step is
to start replacing host packages. We came one step from doing that. The information for a perfect
error message is already in hand at the failure point.

**8. Neuron docs: `experts_implementation="batched_mm"` for MoE on Trainium.** Undocumented, and
without it Qwen3-MoE does not run at all.

---

## What is measured, and at what confidence

Included because the headline has been revised five times, and a reader deserves to know which
claims to push on.

| claim | confidence | basis |
|---|---|---|
| Kernel Hub mechanism works on Neuron, unpatched, on native | **high** | e2e on Qwen3, shim asserted absent, execution counters |
| All three kernels correct on both stacks | **high** | call counters + negative controls + two NKI versions |
| Kernels transfer across model families unchanged | **high** | Qwen3 dense and MoE, zero code changes |
| Dispatch was 322x recoverable on torch-xla | **high** | two verified fixes, controls re-run last, bit-identical output |
| Our kernels are optimal for unfusable ops | **high** | marginal traffic = 1.00x the theoretical floor, solved by regression over two call counts |
| The device gap is not a compiler-flag artifact | **high** | five flag settings; the quantity a flag would move is already at its minimum |
| Attention wins at seq 2048–3072 | **medium-high** | device time, reproduced across two runs to 4 s.f. |
| Fused RMSNorm+MLP wins at H=1024/I=3072 | **medium** | wall clock only; dispatch not separated; single run |
| Native eager is 3–4x slower than the XLA graph path | **medium** | two sequence lengths, tight IQRs; mechanism not directly verified |
| Native has no whole-forward fusion | **low — explanation, not result** | consistent with the numbers and with eager dispatch; the direct traffic-scaling test needs profiling not wired up on native |
| Multi-core SPMD would improve both candidates | **low — untested** | inferred from kernel design and from Finding #26 |

---

## The methodological finding, which may be the most transferable output

Five times in this project a confident, plausible, well-supported conclusion was wrong. They fall
into two classes, and the second is much more dangerous.

**Class one: the measurement was invalid, and a guard catches it.**

| | looked like | actually was |
|---|---|---|
| accuracy | "RMSNorm validated, bit-identical" | kernel never ran; fallback compared against itself |
| benchmark | "NKI is 8–400x slower" | outputs discarded, so XLA eliminated the computation — timed an empty graph |
| torch.compile | "NKI is incompatible with torch.compile" | our loader never registered the module in `sys.modules` |

The common thread: **on a lazy-execution accelerator backend, both correctness and performance
measurements fail silently by default.** A fallback is numerically correct. An eliminated computation
is fast. Nothing errors. The guards now in the harness — execution call counters, a scaling gate that
suppresses overhead-dominated ratios, mandatory controls, negative controls — exist because each of
these cost a cycle, and the second was caught *by* the guard built after the first.

**Class two: the measurements were all valid and the conclusion was still wrong.** No guard catches
this, because nothing is broken.

The 52 ms per-call cost was real, flat across a 112x sweep of problem size, and reproduced five times
within 1%. It was attributed to graph-transition overhead, and that hypothesis survived four
experiments: varying interleaving, varying data volume, ruling out recompilation, and swapping our
kernels for production ones. Every result came back consistent with it.

It survived because **all four measured wall-clock time at the framework level and none could see
inside the 52 ms.** No further variant of that instrument could have falsified it. What did: a device
profile (0.609 ms device against 1459 ms wall — a 2400x gap that invalidated every device-side
explanation at once), then a Python profile that named the function in one run. Thirty-five minutes,
after about five hours of framework-level experiments.

Three practices came out of it, and they generalise past this project:

1. **When a hypothesis has survived several tests and the story still does not close, change
   instrument rather than adding another variant.** Repeated survival is evidence about the
   instrument as much as about the hypothesis.
2. **Measure the two ends against each other early.** Device time versus wall time is one number
   each. Their ratio invalidated a whole class of explanation. It should have been first, not fifth.
3. **Enumerating candidates inside one framing feels like rigour and is not.** When the cost was
   believed to be inside the NEFF, three candidate explanations were written out and ranked. All
   three were device-side, because the framing had already concluded that. The true answer was not
   ranked low — it was absent.

Two further instances are worth naming because each had a different mechanism:

- **A correct calculation applied to a non-linear quantity.** Dividing HBM traffic by call count said
  our kernels moved 3.00x more data than necessary, which reads as a spilled intermediate — and a
  recent kernel change had introduced exactly such a temporary, so there was a ready culprit. Landing
  on exactly 3.00x for two different ops independently is the tell. Traffic is not linear in N, and
  solving `traffic(N) = FIXED + N × MARGINAL` shows the kernels are optimal. **Vary N before dividing
  by N.**
- **A ratio moving because its denominator moved.** Twice. Once with `--lnc 1`, where the best
  NKI/torch ratio in the table was NKI standing still while torch got 91% slower. Then again at model
  scale on native, where a 1.97x "speedup" is a 4.32x slower baseline. **A ratio is two numbers.**

And one that is about writing rather than measuring: **a caveat in the text is not a caveat in the
conclusion.** The limitation "this chained microbenchmark is NKI's worst case" was written into the
findings document *before* the recommendation was drafted, and the recommendation then reasoned from
the number as though the caveat were not there. Either measure the thing the caveat is about, or let
it constrain the claim.

---

## What is not done

- **Backward kernels.** All three declare `has_backward = False`, so training mode falls back. If
  training matters for the beta, this is real work: `rope_hf` has a backward path, `nl.silu_dx`
  exists, RMSNorm backward would need writing.
- **`torch.compile`.** All three declare `can_torch_compile = False`. Separately, `torch_neuronx`'s op
  overrides are not fake-tensor safe, which breaks `torch.compile` on nearly every transformer —
  the override list includes `Embedding`, `Softmax` and `CrossEntropyLoss`. Filed with a reproducer;
  outside this project's scope.
- **Hub upload.** Nothing published. Blocked on the repo-home decision, and possibly on the loader's
  variant-directory expectation.
- **Attention through the Kernel Hub.** The best performance result bypasses the mechanism this PoC is
  about. Wiring it through transformers' attention interface is the obvious next step.
- **Device-time profiling on native.** Which is why the fused-MLP number is provisional.
- **Finding #24's value on native.** It was applied to both native runs, so its contribution there is
  unmeasured. Cheap to get.
- **Multi-core SPMD anything.** The largest gap, and the gating question above.
- **Qwen3-MoE on native**, and any MoE-specific kernel. The best-scoped MoE work identified is a NKI
  kernel for the routing `sort`/`histc` step — it unblocks the default MoE path, the compiler error
  itself recommends NKI for it, and it is blocked by none of our findings.

---

## Closing assessment

The mechanism is sound and cheap. A forward-method swap is the right interception point for Neuron,
it works on the supported stack with no changes to HuggingFace's code, and the entries needed to make
it reachable are a one-block addition. Correctness is not in question: three kernels, two model
families, two stacks, two NKI versions, execution proven rather than inferred.

The performance story is narrower than the mechanism, and pointed the wrong way by default. The ops
the Kernel Hub intercepts most widely — normalisations, rotary embeddings, activations — are exactly
the ops that lose from being intercepted, because they are small, memory-bound, and already fused.
The ops that win are large fused regions with real arithmetic: flash attention, and a fused
RMSNorm+MLP. Both of those exist in `nkilib` today, both beat the compiler in a measured shape
window, and both are being run in a configuration they were not designed for.

So the recommendation is not "yes" or "no" but a redirection: **take the cheap compatibility win now,
and decide the performance question by answering whether a per-layer swap can launch multi-core.**
Everything else — weight layout, the compile boundary, the dispatch caches — is downstream of that
one question.

Two things are worth doing regardless of how that resolves. The `_detect_target` cache is a one-line
fix worth 86x per call to every eager NKI user. And a missing `neuronx-cc` on `PATH` should raise
instead of hanging forever, because the current behaviour points a new user directly at replacing
their driver.
