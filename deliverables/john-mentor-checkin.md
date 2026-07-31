# Draft check-in for John (internship mentor) — Weeks 3-6

**Status: DRAFT, NOT SENT.** Review before sending.

**UPDATED after Weeks 4-6 completed.** The earlier version of this draft asked John to decide
what Week 4 should be and whether Week 5 should become a gap analysis. Both have since been
done, and the Week 4 MFU measurement **changed the project's conclusion**, so the message is
rewritten around that rather than around scheduling questions.

The short version now leads with the finding, because it is the kind of result a mentor needs to
hear before it reaches the kernels team, not after.

---

## The one thing to read first

MFU with the NKI kernels is **0.02% against a 5.06% baseline — 208x slower.** Root cause: every
`@nki.jit` invocation from eager PyTorch/XLA costs ~53 ms of fixed overhead regardless of
problem size, which is more than the entire 42 ms baseline forward pass. It is an
integration-model result, not a kernel-quality one: the Kernel Hub wants many small kernel
invocations, NKI charges ~53 ms each, and nki-library's kernels are built as few large ones.

The kernels themselves are correct — verified on two model architectures. But the eager
per-layer path cannot be made fast, and the decisive follow-up question (does graph mode
amortize the cost?) could not be answered because `torch.compile` doesn't work on this stack.

---

## Short version (Slack)

> Hi John — big update, and the headline isn't what I expected going in.
>
> **The kernels work and they're 208x slower.** MFU 0.02% vs a 5.06% baseline on Qwen3-0.6B at
> full depth. I chased the cause rather than just reporting the number: **every `@nki.jit`
> invocation from eager PyTorch/XLA costs ~53 ms of fixed overhead, independent of problem
> size.** Flat across a 112x range in input size, reproduced four times within 1%. That's more
> than the entire 42 ms baseline forward pass, so at 169 kernel calls per step nothing else
> matters. Ruled out interleaving, host dispatch, my kernels (nki-library's production `rope_hf`
> shows the same figure), recompilation, and sync artifacts.
>
> It's an integration-model result, not a kernel-quality one. The Kernel Hub wants many small
> kernel invocations; NKI charges ~53 ms each; nki-library's kernels are designed as a few large
> fused megakernels. Those three facts are in direct tension and kernel quality doesn't resolve
> them. Notably three separate findings now converge on that same mismatch — weight layout,
> single-core width limits, and now invocation cost.
>
> **The one escape, and I couldn't test it.** If that ~53 ms is a per-invocation framework
> boundary cost, graph mode should amortize it — kernels become part of one compiled graph
> entered once per step instead of 169 times. That would change the recommendation completely.
> But `torch.compile` doesn't work on this stack *at all* — plain `F.silu` with no NKI anywhere
> fails across openxla/inductor/eager in both dtypes. So a NKI failure would be
> indistinguishable from compile being broken generally, and I refused to record a NKI result
> from it.
>
> **This is the single most valuable experiment left in the project** and I can't run it here.
> Two questions for you: is there a Neuron stack where torch.compile works that I could get onto
> (the Native PyTorch beta compile path?), and/or do the NKI/torch-neuronx folks already *know*
> whether NKI invocation cost is paid once at compile time or per call? Asking might be faster
> than measuring.
>
> **Also worth knowing: ~53 ms looks like a misconfiguration, not a design point.** Before I
> write it up as a fundamental property, is that plausibly expected on SDK 2.31 / NKI 0.5.0 via
> torch-xla eager? If it's a known issue the whole conclusion changes.
>
> **What did land.** Weeks 3-6 are done. Three kernels (RMSNorm, RoPE, SiLU) validated on Qwen3
> dense *and* Qwen3-MoE with zero code changes for the MoE case — logits cos_sim 1.000001 and
> 1.000002. RoPE is a real port of nki-library's `rope_hf`. MoE gap analysis written. PoC
> document drafted. Along the way: found that `use_kernels=True` can't reach Neuron at all
> (silent no-op) and verified a ~3-line transformers fix takes Qwen3 from 0 to 9 swapped layers;
> and found Qwen3-MoE won't run on Neuron by default at all because the experts path uses
> `torch.sort`/`histc` → unsupported HLO (fix: `experts_implementation="batched_mm"`,
> undocumented anywhere).
>
> **Three things I got wrong, all caught by measurement.** Week 2's accuracy numbers were
> measuring the PyTorch fallback, not NKI — the kernel never ran. My first benchmark reported
> 8-400x slower because I discarded outputs and XLA eliminated the computation. And a harness
> bug of mine produced a convincing-looking "NKI is incompatible with torch.compile" that was
> just a missing `sys.modules` registration. On a lazy-execution backend both correctness and
> performance measurements fail silently by default — a fallback is numerically correct, an
> eliminated computation is fast, a harness bug looks like a platform limit. That pattern is
> probably the most transferable thing in the PoC and it's written up as such.
>
> **Two decisions I'd still like from you:** (1) do I send the Samir draft or do you want to
> review/introduce, and (2) the nki-library MLP bug and the `torch_neuronx` `torch.neuron`
> one-liner — file them myself or route through you?
>
> `deliverables/poc-document.md` has the full recommendation. Short version: yes invest, but in
> answering the graph-mode question and four small upstream fixes — not in a kernel-porting
> program.

---

## Longer version, if he wants the detail

### Status against your week-by-week plan

| Week | Plan | Outcome |
|------|------|---------|
| 1 | Neuron device path verification | done |
| 2 | RMSNorm kernel + validation | done, **but the accuracy numbers were invalid and have been replaced** (see below) |
| 3 | Hub packaging, RoPE, register neuron entries, `use_kernels=True` | RoPE done (production port), entries done locally, packaging partial (blocked on repo home), `use_kernels=True` **blocked upstream** |
| 4 | SiLU, full Qwen3 e2e, MFU | SiLU **done early**; MFU pending your call |
| 5 | Qwen3-MoE | **gated** — see below |
| 6 | PoC doc + recommendation | on track, content largely accumulated |

Validated, all on trn2 with NKI execution asserted via call counters:

| Kernel | Interception point | Upstream registrations | Accuracy |
|--------|-------------------|------------------------|----------|
| RMSNorm | `@use_kernel_forward_from_hub("RMSNorm")` | 115 | 11/11 cases |
| RoPE | `@use_kernel_func_from_hub("rotary_pos_emb")` | 95 model files | 20/20 + 6/6 guards |
| SiLU | `@use_kernel_forward_from_hub("SiLU")` | 1 decoration, covers all `ACT2FN["silu"]` | 9/9 cases |

E2E on Qwen3: RMSNorm 9× per forward, RoPE 2×, SiLU 2×, zero fallbacks, logits cos_sim 1.000001.
Qwen3 already opts into all three interception points upstream, so no model changes were needed.

Coverage is also better than the guide estimated — 115 RMSNorm registrations vs 87, and 95 RoPE
model files vs 66.

### The thing I'd most like your judgement on

The PoC question is "should Neuron invest in first-class HF Kernel Hub support?" Three weeks in,
my answer is *yes, but the investment isn't what we assumed*.

We assumed the work was porting kernels. It isn't. `nkilib` turns out to be already installed in
the DLAMI venv, and its production kernels are directly callable from PyTorch/XLA — I validated
the installed `rope_hf` at cos_sim 1.000001 and the `mlp` kernel too, within its size limits. So
an HF kernel can be a ~40-line wrapper rather than a hand-port. And hand-porting doesn't scale
to what matters anyway: RoPE needed ~15 lines of deps inlined, the MLP kernel's dependency
closure is ~7,250 lines across 22 files.

So the recommendation shape becomes: three or four small upstream fixes plus one design
decision, not a porting program. Each fix is smaller than a single kernel port.

The counterweight is Finding #19 (dispatch overhead) and #18 (fused MLP size limit), which
together suggest the eager per-layer swap model may be structurally launch-bound until fused
kernels work. That's the tension the Week 6 recommendation has to resolve, and it's the part I'd
most value your steer on before I write it.

### Corrections to my own work

Flagging these because they changed conclusions, not just numbers.

**Week 2's accuracy results were measuring the PyTorch fallback.** `@nki.jit` requires XLA
tensors, so kernels carry a device guard, and the Week 2 tests built inputs with `torch.randn`
— CPU tensors. Every case took the fallback and compared it against a mathematically identical
reference, reporting `max_diff = 0.00e+00`. The perfection *was* the tell; a reduction kernel on
hardware should differ by ~1e-4. The kernel turned out correct but had never been executed.
Every test now asserts via a call counter that the NKI branch ran.

**My first benchmark was meaningless.** Reported 8-400x slower than eager. I discarded the
outputs, so XLA had no live result and eliminated the computation — I was timing an empty graph.
The tell was latency not varying with tensor size. The script now refuses to report ratios
unless latency demonstrably scales with problem size.

**Finding #14 was wrong.** I'd written up the two NKI import paths as having disjoint
capabilities. They're two *versions* — `import nki` is 0.5.0, `from neuronxcc import nki` is an
older bundled build, and `nl.arange` was removed in 0.5.0 in favour of `nl.ds`. So two of our
three kernels are written against a removed API. That's our tech debt (a few hours to fix), not
an upstream question. I'd also claimed `hasattr(nl, "arange")` was True under the new package;
it isn't — I'd read that off a probe that imported the *old* path.

The common thread, and I think the most transferable output of the PoC: **on a lazy-execution
accelerator backend, both correctness and performance measurements fail silently by default.** A
fallback is numerically correct; an eliminated computation is fast. Every measurement needs an
independent check that it exercised the thing under test — a call counter for correctness, a
size-scaling gate for performance. Neither is standard practice.

### Decisions I'm asking for

Reduced to the ones that are still open — the Week 4/5 scheduling questions resolved themselves
by getting done.

1. **Can I get onto a stack where `torch.compile` works on Neuron?** This is the blocker on the
   most valuable remaining experiment. The Native PyTorch beta compile path seems like the
   candidate.
2. **Is ~53 ms per NKI invocation expected on this SDK?** If the NKI/torch-neuronx teams already
   know the answer, asking beats measuring. And if it's a known issue rather than a design point,
   the PoC's conclusion changes materially.
3. **Samir**: do I reach out directly, or would you rather review/introduce first?
4. **The two Neuron-internal items** (nki-library MLP divide-by-zero, `torch_neuronx` setting
   `torch.neuron`): file them myself, or route through you?
5. **Does the PoC recommendation land right?** It says invest, but in the graph-mode question
   and four small upstream fixes rather than kernel porting — and it says that if graph mode
   *doesn't* amortize the cost, Neuron should not invest further in this integration point. That
   second half is a stronger negative than I'd have predicted at Week 1, so I'd like a sanity
   check before it goes to Hanbo/Karthick.

### Already done, no decision needed

- Migrated RMSNorm and SiLU off the removed `nl.arange` API onto `nl.ds` / NKI 0.5.0. Turned out
  to *improve* accuracy ~50x on fp32 (max_diff 1e-4 → 1e-6) because the fix required computing
  the reduction in fp32, which is what PyTorch's RMSNorm does anyway.
- Week 5 gap analysis, with a revised recommendation: the best MoE NKI target is the routing
  `sort`/`histc` step, **not** the expert matmul. It unblocks the default MoE path on Neuron,
  the compiler error itself recommends NKI for it, and it's blocked by neither the weight-layout
  question nor the single-core width limit.

### What I'd do next if you just said "keep going"

1. Get the graph-mode answer, by whatever path is fastest — a working stack or a conversation.
2. File the upstream items (they have external latency, so earlier is better).
3. If graph mode looks viable: the MoE routing kernel, since it's small and unblocked.
4. If it doesn't: stop kernel work and write up the negative recommendation properly. That's a
   real result and worth delivering cleanly rather than padding.

---

## Notes to self before sending

- Lead with status against *his* plan, not with findings. He wrote the schedule; the schedule
  changes are the headline.
- The asks are decisions, not tasks. Don't hand a mentor a routing queue.
- Surface the three self-corrections. Hiding them would be worse, and calibration is part of
  what he's assessing.
- Don't bury the `use_kernels=True` blocker — it's the one guide goal not met, and it's better
  he hears the root cause and the verified fix from me than discovers the gap later.
