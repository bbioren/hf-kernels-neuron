# Draft check-in for John (internship mentor) — end of Week 3

**Status: DRAFT, NOT SENT.** Review before sending.

John wrote the project guide and the week-by-week schedule, so this is framed as a status
check-in plus the places his plan needs to change — not as a list of bugs to route. The asks are
decisions that are genuinely his: what Week 4 should be, whether Week 5 becomes a gap analysis,
and whether I reach out to Samir directly or he introduces.

---

## Short version (Slack)

> Hi John — Week 3 wrap-up, and I need your call on a couple of schedule things.
>
> **Where it stands vs your plan.** Weeks 1-3 done, and Week 4's SiLU kernel landed early, so
> three NKI kernels (RMSNorm, RoPE, SiLU) now swap into a stock Qwen3 and run on trn2 with
> logits matching at cos_sim 1.000001. RoPE is a real port of nki-library's production
> `rope_hf`, not a tutorial derivative. Accuracy suites: 11/11, 20/20 (+6/6 guard cases), 9/9,
> all with NKI execution asserted rather than assumed.
>
> **One Week 3 goal I couldn't hit, and it's not mine to fix.** "Confirm `use_kernels=True`
> alone triggers the swaps" is impossible today — transformers derives the device from
> `model.device.type`, which on Neuron is `"cpu"` or `"xla"`, never `"neuron"`, and because it
> passes a `Device` object rather than a string it skips validation entirely. So it's a silent
> no-op: `kernelize()` returns success with every layer untouched. I found the minimal fix
> (~3 lines in transformers) and verified it's sufficient — applied in-process it takes Qwen3
> from 0 to 9 swapped layers. So the goal is met in substance, just not through the intended
> entry point.
>
> **Three things that change your schedule, and I'd rather you decide than me:**
>
> *1. Week 5 (Qwen3-MoE) is gated.* I ran the MLP spike early to derisk it and found the fused
> MLP kernel can't run single-core above `intermediate_size = 4096` — sharp boundary, 10 configs,
> fails inside its own tile arithmetic with a divide-by-zero. That excludes Qwen3-8B (12288),
> Llama-3-8B and Mistral-7B (14336). MoE kernels are fused the same way and will likely hit
> related issues, plus there's an unresolved design question about weight layouts that HF would
> need to answer. **Is a gap analysis an acceptable Week 5 outcome, or do you want me to push
> for an implementation?** I think the gap analysis is genuinely the more useful artifact here,
> but it's your call since it changes what ships.
>
> *2. Week 4 MFU — is it still the right spend?* Two complications. `use_kernels=True` can't
> route to Neuron, so the measurement has to go through my own kernelize helper (fine, but it's
> a caveat on a customer-facing number). And per-layer benchmarking turned out useless: NKI
> dispatch costs ~0.36 ms of *host* time per call vs ~0.011 ms for eager, so at 217 kernel calls
> per Qwen3-8B forward there's ~76 ms/step of host overhead that swamps any per-kernel signal.
> Full-model MFU is the only instrument left.
>
> Fair warning: given that overhead there's a real chance MFU *with* the kernels is worse than
> without. I think that's still a valuable result — "mechanism works, kernels are correct, eager
> per-layer swap is launch-bound until fusion lands" — but I'd rather flag it now than surprise
> you at Week 6. **Do you want me to spend Week 4 on MFU anyway, or is something else more
> valuable with the time?**
>
> *3. Do I contact Samir directly?* I have a draft for him: Hub repo home
> (`aws-neuron/` vs `kernels-community/`, my lean is `aws-neuron/`), two small `kernels`-side
> fixes, and the weight-layout design question. Happy to send it, but you may want to look first
> or introduce me — your read on the right etiquette there.
>
> **Two things I got wrong that you should know about**, since they affect what I'd recommend:
> Week 2's accuracy numbers were measuring the PyTorch fallback rather than NKI (the kernel
> never executed — `@nki.jit` needs XLA tensors and the tests fed CPU ones), and my first
> benchmark reported every kernel 8-400x slower than eager because I discarded the outputs and
> XLA eliminated the computation. Both are fixed and both are now documented as findings, since
> the underlying pattern — that on a lazy backend correctness *and* performance measurements
> fail silently by default — is probably the most transferable thing in the PoC.
>
> Also: two smaller Neuron-internal items I'd like your read on who to talk to — the nki-library
> MLP bug above, and `torch_neuronx` not setting a `torch.neuron` attribute (one line, and it
> unblocks two separate things in the HF `kernels` library). Happy to file both myself if you'd
> rather I just go.
>
> Full writeup in `deliverables/week-3.md`, findings in `docs/poc-findings.md`. Happy to walk
> through any of it.

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

1. **Week 5**: is a gap analysis an acceptable outcome, or push for a MoE implementation?
2. **Week 4**: MFU as planned, or is something else a better use of the time given #19?
3. **Samir**: do I reach out directly, or would you rather introduce / review first?
4. **The two Neuron-internal items** (nki-library MLP bug, `torch_neuronx` attribute): file them
   myself, or route through you?
5. **Our own tech debt**: I'd like to spend a few hours rewriting RMSNorm and SiLU off the
   removed `nl.arange` API before anything goes to the kernels team. Shipping PoC kernels built
   on a removed API seems like a bad look. Agree?

### What I'd do next if you just said "keep going"

In this order: rewrite the two kernels onto `nl.ds` (hours), then full-size Qwen3-8B MFU with
launch count reported alongside, then start the Week 6 document since most of its content is
already accumulated. File the two Neuron-internal items in parallel since they have external
latency.

---

## Notes to self before sending

- Lead with status against *his* plan, not with findings. He wrote the schedule; the schedule
  changes are the headline.
- The asks are decisions, not tasks. Don't hand a mentor a routing queue.
- Surface the three self-corrections. Hiding them would be worse, and calibration is part of
  what he's assessing.
- Don't bury the `use_kernels=True` blocker — it's the one guide goal not met, and it's better
  he hears the root cause and the verified fix from me than discovers the gap later.
