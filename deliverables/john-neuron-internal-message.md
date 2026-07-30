# Draft message to John — Neuron-internal items from the HF Kernel Hub PoC

**Status: DRAFT, NOT SENT.** Review before sending.

**Why this exists:** most of the open questions from Week 3 are *not* Samir's. Of the items I'd
been treating as "upstream asks", three are Neuron-internal, one turned out to be our own tech
debt, and only two genuinely need HuggingFace. Samir being 9h ahead doesn't block the majority
of it.

**Assumption to correct if wrong:** I've written this assuming John is the right Neuron-side
person to route nki-library / NKI-team / torch_neuronx items and to make the internal calls on
methodology. If routing goes through someone else, the content still stands — just re-address.

---

## Who owns what (the useful split)

| Item | Owner | Needs Samir? |
|---|---|---|
| Fused MLP divides by zero single-core when `I > 4096` (#18) | **nki-library team** | no |
| `torch_neuronx` should set a `torch.neuron` attribute (#7/#12) | **torch_neuronx team** | no |
| NKI tutorials teach a removed API (`nl.arange`) (#14) | NKI docs — *minor, FYI only* | no |
| Rewrite our RMSNorm + SiLU off the removed API | **us** | no |
| MFU measurement methodology | **us / internal call** | no |
| transformers device-routing fix (#9) | HuggingFace | only for "where should the helper live" |
| Hub repo home, `nkilib` allowlist, weight-layout hook (#16/#17) | HuggingFace | **yes** |

So: four things can move today, and one HF item (the routing fix) can be filed as a GitHub
issue without waiting for him.

---

## Short version (Slack to John)

> Hi John — wrapping Week 3 of the HF Kernel Hub PoC. Three NKI kernels (RMSNorm, RoPE, SiLU)
> now swap into a stock Qwen3 and run on trn2, logits matching at cos_sim 1.000001. RoPE is a
> real port of nki-library's production `rope_hf`.
>
> Four things that are ours, not HuggingFace's. Two are bugs I'd like to route, one is a docs
> problem, one is a call I'd like your read on.
>
> **1. nki-library bug — fused MLP can't run single-core above intermediate_size 4096.**
> I ran a spike calling `nkilib.core.mlp.mlp` directly. It works and matches Qwen3MLP
> (cos_sim 0.999979–0.999995), but only at toy sizes. Above I=4096 it fails to compile with
> `'floordiv' does not allow division by zero` inside its own tile arithmetic
> (`mlp_cte_tile_info.py:236` → `kernel_helpers.py:104`). Sharp boundary, 10 configs across
> three hidden_size values — passes iff I ≤ 4096, so 4096 passes and 4224 fails. Not fixed by
> seqlen, `force_cte_mode`, or `mode=PREFILL`.
>
> That excludes every real model: Qwen3-8B is I=12288, Llama-3-8B and Mistral-7B are I=14336.
> My guess is the CTE sharding heuristic forcing `shard_on_inter` above I=4096 while we launch
> single-core with no SPMD grid. Two questions for whoever owns it: is single-core above
> I=4096 meant to work, and if not, can the limit be a `kernel_assert` rather than a
> divide-by-zero deep in tile math? Nothing documents an I≤4096 limit — the documented
> constraint is `H % 128 == 0`, which I=12288 satisfies.
>
> This one matters beyond the bug: it blocks the whole fused-kernel direction for HF, and a
> wrapper can't work around it. Repro is `scripts/spike_nkilib_mlp.py`, ~2 min.
>
> **2. torch_neuronx should set a `torch.neuron` attribute.** One line, and it unblocks two
> separate things in the HF `kernels` library. `kernels` checks `hasattr(torch, "neuron")` to
> decide the backend; it's False even after `import torch_neuronx`, so `_backend()` reports
> `CUDA(12.8)` on a Neuron DLAMI. Consequences: (a) Hub build-variant resolution can't find a
> neuron variant, and (b) `python_depends.json` *already* whitelists `nki` under a `neuron`
> section, but that table is never consulted — so a Neuron kernel literally cannot declare
> `python-depends: ["nki"]` and has to ship `[]` while importing nki. Who owns that?
>
> **3. Minor, mostly FYI: the public NKI tutorials teach a removed API.** `nl.arange` was
> removed in NKI 0.5.0 (replaced by `nl.ds` slicing), but the tutorials still use it — which is
> how I wrote two of our three kernels against it without noticing. Both `import nki` (0.5.0)
> and `from neuronxcc import nki` (older, bundled in the compiler) resolve on the DLAMI, so
> nothing errors at import; you only find out at kernel-compile time.
>
> Fixing our kernels is a few hours and I'll just do it — no ask there. Only flagging it in case
> it's worth a nudge to whoever owns the NKI docs, since it'll mislead other kernel authors the
> same way.
>
> **4. A call I'd like your read on: how to measure MFU.** This is the last technical
> deliverable and the PoC's recommendation hinges on it. Two complications:
> - `use_kernels=True` can't route to Neuron yet (needs a small transformers fix I've verified),
>   so the measurement has to go through our own kernelize helper. Fine by me, but it's a
>   caveat on a customer-facing number — happy to be told to wait for upstream instead.
> - Per-layer microbenchmarking turned out to be useless here: NKI dispatch costs ~0.36 ms of
>   *host* time per call vs ~0.011 ms for eager. At 217 kernel calls per Qwen3-8B forward
>   that's ~76 ms/step of host overhead, and it swamps any per-kernel signal. So full-model MFU
>   is the only instrument left, and I want to report launch count alongside it.
>
> Fair warning: given that dispatch overhead there's a real chance MFU *with* the kernels is
> worse than without. I think that's still a useful result — "mechanism works, kernels are
> correct, eager per-layer swap is launch-bound until fusion lands" — but I'd rather flag it now
> than surprise anyone at Week 6.
>
> Separately I've got a draft for Samir on the HuggingFace-side items (Hub repo home, two small
> `kernels` fixes, one design question). Happy to send that or have you look first.

---

## Longer context, if he wants it

### Where the PoC stands

Three NKI kernels execute inside a stock Qwen3 forward on Trainium via the HF Kernel Hub:

| Kernel | Interception point | Upstream registrations | Accuracy |
|--------|-------------------|------------------------|----------|
| RMSNorm | `@use_kernel_forward_from_hub("RMSNorm")` | 115 | 11/11 cases |
| RoPE | `@use_kernel_func_from_hub("rotary_pos_emb")` | 95 model files | 20/20 + 6/6 guards |
| SiLU | `@use_kernel_forward_from_hub("SiLU")` | 1 decoration covering all `ACT2FN["silu"]` | 9/9 cases |

E2E on Qwen3: RMSNorm 9× per forward, RoPE 2×, SiLU 2×, zero fallbacks, logits cos_sim
1.000001. Qwen3 already opts into all three interception points, so no transformers-side model
changes are needed.

The mechanism works. What's unresolved is whether it's *fast*, and whether the fused kernels —
where the actual performance is — can fit the model at all.

### Two things I got wrong, for calibration

Worth knowing because both were caught by measurement rather than review, and both were the
same class of error:

- **Week 2's accuracy numbers were measuring the PyTorch fallback, not NKI.** `@nki.jit` needs
  XLA tensors, so kernels carry a device guard, and the tests fed CPU tensors — so every case
  silently took the fallback and compared it against a mathematically identical reference,
  reporting a perfect `max_diff = 0.00e+00`. The perfection *was* the bug. The kernel turned
  out correct but had never been executed. Tests now assert via a call counter that the NKI
  branch ran.
- **My first benchmark reported every kernel 8–400× slower than eager.** Meaningless — I
  discarded the outputs, so XLA eliminated the computation and I timed an empty graph. The tell
  was latency not varying with tensor size.

Generalizable point, and probably the most transferable thing in the PoC: on a lazy-execution
backend, *both* correctness and performance measurements fail silently by default. A fallback is
numerically correct; an eliminated computation is fast. Every measurement needs an independent
check that it exercised the thing under test.

### The strategic finding

`nkilib` is already installed in the Neuron DLAMI venv, and its production kernels are directly
callable from PyTorch/XLA — I validated the installed `rope_hf` at cos_sim 1.000001, and the
`mlp` kernel too (within its size limits). So an HF kernel can be a ~40-line wrapper rather than
a hand-port.

That changes the recommendation shape. Hand-porting doesn't scale to what matters: RoPE needed
~15 lines of deps inlined, the MLP kernel's dependency closure is ~7,250 lines across 22 files.
So the PoC's ask becomes a handful of small upstream fixes rather than a kernel-porting program
— *if* item 1 above gets resolved and HF whitelists `nkilib`.

One caveat I'll state in the writeup: a thin wrapper couples the HF kernel repo to both `nkilib`
and `neuronx-cc` versions, and nki-library's own README warns `main` isn't guaranteed compatible
with a given compiler. More tractable than maintaining hand-ports of 7,000-line kernels, but not
free.

---

## Follow-ups to track after sending

- [ ] #18 routed to nki-library owner — get a name and a read on whether single-core >4096 is supported
- [ ] `torch.neuron` attribute — owner identified in torch_neuronx
- [ ] NKI docs currency (`nl.arange` removed but still taught) — FYI only, no owner needed
- [ ] MFU methodology — decision: measure now via our helper, or wait for the transformers fix
- [ ] Green light to send the Samir draft, or he sends/introduces
- [ ] Our own work item: rewrite RMSNorm + SiLU onto `nl.ds` / `import nki` (few hours)
