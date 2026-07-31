# HuggingFace Kernel Hub on Trainium — PoC findings and recommendation

**Author:** Ben Bioren (internship PoC)
**Date:** 2026-07-31
**Audience:** Neuron kernels team (Hanbo Wang, Karthick Gopalswamy); Matt McClean; HF kernels team (Samir)
**Question asked:** should Neuron invest in first-class HuggingFace Kernel Hub support?

---

## Recommendation

**Yes — but not in the eager per-layer path, which is what the Kernel Hub currently offers on
Neuron, and which we measured as unusable for performance.**

Three findings, in order of how much they should change your plans:

1. **The mechanism works and is correct.** Three NKI kernels (RMSNorm, RoPE, SiLU) swap into
   stock Qwen3 dense *and* Qwen3-MoE via the Kernel Hub and produce matching logits
   (`cos_sim` 1.000001 / 1.000002). The interception surface is large — 115 RMSNorm and 95
   rotary registrations upstream — and Qwen3 already opts into all of it. No transformers-side
   model changes were needed.

2. **It is 208x slower.** MFU drops from 5.06% to 0.02%. Every `@nki.jit` invocation from eager
   PyTorch/XLA costs **~53 ms of fixed overhead regardless of problem size** — more than the
   entire 42 ms baseline forward pass. At 169 kernel calls per step, nothing else matters.
   This is an integration-model result, not a kernel-quality result.

3. **It cannot be reached anyway.** `use_kernels=True` cannot select the `"neuron"` device, and
   fails as a *silent no-op*. We found the fix (~3 lines in transformers) and verified it takes
   Qwen3 from 0 to 9 swapped layers.

So the investment we recommend is **not** a kernel-porting program. It is:

| Priority | Investment | Cost | Why |
|---|---|---|---|
| 1 | **Answer the graph-mode question** — can a NKI kernel live in a compiled graph with invocation cost paid once? | days | Decides whether this integration is viable at all. Everything else is contingent on it. |
| 2 | **Four small upstream fixes** (device routing, `_backend()`, `nkilib` allowlist, weight-layout contract) | ~a week total, spread across three teams | Each is smaller than one kernel port. Together they make the mechanism reachable and honest. |
| 3 | **A NKI kernel for MoE routing** (`sort`/`histc`) | small | Unblocks Qwen3-MoE on Neuron *entirely*, and is blocked by none of our other findings. |
| 4 | Kernel porting | — | **Defer.** Hand-porting doesn't scale, thin wrappers work but need item 2, and no kernel is a speedup until item 1 resolves. |

If item 1 comes back negative — the ~53 ms survives compilation — then the honest answer is
that the Kernel Hub's per-layer model cannot host nki-library's fused-megakernel design, and
Neuron should not invest further in this integration point. That would be a valuable answer
too, and it is cheap to obtain.

---

## What was built and validated

Three kernels, all execution-verified on trn2 with negative controls:

| Kernel | Type | Source | Accuracy | Upstream reach |
|--------|------|--------|----------|----------------|
| RMSNorm | layer swap (`RMSNorm`) | NKI tutorial — production kernel unusable, see below | 11/11 cases, fp32 max_diff 4.8e-07…8.1e-06 | 115 registrations |
| RoPE | **function** swap (`rotary_pos_emb`) | **ported from production `nkilib/core/embeddings/rope_hf.py`** | 20/20 cases + 6/6 guard cases | 95 model files |
| SiLU | layer swap (`SiLU`) | native `nl.silu` | 9/9 cases | 1 decoration covering all `ACT2FN["silu"]` users |

End-to-end:

| Model | Result |
|-------|--------|
| Qwen3 dense (2 layers) | RMSNorm 9x, RoPE 2x, SiLU 2x per forward, zero fallbacks, logits `cos_sim 1.000001` |
| Qwen3 dense (28 layers, seq 512) | 169 NKI calls/step, all engaged, used for the MFU measurement |
| **Qwen3-MoE** | all three transfer with **zero code changes**, logits `cos_sim 1.000002` |

That last row is the load-bearing evidence for the per-kernel thesis: the same three kernels
work across two architectures unmodified, because the interception points are shared.

---

## Finding 1 — the mechanism is right, and the interception points already exist

The Kernel Hub's forward-swap is the correct interception point for Neuron in principle.
`kernelize()` walks the model, matches layer names against a mapping, and replaces
`forward()` method pointers while leaving weights in place. Nothing about that is
CUDA-specific, and `"neuron"` is already a first-class device in the `kernels` library.

Coverage is better than the project brief assumed:

| Interception point | Registrations | Note |
|---|---|---|
| `RMSNorm` (layer) | **115** | brief estimated 87 |
| `rotary_pos_emb` (function) | **95 model files** | brief estimated 66 |
| `SiLU`, `GeLU`, `GeluTanh`, `NewGELU`, `FastGELU`, `QuickGELU` | 1 each in `activations.py` | one kernel covers every model using that `ACT2FN` entry |
| `MegaBlocksMoeMLP` | 2 | only live MoE name; `Llama4TextMoe` is commented out |

Two API subtleties that cost us time and will cost others the same:

- Layer repos resolve `kernel.layers.<name>`; **function** repos resolve `<name>` at module top
  level. A function kernel placed in the `layers` namespace is silently not found.
- `has_backward` defaults to **True** for function kernels and **False** for layers. Ours are
  all inference-only, so it must be set explicitly or `kernelize()` will select them in
  training mode and produce wrong gradients.

Also worth flagging: function-kernel replacement is **process-global**. `@use_kernel_func_from_hub`
creates one `Func` instance shared by every layer and every model in the process, so kernelizing
one model changes RoPE for all of them.

Several `_KERNEL_MAPPING` entries — `SwiGLUMLP`, `GeGLUMLP`, `Linear` — are registered by **no
model**, so they are unreachable via the decorator path. `SwiGLUMLP` matters, because fused MLP
is where MLP performance is; it requires the separate fusion API.

---

## Finding 2 — `use_kernels=True` cannot reach Neuron, and fails silently

The headline user-facing gap.

transformers' `kernelize(model, mode)` has no `device` parameter and derives everything from
`model.device.type`, which on Neuron is `"cpu"` (params on host) or `"xla"` (on device) —
never `"neuron"`. And because transformers passes a `Device` **object** while
`kernels.kernelize` only validates device types given as **strings**, `Device(type="xla")`
passes through unvalidated, matches nothing, and returns success with every layer untouched.

So the failure mode is a silent no-op, not an error.

**The fix, and we verified it is sufficient.** One branch in
`transformers/integrations/hub_kernels.py`:

```python
device_type = model.device.type
if device_type == "cuda" and is_rocm_platform():
    device_type = "rocm"
elif device_type == "xla" and _is_neuron_xla():     # <- the fix
    device_type = "neuron"
device = Device(type=device_type)
```

where `_is_neuron_xla()` checks `xm.xla_device_hw(xm.xla_device()) == "NEURON"` — confirmed to
return exactly that on trn2, no new dependency, fails closed.

Applied in-process this takes Qwen3 from **0 → 9 swapped layers** with logits `cos_sim
1.000001`. Two further sites need the same treatment: `kernel_config.py::infer_device` and
`kernels/layer/kernelize.py::_find_device`.

**Worth noting how we nearly got this wrong.** Our first proposal was to patch
`kernels._find_device`. That would have done nothing — transformers computes the device itself
and never calls it on this path. The e2e test caught it. "Propose a fix" and "verify the fix
works" are different activities, and for an ask aimed at another team the second is what makes
it credible.

---

## Finding 3 — eager NKI invocation costs ~53 ms, and that decides everything

Qwen3-0.6B, full 28 layers, seq 512, bf16, forward only, single logical core. Denominator: 632
TFLOPS/device (TensorEngine) ÷ 2 for LNC2 = **316 TFLOPS**. FLOP count computed explicitly
(670.42 GFLOP/step, breakdown in `deliverables/week-4.md`).

| Configuration | Step time | MFU | NKI calls/step |
|---|---|---|---|
| baseline | **41.95 ms** | **5.06 %** | 0 |
| NKI SiLU only | 1,495.54 ms | 0.14 % | 28 |
| all three kernels | **8,753.65 ms** | **0.02 %** | 169 |

Per-call added cost: 51.9 ms (SiLU only), 51.6 ms (all three). Uniform — which pointed at a
fixed charge rather than anything kernel-specific. Confirmed by sweeping problem size:

**52.7–54.6 ms across a 112x range in rows (128 → 14336). Completely flat.** One call on 28x
the data costs 1.02x one call on 1x the data.

Ruled out: interleaving (28 adjacent calls cost the same as 28 separated by torch ops), host
dispatch (0.36 ms/call, 1/145 of it), our kernels (production `rope_hf` shows the same figure),
recompilation (zero compiles during the timed loop), and host-sync artifacts (the cost is
incurred inside a single `mark_step`). Reproduced four times, stable to within 1%.

### Why this is structural

**At ~53 ms per invocation, one NKI call costs more than the entire baseline forward pass.** So
in eager mode any per-layer NKI swap loses, and swapping more layers loses harder. Even a
perfect one-fused-call-per-layer kernel would cost 28 × 53 ms = 1.5 s/step against 42 ms.

nki-library's kernels are designed as **large fused megakernels** that amortize invocation cost
across a whole transformer block. The Kernel Hub's per-layer forward swap **maximizes**
invocation count. That is the core mismatch of this integration, and three independent findings
converge on it:

| Finding | Direction it approaches from |
|---|---|
| #17 weight layout | fused kernels want weights arranged for their tiling; `kernelize()` never touches parameters |
| #18 single-core width limit | fused MLP won't compile single-core above `intermediate_size` 4096 — it assumes SPMD |
| **#20 invocation cost** | **fine-grained invocation is ~53 ms each** |

The per-layer swap works cleanly for exactly the ops that sit on the benign side of all three:
single-op, single-core, weights read as-is, no metadata. RMSNorm, RoPE and SiLU are the *easy*
cases. The mechanism handling them well is not evidence it will handle fused kernels.

### The decisive open question

If ~53 ms is a per-invocation framework-boundary cost, **graph mode should amortize it** — the
kernels become part of one compiled graph entered once per step instead of 169 times. If it is
intrinsic to NEFF execution, compilation won't help.

**We could not answer it.** `torch.compile` fails on this stack for **plain PyTorch** — `F.silu`
with no NKI anywhere fails identically across `openxla`, `inductor` and `eager` backends in both
dtypes. A NKI failure would be indistinguishable from compilation being broken generally, so the
experiment refuses to report a NKI result.

This is the single highest-value remaining experiment in the project, and it needs a stack where
`torch.compile` works on Neuron.

---

## Finding 4 — porting: hand-porting doesn't scale, but it may not be necessary

The brief assumed the work was porting kernels. Two findings reframe that.

**Week 2's blanket claim was drawn from one kernel and is wrong as a generalization.**

| Kernel | Standalone? | Quantization forced? | Verdict |
|--------|-------------|---------------------|---------|
| `rmsnorm/rmsnorm_quant.py` | no | **yes** — `QuantizationType.NONE` unsupported | must reimplement |
| `embeddings/rope_hf.py` | **yes** | no | **already HF-shaped**; ported directly |
| `mlp/mlp.py` | **yes** | no — quant *and* norm both opt-in | feasible, blocked by #18 |

RMSNorm is the outlier, not the archetype. And `rope_hf` is *better* positioned than the
tutorials — it takes precomputed cos/sin, returns a tuple, uses `rotate_half`, and is GQA-aware.
There is no rotary tutorial anywhere in nki-samples, so for RoPE the production library was the
only source and the better one.

**Dependency inlining cost varies by ~480x.** RoPE needed ~15 lines (3 symbols). The MLP
kernel's closure is **7,249 lines across 22 files**. Hand-porting does not reach the kernels
that matter for performance.

**But `nkilib` is already installed and its kernels are directly callable.** Verified: the
installed production `rope_hf` driven from PyTorch/XLA gives `cos_sim 1.000001`, and
`nkilib.core.mlp.mlp` matches Qwen3MLP at `cos_sim 0.999979–0.999995`. So a thin wrapper is
~40 lines with no vendoring:

```python
class NeuronRoPE(nn.Module):
    def forward(self, q, k, cos, sin, unsqueeze_dim=1):
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        return rope_hf(q, k, q_out, k_out, cos=cos, sin=sin)
```

One gotcha worth publishing: destination-passing is **vestigial across the XLA boundary**.
Output tensors must still be passed as shape templates, but results come back via the return
value — reading the mutated arguments gives zeros. nki-library's own tests use
`must_alias_input`, which leads a reader straight to the wrong strategy.

**The blocker is policy, not code:** `python_depends.json` whitelists `nki` for the neuron
backend but not `nkilib`, and the neuron table is unreachable anyway (Finding 5). Adding
`nkilib` is a four-line change with `nki` already there as precedent.

Tradeoff to state honestly: a thin wrapper couples the kernel repo to both `nkilib` and
`neuronx-cc` versions, and nki-library's README warns `main` isn't guaranteed compiler-compatible.
More tractable than maintaining hand-ports of 7,000-line kernels, but not free.

---

## Finding 5 — the environment fights you, in ways that compound

| Issue | Effect |
|---|---|
| `_backend()` reports `CUDA(12.8)` on a Neuron DLAMI | root cause is `hasattr(torch, "neuron")` being False even after `import torch_neuronx`. Breaks build-variant resolution **and** dependency validation. |
| A Neuron kernel cannot declare `python-depends: ["nki"]` | consequence of the above — the whitelist entry exists but its table is never consulted. Kernels must under-declare (`[]`) while importing `nki`. |
| Two NKI generations coexist | `import nki` is 0.5.0; `from neuronxcc import nki` is an older bundled build. `nl.arange` and `nl.mgrid` were **removed** in 0.5.0 (use `nl.ds`), but the public tutorials still teach them. We wrote two kernels against a removed API without noticing, then migrated them. |
| Qwen3-MoE won't run on Neuron at all by default | `grouped_mm_experts_forward` uses `torch.sort`/`torch.histc` → unsupported `sort` HLO. Fix is `experts_implementation="batched_mm"`; nothing documents this. |
| `torch.compile` doesn't work on this stack | blocks the decisive question above |

Fixing `_backend()` is one line in `torch_neuronx` and resolves the first two at once. It does
**not** fix device routing — that is Finding 2's separate transformers change. Two distinct
problems that are easy to conflate into one ticket.

The MoE item is worth surfacing in Neuron's model-support docs independent of the Kernel Hub: a
customer gets a compiler error naming an HLO op with no hint that a config flag resolves it.

---

## The methodological finding, which may be the most transferable output

Three times in this project, a plausible-looking measurement turned out to be measuring nothing.
The pattern is consistent enough to state as a conclusion.

| # | Symptom | Actual cause | Would have concluded |
|---|---------|--------------|---------------------|
| 8 | `max_diff = 0.00e+00`, "bit-identical" | `@nki.jit` needs XLA tensors; tests fed CPU tensors, so the kernel never ran and the fallback was compared to itself | "RMSNorm kernel validated" — for a week |
| 19 | latency independent of problem size | outputs discarded, so XLA eliminated the computation; timed an empty graph | "NKI is 8–400x slower" (wrong by 68x, in the flattering direction) |
| 21 | `ModuleNotFoundError: No module named 'neuron_silu'` under compile | our loader never registered the module in `sys.modules`; Dynamo re-imports by name | "NKI kernels are incompatible with torch.compile" |

**On a lazy-execution accelerator backend, both correctness and performance measurements fail
silently by default.** A fallback is numerically correct. An eliminated computation is fast. A
harness bug looks like a platform limitation. None of these produce an error.

What we now do, and would recommend to anyone doing kernel work on this stack:

1. **Assert execution, don't infer it.** A call counter on the dispatch targets proves the
   kernel ran. Numerical agreement cannot.
2. **Treat a suspiciously good result as a bug report.** For a reduction kernel, an exact-zero
   diff means it didn't run. Interpretation is op-dependent: elementwise ops *should* be
   bit-identical.
3. **Gate performance numbers on a scaling check.** If latency doesn't respond to problem size,
   suppress the result rather than caveat it. Ours does this and it prevented shipping a number
   wrong by 68x.
4. **Always run a control that excludes the thing under test.** The `torch.compile` experiment
   only avoided a false finding because it compiled plain `F.silu` first.
5. **Include negative controls.** Every accuracy suite checks that it can *fail* — against a
   wrong reference and against the unmodified input.

This cost roughly a cycle each time, and one of the three was caught only because the guard
built after the previous one fired.

---

## What we are asking for

| # | Ask | Owner | Size | Status |
|---|-----|-------|------|--------|
| 1 | Answer: can a NKI kernel be invoked from a compiled graph with invocation cost paid once? | NKI / torch-neuronx | a question, maybe already known | **decisive** |
| 2 | Route XLA-on-Neuron to `Device(type="neuron")` (3 sites) | transformers + `kernels` | ~12 lines | **fix verified sufficient** |
| 3 | Set `torch.neuron` so `_backend()` reports neuron | `torch_neuronx` | 1 line | unblocks two things |
| 4 | Add `nkilib` to `python_depends.json` under `neuron` | HF `kernels` | ~6 lines | `nki` is precedent |
| 5 | Weight-layout contract for fused kernels (a `prepare_weights` hook, or kernels absorb layout) | HF `kernels` + nki-library | design decision | blocks all fused ports |
| 6 | Fused MLP divide-by-zero single-core above `intermediate_size` 4096 | nki-library | bug fix | boundary measured, 10 data points |
| 7 | Document `experts_implementation="batched_mm"` for MoE on Neuron | Neuron docs | doc change | customer-facing |
| 8 | NKI tutorials teach `nl.arange`, removed in 0.5.0 | NKI docs | doc change | misleads new authors |

Full detail with exact code locations and ready-to-paste patches in `docs/upstream-fixes.md`.

**Sequencing matters.** Item 1 gates whether items 5 and 6 are worth anyone's time — there is
no point designing a weight-transformation contract for kernels that cannot be a speedup.
Items 2, 3, 4 and 7 are worth doing regardless, because they are about correctness,
reachability and documentation rather than performance.

---

## What is not done

Stated plainly so nobody inherits a false impression:

- **No backward kernels.** All three are `has_backward=False`; training mode falls back.
  nki-library's `rope_hf` has a backward path and `nl.silu_dx` exists, so this is feasible but
  unbuilt.
- **No torch.compile support.** `can_torch_compile=False` on all three, and the stack can't
  test it.
- **No Hub upload.** Flat layout validated as loadable, `digest` confirmed optional, minimum
  repo is two files. Blocked on the repo-home decision (`aws-neuron/` vs `kernels-community/`)
  and on item 4 for an honest dependency declaration.
- **No fused MLP.** Blocked by items 5 and 6.
- **No MoE-specific kernel.** Gap analysis instead — see `deliverables/week-5-moe-gap-analysis.md`.
  The best target turns out to be the routing `sort`/`histc`, not the expert matmul.
- **MFU measured on Qwen3-0.6B, not 8B.** Full depth, so it is a real model, but not the largest
  one. Given the ~53 ms/invocation result, a larger model would only make the ratio worse.
- **Single core only.** Eager per-layer swap doesn't manage multi-core; SPMD was stripped from
  the RoPE port.

---

## Closing assessment

The optimistic read of this PoC is that the hard parts worked. The Kernel Hub mechanism is
sound on Neuron, the kernels are correct on two model architectures, the upstream interception
surface is large and already wired, and the fixes needed to reach it are small and well
understood.

The honest read is that we measured the thing the brief was really asking about — whether this
makes models faster — and on the stack a customer would use today, it makes them 208x slower for
a reason that is structural rather than incidental. The Kernel Hub wants many small kernel
invocations; NKI on this stack charges ~53 ms for each one; nki-library's kernels are built as
few large ones. Those three facts are in direct tension and no amount of kernel quality
resolves them.

That tension has one plausible escape, and it is cheap to test: if graph mode amortizes the
per-invocation cost, the whole picture changes and the Kernel Hub becomes a good bet for Neuron.
**That experiment is the single most valuable thing anyone can do next**, and it is worth more
than any further kernel work. We would recommend running it before committing engineering time
to anything else in this document.
