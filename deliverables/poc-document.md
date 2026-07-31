# HuggingFace Kernel Hub on Trainium — PoC findings and recommendation

**Author:** Ben Bioren (internship PoC)
**Date:** 2026-07-31
**Audience:** Neuron kernels team (Hanbo Wang, Karthick Gopalswamy); Matt McClean; HF kernels team (Samir)
**Question asked:** should Neuron invest in first-class HuggingFace Kernel Hub support?

---

## Recommendation

**Yes. Fix two caching bugs in NKI's dispatch path first — one of them is a one-line change worth
102x — then invest at the granularity that can actually win, which is not one small layer at a
time.**

Four findings, in order of how much they should change your plans:

1. **A one-line bug was costing 102x per kernel call.** `nki/framework/compiled.py::_compile_opts()`
   calls `resolve_target()` on *every* invocation, which forks `neuron-ls` to ask the hardware what
   it is: **~52 ms per call**. It sits outside `_nki_compile_cache` because its result is part of
   the cache key, so a cache *hit* still pays it in full. Caching it (`lru_cache`, or setting
   `NEURON_PLATFORM_TARGET_OVERRIDE`) takes per-call cost from 51.74 ms to 0.49 ms with
   bit-identical accuracy, and takes model-level MFU from **0.02% to 1.50%** — a 62x recovery.
   This is not specific to the Kernel Hub. Anything calling NKI kernels per-layer from eager
   PyTorch pays it.

2. **The mechanism works and is correct.** Three NKI kernels (RMSNorm, RoPE, SiLU) swap into stock
   Qwen3 dense *and* Qwen3-MoE via the Kernel Hub and produce matching logits (`cos_sim` 1.000001 /
   1.000002). The interception surface is large — 115 RMSNorm and 95 rotary registrations upstream —
   and Qwen3 already opts into all of it. No transformers-side model changes were needed. The
   device profile confirms the compiled kernels are efficient: a 28-call NEFF executes in 0.609 ms
   at 43% memory-bandwidth utilisation and 95% engine active time.

3. **Even fixed, per-layer swapping of small ops is still a 3.4x net loss** (42.04 ms/step baseline
   vs 141.43 ms kernelized; MFU 5.05% vs 1.50%). The residual is ~0.59 ms of host dispatch per call
   against 0.02 ms of device time, and cProfile attributes it to `create_computation` rebuilding the
   XLA computation and its HLO protobufs *on every invocation* — the same class of bug as item 1,
   two orders of magnitude smaller. A plain torch op costs 0.02–0.03 ms in the same position, so
   NKI eager dispatch remains **15–20x a torch op's**.

4. **It cannot be reached anyway.** `use_kernels=True` cannot select the `"neuron"` device, and
   fails as a *silent no-op*. We found the fix (~3 lines in transformers) and verified it takes
   Qwen3 from 0 to 9 swapped layers.

Break-even follows directly from item 3: a swapped kernel is a net win only if it saves more than
~0.59 ms of torch time per call. Torch SiLU on `[512, 3072]` costs 0.02–0.04 ms, so these ops are
**15–30x underwater**. Winning requires either dispatch cost at torch-op levels, or kernels that
replace far more work per call — fused kernels, which is what `nkilib` actually ships and what
Findings #17 and #18 say the Kernel Hub cannot currently express.

So the recommended investment is:

| Priority | Investment | Cost | Why |
|---|---|---|---|
| 1 | **Cache `_detect_target()`** in `nki/compiler/target.py` | one decorator | 102x per call, verified accuracy-neutral. Benefits all eager NKI usage, not just this integration. |
| 2 | **Cache the per-call XLA computation build** (`create_computation` + pyhlo scribe) | unknown, needs scoping | The remaining 0.59 ms/call, and the difference between 3.4x slower and plausibly near parity. Not yet attempted — see *What is not done*. |
| 3 | **Four small upstream fixes** (device routing, `_backend()`, `nkilib` allowlist, weight-layout contract) | ~a week, three teams | Each smaller than one kernel port. Together they make the mechanism reachable and honest. |
| 4 | **A NKI kernel for MoE routing** (`sort`/`histc`) | small | Unblocks Qwen3-MoE on Neuron *entirely*, and is blocked by none of our other findings. |
| 5 | Kernel porting at per-layer granularity | — | **Defer** until item 2 lands. Until dispatch is cheaper, small-op swaps lose on arithmetic, however good the kernel is. |

The previous version of this document made item 1 "answer the graph-mode question — can a NKI kernel
live in a compiled graph with invocation cost paid once?" **That question is now answered, and it
was the wrong question.** torch-xla is already a graph runtime: 28 NKI calls demonstrably fuse into
one HLO graph and one device execution, and still cost 28x. Graph batching was never going to help,
because the cost was on the host before `mark_step`. That ask is withdrawn.

### What would change this recommendation

If item 2 turns out to be infeasible — if the XLA computation genuinely must be rebuilt per call —
then per-layer NKI swapping stays a net loss for small ops, and the honest conclusion is that the
Kernel Hub's granularity cannot host `nki-library`'s fused design. That is still worth knowing, and
it is now a scoping question for the NKI team rather than an open experiment.

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

## Finding 3 — eager NKI invocation costs ~53 ms, and the cause is a subprocess

Qwen3-0.6B, full 28 layers, seq 512, bf16, forward only, single logical core. Denominator: 632
TFLOPS/device (TensorEngine) ÷ 2 for LNC2 = **316 TFLOPS**. FLOP count computed explicitly
(670.42 GFLOP/step, breakdown in `deliverables/week-4.md`).

| Configuration | Step time | MFU | NKI calls/step |
|---|---|---|---|
| baseline | **41.95 ms** | **5.06 %** | 0 |
| NKI SiLU only | 1,495.54 ms | 0.14 % | 28 |
| all three kernels | **8,753.65 ms** | **0.02 %** | 169 |
| **all three, with `_detect_target` cached** | **141.43 ms** | **1.50 %** | 169 |
| all three, cached, seq 2048 | 223.99 ms | 4.81 % | 169 |
| baseline, seq 2048 (for that row) | 108.76 ms | 9.90 % | 0 |

That last row is the important one and it arrived late. The rest of this section is the
investigation that produced it, kept in order because the wrong turn in the middle is instructive.
The short version: **the ~53 ms is an uncached `neuron-ls` subprocess forked on every kernel
invocation.** Caching it is one decorator, costs 102x per call, and changes nothing numerically.

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

### The question we thought was decisive, and why it was the wrong one

The reasoning above led to what looked like the pivotal question: if ~53 ms is a per-invocation
framework-boundary cost, graph mode should amortize it — the kernels become part of one compiled
graph entered once per step instead of 169 times. This document previously named that the
single highest-value remaining experiment and asked for a stack where `torch.compile` works.

**Both halves of that were wrong.**

`torch.compile` is not broken on this stack. `torch` 2.9.1 and `torch_xla` 2.9.0 are a matched
pair, `openxla` is registered, and `add`/`mul`/`relu` all compile on XLA tensors. What fails is a
specific set of ops — `silu`, `gelu`, `Embedding`, `Softmax`, `CrossEntropyLoss`, `topk`, `argmax`,
`Dropout` — because `torch_neuronx` replaces them with XLA user computations whose dispatch
predicate accepts a `FakeTensor` and then rejects it. That is a real upstream bug affecting nearly
any transformer, and it is filed separately (Finding #23). It was not a reason to stop.

More importantly, **`torch.compile` was never the right instrument.** torch-xla is *already* a lazy
graph runtime; ops accumulate into an HLO graph and compile at `mark_step()`. So the question was
answerable with no `torch.compile` at all, by counting device executions:

| variant | wall | device executions | per call |
|---------|------|-------------------|----------|
| 28 NKI calls, 1 `mark_step` | 1446.37 ms | **1** | 51.66 ms |
| 1 NKI call, 1 `mark_step` | 52.80 ms | 1 | 52.80 ms |
| 28 torch ops, 1 `mark_step` | 1.23 ms | 1 | 0.04 ms |
| 1 torch op, 1 `mark_step` | 0.25 ms | 1 | 0.25 ms |

The 28 NKI calls already share one graph and one device execution (196-node graph), and still cost
28x. Graph batching was never going to help. The control scales sublinearly, so the harness sees
batching when batching works — and note that `F.silu` on Neuron is itself an XLA user computation,
so **28 XLA custom calls cost 1.23 ms while 28 NKI custom calls cost 1446 ms.** The problem was
never that custom calls don't fuse.

That relocated the cost to somewhere batching can't reach, and the profile finished the job:

| instrument | result |
|---|---|
| neuron-explorer on the 28-call NEFF | device `total_time` **0.609 ms**, 43% MBU, 95% active |
| torch-xla counters | `ExecuteTime` 0.92 ms, `LazyTracing` 0.28 ms, `TransferToDevice` 0, `CompileTime` 0 |
| wall-clock split | **99.9% of 1459 ms spent before `mark_step`** |
| cProfile of one call | 51 of 52 ms in `select.poll` ← `subprocess.check_output` ← `_detect_target` |

A 2400x gap between device time and wall time eliminates every device-side explanation at once.

The cause, confirmed by reading the source: `_compile_opts()` calls `resolve_target()` on every
invocation, which forks `neuron-ls` and parses its stdout. `CompileOptions` is part of the compile
cache *key*, so a cache hit still pays it. Two fixes verified in one process, baseline re-run last
as a control, accuracy asserted on every variant:

| variant | per call | speedup | cos_sim |
|---------|----------|---------|---------|
| baseline | 51.74 ms | — | 0.999938 |
| `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` | 0.50 ms | **102.8x** | 0.999938 |
| `lru_cache(_detect_target)` | 0.49 ms | **105.5x** | 0.999938 |
| baseline again | 51.43 ms | — | 0.999938 |

### What survives, and what it means

The amortization argument in "Why this is structural" above turns out to be **right for the wrong
reason.** Re-derived from the corrected mechanism: the residual dispatch cost after the fix is
~0.59 ms/call against 0.02 ms of device time, attributable to `create_computation` rebuilding the
XLA computation and its HLO protobufs on every invocation. A torch op costs 0.02–0.03 ms. So a
swapped kernel must save >0.59 ms of torch time per call to break even, and these ops are 15–30x
short of that.

The mismatch is therefore **not** that NKI can't fuse into the graph — it demonstrably does, at 43%
memory-bandwidth utilisation. It is that NKI's eager per-call dispatch is too expensive to amortise
over one small layer, so the granularity that wins is the granularity the Kernel Hub cannot
express. Findings #17 and #18 arrive at the same place from weight layout and sharding.

### The residual amortises with scale, which is measured and matters for the recommendation

If the residual is fixed per call, the penalty must shrink as work per call grows. NKI call count is
set by model depth (169, fixed), so raising sequence length tests this directly:

| run | baseline | kernelized | MFU kern | penalty | added per call |
|-----|----------|------------|----------|---------|----------------|
| seq 512 | 42.04 ms | 141.43 ms | 1.50% | 3.36x | 0.588 ms |
| seq 2048 | 108.76 ms | 223.99 ms | **4.81%** | **2.06x** | 0.682 ms |

2.59x more baseline work, but only 1.16x more cost per call. The penalty nearly halves, and
kernelized MFU at seq 2048 is approaching baseline MFU at seq 512.

Qualifications, because the trend reads more encouraging than the arithmetic supports: 1.16x is not
1.0x, so ~16% of the residual does scale with problem size and it is *near*-fixed rather than fixed.
And extrapolating, a step needs ~1150 ms of real work for this overhead to fall below 10% — roughly
10x seq 2048 on a 0.6B model. Reachable at production scale, but it means per-layer swapping
approaches parity only there, not on the models someone would try first.

**Parity is also not the goal.** Reaching it means the kernels stop costing anything. A *speedup*
additionally requires each kernel to beat the torch op it replaces, which this PoC has not
demonstrated for any of the three.

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
| `torch.compile` fails on most transformers | **not** a broken stack: `add`/`mul`/`relu` compile fine. `torch_neuronx` overrides `silu`, `gelu`, `Embedding`, `Softmax`, `CrossEntropyLoss`, `topk`, `argmax`, `Dropout` with XLA user computations that accept a `FakeTensor` and then reject it, so Dynamo cannot trace them. `torch_xla.compile()` works around it. |
| Every NKI call forks a subprocess | `_detect_target()` runs `neuron-ls` per invocation, ~52 ms, outside the compile cache. One decorator fixes it (Finding #24). |
| Every NKI call rebuilds its XLA computation | `create_computation` + pyhlo scribe + 168 protobuf enum lookups per call, ~0.59 ms. The residual after the fix above. |

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

### The fourth instance, which is a different failure mode and the more dangerous one

The three above are harness bugs: the measurement was invalid. The fourth had **valid measurements
and an invalid conclusion**, which is harder to catch because nothing is broken.

Findings #20 and #21 attributed the ~53 ms to graph-transition or NEFF-switching cost. That
hypothesis survived four separate experiments — varying interleaving, varying data volume, ruling
out recompilation, swapping our kernels for production ones — and every result came back consistent
with it. It was wrong. The cost was a `neuron-ls` subprocess on the host.

It survived because **all four experiments measured wall-clock time at the framework level, and
none of them could see inside the 52 ms.** More variants of the same instrument would never have
falsified it. Two changes of instrument did, immediately: a device profile (0.609 ms of device time
against 1459 ms of wall time) and then a Python profile (which named the function).

There is also a smaller lesson in how the hypothesis was defended. Finding #21 listed three
candidate explanations ranked by plausibility. All three were device-side, because the framing had
already concluded the cost was inside the device execution. The missing candidate — that the cost
never reached the device — was not ranked low, it was absent. **Enumerating candidates within a
single framing feels like rigour and is not.**

Two practices follow, and they generalise past this stack:

- **When a hypothesis has survived several tests and the story still doesn't close, change
  instrument rather than adding a variant.** Repeated survival is evidence about the instrument as
  much as about the hypothesis.
- **Measure the two ends against each other.** Device time versus wall time is one number each,
  and their ratio invalidated an entire class of explanation at once. It should have been the first
  thing measured, not the fifth.

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
| 1 | **Cache `_detect_target()`** — it forks `neuron-ls` on every kernel invocation, ~52 ms, outside the compile cache | NKI | one decorator | **102x verified, accuracy-neutral** |
| 2 | **Scope caching the per-call XLA computation build** (`create_computation` + pyhlo scribe, ~0.59 ms/call) | NKI / torch-neuronx | unknown | the difference between 3.4x slower and near parity |
| 3 | Route XLA-on-Neuron to `Device(type="neuron")` (3 sites) | transformers + `kernels` | ~12 lines | **fix verified sufficient** |
| 4 | Set `torch.neuron` so `_backend()` reports neuron | `torch_neuronx` | 1 line | unblocks two things |
| 5 | Add `nkilib` to `python_depends.json` under `neuron` | HF `kernels` | ~6 lines | `nki` is precedent |
| 6 | Make `torch_neuronx`'s op overrides fake-tensor safe (exclude fake/meta in the dispatch predicate; add abstract impls) | `torch_neuronx` | small per op | breaks `torch.compile` on nearly every transformer |
| 7 | Weight-layout contract for fused kernels (a `prepare_weights` hook, or kernels absorb layout) | HF `kernels` + nki-library | design decision | blocks all fused ports |
| 8 | Fused MLP divide-by-zero single-core above `intermediate_size` 4096 | nki-library | bug fix | boundary measured, 10 data points |
| 9 | Document `experts_implementation="batched_mm"` for MoE on Neuron | Neuron docs | doc change | customer-facing |
| 10 | NKI tutorials teach `nl.arange`, removed in 0.5.0 | NKI docs | doc change | misleads new authors |

Full detail with exact code locations and ready-to-paste patches in `docs/upstream-fixes.md`.

**Sequencing matters.** Item 1 is the highest ratio of value to effort in this document and is
independent of everything else — it benefits any eager NKI usage, not just the Kernel Hub. Item 2
gates whether items 7 and 8 are worth anyone's time: there is no point designing a
weight-transformation contract for kernels that cannot be a net speedup. Items 3, 4, 5, 9 and 10
are worth doing regardless, because they concern correctness, reachability and documentation
rather than performance. Item 6 is unrelated to this integration and is filed because we found it.

**A previous version of this table led with "answer the graph-mode question."** That is now
answered and was the wrong question — 28 NKI calls already share one HLO graph and one device
execution and still cost 28x, so graph batching was never the lever. Withdrawn.

---

## What is not done

Stated plainly so nobody inherits a false impression:

- **No backward kernels.** All three are `has_backward=False`; training mode falls back.
  nki-library's `rope_hf` has a backward path and `nl.silu_dx` exists, so this is feasible but
  unbuilt.
- **No torch.compile support.** `can_torch_compile=False` on all three. `torch.compile` does work
  on this stack for ops `torch_neuronx` hasn't overridden, so this is now testable — we did not
  test it, because Finding #21 showed graph batching is not the lever for the performance problem.
- **The residual 0.59 ms/call was not fixed, only attributed.** Caching `create_computation` is a
  larger intervention than one decorator and sits inside `torch_xla`'s op-registry path. Whether it
  is feasible is a scoping question for the NKI team, not a claim we are making. Until it is
  answered, "the kernels could be near parity" is a hypothesis, not a result.
- **No Hub upload.** Flat layout validated as loadable, `digest` confirmed optional, minimum
  repo is two files. Blocked on the repo-home decision (`aws-neuron/` vs `kernels-community/`)
  and on item 4 for an honest dependency declaration.
- **No fused MLP.** Blocked by items 5 and 6.
- **No MoE-specific kernel.** Gap analysis instead — see `deliverables/week-5-moe-gap-analysis.md`.
  The best target turns out to be the routing `sort`/`histc`, not the expert matmul.
- **MFU measured on Qwen3-0.6B, not 8B.** Full depth, so it is a real model, but not the largest
  one. With the fix applied this caveat cuts the *other* way, and we measured it via sequence length
  rather than model size: 4x the sequence narrows the penalty from 3.36x to 2.06x. An 8B model would
  narrow it further. We did not run 8B at full depth.
- **No demonstration that any kernel beats the torch op it replaces.** Every performance result here
  is about dispatch overhead. Whether NKI RMSNorm is intrinsically faster than torch RMSNorm at a
  given shape is unmeasured, and it is a separate question from all of the above.
- **Single core only.** Eager per-layer swap doesn't manage multi-core; SPMD was stripped from
  the RoPE port.

---

## Closing assessment

The optimistic read of this PoC is that the hard parts worked. The Kernel Hub mechanism is
sound on Neuron, the kernels are correct on two model architectures, the upstream interception
surface is large and already wired, and the fixes needed to reach it are small and well
understood.

The honest read is that we measured the thing the brief was really asking about — whether this
makes models faster — and it does not, yet. But the reason changed twice under measurement, and
where it landed is more encouraging than where it started.

The first answer was **208x slower, structurally**. That was wrong. Most of it was a single
uncached subprocess: `_detect_target()` forking `neuron-ls` on every kernel invocation, ~52 ms a
time, sitting outside the compile cache because its result is part of the cache key. One decorator
recovers 102x per call and takes the model from 208x slower to 3.4x slower, with bit-identical
numerics. That bug is not specific to the Kernel Hub, and anyone invoking NKI kernels per-layer
from eager PyTorch is paying it right now.

The second answer is **3.4x slower, for a smaller and better-understood reason**: ~0.59 ms of
host-side dispatch per call, against 0.02 ms of device time, spent rebuilding the XLA computation
and its HLO protobufs on every invocation. That is the same shape of problem as the first — work
that is cacheable per `(kernel, shape, dtype)` being redone per call — and whether it is fixable is
a scoping question we have handed over rather than answered.

What survives from the original structural argument, re-derived correctly: a swapped kernel must
save more than ~0.59 ms of torch time per call to be a net win, and RMSNorm, RoPE and SiLU at these
shapes are 15–30x short. So per-layer swapping of *small* ops cannot win on arithmetic, however
good the kernel is. Winning requires replacing more work per call — fused kernels — which is what
`nki-library` actually ships and what the Kernel Hub's per-layer contract cannot currently express.
Findings #17 and #18 reach the same conclusion from weight layout and from sharding.

There is a third route, and it is the one the data actually points at: **more work per call without
changing the kernels at all.** The residual is near-fixed per call, so 4x the sequence length halves
the penalty (3.36x → 2.06x). That direction is free — it needs no engineering, only larger models
and longer sequences than a 0.6B at seq 512. It does not reach a speedup on its own, but it means
the gap we measured is the *worst* case rather than the representative one.

Note what this is *not* evidence of. The kernels are fine: a 28-call NEFF executes in 0.609 ms at
43% memory-bandwidth utilisation and 95% engine active time. The Kernel Hub mechanism is fine: it
swapped three kernels into two model architectures with matching logits and no model-code changes.
Neither the kernel quality nor the integration design caused the regression we spent most of this
project measuring.

**The single most valuable next step is item 1 — cache `_detect_target()`.** It is one decorator, it
is verified, and it is worth 102x per call to every eager NKI user on the platform. Item 2 then
decides whether this integration can reach parity, and it is a scoping question rather than an
experiment. We would recommend both before committing engineering time to kernel porting.

One closing note on process, since it changed the answer more than any single measurement did. This
document previously stated the 208x figure with a structural explanation, and that explanation had
survived four experiments. It was overturned by asking one question the experiments could not
answer — how much of the wall time is actually on the device — and the answer was 0.04%. The
measurements were never wrong. The framing was, and no additional variant of the same measurement
would have revealed it.
