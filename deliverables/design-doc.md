# HuggingFace Kernel Hub on Trainium — design and results

**Author:** Ben Bioren
**Reviewers:** Pinak Panigrahi (manager), John Gray (mentor)
**Status:** for review
**Companions:** [`results/`](../results/README.md) for every number with provenance,
[`docs/CODE_GUIDE.md`](../docs/CODE_GUIDE.md) for a reading order through the code,
[`docs/poc-findings.md`](../docs/poc-findings.md) for the full findings log

---

## The correction that prompted this document

You said there should not be a slowdown, and that we should see a speedup. **Both are right, and my
reporting had been leading with the wrong number and testing the wrong kernels.** Two things have
changed since that feedback.

### 1. The slowdown was two caching bugs, and both are now fixed and measured

The figure that travelled — "kernelizing Qwen3 is 208x slower" — is real but *pre-fix*, caused by a
one-line bug in NKI's dispatch path. Fixing it left 3.31x. Finding the second bug of the same kind
leaves **1.62x at seq 512 and 1.37x at seq 2048**:

| stage | added cost per kernel call | model slowdown (seq 512) |
|---|---|---|
| as shipped | 52.25 ms | 206x |
| + memoise `_detect_target` (Finding #24) | 0.605 ms | 3.31x |
| + register the XLA computation once (Finding #28) | **0.162 ms** | **1.62x** |
| device floor — what no dispatch fix can remove | 0.0495 ms | — |

**322x** off the per-call overhead, from two one-line changes. Both are the same bug: a cache exists
in the framework and the surrounding code path defeats it. Neither is a property of per-layer kernel
dispatch on Neuron.

### 2. There IS a speedup — on the right kernel, at the right sequence length

The three kernels I had been measuring — RMSNorm, RoPE, SiLU — *cannot* produce a speedup, and that is
now a proven statement rather than a disappointing result. They are small, memory-bound, and the
compiler already fuses them into their neighbours: across a 28-op chain, torch's marginal HBM traffic
is ~0 MB/call because the chain collapses into a single pass, while the NKI kernels sit at exactly the
unfused floor of 6.29 MB/call. **The kernels are optimal and still lose, because you cannot beat not
touching memory.**

So I tested the candidate the analysis actually favoured: `nkilib`'s flash attention kernel. Device
time, Qwen3-0.6B head geometry, causal, single core:

| seq | NKI flash | torch eager | result | torch HBM/layer vs NKI |
|---|---|---|---|---|
| 512 | 0.2463 ms | 0.1225 ms | 2.01x slower | 0.4x |
| 1024 | 0.4939 ms | 0.4269 ms | 1.16x slower | 0.8x |
| **2048** | **1.1438 ms** | **1.6902 ms** | **1.48x FASTER** | **8.3x** |
| **3072** | **1.8484 ms** | **3.9062 ms** | **2.11x FASTER** | **14.9x** |
| 4096 | 2.8295 ms | 1.5784 ms | 1.79x slower | 5.9x |

Flash attention is an *algorithmic* restructuring — it never materialises the `[heads, S, S]` score
matrix — and a compiler does not derive that; it fuses elementwise chains, it does not re-derive the
algorithm. Below seq ~1500 the compiler can keep the score matrix resident and there is nothing to
win. Above it, the `S²` matrix stops fitting and the kernel wins by up to 2.11x.

It is a **window, not a threshold**, and the upper edge is the more interesting half: at 4096 torch's
traffic *drops 47%* while its score matrix grows, so the compiler switches strategy there and becomes
competitive again. The window closes because the compiler improves, not because the kernel degrades.

### What this changes

The recommendation is no longer "the mechanism works but there is nothing to gain." It is:
**point the mechanism at attention, fix the two dispatch caches, and state the sequence range.** §4
and §7b are the evidence; §8 is what is still not done.

---

## 1. What this project is

The HuggingFace `kernels` library replaces `nn.Module.forward()` methods at runtime with optimised
implementations pulled from the Hub. It is merged to transformers mainline. If Neuron registers
`"neuron"` entries in its kernel mapping, every HF model with RMSNorm, rotary embeddings or standard
activations can pick up NKI kernels automatically, in eager mode, with graceful fallback.

That makes it the highest-leverage HF integration point for Neuron on paper: **per-kernel work that
scales across the model zoo** rather than per-model work. The PoC's job was to prove the mechanism
end to end and recommend whether to invest.

### Goals

1. Package NKI kernels so the HF Kernel Hub can load them on Trainium
2. Validate end-to-end on a stock HuggingFace model, unmodified
3. Measure the performance impact honestly, with the MFU denominator stated
4. Recommend whether Neuron should build first-class support

### Non-goals

Backward kernels, `torch.compile` support, Hub publishing (no external side effects were permitted),
and multi-core execution. All are listed under §8.

---

## 2. How the integration works

### The interception mechanism

`kernelize()` walks the model tree, matches each layer's class name against a
`(layer_name, device) -> kernel` mapping, and rebinds `forward`. Weights stay in place; only the
method pointer changes.

```
model                          kernelize(model, device="neuron")
 └─ Qwen3DecoderLayer                    │
     ├─ input_layernorm  ────────────────┤ matches "RMSNorm"  -> NeuronRMSNorm.forward
     ├─ self_attn                        │
     │   └─ rotary_emb   ────────────────┤ matches "rotary_pos_emb" -> NKI RoPE (function)
     └─ mlp                              │
         └─ act_fn      ─────────────────┘ matches "SiLU"     -> NeuronSiLU.forward
```

Two kinds of replacement are needed, and they are not interchangeable:

- **Layer replacement** for RMSNorm and SiLU — a stateless `nn.Module` subclass whose `forward()`
  reads weights off the adopting module via `self`. No `__init__`.
- **Function replacement** for RoPE, because `apply_rotary_pos_emb` is a free function, not a
  module. This needs a `FuncRepository`, and its default for `has_backward` is `True` — the opposite
  of the layer case, which is an easy footgun.

### Kernel structure

Single-file, per transformers PR #46754:

```
kernels/neuron_rmsnorm/
  __init__.py      NeuronRMSNorm class + `class layers:` namespace + the @nki.jit kernel
  metadata.json    {"backend": {"type": "neuron"}}
```

Each kernel carries a device guard: `@nki.jit` requires XLA tensors, so a CPU tensor must fall back
to PyTorch. **That guard is also the project's most dangerous property** — see §5.

### What had to be built around it

`use_kernels=True` **cannot reach Neuron at all.** transformers derives the device from
`model.device.type`, which on Neuron is `"xla"`, never `"neuron"`. Because a `Device` object is
passed rather than a string, validation is skipped, nothing matches, and `kernelize()` returns
success with every layer unchanged — a silent no-op.

The fix is ~12 lines across three sites, and it is verified sufficient: it takes Qwen3 from 0 to 9
swapped layers. Until it lands, everything here routes through
`scripts/neuron_kernel_registration.py::kernelize_for_neuron()`, which calls the `kernels` library
directly with `device="neuron"`. That shim patches a function object in-process only; it does not
touch the installed venv, because a venv edit would be irreproducible for a customer and would
misrepresent the integration's real state.

---

## 3. What was built and validated

Three kernels, all execution-verified on trn2:

| Kernel | Interception point | Upstream reach | Source | Accuracy |
|---|---|---|---|---|
| RMSNorm | `RMSNorm` (layer) | 115 registrations | tutorial-derived, see below | 11/11 cases |
| RoPE | `rotary_pos_emb` (function) | 95 model files | **port of `nkilib/core/embeddings/rope_hf.py`** | 20/20 + 6/6 guards |
| SiLU | `SiLU` (layer) | 1 decoration, covers all `ACT2FN["silu"]` | tutorial-derived | 9/9 cases |

End-to-end on **stock, unmodified** models:

| model | logits `cos_sim` | kernels engaged | fallbacks |
|---|---|---|---|
| Qwen3 dense | 1.000001 | RMSNorm 9, RoPE 2, SiLU 2 | 0 |
| Qwen3-MoE | 1.000002 | RMSNorm 9, RoPE 2, SiLU 2 | 0 |

The MoE row is the load-bearing one for the per-kernel thesis: **the same three kernels transferred
to a second architecture with zero code changes.**

### Why two kernels are tutorial-derived rather than ported

This looks like a shortcut and is not. Standalone versions do not exist in nki-library:

- `nkilib/core/rmsnorm/` contains one kernel, `rmsnorm_quant.py`, which fuses RMSNorm with FP8
  quantisation and **always quantises** — `QuantizationType.NONE` is not a validated input. The only
  plain-RMSNorm code is `_rms_normalize_tile()`, an internal subroutine.
- `nkilib/core/` has **no activations module**: `attention, cumsum, embeddings, max, mlp, moe,
  moe_block, output_projection, qkv, quantization, rmsnorm, router_topk, subkernels, topk, utils`.
  SiLU exists only inside `mlp/mlp.py`.
- `embeddings/rope_hf.py` is standalone and already HF-shaped. It is the one op of the three that was
  ported directly.

That is itself the central finding arriving early: **the ops the Kernel Hub can intercept mostly do
not exist as separable units in nki-library, because nki-library ships fused megakernels.** I
recorded it in Week 2 and did not let it inform what I expected from the Week 4 measurement.

---

## 4. Performance: the actual result

### Method

Qwen3-0.6B, full 28 layers, bf16, forward only, single logical core, `NEURON_CC_FLAGS` unset.
That last choice is not an oversight and is checked in [§7a](#7a-was-any-of-this-a-compiler-flag-artifact).

**Denominator stated explicitly**, because Trn2 has two conventions plus an LNC subtlety:
632 TFLOPS/device (TensorEngine bf16) ÷ 2 for LNC2 = **316 TFLOPS per core**. The published 667
figure includes VectorEngine and ScalarEngine. FLOPs per step are computed explicitly
(670.42 GFLOP), not estimated.

**Every row below has been re-run on a second physical instance** with a version-matched stack. The
re-run column is that run. Absolute step times land a few percent higher on the replacement host; the
*ratios* are what reproduce, several to three significant figures.

### MFU

| configuration | step ms | MFU | vs baseline | re-run step ms | re-run ratio |
|---|---|---|---|---|---|
| baseline, seq 512 | 42.04 | 5.05% | — | 44.36 | — |
| all 3 kernels, seq 512, **before the fix** | 8753.65 | 0.02% | 208x slower | 8873.67 | 206x |
| all 3 kernels, seq 512, **after the fix** | 141.43 | 1.50% | 3.36x slower | 146.67 | 3.31x |
| baseline, seq 2048 | 108.76 | 9.90% | — | 109.65 | — |
| all 3 kernels, seq 2048, after the fix | 223.99 | 4.81% | **2.06x slower** | 226.16 | **2.06x** |

The seq-2048 row is the most encouraging measurement here and the easiest to skim past. Call count is
fixed by model depth, so a longer sequence means more work per call, not more calls. 2.59x more
baseline work costs only 1.16x more per call — so the residual overhead is near-*fixed*, and the
penalty nearly halves just by moving to a longer sequence. At production sequence lengths it matters
proportionally less again.

### Root cause of the 208x

`nki/framework/compiled.py::_compile_opts()` calls `resolve_target()` on **every** kernel
invocation. With no override set that falls through to `_detect_target()`:

```python
def _detect_target() -> str:
    if shutil.which("neuron-ls") is None:
        return "trn3"
    out = subprocess.check_output(["neuron-ls"], text=True, timeout=10, ...)
```

It forks a process and parses `neuron-ls` stdout to ask what hardware it is on. **~52 ms per call.**

NKI does maintain `self.func._nki_compile_cache`, but `CompileOptions` is what identifies a compiled
kernel, so target resolution runs while *building the cache key*. **A cache hit still pays the
subprocess in full.**

Verified fix, two independent methods, in one process, baseline re-run last as a control:

| variant | ms/call | speedup | cos_sim | re-run ms/call | re-run speedup |
|---|---|---|---|---|---|
| baseline (no override) | 51.74 | — | 0.999938 | 52.11 | — |
| `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` | 0.50 | **102.8x** | 0.999938 | 0.49 | **105.3x** |
| `lru_cache(_detect_target)` | 0.49 | **105.5x** | 0.999938 | 0.47 | **110.5x** |
| baseline again (control) | 51.43 | — | 0.999938 | 52.07 | — |

Cosine similarity is identical to six decimal places across all four, so neither fix changes what
gets compiled. The override is set to whatever `_detect_target()` returns *on the host*, never a
hardcoded string — a wrong target would compile for the wrong generation and could be silently wrong
rather than an error.

**This is not Kernel Hub specific.** Anything invoking NKI kernels per-layer from eager PyTorch pays
it today.

### Where the remaining 3.36x goes

Profiling a real forward pass and summing device time across the emitted NEFFs:

| | device ms | HBM MB | wall ms |
|---|---|---|---|
| baseline | 14.329 | 2662.4 | 46.65 |
| kernelized | 22.722 | 3779.9 | 146.65 |

**Dispatch 91.608 ms (91.6%), device 8.392 ms (8.4%).** Per call: dispatch 0.5421 ms, device
0.0497 ms — dispatch is ~11x larger.

This split has now been computed from **four independent wall-time pairs across two physical
instances** — 46.65/146.65, 47.52/144.19, 50.138/144.65 and 54.783/153.43 — giving device shares of
8.4%, 8.6%, 8.9% and 8.5%, and projections of 1.18x, 1.18x, 1.17x and 1.15x. The conclusion does not
depend on which pair is used.

**Which is worth spelling out, because the two inputs have very different precision and it is the
imprecise one the headline leans on.** Device time repeats to ~0.3% across those runs (baseline
14.318 / 14.329 / 14.330 ms; kernelized 22.676 / 22.696 / 22.722 / 22.740) because it is a hardware
trace of a fixed NEFF. Wall time swings 17% (baseline 46.65 to 54.78 ms) because it includes host
scheduling. So **the ~8.4 ms device gap is the solid number**; the percentage and the projection
inherit the wall noise and are honestly quoted as ranges — 8.4–8.9% device, 1.15–1.18x projected —
rather than as point estimates.

The dispatch residual is the same class of bug as the first one, two orders of magnitude smaller:
cProfile puts it in `create_computation` rebuilding the XLA computation and its HLO protobufs *on
every invocation*, on a warm path where the kernel has already run. Whether it is cacheable is the
top open question (§8) — I did not attempt it, because it sits inside `torch_xla`'s op-registry path
and a wrong guess there could be silently incorrect rather than an error.

### The 8.4%: why any device cost at all

A NKI kernel reaches the compiler as an **opaque custom call, and the compiler cannot fuse across
it.** Replacing a torch op therefore does not just add dispatch cost — it removes a fusion the
compiler was already performing. Each swapped op materialises to HBM where the data previously
stayed resident.

The kernels are not at fault, and this is measurable rather than asserted. Solving
`traffic(N) = FIXED + N × MARGINAL` across N=1 and N=28:

| | marginal traffic/call | vs unfused floor |
|---|---|---|
| NKI SiLU | 6.29 MB | **1.00x** |
| torch SiLU | ~0.00 MB | 0.00x |
| NKI RMSNorm | 6.29 MB | **1.00x** |
| torch RMSNorm | ~0.00 MB | 0.00x |

The unfused floor for a `[512, 3072]` bf16 tile is 2 tiles = 6.29 MB — one read in, one write out.
**NKI's marginal traffic is exactly that**, so the kernels spill nothing and are optimal for an op
that cannot fuse. Torch's traffic is independent of N, which is only possible by fusing the chain
into one pass.

For memory-bound ops fusion *is* the optimisation, so a swapped kernel competes against not touching
memory at all. In the chained microbenchmark that is 2.5–2.7x; in a real model, where these ops sit
between matmuls, it is the 8.4% above.

The chain-length dependence confirms this from a second direction. Comparing NKI against torch at
both call counts:

| op | N=1 | N=28 |
|---|---|---|
| SiLU | 1.91x | 2.72x |
| RMSNorm | 2.37x | 2.56x |

The ratio *grows* with chain length, which is what a fusion explanation predicts and a
kernel-quality explanation does not: a longer chain gives torch more to fuse and leaves NKI
unchanged. It also confirms N=28 is the worst case rather than the typical one. (N=1 is not the fair
number either — at N=1 the comparison is dominated by the custom call's 12.58 MB of fixed NEFF setup
traffic. The marginal-traffic regression, not either ratio, is the instrument that separates fixed
from per-call cost, which is why it is the one above.)

This comparison was invisible until a bug was fixed: the summariser keyed profile pairs on op name
alone, so the N=1 and N=28 directories collided and whichever came last in the argument list won. It
happened to be N=28 for both, so the published ratios were right — by argument order, not by
construction.

### The fused MLP, which should have been the win

The fused MLP replaces a whole region — gate + up + SiLU + down — so it does the fusion internally
instead of interrupting the compiler's, and it contains two real matmuls, so there is compute to
optimise. It is the one candidate where a speedup was plausible.

I had written it off on Finding #18 (won't compile single-core above `intermediate_size` 4096). But
#18's own data shows `hidden_size=1024, intermediate_size=3072` **passes** at cos_sim 0.999995 —
exactly Qwen3-0.6B's MLP shape. So it works for the benchmarked model, and I had never timed it.

| shape | blocks | impl | device ms | per block | HBM MB | MBU | cos_sim |
|---|---|---|---|---|---|---|---|
| H=1024, I=3072 | 28 | NKI | 8.321 | 0.2972 | 2172.6 | 36.5% | 0.999979 |
| H=1024, I=3072 | 28 | torch | **2.782** | **0.0993** | **1059.1** | 53.2% | 0.999979 |
| H=4096, I=4096 | 8 | NKI | 11.625 | 1.4532 | 3288.3 | 39.5% | 0.999977 |
| H=4096, I=4096 | 8 | torch | **4.180** | **0.5225** | **1619.1** | 54.1% | 0.999977 |

NKI/torch is **2.99x** and **2.78x**. The gap barely narrows at the largest shape it can run
single-core, traffic stays at ~2x, and torch gets consistently better bandwidth utilisation — so it
is not a shape artifact.

**Interpretation, and it reframes #18:** nki-library kernels are built for the NxDI inference
pipeline — multi-core SPMD, large shapes, often quantised. Single-core, a kernel has one core's SBUF
and tiles far more finely than designed, paying a HBM round-trip at every tile boundary. A `floordiv`
by zero when `intermediate_size > 4096` is what a shard-count calculation looks like with no shard
grid. **#18 is a design boundary, not a bug — I filed it as a divide-by-zero to fix and should have
read it as the kernel telling me single-core is not its execution model.**

The `kernelize()` contract gives a kernel exactly one core, no launch grid, and weights in whatever
layout the model already has. That is the mismatch, and it is now a measured 3x rather than an
inference from weight layouts and compile errors.

---

## 5. Methodology, which may be the most transferable part

On a lazy-execution accelerator backend, **both correctness and performance measurements fail
silently by default.** A fallback is numerically correct. An eliminated computation is fast. A
harness bug looks like a platform limit. None of these produce an error.

Six times a plausible-looking measurement turned out to be measuring something else:

| # | Symptom | Actual cause | Would have concluded |
|---|---|---|---|
| 8 | `max_diff = 0.00e+00`, "bit-identical" | tests fed CPU tensors, so the kernel never ran and the fallback was compared to itself | "RMSNorm validated" — for a week |
| 19 | latency independent of problem size | outputs discarded, XLA eliminated the computation; timed an empty graph | "NKI is 8–400x slower" |
| 21 | `ModuleNotFoundError` under compile | our loader never registered the module in `sys.modules` | "NKI is incompatible with torch.compile" |
| 24 | ~52 ms/call, flat across a 112x sweep | **valid measurement, invalid conclusion** — attributed to graph transitions, actually a subprocess | "per-layer NKI is structurally launch-bound" |
| 25 | traffic 3.00x the floor at N=1 | divided a non-linear quantity by N; fixed NEFF traffic dominates at small N | "the kernels spill an fp32 intermediate" |
| 26 | torch 12.1 MB/block against an 18.9 MB weight set | shared one weight set across 28 blocks, letting the compiler amortise one load | "the fused MLP is 4.3x slower" |

The first three are harness bugs — a guard catches them. **The last three are worse: valid
measurements with invalid conclusions, which no guard catches because nothing is broken.**

### Guards now in place

1. **Assert execution, don't infer it.** A call counter on the dispatch targets proves the kernel
   ran; numerical agreement cannot.
2. **Treat a suspiciously good result as a bug report.** For a reduction kernel, an exact-zero diff
   means it didn't run. Op-dependent: elementwise ops *should* be bit-identical.
3. **Gate performance numbers on a scaling check.** If latency doesn't respond to problem size,
   suppress the result rather than caveat it.
4. **Compare device time against wall time before forming any hypothesis.** Two numbers. Ours
   differed by 2400x, and that single ratio invalidated four experiments' worth of conclusions.
5. **Vary N before dividing by N.** Fixed and marginal costs are different quantities.
6. **Ask what the harness lets each side amortise** that a real model could not.

### Two lessons worth stating separately

**When a hypothesis has survived several tests and the story still doesn't close, change instrument
rather than adding a variant.** The graph-transition hypothesis survived four framework-level timing
experiments because none of them could see inside the 52 ms. What killed it was one comparison:
**0.609 ms of device time for the whole 28-call NEFF against 1459 ms of wall time** — a ~2400x ratio
that eliminates every device-side explanation simultaneously, regardless of which one you favour. A
device profile and a cProfile settled it in about 35 minutes, after roughly five hours of the wrong
approach.

**A caveat in the text is not a caveat in the conclusion.** I wrote "this chained microbenchmark is
an upper bound, the in-situ magnitude is unmeasured" into the findings doc, and then drafted a
recommendation from that number anyway. Measuring it moved the conclusion materially.

---

## 6. Findings inventory

Full log in [`docs/poc-findings.md`](../docs/poc-findings.md). The ones that matter for review:

| # | Finding | Severity |
|---|---|---|
| 9 | `use_kernels=True` cannot reach the `"neuron"` device — silent no-op. Fix verified sufficient | High |
| 17 | `kernelize()` has no weight-transformation hook; fused kernels want a different layout. Now quantified: 3.533 ms / 1172 MB one-time transpose | High |
| 18 | Fused MLP won't compile single-core above `intermediate_size` 4096 — **reframed as a design boundary by #26** | High |
| 22 | Qwen3-MoE won't run on Neuron at all by default (`torch.sort`/`histc` → unsupported `sort` HLO). Fix: `experts_implementation="batched_mm"`, undocumented | High |
| 23 | `torch_neuronx` op overrides aren't fake-tensor safe, breaking `torch.compile` on nearly every transformer | High |
| **24** | **The 208x: uncached `neuron-ls` subprocess per invocation. Fix verified, 102x/call** | **Critical** |
| 25 | Each NKI call is an optimisation barrier; kernels are provably optimal, the loss is forfeited fusion | Critical |
| 26 | Fused MLP also loses (~3x) single-core; nkilib kernels need an SPMD grid | Critical |

---

## 7. Recommendation

**Yes, invest — but the investment is two caching fixes and a scoping question, not a kernel-porting
programme.**

| # | Ask | Owner | Size | Status |
|---|---|---|---|---|
| 1 | Cache `_detect_target()` | NKI | one decorator | **Verified 102x, accuracy-neutral.** Benefits all eager NKI use |
| 2 | Scope caching the per-call `create_computation` rebuild | NKI / torch-neuronx | unknown | **91.6% of the remaining gap.** Takes 3.36x → ~1.18x |
| 3 | Can a NKI custom call participate in compiler fusion? | NKI / compiler | a question | Decides the last ~18% |
| 4 | NKI kernel for MoE routing (`sort`/`histc`) | NKI | small | Unblocks Qwen3-MoE on Neuron **entirely** |
| 5 | Route XLA-on-Neuron to `Device(type="neuron")` | transformers + `kernels` | ~12 lines | Fix verified sufficient |
| 6 | Make `torch_neuronx` overrides fake-tensor safe | `torch_neuronx` | small per op | Unrelated to this integration; filed because we found it |

**Do not port more small memory-bound ops for performance.** RMSNorm, RoPE and activations are
excellent *mechanism* demonstrations and cannot be wins — they are small, memory-bound, and already
fused. Judge a candidate by **whether it replaces a region the compiler would otherwise fuse**, not
by how many models call the op.

The uncomfortable corollary: **the ops the Kernel Hub is best at intercepting have the least to gain
from it.** 115 RMSNorm registrations, 95 RoPE model files, one decoration covering every `ACT2FN`
activation — all small, memory-bound, already fused. Reach and benefit are inversely correlated.

That is an argument for aiming the mechanism differently, not for abandoning it, and §7b is what it
looks like when you do: **attention is both widely intercepted and genuinely improvable**, and it is
where the 1.48–2.11x lives. The priority list that follows from this is attention first, dispatch
caches second (they are what makes an attention swap worth doing at all), and the small ops kept only
as mechanism proof and fallback coverage.

---

## 7b. Where the speedup is

Findings #25 and #26 produced a criterion, and then found no candidate that met it:

> A kernel wins when it replaces a region the compiler would **not** otherwise fuse well, **and**
> there is real arithmetic to restructure.

| candidate | replaces a region the compiler doesn't fuse? | real arithmetic? | measured |
|---|---|---|---|
| RMSNorm, RoPE, SiLU | no — already fused | no — memory-bound | 2.5–2.7x slower (chained) |
| fused MLP | yes | yes | 2.99x slower — needs an SPMD grid it isn't given |
| **flash attention** | **yes — flash is algorithmic, not a fusion** | **yes — 2 matmuls, causal skipping** | **1.48–2.11x FASTER, seq 2048–3072** |

Attention is the first candidate that passes both halves, and it passes them for a reason that is not
incidental. Flash attention avoids materialising the score matrix by restructuring the computation
around an online softmax. That is a different algorithm, not a fusion of the same one, and a compiler
will not find it. `attention_cte` additionally skips the upper-triangle score tiles entirely under
causal masking, rather than computing them and masking to `-inf`.

It also **worked first try** against the HF-native layout: `tp_q=True, tp_k=True, tp_out=False` maps
straight onto `(batch·heads, seq, head_dim)`, GQA is expressed natively as `batch_size_kv <
batch_size` with no K/V replication, and it runs with no SPMD grid — the exact property the fused MLP
lacked. Correctness was `cos_sim 1.000010` against a CPU fp32 reference on the first run. This is the
first nkilib kernel in the project that dropped into the Kernel Hub's calling convention without a
fight, and that is a data point about porting cost as well as performance.

### Why there is a lower edge

At seq 512 torch moves **3.16 MB/layer** — *below* the 6.29 MB it costs merely to read q, k, v and
write the output once. The only way that is possible is that the compiler fused the whole chain and
kept the score matrix resident. So at short sequences the compiler is already achieving flash
attention's central advantage, and the kernel pays an HBM round-trip at its custom-call boundary to
buy something it does not get. That is the same fusion-barrier mechanism as §4, arrived at from the
opposite direction.

### Why there is an upper edge, which I initially got backwards

My first reading of the 4096 reversal was that the NKI kernel had run out of single-core SBUF — K and V
are 8.4 MB each at that length and `attention_cte` only sections K/V above 10K tokens. Tidy, and it
matched an existing finding.

The traffic column says it is wrong. Torch's HBM traffic per layer goes 279.86 → 748.70 → **395.05** MB
across seq 2048 → 3072 → 4096: it *drops 47%* while the score matrix it is supposedly materialising
grows from 302 MB to 537 MB. At 3072 it moves 2.5x the score matrix; at 4096 it moves 0.74x — less than
one copy. Meanwhile NKI's traffic is exactly linear at every point (16.78 MB per 1024 tokens) and its
time is exactly on trend.

**Nothing degraded on the kernel side. The compiler got better.** The leading explanation is that XLA
switches attention strategy above some threshold; that is checkable by diffing the HLO either side of
it, which has not been done, so it is stated as what the traffic supports rather than as confirmed.

Worth recording how close this came to shipping the other way round: the SBUF story was plausible,
matched a prior finding, and blamed the kernel. It survived only because the sweep records traffic per
configuration rather than just time. A one-number benchmark would have produced a confident wrong
answer.

### How to quote this

The same kernel is 2.01x slower at seq 512 and 2.11x faster at seq 3072. **Any figure from this section
without its sequence length is misleading**, which is the mistake sticking point #18 records this
project already making once. And the window has an upper edge as well as a lower one — that is the part
most likely to be dropped in retelling.

---

## 7a. Was any of this a compiler-flag artifact?

No — and the reason is stronger than "we tried some flags."

This was the project's top open item, and it deserved to be: every measurement ran with
`NEURON_CC_FLAGS` unset, and a bad compiler default would be the cheapest possible explanation for
the whole slowdown. It is also the most plausible *technical* form of the objection that there should
not be a slowdown at all. So it was checked in two halves.

**Wall clock.** Five settings — `{unset, --target trn2, +--lnc 1, +--lnc 2, +-O2}` — one subprocess
and one isolated compile cache each, so no setting can inherit another's NEFF. NKI's own time moves
by **1.02x** across all five (13.82–14.15 ms). No setting rescues it.

The first version of this probe thresholded on whether the NKI/torch *ratio* moved, saw a 1.53x
spread, and read that as "flags matter." Reading the columns separately reverses it: the spread is
one setting (`--lnc 1`) making *torch* 2.3x slower, which flatters the ratio without helping NKI at
all. A ratio can move because its denominator moved. The probe now reports both spreads.

**Device time**, which is the half that matters, because Findings #25 and #26 are device-time claims
and the wall-clock probe is ~97% dispatch by construction (0.494 ms/call *is* the post-fix dispatch
floor). Same five settings, profiling at N=1 and N=28 so marginal traffic can be solved for:

| `NEURON_CC_FLAGS` | NKI ms | torch ms | ratio | NKI MB/call | vs unfused floor |
|---|---|---|---|---|---|
| (unset) | 0.608 | 0.224 | 2.72x | 6.29 | 1.00x |
| `--target trn2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |
| `--target trn2 --lnc 1` | 0.580 | 0.429 | 1.35x | 6.29 | 1.00x |
| `--target trn2 --lnc 2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |
| `--target trn2 -O2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |

NKI device time spread **1.05x**. NKI marginal traffic spread **1.00x** — pinned at exactly the
unfused floor under every setting.

**Why this is the strong form.** The weak result would be "five settings tried, none better," which
leaves a sixth open and invites the reviewer to suggest one. The actual finding is that *the quantity
a better setting would have to move is already at its theoretical minimum.* NKI moves one tile in and
one tile out per call — 6.29 MB for a `[512, 3072]` bf16 tile — which is the least an unfusable
operation can move. There is no headroom for a flag to find. The device gap is structural: an opaque
custom call cannot be fused into its neighbours, and compiler flags do not reach that.

**The 1.35x row is a trap.** It is the best ratio in the table and it is not an improvement: NKI
barely moves (0.608 → 0.580) while torch gets 91% slower (0.224 → 0.429). Reading it as progress is
exactly the mistake the first wall-clock probe made.

Both probes now run as harness stages, so a future SDK or compiler change re-tests this
automatically rather than depending on someone remembering that it was once a question.

---

## 8. What is not done

- **The original raw artifacts are gone.** The first trn2 instance expired and every artifact lived
  in `/tmp` on it. Every measurement has since been re-run on a replacement instance and those
  artifacts *are* committed under `results/raw/` — see [§7a](#7a-was-any-of-this-a-compiler-flag-artifact)
  and the reproduction table in [`results/README.md`](../results/README.md) — but a re-run is a
  reproduction, not a recovery. Where an artifact and a transcribed number differ, both are stated.
- **The ~1.15–1.18x figure is a projection**, computed from the in-situ decomposition, and it is now
  partly superseded: it assumed dispatch went to *zero*, and the measured numbers with both fixes are
  1.62x (seq 512) and 1.37x (seq 2048). Those sit between the old 3.31x and the projection, which is
  what the projection predicted, since Finding #28 removes about two thirds of the dispatch term
  rather than all of it. Quote the measured pair, not the projection.
- **Neither dispatch fix is shipped.** Both are runtime monkeypatches, verified accuracy-neutral on
  this stack, with the NKI source fingerprinted so they refuse to apply to a version they were not
  written against. The deliverable is the diagnosis and the verification, not a deployable patch.
- **The attention win is a 4-layer microbenchmark, not a model.** Device time for chained attention
  layers with distinct K/V. In situ, attention sits between QKV and O projections that force HBM
  boundaries anyway, so the custom-call boundary should cost *less* than it does here — but that has
  not been measured, and Finding #25 made exactly this error in the opposite direction.
- **The attention kernel is not wired into the Kernel Hub.** It is called directly. transformers has
  an attention interface that is the right interception point, and mapping `attention_cte` onto it is
  the obvious next step, but it was not done.
- **Where XLA's attention strategy switches is unknown.** It sets the upper edge of the speedup window
  and therefore the size of the opportunity. Diffing the HLO either side of seq 4096 would answer it.
- **Whether `create_computation` is cacheable is unknown.** Not attempted, deliberately.
- **No backward kernels.** All three are `has_backward=False`; training falls back.
- **No `torch.compile` support.** Now testable (Finding #23 explains what actually fails); not tested,
  because graph batching is not the lever.
- **No Hub upload.** Flat layout validated as loadable, minimum repo is two files. Blocked on the
  repo-home decision and on an honest dependency declaration.
- **No multi-core / SPMD measurement**, which is the configuration nki-library kernels were built
  for. This is the largest gap in the negative half of the recommendation.
- **MFU on Qwen3-0.6B, not 8B.** Full depth, so a real model, but not the largest.

---

## 9. How to review this

```
results/README.md          every number, with the producing script and command
docs/CODE_GUIDE.md         reading order through the code
docs/poc-findings.md       full findings log, including the wrong turns, annotated in place
deliverables/poc-document.md   the longer narrative version of this document
```

Reproduce on a fresh trn2:

```bash
./scripts/sync_to_trn2.sh
ssh trn2 'cd hf-kernels-neuron && make test-all'    # correctness, 5 suites
ssh trn2 'cd hf-kernels-neuron && make results'     # every measurement -> results/raw/
```

Wrong conclusions are annotated in place rather than deleted, so the reasoning trail is reviewable.
Where a claim is a projection or an upper bound, it says so.
