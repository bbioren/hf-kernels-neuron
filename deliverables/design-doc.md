# HuggingFace Kernel Hub on Trainium — design and results

**Author:** Ben Bioren
**Reviewers:** Pinak Panigrahi (manager), John Gray (mentor)
**Status:** for review
**Companions:** [`results/`](../results/README.md) for every number with provenance,
[`docs/CODE_GUIDE.md`](../docs/CODE_GUIDE.md) for a reading order through the code,
[`docs/poc-findings.md`](../docs/poc-findings.md) for the full findings log

---

## The correction that prompted this document

You said there should not be a slowdown. **You are right, and my reporting has been leading with
the wrong number.**

The figure that travelled — "kernelizing Qwen3 is 208x slower" — is real but it is *pre-fix*. It was
caused by a one-line bug in NKI's dispatch path, not by the integration. And the figure that
replaced it — "2.5–2.7x slower on device" — is real but comes from a chained microbenchmark that is
deliberately NKI's worst case.

Measured in a real forward pass, the honest decomposition is:

| term | ms/step | share of the gap |
|---|---|---|
| **dispatch** — framework overhead, per kernel call | **91.608** | **91.6%** |
| **device** — forfeited compiler fusion | 8.392 | 8.4% |
| total gap vs baseline | 100.0 | 100% |

So: **there is no structural slowdown.** There is a framework bug worth 102x per call that is
already fixed and verified, a second caching bug of the same kind that accounts for most of what
remains, and an ~8% device cost from replacing ops the compiler was already fusing. With the
dispatch path fixed the projection is **~1.18x slower** — near parity, not a regression.

Everything below is the evidence for that, the design it sits on, and what I got wrong on the way.

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

**Denominator stated explicitly**, because Trn2 has two conventions plus an LNC subtlety:
632 TFLOPS/device (TensorEngine bf16) ÷ 2 for LNC2 = **316 TFLOPS per core**. The published 667
figure includes VectorEngine and ScalarEngine. FLOPs per step are computed explicitly
(670.42 GFLOP), not estimated.

### MFU

| configuration | step ms | MFU | vs baseline |
|---|---|---|---|
| baseline, seq 512 | 42.04 | 5.05% | — |
| all 3 kernels, seq 512, **before the fix** | 8753.65 | 0.02% | 208x slower |
| all 3 kernels, seq 512, **after the fix** | 141.43 | 1.50% | 3.36x slower |
| baseline, seq 2048 | 108.76 | 9.90% | — |
| all 3 kernels, seq 2048, after the fix | 223.99 | 4.81% | **2.06x slower** |

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

| variant | ms/call | speedup | cos_sim |
|---|---|---|---|
| baseline (no override) | 51.74 | — | 0.999938 |
| `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` | 0.50 | **102.8x** | 0.999938 |
| `lru_cache(_detect_target)` | 0.49 | **105.5x** | 0.999938 |
| baseline again (control) | 51.43 | — | 0.999938 |

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

The uncomfortable corollary, and the sharpest result here: **the ops the Kernel Hub is best at
intercepting have the least to gain from it.** 115 RMSNorm registrations, 95 RoPE model files, one
decoration covering every `ACT2FN` activation — all small, memory-bound, already fused. Reach and
benefit are inversely correlated. That argues for pointing the mechanism at coarser ops, not for
abandoning it.

---

## 8. What is not done

- **Raw artifacts are missing.** The trn2 instance expired and every artifact lived in `/tmp` on it.
  Numbers survive in commit messages; scripts survive. `make results` regenerates everything into
  `results/raw/`. See [`results/raw/README.md`](../results/raw/README.md).
- **No run has confirmed the results are independent of compiler flags.** `NEURON_CC_FLAGS` was
  unset throughout. This is the one config choice that could invalidate the device comparisons, and
  it is the most plausible technical form of "there shouldn't be a slowdown." Now *instrumented*:
  `scripts/probe_compiler_flags.py` runs as the fourth stage of `make results` and reports whether
  the NKI/torch ratio moves across `{unset, --target trn2, +--lnc 1, +--lnc 2, +-O2}`.
- **The ~1.18x figure is a projection**, computed from the in-situ decomposition. Not measured.
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
