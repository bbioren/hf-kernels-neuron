# Worklog

## CURRENT STATE — read this first

**Branch `main`.** Everything is committed and pushed; `week-3` was merged in and work now goes
directly to `main`. All five test suites pass on trn2. Weeks 3–6 are done, plus two later sessions
that changed the headline twice.

**Live deliverable: [`deliverables/poc-document.md`](deliverables/poc-document.md)**, maintained as a
living document rather than a report — it carries the current recommendation, the current numbers,
and a "where we are struggling" section. [`docs/poc-findings.md`](docs/poc-findings.md) is the full
findings log (33 findings).

**Two things below are superseded and left in place deliberately, because the corrections are the
useful part.** Sessions 1–7 describe the torch-xla stack. Two later results changed the conclusions:

- **A speedup exists** (session 8). `nkilib` flash attention beats the compiler 1.48x at seq 2048 and
  2.11x at 3072 on device. The project had concluded no speedup was available anywhere.
- **Both integration gates were our own mistake** (session 9). On the Native PyTorch drop,
  `model.device.type` is `"neuron"` and `_backend()` returns `Neuron()`, so stock `use_kernels=True`
  works unpatched. The headline upstream ask — the `xla`→`neuron` device-resolution patch — is
  withdrawn. One ask remains: the `"neuron"` mapping entries.
- **And every performance number in sessions 1–7 is torch-xla-scoped.** On native the sign of the
  headline flips (1.62x slower → 1.97x "faster") *because the baseline is 4.32x worse*. See
  Finding #32 before quoting anything.

### The result that matters (sessions 1–7, torch-xla)

**Every `@nki.jit` invocation was forking a subprocess. Fixing it is one decorator and worth 102x
per call.**

`nki/framework/compiled.py::_compile_opts()` calls `resolve_target()` on every invocation, which
falls through to `_detect_target()`, which runs `neuron-ls` to ask the hardware what it is. ~52 ms
per kernel call. It sits *outside* `_nki_compile_cache`, because its result is part of the cache
key — so a cache **hit** still pays it in full. Finding #24.

| | step time | MFU | penalty |
|---|---|---|---|
| baseline, seq 512 | 42.04 ms | 5.05% | — |
| kernelized, before fix | 8753.65 ms | 0.02% | 208x |
| kernelized, after fix | 141.43 ms | 1.50% | 3.36x |
| kernelized, after fix, seq 2048 | 223.99 ms | 4.81% | **2.06x** |

Verified two ways (env override and `lru_cache`), baseline re-run last as a control, cosine
similarity identical to six decimals across all variants. This is **not Kernel Hub specific** —
anything invoking NKI kernels per-layer from eager PyTorch is paying it today.

**I had this wrong, and the correction is the more useful story.** The previous version of this
summary said the ~53 ms was a graph-transition cost, that the mismatch was structural, and that the
decisive experiment needed a stack where `torch.compile` works. All three were wrong:

- `torch.compile` is not broken here. `add`/`mul`/`relu` compile fine on XLA. What fails is the set
  of ops `torch_neuronx` overrides with XLA user computations (`silu`, `gelu`, `Embedding`,
  `Softmax`, `CrossEntropyLoss`, `topk`, `argmax`, `Dropout`) because the dispatch predicate accepts
  a `FakeTensor` and then rejects it. Real bug, filed separately (#23), but not a blocker.
- `torch.compile` was never the right instrument. torch-xla is *already* a graph runtime. 28 NKI
  calls fuse into **one** HLO graph and **one** device execution (196 nodes) and still cost 28x.
- The cost was never on the device. The NEFF containing all 28 calls executes in **0.609 ms** at
  43% memory-bandwidth utilisation and 95% engine active time.

The wrong hypothesis survived four experiments — interleaving, data volume, recompilation, and
our-kernels-vs-production — because **all four measured wall-clock time at the framework level and
none of them could see inside the 52 ms.** Two changes of instrument settled it in about 35 minutes:
a device profile (0.609 ms device vs 1459 ms wall) and then a cProfile, which named the function.

### What still holds — and the harder finding underneath it

Even fixed, the kernels are a net loss. ~0.59 ms/call of dispatch remains against 0.02 ms of device
time, in `create_computation` rebuilding the XLA computation and its HLO protobufs *on every call* —
same class of bug, 100x smaller.

**But dispatch is no longer the binding constraint. Finding #25 is.** Measured on device with
dispatch excluded entirely (`neuron-explorer` `total_time`, N=28 chained):

| | device ms | HBM traffic | marginal traffic/call | MBU |
|---|---|---|---|---|
| NKI SiLU | 0.607 | 188.7 MB | 6.29 MB = **1.00x** the unfused floor | 43.2% |
| torch SiLU | **0.224** | **6.3 MB** | **~0.00 MB** | 3.9% |
| NKI RMSNorm | 1.625 | 188.8 MB | 6.29 MB = **1.00x** the floor | 16.2% |
| torch RMSNorm | **0.637** | **6.4 MB** | **~0.00 MB** | 1.4% |

NKI is 2.71x slower on SiLU and 2.55x on RMSNorm **with dispatch cost removed**. The kernels are not
at fault: their marginal traffic is exactly the theoretical floor for an unfused op — one read in, one
write out, nothing spilled. Torch's traffic is independent of N, which is only possible if the chain
fused into a single pass.

So a NKI custom call is an **optimisation barrier**. The compiler cannot fuse across it, and each swap
forces a HBM round-trip where the data previously stayed resident. For memory-bound ops fusion *is*
the optimisation, so the kernel is competing against not touching memory at all and cannot win.

### And then the in-situ measurement moderated that, which is the sixth correction

The 2.5–2.7x above is from a **chained** microbenchmark — 28 identical ops back to back — which is
simultaneously the compiler's best case and NKI's worst. I wrote that caveat into the finding and then
drafted the recommendation as if it weren't there, demoting the dispatch fix on the microbenchmark's
strength. Measuring it in a real forward pass reversed the ranking:

| term | value | share of the 100 ms wall gap |
|---|---|---|
| device gap (14.329 → 22.722 ms) | 8.392 ms | **8.4%** |
| dispatch gap | 91.608 ms | **91.6%** |

Per NKI call: device 0.0497 ms, dispatch 0.5421 ms — dispatch is ~11x larger. The fusion barrier is real
(1.59x device time, 1.42x HBM traffic) and second-order.

**So caching `create_computation` is the decisive fix after all**, worth roughly 3.4x → **1.18x**, and
the fusion question decides whether the last ~18% is recoverable. Break-even is **close but not
reached**, not unreachable.

What survives all of it: **the ops the Kernel Hub is best at intercepting have the least to gain from
it.** RMSNorm 115 registrations, RoPE 95 model files, every `ACT2FN` activation via one decoration — all
small, memory-bound, already fused. Reach and benefit are inversely correlated. That is a reason to point
the mechanism at different ops, not to abandon it.

Four findings converge from independent directions: #17 (weight layout), #18 (sharding), #24 (dispatch
cost, ~92% of the regression) and #25 (compiler fusion, ~8%).

**The lesson from the over-claim, stated separately because it is subtle:** a caveat written into a
finding is not a caveat honoured in the conclusion drawn from it. The representativeness limitation was
in the document, in writing, before the recommendation was drafted — and the recommendation reasoned past
it. Either measure the thing the caveat is about, or let it constrain the claim.

### What else landed

- **RMSNorm + SiLU migrated off the removed `nl.arange` API** onto `nl.ds` / NKI 0.5.0. Required
  computing the reduction in fp32, which *improved* accuracy ~50x on fp32 (max_diff 1e-4 → 1e-6)
  and made bf16 bit-identical, because that is what PyTorch's RMSNorm does anyway.
- **Qwen3-MoE: all three kernels transfer with zero code changes** (logits cos_sim 1.000002).
  Load-bearing evidence for the per-kernel-not-per-model thesis. But Qwen3-MoE **does not run on
  Neuron at all by default** — the experts path uses `torch.sort`/`histc` → unsupported `sort`
  HLO. Fix: `experts_implementation="batched_mm"`, documented nowhere (#22).
- **MoE recommendation reversed:** the best NKI target is the routing `sort`/`histc` step, not the
  expert matmul. It unblocks the default MoE path, the compiler error itself recommends NKI for
  it, and it is blocked by neither #17 nor #18.
- **Week 4, 5 and 6 deliverables written**, plus the MoE gap analysis.

### Three times I fooled myself, all caught by measurement

Worth reading as a set, because the pattern is the most transferable output of the project:

| # | Looked like | Actually was |
|---|---|---|
| 8 | "RMSNorm validated, bit-identical" | kernel never ran; fallback compared to itself |
| 19 | "NKI is 8-400x slower" | outputs discarded, XLA eliminated the computation — timed an empty graph |
| 21 | "NKI is incompatible with torch.compile" | my loader didn't register the module in `sys.modules` |
| **24** | **"per-layer NKI swapping is structurally launch-bound"** | **an uncached `neuron-ls` subprocess; 102x recoverable with one decorator** |
| **25** | **"the NKI kernels spill an fp32 intermediate to HBM"** | **artifact of dividing non-linear traffic by N. Marginal traffic is exactly 1.00x the floor — the kernels are optimal** |

On a lazy-execution backend, **both correctness and performance measurements fail silently by
default.** A fallback is numerically correct. An eliminated computation is fast. A harness bug
looks like a platform limitation. None of them error. The guards now in place (execution call
counters, a scaling gate, mandatory controls, negative controls) are in
`tests/nki_test_utils.py` and `scripts/benchmark_kernels.py`, and #19 was caught *by* the guard
built after #8.

**The fourth one is a different failure mode and it is the one I would most want to avoid
repeating.** The first three are harness bugs: the measurement was invalid, and a guard catches
them. In the fourth the measurements were all *valid* — ~52 ms/call really is the cost, reproduced
five times within 1% — and the conclusion drawn from them was wrong. No guard catches that, because
nothing is broken.

Two practices came out of it:

- **When a hypothesis has survived several tests and the story still doesn't close, change
  instrument rather than adding another variant.** Repeated survival is evidence about the
  instrument as much as about the hypothesis.
- **Measure the two ends against each other early.** Device time vs wall time is one number each.
  Their ratio was 2400x and it invalidated an entire class of explanation at once. It should have
  been the first thing measured, not the fifth.

There is also a smaller trap worth naming. When I believed the cost was inside the NEFF, I wrote out
three candidate explanations ranked by plausibility. All three were device-side, because the framing
had already concluded the cost was in the execution. The true answer wasn't ranked low — it was
absent. **Enumerating candidates inside a single framing feels like rigour and isn't.**

And #25 above is a fifth instance, caught before it shipped, with yet another mechanism: **a correct
calculation applied to a quantity that isn't linear.** Dividing HBM traffic by call count said the
kernels move 3.00x more data than necessary, which reads as a spilled fp32 intermediate — and the
`nl.arange` migration had introduced exactly such a temporary, so there was a ready culprit. Landing on
exactly 3.00x for both ops independently is the tell; a real inefficiency doesn't hit a round number
twice. Traffic isn't linear in N because a small NEFF carries fixed setup traffic, and solving
`traffic(N) = FIXED + N x MARGINAL` gives marginal = 1.00x the floor. **Vary N before dividing by N.**
Same class of error as #19, same fix: vary the independent variable and look at the shape.

### What needs you

See **BLOCKED — NEEDS INPUT** at the bottom. Short version, and it changed:

1. **Is the per-call `create_computation` rebuild cacheable?** 91.6% of the remaining regression, and
   closing it takes 3.4x slower to ~1.18x. The largest available improvement and the top technical ask
   (B12). Not attempted — it sits inside `torch_xla`'s op-registry path and a wrong guess there could be
   silently incorrect rather than an error.
2. **Who owns `nki/compiler/target.py`, and do I write the CR?** One decorator, 102x per call,
   reproducer ready. Highest value-to-effort item, and correct regardless of everything else (B10).
3. **Can a NKI custom call participate in compiler fusion?** Decides whether the last ~18% is
   recoverable — i.e. whether this becomes a win or merely approaches parity (B14).
4. **Sanity-check the recommendation before it goes out.** The headline has been revised four times,
   each time by measuring something the previous round assumed. Current version: one framework bug is
   ~92% of the regression, a fusion cost is ~8%, and with both addressed the kernels land near parity (B9).
5. **Am I still in scope?** I am now two layers below the Kernel Hub — NKI's dispatch path, and the
   compiler's fusion behaviour (B13).

Both draft messages are rewritten. The previous Samir draft would have told the HF team their
per-layer granularity might be structurally wrong for Neuron, on the strength of a number whose
cause turned out to be ours — worth not sending.

Two earlier asks are gone: getting a stack where `torch.compile` works (withdrawn, it wasn't
blocking anything) and whether ~53 ms is expected on this SDK (answered — it's a bug).

### What got done

Three NKI kernels now execute inside a real Qwen3 forward on Trainium through the HF
Kernel Hub mechanism, with execution *proven* rather than assumed:

| Kernel | Type | Accuracy | Source |
|--------|------|----------|--------|
| RMSNorm | layer (`RMSNorm`, 115 registrations) | 11/11, NKI verified | tutorial (re-validated) |
| RoPE | function (`rotary_pos_emb`, 95 model files) | 20/20 + 6/6 guards | **nki-library `rope_hf`, ported** |
| SiLU | layer (`SiLU`, covers all `ACT2FN["silu"]`) | 9/9, NKI verified | `nl.silu` native |

E2E on Qwen3: RMSNorm 9×, RoPE 2×, SiLU 2× per forward, all `fallback=0`, logits
`cos_sim 1.000001`. All four suites exit 0.

### The two things you most need to know

**1. Week 2's accuracy results were not measuring NKI.** The kernel never ran. It gates
on `device.type != "cpu"` and the tests fed CPU tensors, so every case took the PyTorch
fallback and compared it against a mathematically identical reference — reporting a
perfect `max_diff = 0.00e+00`. The tell was that perfection: a reduction kernel on
hardware *must* differ by ~1e-4. The kernel turned out to be correct, but it was
untested. Fixed the harness (`tests/nki_test_utils.py`) so every case asserts the NKI
branch executed. Finding #8, and the most valuable customer-facing result of the week:
on Neuron the dangerous outcome isn't a crash, it's a no-op that looks like success.

**2. `use_kernels=True` cannot reach the `"neuron"` device path.** This Week 3 goal
cannot be met today. It fails as a *silent no-op*: transformers passes a `Device` object,
`kernelize` only validates device *strings*, so `Device(type="xla")` sails through and
matches nothing. I initially proposed the wrong fix (patching `kernels._find_device` —
never called on that path); the e2e test caught it. The correct fix is one branch in
`transformers/integrations/hub_kernels.py`, and I **verified it is sufficient**: applied
in-process it takes Qwen3 from 0 → 9 swapped layers. Finding #9.

**3. `nkilib` is already installed, and thin-wrapper porting works today.** Found late, and
it's the most consequential finding for *planning*. Every production kernel is importable
from the venv, and calling installed `rope_hf` directly from PyTorch/XLA gives
`cos_sim 1.000001`. Hand-porting doesn't scale — RoPE needed ~15 lines inlined, the MLP
kernel's closure is 7,249 lines across 22 files. Thin wrappers are a few dozen lines. The
blocker is the `python-depends` allowlist, i.e. policy, not code. So the ask to HF shrinks
from "fund a porting program" to four items, three of which are each smaller than one kernel
port. Finding #16.

### Other findings worth your attention

- **#12** HF already whitelists `nki` as a Neuron `python-depends` — but `_backend()`
  reports cuda on the DLAMI, so the entry is unreachable and kernels must under-declare
  their own dependency. Same `hasattr(torch, "neuron")` root cause as #7, now breaking
  two things. Fixing `_backend()` is the highest-leverage single change.
- **#14** `nki` and `neuronxcc.nki` have different capabilities and **neither is a
  superset**. RMSNorm/SiLU need one, RoPE needs the other; this repo requires both.
  `hasattr` lies about `nl.arange`, so you find out at compile time, per kernel.
- **#13** nki-library *does* have a standalone HF-shaped RoPE (`rope_hf`), undocumented
  in the public API reference. This **reverses the Week 2 narrative**: "nki-library is
  too fused to port" is per-kernel, not a general truth.
- **#15** Several `_KERNEL_MAPPING` entries are registered by no model and are
  unreachable — including `SwiGLUMLP`, which matters because fused MLP is where the real
  MLP performance is. It needs the separate fusion API, not a per-layer swap.
- **#17** The fused MLP is blocked by **weight layout**, not by the kernel. All three HF
  `nn.Linear` weights are transposed vs what `nkilib.mlp()` wants, the transpose must be
  materialized, and `kernelize()` has no parameter hook — only `forward` rewriting. All four
  workarounds are bad. This blocks *every* fused-kernel port on Neuron, so it needs a
  kernels-team design decision before Week 5 planning.

### What's left / needs your input

See **BLOCKED — NEEDS INPUT** at the bottom. Short version: the Hub repo-home decision
(Samir) blocks publishing; three small upstream fixes are owned by other teams; and MFU
measurement is Week 4. I did not touch the steering doc's week-by-week status — worth
updating Week 3 to reflect that the `use_kernels=True` goal is blocked upstream rather
than done.

One correction to the steering doc's assumptions: coverage is larger than estimated —
115 RMSNorm registrations (est. 87) and 95 RoPE model files (est. 66).

---

Branch: `week-3`. Session start: 2026-07-29 17:04 UTC.

Goal per steering doc Week 3: package RMSNorm for Hub, add RoPE, register `"neuron"`
entries in `_KERNEL_MAPPING`, confirm `use_kernels=True` alone triggers the swaps.
Stretch: start SiLU.

---

## PLAN

Ordered by dependency. Each task ends with a verification step run on trn2 over SSH.

| # | Task | Depends on | Verification |
|---|------|-----------|--------------|
| T0 | Env setup on trn2: clone, install deps, re-run Week 2 tests to confirm green baseline | — | `test_rmsnorm_accuracy.py` + `test_qwen3_layer.py` pass on trn2 with NKI backend |
| T1 | Investigate function-replacement API in `kernels` 0.15.2. Does `FuncRepository` / `LocalFuncRepository` exist? If not, what is the actual mechanism for replacing a free function like `apply_rotary_pos_emb`? | T0 | Read installed source on trn2; write finding to `docs/poc-findings.md` |
| T2 | Investigate transformers `_KERNEL_MAPPING` structure + the `use_kernels=True` code path. Determine exactly where a `"neuron"` entry must go and what layer names are registered | T0 | Locate registration site in installed transformers; document |
| T3 | Investigate RoPE in `aws-neuron/nki-library`: does a standalone rotary kernel exist? Log porting friction | — (parallel) | `docs/nki-library-porting-analysis.md` updated with RoPE case study |
| T4 | Write NKI RoPE kernel, single-file pattern, `kernels/neuron_rope/` | T1, T3 | Compiles + runs on trn2 |
| T5 | RoPE accuracy test vs `apply_rotary_pos_emb` | T4 | cosine sim > 0.999 across Qwen3 shapes, NKI backend confirmed |
| T6 | Register `"neuron"` entries in `_KERNEL_MAPPING` for RMSNorm (+ RoPE if T1 allows), confirm `use_kernels=True` alone fires the swap | T2, T5 | e2e script on trn2 shows swap without explicit `use_kernel_mapping` |
| T7 | Hub-style packaging: validate flat repo structure (no `build/`, no kernel-builder) is loadable as a Hub-format repo. No upload | T0 | Structure validation script passes on trn2 |
| T8 | SiLU NKI kernel (stretch) | T5 | cosine sim > 0.999 |
| T9 | Update tracking docs + `deliverables/week-3.md` | all | — |

Non-goals this session (out of scope / guardrailed): Hub upload, pushing to remote,
contacting Samir re: repo home, MFU measurement (Week 4).

---

## LOG

### 2026-07-29 17:04 UTC — Session start
Read steering doc in full. Surveyed repo: Week 2 RMSNorm kernel is single-file
(`kernels/neuron_rmsnorm/__init__.py`), tests exist for isolated layer + 2-layer Qwen3
model. Read all tracking docs to understand established findings (9 sticking points,
6 findings, RMSNorm porting analysis).

Created branch `week-3` off `main` @ `dc126ed`.

trn2 reachable (`ip-172-31-39-187`, 1 neuron device present). Venv at expected path
`/opt/aws_neuronx_venv_pytorch_2_9`. Repo was NOT cloned on the fresh instance —
cloned it. Dependency install started in background.

### 2026-07-29 17:10 UTC — T0 env setup done, baseline "green" (but see 17:25)
trn2 deps installed: `kernels 0.15.2`, `transformers 5.15.0.dev0`, `torch 2.9.1+cu128`.
Ran `tests/test_rmsnorm_accuracy.py` on trn2 — 8/8 shapes pass, cos_sim 1.000000.
Added `scripts/sync_to_trn2.sh` (rsync) so I can iterate without committing/pushing.

### 2026-07-29 17:15 UTC — T1/T2 investigation: function replacement + _KERNEL_MAPPING
Read the installed source rather than the docs (docs lag the API — established Week 1).

**Function replacement exists and is usable.** `kernels` 0.15.2 exports
`FuncRepository`, `LocalFuncRepository`, `LockedFuncRepository`, `use_kernel_func_from_hub`.
Mechanics, from `kernels/layer/func.py`:
- `LocalFuncRepository(repo_path, *, func_name)` → `load()` → `get_local_kernel(repo_path)`
  then `getattr(kernel, func_name)`.
- **Important asymmetry:** layer repos look up `kernel.layers.<name>`; func repos look up
  `<name>` at the **module top level**. So a function kernel must NOT be inside the
  `layers` namespace. Easy to get wrong given the layer pattern.
- Flags are read off the *function object*: `has_backward` defaults to **True** for funcs
  (it defaults to False for layers). Must set explicitly.

**transformers side.** `_KERNEL_MAPPING` is built lazily in
`transformers/integrations/hub_kernels.py::_build_kernel_mapping()`, then merged with
`_FUNCTION_KERNEL_MAPPING`. `"rotary_pos_emb"` already exists there with `cuda`, `rocm`,
and `xpu` entries — but **no `"neuron"`**. That gap is a one-block addition.

**Qwen3 already opts in** to both interception points (verified, not assumed):
`Qwen3RMSNorm.kernel_layer_name == 'RMSNorm'`; `apply_rotary_pos_emb` is a `Func`
nn.Module instance with `kernel_layer_name == 'rotary_pos_emb'` and forward signature
`(self, q, k, cos, sin, unsqueeze_dim=1)`.

**Coverage is better than the steering doc estimated:** 110 model files use the RMSNorm
layer kernel (est. 87), 95 use the rotary_pos_emb func kernel (est. 66).

### 2026-07-29 17:20 UTC — T3 done (delegated): nki-library HAS a HF-shaped RoPE kernel
Delegated the nki-library survey to a sub-agent. Headline: unlike RMSNorm, **RoPE does
not need to be invented** — `nkilib/core/embeddings/rope_hf.py` exists and is written for
HuggingFace tensor layout.

- `rope_hf(q, k, q_out, k_out, cos=None, sin=None, rope_cache=None, backward=False)`
- `[batch, heads, seq, head_dim]`, separate `q_heads`/`k_heads` (GQA), returns a **tuple**
- accepts **precomputed** cos/sin (no internal theta) — matches HF exactly
- `rotate_half` convention, same as Qwen3/Llama
- has a backward path
- only 3 internal imports (`kernel_assert`, `div_ceil`, `get_verified_program_sharding_info`),
  all trivially inlinable — vs RMSNorm's much heavier dependency web
- also documents the key constraint: **`seq_len % (128 * LNC) == 0`** and 4D-only
- and a doc bug: `rope_hf` is absent from the public API reference, and the docs cite a
  non-existent import path `nkilib.core.rope.RoPE` (real: `nkilib.core.embeddings.rope`)

Also confirmed: **NKI has no concatenation primitive.** `rotate_half`'s
`torch.cat((-x2, x1), -1)` is expressed by pre-allocating the full-width destination and
writing into disjoint slices, folding the negation into `op=nl.subtract`.

### 2026-07-29 17:25 UTC — STOP. Week 2's accuracy results were not measuring NKI.
While probing the device path I noticed the baseline reported `max_diff = 0.00e+00` on
every shape. For a hardware kernel that is *wrong* — real NKI reductions differ from
PyTorch by ~1e-4. Exact zero means both sides ran the same code.

Wrote `scripts/probe_nki_execution.py` to instrument the branch. Confirmed on trn2:

| Probe | Result |
|---|---|
| CPU tensors — which branch ran? | NKI = **0**, fallback = **1** |
| `@nki.jit` with CPU tensors | `RuntimeError: Expected all tensors ... to be XLA tensors` |
| `@nki.jit` with XLA tensors (`xla_device_hw` = `NEURON`) | runs; cos_sim 1.000000, max_diff **1.731e-04** |
| full `NeuronRMSNorm` layer on XLA | NKI = **1**, fallback = 0; cos_sim 1.000000, max_diff 1.297e-04 |

So: **the kernel is correct, but it had never been executed.** `NeuronRMSNorm.forward`
guards on `device.type != "cpu"`, and the tests fed CPU tensors, so every Week 2 test
silently took the PyTorch fallback and compared it against a mathematically identical
reference. The test even printed "Backend: NKI kernel (NeuronCores)" — it was reporting
`_HAS_NKI` (importable), not "did it run".

Logged as Finding #8 (Critical). This is the most valuable customer-facing finding so far:
on Neuron the dangerous outcome isn't a crash, it's a no-op that looks like success.

### 2026-07-29 17:30 UTC — T2 verified: `use_kernels=True` cannot reach "neuron"
`scripts/probe_neuron_device_path.py` on trn2. Two independent blockers:
- transformers `kernelize(model, mode)` has **no `device` param**; device comes solely from
  `model.device.type`. (The underlying `kernels.kernelize` *does* accept `device`.)
- Neuron reports `"cpu"` (params on host → neuron mapping silently ignored) or `"xla"`
  once moved to device → **hard error** `Unsupported device type 'xla'`. Nothing maps
  `"xla"` → `"neuron"`. `hasattr(torch, "neuron")` is still False even after
  `import torch_neuronx`.

Minimal upstream fix identified and, importantly, *feasible*: `xm.xla_device_hw()` returns
exactly `"NEURON"` on trn2, so `kernels._find_device` can map xla→neuron in ~3 lines.
Logged as Finding #9 (High).

---

## DECISIONS

**D1. Re-validate RMSNorm on XLA before writing RoPE, and fix the test harness first.**
Finding #8 means the Week 2 result is unproven. Building RoPE on the same harness would
reproduce the same false confidence. So I inserted a task (T0b) ahead of RoPE: a shared
`tests/nki_test_utils.py` that places tensors on the XLA device and *asserts via a call
counter that the NKI branch executed*. Every kernel test from here uses it. Cost: ~1h.
Worth it — otherwise every accuracy number in the PoC is suspect.

**D2. Treat exact-zero numerical diff as a test FAILURE for hardware kernels.**
Assert `0 < max_diff < tol` rather than `max_diff < tol`. A perfect match means the kernel
didn't run. This is the single cheapest guard against Finding #8 recurring.

**D3. Port RoPE from `nki-library`'s `rope_hf`, not from a tutorial.**
Opposite of the RMSNorm decision, and worth calling out because it reverses the Week 2
narrative. For RMSNorm the production kernel was unusable (fused with FP8 quant, no
standalone path) so we used a tutorial. For RoPE there *is* no tutorial anywhere in
nki-samples, and the production kernel is already HF-shaped. So RoPE is the honest
"port the production kernel" case study the PoC needs.

**D4. Do not attempt to patch installed transformers/kernels to force the neuron path.**
Tempting (it would make `use_kernels=True` "work"), but it mutates a shared venv, isn't
reproducible for a customer, and would misrepresent the state of the integration. Instead:
document the gap precisely, write the proposed patch as a diff in the docs, and validate
via the `kernels`-library entry point with an explicit `device="neuron"`. The gap *is* the
finding — hiding it would destroy the PoC's value.

**D5. Scope guard — no Hub upload, no remote pushes, no contacting Samir.**
All are outside this session's guardrails. The Hub question (T7) is answered by validating
repo *structure* locally, not by uploading. Samir coordination is logged for the human.

### 2026-07-29 17:35 UTC — T0b done: harness fixed, RMSNorm re-validated for real
Added `tests/nki_test_utils.py`: `require_neuron()` (refuses to report results off
Neuron hardware, checks `xla_device_hw() == "NEURON"`), `nki_call_counter()` (patches the
kernel module's dispatch targets and asserts nki>0 / fallback==0), dtype-aware
`max_abs_diff` tolerance with cos_sim as the primary gate.

`tests/test_rmsnorm_nki.py`: **11/11 pass, every case nki=1 fallback=0.** fp32 max_diff
1.2e-05 → 3.9e-04, bf16 3.1e-02 (expected at 8 mantissa bits; cos_sim still 0.999993).
Non-zero diffs are the corroborating evidence of real hardware execution.

bf16 first failed a 1e-2 tolerance I'd calibrated for fp32. Added `tol_for_dtype()`
rather than loosening globally — bf16 genuinely cannot hold 1e-2 absolute at these
magnitudes.

Kept `test_rmsnorm_accuracy.py` with a warning header instead of deleting it (guardrail:
don't delete files I didn't create — and it's a useful reproduction of the failure mode).

### 2026-07-29 17:45 UTC — T4/T5 done: NKI RoPE ported and validated
`kernels/neuron_rope/` — ported from nki-library `rope_hf`. **20/20 cases pass, all
nki=1 fallback=0**: GQA/MQA, batch 1-4, seq 128-1024, head_dim 64/128, fp32+bf16,
2D and 3D cos/sin.

Every case is bit-identical (`max_diff = 0.000e+00`). Unlike RMSNorm that is *correct*
here: RoPE is elementwise, so both backends perform the same three IEEE ops in the same
order. But "bit-identical everywhere" also looks exactly like a test that measures
nothing, so I added negative controls to prove the comparison discriminates:
- vs negated-sin reference: cos_sim **0.430** (correctly rejects)
- vs unrotated input: cos_sim **0.500** (proves the kernel actually rotated)
- vs correct reference: cos_sim **1.000000**

Also verified the `seq_len % 128` fallback fires, warns loudly, and stays correct.

### 2026-07-29 18:00 UTC — T6 done: e2e Qwen3, and the upstream fix VERIFIED
`tests/test_qwen3_neuron_e2e.py` on a real 2-layer Qwen3 at seq_len 128:
- **NKI RMSNorm executed 9×** (4/layer + final norm), fallback 0
- **NKI RoPE executed 2×** (1/layer), fallback 0
- logits cos_sim **1.000001**, max_diff 5.29e-05 vs the unkernelized on-device model

Two things I got wrong and the tests caught — both now corrected in Finding #9:

1. **The proposed upstream fix was in the wrong place.** I had recommended patching
   `kernels._find_device`. transformers' wrapper computes
   `Device(type=model.device.type)` itself and passes it explicitly, so `_find_device`
   is never called on the `use_kernels=True` path. Patching it would have done nothing.
   The fix must go in `transformers/integrations/hub_kernels.py`.
2. **The failure mode is a silent no-op, not an error.** transformers passes a `Device`
   *object*, and `kernelize` only validates device types given as *strings*. So
   `Device(type="xla")` sails through unvalidated, matches nothing, and every layer
   quietly keeps its original forward. `kernelize()` returns success.

With the corrected one-branch fix applied in-process, `use_kernels=True` goes from
**0 → 9 swapped layers**, logits cos_sim 1.000001. The recommendation is now
demonstrated rather than hypothesised.

Instrumentation gotcha worth remembering: you must patch the module object the
*repository* loaded (`get_local_kernel()`, which caches), not a fresh
`load_kernel_module()` copy. Patching the wrong one reports nki=0 while the kernel is
in fact running — a false negative of exactly the shape of Finding #8.

### 2026-07-29 18:10 UTC — T7 done: Hub packaging, and a compounding bug
`kernels/python_depends.json` already whitelists `nki` under a `neuron` backend section
— HF anticipated NKI kernels. But `validate_dependencies()` consults the table for
whatever `_backend()` reports, and that is `CUDA(version=12.8)` here, so declaring
`python-depends: ["nki"]` raises `unsupported kernel dependency: nki`. A Neuron kernel
must therefore under-declare (`[]`) to load at all. Finding #12.

Same `hasattr(torch, "neuron")` root cause as Finding #7, now breaking two things.
Fixing `_backend()` is the highest-leverage single change; it does *not* fix device
routing (that's Finding #9's separate transformers change). Two distinct fixes.

Also measured metadata.json field requirements: `digest` is optional (dropping the
sha256 boilerplate), everything else required. Minimum viable kernel repo is 2 files.

### 2026-07-29 18:05 UTC — T8 done: SiLU kernel (stretch goal)
`nl.silu` exists natively, so nothing to port. 9/9 pass, all nki=1 fallback=0, including
Qwen3-8B's 12288 MLP width and bf16. Negative controls: vs GELU 0.991, vs raw input 0.837.

Hit Finding #14 writing this: SiLU failed all 9 shapes under the top-level `nki` package
with `failed to resolve name 'nki.language.arange'`. Then I tried standardising all three
kernels onto `neuronxcc.nki` and broke all 20 RoPE cases with
`NotImplementedError: math.trunc() is not supported for scalar`. Reverted. The two
packages are genuinely disjoint in capability; each kernel is pinned to the one its idiom
needs.

Also corrected a claim I'd written into the SiLU docstring before verifying it: I said
`"SwiGLUMLP"` was the fusion interception point. It's in `_KERNEL_MAPPING` but **no model
registers it** — `Qwen3MLP` has no decorator at all. Fused MLP needs the separate
`register_kernel_replacements_and_fusions` API. Caught by grepping before shipping the
claim; noted because it's the second time this session that writing the verification
changed the conclusion.

### 2026-07-29 18:15 UTC — T9: docs, deliverable, hygiene
- `deliverables/week-3.md` — full writeup.
- Filled the two empty sections of `docs/customer-experience.md` (API friction, runtime
  issues), added a "Silent Failure Modes" section, and reordered the gap table by
  blocking impact with owner + effort.
- `docs/porting-recommendations.md` — corrected the stale `python-depends` claim, wrote
  up the real RoPE porting friction.
- Fixed the **Makefile**, which was broken: `lint` referenced
  `kernels/neuron_rmsnorm/layers.py` (deleted in Week 2) and `test` assumed pytest-style
  tests that ours aren't. Added `test-nki` / `test-e2e` / `probe` / `registration` /
  `sync` targets.
- Closed a gap in my own work: **RMSNorm was still falling back silently** while RoPE and
  SiLU warned — the exact behaviour that caused Finding #8, in the kernel that caused it.
  Added `_warn_once` + `_nki_unsupported_reason` to match the other two.
- Expanded RoPE guard coverage to 6/6 (seq_len%128, unsqueeze_dim, 3D input, odd
  head_dim, cos width mismatch, dtype mismatch).

Final verification: all four suites exit 0 on trn2.

---

## DECISIONS (continued)

**D6. Pin each kernel to the NKI package its idiom requires, rather than rewriting.**
On hitting Finding #14 I could have rewritten RoPE's `div_ceil` to avoid `//` and
standardised on `neuronxcc.nki`. I chose not to: RoPE is a *port*, and its value as a
case study depends on staying close to the nki-library source. Rewriting idioms to
satisfy an undocumented package split would have hidden the finding. Documented the split
instead and pinned per-kernel.

**D7. `has_backward = False` on all three kernels.**
nki-library's `rope_hf` has a real backward rotation, and `nl.silu_dx` exists, so backward
kernels are feasible. But we have not wired autograd, so claiming `has_backward=True`
would let `kernelize` select these in TRAINING mode and produce wrong gradients. False is
the honest value. Note the default for *function* kernels is True, so this had to be set
explicitly — an easy footgun.

**D8. Treat a bit-identical result as op-dependent, not universally suspicious.**
Refined D2. For reduction ops (RMSNorm) a zero diff means the kernel didn't run. For
elementwise ops (RoPE, SiLU) it's the correct expectation, since both backends perform
the same IEEE ops in the same order. The call counter — not the diff — is the
authoritative execution proof; `expect_bit_identical` only controls an advisory note.
Added negative controls so "bit-identical" is never accepted on trust.

**D9. Did not modify the steering doc.**
It's your source of truth and it records week-by-week status. Week 3's
`use_kernels=True` goal is blocked upstream rather than achieved, and coverage numbers
turned out higher than estimated. Flagged here rather than editing it myself.

---

## BLOCKED — NEEDS INPUT

**B1. Hub repo home: `kernels-community/` vs `aws-neuron/`.**
Blocks publishing and determines the `repo_id` in the upstream diff
(`scripts/neuron_kernel_registration.py::PROPOSED_UPSTREAM_DIFF`).
*Recommendation:* `aws-neuron/`, so Neuron owns versioning and can ship fixes without
waiting on kernels-community review. Requires the Samir conversation — external
communication, out of scope for this session.

**B2. Who drives the three upstream fixes?** All are small; none are ours to merge.
- Finding #9 → transformers, ~5 lines across 3 sites (`hub_kernels.kernelize`,
  `kernel_config.infer_device`, `kernels._find_device`). **Verified sufficient.**
- Findings #7/#12 → `torch_neuronx` setting a `torch.neuron` attribute. One line,
  unblocks variant resolution *and* dependency declaration.
- Finding #14 → NKI team decision on which import path is supported long-term.

*Recommendation:* file all three now with the reproductions from this branch. Finding #9
has a verified patch and a demo, so it's the strongest one to lead with.

**B3. Is inference-only acceptable for the beta story?** All three kernels are
`has_backward=False`, so training mode falls back. If training matters for the beta,
backward kernels are a real work item (nki-library's `rope_hf` has the backward path;
`nl.silu_dx` exists; RMSNorm backward would need writing).

**B4. Publishing to the Hub was explicitly out of scope this session** (guardrail: no
external side effects). Everything needed is ready: flat layout validated, `digest`
confirmed optional, minimum repo is 2 files. Unblocked by B1.

---

## SUGGESTIONS (out of scope, logged not done)

- **`kernels`-side kernel report.** The single highest-value customer-experience
  improvement would be an API answering "which implementation is live on which layer".
  Numerical correctness cannot distinguish acceleration from fallback, so today there is
  no way for a user to verify. Worth proposing upstream.
- **A `--strict` mode for `kernelize`** that errors instead of falling back. `use_fallback=False`
  exists in the kernels library but transformers doesn't expose it.
- **RoPE `seq_len` padding.** Rather than falling back at `seq_len % 128 != 0`, pad to a
  multiple of 128 and slice. Would make the kernel usable at arbitrary sequence lengths,
  which is how HF actually calls it. Needs a perf check — padding may cost more than it saves.
- **Port `nl.gelu` / `gelu_apprx_tanh` activations.** Cheap now that the pattern is
  established; each covers a whole model family via one `activations.py` decoration.
  Same caveat as SiLU: elementwise activations are memory-bound, so don't expect wins.

### 2026-07-29 18:30 UTC — The strategy finding I nearly missed
Per the "don't stop early" instruction I went looking for remaining in-scope work and
noticed `docs/nki-library-porting-analysis.md` had never been updated with the RoPE case
study — the steering doc explicitly says to update it per kernel investigated. While
writing that up I delegated an MLP-kernel analysis, and its report flagged (as *inferred,
not verified*) that nkilib might ship bundled with neuronx-cc.

I verified it, and it's better than that: **`nkilib` is a normal install in the Neuron
venv**, with every production kernel importable — `rope_hf`, `mlp`, `moe_cte`,
`router_topk`, `rmsnorm_quant`.

Then I tested whether the installed production kernel is actually *callable* from
PyTorch/XLA:

| Strategy | Result |
|---|---|
| preallocated outs, read **return value** | **q cos_sim 1.000001, k cos_sim 1.000000** |
| preallocated outs, read **mutated args** | cos_sim **0.000000** |

So a **thin wrapper over the production kernel works today**. Destination-passing is
vestigial across the XLA boundary — outputs are shape templates, results come back via the
return value. (nki-library's own tests use `must_alias_input`, which points you at the
second strategy and silently gives zeros. Worth flagging upstream.)

**This is the most consequential finding of the session for planning**, and it reframes the
Week 2 conclusion. Hand-porting doesn't scale: RoPE needed ~15 lines inlined, the MLP
kernel's closure is 7,249 lines across 22 files (~480x). Thin wrappers are a few dozen
lines. So the ask shrinks from "fund a porting program" to four items, three of which are
each smaller than one kernel port. Blocker is the `python-depends` allowlist — policy, not
code. Finding #16.

Also logged **Finding #17**: the fused MLP is blocked by weight layout, not by the kernel.
All three HF `nn.Linear` weights are transposed vs what `nkilib.mlp()` wants, the transpose
must be materialized, and `kernelize()` has no parameter hook — it only rewrites `forward`.
All four workarounds are bad (in-place mutation silently breaks `save_pretrained`; a
transposed copy ~doubles MLP weight memory; lazy-cache adds a stall; per-forward transposing
erases the speedup). This blocks *every* fused-kernel port on Neuron, so it's a design
decision for the kernels team, not a PoC choice. Arguably more valuable output than the
kernel would have been.

Corrected the Week 2 over-generalization while I was there: "nki-library kernels are too
fused to port" is true of RMSNorm and false of RoPE and MLP (both have quant opt-in or
absent). RMSNorm is the outlier, not the archetype — a conclusion drawn from one kernel.

**D10. Did not build a thin-wrapper kernel, only proved it works.**
Tempting to ship a `neuron_rope_thin/` alongside the hand-port. Didn't: it would have to
under-declare an undeclarable dependency (Finding #12), so it isn't a shippable artifact,
and the PoC's value is the hand-port's documented friction. The experiment script records
the feasibility result, which is the part the team needs.

---

## Session 2 — 2026-07-29 19:50-20:15 UTC

Continued after the Week 3 wrap-up. Two pieces of work, both of which produced findings that
change the plan, and both of which required me to correct something I'd previously asserted.

### T10 — MLP derisking spike (`scripts/spike_nkilib_mlp.py`)

Ran the 1-2 day spike I'd recommended for Week 4. It paid for itself immediately.

**Positive:** the production fused MLP kernel **is** drivable directly from PyTorch/XLA with HF
weights. No vendoring. cos_sim vs Qwen3MLP: 0.999989 (fp32), 0.999995 (bf16), 0.999979
(H=4096 I=4096). Supports the Finding #16 thin-wrapper thesis.

**Blocking — Finding #18:** it cannot run single-core when `intermediate_size > 4096`. Sharp
boundary, 10 configs across three `hidden_size` values, **passes iff I <= 4096** (4096 passes,
4224 fails). Not fixed by seq len, `force_cte_mode`, or `mode=PREFILL`. Fails inside the
kernel's own tile arithmetic (`'floordiv' does not allow division by zero`), almost certainly
the CTE sharding heuristic forcing `shard_on_inter` above I=4096 with no SPMD grid.

Every real model is excluded: Qwen3-8B I=12288, Llama-3-8B and Mistral-7B I=14336. Unlike #17,
**a wrapper cannot work around this.** It also turns Week 2's general "SPMD assumptions may not
fit the per-layer swap model" worry into a measured fact: SPMD-strippability is per-kernel, and
for fused kernels it may not hold at all.

**Correction to Finding #17.** I had asserted non-contiguous transposed tensors are rejected and
the transpose must be materialized. Reasoned, not measured — and the measurement contradicts the
premise. The kernel accepts a device-side `.t()` and is numerically correct, and
`is_contiguous()` returns True on XLA even after `.t()` (XLA normalizes layout, so `.t()` is a
real graph op, not a stride view). The memory cost and the missing hook stand; whether
per-forward transposing is expensive depends on whether XLA hoists it, which I have not
profiled. Withdrew the two claims that don't hold; marked the cost table as candidate.

My *first* version of the spike also got its own methodology wrong — transposed on host then
called `.to(device)`, letting the transfer materialize the result, which proved nothing about
non-contiguous handling. Fixed to transpose on device.

### T11 — Per-kernel benchmark (`scripts/benchmark_kernels.py`)

Set out to test the prediction that RMSNorm/RoPE help and SiLU doesn't. **Could not answer it**,
and that is the result.

v1 reported every kernel 8-400x slower than eager. Those numbers were garbage. The tell: latency
didn't vary with tensor size — RMSNorm 0.55 ms at both S=128 and S=2048 (16x the data), eager
0.07 ms throughout. I never consumed the outputs, so XLA eliminated the computation and I timed
an empty graph.

After fixing that, latency does respond to size but only 1.1-1.3x for 8x data → **90%+ fixed
cost**. So even corrected, per-layer microbenchmarking cannot resolve kernel quality here. The
script now **refuses to report ratios** unless latency scales with problem size, and prints why.
On repeat runs the gate passes (1.26x) or fails (1.12x) at identical shapes — itself evidence
the measurement is at the noise floor.

**The signal that did survive — Finding #19:** host-side dispatch is **~0.36 ms/call for NKI vs
~0.011 ms for eager**, reproducible across runs (~25-33x). At 217 kernel calls per Qwen3-8B
forward that's **~76 ms of host-side overhead per step**, serial and independent of batch/seqlen.
Upper bound on serial cost since some may overlap.

This makes fusion *more* important, not less — one fused call replaces several dispatches — so
#17 and #18 go up in priority on this evidence.

**Finding #19 is Finding #8 in a different costume**, and I've written it up as a pattern rather
than an isolated slip: on a lazy-execution backend both correctness and performance measurements
fail silently by default (a fallback is numerically correct; an eliminated computation is fast).
Every measurement needs an independent check that it exercised the thing measured — call counter
for correctness, scaling gate for performance.

---

## DECISIONS (continued)

**D11. Suppress the benchmark ratios rather than report them with caveats.**
I could have published "NKI is 8-400x slower" with a footnote. That would have been actively
harmful — someone would have quoted the headline. The script now returns exit 2 and explains the
suppression. The dispatch-overhead number is reported because it's reproducible and
methodologically clean.

**D12. Do not attempt to fix Finding #18 from our side.**
Tempting to hunt for a config that dodges the divide-by-zero, or to launch SPMD from the wrapper.
Didn't: (a) the boundary is in nki-library's tile math and belongs to that team, (b) an SPMD
launch from a per-layer HF swap is an architecture question for the kernels team, not something
to improvise, and (c) a workaround would obscure a bug that should be reported. Logged as
upstream Fix 5 with the reproduction.

**D13. Promoted Fix 5 above the Finding #17 question in filing order.**
#17 is a design question about how to transform weights for a kernel that currently cannot
compile at any useful size. Sequencing the design conversation first would waste the kernels
team's time.

---

## BLOCKED — NEEDS INPUT (updated)

Additions to B1-B4 from session 1:

**B5. Fused-kernel work is blocked pending two upstream items.** Do not start the fusion-API
integration. Finding #18 (nki-library, no workaround) then Finding #17 (HF design decision).
The standalone spike is done, so there is no further derisking to do from our side.
*Recommendation:* file #18 now with the boundary table; it's a clean bug report.

**B6. Week 4 MFU needs a methodology decision.** Per-layer microbenchmarking is dead (#19), so
MFU on a full model is the only instrument left — but Finding #9 means `use_kernels=True` won't
route to Neuron, so the measurement has to go through `kernelize_for_neuron()`. Worth deciding
whether that's an acceptable basis for a customer-facing number, or whether the upstream fix
should land first. *Recommendation:* measure now via the helper, and state the caveat.
Also report launch count alongside MFU, per #19.

---

## Session 3 — 2026-07-31 06:45-08:30 UTC — Weeks 4, 5, 6

Instruction was to keep going as far as possible. Worked through the remaining project in
priority order: tech debt, then MFU, then MoE, then the PoC document.

### T12 — Migrated RMSNorm + SiLU to NKI 0.5.0 (`nl.ds`)

Cleared the Finding #14 tech debt. Both kernels used `nl.arange` + `mask=`, removed in 0.5.0,
which pinned them to the older bundled API. Validated the replacement pattern first
(`scripts/probe_nki05_api.py`) across ragged tails of 0, 44, 122, and a 1-row case with zero
full tiles.

Four 0.5.0 API differences and two structural constraints found, all documented in the probe:
`nl.arange`/`nl.mgrid` removed; `nl.load`/`nl.store` have no `mask`; `tile.broadcast_to(...)`
method doesn't resolve (use `nl.broadcast_to`); `tile / python_int` is rejected. And: kernels
must be module-level with module-global `nl` imports (a closure gives
`failed to resolve name 'nl.ndarray'`, *identical text* to a genuinely missing API), and no
inner function definitions.

**The bf16 case initially failed** with `nisa.tensor_scalar_arith operand0 must be float32` —
the `[rows,1]` reciprocal can't be bf16. Fixing it by computing the reduction in fp32 also made
the kernel match PyTorch's RMSNorm, which upcasts for the variance. So a *required* fix turned
out to be a correctness improvement: fp32 max_diff 1.2e-05…3.9e-04 → 4.8e-07…8.1e-06 (~50x),
and bf16 became bit-identical.

### T13 — MFU measurement (Week 4)

Baseline 41.95 ms/step, MFU 5.06%. Kernelized 8753.65 ms/step, MFU 0.02%. Denominator stated
explicitly (632 TFLOPS/device TensorEngine ÷ 2 for LNC2 = 316) and FLOP count printed so it's
auditable.

Did **not** stop at "208x slower". Added `--only` for per-kernel attribution: SiLU alone gave
51.9 ms/call, all three gave 51.6 ms/call — uniform, which pointed at a fixed charge. Then swept
problem size: **52.7-54.6 ms across a 112x range in rows.** Completely flat. One call on 28x the
data costs 1.02x one call on 1x. Then ruled out the alternatives one at a time.

Two operational notes: full-model compiles exceed the SSH command timeout, so added
`scripts/run_detached.sh`. And a run I killed left a **stale compile-cache lock** that made the
next run wait forever on a compiler that no longer existed — verified no live compiler and no
completed NEFF before removing only that lock file.

### T14 — Qwen3-MoE (Week 5)

Dense kernels transfer with zero changes. But first had to discover that Qwen3-MoE doesn't run on
Neuron at all by default (`sort` HLO unsupported via `torch.histc`), and probe the four experts
implementations to find `batched_mm` works. Finding #22.

That reframed the MoE recommendation: the valuable NKI target is the routing histogram, not the
expert matmul.

### T15 — The decisive experiment, which failed honestly (Finding #21)

Wrote `scripts/experiment_torch_compile_nki.py` to answer whether graph mode amortizes the
per-invocation cost. **v1 produced a convincing false finding** —
`ModuleNotFoundError: No module named 'neuron_silu'` under compile, which reads as "NKI is
incompatible with torch.compile". It was my own loader failing to register the module in
`sys.modules`, which Dynamo needs to re-import a traced function's defining module.

Fixed that, then added a mandatory plain-PyTorch control — and the control fails too, across
`openxla`/`inductor`/`eager` and both dtypes. So the question is unanswerable here. The script
now *skips* the NKI case when no control passes, rather than emitting a failure that would read
as a NKI result.

### T16 — Deliverables

`deliverables/week-4.md`, `week-5-moe-gap-analysis.md`, and **`poc-document.md`** (the final
one). Updated the steering doc's week-by-week status and definition of done, and rewrote both
draft messages (John, Samir) around the new findings — the John one had been asking him to decide
Week 4/5 scope, which had since resolved by getting done.

---

## SESSION 4 — Root-causing the slowdown properly

Prompted by a fair push: I had reported the graph-mode question as blocked, and was asked why I
didn't just run it. Two things were wrong with that position, and both were mine.

### T17 — `torch.compile` is not broken here

`scripts/diagnose_torch_compile.py`. I had concluded "torch.compile doesn't work on this stack"
from one error message. In fact `torch` 2.9.1 / `torch_xla` 2.9.0 are a matched pair, `openxla` is
registered, and `add`/`mul`/`relu` all compile on XLA tensors. Only `silu` failed — and the full
traceback showed why: `torch_neuronx` replaces it with an XLA user computation whose dispatch
predicate is `if any(is_xla_tensor(it) or is_xla_device(it) ...)`. A `FakeTensor` on `xla:0`
satisfies that, gets routed into `_xla_user_computation`, and is rejected. No abstract impls, so
Dynamo can't trace it. The override list includes `Embedding`, `Softmax` and `CrossEntropyLoss`, so
this hits nearly every transformer. Filed as Finding #23. `torch_xla.compile()` works around it.

### T18 — The question didn't need `torch.compile` at all

`scripts/probe_neff_count.py`. torch-xla is *already* a lazy graph runtime, so I counted device
executions directly with its `ExecuteTime` metric instead of using compile as a proxy. Steady
state, `compiles = 0`, three samples within 0.1%:

| variant | wall | device executions | per call |
|---|---|---|---|
| 28 NKI calls, 1 `mark_step` | 1446.37 ms | **1** | 51.66 ms |
| 1 NKI call | 52.80 ms | 1 | 52.80 ms |
| 28 torch ops | 1.23 ms | 1 | 0.04 ms |
| 1 torch op | 0.25 ms | 1 | 0.25 ms |

The 28 calls already shared one graph and one execution (196-node graph) and still cost 28x.
`execs(pre-sync) = 0` closed the one hole — no `nki.jit` call secretly flushes the graph. The
control scales sublinearly, so the harness sees batching when batching works. Variant C turned out
stronger than designed: `F.silu` on Neuron is itself an XLA user computation, so **28 XLA custom
calls cost 1.23 ms and 28 NKI custom calls cost 1446 ms** — the problem was never that custom calls
don't fuse.

I then wrote this up concluding "the cost is inside the compiled NEFF." Also wrong.

### T19 — Changing instrument, twice, and finding it in 35 minutes

- `scripts/profile_nki_call_cost.py` + neuron-explorer: the NEFF containing all 28 calls executes
  in **0.609 ms** at 43% MBU and 95% active. `activate_instruction_count = 112` = 28x4 confirms all
  28 are in it. HBM traffic ~30 tiles in and out, so each call does round-trip through HBM with no
  inter-call fusion — but that costs 0.6 ms, not 1446.
- `scripts/probe_where_is_the_time.py`: **99.9% of wall time is spent before `mark_step`.** (First
  version of this misread torch-xla's metric accumulators as seconds; they're nanoseconds, which
  produced a nine-digit millisecond figure obvious enough to catch.)
- `scripts/probe_inside_one_call.py`: cProfile put 51 of 52 ms in `select.poll` under
  `subprocess.check_output` under `_detect_target` under `resolve_target` under `_compile_opts`.

Reading the source confirmed it and explained why the compile cache doesn't help: `CompileOptions`
is the cache *key*, so target resolution runs before any lookup. A cache hit pays in full.

### T20 — Verifying the fix, and re-measuring

`scripts/probe_target_override_fix.py`: two fixes, one process, baseline re-run last as a control,
accuracy asserted on every variant. 51.74 → 0.50 ms (env override) and 0.49 ms (`lru_cache`),
cos_sim 0.999938 unchanged across all four. The override is set to whatever `_detect_target()`
returns on the host rather than a hardcoded string, since a wrong target would compile for the wrong
hardware and could be silently wrong rather than an error.

Then `measure_mfu.py --fix-target-detection`: **8753.65 → 141.43 ms/step, MFU 0.02% → 1.50%.**
And because the residual looked fixed-per-call, I tested that too rather than asserting it — seq
2048 gives 2.06x slower instead of 3.36x, with per-call cost up only 1.16x for 2.59x the work.

All five suites re-verified after the change. Finding #24, and the docs updated throughout.

---

## DECISIONS (continued)

**D14. Root-cause the slowdown instead of reporting it.** "208x slower" is a number; "~53 ms
fixed per invocation, flat across 112x problem size, and here are the five alternatives ruled
out" is a finding someone can act on. The attribution work took longer than the measurement and
was worth more.

**D15. Refuse to answer the graph-mode question rather than answer it badly.** I could have
reported "NKI fails under torch.compile" from v1 of that experiment. It would have been wrong and
it would have been quoted. Added a mandatory control and made the script skip rather than guess.

**D16. Did not soften Finding #20 on the hope that graph mode rescues it.** As measured, on the
stack a customer would use today, eager per-layer NKI swapping is not performance-viable. Whether
a future path fixes it is a separate, clearly-labelled claim.

**D17. Wrote the PoC recommendation with a real negative branch.** It says invest — but it also
says that if graph mode doesn't amortize the cost, Neuron should *not* invest further in this
integration point. A PoC that can only conclude "yes" isn't worth running, and the negative
branch is cheap to test.

**D18. MFU on Qwen3-0.6B at full depth rather than Qwen3-8B at reduced depth.** A real model end
to end beats a width-proxy, and given ~53 ms/invocation a bigger model only makes the ratio
worse. Stated as a limitation rather than hidden.

---

## BLOCKED — NEEDS INPUT (final)

**~~B7. Get onto a stack where `torch.compile` works on Neuron.~~ WITHDRAWN.** This was the top ask
and it was wrong. `torch.compile` is not broken here — `add`/`mul`/`relu` compile fine on XLA; only
ops `torch_neuronx` overrides fail (#23). More importantly `torch.compile` was never the right
instrument: torch-xla is already a graph runtime, and 28 NKI calls already fuse into one HLO graph
and one device execution while still costing 28x. Nothing to get onto a different stack for.

**~~B8. Is ~53 ms per NKI invocation expected on SDK 2.31 / NKI 0.5.0?~~ ANSWERED — no, it's a bug.**
It is an uncached `neuron-ls` subprocess in `_detect_target()`, called on every invocation from
`_compile_opts()`, outside the compile cache because its result is part of the cache key. One
`lru_cache` is worth 102x per call. Finding #24. The instinct that it "looked like a
misconfiguration rather than a design point" was right and I should have pushed on it sooner.

**B9. Sanity-check the PoC recommendation before it goes to Hanbo/Karthick.** Still open, and now
more necessary: the recommendation *inverted*. It went from "yes but defer kernel work, and answer
the graph-mode question first" to "yes, fix two caching bugs first, then reassess." Worth a mentor's
read before it goes out.

**B10. Who owns `nki/compiler/target.py`, and do I write the CR?** Highest-value item in the
project, one decorator, reproducer in `scripts/probe_target_override_fix.py`. Not Kernel Hub
specific — any eager per-layer NKI use pays this today.

**B11. Is `NEURON_PLATFORM_TARGET_OVERRIDE` customer-supported or internal-only?** Decides whether
it can be documented as a workaround. Related and worth raising together: with no `neuron-ls` on
PATH, `_detect_target()` silently returns `"trn3"`, so it would compile for the wrong generation
rather than fail loudly.

**B12. Is the residual `create_computation` cost cacheable? TOP TECHNICAL ASK.** ~0.59 ms/call, rebuilt
on every invocation. Same class of bug as B10, 100x smaller. Briefly demoted when Finding #25's
microbenchmark suggested a 2.5–2.7x device deficit would survive closing it; the in-situ measurement put
it back — **dispatch is 91.6% of the remaining regression** and closing this takes 3.4x slower to ~1.18x.
Not attempted: it sits inside `torch_xla`'s op-registry path, it is a much larger change than one
decorator, and a wrong guess could produce silently incorrect results rather than an error.

**B14. Can a NKI custom call participate in compiler fusion, or be made transparent to the fusion
pass?** Finding #25: the compiler cannot fuse across a NKI custom call, so each swapped op round-trips
through HBM where the data previously stayed resident across a fused region. Our kernels move exactly
the theoretical minimum traffic for an unfused op, so this is not kernel quality.

Magnitude, and this is the part I initially got wrong: **2.5–2.7x on device in a chained microbenchmark,
but 8.4% of the regression in a real forward pass.** So it decides whether the last ~18% is recoverable
after B12 lands — whether these kernels become a win or merely approach parity — rather than whether the
integration is viable at all. Briefly filed as the top ask on the microbenchmark's strength; the in-situ
measurement demoted it below B12.

If the answer is yes, the last 18% is recoverable. If no, per-layer swapping of small memory-bound ops
tops out just below parity, and the only shape that wins outright is a kernel spanning a whole fused
region — which is what nkilib ships and what #17/#18 say the Kernel Hub cannot express. A question rather
than an experiment, and I can't answer it from here.

**B13. Scope check.** I am now a layer below the Kernel Hub, inside NKI's dispatch path. In scope
for this PoC, or hand off and return to kernels?

Still open from earlier sessions: B1 (Hub repo home — Samir), B2 (who drives the upstream fixes),
B3 (is inference-only acceptable for beta), B4 (Hub upload, gated on B1), B5 (fused-kernel work
blocked by #17/#18), B6 (superseded — MFU methodology resolved).

**Also, unchanged and still true: 30-odd commits exist only on this laptop.** Nothing has been
pushed, per the guardrail. That is a single point of failure for the whole project and worth
resolving deliberately rather than by accident.

---

## SUGGESTIONS (out of scope, logged not done)

- **A NKI kernel for the MoE routing `sort`/`histc`.** Best-scoped MoE work identified: unblocks
  the default Qwen3-MoE path on Neuron, compiler explicitly recommends NKI for it, blocked by
  neither #17 nor #18. Building blocks exist in `nkilib/core/topk` and `core/router_topk`.
- **`model.get_kernel_report()` in the `kernels` library.** The single highest-value
  customer-experience improvement — today a user cannot distinguish acceleration from a silent
  no-op, because a fallback is numerically correct.
- **RoPE `seq_len` padding** to remove the `% 128` constraint. Needs a perf check. Worth more now
  than when this was written: with #24 fixed, dispatch is ~0.59 ms/call rather than ~52 ms, so
  padding cost is no longer swamped and the tradeoff is actually measurable.
- **GeLU-family activation kernels.** Cheap now that the pattern is established, but elementwise
  activations are memory-bound and sit far below the ~0.59 ms/call break-even threshold (#24). Still
  *not* recommended for a perf win — though the reason has changed from "invocation cost swamps
  everything" to "these ops are too small to clear break-even." Fine as coverage work, not as
  performance work.

---

## DECISIONS (session 4)

**D19. Ran the experiment instead of reporting it blocked.** I had recorded the graph-mode question
as unanswerable because `torch.compile` fails here, and was pushed on why I didn't just run it. That
push was correct on both counts. "Blocked" was doing work that "I haven't checked carefully enough"
should have been doing. *Rejected:* leaving Finding #21 as "needs a different stack" — it would have
gone into the PoC as the top ask, and it was wrong.

**D20. Attacked the question directly rather than through a proxy.** torch-xla is already a graph
runtime, so "does graph mode help" was answerable by counting device executions. Using
`torch.compile` as the instrument added a dependency on a broken-looking path and answered a
narrower question anyway. *Rejected:* fixing `torch.compile` first — days of work to answer something
one metric read settles.

**D21. Changed instrument rather than adding another variant.** After the "it's inside the NEFF"
conclusion, the tempting next step was another timing variant. Instead: a device profile, then a
Python profile. That took ~35 minutes after ~5 hours of framework-level experiments that could not
have falsified the hypothesis. Recorded as the project's main methodological output.

**D22. Kept the wrong conclusions visible instead of overwriting them.** #20's mechanism, #21's
"inside the NEFF" claim, and #21's three-candidate list are annotated in place rather than deleted,
and `week-4.md` has a superseded box rather than a rewrite. Two reasons: the measurements are still
correct and only the attribution was wrong, and the failure mode — a hypothesis surviving four
experiments because the instrument couldn't see the answer — is more instructive than a clean
document. *Rejected:* silently correcting. Anyone re-deriving this would hit the same trap.

**D23. Reported "still a net loss" alongside the 102x.** The fix is a good result and the temptation
was to lead with it. But kernelized is still 3.4x slower than baseline, and break-even needs a kernel
to save >0.59 ms/call while ours are 15–30x short. Both the README and the PoC state the loss
explicitly next to the win. *Rejected:* headlining 102x alone — it would get quoted as "the kernels
work now."

**D24. Measured amortisation rather than asserting it.** I had written "a larger model should narrow
the gap, we did not measure how much" into a deliverable. A hypothesis shouldn't sit in a deliverable
unlabelled, so I ran seq 2048: 3.36x → 2.06x, per-call cost up only 1.16x for 2.59x the work. Also
reported the qualification — 1.16x is not 1.0x, so ~16% does scale with size.

**D25. Did not attempt the `create_computation` fix.** Same class of bug, and it would plausibly take
the kernels to near parity, which is tempting. But it sits inside `torch_xla`'s op-registry path, it
is a much larger intervention than one decorator, and a wrong guess there could produce silently
incorrect results rather than an error. Filed as a scoping question (B12) with the attribution
attached. *Rejected:* patching it speculatively to get a better headline number.

**D26. Added `scripts/run_all_tests.py` rather than fighting the Makefile.** `make test` shells out
per suite, which `run_detached.sh` cannot launch since it execs `python` directly — and the e2e suites
exceed the SSH timeout, so detached running is mandatory. One suite per subprocess, so a crash can't
mask the others and each gets a clean Neuron runtime.

---

## SESSION 5 — Do the kernels actually beat the ops they replace?

Answering the one item under "What is not done" that mattered. Every performance number in the
project so far measured *dispatch* cost; none said whether the kernels are any good. And the two
possible answers point opposite ways — if NKI is faster on device, dispatch is the only thing between
here and a win; if slower, fixing dispatch never produces one.

### T21 — Device-time comparison, dispatch excluded

`scripts/profile_nki_vs_torch_device.py` profiles one `(op, impl, N)` configuration and writes a
NEFF+NTFF; `scripts/summarise_device_profiles.py` reads them via `neuron-explorer` and compares.
Identical work both ways — N chained applications, same shape, dtype and compiler defaults.

| config | device ms | ms/call | HBM r+w | MBU | active |
|---|---|---|---|---|---|
| silu / NKI / N=28 | 0.607 | 0.0217 | 188.7 MB | 43.2% | 95.1% |
| silu / torch / N=28 | **0.224** | **0.0080** | **6.3 MB** | 3.9% | 97.8% |
| rmsnorm / NKI / N=28 | 1.625 | 0.0581 | 188.8 MB | 16.2% | 99.1% |
| rmsnorm / torch / N=28 | **0.637** | **0.0227** | **6.4 MB** | 1.4% | 94.4% |

NKI 2.71x slower on SiLU, 2.55x on RMSNorm, with ~30x the HBM traffic.

### T22 — The first attribution was wrong, and nearly shipped

Dividing traffic by N at N=1 says NKI moves **exactly 3.00x** the necessary traffic for both ops. That
reads as a spilled intermediate, and the `nl.arange` migration had introduced an fp32 temporary, so
there was a ready culprit. Landing on 3.00x twice independently is what made me check.

Traffic isn't linear in N — a small NEFF carries fixed setup traffic. Two call counts solve both terms:

| config | traffic(1) | traffic(28) | marginal/call | vs floor | fixed |
|---|---|---|---|---|---|
| silu / NKI | 18.87 MB | 188.74 MB | **6.29 MB** | **1.00x** | 4.0 tiles |
| silu / torch | 6.29 MB | 6.29 MB | **0.00 MB** | 0.00x | 2.0 tiles |
| rmsnorm / NKI | 18.88 MB | 188.76 MB | **6.29 MB** | **1.00x** | 4.0 tiles |
| rmsnorm / torch | 7.88 MB | 6.42 MB | **~0.00 MB** | 0.00x | 2.5 tiles |

The unfused floor for a `[512, 3072]` bf16 tile is 2 tiles = 6.29 MB. **NKI's marginal traffic is
exactly that.** The kernels spill nothing and are optimal for an op that cannot fuse. Torch's traffic
is independent of N, which is only possible if the chain fused into one pass.

`scripts/analyse_fusion_barrier.py` does this regression and prints the attribution, so the wrong
version can't be reproduced by accident.

### T23 — The conclusion

A NKI custom call is opaque to the compiler, so nothing fuses across it. Replacing a torch op with a
NKI kernel doesn't merely add dispatch cost — it *removes* a fusion opportunity the compiler was
already exploiting. For memory-bound ops fusion is the entire optimisation, so the kernel is competing
against not touching HBM at all and cannot win however well written.

Finding #25. It makes break-even **unreachable** for these ops rather than distant, which is a
stronger and different claim than #24's "15–30x short". Propagated to the PoC document (recommendation
restructured, item 2 is now the fusion question), README, findings, steering blockers and this file.

## DECISIONS (session 5)

**D27. Measured kernel quality rather than leaving it in "What is not done".** It had been sitting
there as a known gap for a full session while the recommendation was written around dispatch cost.
That was the wrong order — the recommendation depended on it. *Rejected:* shipping a recommendation
whose central arithmetic (break-even) had an unmeasured term in it.

**D28. Profiled device time rather than reasoning from wall clock.** Wall clock conflates dispatch and
device, and dispatch was known to dominate, so no wall-clock experiment could have answered this. Same
lesson as session 4, applied deliberately this time instead of after five hours.

**D29. Checked whether traffic was linear before dividing by it.** The 3.00x reading was plausible and
had a ready culprit in our own recent kernel change, which is exactly when a wrong finding gets
accepted. Two call counts cost one extra profile run. *Rejected:* reporting "the kernels spill an fp32
intermediate" — it would have sent someone to optimise a kernel that is already optimal.

**D30. Stated the negative more strongly rather than less — and overdid it.** #25 changed "these ops
don't win yet" to "these ops cannot win", on the strength of a chained microbenchmark. **D32 reverses
this.** Recorded rather than deleted because the error is instructive: I noted in writing that the
microbenchmark was an upper bound, and then drafted the recommendation from the number anyway.

**D31. Kept the fusion question open rather than concluding it is fundamental.** Whether a NKI custom
call *could* participate in fusion is a compiler-team question I cannot answer from here, so it is filed
(B14) rather than assumed shut.

**D32. Measured the in-situ fusion penalty instead of shipping the microbenchmark figure.** The
recommendation had been reordered around 2.5–2.7x. In a real forward pass the device gap is 8.4% of the
regression and dispatch is 91.6%, so B12 went back above B14 and break-even went from "unreachable" to
"~1.18x with perfect dispatch". *Rejected:* leaving the microbenchmark number as the headline with the
caveat in a footnote — which is precisely what the previous round did, and the caveat did not stop the
wrong conclusion being drawn.

**D33. Added a confidence table to the PoC rather than levelling everything to one voice.** The document
now separates high-confidence measurements from the two projections (the ~1.18x figure, and whether a
region-spanning kernel beats the compiler) and says explicitly which claims a reader should push on.
Given the headline has been revised four times, flagging the load-bearing assumptions is more useful than
sounding uniformly certain.

---

## SESSION 6 — Review readiness, and a headline correction I should have made myself

Trigger: manager and mentor said **there shouldn't be a slowdown**, and asked to see the code and the
results. Both halves of that turned out to be fair.

### T24 — They were right, and my reporting was the problem

Re-reading my own numbers rather than defending them: the in-situ split is **91.6% dispatch, 8.4%
device**. So there is no structural slowdown. There is a framework bug worth 102x per call (fixed and
verified), a second caching bug of the same kind accounting for most of the rest, and ~8% from
replacing ops the compiler was already fusing. Projected with dispatch fixed: **~1.18x**.

I had been leading with whichever figure was newest and most dramatic. Three were in circulation, all
true, all differently misleading:

| figure | true of | why it misleads |
|---|---|---|
| 208x slower | pre-fix state | a one-line bug caused it |
| 2.5–2.7x on device | chained microbenchmark | that benchmark is deliberately NKI's worst case |
| 8.4% device / 91.6% dispatch | a real forward pass | representative, and it arrived last |

Logged as sticking point #18, because it cost reviewer trust and the fix is a habit rather than a
tool: before quoting a ratio, ask what configuration it is true of and whether anyone cares about
that configuration.

### T25 — The instance expired mid-session, taking every raw artifact

All measurement artifacts lived in `/tmp` on trn2: JSON outputs, NEFF/NTFF profile pairs, detached
logs. Gone. The numbers survived only because each run's stdout had been pasted into a commit message.

So the results were reproducible but not **auditable** — a reviewer could not open a file and check a
figure. That is exactly what had been asked for.

Sticking point #17. The irritating part: the project explicitly tracked "47 commits exist only on one
laptop" and missed the sharper version one layer down — the evidence lived on a machine with a
*shorter lifetime than the laptop*. Ephemerality was being reasoned about at the wrong level.

### T26 — Built the results tree that should have existed from day one

- `results/measurements.json` — 18 measurements, each with value, producing script, exact command,
  commit SHA, and a provenance status. Every entry currently reads `transcribed` rather than
  `in_repo`, and the file says so up front instead of implying file-backed numbers.
- `scripts/render_results.py` — generates `results/README.md` from that JSON, so a number cannot drift
  between the two. The rendered doc leads with the in-situ split and names the two figures most likely
  to be quoted out of context.
- `scripts/regenerate_results.py` — 21 stages, strictly sequential (two Neuron processes contend for
  cores), writing into `results/raw/<stage>/` with an `index.json` of command, exit code, duration and
  artifacts. `RAW/` placeholders in each stage's argv resolve into the artifact tree, so `--outdir`
  and `--json-out` land in the repo rather than `/tmp`.
- `results/raw/README.md` — states plainly why the directory is empty, and the lesson.

### T27 — Instrumented the one open item I could not run

Every measurement ran with `NEURON_CC_FLAGS` unset. That is the single configuration choice that could
invalidate the device comparisons, and it is the most plausible technical form of "there shouldn't be
a slowdown." No hardware, so I could not test it — instead `scripts/probe_compiler_flags.py` sweeps
`{unset, --target trn2, +--lnc 1, +--lnc 2, +-O2}` and reports whether the NKI/torch **ratio** moves,
not the absolute times.

Two details that matter for it to be correct: flags are a compile-time input, so each setting runs in
its own subprocess (changing them in-flight would silently reuse a NEFF built under the previous
setting), and each gets its own `NEURON_COMPILE_CACHE_URL`. Verdict threshold: ratio spread <1.25x
closes the item.

Wired in as **stage 4** of `make results`, deliberately early — if the ratio does depend on flags then
every device measurement after it needs re-running, and that is worth learning in minute five.

### T28 — Design doc and code guide

`deliverables/design-doc.md` opens with the correction rather than defending the old framing, then
covers the interception mechanism, what was built, performance with the denominator stated, the
methodology, the findings inventory, the recommendation, and what is not done. §3 answers the direct
question about why two kernels are tutorial-derived: nkilib has no standalone RMSNorm (always
quantises) and no activations module at all, so those ops exist only inside fused megakernels. RoPE is
the one standalone op and it *was* ported.

`docs/CODE_GUIDE.md` gives a reading order (a kernel → the guard harness → a test → the shim → e2e)
plus an inverse index from every script to the number it produced. The root-cause chain is listed in
the order it should be read, because the order is the argument.

Verified rather than asserted: a checker confirms every markdown link resolves, all 19 file references
exist, all 47 load-bearing figures appear in `measurements.json`, and the one projection is labelled
as such near where it is stated.

### T29 — Tidy

All 54 `.py` files already had docstrings, so the work was narrow: superseded headers on
`experiment_torch_compile_nki.py` (its premise was wrong three ways) and `tests/test_qwen3_layer.py`
(predates the execution guards, so it can pass on a silent fallback), usage sections added to four
scripts, and a docstring on `tests/__init__.py`. Superseded files are kept and marked, not deleted —
`docs/CODE_GUIDE.md` has a table explaining why each one is still there.

## DECISIONS (session 6)

**D34. Conceded the reviewers' point instead of defending the number.** The 208x and the 2.5–2.7x were
both defensible in isolation, and defending them would have been easy and wrong. The design doc opens
by saying my reporting led with the wrong figure. *Rejected:* a "clarification" framing that kept the
dramatic number and added context underneath it.

**D35. Made `measurements.json` the source of truth and generated the prose from it.** Two copies of a
number is two chances to be wrong, and this project already had numbers drifting between docs.
*Rejected:* a hand-maintained results table, which is what every previous doc in this repo used and is
why the findings index went four findings stale.

**D36. Marked every number `transcribed` rather than quietly presenting them as file-backed.** They are
commit-message-backed, which is weaker, and a reviewer asking to see results deserves to know which.
*Rejected:* regenerating artifacts from memory to fill `results/raw/`, which would have been
fabrication.

**D37. Instrumented the compiler-flag control rather than logging it as a caveat.** It was already
written down as an open item and had sat there unaddressed, which is the same failure as "a caveat in
the text is not a caveat in the conclusion." Making it stage 4 of the harness means it answers itself
on first contact with hardware. *Rejected:* another line in "what is not done."

**D38. Kept superseded scripts and annotated them.** Four scripts and two tests embody reasoning that
turned out wrong. Deleting them would make the repo look tidier and hide how the conclusions were
reached, which for a PoC whose main output is methodology is the wrong trade.

## BLOCKED — NEEDS INPUT (session 6)

**B15. A trn2 instance.** Nothing further can be measured or verified without one. First action:
`./scripts/sync_to_trn2.sh && ssh trn2 'cd hf-kernels-neuron && make results'`. That answers B16 as a
side effect and repopulates `results/raw/`.

**B16. Does the NKI/torch ratio depend on `NEURON_CC_FLAGS`?** Instrumented, unanswered. If the ratio
moves, Findings #25 and #26 need re-running under the better setting before being cited.

Still open and unchanged: B9 (sanity-check the recommendation), B10 (who owns `target.py`, do I write
the CR), B12 (is `create_computation` cacheable — top technical ask), B14 (can a NKI custom call
participate in fusion), B1 (Hub repo home, Samir).

**And still true: 47 commits exist only on this laptop.**

---

## SESSION 7 — A new instance, and turning transcribed numbers back into evidence

Session 6 ended with 18 measurements, every one marked `transcribed`, and a harness written but never
run. This session got hardware, ran it, and found that the harness was wrong in four places — three of
them silently.

### T30. Set up the replacement instance

New trn2 at `16.51.184.34` (`i-0b05f044388db8080`), `trn2.3xlarge`, 1 device, 4 NeuronCores, LNC2,
96 GB. Same instance type and core configuration as the expired host, which is a precondition for
comparing anything. `~/.ssh/config` updated, backed up first.

Verified the environment matched the recorded one *before* trusting any re-run, because a version
difference would make a disagreement uninterpretable — I would not know whether a changed number meant
a changed stack or a changed machine. torch 2.9.1+cu128, torch_xla 2.9.0, neuronx-cc
2.26.6360.0+6f180f47 came with the DLAMI; `kernels` 0.15.2 and transformers 5.15.0.dev0 installed from
`requirements.txt`, which pins transformers to commit `bb3ffb97` rather than tracking main — that pin
earned its keep here. `nki` reports `0.5.0+28631259367.ga768afa6` where `measurements.json` had plain
`0.5.0`; same version, and the build suffix is now recorded.

### T31. Ran the full harness, and it exposed four of its own bugs

All 23 stages exited 0 and the numbers reproduced. But watching it run surfaced four bugs, of which
three produced no error:

1. **`sync_to_trn2.sh --delete` destroyed 19 stages' artifacts.** Sticking point #19. Same failure as
   the original artifact loss — results living where something else deletes them — with a different
   destroyer, found while fixing the first one.
2. **A summing consumer plus a non-clearing producer double-counted device time.** Sticking point #20.
   Reported 16.9% device instead of 8.4%, silently. Fixed in both the producer and the consumer,
   because either alone still leaves a hole. Looking for the same shape deliberately found two more
   scripts with it.
3. **The summary read wall times from the command line,** so the harness passed constants measured on
   the *expired* host next to fresh device times. Right by luck, wrong in principle. The producer now
   writes `wall_times.json` and the consumer reads it, so a run is self-contained.
4. **The `RAW/` placeholder only matched its prefix form,** so a bare `RAW` passed through literally
   and 8 profile directories landed in `./RAW/`. Sticking point #21.

Three path bugs in one harness, all of the same shape: a file written somewhere nobody checked, all
invisible because every stage exited 0. A harness that says "all stages ok" while writing to the wrong
directory is worse than one that fails.

### T32. Made the provenance claims checkable

`scripts/check_measurement_provenance.py`. Verifies that every `status` is a documented value, that
anything claiming to be file-backed names an artifact, and that every named path actually exists. Its
first run failed with 15 missing artifacts — correctly, because they were still only on the instance.
That is the check earning its place immediately: "the artifact is somewhere" had been a belief.

Also moved the projection out of my head and into the script that owns its inputs.
`sum_model_device_time.py` now computes and labels `baseline wall + device gap`, and emits the whole
decomposition as JSON. It had been arithmetic in a commit message, which means it silently keeps the
old walls when the walls change — exactly the drift `measurements.json` exists to prevent.

### T33. Closed the top open item, in both halves

`probe_compiler_flags.py` (wall clock) had already shown NKI invariant across five flag settings at
1.02x spread. But it is ~97% dispatch by construction, so it could not speak to the device-time
findings the recommendation actually rests on. Wrote `probe_device_time_under_flags.py` for the device
half: same five settings, isolated compile cache each, profiling at N=1 and N=28 so marginal traffic
can be solved for.

| `NEURON_CC_FLAGS` | NKI ms | torch ms | ratio | NKI MB/call | vs floor |
|---|---|---|---|---|---|
| (unset) | 0.608 | 0.224 | 2.72x | 6.29 | 1.00x |
| `--target trn2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |
| `--target trn2 --lnc 1` | 0.580 | 0.429 | 1.35x | 6.29 | 1.00x |
| `--target trn2 --lnc 2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |
| `--target trn2 -O2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |

NKI device time spread 1.05x. NKI marginal traffic spread 1.00x — pinned at exactly the unfused floor
under every setting.

This is the strong form of the result, and it is worth being precise about why. The weak version is
"we tried five settings and none helped", which leaves a sixth setting open. The actual finding is that
the quantity a better setting would have to move is already at its theoretical minimum: NKI moves one
tile in and one tile out per call, which is the least an unfusable operation can move. There is no
headroom for a flag to find. The device gap is structural.

The 1.35x row is the trap. It is the best ratio in the table and it is not an improvement — NKI barely
moves (0.608 → 0.580) while torch gets 91% slower. That is the same mistake the first version of the
wall-clock probe made, which is why that probe now reports the two spreads separately.

### T34. Reproduction results

Every measurement re-run. Absolute step times run a few percent higher on this physical host; the
ratios reproduce, several to 3 significant figures.

| quantity | original | re-run |
|---|---|---|
| baseline seq512 | 42.04 ms | 44.36 ms |
| kernelized seq512 fixed | 141.43 ms | 146.67 ms |
| slowdown, fixed | 3.36x | 3.31x |
| slowdown, no fix | 208x | 206x |
| Finding #24 fix | 102.8x / 105.5x | 105.3x / 110.5x |
| host-issue share | 99.9% | 99.9% |
| NKI marginal traffic vs floor | 1.00x | 1.00x |
| device NKI/torch silu, rmsnorm | 2.70x / 2.55x | 2.72x / 2.56x |
| in-situ device share | 8.4% | 8.9% |
| projected with dispatch fixed | 1.18x | 1.17x |

The in-situ split has now been computed from three independent wall-time pairs across two physical
instances — 8.4%, 8.6%, 8.9% device, projecting 1.18x, 1.18x, 1.17x. The conclusion does not depend on
which pair is used, which matters because the walls are the noisiest input to it.

### T35. A new result from fixing a bug

Fixing the summariser's pairing (it keyed on op name only, so `_n1` and `_n28` directories collided and
the last one won) made the N=1 comparison visible for the first time:

| op | N=1 ratio | N=28 ratio |
|---|---|---|
| silu | 1.91x | 2.72x |
| rmsnorm | 2.37x | 2.56x |

The ratio grows with chain length, which is the fusion story confirming itself from a second direction:
the longer the chain, the more fusion torch gets, and the worse NKI looks by comparison. It also means
the N=28 figure is the worst case and the N=1 figure is the fairer single-op comparison — though N=1 is
dominated by the custom call's 12.58 MB of fixed NEFF setup traffic, which is why the marginal-traffic
regression, not either ratio, is the right instrument.

---

## SESSION 8 — B12 answered, and the first speedup

Two results, and the second retires the project's central negative claim.

### T36 — The residual was a second bypassed cache (Finding #28)

B12 had been the top technical ask since session 5: is the ~0.53 ms/call `create_computation` rebuild
cacheable? It had been deliberately left alone because cProfile put it inside `torch_xla`'s
op-registry path and a wrong guess there could be silently incorrect rather than raise.

Reading the source answered it before any measurement. `torch_xla/core/xla_op_registry.py` defines
`Op`, which holds `self._computations = dict()` and whose docstring asks callers to register ops
globally "in order to amortize the lowering cost". `nki/framework/_torch_xla.py::TorchXlaKernel.__call__`
applies `@xla_hlo_call` **inside** `__call__`, so every invocation constructs a fresh `Op` with a
fresh empty memo. The cache is not cold by accident; it is newly created, and therefore always empty.

Beside Finding #24 the pair is striking — both are *a cache exists and the code path defeats it*:

| | the cache | how it is defeated |
|---|---|---|
| #24 | `func._nki_compile_cache` | target resolution runs while building the cache *key* |
| #28 | `Op._computations` | the memo lives on an object recreated per call |

Verified in #24's shape: 0.5278 → 0.1828 ms/call (2.89x), 86 hits against 1 miss, cos_sim
**bit-identical to 16 digits**, baseline re-run last as a control. The patch asserts five structural
landmarks in the installed source and prints its hash, refusing to patch an NKI it does not recognise.

Model level: **52.25 → 0.605 → 0.162 ms/call, 322.5x total.** Slowdown 3.31x → **1.62x** at seq 512
and 2.06x → **1.37x** at seq 2048. The ~1.18x projection from session 5 assumed dispatch went to zero;
these are measured and sit between it and the old 3.31x, which is what the projection predicted.

### T37 — A SPEEDUP EXISTS (Finding #29)

Findings #25/#26 had produced a criterion: a kernel wins when it replaces a region the compiler would
*not* otherwise fuse well **and** there is real arithmetic to restructure. RMSNorm/RoPE/SiLU fail both
halves. The fused MLP passes the second and loses single-core. The one candidate the criterion
actually favours had never been tested.

Flash attention is an algorithmic restructuring, not a fusion — it never materialises the
`[heads, S, S]` score matrix. A compiler fuses elementwise chains; it does not re-derive the
algorithm. And `attention_cte` explicitly supports running without an SPMD grid.

It worked **first try** against the HF-native layout (`tp_q=True, tp_k=True, tp_out=False`), GQA
expressed natively with no K/V replication, `cos_sim 1.000010`. The first nkilib kernel that dropped
into the Kernel Hub's calling convention without a fight.

| seq | NKI ms/layer | torch ms/layer | verdict |
|---|---|---|---|
| 512 | 0.2463 | 0.1225 | 2.01x slower |
| 1024 | 0.4939 | 0.4269 | 1.16x slower |
| **2048** | **1.1438** | 1.6902 | **1.48x FASTER** |
| **3072** | **1.8484** | 3.9062 | **2.11x FASTER** |
| 4096 | 2.8295 | 1.5784 | 1.79x slower |

A **window**, not a threshold, and reproduced across two runs to four significant figures.

**My first explanation of the seq-4096 reversal was backwards, and the traffic column caught it.** I
blamed single-core SBUF pressure in the NKI kernel — plausible, matched Finding #26, and it blamed the
kernel. But torch's traffic goes 279.86 → 748.70 → **395.05** MB as seq goes 2048 → 3072 → 4096: it
*drops 47%* while the score matrix it is supposedly materialising *grows* from 302 to 537 MB. At 4096
it moves less than one copy of the score matrix, which is only possible if it stopped materialising
it. NKI is exactly on its own linear trend at every point. **Nothing degraded on the NKI side; the
compiler got better.** A one-number benchmark would have shipped the wrong story confidently.

## DECISIONS (session 8)

**D39. Verified `kernels`/`transformers` were unmodified before citing them.** Was about to send
Samir file:line citations on the strength of a docstring saying the project patches in-process only.
Checked pip RECORD SHA256s instead (`scripts/verify_kernels_pristine.py`): byte-identical to the
published wheels. So the Neuron plumbing being cited really is HF's code, not ours. *Rejected:*
trusting our own comment.

**D40. Corrected the seq-4096 explanation rather than shipping the tidy one.** See above. *Rejected:*
the SBUF story, which fit an existing finding and blamed the kernel.

---

## SESSION 9 — Native PyTorch, and both gates evaporate

Samir's reply to the week 3–6 summary was, in substance, *you are on the wrong stack*. He was right
about everything except one instruction.

### T38 — Fetched and built the native stack

`s3://huggingface-aws/pytorch-native/drop_jun_25/`. The bucket is in a different region than the CLI
default, and `presign` bakes the endpoint in, so it needs the region passed explicitly. The instance
role has no S3 access to it, so credentials came from the laptop via presigned URLs — which keeps
them off the instance. Python 3.12 venv, 926 MB of wheels.

Result: torch 2.11.0, torch-neuronx 0.1.0, NKI **0.6.0b1**, neuronx-cc 2.0.266551.0a0, torch-mlir.
`torch_xla` is not importable at all.

### T39 — A four-minute hang that was `PATH`, and a near-miss on the environment

First two runs hung silently. Process state showed a 29-thread parent on `futex_do_wait` with a
thread in `waitpid`, and a single-threaded child also on `futex_do_wait`, neither using CPU. That is
a textbook fork-from-a-multithreaded-process deadlock and every observation fit.

**It came with an expensive remedy.** The drop ships driver/runtime debs at build numbers *newer* than
the host's, and Samir's instruction was to install them. "Native wheels on a mismatched production
runtime hangs on first execution" is completely credible. Installing them replaces the host Neuron
driver and would very likely have broken the XLA venv holding every measurement in this project.

`strace -f -e trace=clone,execve` took a minute:

```
execve("/usr/local/sbin/neuronx-cc", ["neuronx-cc", "compile", "module.mlir", ...]) = -1 ENOENT
execve("/usr/local/bin/neuronx-cc",  ...) = -1 ENOENT      ... and 5 more, the whole PATH
```

On the first op needing a compile the runtime forks and `execve`s **`neuronx-cc` by bare name**. It
lives in `native_venv/bin/`, and running `/home/ubuntu/native_venv/bin/python script.py` — an absolute
path — does **not** put the venv's `bin` on `PATH`. Every entry missed, and the child then hung rather
than reporting it. Activating the venv fixed it: the same matmul completes in 1.21 s.

So the debs were never needed. Finding #30, and `scripts/run_native.sh` now activates, asserts
`neuronx-cc` resolves, prints the compiler version, and refuses to run otherwise.

### T40 — Both gates gone (Finding #31)

| | torch-xla | native |
|---|---|---|
| `model.device.type` | `"xla"` | **`"neuron"`** |
| `kernels._backend()` | `CUDA(12.8)` | **`Neuron()`** |
| `hasattr(torch, "neuron")` | False | **True** |
| declaring `["nki"]` | rejected | **accepted** |

Decisive test: stock `hub_kernels.kernelize()` with the shim asserted absent. 9 RMSNorm / 2 RoPE /
2 SiLU swapped, dispatch `nki=9/2/2` with zero fallbacks, logits `cos_sim 1.000001`. All three kernels
also compile and run under NKI 0.6.0b1.

**Proposed Change 1 is withdrawn**, along with Change 1b and the third site in
`kernel_config.infer_device`. **Change 2 — the mapping entries — survives and is the only upstream
blocker.** And the GitHub ticket Samir asked for should not be filed: it would send the kernels team
after a non-bug.

**A false negative on the decisive test, worth recording.** The first run printed
`RoPE swapped: 0 → Gate 2 NOT cleared` while the dispatch counter two lines below read `nki=2`. The
pass criterion was wired to structural inspection, which cannot see function kernels — stock
`kernelize()` `delattr`s the submodule alias in its `finally` block, and the swap mutates
`fn.forward`. Finding #8 established years-of-this-project ago that execution counters are
authoritative; having the principle written down did not put it in the pass condition. The obvious
repair (match qualnames) would also have failed, since the post-swap qualname is *identical*.

### T41 — Native perf, and the sign of the headline flips (Finding #32)

| seq | stack | baseline | kernelized | verdict |
|---|---|---|---|---|
| 512 | xla | 43.94 ms | 71.32 ms | 1.62x slower |
| 512 | native | **189.97 ms** | **96.46 ms** | *1.97x faster* |
| 2048 | xla | 117.78 ms | 161.04 ms | 1.37x slower |
| 2048 | native | **340.74 ms** | **251.86 ms** | *1.35x faster* |

The native baseline is **4.32x slower** than the XLA one. Native kernelized MFU (2.20%) is *below*
XLA kernelized (2.98%). Nothing got faster; the denominator got worse. This is Finding #27's `--lnc 1`
trap at model scale, and "the kernels are 1.97x faster on Native PyTorch" is a true sentence that
would be the most misleading thing this project could emit.

It does give Finding #25 support from the opposite direction: native is eager with no whole-forward
graph, so there is much less fusion to lose, so the barrier costs less. **The fusion penalty only
exists where there is fusion.**

### T42 — Samir's fused RMSNorm+MLP (Finding #33)

He was right that we had missed it: `normalization_type=NormType.RMS_NORM` with
`quantization_type=NONE`. Finding #26 had generalised "nkilib's RMSNorm always fuses quantisation" —
true of `core/rmsnorm/` — into "no non-quantising path exists". Wrong.

| shape | NKI fused | torch | verdict | cos_sim |
|---|---|---|---|---|
| H=1024 I=3072 | **0.6095 ms/block** | 1.0728 | **1.76x FASTER** | 0.999973 |
| H=4096 I=4096 | 2.5377 | 1.7459 | 1.45x slower | 0.999967 |

Second winning candidate, second shape window. Wall clock, so it carries the degraded-baseline caveat.

Two undocumented interface mismatches, both from that one keyword: the norm weight must be **2D**
(HF's is 1D `[H]`), and the hidden tensor must be **3D** where `NO_NORM` accepted 2D — the required
input *rank* depends on a keyword argument. Both fail at compile time, not validation.

**And Finding #18's `I > 4096` boundary is unchanged** on a compiler two generations newer. Identical
threshold, same `floordiv` by zero. Two generations agreeing is much stronger evidence for #26's
reading of it as a design boundary (no SPMD grid) than the original single-compiler report. So the
1.76x is at a shape nobody deploys — Qwen3-8B is I=12288.

## DECISIONS (session 9)

**D41. Did NOT install the deb packages, against an explicit instruction.** Flagged the risk, then
diagnosed before executing. The instruction addressed a problem that did not exist, and following it
immediately would have replaced the host driver to fix a `PATH` bug. *Rejected:* doing what the person
helping you says, immediately, because deferring feels wrong.

**D42. Do not file the GitHub ticket Samir asked for.** Both gates are XLA-path artifacts. *Rejected:*
filing it — it would send him chasing a non-bug in his own library.

**D43. Refused `--fix-op-registry` on native rather than silently skipping it.** Finding #28 patches
`torch_xla`, which is not importable there, so the fix is not merely unapplied — it is meaningless.
*Rejected:* letting a run be labelled as carrying a fix that cannot exist on that stack.

**D44. Reported the native result as the cross-stack table, not as a speedup.** *Rejected:* the
flattering true sentence. Sticking point #18 exists because this habit already cost reviewer trust
once.

**D45. Rewrote `poc-document.md` as a living document.** It had gone three findings stale and its
recommendation argued the opposite of the current conclusion. It now leads with status, carries a
"where we are struggling" section, and defers detail to `poc-findings.md` so there is one source of
truth per fact (27 KB, down from 49 KB). *Rejected:* patching the stale recommendation in place.

## BLOCKED — NEEDS INPUT (current)

**B17. Repo home — REFRAMED, and the earlier lean was wrong.** I had this as a genuine tradeoff
(trust on the default path vs versioning control) on the strength of the comment at
`hub_kernels.py:686` saying the gate covers "repos outside `kernels-community`". The implementation
disagrees: `kernels/utils.py::_check_trust_remote_code` queries an org-level `trustedKernelPublisher`
flag via the Hub API, and `kernels-community` is not special-cased in code — it just has the flag.
Verified live: `kernels-community` has `trustedKernelPublisher: true` and 56 kernels; **`aws-neuron`
already exists** (joint AWS/HF org, 31 models, 180 followers) with 0 kernels and no flag.

Versioning is a `v<N>` branch in a `repo_type="kernel"` repo, so control is just Hub write
permissions and does not depend on the org. **So the ask is `aws-neuron` + the flag**, which gets both
properties. Fall back to `kernels-community/` only if HF declines — and ask their criteria, since
`numKernels: 0` may itself be the blocker. Finding #34. Still blocks the mapping-entry PR, which names
repo IDs.

**B18. Can `kernelize()` express a multi-core SPMD launch?** Now the single gating question for the
whole performance story. Both winning candidates were built for a multi-core grid and are handicapped
single-core. Sits above weight layout (#17), the compile boundary (#18) and the dispatch fixes.

**B19. Device-time profiling on native.** Without it the fused-MLP 1.76x cannot be attributed, and any
native MFU number stays provisional.

**B20. Why is native eager 3–4x slower than the XLA graph path?** The largest single number in the
cross-stack table, and the first thing any reader will ask. Framework question, outside this PoC.

Still open: B10 (who owns `target.py`, do I write the CR — still applies on native), B14 (can a NKI
custom call participate in fusion), B3 (is inference-only acceptable for beta), B4 (Hub upload, gated
on B17).
