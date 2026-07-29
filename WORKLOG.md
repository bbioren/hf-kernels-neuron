# Week 3 Worklog

## SESSION SUMMARY

**Read this first.** Branch `week-3`, **nothing pushed**. All four test suites verified on trn2
(exit 0). Full writeup in `deliverables/week-3.md`; findings with severity in
`docs/poc-findings.md` (#8-#19); upstream asks with patches in `docs/upstream-fixes.md`;
reproduction scripts indexed at the end of the deliverable.

Two sessions. Session 1 = Week 3 proper (below). **Session 2** (at the bottom) ran the MLP
derisking spike and a benchmark attempt, and produced three more findings — including two
corrections to things I'd previously asserted without measuring. If you read only one extra
thing, read Finding #18: the fused MLP cannot run single-core at any realistic
`intermediate_size`, which blocks the fused-kernel direction outright.

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
