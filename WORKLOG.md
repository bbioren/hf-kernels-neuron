# Week 3 Worklog

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
