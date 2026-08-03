# HF Kernel Hub + Neuron: PoC Findings

Tracking pain points, gaps, and observations as we integrate NKI kernels with the HuggingFace Kernel Hub. This is the running log that becomes the PoC document in Week 6.

---

## Documentation Sources

These are the primary references we used. Critically, they describe different layers of the system and sometimes contradict each other (especially around `LocalLayerRepository` API):

| Source | What it covers | URL |
|--------|---------------|-----|
| `kernels` lib docs — Layers | `kernelize()`, `use_kernel_mapping`, `LocalLayerRepository`, layer requirements | https://github.com/huggingface/kernels/blob/main/docs/source/layers.md |
| `kernels` lib docs — Kernel Requirements | Hub repo layout, `metadata.json` schema, backend types, build variants | https://huggingface.co/docs/kernels/kernel-requirements |
| `kernels` lib docs — Quickstart | `get_kernel()`, basic loading | https://huggingface.co/docs/kernels/basic-usage |
| transformers PR #46754 | "Writing kernels" doc — the two-class pattern, `KernelConfig`, module fusion, **single-file kernel example** | https://github.com/huggingface/transformers/pull/46754/files |
| transformers PR #46339 | Extended kernel fusion API via `KernelConfig`, `register_kernel_replacements_and_fusions` | https://github.com/huggingface/transformers/pull/46339 |
| transformers docs — Kernels | `_KERNEL_MAPPING`, `use_kernels=True`, `KernelConfig` | https://huggingface.co/docs/transformers/main/kernels |
| `kernels-community/layer_norm` Hub repo | Example of a published CUDA kernel (build.toml, source layout) | https://huggingface.co/kernels-community/layer_norm/tree/v1 |
| HF Blog — Kernel Hub intro | High-level overview of the Kernel Hub vision | https://huggingface.co/blog/hello-hf-kernels |

**Key insight:** The `kernels` library docs describe the Hub *build artifact* format (multiple directories, build variants, compiled `.so` files). But transformers PR #46754 shows that for local/dev kernels, a kernel is just **one Python file** with a class that has `forward()` + a `layers` namespace class + `metadata.json`. Our initial scaffold was over-engineered because we followed the Hub build docs instead of the transformers integration pattern.

---

## Summary Table

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | `kernels` not installable from GitHub source | Low | Resolved |
| 2 | `LocalLayerRepository` API underdocumented / changed between versions | Medium | Resolved |
| 3 | `metadata.json` required fields not documented for local dev | Medium | Resolved |
| 4 | No `kernel-builder` Neuron build variant | High | Open |
| 5 | Neuron device shows "not detected (CPU-only mode)" despite hardware present | Low | Resolved (see #8) |
| 6 | Kernel can be a single file — our multi-file layout was over-engineered | Low | Resolved |
| 7 | Variant resolver detects CUDA, not Neuron; flat layout works via fallback | Medium | Open |
| 8 | **NKI kernels silently fall back on CPU tensors — no warning. Invalidated our Week 2 accuracy results** | **Critical** | Resolved (methodology fixed) |
| 9 | **`use_kernels=True` cannot reach the `"neuron"` device path at all** | **High** | Open (upstream fix identified) |
| 10 | Function-kernel replacement is process-global, not per-model | Medium | Open |
| 11 | transformers kernel-decorator coverage is much wider than assumed (110 RMSNorm / 95 RoPE model files) | Positive | Confirmed |
| 12 | HF already whitelists `nki` as a Neuron dependency — but the entry is unreachable, so kernels must under-declare | High | Open |
| 13 | nki-library HAS a standalone HF-layout RoPE kernel (`rope_hf`), undocumented in the public API reference | Positive | Confirmed |
| 14 | ~~Two NKI import paths with different capabilities~~ → **version skew: `nki`=0.5.0, `neuronxcc.nki`=older bundled. `nl.arange` was removed. Our RMSNorm+SiLU use a removed API** | Medium | **Corrected** — now our tech debt, not an upstream ask |
| 15 | Interception-point inventory: several `_KERNEL_MAPPING` entries are unreachable (incl. `SwiGLUMLP`) | Reference | Confirmed |
| 16 | **`nkilib` is already installed and its kernels are directly callable — thin-wrapper porting is feasible today** | **High** | Open (policy blocker only) |
| 17 | Fused MLP: `kernelize()` has no weight-transformation hook | High | Open (design decision) — **premise partly corrected, see #17** |
| 18 | **Fused MLP kernel cannot run single-core when `intermediate_size > 4096` — excludes every real model** | **High** | Open (nki-library bug; no wrapper workaround) |
| 19 | Eager NKI *host* dispatch costs ~0.36 ms/call; per-layer microbenchmarking can't resolve kernel quality | Medium | **Reconciled by #24** — 0.36–0.59 ms is the dispatch floor; #20's 52 ms sat on top of it |
| 20 | Every `@nki.jit` invocation from eager PyTorch/XLA costs ~53 ms, independent of problem size. MFU 5.06% → 0.02% | Critical | **Measurement stands; mechanism superseded by #24** |
| 21 | ~~The decisive graph-mode question can't be answered here~~ → **answered: 28 NKI calls already share one HLO graph and one device execution and still cost 28x. Graph batching was never the lever** | High | **Resolved** — conclusion stands, one sub-claim corrected by #24 |
| 22 | Qwen3-MoE won't run on Neuron with transformers' default experts impl (`sort` unsupported); `batched_mm` fixes it. All 3 kernels then transfer unchanged | High | Workaround found; doc gap open |
| 23 | `torch_neuronx`'s op overrides aren't fake-tensor safe, breaking `torch.compile` on nearly every transformer (`Embedding`, `Softmax`, `CrossEntropyLoss`, …) | High | Open — outside this project's scope, reproducer included |
| 24 | **THE ROOT CAUSE: `_detect_target()` forks `neuron-ls` on every kernel invocation, ~52 ms, outside the compile cache. One `lru_cache` = 102x/call, MFU 0.02% → 1.50%** | **Critical** | **Fix verified, accuracy-neutral. Residual ~0.59 ms/call in `create_computation` is 91.6% of what's left** |
| 25 | Each NKI call is an optimisation barrier — the compiler can't fuse across it. Kernels are provably optimal (marginal traffic 1.00x the unfused floor); the loss is the forfeited fusion | Critical | **2.5–2.7x in a chained microbenchmark, but 8.4% of the regression in situ.** Decides the last ~18% after #24's residual |
| 26 | **The fused MLP — the one kernel that could win — loses by 2.8–3.0x on device too, at both shapes it can run single-core. `nkilib` kernels need an SPMD grid; #18 is a design boundary, not a bug** | **Critical** | **Answers "where would a speedup come from?": nowhere, in this configuration.** Also explains why RMSNorm/SiLU had to be tutorial-derived — no standalone versions exist in nki-library |
| 27 | **The device gap is not a compiler-flag artifact.** NKI is invariant across `{unset, --target trn2, ±--lnc 1/2, -O2}`: 1.02x on wall clock, 1.05x on device, and marginal traffic pinned at **1.00x the unfused floor** under every setting | **Critical** | **Closed.** Not "no better flag found" — the quantity a flag would have to move is already at its theoretical minimum, so #25 and #26 are structural. Both probes are harness stages, so a compiler upgrade re-tests it |
| 28 | **B12 answered: the residual is a SECOND bypassed cache.** `torch_xla`'s `Op` already memoises the built computation; NKI applies `@xla_hlo_call` inside `__call__`, so a fresh `Op` with an empty memo is built per call. 0.53 → 0.18 ms/call, bit-identical output | **Critical** | **Closed.** Model slowdown 3.31x → **1.62x** (seq 512) and 2.06x → **1.37x** (seq 2048). 52.25 → 0.605 → 0.162 ms/call, 322x total. Two upstream asks now, both one line, both in NKI's dispatch path |
| 29 | **A SPEEDUP EXISTS.** `nkilib` flash attention (`attention_cte`) beats the compiler's eager attention by **1.48x at seq 2048 and 2.11x at seq 3072** on device; it loses 2.01x at seq 512 and 1.79x at 4096. A WINDOW, not a threshold | **Critical** | **First winning candidate**, and for the reason #25's criterion predicted: flash is an algorithmic restructuring the compiler does not derive. Worked first try against the HF layout. The window's upper edge is the compiler *improving* at 4096 (its traffic drops 47% while the score matrix grows), not the kernel degrading |

---

## Detailed Findings

### 1. `kernels` library not pip-installable from GitHub

**What happened:** `pip install git+https://github.com/huggingface/kernels.git@main` fails because the repo has no `pyproject.toml` or `setup.py` at root. It's a Rust/Python hybrid built via `maturin` and published to PyPI separately.

**Impact:** Minor — you just use `pip install kernels` from PyPI. But it means you can't easily test unreleased features from main unless you build locally with maturin + Rust toolchain.

**Recommendation:** Document that developers should pin a PyPI release, not install from source. If Neuron-specific patches land in `kernels` before a release, we'd need to build from source (requires Rust + maturin setup).

---

### 2. `LocalLayerRepository` API differs from documentation

**What happened:** The HF docs (layers.md on GitHub) show `LocalLayerRepository(repo_path, package_name, layer_name)` but v0.15.2 only accepts `(repo_path, *, layer_name)` — `package_name` was removed.

**Impact:** Medium — anyone following the docs to write a local kernel will hit a `TypeError` immediately. The docs lag behind the actual API.

**Recommendation:** For the PoC doc, note that the kernels library is pre-1.0 and APIs change between minor versions. Pin `kernels>=0.15,<0.16` and verify constructor signatures against the installed version, not the docs.

---

### 3. `metadata.json` required fields unclear for local development

**What happened:** `LocalLayerRepository.load()` calls `get_local_kernel()` which requires a `metadata.json` with specific fields: `name`, `id`, `version`, `license`, `python-depends`, `backend`, `digest`. The kernel requirements docs describe this for Hub publishing but don't emphasize it's also needed for local dev via `LocalLayerRepository`.

**Fields we needed:**
```json
{
  "name": "neuron-rmsnorm",
  "id": "neuron_rmsnorm",
  "version": 0,
  "license": "Apache-2.0",
  "python-depends": [],
  "backend": { "type": "neuron" },
  "digest": { "algorithm": "sha256", "files": {} }
}
```

**Impact:** Medium — the error messages are clear once you know what to look for, but the local dev workflow isn't documented as a first-class path. You essentially have to replicate Hub structure even for local testing.

**Recommendation:** A `kernels init-local` CLI command or a simpler `LocalLayerRepository` that doesn't require metadata would lower the barrier. Flag this in the PoC as friction for kernel developers.

---

### 4. No `kernel-builder` Neuron build variant [PARTIALLY RESOLVED]

**What happened:** The `kernel-builder` tool has no documented Neuron target. We investigated whether this actually blocks Hub publishing for NKI kernels.

**Investigation (2026-07-22):**

We dug into the `kernels` library source on trn2 and found:
- `kernels/backends.py` has a `Neuron` class that parses `"neuron"` variant strings ✓
- `kernels/variants.py` can parse `torch29-neuron-x86_64-linux` as a valid variant ✓
- `parse_backend("neuron")` works ✓

**BUT:** The variant resolver in `get_local_kernel()` auto-detects the backend from torch's build config. On the DLAMI, torch reports `cu128` (CUDA), NOT `neuron`. So a Hub repo with a `build/torch29-neuron-x86_64-linux/` variant directory won't resolve — the loader looks for a CUDA variant, doesn't find one, and fails.

**Why our flat structure works:** When variant resolution fails, `get_local_kernel()` has a fallback: it tries to import `repo_path` directly (as if the repo root IS the kernel). That's why our current layout (`__init__.py` + `metadata.json` at repo root, no `build/` subdirectory) works — it hits the fallback path.

**The real blocker:** `LocalLayerRepository.load()` calls `get_local_kernel(repo_path)` WITHOUT passing a `backend` arg. There's no way to override the backend detection. Even if you pass `device="neuron"` to `kernelize()`, that only affects which kernel mapping entry to look up — it does NOT tell the variant resolver which variant directory to load.

**Conclusion:** Hub publishing for Neuron kernels would need either:
1. A change to `LocalLayerRepository` / `get_local_kernel` to accept a backend override (so `device="neuron"` propagates to variant resolution)
2. OR: publish kernels in the flat format (no variant dirs) and rely on the fallback path — this works today but feels like a hack
3. OR: The DLAMI's torch needs to report `neuron` as its backend (via `torch.neuron` attribute) — this check exists in `backends.py` line 198 (`if hasattr(torch, "neuron")`) but doesn't fire on current DLAMI

**Impact:** Medium — we can still publish to the Hub using the flat format today. The variant system isn't strictly needed for pure-Python NKI kernels (there's only one "build" — it's just Python source). But it means multi-backend repos (CUDA + Neuron in one package) won't work until the resolver is fixed.

---

### 5. Neuron device detection shows "CPU-only mode" [INVESTIGATING]

**What happened:** `neuron-ls` shows the device (1 device, 4 NeuronCores, 96 GB HBM) but the verification script reports "Neuron device: not detected (CPU-only mode)". This is likely because our detection code uses an XLA call that requires explicit device initialization, which doesn't happen at import time in eager mode.

**Impact:** Low — this is a cosmetic issue in our test script, not a real problem. The `kernelize()` path works with `device="neuron"` as a string override regardless of runtime device detection. NKI kernels will compile and execute on NeuronCores when actually called (Week 2 will confirm).

**Recommendation:** For eager-mode kernel development, device detection isn't needed — you pass `device="neuron"` explicitly. The kernels library's auto-detection path (inferring device from model parameters) may need Neuron-specific logic if it doesn't already handle this.

---

### 6. Minimal kernel structure is simpler than Hub docs suggest

**What happened:** We initially built a multi-file layout (`__init__.py`, `layers.py`, `nki_rmsnorm.py`, `metadata.json`) following the Hub kernel-requirements docs. But transformers PR #46754 reveals the actual pattern is much simpler — **a single Python file** with:

```python
class NeuronRMSNorm(nn.Module):
    def forward(self, hidden_states):
        ...

class layers:
    NeuronRMSNorm = NeuronRMSNorm
```

Plus a `metadata.json`. That's the complete kernel. The `layers` thing isn't a separate file — it's a plain class used as a namespace to expose kernel classes.

**Root cause:** Two different docs describe two different things:
- `kernels` library kernel-requirements docs → describe the Hub *build artifact* (compiled CUDA kernels with build variants, `.so` files, etc.)
- transformers PR #46754 "Writing kernels" → describes the *author-facing* pattern for Python-only kernels (NKI kernels are pure Python via `@nki.jit`)

For NKI kernels (which are Python, not compiled C++/CUDA), the simpler single-file pattern is correct.

**Impact:** Low — our multi-file version still works, it's just more files than needed. Will simplify going forward.

**Recommendation:** The PoC doc should clarify that NKI kernels follow the single-file pattern from PR #46754, not the multi-directory CUDA build pattern. NKI kernels are Python source, so they don't need the `kernel-builder` compilation step.

---

## Environment (verified 2026-07-22)

| Package | Version |
|---------|---------|
| kernels | 0.15.2 |
| transformers | 5.15.0.dev0 (from main) |
| torch | 2.9.1+cu128 |
| torch_neuronx | available |
| neuronx-cc | 2.26.6360.0+6f180f47 |
| Python | 3.12.3 |
| Instance | trn2.3xlarge (1 device, 4 NeuronCores, 96 GB) |

## Setup friction (for PoC "ease of use" section)

- DLAMI has torch in `/opt/aws_neuronx_venv_pytorch_2_9/`, not system-wide. Must `source activate` or use `.pth` hack.
- Ubuntu 24.04 blocks system pip installs (PEP 668). Venv required.
- `torch_neuronx` import triggers runtime initialization and fails if helper binaries (`libneuronpjrt-path`) aren't on PATH. Venv must include `bin/` from the Neuron environment.
- Net result: ~15 min from "fresh instance" to "verification script passes." Acceptable for developers, but the kernel Hub vision of "just `pip install kernels` and go" doesn't apply on Neuron today.

---

## Week 1 Results

**Goal:** Verify neuron device path, prove forward-swap mechanism works.

**Result:** All 4 tests pass ✓
1. `kernelize(device="neuron")` — accepted without error
2. `LocalLayerRepository` — loads local kernel package
3. Forward swap — confirmed (output changes from sentinel to real computation)
4. Fallback — works (unmapped layers keep original forward)

**Conclusion:** The HF Kernel Hub mechanism works end-to-end for Neuron. The device path is merged, the layer loading works, stateless kernels swap correctly. The remaining work is kernel implementation (NKI), not plumbing.

---

# Week 3 Findings

## 8. NKI kernels silently fall back on CPU tensors — and this invalidated our Week 2 accuracy results [CRITICAL]

**This is the most important finding of the PoC so far.**

**What happened:** The Week 2 accuracy test reported `cos_sim = 1.000000` and
`max_diff = 0.00e+00` for all 8 shapes. Bit-identical output is a red flag: a real
NKI kernel on hardware does its reductions in a different order than PyTorch and
will differ by ~1e-4 in fp32. Exact zero means the two sides were running the
*same code*.

They were. `NeuronRMSNorm.forward` gates the kernel on:

```python
if _HAS_NKI and hidden_states.device.type != "cpu":
    output_2d = _nki_rmsnorm_kernel(...)   # NKI
else:
    output_2d = _pytorch_rmsnorm(...)      # fallback
```

The Week 2 tests build inputs with `torch.randn(...)` — CPU tensors. So every test
took the `else` branch. `_pytorch_rmsnorm` is mathematically identical to
`Qwen3RMSNorm.forward`, so it compared the fallback against itself and reported a
perfect score.

**Verified with instrumentation** (`scripts/probe_nki_execution.py`, run on trn2):

| Probe | Result |
|-------|--------|
| CPU tensors, count which branch runs | NKI calls = **0**, fallback calls = **1** |
| `@nki.jit` called directly with CPU tensors | `RuntimeError: Expected all tensors in the given list to be XLA tensors` |
| `@nki.jit` on XLA tensors (`xla_device_hw` = `NEURON`) | Compiles + runs. cos_sim `1.000000`, max_diff **`1.731e-04`** |
| Full `NeuronRMSNorm` layer on XLA tensors | NKI calls = **1**, fallback = 0. cos_sim `1.000000`, max_diff `1.297e-04` |

**Two conclusions, and they point opposite directions:**

1. *The kernel itself is correct.* On real Neuron hardware it produces
   `cos_sim = 1.000000` with a `~1.3e-4` max absolute difference, which is exactly
   the rounding behaviour you expect from a genuine hardware reduction. RMSNorm
   passes the quality bar — **now** it's actually been measured.
2. *Our test methodology was wrong and would have stayed wrong.* Nothing in the
   test surfaced that the kernel wasn't running. The test even printed
   `"Backend: NKI kernel (NeuronCores)"`, because it was reporting `_HAS_NKI`
   (is NKI importable) rather than "did the NKI branch execute".

**Impact on customers:** severe, and this is the generalizable part. `@nki.jit`
hard-errors on CPU tensors, so *any* HF Neuron kernel needs a device guard, and the
natural way to write that guard produces a silent fallback. A customer sets
`use_kernels=True`, sees no warning, sees correct numbers, and concludes they have
NKI acceleration. They have eager PyTorch. There is no signal anywhere — no log
line, no attribute, no counter.

**Recommendations:**
- Every Neuron kernel accuracy test must assert the NKI branch actually executed,
  not just that the output is numerically close. A numerically-perfect result is
  *evidence of failure* for a hardware kernel. We now do this via a call counter
  (`tests/nki_test_utils.py::nki_call_counter`).
- Kernels should `logger.warning_once` when they fall back, naming the reason.
- The `kernels` library should expose which implementation is live per layer
  (e.g. `model.get_kernel_report()`), so users can verify rather than trust.
- Ban exact-zero diffs as a pass condition. Assert `0 < max_diff < tol` for
  hardware kernels.

## 9. `use_kernels=True` cannot reach the `"neuron"` device path [HIGH]

The Week 3 goal "confirm `use_kernels=True` alone triggers the swaps on Neuron"
**cannot be met today**, for two independent reasons. Verified empirically in
`scripts/probe_neuron_device_path.py`.

**Reason A — transformers has no device override.**

```python
# transformers/integrations/hub_kernels.py
def kernelize(model: "PreTrainedModel", mode: "Mode | None" = None):
    ...
    device_type = model.device.type          # <- only source of truth
    device = Device(type=device_type)
```

The signature is `(model, mode)`. There is no `device` parameter, unlike the
underlying `kernels.kernelize(model, *, mode, device=None, use_fallback=True)`,
which does accept one. So a caller cannot ask for `"neuron"`.

**Reason B — Neuron never reports `"neuron"` as a device type.**

| Model state | `model.device.type` | `_find_device()` | Outcome |
|---|---|---|---|
| As constructed (eager, params on host) | `"cpu"` | `Device(type='cpu')` | neuron mapping **silently ignored** |
| Moved to Neuron via `xm.xla_device()` | `"xla"` | `Device(type='xla')` | **hard error**: `Unsupported device type 'xla'. Supported device types are: cpu, cuda, mps, neuron, npu, rocm, xpu` |

`"xla"` is not in the supported set, and nothing maps `"xla"` → `"neuron"`.
Separately, `hasattr(torch, "neuron")` is `False` even after
`import torch_neuronx`, so `_has_neuron_ops()` is also `False`.

So the user-facing path has three failure modes and zero success modes: ignore the
mapping, crash on an unsupported device, or fail the `_has_neuron_ops` gate.

**What still works:** calling the `kernels` library directly with an explicit
device, which is what our tests do and what Weeks 1-2 validated:

```python
kernelize(model, device="neuron", mode=Mode.INFERENCE)   # kernels lib, not transformers
```

**Correction to Reason B, found by testing the fix.** The failure is not actually a
hard error on the `use_kernels=True` path — it is a *silent no-op*, which is worse.
transformers passes a `Device` **object** rather than a string, and
`kernels.kernelize` only calls `_validate_device_type` when it receives a string:

```python
if device is None:      device_type = _find_device(model)
elif isinstance(device, str):
                        _validate_device_type(device); device_type = Device(type=device)
else:                   device_type = Device(device.type)      # <- unvalidated
```

So `Device(type="xla")` passes straight through, matches no mapping entry, and every
layer quietly keeps its original forward. `kernelize()` returns successfully. The
`Unsupported device type 'xla'` error only appears if you call the kernels library
directly with `device="xla"` as a string.

**Minimal upstream fix — and where it actually goes.** Our first proposal was to
patch `kernels._find_device`. **That would not have worked**, and the e2e test caught
it: the transformers wrapper computes the device itself and passes it explicitly, so
`_find_device` is never consulted on the `use_kernels=True` path. The fix has to be in
transformers:

```python
# transformers/integrations/hub_kernels.py::kernelize
device_type = model.device.type
if device_type == "cuda" and is_rocm_platform():
    device_type = "rocm"
elif device_type == "xla" and _is_neuron_xla():     # <- the fix
    device_type = "neuron"
device = Device(type=device_type)
```

`_is_neuron_xla()` checks `xm.xla_device_hw(xm.xla_device()) == "NEURON"`, which we
confirmed returns exactly `"NEURON"` on trn2 — reliable, no new dependency.

**Verified sufficient.** Applied in-process (`tests/test_qwen3_neuron_e2e.py` test 2),
this single branch takes Qwen3 from **0 → 9 swapped RMSNorm layers** through the
transformers `use_kernels` path, with logits `cos_sim = 1.000001`. So the change is
not merely proposed, it is demonstrated. The same fix is worth applying to
`kernels._find_device` as well (Change 1b), for callers that rely on auto-detection.

A complementary, independently useful fix: have `torch_neuronx` set a `torch.neuron`
attribute so `_has_neuron_ops()` fires. That alone is *not* sufficient — it changes
neither the transformers device computation nor `_find_device`'s return.

**Lesson worth carrying into the PoC:** we would have shipped a wrong recommendation
had we not built a test that actually applies the proposed patch. "Propose a fix" and
"verify the fix works" are different activities, and for an upstream ask aimed at
another team, the second one is what makes the recommendation credible.

## 10. Function-kernel replacement is process-global, not per-model [MEDIUM]

`@use_kernel_func_from_hub("rotary_pos_emb")` replaces the module-level
`apply_rotary_pos_emb` with a *single instance* of a generated `Func(nn.Module)`
wrapper. `@use_kernelized_func(apply_rotary_pos_emb)` on `Qwen3Attention` then
stores that same instance into each attention module's `_hidden_kernels` dict.

Verified: the object in `model.layers.N.self_attn._hidden_kernels["rotary_pos_emb"]`
**is** (identity, not equality) `modeling_qwen3.apply_rotary_pos_emb`.

Consequences:
- Every layer of every Qwen3 instance in the process shares one wrapper object.
- The call site `apply_rotary_pos_emb(query_states, key_states, cos, sin)` resolves
  the module-level global, so an in-place swap is what makes the replacement
  visible — by design.
- But it means you cannot have two models in one process with different RoPE
  kernels, and kernelizing one model mutates RoPE for all of them.

Not a blocker for the PoC. Worth flagging for anyone doing multi-model serving, and
worth confirming with the HF team that this is intentional rather than incidental.

## 11. transformers kernel-decorator coverage is wider than we assumed [POSITIVE]

The steering doc estimated 87 model files with RMSNorm and 66 with rotary
embeddings. Actual counts in transformers `5.15.0.dev0`:

| Decorator | Kernel name | Model files |
|-----------|-------------|-------------|
| `@use_kernel_forward_from_hub("RMSNorm")` | `RMSNorm` | **110** |
| `@use_kernel_func_from_hub("rotary_pos_emb")` | `rotary_pos_emb` | **95** |

Qwen3 opts into **both**, so no transformers-side changes are needed to target it:

```python
@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module): ...

@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1): ...

@use_kernelized_func(apply_rotary_pos_emb)
class Qwen3Attention(nn.Module): ...
```

This strengthens the leverage argument: two Neuron kernels would light up 110 and 95
model files respectively, if the device-path gap in Finding #9 is closed. The
interception points already exist upstream and are already wired into the models —
the only missing pieces are the `"neuron"` mapping entries and device detection.

`"rotary_pos_emb"` already has `cuda`, `rocm`, and `xpu` entries in
`_FUNCTION_KERNEL_MAPPING`. There is no `"neuron"` entry. That is a one-block
addition, not an architectural change.

## 12. HF already whitelists `nki` for Neuron — but the entry is unreachable [HIGH]

Good news first: `kernels/python_depends.json` in 0.15.2 contains a `neuron` backend
section, and it whitelists exactly one dependency — `nki`:

```json
"backends": {
  "cuda":   { "nvidia-cutlass-dsl": {...} },
  "neuron": { "nki": { "nix": [], "python": [{ "pkg": "nki", "import": "nki" }] } },
  ...
}
```

So HuggingFace has already anticipated NKI kernels and made room for them. That is a
meaningfully better starting position than `docs/porting-recommendations.md` assumed
(it recorded the allowlist as only `einops` + `nvidia-cutlass-dsl`; corrected there).

**But the entry cannot be used.** `_import_from_path()` calls:

```python
validate_dependencies(module_name, metadata.python_depends, _backend())
```

and `validate_dependencies` looks the declared dependency up in
`general` first, then `backends[backend.name]`. On the Neuron DLAMI:

```
_backend()      = CUDA(version=Version('12.8'))
_backend().name = 'cuda'
```

so it consults the **cuda** table, which contains only `nvidia-cutlass-dsl`. Verified
against a real copy of our RoPE kernel:

| `python-depends` | Result |
|---|---|
| `[]` | loads OK |
| `["nki"]` | `ValueError: Kernel module 'neuron_rope' uses unsupported kernel dependency: nki` |
| `["nkilib"]` | `ValueError: ... unsupported kernel dependency: nkilib` |

**This is Finding #7 compounding.** `_backend()` misreports because
`hasattr(torch, "neuron")` is False, and that single root cause now breaks two
separate things: build-variant resolution *and* dependency validation. A Neuron kernel
today is forced to ship `python-depends: []` while importing `nki` — i.e. to
under-declare its dependencies in order to load at all. That works only because the
DLAMI happens to have NKI preinstalled. On a host without it, the kernel would fail at
import with a bare `ImportError` instead of the intended actionable
`"requires Python dependency nki. Please install with: pip install nki"`.

**Recommendation.** Fixing `_backend()` to report `neuron` on Neuron hosts is now the
highest-leverage single upstream change, because it unblocks variant resolution and
dependency declaration together. The root cause is the same `hasattr(torch, "neuron")`
check, so `torch_neuronx` setting that attribute is the cleanest fix — and unlike its
effect on device *routing* (where it does nothing, per Finding #9), here it is
genuinely sufficient.

**Separately:** `nkilib` is not whitelisted. The ask to add it is now much more
concrete, because `nki` establishes the precedent and the exact JSON shape to copy in
the same file. That is the prerequisite for the "thin wrapper" porting strategy
(Option D in `docs/nki-library-porting-analysis.md`).

### metadata.json requirements, measured

Probed by removing one field at a time:

| Field | Required? |
|-------|-----------|
| `name`, `id`, `version`, `license`, `python-depends`, `backend` | **required** — parse fails without any of them |
| `digest` | **optional** — loads fine without it |

So the minimum viable Neuron kernel repo is two files, and the `digest` boilerplate we
have been carrying (`{"algorithm": "sha256", "files": {}}`) can be dropped:

```
kernels/neuron_<op>/
├── __init__.py      # kernel class + `class layers:`  (or a top-level function)
└── metadata.json    # name, id, version, license, python-depends, backend
```

Confirmed both our real kernels load from this flat layout, and that the func kernel
correctly has **no** `layers` namespace while the layer kernel does.

## 14. The two NKI import paths are not interchangeable, and neither is a superset [HIGH]

Both of these import cleanly on the DLAMI:

```python
import nki, nki.language as nl, nki.isa as nisa            # top-level package
import neuronxcc.nki as nki, neuronxcc.nki.language as nl  # compiler-bundled
```

They are different implementations with **different capabilities**, and a kernel is
effectively pinned to whichever one supports its idiom. Discovered by writing the SiLU
kernel against the top-level `nki` (matching nki-library's own imports) and having every
shape fail to compile.

| Idiom | top-level `nki` | `neuronxcc.nki` |
|---|---|---|
| `nl.arange` index tensors + `mask=` | **fails**: `error: failed to resolve name 'nki.language.arange'` | works |
| `//` (floor-div) on tensor shape values | works (shapes are plain ints) | **fails**: `NotImplementedError: math.trunc() is not supported for scalar` |
| `reshape_dim` / `permute` / `nisa.tensor_tensor` on sliced dests | works | untested (blocked by the above) |

Consequences, verified by swapping the imports in each kernel and re-running the suites:

| Kernel | Idiom used | Required package |
|--------|-----------|------------------|
| `neuron_rmsnorm` | `nl.arange` + mask (tutorial style) | `neuronxcc.nki` |
| `neuron_silu` | `nl.arange` + mask | `neuronxcc.nki` |
| `neuron_rope` | slicing + `div_ceil` (nki-library style) | top-level `nki` |

### CORRECTION (measured 2026-07-29, `scripts/probe_nki_versions.py`) — this is version skew, and it is OUR tech debt

The framing above is wrong, and so is the recommendation. It is not a capability split
between two peer packages; **it is two generations of NKI**, and the "missing" API was
deliberately removed.

| | `import nki` | `from neuronxcc import nki` |
|---|---|---|
| Version | **0.5.0**+28631259367.ga768afa6 | unknown (compiled `.so` bundled in neuronx-cc) |
| Form | standalone pure-Python package | compiled extension inside the compiler |
| `nl.arange` | **absent** | present |
| `nl.mgrid` | **absent** | present |
| `nl.ds` | present | present |
| `nl.load` / `store` / `affine_range` / `shared_hbm` / `silu` | present | present |

`nl.arange` and `nl.mgrid` were **removed** in NKI 0.5.0, with `nl.ds` slicing as the
replacement — and `nl.ds` exists in *both*. So there is a single forward-compatible way to
write these kernels, and top-level `nki` is the going-forward surface. Corroborated
independently by the Native PyTorch beta setup notes, which list `nl.arange` as
"Removed; use `nl.ds` slicing" for NKI 0.5.0.

**Two claims withdrawn:**

1. *"`hasattr(nl, 'arange')` returns True under the top-level package."* **False.** It returns
   `False`. That reading came from `scripts/probe_nki_api.py`, which imports
   `neuronxcc.nki.language` — the *old* path. I compared one probe's output against another
   probe's import and drew a conclusion about a package I hadn't actually queried. The
   "no import-time feature detection" argument goes with it.
2. *"Neither is a superset, so a multi-kernel repo genuinely needs both."* **False.** Our repo
   needs both only because two of our kernels use a removed API. Rewritten against `nl.ds`,
   all three would target top-level `nki`.

**What actually holds:** the `//`-on-shape-values failure (`math.trunc() is not supported for
scalar`) is a limitation of the **old** bundled path, not of 0.5.0. That is consistent with
version skew rather than a split — the newer package is simply better.

**Revised action — this is a work item for us, not an ask for the NKI team.**

Our RMSNorm and SiLU kernels use `nl.arange` + `mask=`, a removed idiom, which is why they
are pinned to the legacy compiled path. RoPE already uses the current API. So:

- Rewrite `kernels/neuron_rmsnorm` and `kernels/neuron_silu` to tile with `nl.ds` instead of
  `nl.arange` index tensors, and switch both to `import nki`.
- Then all three kernels target NKI 0.5.0 on one import path, and the per-kernel import
  pinning documented above disappears.
- Re-run all three suites; the execution-asserting harness will confirm nothing regressed.

Estimated effort: a few hours. It also removes a genuine liability — shipping PoC kernels
written against a removed API would be a poor recommendation to hand the kernels team.

**One thing still worth raising with the NKI team**, much narrower than the original ask: the
public NKI tutorials (and possibly the NKI Bootcamp reference kernels) teach the `nl.arange`
idiom, which no longer exists in 0.5.0. That is a docs-currency problem that will mislead every
new kernel author, and it is how we ended up writing two kernels against the old API.

---

*Original analysis retained below for the record.*

**Why this is nastier than it looks.** `hasattr(nl, "arange")` returns **True** under
the top-level package — the attribute exists, it just cannot be resolved during kernel
tracing. So there is no reliable feature-detection at import time; you find out at
compile time, per kernel. And the failure text (`failed to resolve name`) does not hint
that the sibling package would work.

It also means a project shipping several NKI kernels cannot standardise on one import
path — ours genuinely needs both, in the same repository. Any "port nki-library kernels
at scale" effort will hit this immediately, because nki-library source uses the
top-level `nki` while the tutorials use `neuronxcc.nki`.

**Recommendation.** Ask the NKI team which package is the supported long-term surface,
and whether the capability gaps are intentional or drift. If the top-level `nki` is the
future, `nl.arange` needs to resolve there; if `neuronxcc.nki` is, shape values need to
support integer division. Until then, kernel authors need a documented compatibility
table — this finding is the start of one.

## 15. Interception-point inventory: what `use_kernels=True` can actually reach [REFERENCE]

Counted from decorator occurrences in transformers `5.15.0.dev0`. This is the real
surface area available to a Neuron kernel, and it is worth stating precisely because
`_KERNEL_MAPPING` contains entries that **no model registers**.

**Reachable via `@use_kernel_forward_from_hub` (layer swap):**

| Kernel name | Occurrences | Notes |
|---|---|---|
| `RMSNorm` | **115** | the big one. Ported ✓ |
| `MultiScaleDeformableAttention` | 10 | detection models |
| `Qwen3_5GatedDeltaNet` | 4 | |
| `MegaBlocksMoeMLP` | 2 | MoE — relevant to Week 5 |
| `SiLU` | 1 | one decoration in `activations.py` covers every model using `ACT2FN["silu"]`. Ported ✓ |
| `GeLU`, `GeluTanh`, `NewGELU`, `FastGELU`, `QuickGELU` | 1 each | same leverage pattern as SiLU |
| `Llama4TextMoe` | 1 | |

**Reachable via `@use_kernel_func_from_hub` (function swap):**

| Kernel name | Model files | Notes |
|---|---|---|
| `rotary_pos_emb` | **95** | Ported ✓ |
| `ForCausalLMLoss` | — | mapped for cuda training |

**In `_KERNEL_MAPPING` but NOT registered by any model** — dead via the decorator path:
`SwiGLUMLP`, `GeGLUMLP`, `Linear`, `causal_conv1d_fn`, `causal_conv1d_update`.

`SwiGLUMLP` matters most, because a fused gate/up/SiLU/down MLP is where the real MLP
performance is (see `kernels/neuron_silu/__init__.py`). It is *not* reachable by
decorating a layer. Fused replacement goes through a separate, more invasive API —
`register_kernel_replacements_and_fusions()` / `make_parent_class_for_kernel_fusion()`,
driven by `KernelConfig`, which swaps the first named child for the kernel and replaces
its siblings with `nn.Identity()`. That is a different integration shape from the
per-layer forward swap this PoC validated, and it is what a fused NKI MLP kernel would
require.

**Leverage summary.** Three kernels (RMSNorm, RoPE, SiLU) cover the two highest-count
interception points plus the activation family. Because the activation decorations live
in `activations.py` rather than per-model, one activation kernel covers every model
using that `ACT2FN` entry. That is the strongest argument for the per-kernel
(rather than per-model) investment thesis — provided Finding #9's device-routing gap is
closed, since none of this surface is reachable on Neuron today.

**A third device-inference site.** `transformers/utils/kernel_config.py::infer_device()`
repeats the same `param.device.type` logic with a cuda/rocm refinement and no xla/neuron
handling. So the Finding #9 fix needs to be applied in three places to be complete:
`hub_kernels.kernelize`, `kernel_config.infer_device`, and `kernels._find_device`.

## 16. `nkilib` is already installed, and its kernels are directly callable [HIGH — changes the strategy]

The Week 2 porting analysis listed "use nki-library as a pip dependency" (Option D) as
blocked, assuming `nkilib` was neither available nor allowed. **Half of that was wrong.**

`nkilib` is already present in the Neuron venv as a normal site-packages install:

```
/opt/aws_neuronx_venv_pytorch_2_9/lib/python3.12/site-packages/nkilib/
```

Every production kernel imports cleanly (`scripts/probe_nkilib_bundled.py`):
`core.embeddings.rope_hf`, `core.mlp.mlp` (40 params, signature matching GitHub `main`
including recent additions like `gate_up_w_layout` and `dtype_mode`),
`core.moe.moe_cte.moe_cte`, `core.router_topk.router_topk`, `core.rmsnorm.rmsnorm_quant`.

**And the production kernel runs correctly when called directly from PyTorch/XLA.**
Verified on trn2 (`scripts/experiment_nkilib_thin_wrapper.py`) against the installed
`rope_hf`, the same kernel we hand-ported:

| Calling strategy | Result |
|---|---|
| pass preallocated `q_out`/`k_out`, read the **return value** | **q cos_sim 1.000001, k cos_sim 1.000000** |
| pass preallocated outputs, read the **mutated arguments** | cos_sim **0.000000** — never written |

Two things follow.

**(a) Destination-passing is vestigial across the XLA boundary.** The output tensors must
still be passed — they serve as shape/dtype templates — but results come back via the
return value, not by mutation. nki-library's own integration tests use
`"q_out.must_alias_input"`, which would lead a reader straight to the second strategy and
silently produce zeros. Worth documenting for anyone wrapping these kernels.

**(b) A thin-wrapper HF kernel is technically feasible today:**

```python
class NeuronRoPE(nn.Module):
    def forward(self, q, k, cos, sin, unsqueeze_dim=1):
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        return rope_hf(q, k, q_out, k_out, cos=cos, sin=sin)   # no vendoring
```

### Why this changes the recommendation

Week 2 concluded that mass-producing HF wrappers is not automatable because each kernel
needs defusion, interface adaptation, dependency inlining, and SPMD stripping. That
remains true for **self-contained** kernels. The thin-wrapper approach removes all four
steps at once.

The difference in scale is stark. RoPE needed ~15 lines inlined. The MLP kernel's
dependency closure is **7,249 lines across 22 files** (~480x), which makes hand-porting
the larger kernels impractical. Importing it is trivial.

So the ask to HuggingFace shrinks from "support a large porting program" to two small
changes:

1. Add `nkilib` to `kernels/python_depends.json` under the `neuron` backend — four lines,
   with `nki` already present as precedent and the exact JSON shape to copy.
2. Fix `_backend()` so the neuron table is consulted at all (Finding #12).

**The remaining blocker is policy, not code.** `python-depends` whitelists `nki` but not
`nkilib`, and the neuron table is unreachable regardless. A thin wrapper today would have
to under-declare its dependency and rely on `nkilib` happening to be preinstalled — the
same fragility that already applies to `nki`.

**Our hand-ported kernels remain the right choice for this PoC**: self-contained, no
undeclarable dependency, and they document the porting friction, which is the PoC's
deliverable. But they are not the shape to build an engineering program on, and the PoC
should say so plainly.

**Caveat we did not resolve:** nki-library's README warns that GitHub `main` is not
guaranteed compatible with a given compiler version. The installed copy here happens to
match `main`'s MLP signature, but a thin-wrapper strategy inherits a version-coupling
problem between the kernel repo, `nkilib`, and `neuronx-cc` that a vendored kernel does
not have. That tradeoff belongs in the recommendation.

## 17. Fused MLP is blocked by weight layout, and `kernelize()` has no hook for it [HIGH]

The profitable MLP unit is the fused gate/up/SiLU/down, not standalone SiLU (which is
memory-bandwidth bound). `nkilib.core.mlp.mlp` implements exactly that, and — unlike
RMSNorm — quantization and normalization are both **opt-in**, defaulting to `NONE` /
`NO_NORM`. Single-core works; SPMD is optional. Qwen3-8B and Qwen3-0.6B satisfy every
hard constraint on the BF16 path. So the kernel is not the problem.

**The problem is that all three HF MLP weights are transposed relative to what the kernel
wants**, and `kernelize()` provides nowhere to fix that.

| Tensor | HF `nn.Linear.weight` | NKI wants |
|--------|----------------------|-----------|
| `gate_proj` | `[I, H]` | `[H, I]` |
| `up_proj` | `[I, H]` | `[H, I]` |
| `down_proj` | `[H, I]` | `[I, H]` |

### CORRECTION (measured 2026-07-29, `scripts/spike_nkilib_mlp.py`)

The first version of this finding said: *"`.t()` is a free view in torch, but the kernel DMAs
from HBM assuming row-major layout, and non-contiguous tensor failures are a known live issue
on the Neuron beta — so a view cannot safely be passed. The transpose must be materialized."*

**That framing was wrong.** It was reasoned, not measured. What the measurement shows:

- The kernel **accepts** the result of a device-side `.t()` and produces correct output —
  `cos_sim 0.999989` at H=1024, I=3072. There is no non-contiguous rejection.
- On XLA, `is_contiguous()` returns **`True` even after `.t()`**. XLA normalizes layout, so
  `.t()` is not a stride view that can be passed around for free; it becomes a real transpose
  op in the graph, producing a real tensor.

The conclusion partly survives, but the reasoning and the confidence both change:

| Claim | Status |
|-------|--------|
| Non-contiguous transposed tensors are rejected | **withdrawn** — they are accepted |
| A stride-only view could avoid the transpose | **withdrawn** — XLA has no such thing here |
| The transposed tensor is real, so the memory cost is real | **stands** |
| `kernelize()` has no parameter-transformation hook | **stands**, unchanged |
| Per-forward transposing costs HBM traffic | **unverified** |

**The open question is narrower than first written.** Because `.t()` on XLA is a graph op, the
cost depends on whether XLA hoists it out of repeated forwards (paid once — in which case
"lazy transpose + cache" is nearly free and this finding is minor) or re-emits it every step
(in which case the per-forward cost is real and the finding is severe). `is_contiguous()`
cannot answer that; it needs a multi-step profile. Week 4 work.

Until then the table below lists *candidate* costs, not established ones. What **is**
established is the missing hook.

`kernelize()` rewrites `forward` pointers. It never transforms parameters. There is no
"on kernelize" or "on weight load" hook. Candidate approaches:

| Option | Possible cost |
|--------|---------------|
| Mutate parameters in place at kernelize time | `state_dict()` / `save_pretrained()` emit transposed weights that won't load into a stock `Qwen3MLP`. Silent checkpoint corruption. **This one is layout-independent and stands regardless of the profiling result.** |
| Keep a second transposed copy | ~2x MLP weight memory. MLP dominates Qwen3-8B, so near 2x model memory. |
| Transpose lazily on first forward, cache | Same memory cost plus a first-step stall. **May be the right answer** if XLA hoists. |
| Transpose every forward | Potentially three weight-sized HBM round trips per layer per step — unverified, see above |

**Practical note:** this finding is currently moot for real models anyway, because Finding #18
prevents the fused MLP from running single-core at any realistic `intermediate_size`. Resolve
#18 first; #17 only becomes actionable after it.

**Why this is a first-class finding rather than an implementation detail:** it blocks
*every* fused-kernel port on Neuron, not just the MLP. Any kernel whose weight layout
differs from `nn.Linear`'s hits it. The per-layer forward-swap model works cleanly for
weightless ops (RoPE, SiLU) and for ops that read weights as-is (RMSNorm), and breaks down
exactly when a kernel wants weights arranged differently — which is common for
matmul-heavy fused kernels, because that is where layout matters for performance.

**Recommendation.** This is a design decision for the kernels team, not something a PoC
should quietly pick. The two things worth asking upstream:

1. Does the `kernels` library want a weight-transformation hook (something like
   `prepare_weights(module)` called once at kernelize time), with a defined contract about
   whether `state_dict()` reflects original or kernel layout?
2. If not, is the intended pattern that Neuron kernels accept HF-native layouts and eat
   the transpose internally? That is a request to nki-library, not to HF, and would be the
   cleaner division of responsibility — the kernel already knows its own tiling.

The fusion API (`make_parent_class_for_kernel_fusion`) at least gives the chosen strategy
a place to live: siblings collapse to `nn.Identity()`, so the surviving module can hold
transposed weights. But it does not decide the question.

**Effort, for planning:** validating `nkilib.mlp()` standalone against its own
`mlp_torch_ref` on hardware is a 1-2 day spike and should precede any Kernel Hub work.
Landing it in HF via the fusion API, correct *and* faster, is 2-3 weeks. Note also that
nki-library's own MLP tests use `rtol=2e-2`, two orders of magnitude looser than the
`cos_sim > 0.999` bar we hold RMSNorm/RoPE to — decide which bar applies before starting,
because a fused three-matmul kernel will not be bit-identical.

## 18. The fused MLP kernel cannot run single-core when `intermediate_size > 4096` [MEASUREMENT STANDS — reframed by #26: this is a design boundary, not a bug]

> **Read #26 alongside this.** The boundary below is real and reproducible across ten data points.
> The *interpretation* was wrong. This section filed it as an nki-library divide-by-zero to fix; #26
> shows that `nkilib` kernels require a multi-core SPMD launch grid, and a `floordiv` by zero when
> `intermediate_size > 4096` is what a shard-count calculation looks like with no shard grid. The
> kernel is telling us single-core is not its execution model.
>
> Two consequences. The upstream ask changes from "fix the divide-by-zero" to "document the supported
> configuration and emit a better diagnostic". And the fused MLP is **not** blocked-but-promising: at
> `H=1024, I=3072` it compiles and runs fine (see the table below) and is still 2.99x slower on device
> than torch, because single-core it tiles badly and moves 2x the HBM traffic.

Found by the Week 4 derisking spike (`scripts/spike_nkilib_mlp.py`). This is a **sharp,
reproducible boundary**, and it excludes every model anyone would actually want to accelerate.

### The measurement

Calling `nkilib.core.mlp.mlp` single-core (no SPMD launch grid) from PyTorch/XLA, bf16,
B=1, S=128:

| hidden_size | intermediate_size | Result |
|---|---|---|
| 1024 | 3072 | pass — cos_sim 0.999995 |
| 1024 | **4096** | pass — cos_sim 0.999996 |
| 1024 | **5120** | **compile error** |
| 2048 | 4096 | pass — cos_sim 0.999988 |
| 2048 | 6144 | **compile error** |
| 4096 | 4096 | pass — cos_sim 0.999979 |
| 4096 | **4224** | **compile error** |
| 4096 | 4608 | **compile error** |
| 4096 | 5120 | **compile error** |
| 4096 | 8192 | **compile error** |
| 4096 | 12288 | **compile error** |

Ten data points across three `hidden_size` values: **passes iff `intermediate_size <= 4096`**.
Not a ratio effect — H=1024/I=5120 fails while H=4096/I=4096 passes. The boundary is between
4096 and 4224.

Varying sequence length (S=128, 256, 512) does not help. Neither does `force_cte_mode=True`
nor `mode=ComputationMode.PREFILL`.

### The error

```
error: 'floordiv' does not allow division by zero
  nkilib/core/utils/kernel_helpers.py:104   return (numerator + denominator - 1) // denominator
  nkilib/core/utils/tile_info.py:37         tile_count = get_ceil_quotient(tiled_dim_size, tile_size)
  nkilib/core/utils/tile_info.py:59         TiledDimInfo.build(...)
  nkilib/core/mlp/mlp_cte/mlp_cte_tile_info.py:236
                                            build_with_subtiling(bxs_dim_size, bxs_dim_tile_size,
                                                                 bxs_dim_subtile_size)
  nkilib/core/mlp/mlp_cte/mlp_cte.py:262    build_mlp_cte_tile_info(shard_mlp_params, ...)
  nkilib/core/mlp/mlp.py:340                mlp_cte(mlp_params, out, fused_add_out)
```

A tile/subtile size computes to zero, so the ceil-division blows up.

### Why (inferred, consistent with the data)

nki-library's CTE sharding heuristic in `mlp_cte_sharding.py` forces
`shard_on_inter = True` when `intermediate_size > 4096` — which is exactly our boundary. We
launch single-core, so there is no SPMD grid and no worker count to shard across; the
inter-sharding path then derives a zero subtile size.

In other words: **above I=4096 the kernel assumes it is being launched multi-core.** The
`> 4096` threshold in the heuristic and the `> 4096` failure boundary matching exactly is
strong circumstantial evidence, though we have not read the shipped `mlp_cte_sharding.py` to
confirm the mechanism directly.

### Why this matters more than Finding #17

| | Finding #17 (weight layout) | Finding #18 (this) |
|---|---|---|
| Nature | design question — how should weights be transformed | kernel limitation / bug |
| Who can fix | HF kernels team (a hook) or nki-library | nki-library only |
| Workaround from a wrapper | yes, several (all with costs) | **none** |
| Affects | all fused ports | all fused MLP at real sizes |

A wrapper can choose *some* weight-transformation strategy. A wrapper cannot make the kernel's
own tile arithmetic stop dividing by zero.

And every realistic model is on the wrong side of the line:

| Model | intermediate_size | Single-core fused MLP? |
|-------|-------------------|------------------------|
| Qwen3-0.6B | 3072 | yes (but a toy) |
| Qwen3-8B | 12288 | **no** |
| Llama-3-8B | 14336 | **no** |
| Mistral-7B | 14336 | **no** |

So the fused MLP is currently usable only at sizes nobody deploys.

### This is the general SPMD concern, now measured

Week 2's porting analysis listed "SPMD multi-core assumptions don't fit the per-layer swap
model" as a general worry. This is that worry with a number on it. For RoPE, stripping SPMD
was a clean, harmless reduction (`num_shards = 1`). For the MLP it is not optional — the
kernel's tiling *requires* the multi-core path at any useful size.

That is a meaningful update to the porting thesis: **SPMD-strippability is per-kernel, and for
the fused kernels it may not hold.** The HF Kernel Hub swaps one layer at a time on one core;
kernels written for a sharded inference runtime may simply not fit that model, independent of
weight layout or dependency packaging.

### What to do

1. **Report to the nki-library team.** A divide-by-zero in the kernel's own tile arithmetic on
   a legal, documented-as-supported input is a bug. `H % 128 == 0` is the documented
   constraint and I=12288 satisfies it; nothing documents an `I <= 4096` single-core limit.
   Ask whether single-core operation above I=4096 is intended to work, and if not, whether the
   constraint can be asserted clearly instead of failing inside tile math.
2. **Test whether an SPMD launch fixes it.** If the kernel works multi-core at I=12288, the
   question becomes whether an HF per-layer swap can legitimately launch SPMD — which is a
   architecture question for the kernels team, and a more interesting one than the weight
   layout. Not attempted here; it needs a launch-grid setup outside the per-layer swap model.
3. **Do not start the fusion-API integration.** Both #17 and #18 gate it, and #18 has no
   workaround. The 1-2 day spike was worth doing precisely because it surfaced this before
   2-3 weeks went into the integration.

### What the spike *did* establish (the positive result)

Worth stating separately, because it is real: **the production fused MLP kernel is drivable
directly from PyTorch/XLA with HF weights and produces correct results.**

| Config | dtype | cos_sim | max_diff |
|--------|-------|---------|----------|
| H=1024, I=3072 | fp32 | 0.999989 | 1.5e-02 |
| H=1024, I=3072 | bf16 | 0.999995 | 9.6e-03 |
| H=4096, I=4096 | bf16 | 0.999979 | 1.3e-02 |

No vendoring, no reimplementation — `from nkilib.core.mlp.mlp import mlp`, transpose the three
weights, take `[0]` off the returned list. That further supports Finding #16's thin-wrapper
thesis. `max_diff ~1e-2` is expected for a fused three-matmul and is consistent with
nki-library's own `rtol=2e-2` test tolerance; it is well outside the `cos_sim > 0.999`-plus-tight-`max_diff`
bar we hold RMSNorm and RoPE to, which is the tolerance question flagged earlier.

## 19. Eager-mode NKI dispatch costs ~0.36 ms of host time per call [HIGH]

Measured by `scripts/benchmark_kernels.py`. Two results: a performance finding, and a
methodological one about how *not* to measure these kernels.

### The measurement that survived

Host-side cost of issuing one kernel call, without waiting for the device (so this is
Python/dispatch/graph-construction time, not device execution):

| Path | Host enqueue cost | Reproducibility |
|------|-------------------|-----------------|
| NKI SiLU (`@nki.jit` via XLA) | **~0.36 ms/call** | 0.361, 0.372 ms across runs |
| eager `F.silu` | **~0.011 ms/call** | 0.011, 0.015 ms across runs |

**~25-33x more host time per NKI invocation.** What that costs a real model: Qwen3-8B has 36
layers, and our kernels are invoked 6× per layer (4 RMSNorm — input, post-attention, q_norm,
k_norm — plus 1 RoPE and 1 SiLU) plus a final norm = **217 calls per forward**. At ~0.35 ms
of extra host time each, that is **~76 ms of host-side overhead per forward step**.

That overhead is serial, fixed, and does not shrink with batch or sequence length. Unless it
overlaps with device execution it sets a floor on step time that no amount of kernel quality
can beat.

**Caveats, stated plainly:** this is an upper bound on the *serial* cost — some of the enqueue
work may overlap with device execution in a real model, and some of it may be one-time graph
construction that XLA caches across steps. It also says nothing about graph mode; these
kernels declare `can_torch_compile = False`. The Week 4 full-model MFU measurement is what
settles it.

### Why this makes fusion more important, not less

This is a finding about the **eager per-layer integration model**, not about the kernels. And
it is the strongest argument yet for the fused-kernel direction: one fused MLP call replaces
several dispatches, so fusion cuts launch count *as well as* memory traffic. A fused
gate/up/SiLU/down would remove 2 of the 6 per-layer calls and a fused
norm+MLP would remove 3.

So Findings #17 (weight-layout hook) and #18 (single-core `I > 4096` failure) get *more*
important on this evidence, not less. The per-layer swap model that the Kernel Hub is built
around is the model that suffers most from launch overhead.

### The methodological finding: per-layer microbenchmarking is the wrong instrument

The first version of this benchmark reported that every NKI kernel was 8-400x **slower** than
eager — RMSNorm 8x, SiLU 13x, RoPE 405x. **Those numbers were meaningless**, and the tell was
that latency did not vary with tensor size: RMSNorm measured 0.55 ms at both S=128 and
S=2048 (16x the data), while eager sat at 0.07 ms throughout. Timing that is independent of
problem size is not measuring the problem.

Two causes, both mine:

1. **Dead-code elimination.** The benchmark discarded every output. XLA is lazy, so at
   `mark_step()` there was no live result and the computation was never performed. It was
   timing an empty graph. Fixed by consuming outputs via `.sum().item()`.
2. **Overhead domination.** Whatever remained was per-call dispatch, which is fixed and
   therefore identical across shapes.

After fixing (1), latency does respond to size — but only weakly, around **1.1-1.3x for 8x
data**, which decomposes to roughly **90%+ fixed cost** at these shapes. So even the corrected
microbenchmark cannot resolve kernel quality: the signal is buried under per-call overhead on
both paths (and part of that fixed cost is the harness's own sync, which applies to both).

The script now **refuses to report NKI-vs-eager ratios** unless latency demonstrably scales
with problem size, and prints the suppression reason instead. On repeated runs the gate
sometimes passes (1.26x) and sometimes fails (1.12x) at the same shapes, which is itself
evidence that the measurement sits at the noise floor and should not be trusted for ratios.

**This is Finding #8 in a different costume**, and worth noting as a pattern rather than an
isolated slip. Both were measurements that produced confident, plausible-looking numbers while
not exercising the thing under test:

| | Finding #8 | Finding #19 (v1) |
|---|---|---|
| Symptom | `max_diff = 0.00e+00` — too perfect | latency independent of problem size |
| Cause | kernel never ran (CPU tensors → fallback) | computation never ran (output discarded → DCE) |
| Why it fooled us | fallback is numerically correct | overhead is plausible as latency |
| Guard now in place | call counter asserts NKI executed | scaling gate suppresses overhead-dominated results |

The general lesson for the PoC: **on a lazy-execution accelerator backend, both correctness
and performance measurements fail silently by default.** Every measurement needs a check that
it actually exercised the thing being measured — a call counter for correctness, a scaling
check for performance. Neither is standard practice, and both cost us a cycle.

### What to do

- Do **not** quote per-layer NKI-vs-eager latency ratios from this project. The instrument
  can't support them.
- Week 4 MFU on a full model is the measurement that decides whether the kernels help, and it
  naturally accounts for launch overhead in the way a customer would experience it.
- When reporting MFU, report launch count too. If step time is launch-bound, MFU alone will
  look like a kernel-quality problem when it is an integration-model problem.
- Worth asking the HF kernels team whether per-call dispatch cost is a known concern for
  non-CUDA backends, and whether there is a batching or persistence mechanism.

## 20. Every `@nki.jit` invocation from eager PyTorch/XLA costs ~53 ms, independent of problem size [MEASUREMENT STANDS — mechanism SUPERSEDED by #24]

> **Read #24 first.** The measurements below are correct and reproduced five times. The
> *explanation* is wrong. The cost is an uncached `neuron-ls` subprocess forked on every
> invocation inside NKI's dispatch path, not a graph-transition or NEFF-switching cost. Caching
> the target detection removes 102x of it with no accuracy change, which takes the model-level
> regression from 208x slower to 3.4x slower. Everything below about *ruling out* problem size,
> interleaving, recompilation, and our-kernels-vs-production remains valid and is what eventually
> made #24 findable. The section "Why this happens, and the structural point" is superseded.

**This is the finding the recommendation turns on.** The mechanism works and the kernels are
correct, but in eager mode a per-layer NKI swap cannot be performance-competitive on this
stack — not because of kernel quality, but because each invocation carries a fixed cost larger
than an entire baseline forward pass.

### The measurement

Qwen3-0.6B (28 layers, full depth), seq 512, bf16, forward only, single logical NeuronCore:

| Configuration | Step time | MFU (per logical core) | NKI calls/step |
|---|---|---|---|
| baseline, no kernels | **41.95 ms** | **5.06 %** | 0 |
| NKI SiLU only | 1,495.54 ms | 0.14 % | 28 |
| NKI RMSNorm + RoPE + SiLU | **8,753.65 ms** | **0.02 %** | 169 |

Steady state, not a compile artifact: zero compilations during the timed loop, and step time
stable to within 0.2% (IQR 8746–8764 ms).

Per-call added cost: **51.9 ms** (SiLU only) and **51.6 ms** (all three). Uniform.

### It is a fixed cost per call, not per unit of work

`scripts/experiment_nki_graph_break.py`, single NKI SiLU call, output consumed:

| rows | tiles | NKI | torch `F.silu` | ratio |
|------|-------|-----|----------------|-------|
| 128 | 1 | 54.57 ms | 0.250 ms | 218x |
| 256 | 2 | 53.38 ms | 0.250 ms | 213x |
| 512 | 4 | 52.72 ms | 0.269 ms | 196x |
| 1024 | 8 | 53.52 ms | 0.588 ms | 91x |
| 4096 | 32 | 53.86 ms | 0.304 ms | 177x |
| 14336 | 112 | 53.75 ms | 0.501 ms | 107x |

**52.7 – 54.6 ms across a 112x range in problem size.** Flat. The compute is negligible against
the fixed cost — 14336×3072 bf16 in and out is ~176 MB of traffic, which at Trn2 HBM bandwidth
is well under a millisecond.

Additional structure, same script:

| Variant | Result | Reading |
|---|---|---|
| A: 28 NKI calls back to back | 1451.8 ms (51.9 /call) | cost is per call |
| B: 28 NKI calls, torch op between each | 1451–1478 ms (51.6 /call) | **interleaving is NOT the cause** — A ≈ B |
| C: 28 torch `F.silu` calls | 0.76 ms (0.03 /call) | control |
| D: 1 NKI call on 28x the data | 52.97 ms | one big call ≈ one small call |
| E: 1 NKI call, base shape | 52.03 ms | **D/E = 1.02x for 28x the data** |

A and B being equal rules out the graph-break-from-interleaving hypothesis: adjacent NKI calls
are just as expensive as ones separated by framework ops. And note A contains a single
`mark_step`, so the 28 × 52 ms is incurred *inside one graph execution* — it is device-side, not
host-side synchronization.

### It is not our kernels, and it is not host dispatch

- **Not our kernels.** nki-library's production `rope_hf` shows the same ~52 ms per call in the
  same run. All three kernels — two hand-written, one a production port — land on the same
  number.
- **Not host dispatch.** Finding #19 measured host-side enqueue at ~0.36 ms/call. That is 1/145
  of this. Finding #19's conclusion was directionally right (eager NKI is invocation-bound) but
  understated the magnitude by two orders of magnitude, because it measured only the host side.
- **Not compilation.** Zero compiles during the timed loop.

### This retroactively vindicates the validity gate

`scripts/benchmark_kernels.py` earlier measured a single SiLU NKI call at ~0.78 ms and its
validity gate **suppressed the result** because latency did not scale with problem size. That
suppression was correct: the true figure is ~53 ms, and 0.78 ms was measuring an
eliminated computation. Had the gate not been there, the PoC would have carried a number that
was wrong by 68x — in the flattering direction.

Worth recording as evidence the guard earns its keep, not just as a process note.

### Why this happens, and the structural point

> **SUPERSEDED by #24.** The paragraph immediately below guessed at NEFF setup and HBM
> round-tripping, and explicitly flagged that it had not been profiled. Profiling showed the
> device executes a 28-call NEFF in 0.609 ms at 43% memory-bandwidth utilisation, so the device
> side was never the problem. The real cause is a per-call `neuron-ls` subprocess on the host.
> Kept here unedited because the reasoning that follows it — the amortisation argument — turns out
> to be *right for the wrong reason*, and #24 re-derives it from the corrected mechanism.

The cost behaves like a fixed per-invocation charge for entering and leaving a NKI custom call
from the framework graph — plausibly NEFF setup plus HBM round-tripping of inputs and outputs,
though we have not profiled to attribute it precisely.

The structural consequence matters more than the mechanism:

**nki-library's kernels are designed as large fused megakernels, and the HF Kernel Hub's
per-layer forward swap is the exact opposite shape.** A fused kernel amortizes the invocation
cost across a whole transformer block; a per-layer swap pays it 6 times per layer. This is the
same mismatch Findings #17 and #18 found from the weight-layout and sharding directions,
now visible from the cost direction — and it is the most quantitative form of it.

Arithmetic that makes the point: at ~53 ms per invocation, the *entire* 42 ms baseline forward
pass is cheaper than one NKI call. So in eager mode, on this stack, **any** per-layer NKI swap
loses, and swapping more layers loses harder. Even a perfectly fused one-call-per-layer kernel
would cost 28 × 53 ms = 1.5 s/step against a 42 ms baseline.

### What would change the answer

1. **Graph mode / `torch.compile`.** All three kernels declare `can_torch_compile = False`. If
   NKI kernels can be embedded in a compiled graph so the per-invocation cost is paid at compile
   time rather than per call, the entire picture changes. **This is now the most important open
   question in the project**, and it is more important than any remaining kernel work.
2. **Confirmation that ~53 ms is not expected.** It is large enough to look like a
   misconfiguration rather than a design point. Worth asking the NKI team directly: is this the
   expected cost of invoking a NKI kernel from eager torch-xla on SDK 2.31 / NKI 0.5.0, or a
   known issue? If it is expected, that is a strong statement about eager NKI in general, well
   beyond this PoC.
3. **One kernel per model, not per layer.** A single NKI call covering the whole forward would
   amortize the cost — but that is a megakernel, i.e. not the Kernel Hub model at all.

### What this does not invalidate

Worth stating clearly, because the correctness work stands independently:

- All three kernels are numerically correct, execution-verified, with negative controls.
- The Kernel Hub interception mechanism works on Neuron: layer swap, function swap, graceful
  fallback, 115 + 95 upstream registration points reachable.
- Findings #9, #12, #14, #16 (device routing, dependency allowlist, version skew, thin-wrapper
  feasibility) are all unaffected — they are about whether the mechanism can be *reached*, not
  how fast it runs.

The PoC's question was "should Neuron invest in first-class HF Kernel Hub support?" This finding
does not answer no. It relocates the answer: **the eager per-layer path is not the one to
invest in, and the compile path is now the question that matters.**

## 21. ANSWERED: graph batching already happens, and it does not help [CRITICAL — conclusion stands, one sub-claim corrected by #24]

> **One correction.** The original title of this section ended "...the ~52 ms is inside the
> compiled NEFF." That last part is wrong: the device executes the whole 28-call NEFF in 0.609 ms.
> The cost is on the host, before `mark_step`. See #24. The section's main conclusion — that graph
> batching is already happening and cannot recover the cost — is unaffected and is in fact
> *strengthened*: host-side per-call work is exactly the kind of thing graph batching cannot touch.
> The reasoning below is left intact, since walking from "it must be in the NEFF" to "it is a
> subprocess on the host" is the useful part.

Finding #20 makes one question decisive: does graph mode amortize the ~53 ms per-invocation NKI
cost? If yes, the recommendation is "build on the compile path". If no, per-layer NKI swapping
is not viable in any mode.

**Answered: no. And answering it did not require `torch.compile` at all.**

### The framing was wrong, and that is why it looked blocked

This finding originally read "the decisive question cannot be answered on this stack, because
`torch.compile` is broken here." That was wrong twice over, and both errors are the kind a
reviewer would catch, so they are recorded rather than quietly overwritten.

First, **`torch.compile` was never the right instrument.** torch-xla is *already* a lazy graph
runtime: operations accumulate into an HLO graph and compile/execute at `mark_step()`. Asking
"would graph mode help" while running on torch-xla asks a question the runtime already answers.
`torch.compile` would change *when* tracing happens, not whether the NKI calls land in one graph.

Second, **`torch.compile` is not broken here.** `scripts/diagnose_torch_compile.py` shows `torch`
2.9.1 and `torch_xla` 2.9.0 are a matched pair, `openxla` is registered, and `add` / `mul` /
`relu` all compile and run correctly on XLA tensors. Only specific ops fail, for a specific and
fixable reason — see Finding #23. The original conclusion stopped one level short of a real bug.

### The measurement that answers it

`scripts/probe_neff_count.py` counts real device executions using torch-xla's own counters
(`ExecuteTime` gets one sample per graph launch), so batching is directly observed rather than
inferred. 28 NKI calls issued before a single `mark_step`; steady state, `compiles = 0` on every
variant, three samples each agreeing within 0.1%:

| variant | wall | device executions | per call |
|---------|------|-------------------|----------|
| A. 28 NKI calls, 1 `mark_step` | 1446.37 ms | **1** | 51.66 ms |
| B. 1 NKI call, 1 `mark_step` | 52.80 ms | 1 | 52.80 ms |
| C. 28 torch ops, 1 `mark_step` | 1.23 ms | 1 | 0.04 ms |
| D. 1 torch op, 1 `mark_step` | 0.25 ms | 1 | 0.25 ms |

Three things follow, all directly observed rather than argued:

1. **The 28 NKI calls DID share one graph and one device execution.** `execs(total) = 1`, and
   `TensorsGraphSize` reports a 196-node graph (28 x 7 nodes). This validates the inference drawn
   from variant A of `experiment_nki_graph_break.py`, which had been asserted, not checked.
2. **NKI calls do not self-synchronise.** `execs(pre-sync) = 0` on every variant: the counter had
   not advanced before `mark_step`. No `@nki.jit` call is secretly flushing the lazy graph. This
   was the one hole that could have invalidated the entire line of reasoning.
3. **Cost is linear in the number of NKI custom calls, inside a single execution.** A/B is
   **27.39x for 28x the calls**. The control is sublinear (C/D = 4.9x for 28x), which proves the
   harness can see batching when batching works.

### The control turned out stronger than it was designed to be

Variant C uses `F.silu`, which on Neuron is *itself* lowered to an XLA user computation by
`torch_neuronx` (visible in the traceback in Finding #23: `Silu.forward_impl` ->
`_xla_user_computation`). So C is 28 XLA custom calls plus 28 multiplies, and it costs 1.23 ms.

**28 XLA custom calls: 1.23 ms. 28 NKI custom calls: 1446 ms.** So the cost is not "custom calls
don't fuse" and not "XLA can't batch opaque nodes". It is specific to how NKI kernels are lowered
and scheduled inside a NEFF.

### What this means

Graph mode is *already applied*. The calls are already in one HLO module and one device execution,
and the cost is still paid per call. `torch.compile` cannot recover it, because there is no
batching left to do at the framework level — it already happened.

That relocates the problem. The ~52 ms is paid per NKI custom call, independent of tile size
(Finding #20 variant D: 28x the data in one call costs 1.02x). Candidate explanations, in
descending plausibility *as they stood at this point*:

1. Each NKI custom call is lowered as its own schedulable unit the runtime switches between —
   pipeline drain, HBM round-trip for inputs and outputs, possibly a NEFF-region switch.
2. A fixed synchronisation or barrier is emitted per custom call.
3. Real device compute time. Implausible at this magnitude: SiLU on `[512, 3072]` bf16 touches
   ~3 MB, order microseconds at HBM bandwidth, and the cost is flat across a 112x range of
   problem sizes.

> **All three were wrong, and the list itself was the mistake.** Every candidate is device-side,
> because the conclusion "one device execution still costs 28x" had been read as "therefore the
> cost is in that execution." The unexamined alternative was that the cost never reached the
> device at all. Profiling the NEFF (0.609 ms device time against 1459 ms wall) ruled out all
> three at once and pointed at the host. See #24.
>
> This is worth keeping visible: the error was not picking the wrong item from the list, it was
> not noticing the list was missing an option. Enumerating candidates within one framing feels
> like rigour and is not.

Distinguishing these needs a device profile, not another framework-level experiment. Either way
the owner is the compiler/runtime, not the Kernel Hub integration and not the kernels.

### Consequence for the recommendation

The PoC's previous top ask — "get us onto a stack where `torch.compile` works, then re-run" — is
**withdrawn**. It would not have told us anything, because the batching it would have produced is
already being produced. The replacement ask is narrower and answerable by the NKI/compiler team:
*why does each NKI custom call inside a single NEFF cost ~52 ms, and is that reducible?*

### A harness bug worth recording, because it would have produced a false finding

The first version of this experiment *did* report a NKI-specific failure:

```
torch._dynamo.exc.InternalTorchDynamoError: ModuleNotFoundError: No module named 'neuron_silu'
```

That is entirely an artifact of our own test harness. `load_kernel_module()` loads kernels via
`importlib.util.spec_from_file_location` (necessary because our `kernels/` directory shadows the
`kernels` pip package) but never registered the result in `sys.modules`. Dynamo re-imports a
traced function's defining module by name, so it could not find it.

Read naively, that error says "NKI kernels don't survive torch.compile". It says nothing of the
kind. Fixed by registering the module in `sys.modules` in `load_kernel_module()`.

This is the third time in this project that a plausible-looking measurement turned out to be an
artifact of the measurement itself (Findings #8, #19, and now this). The pattern is consistent
enough to be worth stating as a conclusion rather than an anecdote — see the Week 6 document.

### Eager cost reproduced a fourth time

Incidental but useful: the eager NKI per-call cost reproduced again at **52.09 ms/call**
(8 calls, 416.71 ms). Across four independent measurements — SiLU-only in a model (51.9),
all three kernels in a model (51.6), the back-to-back experiment (51.85), and this control
(52.09) — the figure is stable to within 1%. Finding #20 is not a fluke.

### What to do

1. **Do not spend effort on getting `torch.compile` working for this question.** Answered above:
   the graph batching it would provide is already happening. (`torch.compile` is still worth
   fixing for its own sake — see Finding #23 — just not as the route to this answer.)
2. **Ask the NKI / compiler team the narrowed question:** why does each NKI custom call inside a
   single NEFF cost ~52 ms independent of tile size, and is it reducible? Include the differential
   that makes it sharp: 28 `torch_neuronx` XLA user computations in the same position cost 1.23 ms
   total, while 28 NKI custom calls cost 1446 ms.
3. **Profile one NEFF containing N NKI calls** to separate "per-custom-call scheduling overhead"
   from "emitted barrier" from "real device time". This is now the only open technical question
   behind Finding #20.
4. **Do not soften Finding #20.** On the stack a customer would use today, eager per-layer NKI
   swapping is not performance-viable, and the most plausible escape hatch has now been measured
   and ruled out rather than left as an open hope.

## 22. Qwen3-MoE does not run on Neuron with transformers' default experts implementation [HIGH — customer-facing]

Unrelated to NKI kernels, and it blocks the model entirely.

`Qwen3MoeExperts.forward` dispatches through `transformers/integrations/moe.py`, and the
default implementation `grouped_mm_experts_forward` calls `torch.sort` and `torch.histc`.
`histc` lowers to a `sort` HLO, which the Neuron compiler rejects:

```
RuntimeError: RunNeuronCCImpl: [ERROR] [NCC_EVRF029]
  Operation sort is not supported on trn2. Use supported equivalent operation like TopK
  or replace it with an alternate implementation via Neuron Kernel Interface (NKI).
  %sort.0 = u32[256]{0} sort(%reshape.371), dimensions={0}, ...
```

This fires on a **plain forward pass with no kernelization at all**, so it is a property of
stock transformers on Neuron.

### The workaround, and it is a one-liner

transformers exposes four experts implementations via `config._experts_implementation`.
Probed on trn2 (`tests/test_qwen3_moe_e2e.py`):

| `experts_implementation` | Runs on Neuron? | Why |
|---|---|---|
| default (`grouped_mm`) | **no** | `torch.sort` + `torch.histc` → unsupported `sort` HLO |
| `batched_mm` | **yes** | no `sort`/`histc`/`nonzero`/`unique`/`bincount` in its path |
| `deepgemm` | not reached | probe short-circuits on first success |
| `sonicmoe` | not reached | " |

So on Neuron:

```python
config = Qwen3MoeConfig(..., experts_implementation="batched_mm")
```

Nothing documents this. A customer trying Qwen3-MoE on Trainium gets a compiler error naming
an HLO op, with no indication that a config flag fixes it. Worth surfacing in Neuron's
model-support docs regardless of what happens with the Kernel Hub.

### With that set, all three dense kernels transfer to MoE unchanged

Week 5's "reuse RMSNorm/RoPE/SiLU" goal, measured rather than assumed. Qwen3-MoE, 2 layers,
4 experts, top-k 2, seq 128:

| Kernel | Dispatch | Expected |
|--------|----------|----------|
| RMSNorm | `nki=9 fallback=0` | 9 = 4/layer + final norm — same structure as dense |
| RoPE | `nki=2 fallback=0` | 2 = 1/layer |
| SiLU | `nki=2 fallback=0` | discovered, not predicted |

Logits `cos_sim = 1.000002` against the unkernelized model. **Zero code changes to the
kernels.** The interception points are shared with the dense model, so the per-kernel
investment does carry across model families — which is the load-bearing claim behind the
"per-kernel, not per-model" thesis, now demonstrated on a second architecture.

### It also reframes which MoE kernel is worth writing

The gap analysis assumed the valuable MoE NKI kernel was the blockwise expert matmul
(`nkilib/core/moe/moe_cte`). This finding suggests otherwise: the thing actually blocking
Qwen3-MoE on Neuron is a **routing histogram**, and the compiler error itself recommends NKI
as the remedy ("replace it with an alternate implementation via NKI").

So a small NKI kernel for the `sort`/`histc` step — or wiring up `nkilib/core/router_topk` and
`core/topk` — would unblock the default MoE path on Neuron entirely. That is a much smaller,
better-scoped piece of work than porting the expert matmul, and unlike the expert matmul it is
not blocked by Findings #17 or #18.

Best MoE-related next step, and it is a genuinely new recommendation from this session.

---

## 23. `torch_neuronx`'s op overrides are not fake-tensor safe, which breaks `torch.compile` on essentially every transformer [HIGH — outside this project's scope, clear owner]

Found while dismantling the original Finding #21. It is unrelated to NKI and to the Kernel Hub,
but it is a concrete upstream bug with a one-line root cause and a large blast radius, so it is
recorded here for routing rather than dropped.

### The symptom

```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
call_function <function silu ...>(*(FakeTensor(..., device='xla:0', size=(64, 64)),), **{}):
got RuntimeError('Expected all tensors in the given list to be XLA tensors.
Element at index 0 is not an XLA tensor. Got: XLAFloatType')
```

### It is not what it looks like

`torch.compile` works fine on this stack. Measured with `scripts/diagnose_torch_compile.py`:

| case | CPU tensors | XLA tensors |
|------|-------------|-------------|
| `add` | OK | OK |
| `mul` | OK | OK |
| `relu` | OK | OK |
| `silu` | OK | **fails** |

`torch` 2.9.1 / `torch_xla` 2.9.0 are a matched pair, and `openxla` is present in
`torch._dynamo.list_backends()`. So this is per-op, not per-stack.

### Root cause

`torch_neuronx` replaces a set of ATen ops with hand-written XLA user computations. The dispatch
predicate is in `torch_neuronx/xla_impl/base.py`:

```python
def wrapper(*args, **kwargs):
    if any(is_xla_tensor(it) or is_xla_device(it)
           for it in chain(args, kwargs.values())):
        return custom_call_cls.apply(*args, **kwargs)   # -> _xla_user_computation
    return func(*args, **kwargs)
```

A `FakeTensor` carrying `device='xla:0'` satisfies that predicate. It is routed into
`torch_xla._XLAC._xla_user_computation`, which requires *real* XLA tensors and rejects it. There is
no meta/abstract implementation for these overrides, so Dynamo cannot trace through them at all.

The predicate needs to exclude fake/meta tensors (and the overrides need abstract impls) for these
ops to be traceable.

### Blast radius

The overridden ops, from `torch_neuronx/xla_impl/ops.py`:

`gelu`, `silu`, `randn`, `CrossEntropyLoss`, `Dropout`, `Embedding`, `clip_grad_norm_`, `argmax`,
`Softmax`, `topk`, `upsample_nearest2d`

That list contains `Embedding`, `Softmax` and `CrossEntropyLoss`. Every transformer forward
contains an embedding and a softmax; every training step contains a loss. So `torch.compile` on a
transformer on this stack will hit this on essentially any model, not as an edge case.

This is consistent with, and may partly explain, the TTFI and torch.compile-coverage difficulties
reported by other teams — though that connection is inference, not something measured here.

### Workaround that does work

`torch_xla.compile()` succeeds where `torch.compile(backend="openxla")` fails, because it wraps
lazy-tensor execution with explicit graph boundaries instead of routing through Dynamo and fake
tensors. Verified: `torch_xla.compile(f)` on `F.silu` returned the same value as eager
(`sum = 902.0360` both ways).

### Why it is in this document but not in the recommendation

It is not a Kernel Hub issue, not a NKI issue, and fixing it would not change any performance
result in this PoC (Finding #21 shows graph batching already happens and does not help). It is
filed here so it can be routed to whoever owns `torch_neuronx`, with the reproducer in
`scripts/diagnose_torch_compile.py`.

---

## 24. The 208x regression was an uncached `neuron-ls` subprocess per kernel invocation [CRITICAL — supersedes the mechanism in #20 and #21]

Findings #20 and #21 correctly measured a fixed ~52 ms cost per `@nki.jit` invocation and
correctly ruled out problem size, interleaving, recompilation, our kernels, and graph batching.
They attributed it to the wrong thing. This finding has the actual cause, a verified fix, and the
re-measurement.

### Root cause

`nki/framework/compiled.py::_compile_opts()` calls `resolve_target()` on **every** invocation:

```python
def _compile_opts(self):
    opts = CompileOptions(
        target=resolve_target(self.func, self.target),   # <-- every call
        lnc=self.lnc,
        ...
```

With `NEURON_PLATFORM_TARGET_OVERRIDE` unset and no explicit target, `resolve_target()` falls
through to `nki/compiler/target.py::_detect_target()`:

```python
def _detect_target() -> str:
    if shutil.which("neuron-ls") is None:
        return "trn3"
    out = subprocess.check_output(["neuron-ls"], text=True, timeout=10, stderr=subprocess.PIPE)
    for line in out.splitlines():
        if line.startswith("instance-type:"):
            ...
```

It forks a process and runs `neuron-ls` to ask the hardware what it is. That costs ~52 ms.

**The compile cache cannot help.** NKI does maintain `self.func._nki_compile_cache`, but
`CompileOptions` is what identifies a compiled kernel, so target resolution happens while
*building the cache key*. A cache **hit** still pays the subprocess in full.

That explains every previously puzzling property at once: the cost is fixed because forking
`neuron-ls` does not depend on tensor size (flat across a 112x sweep); it is per call because
nothing caches it; it is invisible to graph batching because it happens on the host before
anything reaches the graph; and it is identical for our kernels and production `nkilib` ones
because it is in the shared dispatch path.

### How it was localised

Each step is a separate script, and each one narrows the search:

| step | script | result |
|------|--------|--------|
| are N calls one graph? | `probe_neff_count.py` | 28 calls -> **1** graph, **1** device execution, 196-node graph, `execs(pre-sync) = 0` |
| what does the device do? | `profile_nki_call_cost.py` + neuron-explorer | NEFF `total_time` **0.609 ms**, 43% MBU, 95% active, `activate_instruction_count = 112` (28x4) |
| host or device? | `probe_where_is_the_time.py` | wall 1459 ms, **99.9% before `mark_step`**; ExecuteTime 0.92 ms, LazyTracing 0.28 ms, TransferToDevice 0, CompileTime 0 |
| which function? | `probe_inside_one_call.py` | cProfile: 51 of 52 ms in `select.poll` under `subprocess.check_output` under `_detect_target` |

The decisive step was the third: a 2400x gap between device time and wall time eliminates every
device-side explanation simultaneously, regardless of which one you favour.

The arithmetic closes independently: 169 NKI calls/step x 51.8 ms = 8754 ms, against 8753.65
ms/step measured by `measure_mfu.py`.

### The fix, verified two ways

`scripts/probe_target_override_fix.py` tests both in one process, re-runs the baseline last as a
control, and checks accuracy on every variant:

| variant | per call | speedup | cos_sim |
|---------|----------|---------|---------|
| baseline (no override) | 51.74 ms | — | 0.999938 |
| Fix A: `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` | 0.50 ms | **102.8x** | 0.999938 |
| Fix B: `lru_cache(_detect_target)` | 0.49 ms | **105.5x** | 0.999938 |
| baseline again (control) | 51.43 ms | — | 0.999938 |

Two deliberate design choices in that test. The override is set to whatever `_detect_target()`
returns *on this host*, never a hardcoded string — a wrong target would compile for the wrong
hardware, which could be silently wrong rather than an error. And accuracy is asserted on every
variant, because a faster wrong answer is a bug, not a fix. Cosine similarity is identical to six
decimal places across all four, so neither fix changes what gets compiled.

Fix A is a customer-side workaround available today. Fix B is what an upstream fix plausibly looks
like: one decorator, no user action, no API change.

### Re-measured MFU

`measure_mfu.py --fix-target-detection`. Qwen3-0.6B, 28 layers, seq 512, batch 1, forward only,
single logical core, denominator 632/2 = 316 TFLOPS:

| | step time | MFU | vs baseline |
|---|---|---|---|
| baseline | 42.04 ms | 5.05% | — |
| kernelized, before fix | 8753.65 ms | 0.02% | 208x slower |
| kernelized, after fix | **141.43 ms** | **1.50%** | **3.4x slower** |

169 NKI launches, zero fallbacks, IQRs non-overlapping.

**The fix recovers 62x, and the kernels are still a 3.4x net loss.** The headline is not
"it was just a bug."

### What remains: a second, smaller instance of the same class of bug

Remaining added cost is `141.43 - 42.04 = 99.4 ms` over 169 calls = **0.588 ms/call**, against
0.02 ms/call of device time. `probe_inside_one_call.py --fix-target-detection` profiles it:

```
nki/framework/_torch_xla.py:138        __call__
torch_xla/core/xla_op_registry.py:24   __call__
torch_xla/core/xla_builder.py:817      create_computation
torch_neuronx/pyhlo/scribe.py:606      __init__      x56
protobuf/internal/enum_type_wrapper    __getattr__   x168
```

Every invocation rebuilds the XLA computation and its HLO protobufs from scratch, on a warm path
where the kernel has already run several times. Same shape of problem as the subprocess — per-call
work that is cacheable per `(kernel, shape, dtype)` — two orders of magnitude smaller. A plain
torch op in the same conditions costs 0.02–0.03 ms, so **NKI eager dispatch is still ~15–20x a
torch op's**, and it is dominated by protobuf construction rather than anything architectural.

Whether *this* is also fixable was not tested. It is a larger intervention than one decorator and
sits inside `torch_xla`'s op-registry path, so it is filed as a question for the NKI team, not a
claim.

### The residual is near-fixed per call, so it amortises — measured, not assumed

If the residual is fixed per call, the relative penalty must shrink as work per call grows. That is
testable without changing anything: NKI call count is set by model depth (169 for Qwen3-0.6B), so
raising sequence length adds work per call while holding call count constant.
`scripts/compare_mfu_runs.py` compares the two runs:

| run | baseline | kernelized | MFU base | MFU kern | penalty | added/call |
|-----|----------|------------|----------|----------|---------|------------|
| seq 512 | 42.04 ms | 141.43 ms | 5.05% | 1.50% | 3.36x | 0.588 ms |
| seq 2048 | 108.76 ms | 223.99 ms | 9.90% | **4.81%** | **2.06x** | 0.682 ms |

Baseline work grew 2.59x; added cost per call grew only **1.16x**. So the overhead is near-fixed per
call and the penalty nearly halves, from 3.36x to 2.06x. Kernelized MFU at seq 2048 (4.81%) is
approaching the *baseline* MFU at seq 512 (5.05%).

Two honest qualifications. The 1.16x growth is not 1.0x, so roughly 16% of the residual does scale
with problem size — it should be described as *near*-fixed, not fixed. And extrapolating: at
0.682 ms/call over 169 calls, a step needs ~1150 ms of real work for the overhead to fall below 10%,
which is about 10x more than seq 2048 on a 0.6B model. Reachable with a larger model and longer
sequences, but it means per-layer swapping only approaches parity at production scale.

Parity is also not the goal. Reaching it would only mean the kernels stop *costing* anything; a
speedup additionally requires the kernels to beat the torch ops they replace, which this PoC has
not demonstrated for any of the three.

### This reconciles Finding #19

Finding #19 recorded ~0.36 ms of host dispatch per call, which appeared to contradict 52 ms. Both
are real and they measure different things: **0.36–0.59 ms is the dispatch floor, and the 52 ms was
the subprocess stacked on top of it.** #19 was measuring the residual all along. Neither figure
needs retracting; #19's *conclusion* (that per-layer microbenchmarking mispredicts in-model cost)
also still holds, for the reason given there.

### Break-even, from measured numbers

A swapped kernel is a net win only if it saves more than ~0.59 ms of torch time per call. Torch
SiLU on `[512, 3072]` bf16 costs 0.02–0.04 ms. These ops are **15–30x underwater**. Winning
requires either dispatch cost at torch-op levels, or kernels that replace far more work per call.

The second option means fused kernels — which is exactly what `nkilib` ships, and exactly what
Findings #17 (weight layout) and #18 (single-core width limit) say the Kernel Hub cannot currently
express. So the PoC's central thesis survives with a corrected and much sharper mechanism: the
mismatch is not that NKI cannot fuse into the XLA graph (it demonstrably does — one graph, one
execution, 43% MBU), it is that **NKI's eager per-call dispatch cost is too high to amortise over
one small layer, so the granularity that wins is the granularity the Kernel Hub cannot express.**

### The methodological point, which is the most transferable output here

The graph-transition hypothesis in #20/#21 was wrong, and it survived four separate experiments:
varying interleaving, varying data volume, ruling out recompilation, and swapping our kernels for
production ones. Every one of those came back consistent with it.

It survived because **every one of those experiments measured wall-clock time at the framework
level, and none of them could see inside the 52 ms.** More variants of the same instrument would
never have falsified it. What falsified it was changing instrument: a device profile, then a
Python profile.

A hypothesis that keeps surviving tests is not necessarily right. It may only be untestable by the
instrument in use. When a hypothesis has survived several tests and the story still does not close,
the next move is a different *kind* of measurement, not another variant of the same one.

---

## 25. Each NKI call is an optimisation barrier: the compiler cannot fuse across it, and for memory-bound ops fusion is the whole optimisation [CRITICAL — the real structural limit]

Every performance finding before this one measured **dispatch** cost: ~52 ms/call from an uncached
subprocess (#24, fixed), ~0.59 ms/call residual from rebuilding the XLA computation (#24, open),
0.02 ms/call of device time. None of them said whether the kernels are any *good*.

That gap mattered, because the two possibilities point opposite ways. If NKI is faster than torch on
device, dispatch is the only thing between here and a win and Fix 7 is the whole ballgame. If NKI is
slower on device, fixing dispatch never produces a speedup and per-layer swapping of these ops is a
dead end on merit rather than on plumbing.

**It is the second one, and the reason is not kernel quality.**

### The measurement

Device time only, from `neuron-explorer` `total_time`, for identical work computed both ways: N
chained applications of the op, same shape, dtype and compiler defaults (`NEURON_CC_FLAGS` unset, as
everywhere else in this project). Wall-clock dispatch cost is excluded by construction.

*The "compiler defaults" qualifier used to be a live caveat here. It is now discharged by
[Finding #27](#27-the-device-gap-is-not-a-compiler-flag-artifact-and-the-reason-is-structural-critical--closes-the-last-thing-that-could-have-invalidated-25-and-26):
NKI device time varies by 1.05x across five flag settings and its marginal traffic is pinned at the
unfused floor under all of them, so nothing below is a configuration artifact.*

| config | device ms | ms/call | HBM r+w | MBU | active |
|--------|-----------|---------|---------|-----|--------|
| silu / NKI / N=28 | 0.607 | 0.0217 | 188.7 MB | 43.2% | 95.1% |
| silu / torch / N=28 | **0.224** | **0.0080** | **6.3 MB** | 3.9% | 97.8% |
| rmsnorm / NKI / N=28 | 1.625 | 0.0581 | 188.8 MB | 16.2% | 99.1% |
| rmsnorm / torch / N=28 | **0.637** | **0.0227** | **6.4 MB** | 1.4% | 94.4% |

**NKI is 2.71x slower on SiLU and 2.55x on RMSNorm, with ~30x the HBM traffic.**

### The attribution needed care, and the first pass of it was wrong

The obvious next step is to divide traffic by N and compare against a floor. Doing that at N=1 says
NKI moves 3.00x the necessary traffic for both ops, which reads as a spilled intermediate — plausibly
the fp32 temporary introduced when these kernels were migrated to `nl.ds` (Finding #14 correction).

That conclusion is an artifact. **Traffic is not linear in N**: a small NEFF carries fixed setup
traffic that dominates at N=1. With two call counts, both terms are solvable:

```
traffic(N) = FIXED + N x MARGINAL
```

| config | traffic(1) | traffic(28) | marginal/call | vs floor | fixed |
|--------|-----------|-------------|---------------|----------|-------|
| silu / NKI | 18.87 MB | 188.74 MB | **6.29 MB** | **1.00x** | 4.0 tiles |
| silu / torch | 6.29 MB | 6.29 MB | **0.00 MB** | 0.00x | 2.0 tiles |
| rmsnorm / NKI | 18.88 MB | 188.76 MB | **6.29 MB** | **1.00x** | 4.0 tiles |
| rmsnorm / torch | 7.88 MB | 6.42 MB | **~0.00 MB** | 0.00x | 2.5 tiles |

The unfused floor for a `[512, 3072]` bf16 tile is 2 tiles = 6.29 MB: one read in, one write out, the
minimum for an op that cannot fuse with its neighbours.

**NKI's marginal traffic is exactly 1.00x that floor, for both ops.** The kernels spill nothing and
are optimal for an unfused op. **Torch's traffic is independent of N**, which is only possible if the
entire chain fused into a single pass.

So the kernels are blameless and the whole gap has one cause.

### The cause

A NKI kernel reaches the compiler as an opaque custom call. **The compiler cannot fuse across it.**
So replacing a torch op with a NKI kernel does not merely add dispatch cost — it *removes* a fusion
opportunity the compiler was already exploiting. Each swapped op is forced to materialise its output
to HBM and re-read it, where before it stayed resident across a fused region.

For memory-bound operations — elementwise activations, normalisations — fusion *is* the optimisation.
There is no arithmetic to speed up; the only thing that matters is how many times the data crosses
the HBM boundary. A NKI kernel for such an op is therefore competing against "not touching HBM at
all", and cannot beat it outright on that axis however well written it is.

The MBU column makes this visible from the other side: NKI runs at 43.2% memory-bandwidth
utilisation (it is bandwidth-bound, moving 188 MB), torch at 3.9% (it barely touches memory).

### Why this matters, and how much — the second part was initially overstated

Findings #20 and #24 said per-layer swapping is *expensive*. This says something different in kind:
swapping a small memory-bound op **forfeits a compiler optimisation**, so the kernel starts behind
regardless of dispatch cost. Fixing dispatch is necessary but not sufficient.

> **The magnitude claim originally made in this section was wrong, and is corrected below.** It said
> break-even is "unreachable, not merely distant", on the strength of the chained microbenchmark. In a
> real forward pass the device gap is **8.4%** of the regression against **91.6%** dispatch, so with
> dispatch fixed these kernels land near **1.18x** slower, not 2.5–2.7x. Break-even is close but not
> reached. See "MEASURED IN SITU" below, and do not quote the 2.5–2.7x figure without it.

It also explains the shape of the entire project. The ops the Kernel Hub is *best* at intercepting —
RMSNorm (115 upstream registrations), RoPE (95 model files), activations (one decoration covering all
of `ACT2FN`) — are precisely the ops that lose most from being intercepted. They are small, memory
bound, and already fused. The mechanism's reach and its usefulness are inversely correlated.

What *can* win is a kernel that replaces a whole fused region, so the kernel performs the fusion
internally instead of blocking the compiler's. That is exactly what `nkilib`'s fused megakernels are,
and exactly what Findings #17 (weight layout) and #18 (single-core width limit) say the Kernel Hub
cannot currently express. Four findings now converge on the same conclusion from four independent
directions:

| Finding | Direction | Says |
|---------|-----------|------|
| #17 | weight layout | fused kernels need weights `kernelize()` can't produce |
| #18 | sharding | fused kernels assume SPMD; won't compile single-core at real widths |
| #24 | dispatch cost | per-call overhead, ~0.59 ms after the big fix |
| **#25** | **compiler fusion** | **per-layer swap destroys the fusion that makes small ops fast** |

### MEASURED IN SITU — the penalty is real but second-order, and this section's framing over-claimed

The limitation flagged below ("the magnitude in situ was not measured") has since been measured, and
it changes how much weight this finding should carry. `scripts/profile_model_device_time.py` profiles
the real Qwen3-0.6B forward with `NEURON_RT_INSPECT`, and `scripts/sum_model_device_time.py` sums
device time across the emitted NEFFs to decompose the wall-clock gap:

```
wall_k - wall_b  =  (device_k - device_b)  +  (dispatch_k - dispatch_b)
```

| | NEFFs | device time | HBM traffic | activates |
|---|---|---|---|---|
| baseline | 1 | 14.329 ms | 2662.4 MB | 8,038 |
| kernelized | 1 | 22.722 ms | 3779.9 MB | 16,468 |

| term | value | share of wall gap |
|------|-------|-------------------|
| wall gap (46.65 → 146.65 ms, profiled run) | 100.00 ms | 100% |
| **device gap** | **8.392 ms** | **8.4%** |
| **dispatch gap** | **91.608 ms** | **91.6%** |

Per NKI call, at 169 calls/step: **device 0.0497 ms, dispatch 0.5421 ms.** Dispatch is ~11x larger.

So the fusion barrier is real in situ — device time rises 1.59x and HBM traffic 1.42x — but it is
**second-order**. The 2.5–2.7x figure holds only where 28 identical ops sit back to back, which is
simultaneously the compiler's best case and NKI's worst. In a real model these ops are separated by
matmuls, and most of the device work is matmul regardless.

**This reverses the ranking in "What to do" below and in the PoC recommendation.** With dispatch
removed, the model would run at roughly `46.65 + 8.39 = 55 ms` against a 46.65 ms baseline — about
**1.18x slower, not 3.4x**. So caching `create_computation` is decisive after all, and the fusion
question is important but not the binding constraint at model scale.

Break-even is therefore **close but not reached**, rather than unreachable: ~1.18x with perfect
dispatch, and closing the last 18% would require the kernels to beat torch on device, which the fusion
barrier prevents. That is a materially weaker claim than "these ops cannot win", and it is the correct
one.

**The methodological failure here is worth naming, because it is subtle.** The caveat immediately below
was written *before* the recommendation was drafted, and the recommendation was then written as though
it did not exist — treating the microbenchmark number as the operative one and demoting the dispatch fix
on its strength. **A caveat in the text is not a caveat in the conclusion.** Either measure the thing
the caveat is about, or let the caveat constrain what you claim.

Caveats on the in-situ numbers themselves: HBM traffic in the model includes weights, so the 1.42x
ratio dilutes the activation-only effect and should not be read as the fusion penalty directly. And
wall times here (46.65 / 146.65) come from the profiled run rather than `measure_mfu` (42.04 / 141.43),
since `NEURON_RT_INSPECT` adds a few ms; the decomposition uses one consistent pair throughout.

### Honest limits of this measurement

- **The chained microbenchmark maximises the fusion advantage.** 28 identical ops back to back is the
  best possible case for the compiler. In a real model these ops are separated by matmuls, so less
  fusion is available and the real penalty is smaller than 2.7x. It is not zero — Qwen3's SiLU sits
  between the `gate * up` elementwise multiply and the down projection, which is exactly the kind of
  neighbour it would otherwise fuse with — but the magnitude in situ was not measured. **(Now measured;
  see above. It is 8.4% of the regression.)**
- **N=1 numbers are unreliable for traffic attribution**, since fixed NEFF traffic dominates. The
  regression above is the right instrument; the raw N=1 per-call division is the wrong one, and it is
  what produced the false spilled-intermediate reading.
- **This says nothing about compute-bound ops.** A matmul-heavy fused kernel has real arithmetic to
  optimise and is not competing against "stay in registers". #25 is a statement about memory-bound
  ops specifically.
- **Not measured: whether a NKI kernel spanning a fused region beats the compiler on that region.**
  That is the experiment that would confirm the positive half of the recommendation, and it is
  blocked by #17 and #18.

### What to do — reordered after the in-situ measurement

1. **Fix 7 (cache `create_computation`) is the decisive one.** 91.6% of the model-level regression is
   dispatch, and closing it takes the kernels from 3.4x slower to roughly 1.18x. That is the single
   largest available improvement, and this finding does not change that.
2. **Ask the compiler team whether fusion across NKI custom calls is achievable.** Still worth asking —
   it is what stands between ~1.18x and a genuine win — but it is the second question, not the first.
   If a NKI kernel could be made transparent to the fusion pass, or could declare itself fusable, this
   finding dissolves.
3. **Re-scope the porting queue around fusion span, not op popularity.** The question for a candidate
   is "does this replace a region the compiler would otherwise fuse, and does it do that fusion
   better?" — not "how many models call this op?" This holds regardless of the magnitude above.
4. **Keep treating RMSNorm, RoPE and activations as mechanism demonstrations rather than performance
   targets.** They are excellent at the former — small, single-op, no weight-layout issues, no
   sharding. Even with perfect dispatch they are ~18% underwater, so they are not wins. Those are
   different goals and this PoC conflated them for weeks.
5. **Do not quote the 2.5–2.7x figure without its context.** It is a chained-microbenchmark upper bound.
   The in-situ number is 8.4% of the regression, and the two will be confused if the first is stated
   alone.

---

## 26. The fused MLP also loses by ~3x on device — and #18 was a design boundary, not a bug [CRITICAL — answers "where would a speedup come from?"]

Prompted by the question the PoC should have asked in Week 2: *why is there a slowdown at all — we
should be seeing a speedup.* Chasing it produced the clearest statement of the mismatch in this
document, and corrected two earlier errors of mine.

### First, why RMSNorm and SiLU were tutorial-derived rather than ported

Not a shortcut. **Standalone versions do not exist in nki-library.**

- `nkilib/core/rmsnorm/` contains exactly one kernel, `rmsnorm_quant.py`, which fuses RMSNorm with
  FP8 quantisation and always quantises — `QuantizationType.NONE` is not a validated input. The
  closest thing to a plain BF16→BF16 RMSNorm is `_rms_normalize_tile()`, an internal subroutine.
- `nkilib/core/` has **no activations module at all**: `attention, cumsum, embeddings, max, mlp, moe,
  moe_block, output_projection, qkv, quantization, rmsnorm, router_topk, subkernels, topk, utils`.
  SiLU exists only inside `mlp/mlp.py`.
- `embeddings/rope_hf.py` is standalone and already HF-shaped. It is the one op of the three that was
  ported directly.

So the sourcing decision *was* the structural finding, arriving in Week 2: **the ops the Kernel Hub
can intercept mostly do not exist as separable units in nki-library.** It was recorded in
`docs/nki-library-porting-analysis.md` and then not allowed to inform what the Week 4 MFU measurement
was expected to show.

### Second, the fused MLP was written off on a limit it does not hit

Finding #18 established that `nkilib.core.mlp.mlp` fails to compile single-core when
`intermediate_size > 4096`, and that was used to defer all fused-kernel work. But #18's own data shows
`hidden_size=1024, intermediate_size=3072` **passes** at cos_sim 0.999995 — which is exactly
Qwen3-0.6B's MLP shape, the model every MFU number in this project was measured on.

So the one kernel that could plausibly show a speedup works for the benchmarked model, and had never
been timed. It replaces a whole fusable region (gate + up + SiLU + down) rather than interrupting one,
and it contains two real matmuls, so unlike RMSNorm/RoPE/SiLU there is compute to optimise.

### The measurement

`scripts/profile_fused_mlp_vs_torch.py`. Device time from `neuron-explorer`, correctness gated against
a CPU fp32 reference on every run, weights transposed on device (the realistic path).

| H=1024, I=3072, S=512, 28 blocks | device ms | per block | HBM r+w | MBU |
|---|---|---|---|---|
| NKI fused MLP | 8.321 | 0.2972 | 2172.6 MB | 36.5% |
| torch (3 matmuls + silu) | **2.782** | **0.0993** | **1059.1 MB** | 53.2% |
| | | | | **NKI/torch = 2.99x** |

| H=4096, I=4096, S=512, 8 blocks — largest single-core shape per #18 | device ms | per block | HBM r+w | MBU |
|---|---|---|---|---|
| NKI fused MLP | 11.625 | 1.4532 | 3288.3 MB | 39.5% |
| torch | **4.180** | **0.5225** | **1619.1 MB** | 54.1% |
| | | | | **NKI/torch = 2.78x** |

cos_sim 0.999979 and 0.999977, so both are correct. The gap barely narrows with scale
(2.99x → 2.78x), HBM traffic stays at ~2.0x, and torch gets consistently better bandwidth
utilisation at both shapes. **Not a shape artifact.**

### A harness error caught mid-measurement, and it is the same one as #25

The first version reused **one** weight set across all 28 chained blocks. That let the compiler load
the weights once and amortise them over all 28, and torch came out at 12.1 MB/block of traffic against
an 18.9 MB weight set — *less than a single weight load*, which is what exposed it. A real model has
distinct weights per layer, so that amortisation does not exist.

Fixed to one weight set per block (528 MB total) and re-run. `--shared-weights` reproduces the flawed
configuration deliberately. Same class of error as #25's chained microbenchmark: **the harness handed
one side an advantage that does not occur in practice.** Twice now, so it is worth a standing check —
before comparing two implementations, ask what the harness lets each one amortise that a real model
would not.

### The interpretation, which reframes Finding #18

`nki-library` kernels are built for the NxDI inference pipeline: **multi-core SPMD**, large shapes,
frequently quantised. Run single-core, a kernel has one core's SBUF (24–28 MB) to work with, so it
tiles the problem far more finely than it was designed to and pays a HBM round-trip at every tile
boundary. Hence ~2x the traffic at ~40% MBU, against torch's ~54%. The Neuron compiler, targeting that
same single core, picks better tiling.

**So #18 is not a bug to route upstream. It is the kernel telling us single-core is not its execution
model.** A `floordiv` by zero when `intermediate_size > 4096` is what a shard-count calculation looks
like when there is no shard grid. Finding #18 filed it as a divide-by-zero to fix and recommended it as
an nki-library bug; it should have been read as a design boundary. That correction belongs in the
upstream asks — it changes what we are asking for and of whom.

### Where a speedup would actually come from

Putting #25 and #26 together answers the question completely:

| candidate | why it can't win here |
|---|---|
| RMSNorm, RoPE, SiLU | small, memory-bound, already fused by the compiler; the swap forfeits that fusion (#25) |
| fused MLP | spans a fusable region and has real compute, but runs single-core with no SPMD grid, so it tiles badly and moves 2x the traffic (this finding) |

**The mechanism and the kernel library are built for different execution models**, and that is now a
measured 3x on device rather than an inference from weight layouts and compile errors.

A speedup needs the kernels in their intended configuration: multi-core SPMD, `intermediate_size`
beyond 4096, likely quantised. `kernelize()` expresses none of that — it swaps a `forward()` method,
on one device, with weights in whatever layout the model already has.

### Incidentally: Finding #17 now has a number

The on-device weight transpose the kernel requires costs **3.533 ms / 1172 MB** at H=1024/I=3072 and
**6.726 ms / 2223 MB** at H=4096/I=4096, as its own NEFF. That is a one-time load cost rather than
per-step, so it does not belong in the per-step comparison above — but it is the first time the
weight-layout mismatch has been quantified rather than described.

### What to do

1. **Re-file Finding #18.** Not "fix the divide-by-zero" but "document that these kernels require an
   SPMD launch grid, and state the supported configuration." The current error is a poor diagnostic for
   what is actually a usage constraint.
2. **Stop treating the fused MLP as the blocked-but-promising candidate.** It is not blocked by a bug;
   it is being run in a configuration it was not built for, and it loses by 3x there.
3. **If per-layer Kernel Hub integration is to be pursued for performance, the question to answer first
   is whether `kernelize()` can express a multi-core launch.** Not weight layout (#17), not the compile
   limit (#18) — those are downstream of the execution-model mismatch this finding measures.
4. **Do not read this as "NKI kernels are slow."** These kernels are correct to six decimals and are
   presumably good at what they were built for. Nothing here measures them in their intended
   configuration, and this document should not be cited as if it did.

---

## 27. The device gap is not a compiler-flag artifact, and the reason is structural [CRITICAL — closes the last thing that could have invalidated #25 and #26]

Every performance measurement in this project ran with `NEURON_CC_FLAGS` unset. That was recorded as a
caveat from the start, and it was the right caveat: a bad compiler default would be the cheapest
possible explanation for the entire slowdown, and it is the most plausible *technical* form of the
reviewers' objection that there should not be a slowdown at all. It stayed open for two sessions
because it needed hardware.

It is now closed, in two halves, and the second half gives a stronger answer than "we tried some
flags and none helped."

### Half one: wall clock

Five settings — `{unset, --target trn2, +--lnc 1, +--lnc 2, +-O2}` — 28 chained SiLU applications on a
`[512, 3072]` bf16 tile. One subprocess and one isolated compile cache per setting, so no setting can
be served a NEFF another setting compiled.

| `NEURON_CC_FLAGS` | NKI ms | torch ms | ratio |
|---|---|---|---|
| (unset) | 14.096 | 0.728 | 19.37x |
| `--target trn2` | 14.061 | 0.764 | 18.39x |
| `--target trn2 --lnc 1` | 13.821 | 1.708 | 8.09x |
| `--target trn2 --lnc 2` | 14.019 | 0.726 | 19.32x |
| `--target trn2 -O2` | 14.152 | 1.023 | 13.83x |

Spread: ratio 2.39x, **NKI 1.02x**, torch 2.35x.

**The first version of this probe got the answer wrong**, and how it did is worth keeping. It
thresholded on whether the NKI/torch *ratio* moved. The ratio spread was 1.53x on its first run, which
reads as "flags matter, re-run everything." But the columns separately say the opposite: NKI is flat
at 13.8–14.2 ms while torch varies by 2.35x, and the whole ratio spread comes from `--lnc 1` making
*torch* slower. **A ratio can move because its denominator moved.** The probe now reports both spreads
and concludes on whether NKI itself responds.

**Scope limit, which the probe prints itself.** 13.82 ms for 28 calls is 0.494 ms/call, and that *is*
the post-fix dispatch floor (Finding #24). So this run is ~97% dispatch and only a few percent device.
It establishes that the *dispatch* cost is flag-invariant. It says almost nothing about the
device-time claims in #25 and #26, which are the ones the recommendation rests on.

### Half two: device time, which is the half that matters

Same five settings, but profiling device time at **N=1 and N=28**, so marginal traffic per call can be
solved for rather than inferred.

| `NEURON_CC_FLAGS` | NKI ms | torch ms | ratio | NKI MB/call | vs unfused floor |
|---|---|---|---|---|---|
| (unset) | 0.608 | 0.224 | 2.72x | 6.29 | 1.00x |
| `--target trn2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |
| `--target trn2 --lnc 1` | 0.580 | 0.429 | 1.35x | 6.29 | 1.00x |
| `--target trn2 --lnc 2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |
| `--target trn2 -O2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |

NKI device time spread **1.05x**. NKI marginal traffic spread **1.00x**.

### Why this is the strong form of the result

The weak version would be: *five settings were tried and none was better.* That leaves a sixth setting
open, and it invites a reviewer to reasonably ask whether the right one was missed.

The actual finding is different in kind. **The quantity a better setting would have to move is already
at its theoretical minimum.** The unfused floor for a `[512, 3072]` bf16 tile is two tiles — one read
in, one write out — which is 6.29 MB, and NKI's marginal traffic is exactly 6.29 MB under every
setting. That is the least an operation that cannot fuse into its neighbours can possibly move. There
is no headroom for a flag to find, so the absence of a better setting is not evidence about the search;
it is a consequence of the measurement.

The device gap is therefore **structural**: a NKI kernel reaches the compiler as an opaque custom call,
the compiler cannot fuse across it, and compiler flags do not reach that. Findings #25 and #26 stand as
measured.

### The 1.35x row is a trap

It is the best ratio in the table and it is not an improvement. NKI barely moves (0.608 → 0.580) while
torch gets 91% slower (0.224 → 0.429). Reading it as progress would repeat exactly the mistake the
first wall-clock probe made — twice in one investigation, from the same cause: **a ratio is two
numbers, and a change in it does not say which one moved.**

This is also the second time in this project that `--lnc 1` has looked like an improvement and been an
artifact of degrading the baseline. If anyone quotes a favourable NKI/torch ratio under `--lnc 1`, that
is the thing to check first.

### What to do

1. **Close the open item and say what closed it.** Both probes are now harness stages
   (`compiler-flag-control`, `compiler-flag-control-device`), so an SDK or compiler upgrade re-tests
   this automatically instead of depending on someone remembering it was once a question. That matters
   more than the answer: the answer is true of this compiler version.
2. **Stop qualifying #25 and #26 with "on compiler defaults."** The qualification was honest and is now
   discharged. Replace it with the actual result, which is stronger.
3. **When reporting any ratio in this project, report both terms.** Two separate wrong readings came
   from watching a ratio and not its parts. This is now enforced in the probes rather than remembered.
4. **Do not read this as "no compiler work could help."** It says no *flag* helps. Making a NKI custom
   call fusible — so the compiler can see through it — is a compiler change, and it is the one change
   that would move the 6.29 MB. That remains the open question (B14), and this finding sharpens it from
   "can fusion happen?" to "the 6.29 MB/call is the exact quantity a fusible custom call would remove."

---

## 28. B12 answered — the residual is a second bypassed cache, and the slowdown drops to 1.37-1.62x [CRITICAL]

Finding #24 removed ~52 ms/call and left ~0.53 ms/call, which the in-situ decomposition showed was
91% of the remaining regression. That residual was deliberately not attacked: cProfile put it inside
`torch_xla`'s op-registry path, and a wrong guess there could be silently incorrect rather than raise.

Reading the source answered it before any measurement, and the answer is **Finding #24's shape a
second time: a cache exists and the code path throws it away.**

`torch_xla/core/xla_op_registry.py` defines `Op`, which holds

```python
self._computations = dict()      # keyed on pickle.dumps([shapes, kwargs])
```

and whose own docstring says: *"Python based XLA operations should be preferably registered globally,
in order to amortize the lowering cost."*

`nki/framework/_torch_xla.py::TorchXlaKernel.__call__` does the opposite:

```python
@xla_hlo_call                       # -> xla_call -> xla_op_registry.register -> Op(...)
def nki_custom_call(*tensors):
    ...
xla_result = nki_custom_call(*input_tensors)
```

The decorator runs **inside** `__call__`, so every kernel invocation constructs a brand-new `Op` with
a brand-new empty `_computations`. The cache is not cold by accident; it is newly created, and
therefore always empty, on every call.

Put beside #24, the pair is striking:

| | the cache | how it is defeated |
|---|---|---|
| #24 | `func._nki_compile_cache` | target resolution runs while building the cache *key*, so a hit still pays the subprocess |
| B12 | `Op._computations` | the memo lives on an object that is recreated per call |

Neither is a property of per-layer kernel dispatch on Neuron. Both are one misplaced line.

### Why the key is sound, which is what separates a fix from a guess

The lowering closure captures `config = nir.build_config()`. `nir` comes from
`self._cached_compile_to_bir(frontend, converted_inputs, compile_opts)`, already memoised on
`self._generate_cache_key(converted_inputs, compile_opts)`. So same key ⟹ same `nir` ⟹ same `config`
⟹ same closure. **The key is not a judgement about what is safe to share; it is the key NKI already
uses for the object the closure is built from.** Two guards make that concrete: the Op is cached only
when NKI's compile cache is enabled (with `NKI_DISABLE_COMPILE_CACHE`, `nir` is rebuilt per call and
could differ), and a null key from unhashable arguments falls through to the original path.

### Verified in the shape of #24's verification

| variant | ms/call | speedup | cos_sim |
|---|---|---|---|
| baseline (post-#24) | 0.5278 | — | 0.9999424815177917 |
| Op registry cached | 0.1828 | **2.89x** | 0.9999424815177917 |
| baseline again (control) | 0.4943 | — | 0.9999424815177917 |

86 hits, 1 miss, 1 distinct key. Cosine similarity is **bit-identical to 16 digits**, not merely above
a threshold. The baseline is re-run last, so ordering cannot explain it. And the probe reports cache
hit counts, so a timing win cannot be credited to the cache without evidence the cache was used —
2361 hits against 5 misses on the full model.

It also refuses to patch an NKI it does not recognise. The patch reimplements
`TorchXlaKernel.__call__`, so five structural landmarks are asserted in the installed source first and
the source hash is printed for the record (`5f8521daa38d96a2`).

### Model-level effect, which is the number that matters

| stage | seq | baseline | kernelized | slowdown | MFU | added ms/call |
|---|---|---|---|---|---|---|
| no fixes | 512 | 43.06 | 8873.67 | 206.07x | 0.02% | 52.2522 |
| #24 only | 512 | 44.36 | 146.67 | 3.31x | 1.45% | 0.6054 |
| **#24 + B12** | 512 | 43.94 | **71.32** | **1.62x** | 2.98% | 0.1620 |
| #24 only | 2048 | 109.64 | 226.16 | 2.06x | 4.76% | 0.6894 |
| **#24 + B12** | 2048 | 117.78 | **161.04** | **1.37x** | 6.69% | 0.2560 |
| device floor | | | | | | 0.0495 |

**52.25 → 0.605 → 0.162 ms/call.** 86.3x from #24, another 3.7x from B12, **322.5x together.** The
cost is now within 3.3x of the device floor, and 69% of what remains is still dispatch.

### What this does to the project's own framing

The ~1.15–1.18x figure was a **projection** that assumed dispatch went to zero. 1.62x and 1.37x are
**measured**, and they sit between the old 3.31x and that projection — which is exactly what the
projection predicted, since B12 removes about two thirds of the dispatch term rather than all of it.
The claim that the slowdown was a framework bug rather than a property of the approach is now
demonstrated rather than argued.

### What to do

1. **Two upstream asks now, not one, and they are the same kind.** #24: memoise `_detect_target`.
   B12: register the lowering once per compile-cache key. Both are small, both are in NKI's dispatch
   path, both are accuracy-neutral by measurement.
2. **Stop quoting 3.31x.** It is the one-fix number. 1.62x at seq 512 and 1.37x at seq 2048 are
   current.
3. **Neither of these is shipped.** Both are runtime monkeypatches verified on this stack. The
   deliverable is the diagnosis and the verification, not a patch anyone can deploy.

---

## 29. A SPEEDUP EXISTS: NKI flash attention beats the compiler at seq >= 2048 [CRITICAL — this is the answer to "we should see a speedup"]

Every previous candidate in this project lost, and Findings #25 and #26 explained why with a criterion:

> A kernel wins when it replaces a region the compiler would **not** otherwise fuse well, **and**
> there is real arithmetic to restructure.

RMSNorm, RoPE and SiLU fail both halves — small, memory-bound, already fused, so an opaque custom call
*removes* an optimisation. The fused MLP passes the second half and loses 2.99x single-core for lack of
an SPMD grid. What had not been tested was the one candidate the criterion actually favours.

### Why attention is different, stated before measuring

1. **Flash attention is an algorithmic restructuring, not a fusion.** It never materialises the
   `[heads, S, S]` score matrix, using online softmax with running max/sum instead. A compiler fuses
   elementwise chains; it does not re-derive the algorithm. So this is not something the compiler is
   already doing.
2. **There is real arithmetic.** Two matmuls per head, and under causal masking half the score tiles
   can be skipped entirely rather than computed and masked to `-inf`. `attention_cte` does skip them.
3. **It runs single-core.** `nkilib/core/attention/attention_cte.py` states it "can be invoked with 1D
   SPMD grid for LNC2 **or without grid**". That is precisely the property the fused MLP lacked, so
   #26's verdict does not automatically carry over.

It also worked **first try** against the HF-native layout — `tp_q=True, tp_k=True, tp_out=False` maps
directly onto `(batch·heads, seq, head_dim)`, GQA is expressed natively as `batch_size_kv < batch_size`
with no K/V replication, and correctness was `cos_sim 1.000010` against a CPU fp32 reference. Contrast
Finding #13's RoPE (undocumented) and #17/#18's MLP (wrong weight layout, compile boundary). This is
the first nkilib kernel that dropped into the Kernel Hub's calling convention without a fight.

### The result

Device time, single logical core, Qwen3-0.6B head geometry (16 q heads, 8 kv heads, `head_dim` 128),
causal, 4 layers per graph with distinct K/V per layer:

| seq | NKI ms/layer | torch ms/layer | NKI/torch | NKI MB/layer | torch MB/layer | score matrix |
|---|---|---|---|---|---|---|
| 512 | 0.2463 | 0.1225 | 2.01x slower | 8.39 | 3.16 | 8.4 MB |
| 1024 | 0.4939 | 0.4269 | 1.16x slower | 16.78 | 13.12 | 33.6 MB |
| **2048** | **1.1438** | **1.6902** | **1.48x FASTER** | 33.55 | 279.86 | 134.2 MB |
| **3072** | **1.8484** | **3.9062** | **2.11x FASTER** | 50.33 | 748.70 | 302.0 MB |
| 4096 | 2.8295 | 1.5784 | 1.79x slower | 67.11 | 395.05 | 536.9 MB |

The 2048–4096 region was measured twice, in separate runs, with 3072 added the second time. It
reproduces to four significant figures (NKI 1.1420/1.1438 at 2048, 2.8299/2.8295 at 4096), so nothing
below is noise.

**The crossover is between 1024 and 2048, and the traffic column explains it.** At seq 512 torch moves
3.16 MB/layer — *below* the 6.29 MB it costs merely to read q, k, v and write the output once. The only
way that is possible is that the compiler fused the whole chain and kept the score matrix resident, so
**flash attention's central advantage is something XLA on Neuron was already achieving at short
sequences**, and the kernel spends an HBM round-trip at its boundary to buy something it does not get.
That is the same fusion-barrier story as #25, arrived at from a different direction.

As `S` grows the score matrix grows as `S²` while flash's working set grows as `S`. At seq 3072 torch
moves 748.70 MB/layer against NKI's 50.33 — **14.9x more** — and its MBU has risen from 3.6% to 26.8%.
The compiler has stopped being able to keep the scores resident, and the kernel wins by 2.11x.

**NKI's side of the table is textbook flash attention.** Traffic is exactly linear in sequence length —
16.78 MB per 1024 tokens, at every point, with no inflection — and time grows smoothly. The kernel does
what it says it does.

### The seq-4096 reversal is on the TORCH side, and my first hypothesis was backwards

The 4096 row breaks the trend, and my initial reading was that the NKI kernel had run out of single-core
SBUF: at seq 4096 K and V are 8.4 MB each, `attention_cte` only sections K/V above 10K tokens, and a
spill would produce exactly a superlinear cost. That would have been Finding #26 recurring, and it was
a tidy story.

**The traffic column says it is wrong.** Torch's HBM traffic per layer goes 279.86 → 748.70 → **395.05**
MB as sequence goes 2048 → 3072 → 4096. It *drops by 47%* while the score matrix it is supposedly
materialising *grows* from 302 MB to 537 MB. At 3072 torch moves 2.5x the score matrix, consistent with
materialising it and passing over it several times. At 4096 it moves 0.74x the score matrix — **less than
one copy**, which is only possible if it stopped materialising it.

Meanwhile NKI at 4096 is exactly on its own linear trend (67.11 MB = 4 × 16.78) and its time is exactly
on trend too. **Nothing degraded on the NKI side. The compiler got better.**

The leading explanation is that XLA has a threshold above which it switches attention strategy — a
blocked or tiled decomposition rather than a materialised score matrix — and above that threshold it is
competitive again. That is checkable by dumping the HLO for the two configurations and diffing the
fusion structure, which has not been done, so it is stated as the explanation the traffic supports
rather than as a confirmed mechanism.

It is worth recording how close this came to being published the other way round. The SBUF story was
plausible, matched an existing finding, and would have blamed the kernel. The traffic column contradicts
it, and the only reason the contradiction surfaced is that the sweep records traffic per configuration
rather than just time. **A one-number benchmark would have produced a confident wrong answer here.**

### What this changes about the recommendation

This is the first candidate that wins, and it wins for the reason the criterion predicted. The
recommendation moves from "the mechanism works and no speedup is available" to something sharper and
more useful:

1. **Point Kernel Hub interception at attention, not at RMSNorm/RoPE/activations.** The uncomfortable
   corollary of Finding #25 still holds — reach and benefit are inversely correlated, and the ops the
   Hub intercepts most widely have the least to gain — but attention is both widely intercepted
   (transformers has an attention interface) and genuinely improvable.
2. **State the sequence length with the claim, and state that it is a WINDOW.** 2.01x slower at seq 512,
   1.48x faster at 2048, 2.11x faster at 3072, 1.79x slower at 4096 — all the same kernel. Quoting any
   one of those without `seq` is the mistake sticking point #18 records. And the window has an upper
   edge as well as a lower one, which is the part most likely to be dropped in retelling.
3. **The upper edge is the more interesting half.** The window closes because the *compiler* improves at
   4096, not because the kernel degrades. So the size of the opportunity depends on where XLA's own
   attention strategy switches, which nobody on this project knew was a threshold at all. Find it
   properly: dump HLO either side of it.
4. **The dispatch fixes are load-bearing for this result.** At 0.53 ms/call of overhead the 2048 win
   (1.14 ms/layer device) would have been invisible. #24 and B12 are what make an attention kernel worth
   swapping at all — which is the concrete argument for fixing them upstream.
5. **Measure multi-core.** Still the largest gap, and `attention_cte` shards batch across LNC2 cores, so
   it is the configuration this kernel was built for and the numbers above are its handicapped case.
6. **Do not over-read a 4-layer microbenchmark.** This is device time for chained attention layers with
   distinct K/V, not a model. In situ, attention sits between QKV and O projections that force HBM
   boundaries anyway, so the custom-call boundary costs less than it does here — the in-situ effect
   should be *better* than this, but it has not been measured. Finding #25 made the opposite mistake in
   the opposite direction; the lesson is the same.
