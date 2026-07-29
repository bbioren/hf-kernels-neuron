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
| 14 | The two NKI import paths (`nki` vs `neuronxcc.nki`) have different capabilities; neither is a superset | High | Open |
| 15 | Interception-point inventory: several `_KERNEL_MAPPING` entries are unreachable (incl. `SwiGLUMLP`) | Reference | Confirmed |
| 16 | **`nkilib` is already installed and its kernels are directly callable — thin-wrapper porting is feasible today** | **High** | Open (policy blocker only) |
| 17 | Fused MLP: `kernelize()` has no weight-transformation hook | High | Open (design decision) — **premise partly corrected, see #17** |
| 18 | **Fused MLP kernel cannot run single-core when `intermediate_size > 4096` — excludes every real model** | **High** | Open (nki-library bug; no wrapper workaround) |
| 19 | Eager NKI dispatch costs ~0.36 ms host time per call (~25x eager); per-layer microbenchmarking can't resolve kernel quality | High | Open — Week 4 MFU is the right instrument |

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

## 18. The fused MLP kernel cannot run single-core when `intermediate_size > 4096` [HIGH — harder than #17]

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
