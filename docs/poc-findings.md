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

**Minimal upstream fix.** The cleanest change is in `kernels`, not transformers,
because it fixes auto-detection for everyone rather than adding an override that
each framework must thread through. In `kernels/layer/kernelize.py::_find_device`:

```python
dev_type = param.device.type
if dev_type == "xla" and _is_neuron_xla():
    return Device(type="neuron")
```

where `_is_neuron_xla()` checks the XLA runtime reports Neuron hardware. We
confirmed this is detectable today: `xm.xla_device_hw(xm.xla_device())` returns
exactly `"NEURON"` on trn2. See `docs/porting-recommendations.md` for the
proposed patch.

A complementary, independently useful fix: have `torch_neuronx` set a `torch.neuron`
attribute so `_has_neuron_ops()` fires. That alone is *not* sufficient — it does not
change what `_find_device` returns.

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
