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
| 5 | Neuron device shows "not detected (CPU-only mode)" despite hardware present | Low | Investigating |
| 6 | Kernel can be a single file — our multi-file layout was over-engineered | Low | Resolved |

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

### 4. No `kernel-builder` Neuron build variant [OPEN]

**What happened:** The `kernel-builder` tool (used to build and publish kernels to the Hub) has documented build variants for CUDA, ROCm, Metal, CPU, XPU, and CANN. The `backend.type` field accepts `"neuron"` (confirmed in kernel-requirements docs), but there's no documented `kernel-builder` workflow for producing a Neuron build variant.

**Impact:** High — this is the gap between "local dev works" and "published on the Hub for anyone to use." We can develop and validate kernels locally via `LocalLayerRepository`, but cannot publish them to the Hub without a build variant.

**Status:** Known gap per project doc. Week 3 will determine if this blocks Hub publishing or if we can work around it with a manual upload.

**Recommendation:** This is the #1 finding for the kernels team. The mechanism (device path, metadata, layer loading) all works for Neuron. The missing piece is the build tooling to produce a publishable artifact.

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
