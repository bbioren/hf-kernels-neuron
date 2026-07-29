# Sticking Points Log

Running log of things that were harder than expected, took extra time, or blocked progress. Each entry notes the date, what happened, how long it took to resolve, and whether it would affect customers or engineering at scale.

---

## Format

```
### [DATE] Short title
**Time lost:** X min/hours
**Would affect:** customers / engineering / both
**Resolution:** what fixed it
**Takeaway:** what should be different
```

---

### [2026-07-22] `kernels` not installable from GitHub source
**Time lost:** 15 min
**Would affect:** engineering (anyone trying to test unreleased features)
**Resolution:** Install from PyPI instead. Library is Rust/Python hybrid, needs maturin to build from source.
**Takeaway:** Document that devs must use PyPI releases. If Neuron patches land before a release, building from source requires Rust toolchain.

### [2026-07-22] `LocalLayerRepository` API changed — docs show removed param
**Time lost:** 20 min
**Would affect:** both (any kernel author following docs)
**Resolution:** Checked actual constructor signature via `help()` on trn2. v0.15.2 only takes `(repo_path, *, layer_name)`.
**Takeaway:** Pre-1.0 library, APIs shift. Always verify against installed version, not docs. Pin minor version.

### [2026-07-22] `metadata.json` required for local dev (underdocumented)
**Time lost:** 30 min (two round-trips to trn2 to figure out required fields)
**Would affect:** both
**Resolution:** Added `python-depends: []` and `digest: {"algorithm": "sha256", "files": {}}` — both required but not in the "local dev" docs.
**Takeaway:** HF should either simplify LocalLayerRepository to not need metadata, or document the minimum fields clearly.

### [2026-07-22] Neuron DLAMI venv structure — torch not system-wide
**Time lost:** 45 min (tried .pth hack, then standalone venv, finally just activated DLAMI venv)
**Would affect:** customers
**Resolution:** `source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate` — use the DLAMI venv directly.
**Takeaway:** The DLAMI doesn't match the "just pip install" developer experience. Must document the activation step prominently.

### [2026-07-22] `torch_neuronx` import crashes without bin/ on PATH
**Time lost:** 15 min
**Would affect:** customers
**Resolution:** The `.pth` hack (adding site-packages to a local venv) doesn't expose the neuron venv's `bin/` directory. Must fully activate the venv.
**Takeaway:** `torch_neuronx` has a hard dependency on `libneuronpjrt-path` binary being on PATH at import time. This is unusual for a Python package.

### [2026-07-22] Our `kernels/` directory conflicts with the `kernels` pip package
**Time lost:** 20 min
**Would affect:** engineering
**Resolution:** Used `importlib.util.spec_from_file_location()` to load our local kernel by explicit path instead of normal import.
**Takeaway:** Don't name your project directory the same as a pip package you depend on. Or use a different project structure.

### [2026-07-22] Variant resolver detects CUDA, not Neuron
**Time lost:** 1 hour (investigating, writing test scripts, reading source)
**Would affect:** both (blocks Hub publishing with variant structure)
**Resolution:** `hasattr(torch, "neuron")` returns False on current DLAMI. Flat structure (no build/ dir) works via fallback path.
**Takeaway:** `torch_neuronx` should set `torch.neuron` attribute. File bug or PR.

### [2026-07-22] nki-library has no standalone RMSNorm
**Time lost:** 30 min reading source + writing analysis doc
**Would affect:** engineering (blocks mass porting)
**Resolution:** Used tutorial-derived kernel for PoC. Documented that production kernels need unfused entry points.
**Takeaway:** nki-library is designed for the NxDI inference pipeline (always fused with quant). Needs a simple `nkilib.ops.*` API for the HF use case.

### [2026-07-29] Week 2 accuracy results were measuring the PyTorch fallback, not NKI
**Time lost:** ~1 hour (spotting it, writing the instrumented probe, re-validating)
**Would affect:** both — and this is the worst kind of problem because it fails silently
**Resolution:** `@nki.jit` requires XLA tensors and hard-errors on CPU ones. Our kernel
guards with `device.type != "cpu"`, so CPU-tensor tests took the fallback branch every
time. Fixed by adding `tests/nki_test_utils.py`, which places tensors on the XLA device
and asserts via a call counter that the NKI branch actually ran.
**Takeaway:** The tell was `max_diff = 0.00e+00`. For a hardware kernel, a *perfect*
match is evidence of failure, not success — real NKI reductions differ from PyTorch by
~1e-4. Never accept exact-zero diff as a pass. Always assert the kernel executed, not
just that the numbers look right.

### [2026-07-29] `use_kernels=True` can't reach the neuron device path — two independent gaps
**Time lost:** ~45 min investigating (across kernels + transformers source, then a probe)
**Would affect:** customers (this is the headline user-facing gap)
**Resolution:** No fix available locally. transformers' `kernelize(model, mode)` has no
`device` parameter and derives everything from `model.device.type`; Neuron reports
`"cpu"` (mapping ignored) or `"xla"` (rejected as unsupported). Worked around in tests by
calling the `kernels` library directly with `device="neuron"`.
**Takeaway:** Documented the minimal upstream patch — map `"xla"` → `"neuron"` in
`kernels._find_device` using `xm.xla_device_hw()`, which we confirmed returns `"NEURON"`.
Filed as Finding #9. This is the single highest-value upstream change for the project.

---

## Summary Statistics

| Category | Count | Total Time Lost |
|----------|-------|----------------|
| Documentation gaps | 3 | ~65 min |
| Environment/setup | 3 | ~75 min |
| API instability | 1 | ~20 min |
| Architecture mismatch | 3 | ~2.25 hours |
| Silent-failure / test methodology | 1 | ~60 min |
| **Total** | **11** | **~5.75 hours** |
