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

### [2026-07-29] `nki` and `neuronxcc.nki` are not interchangeable — neither is a superset
**Time lost:** ~50 min (SiLU failed on all 9 shapes; then a wrong "standardise on one
package" attempt broke all 20 RoPE cases before I reverted)
**Would affect:** both — anyone porting more than one nki-library kernel hits this
**Resolution:** Pinned each kernel to the package its idiom needs. RMSNorm and SiLU use
`nl.arange` index tensors, which only resolve under `neuronxcc.nki`. RoPE uses slicing
plus `//` on shape values, which only works under top-level `nki` (`neuronxcc.nki`
treats shapes as symbolic scalars and raises
`NotImplementedError: math.trunc() is not supported for scalar`). Our repo genuinely
needs both packages.
**Takeaway:** `hasattr(nl, "arange")` is True under the top-level package even though
the name cannot be resolved at trace time, so there is no import-time feature detection
— you find out at compile time, per kernel. And the error text never hints that the
sibling package would work. nki-library source uses top-level `nki` while the tutorials
use `neuronxcc.nki`, so a mass-porting effort meets this immediately. Needs a supported
compatibility table from the NKI team. Finding #14.

### [2026-07-29] My own test instrumentation gave a false negative
**Time lost:** ~20 min
**Would affect:** engineering (anyone writing kernel tests)
**Resolution:** In the e2e test I patched a freshly `load_kernel_module()`-ed copy of the
kernel, but `LocalLayerRepository` had loaded its *own* module object, so the counters
read nki=0 while the kernel was demonstrably running (logits had changed). Fixed by
instrumenting via `get_local_kernel()`, which caches and returns the same object the
repository used.
**Takeaway:** Ironic and instructive: this is a false negative of exactly the shape
Finding #8 is a false positive of. Whenever you assert on "did the kernel run", confirm
you are observing the same module object the framework loaded — Python module identity
is easy to get wrong when a package is loaded by path.

---

## Summary Statistics

| Category | Count | Total Time Lost |
|----------|-------|----------------|
| Documentation gaps | 3 | ~65 min |
| Environment/setup | 3 | ~75 min |
| API instability | 2 | ~70 min |
| Architecture mismatch | 3 | ~2.25 hours |
| Silent-failure / test methodology | 2 | ~80 min |
| **Total** | **13** | **~7 hours** |

### Where the time actually goes

Two categories dominate, and they are not the ones you would guess from Week 1:

- **Silent failures and test methodology (~80 min).** Nothing crashed. The kernels
  produced correct numbers while not running at all. This cost a week of false
  confidence in Week 2 and was only caught by noticing an implausibly *perfect* result.
- **Undocumented capability splits (~70 min).** Two NKI packages that both import, both
  pass `hasattr`, and fail differently at compile time.

Neither shows up as an error message a customer could search for. That is the through-line
of this PoC: the Neuron + HF Kernel Hub integration mostly fails *quietly*.

---

## 14. Chasing a performance regression to the wrong layer [~6 hours, the largest single item]

**What happened.** Kernelizing Qwen3 made it 208x slower. Root-causing that consumed most of two
sessions and the conclusion was wrong for most of it.

**Time breakdown, because the shape of it is the lesson:**

| activity | time | outcome |
|---|---|---|
| four framework-level experiments (interleaving, data volume, recompilation, our-vs-production kernels) | ~3 h | all consistent with a wrong hypothesis |
| writing up the graph-transition explanation, twice | ~1 h | had to be corrected twice |
| chasing `torch.compile` as the decisive test | ~1 h | wrong instrument entirely |
| device profile + Python profile | **~35 min** | **found it** |
| verifying the fix and re-measuring | ~30 min | 102x per call, 62x at model level |

The two measurements that actually resolved it took 35 minutes. Everything before them was
elaboration within a framing that could not be falsified by the instrument in use.

**Why it was slow.** Every one of the four experiments measured wall-clock time at the framework
level. A fixed per-call cost independent of problem size is genuinely the signature of
graph-transition overhead, so each experiment came back consistent and increased confidence in a
wrong answer. The hypothesis was never tested against a device profile, which would have killed it
immediately: 0.609 ms of device time against 1459 ms of wall time.

**What would have saved the time.** Measuring device time against wall time *first*. It is one
number from `neuron-explorer` and one from `time.perf_counter()`, their ratio was 2400x, and it
invalidates every device-side explanation at once. Total cost maybe 15 minutes, and it should be
the first thing done on any accelerator performance question, before any hypothesis is formed.

**Who else this affects.** Anyone debugging NKI performance from eager PyTorch. The `neuron-ls`
subprocess costs ~52 ms per kernel invocation on any workload, and it presents as "NKI kernels are
slow" rather than as anything pointing at process spawning. A customer would have no reason to
suspect it and no easy way to find it — it took a cProfile of a single call to see.

---

## 15. `pgrep -af <pattern>` over SSH matches its own command line [~10 min, twice]

`ssh trn2 'pgrep -af neuronx-cc || echo free'` always reports a match, because the `bash -c`
wrapper carrying the pattern is itself a running process containing that pattern. First time it
looked like a stale compiler process was holding the Neuron cores; second time I recognised it.

Use `pgrep -af neuronx-cc | grep -v pgrep`, or check for the actual artifact (`model.neff`)
instead of the process. Minor, but it produces a false "cores busy" reading, which on this box
looks identical to the real and fairly common stale-lock situation.

---

## 16. torch-xla metric accumulators are nanoseconds, not seconds [~15 min, nearly a published error]

`torch_xla.debug.metrics.metric_data(name)` returns `(count, accumulator, samples)`. The
accumulator is in **nanoseconds**, while `metrics_report()` prints it formatted as `us`/`ms`. I
read it as seconds and printed `ExecuteTime 919108000.00 ms` — a nine-digit millisecond figure in
a table next to a 1459 ms wall time, which is what made it obviously wrong.

Had the scale been closer to plausible it would have gone into a finding. Worth stating as a
general rule: when a derived number is impossible, the units are the first thing to check, and a
sanity range on any computed timing catches this class of error for free. Cross-check against
`metrics_report()`, which formats the same values with explicit units.
