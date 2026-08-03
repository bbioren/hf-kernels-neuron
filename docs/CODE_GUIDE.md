# Code guide

A reading order through this repo for a reviewer, and a map from every script to the number it
produced.

Start with [`deliverables/design-doc.md`](../deliverables/design-doc.md) for what was built and why,
and [`results/README.md`](../results/README.md) for the numbers. This document is about the code.

---

## 30-second orientation

```
kernels/          the three NKI kernels — the actual deliverable
tests/            accuracy + end-to-end suites, all assert the NKI branch really ran
scripts/          one file per question asked. Named probe_* / experiment_* / profile_* / measure_*
results/          measurements.json (source of truth) + raw/ (regenerable artifacts)
docs/             findings, upstream asks, porting analysis, this guide
deliverables/     design doc, PoC document, weekly writeups, draft messages
```

Nothing needs building. Kernels are pure Python; `@nki.jit` compiles at first call.

---

## Read in this order

### 1. A kernel — `kernels/neuron_rmsnorm/__init__.py` (186 lines)

The whole pattern in one file. Read top to bottom:

- the `@nki.jit` kernel itself — tiling, `nl.load` / `nl.store`, the fp32 reduction
- `_nki_unsupported_reason()` — the device guard, and *why* it must exist
- `NeuronRMSNorm(nn.Module)` — stateless: no `__init__`, `forward()` reads `self.weight` and
  `self.variance_epsilon` off the module it was grafted onto
- `class layers:` — the namespace the Hub loader looks for
- `has_backward = False`, `can_torch_compile = False` — honest metadata

Then compare against the other two, which differ in instructive ways:

| file | what's different |
|---|---|
| `kernels/neuron_rope/__init__.py` | a **port** of `nkilib/core/embeddings/rope_hf.py`. Function replacement, not layer. Six guards for shapes the kernel can't take (`seq_len % 128`, odd `head_dim`, …) |
| `kernels/neuron_silu/__init__.py` | the simplest case — pure elementwise, no weights |
| `kernels/neuron_identity/` | Week 1 scaffolding. Proves the swap mechanism with a kernel that does almost nothing. Read if you want the mechanism without kernel detail |

### 2. The guard harness — `tests/nki_test_utils.py` (322 lines)

**Read this before any test.** It is the answer to the project's most expensive mistake.

`@nki.jit` needs XLA tensors, so kernels fall back on CPU tensors. Week 2's tests built inputs with
`torch.randn` — CPU — so every case took the fallback and compared it against a mathematically
identical reference. `max_diff = 0.00e+00`, eleven passes, and the kernel had never executed.

Three guards per case:

- `require_neuron()` — refuses to report results unless `xla_device_hw() == "NEURON"`
- `nki_call_counter()` — patches the kernel module's dispatch targets and asserts `nki > 0` and
  `fallback == 0`. **This is what proves the kernel ran.** Numerical agreement cannot.
- `tol_for_dtype()` — dtype-aware tolerance, with cosine similarity as the primary gate

`load_kernel_module()` also lives here: our `kernels/` directory shadows the `kernels` pip package,
so kernels load by path via `importlib.util.spec_from_file_location` — and register in `sys.modules`,
because Dynamo re-imports a traced function's defining module by name.

### 3. A test — `tests/test_rmsnorm_nki.py` (203 lines)

Shows the guards in use, including negative controls: each suite checks it can *fail*, against a
wrong reference and against unmodified input. A suite that cannot fail is not evidence.

### 4. The integration shim — `scripts/neuron_kernel_registration.py` (416 lines)

`kernelize_for_neuron(model)`, used by the e2e tests and by `measure_mfu.py`, plus
`PROPOSED_UPSTREAM_DIFF` — the ~12-line transformers change that would make `use_kernels=True` work.

The shim patches a function object **in this process only**. It does not modify the venv, because a
venv edit would be irreproducible for a customer and would hide the gap that is itself the finding.

### 5. End to end — `tests/test_qwen3_neuron_e2e.py` (295 lines)

Stock Qwen3, kernels swapped, logits compared, call counts asserted. Then
`tests/test_qwen3_moe_e2e.py` for the same three kernels on Qwen3-MoE unchanged — including the
discovery that MoE doesn't run on Neuron at all without `experts_implementation="batched_mm"`.

---

## Script → result map

Every number in `results/measurements.json` names its producing script. This is the inverse.

### The performance headline

| script | produces | finding |
|---|---|---|
| `measure_mfu.py` | all MFU rows; `--fix-target-detection` toggles the #24 fix so both are reproducible | #20, #24 |
| `compare_mfu_runs.py` | amortisation: 2.59x more work costs 1.16x more per call | #24 |

### The root-cause chain — read in this order, it is the argument

| # | script | question | answer |
|---|---|---|---|
| 1 | `probe_neff_count.py` | do N NKI calls share one graph? | yes — 1 device execution, 196-node graph, and still 28x the cost |
| 2 | `profile_nki_call_cost.py` | what does the device do? | 0.609 ms for the whole 28-call NEFF, 43% MBU, 95% active |
| 3 | `probe_where_is_the_time.py` | host or device? | **99.9% host, before `mark_step`** |
| 4 | `probe_inside_one_call.py` | which function? | cProfile: 51 of 52 ms in `select.poll` under `subprocess.check_output` |
| 5 | `probe_target_override_fix.py` | does the fix work? | 51.74 → 0.49 ms/call, cos_sim unchanged |

Step 2 vs step 3 is the decisive comparison — 0.609 ms device against 1459 ms wall eliminates every
device-side explanation at once.

### Kernel quality

| script | produces |
|---|---|
| `profile_nki_vs_torch_device.py` | device time, NKI vs the torch op it replaces, one `(op, impl, N)` per run |
| `run_device_profile_sweep.py` | drives that across ops/impls/call-counts |
| `summarise_device_profiles.py` | reads NEFF+NTFF via `neuron-explorer`, tabulates |
| `analyse_fusion_barrier.py` | the `traffic(N) = FIXED + N × MARGINAL` regression — proves the kernels are at the unfused floor |
| `profile_model_device_time.py` + `sum_model_device_time.py` | **the in-situ split: 91.6% dispatch, 8.4% device** |
| `profile_fused_mlp_vs_torch.py` | fused MLP vs torch, 2.99x / 2.78x |
| `spike_nkilib_mlp.py` | the 10-point compile boundary (`I <= 4096`) |

### Diagnostics and environment

| script | question |
|---|---|
| `diagnose_torch_compile.py` | is `torch.compile` broken here? (no — only `torch_neuronx`-overridden ops) |
| `probe_compiler_flags.py` | **does the NKI/torch ratio depend on `NEURON_CC_FLAGS`?** Not yet run — needs hardware |
| `probe_nki_versions.py` | `nki` 0.5.0 vs bundled `neuronxcc.nki` |
| `probe_nki05_api.py` | what NKI 0.5.0 offers after `nl.arange` was removed |
| `probe_neuron_device_path.py` | can `use_kernels=True` reach `"neuron"`? (no) |
| `probe_hub_packaging.py`, `test_hub_structure.py` | what Hub publishing requires |
| `probe_nkilib_bundled.py`, `probe_mlp_signature.py` | is nkilib installed, and what's the MLP signature |
| `smoke_device.py` | minimal "is there a Neuron device" check |

### Harness

| script | purpose |
|---|---|
| `run_all_tests.py` | all 5 suites, one subprocess each so a crash can't mask the others |
| `regenerate_results.py` | 21 stages → `results/raw/`. **This is how you rebuild the evidence** |
| `render_results.py` | `measurements.json` → `results/README.md`. Runs anywhere |
| `sync_to_trn2.sh` | rsync local → trn2 |
| `run_detached.sh` | run under `nohup`; needed because full-model compiles exceed the SSH timeout |

### Superseded — kept deliberately, marked in their docstrings

| script | why it's still here |
|---|---|
| `experiment_torch_compile_nki.py` | its premise was wrong three ways. Kept because the guard it introduced — refuse to report a NKI result unless a plain-PyTorch control compiles first — is why it never emitted a false finding |
| `experiment_nki_graph_break.py` | the four framework-level experiments that supported the wrong hypothesis. The *ruling-out* work in it is what eventually made #24 findable |
| `benchmark_kernels.py` | first benchmark; its scaling gate is what caught Finding #19 |
| `tests/test_rmsnorm_accuracy.py` | the documented reproduction of the silent-fallback failure mode. Header says **DO NOT TRUST THIS TEST'S RESULTS** |
| `tests/test_qwen3_layer.py` | Week 2 e2e, RMSNorm only, predates the guards. The diff against the current suite shows what the guards added |

---

## Running things

```bash
make help              # all targets
make test-all          # 5 suites (needs trn2)
make results           # every measurement -> results/raw/ (30-60 min, needs trn2)
make results-render    # regenerate results/README.md from JSON (runs anywhere)
make rootcause         # replay the Finding #24 chain
make fusion            # replay Finding #25
make insitu            # replay the in-situ device/dispatch split
make lint              # byte-compile everything
```

Two operational notes that cost time to learn:

- **Never run two Neuron processes at once.** They contend for cores and you get
  `Requested:4 Available:0`. Every harness here is strictly sequential for this reason.
- **Full-model compiles exceed the SSH command timeout.** Use `run_detached.sh`, which execs under
  `nohup` with stdin closed.

---

## Conventions

- **Every script's docstring states what it measures, why the question mattered, and how to run it.**
  Several also record what a *previous version* of the script got wrong, because that is usually the
  reusable part.
- **Wrong conclusions are annotated in place, not deleted.** `docs/poc-findings.md` has superseded
  boxes on #18, #20, #21 and #25. The measurements in those findings are still correct; it was the
  attributions that were wrong, and seeing the difference is the point.
- **No number is reported without a control.** Accuracy suites include negative controls; timing
  comparisons re-run the baseline last; the fix verification checks accuracy on every variant.
