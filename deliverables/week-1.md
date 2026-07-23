# Week 1 Deliverable: Neuron Device Path Verification

## Summary

Verified that the HuggingFace `kernels` library's `"neuron"` device path works end-to-end on a trn2.3xlarge instance. A stateless kernel layer swaps into a real `nn.Module.forward()` via `LocalLayerRepository`, with graceful fallback when no kernel is registered. `KernelConfig(use_local_kernel=True)` accepts a `"neuron"` mapping as the project doc specified.

## How kernelize() works (the flow)

`kernelize(model, device="neuron", mode=Mode.INFERENCE)` walks every module in the model's tree. For each module, it checks whether that module's class has been annotated with `@use_kernel_forward_from_hub("SomeName")`. If it has, `kernelize` looks up `"SomeName"` in the active kernel mapping and checks if there's an entry for the current device (`"neuron"`). If a match is found, it loads the kernel class from the registered repository (Hub or local), instantiates it, and replaces the module's `forward()` method with the kernel class's `forward()`. The original module's parameters (`self.weight`, etc.) remain in place — the kernel's `forward` reads them via `self` because it's effectively bound to the original module. If no match is found, the module keeps its original `forward()` unchanged (fallback). This is a purely eager-mode mechanism today — no graph tracing, no compilation, just a method pointer swap.

## Test Results (trn2.3xlarge, 2026-07-22)

| Test | Result |
|------|--------|
| `kernelize(device="neuron")` accepted without error | ✓ PASS |
| `LocalLayerRepository` loads local kernel package | ✓ PASS |
| Forward swap fires (output changes from sentinel to real values) | ✓ PASS |
| Fallback works (unmapped layers keep original forward) | ✓ PASS |
| `KernelConfig(use_local_kernel=True)` accepts neuron mapping | ✓ PASS |
| Device-specific format `{"RMSNorm": {"neuron": "..."}}` accepted | ✓ PASS |

## Versions Tested

| Package | Version | Source |
|---------|---------|--------|
| `kernels` | 0.15.2 | PyPI |
| `transformers` | 5.15.0.dev0 | GitHub main, commit `bb3ffb9703e3acb84f06db1d3799756e977662c2` |
| `torch` | 2.9.1+cu128 | Neuron DLAMI (`/opt/aws_neuronx_venv_pytorch_2_9`) |
| `torch_neuronx` | available | Neuron DLAMI |
| `neuronx-cc` | 2.26.6360.0+6f180f47 | Neuron DLAMI |
| Python | 3.12.3 | Ubuntu 24.04 |
| Instance | trn2.3xlarge | 1 Neuron device, 4 NeuronCores, 96 GB HBM |

## What was validated

- The `"neuron"` device path is live in `kernels` v0.15.2 (the `backend.type: "neuron"` is an accepted value in both `metadata.json` and `kernelize(device=...)`).
- `LocalLayerRepository` loads a local kernel package given `repo_path` + `layer_name` + a valid `metadata.json`.
- Stateless kernel layers (no `__init__`, only `forward()`, reads state via `self.weight` from the adopting module) work correctly.
- The `KernelConfig` transformers API (`use_local_kernel=True`) resolves local paths and triggers the swap.
- Fallback is graceful — layers without a registered kernel keep their original `forward()`.

## What was NOT validated (deferred to Week 2+)

- Actual NKI kernel execution on NeuronCores (current `NeuronRMSNorm` is a PyTorch reference impl, not `@nki.jit`)
- Real model integration (Qwen3 layer, not a toy model)
- Accuracy comparison against reference output
- `has_backward = True` / training mode
- `can_torch_compile = True`

## Key Findings

1. **Kernel can be very simple** — a single Python file with a class defining `forward()` + a `layers` namespace class + `metadata.json`. No compiled artifacts needed for NKI (it's Python via `@nki.jit`).
2. **`metadata.json` is required even for local dev** — `python-depends`, `digest`, and `backend` fields all mandatory. Not obvious from the docs.
3. **`LocalLayerRepository` API changed** — docs show `package_name` arg but v0.15.2 removed it. Pre-1.0 library, APIs shift between minors.
4. **No `kernel-builder` for Neuron** — the `"neuron"` backend type is accepted in metadata but there's no build tooling to produce a publishable Hub artifact. This is the #1 gap for the kernels team.
5. **Two loading paths exist** — `kernels` library directly (`use_kernel_mapping` + `kernelize`) and transformers `KernelConfig`. Both work; `KernelConfig` is the intended user-facing API.

## Scripts

- `scripts/verify_neuron_path.py` — validates all 4 core goals (kernelize, LocalLayerRepository, forward swap, fallback)
- `scripts/verify_kernel_config.py` — validates `KernelConfig(use_local_kernel=True)` with neuron mapping
- `scripts/demo_identity_swap.py` — visual demo of the identity kernel swap

## Next (Week 2)

Replace the PyTorch reference implementation in `NeuronRMSNorm.forward()` with the actual NKI RMSNorm kernel from `nki_samples`. Validate accuracy (cosine sim > 0.999) against `Qwen3RMSNorm` output on trn2.
