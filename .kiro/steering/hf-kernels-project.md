# HuggingFace Kernels on Neuron — Project Context

## What this project is

A 6-week internship PoC: package NKI kernels from `nki_samples` for the HuggingFace `kernels` library (Kernel Hub) and validate that Qwen3 dense runs end-to-end on Trainium with `use_kernels=True` swapping in NKI-accelerated layers. Final deliverable is a PoC document for the HF kernels team.

## Key concepts

- **KERNEL_MAPPING**: dict of `(layer_class_name, device) → kernel_impl`. Adding `"neuron"` entries gives all HF models with RMSNorm/RoPE/SiLU access to NKI kernels automatically.
- **Neuron device path**: the routing branch in `kernels` lib that selects `_NeuronRepos` when on Neuron hardware. Already merged to transformers mainline.
- **kernelize() flow**: walks model tree, matches layer names against KERNEL_MAPPING for current device, hot-swaps `forward()` method pointers. Module weights stay in place.
- **LocalLayerRepository**: local on-disk kernel repo for development without Hub publishing.
- **Stateless kernel**: pure function that reads weights from the existing module, no own state. Must subclass `nn.Module`, define only `forward()`, and declare `has_backward` / `can_torch_compile`.
- **@use_kernel_forward_from_hub("Name")**: decorator on a layer class that makes it swappable. The string name is looked up in KERNEL_MAPPING.
- **use_kernel_mapping context manager**: scopes a mapping for testing without polluting global state.

## Repo layout

```
kernels/
  neuron_identity/    # Week 1 PoC kernel (identity-scale)
  neuron_rmsnorm/     # Week 2 kernel (RMSNorm from nki_samples)
  neuron_rope/        # Week 3 kernel (RoPE)
  neuron_silu/        # Week 4 kernel (SiLU activation)
scripts/
  verify_neuron_path.py   # Validates all 4 Week 1 goals
  demo_identity_swap.py   # Minimal forward-swap demo
tests/                    # pytest accuracy + integration tests
docs/                     # PoC document (Week 6 deliverable)
```

## How to add a new kernel

1. Create `kernels/neuron_<name>/` with `__init__.py`, `layers.py`, and the NKI kernel file
2. In `layers.py`: subclass `nn.Module`, set `has_backward = False`, `can_torch_compile = False`, annotate expected state with type hints, implement `forward()`
3. In `__init__.py`: `from . import layers` and `__all__ = ["layers"]`
4. Register in test scripts via `LocalLayerRepository(repo_path=..., package_name="neuron_<name>", layer_name="Neuron<Name>")`
5. For production: add `"neuron"` entry to `_KERNEL_MAPPING` in transformers (Week 3+)

## Design decisions

- Start `has_backward = False` (inference only) unless nki_samples already has a backward kernel
- Start `can_torch_compile = False` (eager mode only) — Kernel Hub forward-swap works in eager today
- Use reference PyTorch impl first (validates plumbing), then swap in NKI kernel on trn2
- The NKI kernel file uses conditional import (`try: import neuronxcc.nki`) so code is testable off-device

## Target model: Qwen3 dense

- RMSNorm: `Qwen3RMSNorm` — manual implementation (not `torch.nn.functional.rms_norm`), reads `self.weight` and `self.variance_epsilon`
- RoPE: `apply_rotary_pos_emb` — a function, so needs `FuncRepository` not `LayerRepository`
- SiLU: used in MLP gating layer

## Accuracy targets

- Cosine similarity > 0.999 against reference layer output
- For e2e: loss or logits parity vs CPU/CUDA golden reference

## Week-by-week plan

1. Verify neuron device path + minimal identity kernel swap (current)
2. NKI RMSNorm kernel, validate on Qwen3 dense layer
3. Package for Hub, add RoPE, register neuron entries in KERNEL_MAPPING
4. Add SiLU, full Qwen3 dense e2e, measure MFU
5. (Stretch) Qwen3-MoE extension
6. PoC document, review, ship

## Coordination

- Samir (arsamir): HF kernels team contact, Hub repo home decision
- Pinak (panpinak): SA team reviewer
- Hanbo Wang / Karthick Gopalswamy: kernels team (PoC recipients)
- Matt (mmcclean): final deliverable recipient
