"""Neuron entries for the HF kernel mapping, plus the shim that makes them reachable.

This module is the Week 3 integration artifact. It does three separable things:

1. `build_neuron_mapping()` — the `"neuron"` mapping entries for `RMSNorm` and
   `rotary_pos_emb`. This is the payload that would be added to transformers'
   `_KERNEL_MAPPING` / `_FUNCTION_KERNEL_MAPPING` upstream. See
   `PROPOSED_UPSTREAM_DIFF` for the exact form that PR would take.

2. `kernelize_for_neuron()` — kernelizes a model for Neuron *today*, by calling the
   `kernels` library directly with an explicit `device="neuron"` and replicating the
   `_hidden_kernels` attach/detach dance that transformers' wrapper performs (needed
   for function kernels like RoPE).

3. `enable_neuron_device_detection()` — an in-process shim implementing the proposed
   upstream fix for Finding #9: map an XLA device that reports `NEURON` hardware onto
   `Device(type="neuron")`. This exists so we can *demonstrate* that the proposed fix
   is sufficient to make `use_kernels=True` work, rather than merely asserting it.

   It patches a function object in the imported module, in this process only. It does
   NOT modify anything on disk or in the installed venv. `disable_...` reverses it.

Why the shim rather than a real patch: modifying the shared venv would be
irreproducible for a customer and would misrepresent the integration's true state.
The gap is the finding; hiding it would remove the PoC's value.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
KERNEL_REPO_ROOT = PROJECT_ROOT / "kernels"

# Kernel name (as registered by the transformers decorators) -> local repo + symbol.
#
#   "RMSNorm"        <- @use_kernel_forward_from_hub("RMSNorm") on Qwen3RMSNorm
#                       110 model files in transformers 5.15.0.dev0
#   "rotary_pos_emb" <- @use_kernel_func_from_hub("rotary_pos_emb") on
#                       apply_rotary_pos_emb; 95 model files
NEURON_KERNELS = {
    "RMSNorm": {
        "kind": "layer",
        "repo": "neuron_rmsnorm",
        "symbol": "NeuronRMSNorm",
    },
    "rotary_pos_emb": {
        "kind": "func",
        "repo": "neuron_rope",
        "symbol": "apply_rotary_pos_emb",
    },
}


def build_neuron_mapping(only: list[str] | None = None) -> dict:
    """Build the `{kernel_name: {"neuron": Repository}}` mapping for our local kernels.

    Layer kernels resolve `kernel.layers.<symbol>`; function kernels resolve
    `<symbol>` at the module top level. That asymmetry is easy to get wrong, so the
    repo type is recorded explicitly in NEURON_KERNELS rather than inferred.
    """
    from kernels import LocalFuncRepository, LocalLayerRepository

    mapping: dict = {}
    for name, spec in NEURON_KERNELS.items():
        if only is not None and name not in only:
            continue
        repo_path = KERNEL_REPO_ROOT / spec["repo"]
        if not repo_path.exists():
            raise FileNotFoundError(f"kernel repo missing: {repo_path}")
        if spec["kind"] == "layer":
            repo = LocalLayerRepository(repo_path=repo_path, layer_name=spec["symbol"])
        else:
            repo = LocalFuncRepository(repo_path=repo_path, func_name=spec["symbol"])
        mapping[name] = {"neuron": repo}
    return mapping


# --------------------------------------------------------------------------
# Attach/detach of function kernels (mirrors transformers' kernelize wrapper)
# --------------------------------------------------------------------------

def _attach_hidden_kernels(module) -> None:
    """Temporarily register `_hidden_kernels` entries as real submodules.

    `@use_kernel_func_from_hub` turns a free function into an nn.Module instance, and
    `@use_kernelized_func` stashes it in the owning module's `_hidden_kernels` dict —
    deliberately NOT as a submodule. `kernelize()` only walks `named_modules()`, so
    the entries have to be registered for the duration of the call.
    """
    import torch.nn as nn

    for name, fn in getattr(module, "_hidden_kernels", {}).items():
        if name not in dict(module.named_children()):
            if not isinstance(fn, nn.Module):
                raise ValueError(
                    f"_hidden_kernels['{name}'] is not an nn.Module; the underlying "
                    "function must be decorated with @use_kernel_func_from_hub"
                )
            module.register_module(name, fn)


def _detach_hidden_kernels(module) -> None:
    for name in getattr(module, "_hidden_kernels", {}):
        if hasattr(module, name):
            delattr(module, name)


def kernelize_for_neuron(model, mode=None, only: list[str] | None = None, use_fallback: bool = True):
    """Kernelize `model` with our Neuron kernels. Works today, no upstream changes.

    Bypasses transformers' `kernelize()` because that wrapper derives the device from
    `model.device.type` and so can never reach `"neuron"` (Finding #9).
    """
    from kernels import Mode, kernelize, use_kernel_mapping

    if mode is None:
        mode = Mode.INFERENCE

    mapping = build_neuron_mapping(only=only)
    try:
        model.apply(_attach_hidden_kernels)
        with use_kernel_mapping(mapping, inherit_mapping=False):
            kernelize(model, device="neuron", mode=mode, use_fallback=use_fallback)
    finally:
        model.apply(_detach_hidden_kernels)
    return model


# --------------------------------------------------------------------------
# Finding #9 shim: make xla-on-Neuron resolve to Device(type="neuron")
# --------------------------------------------------------------------------

_ORIGINAL_FIND_DEVICE = None


def _kernelize_module():
    """Return the `kernels.layer.kernelize` MODULE, not the same-named function."""
    import importlib

    return importlib.import_module("kernels.layer.kernelize")


def _xla_is_neuron() -> bool:
    """True when the current XLA runtime is backed by Neuron hardware."""
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device_hw(xm.xla_device()) == "NEURON"
    except Exception:
        return False


_ORIGINAL_TF_KERNELIZE = None


def _patched_transformers_kernelize(model, mode=None):
    """transformers' `kernelize()` with the two-line Neuron fix applied.

    Faithful reproduction of `transformers/integrations/hub_kernels.py::kernelize`,
    with one added branch (marked THE FIX). Everything else — the mode defaulting,
    the rocm refinement, the kernel_config handling, the attach/detach, the
    `_use_kernels` flag — mirrors upstream so this demonstrates the patch rather
    than a different code path.
    """
    from kernels import Device, Mode, use_kernel_mapping
    from kernels import kernelize as _kernels_kernelize

    mode = Mode.INFERENCE if not model.training else Mode.TRAINING if mode is None else mode

    device_type = model.device.type
    try:
        from transformers.utils import is_rocm_platform

        rocm = is_rocm_platform()
    except Exception:
        rocm = False

    if device_type == "cuda" and rocm:
        device_type = "rocm"
    elif device_type == "xla" and _xla_is_neuron():
        device_type = "neuron"  # THE FIX
    device = Device(type=device_type)

    try:
        model.apply(_attach_hidden_kernels)
        if getattr(model, "kernel_config", None) is not None:
            inherit_mapping = not model.kernel_config.use_local_kernel
            with use_kernel_mapping(
                model.kernel_config.kernel_mapping, inherit_mapping=inherit_mapping
            ):
                _kernels_kernelize(model, device=device, mode=mode)
        else:
            _kernels_kernelize(model, device=device, mode=mode)
        model._use_kernels = True
    finally:
        model.apply(_detach_hidden_kernels)
    return model


def enable_neuron_device_detection() -> bool:
    """Apply the proposed upstream fix, in this process only.

    Patches TWO places, because they are independent and only the first one is on
    the `use_kernels=True` path:

    1. `transformers.integrations.hub_kernels.kernelize` — this is the one that
       matters. The transformers wrapper computes `Device(type=model.device.type)`
       itself and passes it to `kernels.kernelize(device=...)`, so it never calls
       `kernels._find_device` at all. Patching only the kernels library would have
       no effect on `use_kernels=True`. (We got this wrong on the first attempt;
       the test caught it.)

    2. `kernels.layer.kernelize._find_device` — for any caller that lets the
       kernels library auto-detect, e.g. `kernelize(model)` with no device.

    In-process only. Nothing on disk or in the venv is modified.
    """
    global _ORIGINAL_FIND_DEVICE, _ORIGINAL_TF_KERNELIZE
    from kernels import Device

    # 1. transformers entry point
    if _ORIGINAL_TF_KERNELIZE is None:
        from transformers.integrations import hub_kernels

        _ORIGINAL_TF_KERNELIZE = hub_kernels.kernelize
        hub_kernels.kernelize = _patched_transformers_kernelize

    # 2. kernels auto-detection
    #
    # NOTE: `import kernels.layer.kernelize as kz` does NOT give you the module —
    # `kernels/layer/__init__.py` re-exports the `kernelize` *function*, which
    # shadows the submodule of the same name. Go through importlib instead.
    if _ORIGINAL_FIND_DEVICE is None:
        kz = _kernelize_module()
        _ORIGINAL_FIND_DEVICE = kz._find_device
        original = _ORIGINAL_FIND_DEVICE

        def _find_device_with_neuron(model):
            try:
                param = next(model.parameters())
            except StopIteration:
                return original(model)
            if param.device.type == "xla" and _xla_is_neuron():
                return Device(type="neuron")
            return original(model)

        kz._find_device = _find_device_with_neuron

    return True


def disable_neuron_device_detection() -> None:
    """Remove both shims, restoring stock behaviour."""
    global _ORIGINAL_FIND_DEVICE, _ORIGINAL_TF_KERNELIZE
    if _ORIGINAL_TF_KERNELIZE is not None:
        from transformers.integrations import hub_kernels

        hub_kernels.kernelize = _ORIGINAL_TF_KERNELIZE
        _ORIGINAL_TF_KERNELIZE = None
    if _ORIGINAL_FIND_DEVICE is not None:
        _kernelize_module()._find_device = _ORIGINAL_FIND_DEVICE
        _ORIGINAL_FIND_DEVICE = None


def register_with_transformers(only: list[str] | None = None) -> dict:
    """Register the Neuron entries into the global kernel mapping.

    Approximates what importing transformers would do if the upstream PR in
    PROPOSED_UPSTREAM_DIFF had landed. Combined with
    `enable_neuron_device_detection()`, this makes `use_kernels=True` reach our
    kernels — which is how we verify the proposed fix is actually sufficient.
    """
    from kernels import register_kernel_mapping

    mapping = build_neuron_mapping(only=only)
    register_kernel_mapping(mapping)
    return mapping


# --------------------------------------------------------------------------
# Documentation artifact: what the upstream change looks like
# --------------------------------------------------------------------------

PROPOSED_UPSTREAM_DIFF = '''\
Two changes are needed for `use_kernels=True` to work on Trainium. Neither is
architectural; the interception points already exist and Qwen3 already opts in.

Change 1 has been VERIFIED sufficient: applying it in-process takes Qwen3 from
0 to 9 swapped RMSNorm layers via the transformers `use_kernels` path, with
logits cos_sim 1.000001. See tests/test_qwen3_neuron_e2e.py test 2.

--- CHANGE 1 (required) -------------------------------------------------------
transformers/integrations/hub_kernels.py — resolve XLA-on-Neuron to "neuron".

IMPORTANT: this must go in transformers, NOT in kernels._find_device. The
transformers wrapper computes the Device itself and passes it explicitly to
kernels.kernelize(device=...), so kernels._find_device is never consulted on
this path. Patching only the kernels library has no effect on use_kernels=True.
(We initially proposed the kernels-side fix; the e2e test disproved it.)

Note also that because transformers passes a Device *object* rather than a
string, kernels.kernelize skips _validate_device_type entirely. So the current
behaviour is not an error but a silent no-op: Device(type="xla") matches no
mapping entry and every layer quietly keeps its original forward.

     def kernelize(model: "PreTrainedModel", mode: "Mode | None" = None):
         ...
         device_type = model.device.type
         if device_type == "cuda" and is_rocm_platform():
             device_type = "rocm"
+        elif device_type == "xla" and _is_neuron_xla():
+            device_type = "neuron"
         device = Device(type=device_type)

+def _is_neuron_xla() -> bool:
+    try:
+        import torch_xla.core.xla_model as xm
+        return xm.xla_device_hw(xm.xla_device()) == "NEURON"
+    except Exception:
+        return False

Verified on trn2: xm.xla_device_hw(xm.xla_device()) returns exactly "NEURON",
so the check is reliable and needs no new dependency.

--- CHANGE 1b (recommended, same idea) ----------------------------------------
kernels/layer/kernelize.py — the equivalent fix for callers that let the kernels
library auto-detect, e.g. `kernelize(model)` with no device argument. Not on the
use_kernels=True path, but needed for direct kernels-library users.

     def _find_device(model: "nn.Module") -> Device:
         dev_type = param.device.type
         if dev_type == "cuda":
             ...
+        elif dev_type == "xla" and _is_neuron_xla():
+            return Device(type="neuron")
         return Device(type=dev_type)

--- CHANGE 2 (required) -------------------------------------------------------
transformers/integrations/hub_kernels.py — add the neuron entries.

"rotary_pos_emb" already has cuda / rocm / xpu entries; this adds a sibling.
Repo IDs assume the kernels are published under aws-neuron/ (open question:
kernels-community/ vs aws-neuron/ — needs the Samir conversation).

     _KERNEL_MAPPING = {
         ...
+        "RMSNorm": {
+            "neuron": LayerRepository(
+                repo_id="aws-neuron/rmsnorm",
+                layer_name="NeuronRMSNorm",
+                version=1,
+            )
+        },
     }

     _FUNCTION_KERNEL_MAPPING = {
         "rotary_pos_emb": {
             "cuda": FuncRepository(...),
             "rocm": {...},
             "xpu": {...},
+            "neuron": FuncRepository(
+                repo_id="aws-neuron/rope",
+                func_name="apply_rotary_pos_emb",
+                version=1,
+            ),
         },
     }

--- OPTIONAL / INDEPENDENTLY USEFUL ------------------------------------------
a) torch_neuronx should set a `torch.neuron` attribute. `_has_neuron_ops()` in
   kernels checks `hasattr(torch, "neuron")` and currently always returns False,
   even after `import torch_neuronx`. NOTE: this alone does NOT fix anything —
   it does not change what _find_device returns. Change 1 is still required.

b) kernels.kernelize's docstring lists supported devices as "cuda", "mps",
   "npu", "rocm", "xpu" — omitting both "neuron" and "cpu", which the code does
   support. A reader would conclude Neuron is unsupported.

c) A device override on transformers' kernelize(model, mode) would give callers
   an escape hatch, but Change 1 is the better fix: it makes auto-detection
   correct for every framework rather than requiring each to thread a parameter.
'''


def main():
    """Print the mapping we'd register and the proposed upstream diff."""
    print("=" * 76)
    print("Neuron kernel mapping entries")
    print("=" * 76)
    for name, spec in NEURON_KERNELS.items():
        print(f"  {name:16s} {spec['kind']:6s} kernels/{spec['repo']} :: {spec['symbol']}")
    print()
    try:
        mapping = build_neuron_mapping()
        for name, by_dev in mapping.items():
            print(f"  {name} -> neuron -> {by_dev['neuron']}")
    except Exception as e:
        print(f"  (could not build mapping: {type(e).__name__}: {e})")
    print()
    print(PROPOSED_UPSTREAM_DIFF)
    return 0


if __name__ == "__main__":
    sys.exit(main())
