"""THE DECISIVE TEST: does the stock transformers use_kernels path reach Neuron on native?

Two independent gates blocked `use_kernels=True` on the torch-xla stack. Both are now claimed
to be XLA artifacts. This settles the second one end-to-end, on a real Qwen3 forward.

  Gate 1  kernels._backend() -> hasattr(torch, "neuron") was False, so a Neuron host was
          identified as CUDA(12.8) and validate_dependencies read the wrong section of
          python_depends.json, making an honest "python-depends": ["nki"] manifest illegal.
          Already confirmed gone on native: _backend().name == "neuron", ["nki"] validates.

  Gate 2  transformers' kernelize() computes Device(type=model.device.type) itself. On XLA that
          is "xla", which matches no mapping entry -- and because transformers passes a Device
          OBJECT, kernels skips _validate_device_type, so it is a SILENT no-op rather than an
          error. Every layer quietly keeps its original forward. This is the gate that actually
          blocked the feature, and the one tested here.

On native, model.device.type is "neuron", so stock kernelize() should compute Device("neuron")
and the mapping should fire with NO patching. The test therefore runs the REAL
transformers.integrations.hub_kernels.kernelize -- our enable_neuron_device_detection() shim is
deliberately NOT called, and the probe asserts it was never installed.

WHAT THIS DOES NOT SHOW, and must not be reported as showing. register_with_transformers()
still injects the "neuron" mapping entries from our local kernel dirs. Upstream transformers has
NO neuron entries in _KERNEL_MAPPING, so on a stock install use_kernels=True still does nothing
on Trainium -- not for device-routing reasons any more, but because there is nothing registered
to route TO. In the proposed-upstream-diff terms: Change 1 (device resolution) becomes
unnecessary on native; Change 2 (the mapping entries) is still required.

Execution is proved with call counters, never inferred from logits, per Finding #8: every kernel
here falls back to eager PyTorch when it cannot run, which keeps logits perfect while delivering
no acceleration.

    ./scripts/run_native.sh scripts/probe_native_use_kernels.py

Must go through run_native.sh, or neuronx-cc is off PATH and the first device op hangs.
"""

import json
import pathlib
import sys

import faulthandler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

HANG_TIMEOUT_S = 1200
SEQ_LEN = 128
DEV = "neuron"

RESULT = {"stack": "native", "torch": torch.__version__, "shim_used": False}


def sync():
    """Native has no mark_step; .cpu() on the consumer is what forces materialisation."""
    return


def main():
    from neuron_kernel_registration import register_with_transformers
    from nki_test_utils import cosine_similarity, max_abs_diff, nki_call_counter
    from test_qwen3_neuron_e2e import build_qwen3, count_rmsnorm_swaps, forward_overrides

    from kernels import get_local_kernel
    from transformers.integrations import hub_kernels

    print(f"torch {torch.__version__}")

    # --- gate 1, restated for the record -------------------------------------------------
    from kernels.utils import _backend

    backend = _backend()
    RESULT["gate1"] = {
        "backend_name": getattr(backend, "name", str(backend)),
        "hasattr_torch_neuron": hasattr(torch, "neuron"),
    }
    print(f"kernels._backend()          : {backend}")
    print(f"hasattr(torch, 'neuron')    : {hasattr(torch, 'neuron')}")

    # --- prove the shim is NOT installed -------------------------------------------------
    # If our patch were in place, hub_kernels.kernelize would be _patched_transformers_kernelize.
    kernelize_qualname = getattr(hub_kernels.kernelize, "__qualname__", "?")
    kernelize_module = getattr(hub_kernels.kernelize, "__module__", "?")
    is_stock = "patched" not in kernelize_qualname
    RESULT["kernelize_is_stock"] = is_stock
    RESULT["kernelize_qualname"] = f"{kernelize_module}.{kernelize_qualname}"
    print(f"hub_kernels.kernelize is    : {kernelize_module}.{kernelize_qualname}")
    print(f"  -> stock, unpatched       : {is_stock}")
    if not is_stock:
        raise RuntimeError("the shim is installed; this test is only meaningful without it")

    # --- build the model ------------------------------------------------------------------
    model, config = build_qwen3()
    model = model.to(DEV)
    ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN)).to(DEV)

    dev_type = model.device.type
    RESULT["model_device_type"] = dev_type
    print(f"\nmodel.device.type           : {dev_type}   <- what stock kernelize() reads")
    if dev_type != "neuron":
        print("  !! not 'neuron' -- Gate 2 would still apply on this stack")

    # reference logits before any swapping
    with torch.no_grad():
        ref_logits = model(ids).logits.cpu()

    # Snapshot the function-kernel forwards BEFORE kernelize, so the swap can be measured by
    # identity change rather than by qualname matching. Function kernels live in
    # module._hidden_kernels (populated once in __init__, never replaced), and kernelize()
    # mutates fn.forward in place. Comparing before/after is robust to however the kernels
    # library chooses to wrap the replacement.
    def hidden_kernel_forwards():
        snap = {}
        for path, m in model.named_modules():
            for name, fn in getattr(m, "_hidden_kernels", {}).items():
                fwd = getattr(fn, "forward", None)
                inner = getattr(fwd, "__func__", fwd)
                snap[f"{path}::{name}"] = (
                    id(inner),
                    getattr(inner, "__qualname__", repr(inner)),
                )
        return snap

    before_fwd = hidden_kernel_forwards()
    print(f"\nfunction-kernel slots found : {len(before_fwd)}")
    for k, (_, qn) in sorted(before_fwd.items()):
        print(f"    {k:56s} -> {qn}")

    # --- the actual test: stock kernelize(), neuron entries registered, no shim ----------
    print("\n>>> register neuron mapping entries, then call STOCK hub_kernels.kernelize()")
    register_with_transformers()
    hub_kernels.kernelize(model)

    overrides = forward_overrides(model)
    n_rms = count_rmsnorm_swaps(model)
    n_silu = sum(1 for qn in overrides.values() if "NeuronSiLU" in qn)

    # Function kernels cannot be counted from named_modules(), and the first version of this
    # probe wrongly reported "RoPE swapped: 0" because of it. Stock kernelize() ends with
    #     finally: model.apply(detach_hidden_kernels)
    # and detach does delattr(module, name), removing the submodule ALIAS. The object itself
    # survives in module._hidden_kernels, so count there instead. (Verified against the
    # installed transformers, hub_kernels.py:603-627.) The dispatch counters below are the
    # authoritative evidence either way.
    after_fwd = hidden_kernel_forwards()
    changed = [k for k in before_fwd if k in after_fwd and after_fwd[k][0] != before_fwd[k][0]]
    n_rope = len(changed)
    RESULT["swaps"] = {"rmsnorm": n_rms, "rope": n_rope, "silu": n_silu}
    # Record both sides. The qualname is IDENTICAL before and after -- the kernels library wraps
    # our function in a freshly generated `Func` module, so only object identity distinguishes
    # them. Recording just one side would read as "nothing changed".
    RESULT["rope_slots_changed"] = {
        k: {
            "before": {"id": before_fwd[k][0], "qualname": before_fwd[k][1]},
            "after": {"id": after_fwd[k][0], "qualname": after_fwd[k][1]},
        }
        for k in sorted(changed)
    }
    RESULT["swaps_note"] = (
        "rope is counted as the number of _hidden_kernels slots whose forward changed identity "
        "across kernelize(), NOT from named_modules(). Two reasons a module walk cannot see it: "
        "stock kernelize() detaches the submodule alias in its finally block "
        "(hub_kernels.py:603-627), and the swap mutates fn.forward rather than adding an "
        "instance attribute to a model submodule. The dispatch counters are authoritative."
    )

    n_layers = config.num_hidden_layers
    expected = {"rmsnorm": 4 * n_layers + 1, "rope": n_layers, "silu": n_layers}
    RESULT["swaps_expected"] = expected
    print(f"    RMSNorm swapped : {n_rms}  (expected {expected['rmsnorm']})")
    print(f"    RoPE    swapped : {n_rope}  (expected {expected['rope']})")
    print(f"    SiLU    swapped : {n_silu}  (expected {expected['silu']})")
    print("    swapped modules:")
    for path, qn in sorted(overrides.items()):
        print(f"        {path:40s} -> {qn}")

    # --- prove the kernels EXECUTE, not merely that forwards were replaced ---------------
    rms_mod = get_local_kernel(PROJECT_ROOT / "kernels" / "neuron_rmsnorm")
    rope_mod = get_local_kernel(PROJECT_ROOT / "kernels" / "neuron_rope")
    silu_mod = get_local_kernel(PROJECT_ROOT / "kernels" / "neuron_silu")

    print("\n>>> forward pass with call counters")
    with nki_call_counter(rms_mod, ["_nki_rmsnorm_kernel"], ["_pytorch_rmsnorm"]) as rc:
        with nki_call_counter(rope_mod, ["_nki_rope_hf"], ["_torch_rope"]) as pc:
            with nki_call_counter(silu_mod, ["_nki_silu_kernel"], ["_torch_silu"]) as sc:
                with torch.no_grad():
                    out_logits = model(ids).logits.cpu()

    cos = cosine_similarity(ref_logits, out_logits)
    diff = max_abs_diff(ref_logits, out_logits)
    RESULT["dispatch"] = {
        "rmsnorm": {"nki": rc.nki, "fallback": rc.fallback},
        "rope": {"nki": pc.nki, "fallback": pc.fallback},
        "silu": {"nki": sc.nki, "fallback": sc.fallback},
    }
    RESULT["logits_cos_sim"] = cos
    RESULT["logits_max_diff"] = diff

    print(f"    RMSNorm dispatch : {rc}")
    print(f"    RoPE    dispatch : {pc}")
    print(f"    SiLU    dispatch : {sc}")
    print(f"    logits cos_sim   : {cos:.6f}")
    print(f"    logits max_diff  : {diff:.3e}")

    # The verdict rests on the DISPATCH COUNTERS, not on the structural walk. Finding #8's
    # lesson generalises: what a module tree looks like after the fact is weaker evidence than
    # a count of which implementation actually ran. It also avoids the function-kernel blind
    # spot that made the first run of this probe report a false negative.
    exec_counts_ok = (
        rc.nki == expected["rmsnorm"]
        and pc.nki == expected["rope"]
        and sc.nki == expected["silu"]
    )
    exec_ok = rc.nki_ran and pc.nki_ran and sc.nki_ran
    swaps_ok = (
        n_rms == expected["rmsnorm"]
        and n_rope == expected["rope"]
        and n_silu == expected["silu"]
    )
    acc_ok = cos > 0.999
    passed = exec_ok and exec_counts_ok and acc_ok

    RESULT["gate2_gone"] = bool(exec_ok and exec_counts_ok)
    RESULT["passed"] = bool(passed)
    RESULT["structural_counts_ok"] = bool(swaps_ok)

    print("\n" + "=" * 76)
    print(f"all three NKI kernels executed             : {'YES' if exec_ok else 'NO'}")
    print(f"execution counts match expected            : {'YES' if exec_counts_ok else 'NO'}")
    print(f"logits match unkernelized model            : {'YES' if acc_ok else 'NO'}")
    print(f"  (supplementary) structural swap counts   : {'match' if swaps_ok else 'differ'}")
    print()
    if passed:
        print("GATE 2 IS GONE on the Native PyTorch stack. No device-resolution patch needed:")
        print("proposed-upstream Change 1 is unnecessary here.")
        print()
        print("STILL REQUIRED: Change 2, the neuron entries in transformers' _KERNEL_MAPPING.")
        print("This probe injected them via register_with_transformers(). On a stock install")
        print("use_kernels=True remains a no-op on Trainium -- now for lack of entries to route")
        print("to, not for lack of device routing.")
    else:
        print("Gate 2 NOT cleared. See counts above.")
    print("=" * 76)
    return 0 if passed else 1


if __name__ == "__main__":
    out = PROJECT_ROOT / "results/raw/native/native_use_kernels.json"
    faulthandler.dump_traceback_later(HANG_TIMEOUT_S, exit=True)
    code = 1
    try:
        code = main()
    except BaseException as e:  # noqa: BLE001
        RESULT["fatal"] = f"{type(e).__name__}: {e}"
        import traceback

        traceback.print_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(RESULT, indent=2) + "\n")
        print(f"\nwrote {out}")
    sys.exit(code)
