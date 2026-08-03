"""NKI SiLU accuracy on Neuron hardware, with execution proof.

SiLU is elementwise like RoPE, so bit-identical output is the expected result
rather than a warning sign. Negative controls included for the same reason as in
test_rope_nki.py: prove the comparison can fail.

Run on trn2:
    python tests/test_silu_nki.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

from nki_test_utils import (
    AccuracyResult,
    CallCounts,
    assert_nki_accuracy,
    cosine_similarity,
    load_kernel_module,
    max_abs_diff,
    nki_call_counter,
    report,
    require_neuron,
    sync,
    tol_for_dtype,
)

NKI_NAMES = ["_nki_silu_kernel"]
FALLBACK_NAMES = ["_torch_silu"]


def reference_silu(x):
    """transformers' SiLUActivation.forward."""
    return nn.functional.silu(x)


def run_case(mod, device, shape, dtype=torch.float32):
    tag = f"{tuple(shape)} {str(dtype).replace('torch.', '')}"
    torch.manual_seed(0)
    x_cpu = torch.randn(*shape, dtype=dtype)
    golden = reference_silu(x_cpu)

    layer = mod.layers.NeuronSiLU().to(device)
    with nki_call_counter(mod, NKI_NAMES, FALLBACK_NAMES) as counts:
        with torch.no_grad():
            out = layer(x_cpu.to(device))
        sync()
        out_cpu = out.cpu()

    return assert_nki_accuracy(
        tag, golden, out_cpu, counts,
        max_diff_tol=tol_for_dtype(dtype),
        expect_bit_identical=True,
    )


def test_discrimination(mod, device):
    """Prove the comparison can fail (see test_rope_nki.py for rationale)."""
    print()
    print("-" * 76)
    print("Discrimination checks (negative controls)")
    print("-" * 76)

    torch.manual_seed(0)
    x = torch.randn(1, 64, 256)
    layer = mod.layers.NeuronSiLU().to(device)
    with torch.no_grad():
        out = layer(x.to(device))
    sync()
    out_cpu = out.cpu()

    # A. vs a different activation (GELU) -> must be rejected
    gelu_ref = nn.functional.gelu(x)
    cos_gelu = cosine_similarity(gelu_ref, out_cpu)
    a_ok = cos_gelu < 0.999
    print(f"  A. vs GELU reference  : cos_sim={cos_gelu:.6f}  "
          f"-> {'discriminates' if a_ok else 'DOES NOT DISCRIMINATE'}")

    # B. vs the raw input (proves the activation was applied)
    cos_ident = cosine_similarity(x, out_cpu)
    b_ok = cos_ident < 0.999
    print(f"  B. vs raw input       : cos_sim={cos_ident:.6f}  "
          f"-> {'activation applied' if b_ok else 'KERNEL IS A NO-OP'}")

    # C. vs correct reference
    silu_ref = reference_silu(x)
    cos_ok = cosine_similarity(silu_ref, out_cpu)
    c_ok = cos_ok > 0.999
    print(f"  C. vs correct SiLU    : cos_sim={cos_ok:.6f}  "
          f"-> {'matches' if c_ok else 'MISMATCH'}")

    ok = a_ok and b_ok and c_ok
    print(f"  {'PASS' if ok else 'FAIL'} — comparison is meaningful")
    return AccuracyResult(
        label="discrimination (negative controls)",
        cos_sim=cos_ok,
        max_diff=max_abs_diff(silu_ref, out_cpu),
        counts=CallCounts(nki=1, fallback=0),
        passed=ok,
    )


def test_fallback_is_loud(mod, device):
    """A too-wide last dim must fall back, warn, and stay correct."""
    print()
    print("-" * 76)
    print(f"Fallback behaviour (last dim > MAX_FREE_DIM = {mod.MAX_FREE_DIM})")
    print("-" * 76)

    width = mod.MAX_FREE_DIM + 128
    torch.manual_seed(0)
    x = torch.randn(1, 2, width)
    golden = reference_silu(x)

    mod._warned.clear()
    layer = mod.layers.NeuronSiLU().to(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with nki_call_counter(mod, NKI_NAMES, FALLBACK_NAMES) as counts:
            with torch.no_grad():
                out = layer(x.to(device))
            sync()
            out_cpu = out.cpu()

    warned = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    cos = cosine_similarity(golden, out_cpu)
    took_fallback = counts.fallback > 0 and counts.nki == 0

    print(f"  dispatch        : {counts}")
    print(f"  took fallback   : {took_fallback}")
    print(f"  warning emitted : {len(warned) > 0}")
    if warned:
        print(f"  warning text    : {str(warned[0].message)[:95]}")
    print(f"  still correct   : cos_sim={cos:.6f}")

    ok = took_fallback and len(warned) > 0 and cos > 0.999
    print(f"  {'PASS' if ok else 'FAIL'}")
    return AccuracyResult(
        label="fallback loud + correct (wide last dim)",
        cos_sim=cos,
        max_diff=max_abs_diff(golden, out_cpu),
        counts=counts,
        passed=ok,
    )


def main():
    device = require_neuron()
    mod = load_kernel_module("neuron_silu")

    if not mod._HAS_NKI:
        print("NKI unavailable — cannot validate the kernel.")
        return 1

    shapes = [
        (1, 8, 64),
        (2, 16, 128),
        (1, 128, 896),
        (1, 64, 4864),      # Qwen3-0.6B intermediate_size
        (1, 32, 12288),     # Qwen3-8B intermediate_size
        (1, 1, 256),        # single token
        (1, 250, 512),      # non-multiple-of-128 rows
        (4, 64, 1024),      # larger batch
    ]

    results = []
    for shape in shapes:
        try:
            results.append(run_case(mod, device, shape))
        except Exception as e:
            import traceback

            print(f"  ERROR {shape}: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append(
                AccuracyResult(
                    label=str(tuple(shape)),
                    cos_sim=0.0,
                    max_diff=float("inf"),
                    counts=CallCounts(),
                    passed=False,
                    notes=[f"exception: {type(e).__name__}"],
                )
            )

    try:
        results.append(run_case(mod, device, (1, 128, 896), dtype=torch.bfloat16))
    except Exception as e:
        print(f"  bf16 case failed: {type(e).__name__}: {e}")

    ok = report(results, "NKI SiLU on Neuron hardware (execution-verified)")
    disc = test_discrimination(mod, device)
    fb = test_fallback_is_loud(mod, device)
    return 0 if (ok and disc.passed and fb.passed) else 1


if __name__ == "__main__":
    sys.exit(main())
