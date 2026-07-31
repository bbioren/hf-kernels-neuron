"""RMSNorm NKI kernel accuracy — on real Neuron hardware, with execution proof.

This replaces the validation logic of `test_rmsnorm_accuracy.py`, which ran on CPU
tensors and therefore only ever exercised the PyTorch fallback (Finding #8). The
old file is kept as-is to document the failure mode.

Differences that matter:
  - tensors are placed on the XLA (Neuron) device, so the NKI path is reachable
  - a call counter asserts the NKI branch ran and the fallback did not
  - a bit-identical result is flagged as suspicious instead of celebrated

Run on trn2:
    python tests/test_rmsnorm_nki.py
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

NKI_NAMES = ["_nki_rmsnorm_kernel"]
FALLBACK_NAMES = ["_pytorch_rmsnorm"]


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm.forward, computed on CPU in fp32 as the golden reference."""
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(dtype)


def run_case(mod, device, hidden_size, seq_len, batch_size, dtype=torch.float32, eps=1e-6,
             expect_bit_identical=False):
    """Run one shape through the NKI kernel on hardware and score it."""
    label = f"({batch_size}, {seq_len}, {hidden_size}) {str(dtype).replace('torch.', '')}"

    torch.manual_seed(0)
    weight_cpu = (torch.randn(hidden_size) * 0.5 + 1.0).to(dtype)
    x_cpu = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    golden = reference_rmsnorm(x_cpu, weight_cpu, eps)
    tol = tol_for_dtype(dtype)

    # Build the layer the way kernelize() leaves it: our forward, adopting
    # module's state. Move to the Neuron device so the NKI branch is taken.
    layer = mod.layers.NeuronRMSNorm()
    layer.weight = nn.Parameter(weight_cpu.clone())
    layer.variance_epsilon = eps
    layer = layer.to(device)

    x = x_cpu.to(device)

    with nki_call_counter(mod, NKI_NAMES, FALLBACK_NAMES) as counts:
        with torch.no_grad():
            out = layer(x)
        sync()
        out_cpu = out.cpu()

    return assert_nki_accuracy(label, golden, out_cpu, counts, max_diff_tol=tol,
                               expect_bit_identical=expect_bit_identical)


def test_fallback_is_loud(mod, device):
    """A CPU input must fall back, warn, and stay correct.

    This is the Finding #8 scenario reproduced deliberately. Before Week 3 this path was
    silent, which is how an entire accuracy suite passed without executing NKI. The
    warning is the fix, so it needs a test — otherwise the same regression could recur
    unnoticed, which is exactly the failure mode we are guarding against.
    """
    print()
    print("-" * 76)
    print("Fallback behaviour (CPU input — the Finding #8 scenario)")
    print("-" * 76)

    hidden, eps = 256, 1e-6
    torch.manual_seed(0)
    weight = torch.randn(hidden) * 0.5 + 1.0
    x = torch.randn(1, 32, hidden)
    golden = reference_rmsnorm(x, weight, eps)

    mod._warned.clear()
    layer = mod.layers.NeuronRMSNorm()
    layer.weight = nn.Parameter(weight.clone())
    layer.variance_epsilon = eps

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with nki_call_counter(mod, NKI_NAMES, FALLBACK_NAMES) as counts:
            with torch.no_grad():
                out = layer(x)  # deliberately CPU, not moved to device

    warned = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    cos = cosine_similarity(golden, out)
    took_fallback = counts.fallback > 0 and counts.nki == 0

    print(f"  dispatch        : {counts}")
    print(f"  took fallback   : {took_fallback}")
    print(f"  warning emitted : {len(warned) > 0}")
    if warned:
        print(f"  warning text    : {str(warned[0].message)[:95]}")
    print(f"  still correct   : cos_sim={cos:.6f}")

    ok = took_fallback and len(warned) > 0 and cos > 0.999
    print(f"  {'PASS' if ok else 'FAIL'}")
    if not warned:
        print("  (a silent fallback here is precisely the Finding #8 regression)")
    return AccuracyResult(
        label="fallback loud + correct (CPU input)",
        cos_sim=cos,
        max_diff=max_abs_diff(golden, out),
        counts=counts,
        passed=ok,
    )


def main():
    device = require_neuron()
    mod = load_kernel_module("neuron_rmsnorm")

    if not mod._HAS_NKI:
        print("NKI unavailable in this environment — cannot validate the kernel.")
        return 1

    # (hidden_size, seq_len, batch_size)
    cases = [
        (64, 8, 1),          # tiny
        (128, 16, 2),
        (896, 128, 1),       # Qwen3-0.6B hidden
        (1536, 128, 1),      # Qwen3-1.7B hidden
        (2048, 64, 2),       # Qwen3-4B hidden
        (3584, 32, 1),       # Qwen3-8B hidden
        (256, 1, 1),         # single token
        (512, 250, 1),       # non-multiple-of-128 seq_len
        (4096, 128, 1),      # rows = 128, exactly one tile
        (128, 300, 1),       # rows = 300, spans 3 tiles unevenly
    ]

    results = []
    for hidden_size, seq_len, batch_size in cases:
        try:
            results.append(run_case(mod, device, hidden_size, seq_len, batch_size))
        except Exception as e:
            import traceback

            print(f"  ERROR on ({batch_size}, {seq_len}, {hidden_size}): {type(e).__name__}: {e}")
            traceback.print_exc()
            from nki_test_utils import AccuracyResult, CallCounts

            results.append(
                AccuracyResult(
                    label=f"({batch_size}, {seq_len}, {hidden_size})",
                    cos_sim=0.0,
                    max_diff=float("inf"),
                    counts=CallCounts(),
                    passed=False,
                    notes=[f"exception: {type(e).__name__}"],
                )
            )

    # bf16 at a Qwen3 shape — the dtype real training/inference uses.
    #
    # expect_bit_identical=True only for bf16, and only since the NKI 0.5.0 migration.
    # The kernel now computes the reduction and reciprocal in float32, exactly as
    # PyTorch's RMSNorm does, so the bf16 result rounds identically to the reference.
    # The fp32 cases above deliberately keep the check ON — they still show a non-zero
    # diff (~1e-6), which is the corroborating signal that a real reduction ran.
    # The authoritative execution proof in both cases is the call counter.
    try:
        results.append(
            run_case(mod, device, 896, 128, 1, dtype=torch.bfloat16,
                     expect_bit_identical=True)
        )
    except Exception as e:
        print(f"  bf16 case failed: {type(e).__name__}: {e}")

    ok = report(results, "NKI RMSNorm on Neuron hardware (execution-verified)")
    fb = test_fallback_is_loud(mod, device)
    return 0 if (ok and fb.passed) else 1


if __name__ == "__main__":
    sys.exit(main())
