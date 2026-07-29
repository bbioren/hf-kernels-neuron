"""Per-kernel latency: NKI vs eager PyTorch on Neuron. WITH A VALIDITY GATE.

WHY: in `kernels/neuron_silu/__init__.py` and the Week 3 deliverable I predicted RMSNorm and
RoPE would help while standalone SiLU would not, and wrote "do not claim a win without
measuring." This measures it.

WHY THE VALIDITY GATE EXISTS (v1 of this script was wrong, and silently so):

The first version reported that every NKI kernel was 8-400x *slower* than eager. Those
numbers were meaningless. The tell was that latency did not change with tensor size —
RMSNorm measured 0.55 ms at both S=128 and S=2048, a 16x difference in data, and the eager
side sat at 0.07 ms throughout. Timing that is independent of problem size is not measuring
the problem.

Two causes:
  1. **Dead code elimination.** The benchmark discarded each output. XLA is lazy, so at
     `mark_step()` there was no live result and the computation was never performed. It was
     timing an empty graph.
  2. **Host-side overhead swamping everything.** Whatever remained was dispatch cost, which
     is fixed per call and therefore identical across shapes.

This is the same failure mode as Finding #8 in a different costume: a measurement that
produces plausible-looking numbers while not exercising the thing under test. So this version
**refuses to report** unless latency demonstrably scales with problem size.

Fixes applied:
  - Outputs are consumed (`.sum().item()`), which forces materialization and a real sync.
  - A scaling check runs first: time the same op at 1x and 8x data. If the larger problem is
    not measurably slower, results are declared overhead-dominated and suppressed.
  - Host-side enqueue cost is measured separately from end-to-end latency, so per-call
    dispatch overhead is visible rather than silently folded into "the kernel is slow".

WHAT THIS IS NOT: not MFU, not end-to-end model throughput, and it says nothing about graph
mode (these kernels declare `can_torch_compile = False`). Week 4 full-model MFU is the number
that matters.

Run on trn2:
    python scripts/benchmark_kernels.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from nki_test_utils import load_kernel_module, require_neuron

SEP = "=" * 86
WARMUP = 5
ITERS = 20

# The larger problem must be at least this much slower for the measurement to be
# considered size-sensitive rather than overhead-dominated.
SCALING_THRESHOLD = 1.25


def mark():
    import torch_xla.core.xla_model as xm

    xm.mark_step()


def consume(out):
    """Force the result to be computed and read back.

    Without this XLA has no live output and eliminates the computation. `.sum()` is cheap
    relative to the ops under test at these sizes; `.item()` forces the sync.
    """
    if isinstance(out, (tuple, list)):
        total = None
        for t in out:
            s = t.float().sum()
            total = s if total is None else total + s
        return total.item()
    return out.float().sum().item()


def time_end_to_end(fn, iters=ITERS, warmup=WARMUP):
    """Median / IQR per-call latency in ms, output consumed so it actually executes."""
    for _ in range(warmup):
        consume(fn())
        mark()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        consume(out)
        samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    return statistics.median(samples), samples[len(samples) // 4], samples[(3 * len(samples)) // 4]


def time_enqueue(fn, iters=ITERS, warmup=WARMUP):
    """Host-side cost of issuing the call, without waiting for the device.

    Isolates per-call Python/dispatch overhead. If this is large for the NKI path it is a
    finding in itself: eager-mode NKI would be launch-bound regardless of kernel quality.
    """
    for _ in range(warmup):
        fn()
    mark()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    mark()
    samples.sort()
    return statistics.median(samples)


def scaling_check(label, make_fn, small_args, large_args, ratio):
    """Verify latency responds to problem size. Returns (valid, detail)."""
    small = time_end_to_end(make_fn(*small_args), iters=10, warmup=3)[0]
    large = time_end_to_end(make_fn(*large_args), iters=10, warmup=3)[0]
    observed = large / small if small > 0 else float("nan")
    valid = observed >= SCALING_THRESHOLD
    detail = (f"{small:.2f}ms at 1x vs {large:.2f}ms at {ratio}x data "
              f"-> {observed:.2f}x")
    print(f"    {label:26s} {detail}   "
          f"{'size-sensitive' if valid else 'OVERHEAD-DOMINATED'}")
    return valid, observed


def main():
    dev = require_neuron()
    print(SEP)
    print("Per-kernel latency: NKI vs eager PyTorch on Neuron (bf16)")
    print(SEP)
    print(f"  warmup={WARMUP} iters={ITERS}; outputs consumed via .sum().item()")
    print(f"  validity gate: larger problem must be >= {SCALING_THRESHOLD}x slower")

    # ------------------------------------------------------------------
    print()
    print("VALIDITY GATE — does measured latency respond to problem size?")
    print()

    silu_mod = load_kernel_module("neuron_silu")

    def make_silu_nki(S, I):
        x = torch.randn(1, S, I, dtype=torch.bfloat16).to(dev)
        layer = silu_mod.layers.NeuronSiLU().to(dev)
        with torch.no_grad():
            return lambda: layer(x)

    def make_silu_eager(S, I):
        x = torch.randn(1, S, I, dtype=torch.bfloat16).to(dev)
        with torch.no_grad():
            return lambda: F.silu(x)

    nki_valid, nki_ratio = scaling_check(
        "SiLU NKI", make_silu_nki, (128, 3072), (1024, 3072), 8
    )
    eager_valid, eager_ratio = scaling_check(
        "SiLU eager", make_silu_eager, (128, 3072), (1024, 3072), 8
    )

    print()
    print("  Host-side enqueue cost (no device wait) — isolates per-call dispatch overhead:")
    nki_enq = time_enqueue(make_silu_nki(128, 3072))
    eag_enq = time_enqueue(make_silu_eager(128, 3072))
    print(f"    SiLU NKI   enqueue {nki_enq:8.3f} ms/call")
    print(f"    SiLU eager enqueue {eag_enq:8.3f} ms/call")
    if nki_enq > 1.0:
        print(f"    => NKI dispatch costs ~{nki_enq:.1f} ms of HOST time per call.")
        print("       At that level an eager per-layer swap is launch-bound: the kernel")
        print("       cannot win no matter how good the device code is. This is a finding")
        print("       about the eager integration model, not about the kernel.")

    print()
    print(SEP)
    if not (nki_valid and eager_valid):
        print("RESULT SUPPRESSED — measurement is overhead-dominated")
        print(SEP)
        print("  Latency did not scale with problem size on at least one path:")
        print(f"    NKI   scaling {nki_ratio:.2f}x for 8x data  "
              f"({'ok' if nki_valid else 'FAILS gate'})")
        print(f"    eager scaling {eager_ratio:.2f}x for 8x data  "
              f"({'ok' if eager_valid else 'FAILS gate'})")
        print()
        print("  Reporting NKI-vs-eager ratios from this data would be misleading: they")
        print("  would describe fixed per-call overhead, not kernel performance. Any")
        print("  'Nx slower' number derived here is an artifact.")
        print()
        print("  What this does legitimately establish:")
        print(f"    - NKI host-side dispatch is ~{nki_enq:.1f} ms/call vs ~{eag_enq:.3f} ms")
        print("      for eager. In eager mode, per-layer NKI kernels are launch-bound.")
        print("    - Per-layer microbenchmarking is the wrong instrument here. The")
        print("      meaningful measurement is full-model MFU, where launch cost is")
        print("      amortized differently and XLA schedules the whole graph.")
        print()
        print("  NEXT: do the Week 4 full-model MFU measurement instead of trying to")
        print("  rescue this microbenchmark. See docs/poc-findings.md Finding #19.")
        print(SEP)
        return 2

    print("Validity gate PASSED — latency responds to problem size on both paths")
    print(SEP)

    # ------------------------------------------------------------------
    # Fixed/variable decomposition. This is the informative analysis: with two
    # measurements at known data ratio r, solve T = fixed + k*work.
    #   T1 = fixed + k*w        T2 = fixed + k*r*w
    #   k*w = (T2 - T1)/(r - 1)     fixed = T1 - k*w
    # ------------------------------------------------------------------
    def decompose(t_small, t_large, ratio):
        variable_small = (t_large - t_small) / (ratio - 1)
        fixed = t_small - variable_small
        pct_fixed = 100.0 * fixed / t_small if t_small > 0 else float("nan")
        return fixed, variable_small, pct_fixed

    print()
    print("Fixed vs size-dependent cost (SiLU, S=128 -> S=1024, 8x data)")
    print()

    silu_nki_small = time_end_to_end(make_silu_nki(128, 3072))[0]
    silu_nki_large = time_end_to_end(make_silu_nki(1024, 3072))[0]
    silu_eag_small = time_end_to_end(make_silu_eager(128, 3072))[0]
    silu_eag_large = time_end_to_end(make_silu_eager(1024, 3072))[0]

    for name, ts, tl in (
        ("NKI", silu_nki_small, silu_nki_large),
        ("eager", silu_eag_small, silu_eag_large),
    ):
        fixed, var, pct = decompose(ts, tl, 8)
        print(f"  {name:6s} total {ts:6.2f}ms at S=128  =  fixed {fixed:6.2f}ms "
              f"+ size-dependent {var:6.3f}ms   ({pct:.0f}% fixed)")

    print()
    print("  Both paths are dominated by fixed cost at these sizes, and part of that fixed")
    print("  cost is this harness (the .sum().item() sync applies to both). So the honest")
    print("  clean signal is the enqueue delta measured above, not the totals.")

    # ------------------------------------------------------------------
    print()
    print(SEP)
    print("THE ACTIONABLE NUMBER: host-side dispatch overhead")
    print(SEP)
    delta = nki_enq - eag_enq
    print(f"  NKI   {nki_enq:.3f} ms/call")
    print(f"  eager {eag_enq:.3f} ms/call")
    print(f"  delta {delta:.3f} ms/call of extra HOST time per NKI kernel invocation")
    print()
    print("  What that costs a real model. Qwen3-8B has 36 layers, and per layer our")
    print("  kernels are invoked: 4 RMSNorm (input, post-attn, q_norm, k_norm) + 1 RoPE")
    print("  + 1 SiLU = 6, plus 1 final norm.")
    calls = 36 * 6 + 1
    print(f"    calls per forward         {calls}")
    print(f"    extra host time / forward {calls * delta:.0f} ms")
    print()
    print("  That is host-side serial overhead, not device time, and it does not shrink")
    print("  with batch or sequence length. Unless it overlaps with device execution it")
    print("  sets a floor on step time that no amount of kernel quality can beat.")
    print()
    print("  This is a finding about the EAGER PER-LAYER INTEGRATION MODEL, not about the")
    print("  kernels. It is also the strongest argument yet for the fused-kernel direction:")
    print("  one fused MLP call replaces several dispatches, so fusion cuts launch count as")
    print("  well as memory traffic. Which makes Findings #17 and #18 more important, not")
    print("  less.")

    print()
    print("  CAVEATS:")
    print("    - Microbenchmark. Per-call overhead may overlap with device work in a real")
    print("      model; the forward-pass estimate above is an upper bound on the serial cost.")
    print("    - Says nothing about graph mode (can_torch_compile = False on these kernels).")
    print("    - Week 4 full-model MFU remains the number that decides the question.")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
