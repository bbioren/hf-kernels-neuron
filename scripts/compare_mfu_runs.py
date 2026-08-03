"""Compare MFU JSON outputs across runs, and report per-call added cost.

Purpose: test whether the residual ~0.59 ms/call of NKI dispatch overhead AMORTISES with more work
per call. Call count is fixed by model depth (169 for Qwen3-0.6B), so increasing sequence length
increases work per call without changing the number of calls. If the added cost per call stays flat
while the baseline step time grows, the overhead is fixed per call and the relative penalty shrinks
as models get bigger.

Usage:
    python scripts/compare_mfu_runs.py run_a.json run_b.json ...
"""

import json
import sys


def main(paths):
    if not paths:
        print(__doc__)
        return 1

    print(f"{'run':<26} {'baseline':>10} {'kernelized':>11} {'MFU base':>9} "
          f"{'MFU kern':>9} {'slower':>8} {'calls':>6} {'added':>9} {'per call':>9}")
    print("-" * 108)

    rows = []
    for p in paths:
        d = json.load(open(p))
        b, k = d["baseline"], d["kernelized"]
        n = k.get("nki_launches_per_step", 0)
        b_ms, k_ms = b["step_s"] * 1e3, k["step_s"] * 1e3
        added = k_ms - b_ms
        per_call = added / n if n else float("nan")
        label = f"{d.get('preset', '?')} seq{d.get('seq', '?')} b{d.get('batch', '?')}"
        rows.append((label, b_ms, k_ms, per_call, added, n,
                     b["mfu_per_core_te"], k["mfu_per_core_te"], 1 / d["speedup"]))
        print(f"{label:<26} {b_ms:9.2f}m {k_ms:10.2f}m {b['mfu_per_core_te']:8.2f}% "
              f"{k['mfu_per_core_te']:8.2f}% {1/d['speedup']:7.2f}x {n:6d} "
              f"{added:8.2f}m {per_call:8.3f}m")

    if len(rows) >= 2:
        print()
        print("AMORTISATION")
        first, last = rows[0], rows[-1]
        work_growth = last[1] / first[1]
        pc_change = last[3] / first[3] if first[3] else float("nan")
        print(f"  baseline work grew {work_growth:.2f}x "
              f"({first[1]:.1f} -> {last[1]:.1f} ms/step)")
        print(f"  added cost per call {first[3]:.3f} -> {last[3]:.3f} ms "
              f"({pc_change:.2f}x)")
        print(f"  penalty {first[8]:.2f}x -> {last[8]:.2f}x slower")
        if pc_change < 1.35:
            print("  -> per-call cost is roughly FIXED, so it amortises: the penalty shrinks as")
            print("     work per call grows. Extrapolating, a large enough op reaches parity.")
            # Work per call needed for the overhead to be, say, 10% of step time.
            pc = last[3]
            print(f"     at {pc:.3f} ms/call fixed overhead and {last[5]} calls, a step would need")
            print(f"     ~{pc * last[5] * 10:.0f} ms of real work for the overhead to be <10%.")
        else:
            print("  -> per-call cost GREW with work per call, so it is not purely fixed overhead.")
            print("     Part of it scales with problem size; do not describe it as fixed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
