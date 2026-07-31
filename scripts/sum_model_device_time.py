"""Sum device time across every NEFF a full model emits, and decompose the wall-clock gap.

A full Qwen3 forward compiles to several NEFFs, so a single `neuron-explorer view` reads only part of
the picture. This sums `total_time` and HBM traffic across all NEFF/NTFF pairs in a directory, then
compares baseline against kernelized to split the model-level regression into its two parts:

    wall_k - wall_b  =  (device_k - device_b)  +  (dispatch_k - dispatch_b)

The first term is the in-situ cost of losing compiler fusion (Finding #25 measured an upper bound of
2.5-2.7x from a chained microbenchmark). The second is per-call dispatch overhead (Finding #24, the
~0.59 ms/call residual). Knowing which dominates decides whether Fix 7 or the fusion question is the
more important ask.

Usage:
    python scripts/sum_model_device_time.py /tmp/prof_model_baseline /tmp/prof_model_kernelized \\
        --wall-baseline 42.04 --wall-kernelized 141.43 --nki-calls 169
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_profile(neff: Path, ntff: Path):
    cmd = ["neuron-explorer", "view", "--output-format", "summary-json",
           "-n", str(neff), "-s", str(ntff)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if out.returncode != 0:
        return None, out.stderr.strip()[:160]
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            return (next(iter(d.values())) if len(d) == 1 else d), None
    return None, "no JSON in output"


def sum_dir(d: str):
    """Sum device time and traffic across every NEFF/NTFF pair under d."""
    root = Path(d)
    if not root.exists():
        return None
    total_ms, hbm, acts, n_read, skipped = 0.0, 0, 0, 0, []
    for neff in sorted(root.rglob("*.neff")):
        stem = neff.name.replace("neff_", "").replace(".neff", "")
        ntff = neff.parent / f"{stem}.ntff"
        if not ntff.exists():
            skipped.append((neff.name, "no matching NTFF"))
            continue
        m, err = read_profile(neff, ntff)
        if m is None:
            skipped.append((neff.name, err))
            continue
        total_ms += m.get("total_time", 0) * 1e3
        hbm += m.get("hbm_read_bytes", 0) + m.get("hbm_write_bytes", 0)
        acts += m.get("activate_instruction_count", 0)
        n_read += 1
    return dict(device_ms=total_ms, hbm=hbm, acts=acts, n=n_read, skipped=skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir")
    ap.add_argument("kernelized_dir")
    ap.add_argument("--wall-baseline", type=float, required=True, help="ms/step, from measure_mfu")
    ap.add_argument("--wall-kernelized", type=float, required=True, help="ms/step")
    ap.add_argument("--nki-calls", type=int, default=169)
    args = ap.parse_args()

    b = sum_dir(args.baseline_dir)
    k = sum_dir(args.kernelized_dir)
    if not b or not k:
        print("missing profile directory")
        return 1

    print("=" * 96)
    print("IN-SITU DECOMPOSITION: how much of the model regression is device, how much dispatch?")
    print("=" * 96)
    for name, r in (("baseline", b), ("kernelized", k)):
        print(f"  {name:<12} {r['n']:2d} NEFF(s)  device {r['device_ms']:8.3f} ms  "
              f"HBM {r['hbm']/1e6:9.1f} MB  activates {r['acts']}")
        for nm, why in r["skipped"]:
            print(f"      skipped {nm}: {why}")

    if not b["n"] or not k["n"]:
        print("\n  could not read profiles from one side; cannot decompose")
        return 1

    d_dev = k["device_ms"] - b["device_ms"]
    d_wall = args.wall_kernelized - args.wall_baseline
    d_disp = d_wall - d_dev

    print()
    print(f"  wall gap        {d_wall:9.2f} ms   ({args.wall_baseline} -> {args.wall_kernelized})")
    print(f"  device gap      {d_dev:9.3f} ms   "
          f"({100 * d_dev / d_wall:5.1f}% of the wall gap)")
    print(f"  dispatch gap    {d_disp:9.3f} ms   "
          f"({100 * d_disp / d_wall:5.1f}% of the wall gap)")
    print()
    print(f"  per NKI call ({args.nki_calls} calls/step):")
    print(f"    device   {d_dev / args.nki_calls:8.4f} ms")
    print(f"    dispatch {d_disp / args.nki_calls:8.4f} ms")
    print(f"  HBM traffic  baseline {b['hbm']/1e6:.1f} MB -> kernelized {k['hbm']/1e6:.1f} MB "
          f"({k['hbm'] / max(b['hbm'], 1):.2f}x)")

    print()
    print("INTERPRETATION")
    frac = d_dev / d_wall if d_wall else 0
    if frac < 0.15:
        print(f"  Device accounts for only {100*frac:.1f}% of the regression, so IN A REAL MODEL the")
        print("  fusion penalty is minor and the slowdown is overwhelmingly dispatch. Finding #25's")
        print("  2.5-2.7x is then a genuine upper bound from the chained microbenchmark rather than")
        print("  a representative figure — and Fix 7 (caching create_computation) matters MORE than")
        print("  #25's framing implies. State both, and do not let the microbenchmark carry the")
        print("  headline.")
    elif frac > 0.5:
        print(f"  Device accounts for {100*frac:.1f}% of the regression, so the fusion barrier is real")
        print("  at model scale, not just in the microbenchmark. #25 is the binding constraint and")
        print("  dispatch work alone cannot fix the model-level regression.")
    else:
        print(f"  Device is {100*frac:.1f}% of the regression — both terms matter and neither fix")
        print("  alone is sufficient. Report the split rather than picking a headline cause.")
    print()
    print("  Caveat: HBM traffic here includes weights, so the kernelized/baseline traffic ratio is")
    print("  diluted compared to the activation-only microbenchmark and should not be read as the")
    print("  fusion penalty directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
