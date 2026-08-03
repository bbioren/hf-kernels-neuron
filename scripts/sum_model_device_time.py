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


def sum_dir(d: str, expect_neffs=None):
    """Sum device time and traffic across every NEFF/NTFF pair under d.

    Summing makes this vulnerable to leftovers: a NEFF from a previous run in the same directory is
    indistinguishable from a fresh one and gets added in, doubling the total with no error. That has
    happened — a re-run reported 2 NEFFs, exactly 2x the device time, and a device share of 16.9%
    instead of 8.4%. `expect_neffs` turns that from a silent wrong answer into a loud one.
    """
    root = Path(d)
    if not root.exists():
        return None

    found = sorted(root.rglob("*.neff"))
    if expect_neffs is not None and len(found) != expect_neffs:
        print(f"  WARNING: {root.name} has {len(found)} NEFF(s), expected {expect_neffs}.")
        print("    Device time is SUMMED across all of them, so extras from a previous run inflate")
        print("    the total. Delete the directory and re-run the producing stage.")
        for p in found:
            print(f"      {p.relative_to(root)}")

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
    # Not required: prefer each profile dir's own wall_times.json, written by
    # profile_model_device_time.py during the same run. A hand-passed constant can outlive the host
    # it was measured on and then sit next to fresh device numbers, which is how an earlier run of
    # this script ended up mixing stale walls with fresh device times.
    ap.add_argument("--wall-baseline", type=float, default=None,
                    help="ms/step. Omit to read wall_times.json from the baseline profile dir.")
    ap.add_argument("--wall-kernelized", type=float, default=None,
                    help="ms/step. Omit to read wall_times.json from the kernelized profile dir.")
    ap.add_argument("--nki-calls", type=int, default=169)
    ap.add_argument("--expect-neffs", type=int, default=1,
                    help="NEFFs each profile dir should contain. Qwen3-0.6B forward emits 1; more "
                         "than that usually means leftovers from a previous run, which would be "
                         "silently summed in. Set to 0 to disable the check.")
    ap.add_argument("--json-out", default=None,
                    help="write the decomposition here. This is the single most quoted result in "
                         "the project, so it should be a file and not only a log line.")
    args = ap.parse_args()
    if args.expect_neffs == 0:
        args.expect_neffs = None

    def wall_from(d, override, label):
        if override is not None:
            print(f"  {label} wall: {override} ms (passed on the command line)")
            return override
        p = Path(d) / "wall_times.json"
        if not p.exists():
            print(f"  ERROR: no --wall-{label} given and {p} does not exist.\n"
                  f"  Re-run profile_model_device_time.py (it now emits wall_times.json), or pass\n"
                  f"  --wall-{label} explicitly.")
            sys.exit(2)
        med = json.loads(p.read_text())["wall_ms_median"]
        print(f"  {label} wall: {med} ms (from {p.name}, this run)")
        return med

    args.wall_baseline = wall_from(args.baseline_dir, args.wall_baseline, "baseline")
    args.wall_kernelized = wall_from(args.kernelized_dir, args.wall_kernelized, "kernelized")

    b = sum_dir(args.baseline_dir, expect_neffs=args.expect_neffs)
    k = sum_dir(args.kernelized_dir, expect_neffs=args.expect_neffs)
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

    # PROJECTION, computed here rather than by hand. This is the number the design doc leads with,
    # and it was previously arithmetic done in a commit message: baseline wall + device gap. Doing
    # it by hand means it silently keeps the old walls when the walls change, which is exactly the
    # drift measurements.json exists to prevent. It is a projection and is labelled as one on every
    # line that prints it: it assumes the dispatch gap goes to zero, which no fix has achieved.
    proj_ms = args.wall_baseline + d_dev
    proj_ratio = proj_ms / args.wall_baseline if args.wall_baseline else float("nan")
    print()
    print("  PROJECTION — if dispatch overhead were eliminated entirely:")
    print(f"    step        {proj_ms:9.2f} ms   (baseline {args.wall_baseline} + device gap "
          f"{d_dev:.3f})")
    print(f"    vs baseline {proj_ratio:9.2f}x   <- NOT MEASURED. Assumes dispatch -> 0, which is")
    print("                            an upper bound on what fixing dispatch can buy.")

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

    if args.json_out:
        out = {
            "baseline": {"dir": args.baseline_dir, "neffs": b["n"],
                         "device_ms": round(b["device_ms"], 3), "hbm_mb": round(b["hbm"] / 1e6, 1),
                         "activates": b["acts"], "wall_ms": args.wall_baseline},
            "kernelized": {"dir": args.kernelized_dir, "neffs": k["n"],
                           "device_ms": round(k["device_ms"], 3), "hbm_mb": round(k["hbm"] / 1e6, 1),
                           "activates": k["acts"], "wall_ms": args.wall_kernelized},
            "nki_calls_per_step": args.nki_calls,
            "wall_gap_ms": round(d_wall, 3),
            "device_gap_ms": round(d_dev, 3),
            "device_gap_pct": round(100 * frac, 1),
            "dispatch_gap_ms": round(d_disp, 3),
            "dispatch_gap_pct": round(100 * (1 - frac), 1),
            "per_call_device_ms": round(d_dev / args.nki_calls, 4),
            "per_call_dispatch_ms": round(d_disp / args.nki_calls, 4),
            "projected_with_dispatch_fixed": {
                "step_ms": round(proj_ms, 2),
                "slowdown_vs_baseline": round(proj_ratio, 2),
                "status": "PROJECTION, not measured — assumes dispatch gap goes to zero",
            },
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
