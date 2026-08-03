"""Drive scripts/profile_nki_vs_torch_device.py across ops, impls and call counts.

WHY THE CALL-COUNT SWEEP MATTERS
At N=28 chained, NKI is ~2.6x slower on device than torch for both SiLU and RMSNorm, with ~30x the
HBM traffic. Two different causes could produce that, and they have opposite implications:

  (a) The kernels are simply worse than what the compiler emits.
  (b) The kernels are fine, but each NKI custom call is an optimisation barrier: the compiler fuses
      a chain of torch ops into one pass, and cannot fuse across an opaque custom call, so every
      NKI call materialises its tile to HBM.

N=1 separates them. With a single op there is no chain to fuse, so both implementations read and
write the same tile once and the comparison is kernel-vs-kernel on merit. If NKI ≈ torch at N=1 but
2.6x slower at N=28, the cause is (b) and the kernels are not the problem.

Each configuration needs its own process and its own NEFF directory, so this runs them as
subprocesses rather than in-process.

Usage:
    python scripts/run_device_profile_sweep.py
    python scripts/run_device_profile_sweep.py --calls 1 28 --ops silu rmsnorm
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROFILER = ROOT / "scripts" / "profile_nki_vs_torch_device.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="+", default=["silu", "rmsnorm"])
    ap.add_argument("--impls", nargs="+", default=["nki", "torch"])
    ap.add_argument("--calls", nargs="+", type=int, default=[1, 28])
    ap.add_argument("--outdir-base", default="/tmp",
                    help="where per-configuration profile dirs go. Default /tmp is fine for "
                         "throwaway runs, but pass results/raw so the profiles land in the "
                         "artifact tree and survive the host — /tmp on a rented instance is the "
                         "least durable location available, and this project has lost artifacts "
                         "to it once already.")
    args = ap.parse_args()

    jobs = [(op, impl, n) for n in args.calls for op in args.ops for impl in args.impls]
    print(f"{len(jobs)} configurations to profile")
    print("(each is a separate process; two Neuron processes must not run concurrently)")

    outdirs, failures = [], []
    for i, (op, impl, n) in enumerate(jobs, 1):
        outdir = f"{args.outdir_base.rstrip('/')}/prof_{op}_{impl}_n{n}"
        outdirs.append(outdir)
        print(f"\n[{i}/{len(jobs)}] {op} {impl} calls={n} -> {outdir}", flush=True)
        subprocess.run(["rm", "-rf", outdir], check=False)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(PROFILER), "--op", op, "--impl", impl,
             "--calls", str(n), "--outdir", outdir],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            failures.append((op, impl, n, proc.returncode))
            print(f"  FAILED exit {proc.returncode} in {dt:.0f}s")
            print("  stderr tail:", proc.stderr.strip().splitlines()[-5:])
        else:
            steady = [l for l in proc.stdout.splitlines() if "iter 3" in l]
            print(f"  ok in {dt:.0f}s  {steady[0].strip() if steady else ''}")

    print()
    print("=" * 80)
    if failures:
        print(f"{len(failures)} configuration(s) failed: {failures}")
    print("Now summarise with:")
    print("  python scripts/summarise_device_profiles.py \\")
    for d in outdirs:
        print(f"      {d} \\")
    print("      --calls <N>")
    print()
    print("N=1 and N=28 directories can be summarised in ONE invocation: the summariser reads the")
    print("call count from each directory's _n<N> suffix, so --calls is only a fallback for")
    print("directories without one. It used to divide every directory by a single --calls, which")
    print("silently divided an N=1 profile by 28.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
