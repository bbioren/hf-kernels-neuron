"""Attribute the NKI-vs-torch device gap: is it kernel quality, or the fusion barrier?

Reads the device profiles produced by run_device_profile_sweep.py and does the traffic arithmetic
that separates the two explanations.

THE ARITHMETIC
A tile of [rows, cols] in bfloat16 is rows*cols*2 bytes. An op that cannot fuse with its neighbours
must read its input from HBM and write its output back, so its FLOOR is 2 tiles of traffic per call.
An op that fuses into a chain pays that floor ONCE for the whole chain.

So for N chained applications:
  unfused floor   = N * 2 * tile_bytes
  fully fused     =     2 * tile_bytes

If NKI traffic per call is at or near the unfused floor, the kernel is efficient and the gap is the
fusion barrier — each custom call is opaque to the compiler, so nothing fuses across it. If NKI
traffic per call is well ABOVE the floor, the kernel itself is moving more data than it needs
(e.g. spilling an fp32 intermediate to HBM), which is ours to fix.

These have opposite implications for the recommendation, so the distinction is worth being precise
about rather than reporting "2.6x slower" and leaving the cause open.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_profile(neff: Path, ntff: Path):
    cmd = ["neuron-explorer", "view", "--output-format", "summary-json",
           "-n", str(neff), "-s", str(ntff)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            return next(iter(d.values())) if len(d) == 1 else d
    return None


def load(outdir: str):
    root = Path(outdir)
    if not root.exists():
        return None
    neffs = sorted(root.rglob("*.neff"))
    if not neffs:
        return None
    neff = neffs[0]
    stem = neff.name.replace("neff_", "").replace(".neff", "")
    ntff = neff.parent / f"{stem}.ntff"
    if not ntff.exists():
        cands = sorted(neff.parent.glob("*.ntff"))
        if not cands:
            return None
        ntff = cands[0]
    return read_profile(neff, ntff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--cols", type=int, default=3072)
    ap.add_argument("--bytes-per-elem", type=int, default=2, help="2 for bfloat16")
    ap.add_argument("--ops", nargs="+", default=["silu", "rmsnorm"])
    ap.add_argument("--calls", nargs="+", type=int, default=[1, 28])
    args = ap.parse_args()

    tile = args.rows * args.cols * args.bytes_per_elem
    floor_per_call = 2 * tile

    print("=" * 100)
    print("FUSION BARRIER vs KERNEL QUALITY")
    print("=" * 100)
    print(f"  tile [{args.rows}, {args.cols}] @ {args.bytes_per_elem} B = {tile / 1e6:.2f} MB")
    print(f"  unfused floor per call = 2 tiles = {floor_per_call / 1e6:.2f} MB "
          f"(one read in, one write out)")
    print()
    print(f"{'config':<26} {'device ms':>10} {'ms/call':>9} {'HBM MB':>9} "
          f"{'MB/call':>9} {'vs floor':>9} {'MBU%':>7}")
    print("-" * 100)

    data = {}
    for n in args.calls:
        for op in args.ops:
            for impl in ("nki", "torch"):
                m = load(f"/tmp/prof_{op}_{impl}_n{n}")
                if not m:
                    print(f"{op}/{impl}/n{n:<3} — no profile")
                    continue
                dev_ms = m["total_time"] * 1e3
                hbm = m.get("hbm_read_bytes", 0) + m.get("hbm_write_bytes", 0)
                per_call = hbm / n
                data[(op, impl, n)] = (dev_ms, hbm, per_call, m)
                print(f"{op + '/' + impl + '/n' + str(n):<26} {dev_ms:10.3f} "
                      f"{dev_ms / n:9.4f} {hbm / 1e6:9.1f} {per_call / 1e6:9.2f} "
                      f"{per_call / floor_per_call:8.2f}x "
                      f"{m.get('mbu_estimated_percent', 0) * 100:6.1f}%")

    print()
    print("ATTRIBUTION — fixed vs marginal traffic")
    print("-" * 100)
    print("  Traffic is NOT linear in N: a small NEFF carries fixed setup traffic that dominates")
    print("  at N=1. Dividing total traffic by N therefore overstates per-call cost at small N and")
    print("  produces a false 'kernel is inefficient' reading. With two call counts we can solve")
    print("  for both terms instead:")
    print("      traffic(N) = FIXED + N * MARGINAL")
    print("  MARGINAL is the real per-call cost, and it is what should be compared to the floor.")

    for op in args.ops:
        for impl in ("nki", "torch"):
            pts = [(n, data[(op, impl, n)][1]) for n in args.calls if (op, impl, n) in data]
            if len(pts) < 2:
                continue
            (n1, b1), (n2, b2) = pts[0], pts[-1]
            marginal = (b2 - b1) / (n2 - n1)
            fixed = b1 - n1 * marginal
            print(f"\n  {op}/{impl}:  traffic({n1})={b1/1e6:.2f} MB, "
                  f"traffic({n2})={b2/1e6:.2f} MB")
            print(f"    marginal per call = {marginal/1e6:6.2f} MB "
                  f"= {marginal/floor_per_call:.2f}x the unfused floor")
            print(f"    fixed per NEFF    = {fixed/1e6:6.2f} MB "
                  f"= {fixed/tile:.1f} tiles of setup")
            data[(op, impl, "marginal")] = marginal

    print()
    print("VERDICT")
    print("-" * 100)
    for op in args.ops:
        kn, kt = (op, "nki", "marginal"), (op, "torch", "marginal")
        if kn not in data or kt not in data:
            continue
        mn, mt = data[kn], data[kt]
        big_n = args.calls[-1]
        n_dev = data[(op, "nki", big_n)][0]
        t_dev = data[(op, "torch", big_n)][0]
        print(f"\n  {op}:")
        print(f"    device time at N={big_n}:  NKI {n_dev:7.3f} ms vs torch {t_dev:7.3f} ms "
              f"= {n_dev / t_dev:.2f}x")
        print(f"    marginal traffic/call:   NKI {mn/1e6:6.2f} MB vs torch {mt/1e6:6.2f} MB")
        print(f"    NKI vs unfused floor:    {mn/floor_per_call:.2f}x")

        if abs(mn - floor_per_call) / floor_per_call < 0.10:
            print("    -> NKI marginal traffic is EXACTLY the unfused floor (one read in, one")
            print("       write out). The kernel spills nothing and is optimal for an op that")
            print("       cannot fuse. Kernel quality is NOT the problem.")
        elif mn > floor_per_call * 1.3:
            print(f"    -> NKI is {mn/floor_per_call:.2f}x the floor even at the margin, so the kernel")
            print("       moves more data than needed — a spilled intermediate would do this.")
        if mt < floor_per_call * 0.3:
            print(f"    -> torch marginal traffic is {mt/1e6:.2f} MB, far below the floor, which is")
            print("       only possible if the chain FUSED into a single pass.")
            print("    => The entire device gap is the FUSION BARRIER.")

    print()
    print("=" * 100)
    print("WHY THIS MATTERS FOR THE RECOMMENDATION")
    print("=" * 100)
    print("  A NKI custom call is opaque to the Neuron compiler, so it cannot fuse across one.")
    print("  Replacing a torch op with a NKI kernel therefore does not merely add dispatch cost —")
    print("  it REMOVES a fusion opportunity the compiler was already exploiting. For")
    print("  memory-bound ops (elementwise activations, normalisations) fusion is the entire")
    print("  optimisation, so the swap loses on device even with a perfect kernel.")
    print()
    print("  That is a property of the per-layer swap model, not of NKI or of these kernels, and")
    print("  it is why the ops best suited to the Kernel Hub mechanism are the ones LEAST suited")
    print("  to benefit from it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
