"""Does the DEVICE-time gap between NKI and torch depend on NEURON_CC_FLAGS?

WHY THIS EXISTS — it closes the other half of an open item
scripts/probe_compiler_flags.py established that NKI's WALL time is invariant across
{unset, --target trn2, +--lnc 1, +--lnc 2, +-O2}: 13.82-14.15 ms, a 1.02x spread. But that probe
measures 28 chained calls at ~0.49 ms/call, which is the post-fix dispatch floor, so it is ~97%
dispatch by construction. It proves the DISPATCH cost is flag-invariant and says almost nothing
about device time.

The device-time claims are the load-bearing ones for the recommendation:
  Finding #25 — NKI is 2.5-2.7x slower than torch on device for SiLU and RMSNorm at N=28
  Finding #26 — the fused nkilib MLP is ~3x slower than torch on device

Both were measured on compiler defaults only. If a different target or LNC setting closed that gap,
the recommendation would change, so it has to be checked rather than assumed.

WHAT IT DOES
Profiles the same op at N=1 and N=28 under each flag setting, into a separate directory tree per
setting, then compares two things per setting:

  device time at N=28        does any setting make the NKI kernel faster on device?
  marginal traffic per call  does any setting let the compiler fuse across the custom call?

The marginal-traffic column is the more diagnostic of the two. NKI's marginal traffic on defaults is
exactly 6.29 MB = one tile in, one tile out = the unfused floor. If a flag setting let the custom
call participate in fusion, marginal traffic would drop BELOW the floor, the way torch's does
(~0 MB). If it stays pinned at the floor under every setting, the fusion barrier is structural and
no compiler flag reaches it.

Usage — run on trn2, expect ~4 minutes per setting (4 profile runs each):
    python scripts/probe_device_time_under_flags.py
    python scripts/probe_device_time_under_flags.py --op rmsnorm
    python scripts/probe_device_time_under_flags.py --outdir-base results/raw/flagcheck
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Same settings as probe_compiler_flags.py, so the two probes are directly comparable.
# None means "leave NEURON_CC_FLAGS unset", which is what every other measurement ran on.
SETTINGS = [
    ("default", None),
    ("target", "--target trn2"),
    ("lnc1", "--target trn2 --lnc 1"),
    ("lnc2", "--target trn2 --lnc 2"),
    ("O2", "--target trn2 -O2"),
]

TILE_MB = 512 * 3072 * 2 / 1e6          # [512, 3072] bf16
UNFUSED_FLOOR_MB = 2 * TILE_MB          # one read in, one write out


def read_metrics(path: Path):
    """profile_dir -> total_time ms and HBM MB, from summarise_device_profiles.py --json-out."""
    import json
    if not path.exists():
        return {}
    out = {}
    for e in json.loads(path.read_text()):
        m = e["metrics"]
        out[e["profile_dir"]] = {
            "device_ms": (m.get("total_time") or 0) * 1e3,
            "hbm_mb": ((m.get("hbm_read_bytes") or 0) + (m.get("hbm_write_bytes") or 0)) / 1e6,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="silu", choices=["silu", "rmsnorm"])
    ap.add_argument("--outdir-base", default="results/raw/flagcheck",
                    help="one subdirectory per flag setting is created under here")
    ap.add_argument("--only", nargs="+", help="run only these settings by name")
    args = ap.parse_args()

    settings = SETTINGS
    if args.only:
        settings = [s for s in SETTINGS if s[0] in args.only]
        if not settings:
            print(f"no setting matches {args.only}; known: {[s[0] for s in SETTINGS]}")
            return 1

    base = (ROOT / args.outdir_base) if not Path(args.outdir_base).is_absolute() \
        else Path(args.outdir_base)

    print("=" * 94)
    print("DEVICE TIME UNDER COMPILER FLAGS: can any setting close the NKI/torch device gap?")
    print("=" * 94)
    print(f"  op = {args.op}, calls = 1 and 28, tile [512, 3072] bf16")
    print(f"  unfused floor per call = 2 tiles = {UNFUSED_FLOOR_MB:.2f} MB")
    print(f"  {len(settings)} setting(s) x 4 profile runs each, sequential")
    print(f"  profiles under {base}")
    print()

    results = {}
    for name, flags in settings:
        outdir = base / name
        print(f"--- setting '{name}': NEURON_CC_FLAGS="
              f"{flags if flags else '(unset)'} ---", flush=True)

        env = dict(os.environ)
        if flags is None:
            env.pop("NEURON_CC_FLAGS", None)
        else:
            env["NEURON_CC_FLAGS"] = flags
        # A shared compile cache would let one setting serve another setting's NEFF, which is
        # exactly the confound this probe exists to avoid.
        env["NEURON_CC_CACHE_DIR"] = str(base / f"_cache_{name}")

        sweep = subprocess.run(
            [sys.executable, "scripts/run_device_profile_sweep.py",
             "--ops", args.op, "--calls", "1", "28", "--outdir-base", str(outdir)],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
        )
        if sweep.returncode != 0:
            print(f"  sweep FAILED exit {sweep.returncode}")
            print("  stderr tail:", sweep.stderr.strip().splitlines()[-4:])
            continue

        # Extract device metrics WITHOUT the flags set: neuron-explorer only reads the artifacts.
        dirs = [str(outdir / f"prof_{args.op}_{impl}_n{n}")
                for n in (1, 28) for impl in ("nki", "torch")]
        jout = outdir / "device_metrics.json"
        summ = subprocess.run(
            [sys.executable, "scripts/summarise_device_profiles.py", *dirs,
             "--json-out", str(jout)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        if summ.returncode != 0:
            print(f"  summarise FAILED exit {summ.returncode}")
            continue

        m = read_metrics(jout)
        key = lambda impl, n: f"prof_{args.op}_{impl}_n{n}"  # noqa: E731
        try:
            n1_nki, n28_nki = m[key("nki", 1)], m[key("nki", 28)]
            n1_tor, n28_tor = m[key("torch", 1)], m[key("torch", 28)]
        except KeyError as e:
            print(f"  missing profile {e}; skipping this setting")
            continue

        # traffic(N) = FIXED + N * MARGINAL, solved from the N=1 and N=28 points.
        marg_nki = (n28_nki["hbm_mb"] - n1_nki["hbm_mb"]) / 27
        marg_tor = (n28_tor["hbm_mb"] - n1_tor["hbm_mb"]) / 27
        results[name] = {
            "flags": flags or "(unset)",
            "nki_ms": n28_nki["device_ms"],
            "torch_ms": n28_tor["device_ms"],
            "ratio": n28_nki["device_ms"] / n28_tor["device_ms"] if n28_tor["device_ms"] else 0,
            "marg_nki": marg_nki,
            "marg_torch": marg_tor,
        }
        r = results[name]
        print(f"  device N=28: NKI {r['nki_ms']:7.3f} ms  torch {r['torch_ms']:7.3f} ms  "
              f"ratio {r['ratio']:6.2f}x")
        print(f"  marginal MB/call: NKI {marg_nki:6.2f}  torch {marg_tor:6.2f}  "
              f"(floor {UNFUSED_FLOOR_MB:.2f})")
        print()

    if not results:
        print("no setting produced usable profiles")
        return 1

    print("=" * 94)
    print("VERDICT")
    print("=" * 94)
    print(f"  {'setting':<10} {'flags':<26} {'NKI ms':>8} {'torch ms':>9} {'ratio':>8} "
          f"{'NKI MB/call':>12} {'vs floor':>9}")
    print("  " + "-" * 90)
    for name, r in results.items():
        print(f"  {name:<10} {r['flags']:<26} {r['nki_ms']:8.3f} {r['torch_ms']:9.3f} "
              f"{r['ratio']:7.2f}x {r['marg_nki']:12.2f} "
              f"{r['marg_nki'] / UNFUSED_FLOOR_MB:8.2f}x")
    print()

    nki_ms = [r["nki_ms"] for r in results.values()]
    ratios = [r["ratio"] for r in results.values()]
    margs = [r["marg_nki"] for r in results.values()]
    nki_spread = max(nki_ms) / min(nki_ms) if min(nki_ms) else 0
    marg_spread = max(margs) / min(margs) if min(margs) else 0

    print(f"  NKI device time spread across settings:  {nki_spread:.2f}x "
          f"({min(nki_ms):.3f}-{max(nki_ms):.3f} ms)")
    print(f"  NKI marginal traffic spread:             {marg_spread:.2f}x "
          f"({min(margs):.2f}-{max(margs):.2f} MB/call)")
    print(f"  best (lowest) NKI/torch ratio:           {min(ratios):.2f}x")
    print()

    # The interesting outcome is whether ANY setting gets NKI below torch, or lets the custom call
    # fuse. Both are single, checkable conditions rather than a threshold judgement.
    if min(ratios) < 1.0:
        print("  A compiler setting makes NKI FASTER than torch on device. Findings #25/#26 are")
        print("  configuration artifacts and every device-time claim needs re-running under it.")
        return 0

    fused = [n for n, r in results.items()
             if r["marg_nki"] < 0.9 * UNFUSED_FLOOR_MB]
    if fused:
        print(f"  Settings {fused} drop NKI marginal traffic below the unfused floor, so the")
        print("  custom call DID participate in fusion there. The fusion barrier is not structural.")
        return 0

    print("  No setting makes NKI faster than torch on device, and NKI marginal traffic stays")
    print(f"  pinned at {min(margs):.2f}-{max(margs):.2f} MB/call — the unfused floor — under every")
    print("  setting. So the device gap is STRUCTURAL: an opaque custom call cannot be fused into")
    print("  its neighbours, and no compiler flag reaches that. Findings #25 and #26 stand as")
    print("  measured, and the open item is closed for device time as well as wall time.")
    print()
    print("  This is the stronger version of the result: it is not that we failed to find a better")
    print("  flag, it is that the quantity a better flag would have to move (marginal traffic) is")
    print("  already at its theoretical minimum for an unfusable op.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
