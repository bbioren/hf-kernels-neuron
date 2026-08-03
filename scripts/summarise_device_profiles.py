"""Read device time out of every NEFF+NTFF pair under a directory and compare.

Companion to scripts/profile_nki_vs_torch_device.py. Runs `neuron-explorer view` on each
NEFF/NTFF pair found, extracts the device metrics, and prints a comparison table so the NKI-vs-torch
question can be answered from device time rather than wall clock.

Metrics pulled per profile:
  total_time                          device execution time for the whole NEFF
  total_active_time_percent           how much of it the engines were busy
  mbu_estimated_percent               memory-bandwidth utilisation
  hbm_read_bytes / hbm_write_bytes    traffic, which reveals whether ops fused
  *_engine_active_time_percent        which engine did the work

Usage:
    python scripts/summarise_device_profiles.py /tmp/prof_silu_nki_n28 /tmp/prof_silu_torch_n28
    python scripts/summarise_device_profiles.py --calls 28 /tmp/prof_*
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

FIELDS = [
    "total_time",
    "total_active_time_percent",
    "mbu_estimated_percent",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "tensor_engine_active_time_percent",
    "vector_engine_active_time_percent",
    "scalar_engine_active_time_percent",
    "gpsimd_engine_active_time_percent",
    "activate_instruction_count",
]


def read_profile(neff: Path, ntff: Path):
    """Run neuron-explorer and return the metrics dict for this NEFF, or None."""
    cmd = ["neuron-explorer", "view", "--output-format", "summary-json",
           "-n", str(neff), "-s", str(ntff)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    neuron-explorer failed: {e}")
        return None
    if out.returncode != 0:
        print(f"    neuron-explorer exit {out.returncode}: {out.stderr.strip()[:200]}")
        return None
    # Output is {"<hash>": {...metrics...}}; take the single inner dict.
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if len(d) == 1:
            return next(iter(d.values()))
        return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--calls", type=int, default=28,
                    help="calls per graph, for the per-call column")
    ap.add_argument("--json-out", default=None,
                    help="write the extracted per-profile metrics here. This is the AUDITABLE "
                         "artifact: every device-time claim in this project comes from these "
                         "numbers, and they are a few KB, whereas the NTFF traces they are "
                         "extracted from are hundreds of MB and are gitignored.")
    args = ap.parse_args()

    rows = []
    extracted = []
    for d in args.dirs:
        root = Path(d)
        if not root.exists():
            print(f"skip {d} (missing)")
            continue
        neffs = sorted(root.rglob("*.neff"))
        if not neffs:
            print(f"skip {d} (no NEFF)")
            continue
        for neff in neffs:
            # Runtime writes <id>_vnc_0.ntff next to neff_<id>_vnc_0.neff
            stem = neff.name.replace("neff_", "").replace(".neff", "")
            ntff = neff.parent / f"{stem}.ntff"
            if not ntff.exists():
                cands = sorted(neff.parent.glob("*.ntff"))
                if not cands:
                    print(f"skip {neff.name} (no NTFF)")
                    continue
                ntff = cands[0]
            print(f"reading {root.name} / {neff.name}")
            m = read_profile(neff, ntff)
            if m:
                rows.append((root.name, m))
                extracted.append({
                    "profile_dir": root.name,
                    "neff": neff.name,
                    "ntff": ntff.name,
                    "metrics": {k: m.get(k) for k in FIELDS},
                })

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(extracted, indent=2) + "\n")
        print(f"\nwrote {args.json_out} ({len(extracted)} profile(s)) — "
              f"this is the auditable artifact; the NTFF traces are gitignored")

    if not rows:
        print("no profiles read")
        return 1

    print()
    print("=" * 110)
    print("DEVICE TIME COMPARISON  (wall-clock dispatch cost excluded by construction)")
    print("=" * 110)
    hdr = (f"{'profile':<30} {'device ms':>10} {'per call':>10} {'active%':>8} "
           f"{'MBU%':>7} {'HBM r+w MB':>11} {'acts':>6}")
    print(hdr)
    print("-" * 110)
    for name, m in rows:
        tot = m.get("total_time", 0) * 1e3
        hbm = (m.get("hbm_read_bytes", 0) + m.get("hbm_write_bytes", 0)) / 1e6
        print(f"{name:<30} {tot:10.3f} {tot / args.calls:10.4f} "
              f"{m.get('total_active_time_percent', 0) * 100:7.1f}% "
              f"{m.get('mbu_estimated_percent', 0) * 100:6.1f}% "
              f"{hbm:11.1f} {m.get('activate_instruction_count', 0):6d}")

    # Pair up nki vs torch for the same op, if both are present.
    print()
    print("VERDICT")
    by_op = {}
    for name, m in rows:
        for op in ("silu", "rmsnorm"):
            if op in name:
                impl = "nki" if "nki" in name else "torch" if "torch" in name else None
                if impl:
                    by_op.setdefault(op, {})[impl] = m

    if not by_op:
        print("  could not pair nki/torch profiles by name; read the table above directly")
        return 0

    for op, impls in sorted(by_op.items()):
        if "nki" not in impls or "torch" not in impls:
            print(f"  {op}: only {list(impls)} measured, need both to compare")
            continue
        n = impls["nki"]["total_time"] * 1e3
        t = impls["torch"]["total_time"] * 1e3
        ratio = n / t if t else float("inf")
        verdict = "NKI FASTER" if ratio < 1 else "NKI SLOWER"
        print(f"  {op:8s} NKI {n:8.3f} ms   torch {t:8.3f} ms   "
              f"NKI/torch = {ratio:6.2f}x   -> {verdict} on device")
        nh = (impls["nki"].get("hbm_read_bytes", 0)
              + impls["nki"].get("hbm_write_bytes", 0)) / 1e6
        th = (impls["torch"].get("hbm_read_bytes", 0)
              + impls["torch"].get("hbm_write_bytes", 0)) / 1e6
        print(f"           HBM traffic: NKI {nh:.1f} MB vs torch {th:.1f} MB "
              f"({nh / th:.2f}x)" if th else "")

    print()
    print("  How to read this: if NKI is SLOWER on device, fixing the remaining dispatch")
    print("  overhead never yields a speedup and per-layer swapping of these ops is a dead end")
    print("  on merit, not just on plumbing. If NKI is FASTER on device, then dispatch cost is")
    print("  the only thing between here and a win, and Fix 7 is the whole ballgame.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
