"""Print the per-call added cost at each stage of the dispatch fixes, from the MFU artifacts.

WHY
The headline of this project moved three times as fixes landed, and each time the comparison was
assembled by hand from separate runs. This reads the committed MFU JSONs and prints the progression
in one place, so "how much of the regression is left" is answered from artifacts rather than from
whichever number was most recently in a commit message.

The stages:
  no fixes         Finding #24's uncached `neuron-ls` subprocess still in the path
  #24 only         target detection lru_cached
  #24 + B12        the XLA computation also registered once per compile-cache key
  device floor     the in-situ device gap, which no dispatch fix can remove

Usage (runs anywhere, reads only JSON):
    python scripts/compare_fix_stages.py
    python scripts/compare_fix_stages.py --raw results/raw --json-out results/fix_progression.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# (label, path relative to the raw root, seq)
SOURCES = [
    ("no fixes", "mfu-kernelized-512-nofix/mfu_512_nofix.json", 512),
    ("#24 only", "mfu-baseline-and-kernelized-512-fixed/mfu_512_fixed.json", 512),
    ("#24 + B12", "mfu-512-both-fixes/mfu_512_both.json", 512),
    ("#24 only", "mfu-2048-fixed/mfu_2048_fixed.json", 2048),
    ("#24 + B12", "mfu-2048-both-fixes/mfu_2048_both.json", 2048),
]

# From the in-situ decomposition. Device time is the one term a dispatch fix cannot touch, so it is
# the floor any of these numbers is converging towards.
INSITU = "insitu-summary/insitu_decomposition.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    raw = ROOT / args.raw

    floor = None
    p = raw / INSITU
    if p.exists():
        floor = json.loads(p.read_text())["per_call_device_ms"]

    rows = []
    for label, rel, seq in SOURCES:
        f = raw / rel
        if not f.exists():
            print(f"  missing {rel} — run `make results`")
            continue
        d = json.loads(f.read_text())
        b, k = d["baseline"]["step_s"] * 1e3, d["kernelized"]["step_s"] * 1e3
        calls = d["kernelized"].get("nki_launches_per_step") or 169
        rows.append({
            "stage": label,
            "seq": seq,
            "baseline_ms": round(b, 2),
            "kernelized_ms": round(k, 2),
            "slowdown": round(k / b, 3),
            "mfu_baseline_pct": round(d["baseline"]["mfu_per_core_te"], 3),
            "mfu_kernelized_pct": round(d["kernelized"]["mfu_per_core_te"], 3),
            "nki_calls": calls,
            "added_ms_per_call": round((k - b) / calls, 4),
            "fixes": d.get("fixes"),
        })

    if not rows:
        return 1

    print("=" * 96)
    print("DISPATCH FIX PROGRESSION — Qwen3-0.6B, 28 layers, bf16, forward, single logical core")
    print("=" * 96)
    print(f"  {'stage':<12} {'seq':>5} {'baseline':>9} {'kernelized':>11} {'slowdown':>9} "
          f"{'MFU':>7} {'added ms/call':>14}")
    print("  " + "-" * 92)
    for r in rows:
        print(f"  {r['stage']:<12} {r['seq']:5d} {r['baseline_ms']:9.2f} {r['kernelized_ms']:11.2f} "
              f"{r['slowdown']:8.2f}x {r['mfu_kernelized_pct']:6.2f}% "
              f"{r['added_ms_per_call']:14.4f}")
    if floor is not None:
        print(f"  {'device floor':<12} {'':>5} {'':>9} {'':>11} {'':>9} {'':>7} {floor:14.4f}")
    print()

    per512 = {r["stage"]: r["added_ms_per_call"] for r in rows if r["seq"] == 512}
    if {"no fixes", "#24 only", "#24 + B12"} <= per512.keys():
        a, b_, c = per512["no fixes"], per512["#24 only"], per512["#24 + B12"]
        print("INTERPRETATION")
        print(f"  Added cost per NKI call: {a:.3f} -> {b_:.3f} -> {c:.3f} ms")
        print(f"    Finding #24 (lru_cache on _detect_target):      {a / b_:7.1f}x")
        print(f"    B12 (register the XLA computation once):        {b_ / c:7.1f}x")
        print(f"    both together:                                 {a / c:7.1f}x")
        if floor:
            print(f"  Remaining dispatch is {c - floor:.3f} ms/call against a device floor of "
                  f"{floor:.3f}, so")
            print(f"  the cost is now within {c / floor:.1f}x of the floor, and "
                  f"{100 * (c - floor) / c:.0f}% of what")
            print("  remains is still dispatch rather than device time.")
        print()
        print("  Both fixes are the SAME bug: a cache exists and the code path defeats it. #24")
        print("  resolved the compile target while building the cache key, so a hit still paid the")
        print("  subprocess. B12 built the computation cache on an object recreated per call.")
        print("  Neither is a property of per-layer kernel dispatch on Neuron.")

    amort = [(r["seq"], r["slowdown"]) for r in rows if r["stage"] == "#24 + B12"]
    if len(amort) > 1:
        amort.sort()
        print()
        print("  Amortisation, both fixes applied:")
        for s, sd in amort:
            print(f"    seq {s:5d}  {sd:.2f}x slower")
        print("  Call count is fixed by model depth, so a longer sequence means more work per call")
        print("  rather than more calls. The residual is near-fixed, so the penalty shrinks.")

    if args.json_out:
        out = {"rows": rows, "device_floor_ms_per_call": floor}
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
