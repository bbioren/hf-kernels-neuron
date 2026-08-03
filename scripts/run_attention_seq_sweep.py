"""Sweep sequence length to find where flash attention starts BEATING the compiler.

WHY A SWEEP RATHER THAN A SINGLE POINT
The first attention measurement, at seq 512, said NKI flash attention was 2.09x slower on device
than torch eager attention — and the traffic explained why in a way that was not anticipated:

    NKI    234.9 MB over 28 layers  =  8.39 MB/layer
    torch   63.0 MB over 28 layers  =  2.25 MB/layer

8.39 MB/layer is almost exactly one [16, 512, 512] bf16 score matrix (8.4 MB). And 2.25 MB/layer is
BELOW the 6.29 MB it costs just to read q, k, v and write the output once. Both facts point the same
way: **the compiler fused the entire chain and never materialised the score matrix to HBM either.**
Flash attention's central algorithmic advantage — never writing the S x S scores — is something XLA
on Neuron was already achieving at this size, so the kernel spends its boundary HBM round-trip to
buy something it does not get.

That is a statement about seq 512, not about attention. The score matrix grows as S^2 while flash's
working set grows as S:

    seq   score matrix (16 heads, bf16)
     512      8.4 MB      fits in the compiler's fused pipeline
    1024     33.6 MB
    2048    134.2 MB
    4096    536.9 MB      cannot possibly stay resident

Somewhere in that range the compiler must start spilling and flash must start winning. Finding that
crossover is the difference between "no speedup is available" and "a speedup is available above seq
N" — and the second is a materially different recommendation, because production sequence lengths are
not 512.

This is also the honest form of the search. Every prior candidate in this project was tested at one
configuration and written off. Attention is the one where the physics says the answer depends on
scale, so testing one point would be choosing the answer.

Each configuration needs its own process and NEFF directory, so they run as subprocesses.

Usage:
    python scripts/run_attention_seq_sweep.py
    python scripts/run_attention_seq_sweep.py --seqs 512 1024 2048 4096 --calls 4
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROFILER = ROOT / "scripts" / "profile_attention_nki_vs_torch.py"
SUMMARISER = ROOT / "scripts" / "summarise_device_profiles.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    ap.add_argument("--calls", type=int, default=4,
                    help="layers per graph. Kept small and CONSTANT across seq lengths so the "
                         "per-layer numbers are comparable and long sequences do not blow up "
                         "compile time. 4 is enough to separate fixed NEFF traffic from marginal.")
    ap.add_argument("--outdir-base", default="results/raw/attn-sweep")
    ap.add_argument("--json-out", default="results/raw/attn-sweep/sweep.json")
    args = ap.parse_args()

    base = ROOT / args.outdir_base
    jobs = [(s, impl) for s in args.seqs for impl in ("nki", "torch")]

    print(f"{len(jobs)} configurations ({len(args.seqs)} seq lengths x 2 impls), "
          f"{args.calls} layers each")
    print("sequential — two Neuron processes contend for cores\n")

    dirs, failures = {}, []
    for i, (seq, impl) in enumerate(jobs, 1):
        outdir = base / f"prof_attn_{impl}_s{seq}_n{args.calls}"
        print(f"[{i}/{len(jobs)}] seq={seq} impl={impl}", flush=True)
        t0 = time.perf_counter()
        p = subprocess.run(
            [sys.executable, str(PROFILER), "--impl", impl, "--seq", str(seq),
             "--calls", str(args.calls), "--outdir", str(outdir)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        dt = time.perf_counter() - t0
        if p.returncode != 0:
            failures.append((seq, impl, p.returncode))
            print(f"    FAILED exit {p.returncode} in {dt:.0f}s")
            for line in p.stdout.strip().splitlines()[-6:]:
                print(f"      {line}")
            for line in p.stderr.strip().splitlines()[-4:]:
                print(f"      ! {line}")
            continue
        wall = [l for l in p.stdout.splitlines() if f"iter {3}" in l]
        acc = [l for l in p.stdout.splitlines() if "correctness" in l]
        print(f"    ok in {dt:.0f}s")
        if acc:
            print(f"    {acc[0].strip()}")
        if wall:
            print(f"    {wall[0].strip()}")
        dirs[(seq, impl)] = outdir

    if not dirs:
        print("no configuration produced a profile")
        return 1

    # ---- extract device metrics for everything in one pass -------------------------------
    print("\nextracting device metrics...", flush=True)
    metrics_json = base / "device_metrics.json"
    p = subprocess.run(
        [sys.executable, str(SUMMARISER), *[str(d) for d in dirs.values()],
         "--calls", str(args.calls), "--json-out", str(metrics_json)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if p.returncode != 0:
        print(f"summariser failed exit {p.returncode}")
        print(p.stdout[-2000:])
        return 1

    entries = json.loads(metrics_json.read_text())

    # Each directory holds TWO NEFFs by design: a 1-layer correctness graph and the args.calls-layer
    # timed graph. Pick the timed one by instruction count — it is the larger.
    timed = {}
    for e in entries:
        d = e["profile_dir"]
        acts = e["metrics"].get("activate_instruction_count") or 0
        if d not in timed or acts > timed[d]["metrics"].get("activate_instruction_count", 0):
            timed[d] = e

    rows = []
    for (seq, impl), outdir in dirs.items():
        e = timed.get(outdir.name)
        if not e:
            continue
        m = e["metrics"]
        hbm = ((m.get("hbm_read_bytes") or 0) + (m.get("hbm_write_bytes") or 0)) / 1e6
        rows.append({
            "seq": seq, "impl": impl,
            "device_ms": round((m.get("total_time") or 0) * 1e3, 4),
            "per_layer_ms": round((m.get("total_time") or 0) * 1e3 / args.calls, 5),
            "hbm_mb": round(hbm, 1),
            "hbm_mb_per_layer": round(hbm / args.calls, 2),
            "mbu_pct": round((m.get("mbu_estimated_percent") or 0) * 100, 1),
            "active_pct": round((m.get("total_active_time_percent") or 0) * 100, 1),
            "acts": m.get("activate_instruction_count"),
        })

    print()
    print("=" * 104)
    print("FLASH ATTENTION vs COMPILER, BY SEQUENCE LENGTH  (device time, single logical core)")
    print("=" * 104)
    print(f"  {'seq':>6} {'impl':>6} {'device ms':>10} {'ms/layer':>10} {'HBM MB/layer':>13} "
          f"{'MBU':>6} {'active':>7} {'score mtx MB':>13}")
    print("  " + "-" * 100)
    for r in sorted(rows, key=lambda r: (r["seq"], r["impl"])):
        score_mb = 16 * r["seq"] * r["seq"] * 2 / 1e6
        print(f"  {r['seq']:6d} {r['impl']:>6} {r['device_ms']:10.3f} {r['per_layer_ms']:10.4f} "
              f"{r['hbm_mb_per_layer']:13.2f} {r['mbu_pct']:5.1f}% {r['active_pct']:6.1f}% "
              f"{score_mb:13.1f}")

    print()
    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    by_seq = {}
    for r in rows:
        by_seq.setdefault(r["seq"], {})[r["impl"]] = r

    crossover = None
    print(f"  {'seq':>6} {'NKI ms/layer':>13} {'torch ms/layer':>15} {'NKI/torch':>10} "
          f"{'NKI MB':>8} {'torch MB':>9} {'traffic ratio':>14}")
    print("  " + "-" * 100)
    for seq in sorted(by_seq):
        pair = by_seq[seq]
        if "nki" not in pair or "torch" not in pair:
            print(f"  {seq:6d}  incomplete pair ({list(pair)})")
            continue
        n, t = pair["nki"], pair["torch"]
        ratio = n["per_layer_ms"] / t["per_layer_ms"] if t["per_layer_ms"] else float("inf")
        tr = n["hbm_mb_per_layer"] / t["hbm_mb_per_layer"] if t["hbm_mb_per_layer"] else float("inf")
        mark = "  <- NKI FASTER" if ratio < 1 else ""
        if ratio < 1 and crossover is None:
            crossover = seq
        print(f"  {seq:6d} {n['per_layer_ms']:13.4f} {t['per_layer_ms']:15.4f} {ratio:9.2f}x "
              f"{n['hbm_mb_per_layer']:8.2f} {t['hbm_mb_per_layer']:9.2f} {tr:13.2f}x{mark}")

    print()
    if crossover is not None:
        print(f"  A SPEEDUP EXISTS. NKI flash attention becomes faster than the compiler's eager")
        print(f"  attention at seq >= {crossover} on device, and the trend is monotone in the")
        print("  traffic column: the compiler's advantage at short sequences comes from keeping the")
        print("  S x S score matrix resident, and that stops being possible as S^2 grows.")
        print()
        print("  This is the first candidate in the project that wins, and it wins for the reason")
        print("  the criterion predicted: flash attention is an ALGORITHMIC restructuring the")
        print("  compiler does not derive, not a fusion the compiler already performs.")
        print()
        print("  Recommendation impact: point Kernel Hub interception at attention, not at")
        print("  RMSNorm/RoPE/activations, and state the sequence length the win requires.")
    else:
        ratios = [(s, by_seq[s]["nki"]["per_layer_ms"] / by_seq[s]["torch"]["per_layer_ms"])
                  for s in sorted(by_seq)
                  if "nki" in by_seq[s] and "torch" in by_seq[s]]
        print("  No crossover within the sequence lengths tested.")
        if len(ratios) > 1:
            first, last = ratios[0], ratios[-1]
            direction = "NARROWING" if last[1] < first[1] else "WIDENING"
            print(f"  The gap is {direction}: {first[1]:.2f}x at seq {first[0]} -> "
                  f"{last[1]:.2f}x at seq {last[0]}.")
            if last[1] < first[1]:
                print("  Extrapolating the trend, a crossover would occur at a longer sequence than")
                print("  was tested. Say that as a trend with a bound, not as a prediction.")
            else:
                print("  The gap widens with sequence length, which contradicts the flash-attention")
                print("  hypothesis and should be reported as such.")

    if failures:
        print(f"\n  {len(failures)} configuration(s) failed: {failures}")

    if args.json_out:
        out = {"calls_per_graph": args.calls, "rows": rows, "crossover_seq": crossover,
               "failures": failures}
        Path(ROOT / args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(ROOT / args.json_out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
