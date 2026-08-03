"""Is the NKI-vs-torch device gap an artifact of running on compiler defaults?

THE OPEN QUESTION THIS CLOSES
Every measurement in this project ran with NEURON_CC_FLAGS unset. That was deliberate and
consistent — changing flags mid-project would have made runs incomparable — but it leaves one
objection unanswered: the reviewers' instinct that "there shouldn't be a slowdown" would be
satisfied if the default target/LNC selection were penalising the NKI path specifically.

That is worth ruling out rather than asserting, because it is the single configuration choice that
could invalidate the device-time comparisons in Findings #25 and #26.

WHAT IT MEASURES
The same fixed amount of work — N chained applications of one op — computed via the NKI kernel and
via the torch op, under several NEURON_CC_FLAGS settings. What matters is not the absolute times
(they will differ between flag settings) but whether the NKI/torch RATIO moves. If the ratio is
stable across settings, the gap is not a flag artifact and the existing conclusions stand. If it
collapses under some setting, that setting is the one every measurement should have used and a
chunk of this project needs re-running.

Flags are a compile-time input, so each setting needs its own process. This script re-execs itself
per setting rather than trying to change flags in-flight, which would silently reuse a cached NEFF
compiled under the previous setting.

Usage — on trn2, from the repo root:
    python scripts/probe_compiler_flags.py                 # sweep all settings
    python scripts/probe_compiler_flags.py --child "..."   # internal, one setting
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# The settings worth testing. "" is what every existing measurement used.
SETTINGS = [
    "",
    "--target trn2",
    "--target trn2 --lnc 1",
    "--target trn2 --lnc 2",
    "--target trn2 -O2",
]

N_CALLS = 28
ROWS, COLS = 512, 3072


def child(flags):
    """Run one flag setting in this process and print a JSON line. Never imported."""
    if flags:
        os.environ["NEURON_CC_FLAGS"] = flags
    else:
        os.environ.pop("NEURON_CC_FLAGS", None)
    # Isolate the compile cache per setting, so a NEFF built under different flags is never reused.
    tag = flags.replace(" ", "_").replace("-", "") or "default"
    os.environ["NEURON_COMPILE_CACHE_URL"] = f"/tmp/nccache_{tag}"

    import functools
    import statistics
    import time

    sys.path.insert(0, str(ROOT / "tests"))
    import torch
    import torch.nn.functional as F

    from nki_test_utils import load_kernel_module, require_neuron

    require_neuron()
    import torch_xla.core.xla_model as xm

    # Finding #24 fix, so the ~52 ms subprocess does not swamp the comparison.
    import nki.compiler.target as nki_target

    nki_target._detect_target = functools.lru_cache(maxsize=1)(nki_target._detect_target)

    dev = xm.xla_device()
    mod = load_kernel_module("neuron_silu")
    if not mod._HAS_NKI:
        print(json.dumps({"flags": flags, "error": "NKI unavailable"}))
        return 1
    layer = mod.layers.NeuronSiLU().to(dev)
    x = torch.randn(ROWS, COLS, dtype=torch.bfloat16).to(dev)

    def bench(fn):
        for _ in range(2):                       # warm: compile under THESE flags
            out = fn()
            xm.mark_step()
            xm.wait_device_ops()
            del out
        s = []
        for _ in range(3):
            xm.mark_step()
            xm.wait_device_ops()
            t0 = time.perf_counter()
            out = fn()
            xm.mark_step()
            xm.wait_device_ops()
            s.append((time.perf_counter() - t0) * 1e3)
            del out
        return statistics.median(s)

    def nki_chain():
        out = x
        for _ in range(N_CALLS):
            out = layer(out)
        return out

    def torch_chain():
        out = x
        for _ in range(N_CALLS):
            out = F.silu(out)
        return out

    nki_ms = bench(nki_chain)
    torch_ms = bench(torch_chain)
    print(json.dumps({
        "flags": flags or "(unset — project default)",
        "nki_ms": round(nki_ms, 3),
        "torch_ms": round(torch_ms, 3),
        "ratio": round(nki_ms / torch_ms, 3) if torch_ms else None,
    }))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child is not None:
        return child(args.child)

    print("=" * 78)
    print("COMPILER-FLAG CONTROL: does the NKI/torch ratio depend on NEURON_CC_FLAGS?")
    print("=" * 78)
    print(f"  {N_CALLS} chained SiLU applications, tile [{ROWS}, {COLS}] bf16, wall clock")
    print("  one subprocess per setting, isolated compile cache per setting")
    print("  the ABSOLUTE times will differ; only the RATIO matters here\n")

    rows = []
    for s in SETTINGS:
        label = s or "(unset — project default)"
        print(f"  running: {label} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--child", s],
            cwd=ROOT, capture_output=True, text=True,
        )
        line = next((l for l in proc.stdout.splitlines() if l.strip().startswith("{")), None)
        if proc.returncode != 0 or not line:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
            print(f"    FAILED exit {proc.returncode}")
            for t in tail:
                print(f"      {t}")
            rows.append({"flags": label, "error": f"exit {proc.returncode}"})
            continue
        r = json.loads(line)
        rows.append(r)
        print(f"    nki {r['nki_ms']:8.3f} ms   torch {r['torch_ms']:8.3f} ms   "
              f"ratio {r['ratio']}x")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  {'flags':<34} {'nki ms':>9} {'torch ms':>9} {'ratio':>8}")
    print("  " + "-" * 62)
    for r in rows:
        if "error" in r:
            print(f"  {r['flags']:<34} {r['error']:>28}")
        else:
            print(f"  {r['flags']:<34} {r['nki_ms']:>9.3f} {r['torch_ms']:>9.3f} "
                  f"{r['ratio']:>7.3f}x")

    ok = [r for r in rows if "ratio" in r and r["ratio"]]
    if len(ok) < 2:
        print("\n  too few settings succeeded to draw a conclusion")
        return 1

    ratios = [r["ratio"] for r in ok]
    nkis = [r["nki_ms"] for r in ok]
    torches = [r["torch_ms"] for r in ok]
    spread = max(ratios) / min(ratios)
    nki_spread = max(nkis) / min(nkis)
    torch_spread = max(torches) / min(torches)

    print(f"\n  spread across settings:  ratio {spread:.2f}x   "
          f"NKI {nki_spread:.2f}x   torch {torch_spread:.2f}x")
    print()

    # The question is not "does the ratio move" — a ratio can move because the DENOMINATOR moved,
    # which says nothing about whether any setting helps NKI. Separate the two.
    if nki_spread < 1.15:
        print(f"  -> NKI is INVARIANT across compiler settings ({min(nkis):.2f}-{max(nkis):.2f} ms,")
        print(f"     {nki_spread:.2f}x spread). **No setting rescues the NKI path.** So the gap is")
        print("     not a compiler-flag artifact, which is what this control set out to establish.")
        if torch_spread > 1.15:
            print(f"     The ratio spread ({spread:.2f}x) is driven by TORCH moving "
                  f"({torch_spread:.2f}x), not NKI.")
            worst = max(ok, key=lambda r: r["torch_ms"])
            print(f"     Specifically '{worst['flags']}' makes torch slower "
                  f"({worst['torch_ms']:.3f} ms), which flatters the ratio without helping NKI.")
        print("     -> The open item is CLOSED for the wall-clock gap. Record it.")
    elif spread < 1.25:
        print("  -> Both sides are stable and the ratio is stable. Gap is not a flag artifact.")
        print("     -> The open item is CLOSED for the wall-clock gap.")
    else:
        best = min(ok, key=lambda r: r["ratio"])
        print(f"  -> NKI itself MOVES with flags ({nki_spread:.2f}x). The best setting is")
        print(f"     '{best['flags']}' at {best['ratio']:.2f}x. Re-run the affected measurements")
        print("     under it before citing #25 or #26 — the default may have penalised NKI.")

    # --- scope limit, stated because it is easy to overclaim from this probe ----------------
    per_call = min(nkis) / N_CALLS
    print()
    print("  SCOPE LIMIT — read before citing this as settling Findings #25/#26:")
    print(f"    This measures WALL time. At {min(nkis):.2f} ms for {N_CALLS} calls that is")
    print(f"    {per_call:.3f} ms/call, which is the post-fix DISPATCH floor — so this run is")
    print("    ~97% dispatch and only a few percent device. It therefore establishes that the")
    print("    DISPATCH cost is flag-invariant, and does NOT by itself settle whether the")
    print("    DEVICE-time gap in #25/#26 depends on flags. For that, profile device time under")
    print("    two settings:")
    print("      NEURON_CC_FLAGS='--target trn2 --lnc 2' python scripts/run_device_profile_sweep.py \\")
    print("          --calls 1 28 --ops silu")
    print("    and compare against the default-flag profiles via analyse_fusion_barrier.py.")
    print()
    print("  Then update results/measurements.json: set the compiler_flags note and record what")
    print("  this closed (dispatch) versus what remains open (device).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
