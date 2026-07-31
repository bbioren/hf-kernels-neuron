"""Do the NKI kernels actually beat the torch ops they replace, on device?

THE GAP THIS FILLS
Every performance number in this project so far measures *dispatch* cost:
  - ~52 ms/call from an uncached `neuron-ls` subprocess (Finding #24, fixed)
  - ~0.59 ms/call residual from rebuilding the XLA computation per call (#24, open)
  - 0.02 ms/call of actual device time
None of it says whether the kernel is any good. That matters for the recommendation: if NKI
RMSNorm is *also* slower than torch RMSNorm on device, then fixing dispatch never produces a
speedup and the whole per-layer approach is pointless regardless. If NKI is faster on device, then
dispatch is the only thing standing between here and a win, and Fix 7 becomes the whole ballgame.

THE MEASUREMENT
Device time only, from neuron-explorer, for the same amount of work computed both ways:
  N chained applications of the op via NKI kernels   vs   N chained applications via torch
Same math, same shapes, same dtype, same compiler flags (defaults). Dispatch cost is excluded by
construction because we read the NEFF's `total_time`, not wall clock.

Note on fairness: with N chained torch ops the compiler can fuse across them, and N chained NKI
custom calls cannot fuse with each other. That is a real advantage of staying in the graph rather
than a measurement artifact, so it is reported rather than controlled away — but N=1 is also
measured so the fusion effect is visible separately.

Usage — one invocation per configuration, since each needs its own NEFF directory:
    python scripts/profile_nki_vs_torch_device.py --op silu    --impl nki   --calls 28
    python scripts/profile_nki_vs_torch_device.py --op silu    --impl torch --calls 28
    python scripts/profile_nki_vs_torch_device.py --op rmsnorm --impl nki   --calls 28
    python scripts/profile_nki_vs_torch_device.py --op rmsnorm --impl torch --calls 28

Then read device time out of each with:
    neuron-explorer view --output-format summary-json -n <neff> -s <ntff> | jq .[].total_time
`scripts/summarise_device_profiles.py` does that across all four and prints the comparison.
"""

import argparse
import functools
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--op", choices=["silu", "rmsnorm"], required=True)
ap.add_argument("--impl", choices=["nki", "torch"], required=True)
ap.add_argument("--calls", type=int, default=28)
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--cols", type=int, default=3072)
ap.add_argument("--outdir", default=None)
ap.add_argument("--iters", type=int, default=4)
args = ap.parse_args()

OUTDIR = args.outdir or f"/tmp/prof_{args.op}_{args.impl}_n{args.calls}"

# Must be set before torch_xla / neuron runtime init.
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = OUTDIR
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import time

import torch

from nki_test_utils import load_kernel_module, require_neuron


def main():
    require_neuron()
    import torch_xla.core.xla_model as xm

    # Apply the Finding #24 fix. Without it every NKI call forks `neuron-ls` for ~52 ms, which
    # does not affect device time but makes the run take minutes for no reason.
    import nki.compiler.target as nki_target

    nki_target._detect_target = functools.lru_cache(maxsize=1)(nki_target._detect_target)

    dev = xm.xla_device()
    x = torch.randn(args.rows, args.cols, dtype=torch.bfloat16).to(dev)

    if args.impl == "nki":
        mod = load_kernel_module(f"neuron_{args.op}")
        if not mod._HAS_NKI:
            print("NKI unavailable — refusing to produce a profile that measures the fallback.")
            return 1
        layer_cls = {"silu": "NeuronSiLU", "rmsnorm": "NeuronRMSNorm"}[args.op]
        layer = getattr(mod.layers, layer_cls)().to(dev)
        if args.op == "rmsnorm":
            # The stateless kernel reads weight/eps off the adopting module.
            layer.weight = torch.nn.Parameter(torch.ones(args.cols, dtype=torch.bfloat16).to(dev))
            layer.variance_epsilon = 1e-6
        apply_op = layer
    else:
        if args.op == "silu":
            apply_op = torch.nn.functional.silu
        else:
            w = torch.ones(args.cols, dtype=torch.bfloat16).to(dev)

            def apply_op(t, _w=w):
                # Same computation Qwen3RMSNorm performs, including the fp32 reduction.
                v = t.to(torch.float32)
                v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
                return (_w * v).to(t.dtype)

    print(f"op={args.op} impl={args.impl} calls={args.calls} "
          f"tile=[{args.rows}, {args.cols}] bfloat16")
    print(f"  inspect dir: {OUTDIR}")
    print(f"  NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '(unset — compiler defaults)')}")

    def graph():
        out = x
        for _ in range(args.calls):
            out = apply_op(out)
        return out

    for i in range(args.iters):
        t0 = time.perf_counter()
        out = graph()
        xm.mark_step()
        xm.wait_device_ops()
        ms = (time.perf_counter() - t0) * 1e3
        print(f"  iter {i}: wall {ms:9.2f} ms  ({ms / args.calls:7.3f} ms/call)"
              f"{'  (compile)' if i == 0 else ''}")
        del out

    print("\n  artifacts:")
    for p in sorted(Path(OUTDIR).rglob("*.neff")):
        print(f"    NEFF {p}  ({p.stat().st_size / 1024:.0f} KiB)")
    for p in sorted(Path(OUTDIR).rglob("*.ntff")):
        print(f"    NTFF {p}  ({p.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
