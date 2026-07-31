"""Profile a NEFF containing N NKI calls, to find out what the ~52 ms per call actually is.

Finding #21 established that N NKI calls fuse into ONE XLA graph and ONE device execution, and
that the cost is still linear in N (27.39x for 28x the calls). So the ~52 ms is paid *inside* the
compiled NEFF. Three candidate explanations remain:

  1. per-custom-call scheduling overhead (pipeline drain, HBM round-trip, region switch)
  2. an emitted synchronisation/barrier per custom call
  3. real device compute time

This script generates the NEFFs so they can be captured with neuron-explorer. Deliberately does
NOT set NEURON_CC_FLAGS: every other measurement in this project ran on compiler defaults, and
changing them here would profile something other than what was measured.

The reference/consume path is kept on-device but trivial, and no CPU reference is computed on
device, to keep the NEFF count low and identifiable (see the profiling skill's guidance).

Usage (run once per N so each gets its own output dir):
    python scripts/profile_nki_call_cost.py --calls 1  --outdir /tmp/prof_n1
    python scripts/profile_nki_call_cost.py --calls 28 --outdir /tmp/prof_n28
"""

import argparse
import os
import sys
from pathlib import Path

# ---- must be set before torch_xla / neuron runtime init ---------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--calls", type=int, default=28)
ap.add_argument("--outdir", default="/tmp/prof_nki")
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--cols", type=int, default=3072)
ap.add_argument("--iters", type=int, default=4, help="executions, so --profile-nth-exec=2 works")
args = ap.parse_args()

os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = args.outdir
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

Path(args.outdir).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import time

import torch

from nki_test_utils import load_kernel_module, require_neuron


def main():
    require_neuron()
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    mod = load_kernel_module("neuron_silu")
    if not mod._HAS_NKI:
        print("NKI unavailable — refusing to profile a fallback path.")
        return 1

    layer = mod.layers.NeuronSiLU().to(dev)
    x = torch.randn(args.rows, args.cols, dtype=torch.bfloat16).to(dev)

    print(f"profiling {args.calls} NKI call(s), tile [{args.rows}, {args.cols}] bf16")
    print(f"  inspect dir: {args.outdir}")
    print(f"  NEURON_RT_VISIBLE_CORES={os.environ['NEURON_RT_VISIBLE_CORES']}")
    print(f"  NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '(unset — compiler defaults)')}")

    def graph():
        out = x
        for _ in range(args.calls):
            out = layer(out)
        return out

    for i in range(args.iters):
        t0 = time.perf_counter()
        out = graph()
        xm.mark_step()
        xm.wait_device_ops()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1e3
        tag = "(compile)" if i == 0 else ""
        print(f"  iter {i}: {ms:9.2f} ms  ({ms / args.calls:7.2f} ms/call) {tag}")
        del out

    print("\nNEFFs written under:", args.outdir)
    for p in sorted(Path(args.outdir).rglob("*.neff")):
        print("   ", p, f"({p.stat().st_size / 1024:.0f} KiB)")
    for p in sorted(Path(args.outdir).rglob("*.ntff")):
        print("   ", p, f"({p.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
