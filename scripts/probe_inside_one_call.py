"""What happens inside ONE @nki.jit call? cProfile the Python stack.

Established so far:
  - 28 NKI calls compile into ONE NEFF that executes in 0.609 ms of device time
    (neuron-explorer: 43% MBU, 95% active — the device execution is efficient)
  - the same work takes ~1459 ms of wall time
  - 99.9% of that wall time is spent BEFORE mark_step, while the host is issuing calls
  - torch-xla's own accounting is tiny: ExecuteTime 0.92 ms, LazyTracing 0.28 ms,
    TransferToDeviceTime 0, CompileTime 0

So the cost is host-side and synchronous, inside the call itself. That contradicts Finding #19,
which recorded ~0.36 ms of host dispatch per call. This resolves the contradiction by profiling
the Python call stack directly rather than inferring from wall clock.

Three measurements:
  1. wall time of a single call with NO mark_step and NO sync at all
     -> if ~52 ms, the cost is synchronous inside the call
  2. cProfile of one call, sorted by cumulative time
     -> names the function responsible
  3. wall time of the 2nd..Nth call with identical shapes
     -> is there a per-call cache miss, or is it paid every time?

Run on trn2:
    python scripts/probe_inside_one_call.py
"""

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch

from nki_test_utils import load_kernel_module, require_neuron

SEP = "=" * 84


def main():
    require_neuron()
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    mod = load_kernel_module("neuron_silu")
    if not mod._HAS_NKI:
        print("NKI unavailable — refusing to report a result.")
        return 1

    layer = mod.layers.NeuronSiLU().to(dev)
    x = torch.randn(512, 3072, dtype=torch.bfloat16).to(dev)

    print(SEP)
    print("INSIDE ONE NKI CALL")
    print(SEP)

    # warm: compile everything, so nothing below is a first-call artifact
    for _ in range(3):
        _ = layer(x)
    xm.mark_step()
    xm.wait_device_ops()

    # ---- 1. a single call, no mark_step, no sync -------------------------------------
    print()
    print("1. single call, NO mark_step, NO wait_device_ops")
    for i in range(6):
        t0 = time.perf_counter()
        out = layer(x)
        t1 = time.perf_counter()
        print(f"     call {i}: {(t1 - t0) * 1e3:8.2f} ms   (nothing synced)")
        del out
    xm.mark_step()
    xm.wait_device_ops()

    # Control: the same thing for a plain torch op, to show the harness isn't the cost.
    print()
    print("   control, plain torch op, same conditions:")
    for i in range(3):
        t0 = time.perf_counter()
        out = torch.nn.functional.silu(x)
        t1 = time.perf_counter()
        print(f"     call {i}: {(t1 - t0) * 1e3:8.2f} ms")
        del out
    xm.mark_step()
    xm.wait_device_ops()

    # ---- 2. cProfile one call ---------------------------------------------------------
    print()
    print("2. cProfile of ONE call, top 25 by cumulative time")
    pr = cProfile.Profile()
    pr.enable()
    out = layer(x)
    pr.disable()
    del out

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    for line in s.getvalue().splitlines():
        print("   ", line)

    print()
    print("   same, sorted by TOTAL time (self time, excludes subcalls):")
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(15)
    for line in s2.getvalue().splitlines():
        print("   ", line)

    xm.mark_step()
    xm.wait_device_ops()
    return 0


if __name__ == "__main__":
    sys.exit(main())
