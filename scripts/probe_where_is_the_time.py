"""Where does the ~52 ms per NKI call actually go?

The device profile (scripts/profile_nki_call_cost.py + neuron-explorer) says the NEFF containing
28 NKI calls executes in 0.609 ms of device time, at 43% memory-bandwidth utilisation and 95%
active time. The same execution takes ~1460 ms of wall time. So ~99.96% of the cost is not on
the device, which contradicts the "it is inside the NEFF" reading in Finding #21 and needs
resolving before either claim is published.

This splits the wall time using torch-xla's own accounting. Unlike the earlier probe, which only
read sample COUNTS, this reads the ACCUMULATORS:

  ExecuteTime          runtime's view of how long the device execution took
  CompileTime          compilation (must be 0 in steady state)
  LazyTracing          building the IR graph on the host
  TransferToDeviceTime host -> device copies
  DeviceLockWait       contention on the device lock

  wall - ExecuteTime  = host-side work outside the execute call
  ExecuteTime - 0.6ms = runtime overhead wrapped around the device execution

Run on trn2:
    python scripts/probe_where_is_the_time.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch

from nki_test_utils import load_kernel_module, require_neuron

SEP = "=" * 84
N_CALLS = 28
# neuron-explorer total_time for the 28-call NEFF. This is a CONSTANT FROM A PRIOR RUN, not
# measured by this script — it cannot be, since reading it requires neuron-explorer on a captured
# NTFF. Re-derive it with scripts/profile_nki_call_cost.py + summarise_device_profiles.py if the
# hardware or compiler changes, and update this. Labelled in the output so a reader does not mistake
# it for a fresh measurement sitting next to this run's fresh wall times.
DEVICE_MS_FROM_PROFILE = 0.609
DEVICE_MS_SOURCE = "prior run, scripts/profile_nki_call_cost.py — NOT measured here"

# torch-xla metric accumulators are NANOSECONDS. metrics_report() prints them formatted with units,
# which makes it easy to assume the raw accumulator is seconds; it is not. Reading them as seconds
# once produced a nine-digit millisecond figure in a table next to a 1459 ms wall time.
NS_TO_MS = 1e-6


def snap():
    """All torch-xla metrics as {name: (count, accumulator_seconds)}."""
    import torch_xla.debug.metrics as met

    out = {}
    for name in met.metric_names():
        d = met.metric_data(name)
        if d is not None:
            out[name] = (d[0], d[1])
    return out


def delta(a, b, name):
    """(count, seconds) increase for metric `name` between snapshots a and b."""
    c0, t0 = a.get(name, (0, 0))
    c1, t1 = b.get(name, (0, 0))
    return c1 - c0, t1 - t0


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

    def graph():
        out = x
        for _ in range(N_CALLS):
            out = layer(out)
        return out

    def sync():
        xm.mark_step()
        xm.wait_device_ops()

    print(SEP)
    print(f"WHERE DOES THE TIME GO? {N_CALLS} NKI calls, one graph, one device execution")
    print(SEP)

    # warm up: compile, then discard
    for _ in range(3):
        out = graph()
        sync()
        del out

    # ---- split one steady-state iteration --------------------------------------------
    rows = []
    for it in range(3):
        sync()
        before = snap()

        t0 = time.perf_counter()
        out = graph()
        t_trace = time.perf_counter()          # host finished issuing ops
        xm.mark_step()
        t_mark = time.perf_counter()           # mark_step returned
        xm.wait_device_ops()
        t_done = time.perf_counter()

        after = snap()
        del out

        wall_ms = (t_done - t0) * 1e3
        host_issue_ms = (t_trace - t0) * 1e3
        mark_ms = (t_mark - t_trace) * 1e3
        wait_ms = (t_done - t_mark) * 1e3

        ec, et = delta(before, after, "ExecuteTime")
        cc, ct = delta(before, after, "CompileTime")
        lc, lt = delta(before, after, "LazyTracing")
        tc, tt = delta(before, after, "TransferToDeviceTime")
        dc, dt = delta(before, after, "DeviceLockWait")

        # Accumulators are nanoseconds — see NS_TO_MS above.
        rows.append(dict(
            wall=wall_ms, issue=host_issue_ms, mark=mark_ms, wait=wait_ms,
            exec_n=ec, exec_ms=et * NS_TO_MS, comp_n=cc, comp_ms=ct * NS_TO_MS,
            trace_n=lc, trace_ms=lt * NS_TO_MS, xfer_ms=tt * NS_TO_MS,
            lock_ms=dt * NS_TO_MS,
        ))
        print(f"  iter {it}: wall {wall_ms:8.2f} ms  "
              f"[issue {host_issue_ms:7.2f} | mark_step {mark_ms:8.2f} | wait {wait_ms:7.2f}]  "
              f"ExecuteTime {et * NS_TO_MS:8.3f} ms x{ec}  compiles {cc}")

    r = rows[len(rows) // 2]

    print()
    print(SEP)
    print("BREAKDOWN (median iteration)")
    print(SEP)
    print(f"  wall time                        {r['wall']:9.2f} ms   100.0%")
    print(f"    host issuing ops (lazy trace)  {r['issue']:9.2f} ms   "
          f"{100 * r['issue'] / r['wall']:5.1f}%")
    print(f"    mark_step()                    {r['mark']:9.2f} ms   "
          f"{100 * r['mark'] / r['wall']:5.1f}%")
    print(f"    wait_device_ops()              {r['wait']:9.2f} ms   "
          f"{100 * r['wait'] / r['wall']:5.1f}%")
    print()
    print("  torch-xla counters (accumulators are nanoseconds; converted here):")
    print(f"    ExecuteTime                    {r['exec_ms']:9.3f} ms  "
          f"({r['exec_n']} execution(s))")
    print(f"    LazyTracing                    {r['trace_ms']:9.3f} ms  "
          f"({r['trace_n']} traced ops)")
    print(f"    TransferToDeviceTime           {r['xfer_ms']:9.3f} ms")
    print(f"    DeviceLockWait                 {r['lock_ms']:9.3f} ms")
    print(f"    CompileTime                    {r['comp_ms']:9.3f} ms  "
          f"({r['comp_n']} compiles — must be 0)")
    print()
    print(f"  device time (neuron-explorer)    {DEVICE_MS_FROM_PROFILE:9.3f} ms  "
          f"{100 * DEVICE_MS_FROM_PROFILE / r['wall']:5.2f}%")
    print(f"    ^ {DEVICE_MS_SOURCE}")

    print()
    print(SEP)
    print("INTERPRETATION")
    print(SEP)
    unaccounted = r["wall"] - r["exec_ms"]
    print(f"  wall - ExecuteTime            = {unaccounted:9.2f} ms  "
          f"(host-side, outside the execute call)")
    print(f"  ExecuteTime - device time     = "
          f"{r['exec_ms'] - DEVICE_MS_FROM_PROFILE:9.2f} ms  "
          f"(runtime overhead around a {DEVICE_MS_FROM_PROFILE} ms device execution)")
    print()
    if r["exec_ms"] > 0.5 * r["wall"]:
        print("  -> The time is inside ExecuteTime, but the device only accounts for")
        print(f"     {DEVICE_MS_FROM_PROFILE} ms of it. So it is RUNTIME overhead wrapped around the")
        print("     device execution, not device work and not framework tracing.")
        print("     That is a runtime/driver cost, and it is plausibly fixable — which would")
        print("     change the PoC recommendation.")
    elif r["issue"] > 0.5 * r["wall"]:
        print("  -> The time is on the HOST while issuing ops, before mark_step. Each")
        print("     @nki.jit call is doing expensive host-side work (tracing, lowering, or")
        print("     kernel re-specialisation) at call time. Also plausibly fixable.")
    else:
        print("  -> Not dominated by either; report the raw split.")

    print()
    print(f"  per NKI call: wall {r['wall'] / N_CALLS:.2f} ms, "
          f"device {DEVICE_MS_FROM_PROFILE / N_CALLS:.4f} ms, "
          f"ratio {r['wall'] / DEVICE_MS_FROM_PROFILE:.0f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
