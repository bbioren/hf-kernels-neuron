"""Decisive probe: does an NKI call join the surrounding XLA graph, or force its own
device execution?

WHY THIS IS THE RIGHT EXPERIMENT
Finding #20 established a ~53 ms fixed cost per @nki.jit invocation. The open question was
framed as "would graph mode (torch.compile) amortise it?" That framing was sloppy, because
torch-xla is ALREADY a graph runtime: ops accumulate into an HLO graph and compile/execute
at mark_step. The graph-break experiment's variant A issued 28 NKI calls with a SINGLE
mark_step and still paid 28 x 52 ms — which only means something if those 28 calls really
were one graph. That is the assumption this probe tests directly, instead of via the
torch.compile proxy.

torch-xla exposes execution counters, so this is directly observable rather than inferred:
  metric "ExecuteTime"  -> one sample per device execution (graph launch)
  metric "CompileTime"  -> one sample per compile

THE MEASUREMENT
For N=28 NKI calls issued before a single mark_step:
  - if ExecuteTime count increases by ~28, each NKI call flushes and executes on its own.
    The 53 ms is per-execution overhead, the calls never share a graph, and torch.compile
    cannot fix it — the fix must make NKI a fusable custom call inside one HLO module.
  - if ExecuteTime count increases by ~1, the 28 calls DID share one graph, and the
    28 x 53 ms is inside a single device execution. Then it is device time or per-custom-call
    launch cost inside the NEFF: a compiler/kernel issue, not a framework-boundary issue.

Reading the counter BEFORE mark_step additionally shows whether nki.jit synchronises
internally (a de-facto mark_step per call).

Run on trn2:
    python scripts/probe_neff_count.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
import torch.nn.functional as F

from nki_test_utils import load_kernel_module, require_neuron

SEP = "=" * 84
N_CALLS = 28
ROWS, COLS = 512, 3072


def execs():
    """Number of device executions so far (count of ExecuteTime samples)."""
    import torch_xla.debug.metrics as met

    d = met.metric_data("ExecuteTime")
    return 0 if d is None else d[0]


def compiles():
    import torch_xla.debug.metrics as met

    d = met.metric_data("CompileTime")
    return 0 if d is None else d[0]


def sync():
    import torch_xla.core.xla_model as xm

    xm.mark_step()
    xm.wait_device_ops()


def main():
    require_neuron()
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

    dev = xm.xla_device()
    mod = load_kernel_module("neuron_silu")
    if not mod._HAS_NKI:
        print("NKI unavailable — refusing to report a result. See docs/poc-findings.md #16.")
        return 1

    # Same layer, shape and dtype as scripts/experiment_nki_graph_break.py, so the
    # execution counts here are directly comparable with that experiment's timings.
    nki_silu = mod.layers.NeuronSiLU().to(dev)

    print(SEP)
    print("DECISIVE PROBE: are N NKI calls one graph execution, or N?")
    print(SEP)
    print(f"  N_CALLS = {N_CALLS}, tile = [{ROWS}, {COLS}], dtype = bfloat16")
    print("  torch-xla is already a graph runtime; this counts real device executions.")

    x = torch.randn(ROWS, COLS, dtype=torch.bfloat16).to(dev)
    w = torch.randn(COLS, dtype=torch.bfloat16).to(dev)

    # ---- warm up so nothing below is measuring a compile -------------------------------
    for _ in range(3):
        _ = nki_silu(x)
        _ = F.silu(x) * w
    sync()

    print()
    print("  available metrics:", [m for m in met.metric_names()][:12])
    print()

    results = {}

    def run_variant(name, body, warmup=2, iters=3):
        """Warm up (to compile), then measure. Counters are read on a warm run so the
        execution count reflects steady state, not the compile path."""
        # Warm up: this is where the compile happens. Timings below must exclude it,
        # otherwise a 1.2 s compile masquerades as per-call cost.
        for _ in range(warmup):
            out = body()
            sync()
            del out

        c_before = compiles()
        samples = []
        for _ in range(iters):
            sync()
            e0 = execs()
            t0 = time.perf_counter()
            out = body()
            e_pre = execs()      # BEFORE mark_step: did the calls self-synchronise?
            sync()
            t1 = time.perf_counter()
            e1 = execs()
            samples.append((t1 - t0) * 1e3)
            del out
        results[name] = dict(
            wall_ms=sorted(samples)[len(samples) // 2],      # median
            all_ms=samples,
            execs_before_sync=e_pre - e0,
            execs_total=e1 - e0,
            compiles=compiles() - c_before,                  # must be 0 in steady state
        )
        return results[name]

    # A: N NKI calls, one mark_step
    def variant_a():
        acc = None
        for _ in range(N_CALLS):
            y = nki_silu(x)
            acc = y if acc is None else acc + y
        return acc

    # B: 1 NKI call, one mark_step
    def variant_b():
        return nki_silu(x)

    # C: N torch ops, one mark_step  (control: proves batching is visible in the counter)
    def variant_c():
        acc = None
        for _ in range(N_CALLS):
            y = F.silu(x) * w
            acc = y if acc is None else acc + y
        return acc

    # D: 1 torch op (control baseline)
    def variant_d():
        return F.silu(x) * w

    for name, fn in [
        (f"A  {N_CALLS} NKI calls, 1 mark_step", variant_a),
        ("B  1 NKI call,  1 mark_step", variant_b),
        (f"C  {N_CALLS} torch ops, 1 mark_step", variant_c),
        ("D  1 torch op,  1 mark_step", variant_d),
    ]:
        r = run_variant(name, fn)
        flag = "" if r["compiles"] == 0 else "  <-- COMPILED, TIME INVALID"
        print(f"  {name:34s} wall {r['wall_ms']:9.2f} ms   "
              f"execs(pre-sync) {r['execs_before_sync']:3d}   "
              f"execs(total) {r['execs_total']:3d}   compiles {r['compiles']}{flag}")
        print(f"  {'':34s}      samples {['%.2f' % s for s in r['all_ms']]}")

    # ---- interpretation ---------------------------------------------------------------
    print()
    print(SEP)
    print("INTERPRETATION")
    print(SEP)

    a = results[f"A  {N_CALLS} NKI calls, 1 mark_step"]
    b = results["B  1 NKI call,  1 mark_step"]
    c = results[f"C  {N_CALLS} torch ops, 1 mark_step"]
    d = results["D  1 torch op,  1 mark_step"]

    print(f"  control: {N_CALLS} torch ops -> {c['execs_total']} execution(s); "
          f"1 torch op -> {d['execs_total']}")
    if c["execs_total"] <= d["execs_total"] + 1:
        print("    control OK: many torch ops batch into a single execution, so the")
        print("    counter does detect graph batching. A negative result below is real.")
    else:
        print("    !! control did NOT batch. The counter may not mean what this assumes;")
        print("    treat the NKI result below as inconclusive.")

    print()
    print(f"  {N_CALLS} NKI calls -> {a['execs_total']} execution(s); "
          f"1 NKI call -> {b['execs_total']}")
    per_call = a["execs_total"] / N_CALLS if N_CALLS else 0
    print(f"  executions per NKI call: {per_call:.2f}")

    if a["execs_before_sync"] >= N_CALLS * 0.8:
        print("    -> Each nki.jit call SELF-SYNCHRONISES: the counter had already advanced")
        print("       before mark_step. The calls never shared a graph. Variant A of the")
        print("       graph-break experiment was NOT one graph, so its 28 x 52 ms says")
        print("       nothing about graph mode either way.")
    elif per_call >= 0.8:
        print("    -> Each NKI call is its OWN device execution even inside one mark_step")
        print("       region. The 53 ms is per-execution overhead. torch.compile would not")
        print("       help: the fix is for NKI to become a fusable custom call within a")
        print("       single HLO module, which is a compiler/runtime change, not a")
        print("       framework-mode change.")
    elif a["execs_total"] <= b["execs_total"] + 1:
        print("    -> The NKI calls DID share one device execution. The 53 ms per call is")
        print("       therefore INSIDE the NEFF: real device time or per-custom-call launch")
        print("       cost. Graph mode is already being applied and does not help; this is")
        print("       a kernel/compiler cost, not an integration cost.")
    else:
        print("    -> Partial batching. Report the raw numbers; needs a profile to resolve.")

    print()
    print(f"  wall time per NKI call in A: {a['wall_ms'] / N_CALLS:.2f} ms")
    print(f"  wall time for B (1 call):    {b['wall_ms']:.2f} ms")
    print(f"  A/B wall ratio: {a['wall_ms'] / max(b['wall_ms'], 1e-9):.2f}x "
          f"for {N_CALLS}x the calls")

    print()
    print("  full metrics report (tail):")
    rep = met.metrics_report().strip().splitlines()
    for line in rep[:40]:
        print("   ", line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
