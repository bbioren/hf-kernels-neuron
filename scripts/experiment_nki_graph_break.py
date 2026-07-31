"""Experiment: why does an NKI call cost ~52 ms inside a model but ~0.8 ms alone?

THE OBSERVATION (scripts/measure_mfu.py):
  Qwen3-0.6B, 28 layers, seq 512, forward only, single logical core.
    baseline (no kernels)      41.95 ms/step
    NKI SiLU only, 28 calls  1495.54 ms/step  -> 51.9 ms of added cost per call
    all three, 169 calls     8753.65 ms/step  -> 51.6 ms of added cost per call
  Uniform per-call cost, independent of which kernel and of how much work it does
  (SiLU on [512, 3072] is trivial). Steady state: zero compiles during the timed loop,
  and step time is stable to within 0.2%.

  But in isolation the same SiLU kernel measured ~0.8 ms end-to-end including a sync.
  So being inside a model forward makes each call ~65x more expensive.

THE HYPOTHESIS: each `@nki.jit` call is an XLA custom call that cannot fuse into the
surrounding graph, so a forward pass containing N NKI calls becomes N+1 separate device
executions instead of one. If switching between NEFFs is expensive — loading a different
NEFF, draining the pipeline, round-tripping tensors through HBM — the cost would be per
*transition*, not per unit of work. That predicts a fixed cost per call regardless of kernel,
which is exactly what was measured.

THE TEST: hold the number of NKI calls fixed and vary only whether other work is
interleaved between them.

  A. N NKI calls back to back, nothing between      -> if cheap, transitions are the cost
  B. N NKI calls with a torch op between each       -> if expensive, transitions are the cost
  C. N torch ops only, no NKI                       -> control
  D. 1 NKI call, N times more data                  -> is cost per call or per byte?

If A is cheap and B is expensive, the cost is interleaving/graph-transition, and the
conclusion is about the *integration model*: a per-layer swap that alternates NKI and
framework ops is structurally penalised, independent of kernel quality.

If A and B are both expensive, the cost is per NKI call in a multi-call graph.

If D scales with data while A/B do not, the cost is fixed per call.

Run on trn2:
    python scripts/experiment_nki_graph_break.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
import torch.nn.functional as F

from nki_test_utils import load_kernel_module, require_neuron

SEP = "=" * 84
N_CALLS = 28   # matches the 28-layer model
ITERS = 8
WARMUP = 3


def sync():
    import torch_xla.core.xla_model as xm

    xm.mark_step()
    try:
        xm.wait_device_ops()
    except Exception:
        pass


def consume(t):
    """Force materialization. Without this XLA can eliminate the whole chain.

    Learned the hard way twice (Finding #19): a discarded output means the computation
    may never run, and the resulting timings look plausible while measuring nothing.
    """
    return float(t.float().sum().item())


def timeit(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        consume(fn())
        sync()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        consume(fn())
        s.append((time.perf_counter() - t0) * 1e3)
    s.sort()
    return statistics.median(s)


def main():
    dev = require_neuron()
    mod = load_kernel_module("neuron_silu")
    if not mod._HAS_NKI:
        print("NKI unavailable")
        return 1

    print(SEP)
    print("Why is an in-model NKI call ~52 ms when an isolated one is ~0.8 ms?")
    print(SEP)
    print(f"  N_CALLS={N_CALLS} (matches the 28-layer model), shape [512, 3072] bf16")
    print("  One mark_step + device wait per variant, so we time whole-graph execution.")

    layer = mod.layers.NeuronSiLU().to(dev)
    x = torch.randn(512, 3072, dtype=torch.bfloat16).to(dev)

    # ---- A: N NKI calls back to back --------------------------------------
    def variant_a():
        out = x
        for _ in range(N_CALLS):
            out = layer(out)
        return out

    # ---- B: N NKI calls, a torch op between each -------------------------
    def variant_b():
        out = x
        for _ in range(N_CALLS):
            out = layer(out)
            out = out * 1.0001          # cheap framework op, forces interleaving
        return out

    # ---- C: N torch ops only (control) ----------------------------------
    def variant_c():
        out = x
        for _ in range(N_CALLS):
            out = F.silu(out)
        return out

    # ---- D: 1 NKI call on N times the data ------------------------------
    x_big = torch.randn(512 * N_CALLS, 3072, dtype=torch.bfloat16).to(dev)

    def variant_d():
        return layer(x_big)

    # ---- E: 1 NKI call on the base shape, for the per-call floor ---------
    def variant_e():
        return layer(x)

    print()
    print("  timing (this compiles a few graphs first) ...")
    results = {}
    for name, fn, desc in [
        ("A", variant_a, f"{N_CALLS} NKI calls, back to back"),
        ("B", variant_b, f"{N_CALLS} NKI calls, torch op between each"),
        ("C", variant_c, f"{N_CALLS} torch F.silu calls (control)"),
        ("D", variant_d, f"1 NKI call on {N_CALLS}x the data"),
        ("E", variant_e, "1 NKI call, base shape (per-call floor)"),
    ]:
        try:
            ms = timeit(fn)
            results[name] = ms
            per_call = ms / N_CALLS if name in ("A", "B", "C") else ms
            unit = "per call" if name in ("A", "B", "C") else "total"
            print(f"    {name}. {desc:44s} {ms:9.2f} ms   ({per_call:8.2f} ms {unit})")
        except Exception as e:
            results[name] = None
            print(f"    {name}. {desc:44s} FAILED: {type(e).__name__}: "
                  f"{str(e).replace(chr(10), ' ')[:80]}")

    # ---- sweep: is there a cliff, or is it flat from the first call? -----
    #
    # An earlier isolated microbenchmark measured ~0.78 ms for a single SiLU call at
    # [128, 3072] — but that measurement failed its own validity gate (latency did not
    # scale with size), so it was suppressed. Sweeping rows settles whether the ~52 ms
    # is flat from the start or appears past some tile count.
    print()
    print("  Single-call sweep over row count (one NKI call, output consumed):")
    print(f"    {'rows':>8s} {'tiles':>6s} {'NKI ms':>10s} {'torch ms':>10s} {'ratio':>8s}")
    sweep = {}
    for rows in [128, 256, 512, 1024, 4096, 14336]:
        xs = torch.randn(rows, 3072, dtype=torch.bfloat16).to(dev)
        try:
            nki_ms = timeit(lambda xs=xs: layer(xs), iters=5, warmup=2)
            t_ms = timeit(lambda xs=xs: F.silu(xs), iters=5, warmup=2)
            sweep[rows] = (nki_ms, t_ms)
            print(f"    {rows:8d} {rows//128:6d} {nki_ms:10.2f} {t_ms:10.3f} "
                  f"{nki_ms/t_ms:7.0f}x")
        except Exception as ex:
            print(f"    {rows:8d}  FAILED: {type(ex).__name__}")

    if sweep:
        vals = [v[0] for v in sweep.values()]
        print(f"    NKI single-call range: {min(vals):.2f} - {max(vals):.2f} ms "
              f"across a {max(sweep)//min(sweep)}x row range")
        if max(vals) / min(vals) < 2.0:
            print("    => flat. The ~52 ms is a fixed cost paid on the FIRST call and every")
            print("       call, not something that appears at scale.")

    # ---- interpretation --------------------------------------------------
    print()
    print(SEP)
    print("INTERPRETATION")
    print(SEP)
    a, b, c, d, e = (results.get(k) for k in "ABCDE")
    if None in (a, b, c, d, e):
        print("  Incomplete data; cannot conclude.")
        return 1

    a_pc, b_pc, c_pc = a / N_CALLS, b / N_CALLS, c / N_CALLS
    print(f"  per-call: A(NKI only) {a_pc:.2f} ms | B(interleaved) {b_pc:.2f} ms | "
          f"C(torch) {c_pc:.2f} ms")
    print(f"  D: one NKI call on {N_CALLS}x data = {d:.2f} ms, vs A total {a:.2f} ms")
    print(f"  E: one NKI call, base shape = {e:.2f} ms   "
          f"(D/E = {d/e:.2f}x for {N_CALLS}x the data)")
    print()
    if d / e < 4:
        print(f"  E->D: {N_CALLS}x the data costs only {d/e:.1f}x the time, so a single call is")
        print("  dominated by FIXED per-call cost, not by the work it does.")
    print()

    print("  Cost model:")
    if d < a / 4:
        print(f"    D ({d:.1f} ms) is far cheaper than A ({a:.1f} ms) for the SAME total data.")
        print("    => cost is PER CALL, not per byte. One big call beats many small ones.")
    else:
        print(f"    D ({d:.1f} ms) is comparable to A ({a:.1f} ms).")
        print("    => cost tracks data volume, not call count.")

    print()
    print("  Interleaving:")
    if b_pc > 1.5 * a_pc:
        print(f"    B ({b_pc:.1f} ms/call) is markedly worse than A ({a_pc:.1f} ms/call).")
        print("    => interleaving framework ops between NKI calls adds cost on top of the")
        print("       per-call cost. Consistent with each NKI call forcing a graph break, so")
        print("       alternating NKI and framework ops multiplies device executions.")
    else:
        print(f"    B ({b_pc:.1f} ms/call) is close to A ({a_pc:.1f} ms/call).")
        print("    => interleaving is NOT the driver; the cost is intrinsic to each NKI call")
        print("       even when calls are adjacent.")

    print()
    print("  What this means for the Kernel Hub integration model:")
    if a_pc > 5 * c_pc:
        print(f"    An NKI call costs ~{a_pc/c_pc:.0f}x a framework op of the same shape even")
        print("    back-to-back. In eager mode, a per-layer swap replaces cheap framework ops")
        print("    with expensive-to-invoke custom calls, so the more layers you swap the worse")
        print("    it gets. That is a property of the integration, not of the kernels.")
        print()
        print("    It also reframes the fused-kernel argument: fusion helps not mainly by saving")
        print("    memory traffic but by collapsing many expensive invocations into one.")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
