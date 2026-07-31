"""Experiment: does graph mode amortize the ~53 ms per-invocation NKI cost?

THIS IS THE MOST IMPORTANT OPEN QUESTION IN THE PROJECT.

Finding #20 established that every `@nki.jit` invocation from eager PyTorch/XLA costs ~53 ms
of fixed overhead, independent of problem size. At that price the entire 42 ms baseline
forward pass of Qwen3-0.6B is cheaper than a single NKI call, so any per-layer NKI swap loses
in eager mode and swapping more layers loses harder. MFU went 5.06% -> 0.02%.

If the cost is a per-invocation *framework boundary crossing*, then compiling the model should
amortize it: the NKI kernels become part of one compiled graph, entered once per step rather
than 169 times. If instead the cost is intrinsic to executing a NKI NEFF, compilation will not
help and the eager conclusion generalizes.

That distinction decides whether the recommendation is "don't invest in this integration" or
"invest in the compile path, not the eager path".

All three of our kernels declare `can_torch_compile = False`, which is the honest setting for
the Kernel Hub metadata today — but that flag governs whether `kernelize()` will select them in
a TORCH_COMPILE mode, not whether the underlying kernel *can* be compiled. This tests the
latter directly, bypassing the Kernel Hub.

WHAT IS MEASURED
  A. eager: N NKI calls                    -> expect ~53 ms/call (reproduces Finding #20)
  B. compiled: N NKI calls                 -> the question
  C. compiled: N torch ops (control)       -> shows compilation itself is working
  D. eager: N torch ops (control)

If B/N is much smaller than A/N, graph mode amortizes the cost and the compile path is viable.
If B ≈ A, the cost survives compilation and eager NKI's problem is NKI's problem.

Also reports whether Dynamo graph-broke, since a graph break around every NKI call would
reproduce eager behaviour while appearing to be "compiled".

Run on trn2:
    python scripts/experiment_torch_compile_nki.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from nki_test_utils import load_kernel_module, require_neuron

SEP = "=" * 84
N_CALLS = 8          # smaller than the 28-layer model: each eager call costs ~53 ms
ITERS = 5
WARMUP = 2


def sync():
    import torch_xla.core.xla_model as xm

    xm.mark_step()
    try:
        xm.wait_device_ops()
    except Exception:
        pass


def consume(t):
    """Force materialization; a discarded output can be eliminated entirely."""
    if isinstance(t, (tuple, list)):
        return sum(float(x.float().sum().item()) for x in t)
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


def pick_backend():
    """Find a working torch.compile backend for XLA on this stack."""
    candidates = ["openxla", "torchxla_trace_once", "aot_torchxla_trace_once", "inductor"]
    available = []
    try:
        from torch._dynamo import list_backends

        available = [str(b) for b in list_backends()]
    except Exception:
        pass
    print(f"  dynamo backends available: {available or '(could not list)'}")
    for c in candidates:
        if not available or c in available:
            return c, available
    return (available[0] if available else None), available


def main():
    dev = require_neuron()
    silu_mod = load_kernel_module("neuron_silu")
    if not silu_mod._HAS_NKI:
        print("NKI unavailable")
        return 1

    print(SEP)
    print("Does graph mode amortize the ~53 ms per-invocation NKI cost?")
    print(SEP)
    print(f"  N_CALLS={N_CALLS}, shape [512, 3072] bf16, outputs consumed")
    print(f"  torch {torch.__version__}")

    backend, _ = pick_backend()
    print(f"  chosen backend: {backend}")

    nki_layer = silu_mod.layers.NeuronSiLU().to(dev)
    x = torch.randn(512, 3072, dtype=torch.bfloat16).to(dev)

    class NkiStack(nn.Module):
        def __init__(self, layer, n):
            super().__init__()
            self.layer = layer
            self.n = n

        def forward(self, t):
            for _ in range(self.n):
                t = self.layer(t)
            return t

    class TorchStack(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, t):
            for _ in range(self.n):
                t = F.silu(t)
            return t

    nki_stack = NkiStack(nki_layer, N_CALLS).to(dev)
    torch_stack = TorchStack(N_CALLS).to(dev)

    results = {}

    # ---- A: eager NKI ----------------------------------------------------
    print()
    print("  A. eager, NKI ...")
    try:
        results["A"] = timeit(lambda: nki_stack(x))
        print(f"     {results['A']:9.2f} ms  ({results['A']/N_CALLS:7.2f} ms/call)")
    except Exception as e:
        results["A"] = None
        print(f"     FAILED: {type(e).__name__}: {str(e)[:120]}")

    # ---- D: eager torch (control) ---------------------------------------
    print("  D. eager, torch (control) ...")
    try:
        results["D"] = timeit(lambda: torch_stack(x))
        print(f"     {results['D']:9.2f} ms  ({results['D']/N_CALLS:7.4f} ms/call)")
    except Exception as e:
        results["D"] = None
        print(f"     FAILED: {type(e).__name__}: {str(e)[:120]}")

    # ---- C: compiled torch control -------------------------------------
    #
    # MUST pass before B means anything. If torch.compile cannot handle plain F.silu on
    # this stack, a failure in B says nothing about NKI. Sweep dtype and backend to give
    # compilation the best chance of working before concluding anything.
    print()
    print("  C. compiled torch CONTROL — must pass for B to be interpretable")
    control_backend = None
    control_dtype = None
    for be in [backend, "openxla", "inductor", "eager"]:
        if be is None:
            continue
        for dt, xt in (("bf16", x), ("fp32", x.float())):
            try:
                torch._dynamo.reset()
                cc = torch.compile(TorchStack(N_CALLS).to(dev), backend=be)
                ms = timeit(lambda cc=cc, xt=xt: cc(xt), iters=3, warmup=2)
                print(f"     backend={be:22s} dtype={dt}  OK  {ms:9.3f} ms")
                if control_backend is None:
                    control_backend, control_dtype = be, dt
                    results["C"] = ms
                    results["C_x"] = xt
            except Exception as e:
                msg = str(e).replace("\n", " ")[:90]
                print(f"     backend={be:22s} dtype={dt}  FAILED  {type(e).__name__}: {msg}")
        if control_backend is not None:
            break

    if control_backend is None:
        print()
        print("     No working torch.compile configuration found on this stack.")
        print("     B cannot be interpreted: a NKI failure under compile would be")
        print("     indistinguishable from compile being broken generally.")
    else:
        print(f"     => using backend={control_backend} dtype={control_dtype} for B")
        backend = control_backend

    # ---- B: compiled NKI — the question ---------------------------------
    print()
    print(f"  B. compiled ({backend}), NKI  <-- the question")
    graph_breaks = None
    if control_backend is None:
        results["B"] = None
        print("     SKIPPED — no working compile configuration (see C).")
    else:
        xb = results.get("C_x", x)
        try:
            torch._dynamo.reset()
            try:
                import torch._dynamo.utils as dynamo_utils

                dynamo_utils.counters.clear()
            except Exception:
                dynamo_utils = None

            stack = NkiStack(nki_layer, N_CALLS).to(dev)
            c_nki = torch.compile(stack, backend=backend)
            results["B"] = timeit(lambda: c_nki(xb), iters=3, warmup=2)
            print(f"     {results['B']:9.2f} ms  ({results['B']/N_CALLS:7.2f} ms/call)")

            if dynamo_utils is not None:
                gb = dynamo_utils.counters.get("graph_break", {})
                graph_breaks = dict(gb)
                total = sum(gb.values())
                print(f"     dynamo graph breaks: {total}")
                for k, v in list(gb.items())[:5]:
                    print(f"       {v:4d}x  {str(k)[:90]}")
        except Exception as e:
            results["B"] = None
            print(f"     FAILED: {type(e).__name__}: {str(e).replace(chr(10), ' ')[:200]}")

    # ---- verdict ---------------------------------------------------------
    print()
    print(SEP)
    print("VERDICT")
    print(SEP)
    a, b, c, d = (results.get(k) for k in "ABCD")

    if a is None:
        print("  Eager NKI baseline failed; cannot compare.")
        return 1

    print(f"  eager    NKI   {a:10.2f} ms  ({a/N_CALLS:8.2f} ms/call)")
    if d is not None:
        print(f"  eager    torch {d:10.2f} ms  ({d/N_CALLS:8.4f} ms/call)")
    if c is not None:
        print(f"  compiled torch {c:10.2f} ms   (control: compilation is working)")
    if b is not None:
        print(f"  compiled NKI   {b:10.2f} ms  ({b/N_CALLS:8.2f} ms/call)")
    print()

    if b is None and control_backend is None:
        print("  INCONCLUSIVE — torch.compile does not work on this stack even for plain")
        print("  F.silu, so nothing can be concluded about NKI under compilation.")
        print()
        print("  This is a real finding, just a different one: the graph-mode question that")
        print("  Finding #20 makes decisive cannot be answered in this environment. Answering")
        print("  it needs either a stack where torch.compile works on Neuron (the torch.compile")
        print("  branch of the Native PyTorch beta) or guidance from the NKI / torch-neuronx")
        print("  teams on the supported way to call a NKI kernel from a compiled graph.")
        print()
        print("  Do NOT record this as 'NKI is incompatible with torch.compile'.")
    elif b is None:
        print("  Compiled torch works but compiled NKI does not.")
        print("  Since the control passed, this IS attributable to the NKI kernel: it does not")
        print("  survive Dynamo tracing on this stack without extra work (custom op")
        print("  registration, or an explicitly opaque boundary).")
        print()
        print("  Next step is a question for the NKI / torch-neuronx teams rather than more")
        print("  measurement: what is the supported way to call a NKI kernel from a compiled")
        print("  graph?")
    elif b < a / 4:
        print(f"  Graph mode AMORTIZES the cost: {a/b:.1f}x faster than eager.")
        print(f"  Per-call cost drops from {a/N_CALLS:.2f} ms to {b/N_CALLS:.2f} ms.")
        print()
        print("  => This changes the recommendation. The per-invocation charge is a framework")
        print("     boundary cost, not intrinsic to NKI execution, and the compile path is")
        print("     where the integration should be built. The Kernel Hub already models this:")
        print("     kernels can declare can_torch_compile=True and be selected only in")
        print("     TORCH_COMPILE modes. Our kernels should be re-evaluated for that flag.")
    else:
        print(f"  Graph mode does NOT help: compiled is {a/b:.2f}x eager.")
        print("  The ~53 ms is intrinsic to executing a NKI kernel here, not a cost of")
        print("  crossing the eager framework boundary.")
        print()
        print("  => Finding #20 generalizes beyond eager, and per-layer NKI kernel swapping")
        print("     is not viable on this stack in any mode. The remaining path would be")
        print("     whole-block or whole-model megakernels, which is not the Kernel Hub model.")

    if graph_breaks:
        print()
        print("  NOTE: dynamo reported graph breaks. If it broke around every NKI call, the")
        print("  'compiled' number above is really eager and should not be read as graph mode.")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
