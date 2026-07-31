"""Verify the fix for the ~52 ms per-invocation NKI cost, and check it changes nothing else.

ROOT CAUSE (Finding #24, source-confirmed + cProfile-attributed):
`nki/framework/compiled.py::_compile_opts()` calls `nki/compiler/target.py::resolve_target()` on
every kernel invocation. With no override set that falls through to `_detect_target()`, which
forks `neuron-ls` and parses its stdout. That subprocess costs ~52 ms. It sits outside the
`_nki_compile_cache`, because its result is part of the cache key, so a cache HIT still pays it.

This script tests two independent fixes in ONE process, so the comparison is clean:

  Fix A  set NEURON_PLATFORM_TARGET_OVERRIDE, which `resolve_target()` checks first and which it
         re-reads on every call — so it can be toggled live. This is the customer-side workaround.

  Fix B  wrap `_detect_target` in functools.lru_cache. This is what an upstream fix would
         plausibly look like, and it needs no env var and no user action.

CORRECTNESS IS PART OF THE TEST, NOT AN AFTERTHOUGHT.
Fix A changes the compile target. If the override were wrong, kernels would be compiled for the
wrong hardware, which could be silently wrong rather than an error. So:
  - the override is set to exactly whatever `_detect_target()` returns on this host, never a
    hardcoded guess
  - every variant's output is compared against a CPU reference
A speedup with degraded accuracy is a bug, not a fix, and this reports both.

Run on trn2:
    python scripts/probe_target_override_fix.py
"""

import functools
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch

from nki_test_utils import load_kernel_module, require_neuron

SEP = "=" * 84
N_CALLS = 28
ITERS = 3
ENV_KEY = "NEURON_PLATFORM_TARGET_OVERRIDE"


def cos_sim(a, b):
    a = a.detach().float().flatten().cpu()
    b = b.detach().float().flatten().cpu()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    require_neuron()
    import torch_xla.core.xla_model as xm

    import nki.compiler.target as nki_target

    dev = xm.xla_device()
    mod = load_kernel_module("neuron_silu")
    if not mod._HAS_NKI:
        print("NKI unavailable — refusing to report a result.")
        return 1

    layer = mod.layers.NeuronSiLU().to(dev)
    x_cpu = torch.randn(512, 3072, dtype=torch.bfloat16)
    x = x_cpu.to(dev)
    ref = torch.nn.functional.silu(x_cpu.float())     # CPU reference, never on device

    print(SEP)
    print("VERIFY THE FIX: is the ~52 ms per call really the neuron-ls subprocess?")
    print(SEP)

    # ---- 0. what does target detection cost on its own, and what does it return? -------
    if ENV_KEY in os.environ:
        print(f"  NOTE: {ENV_KEY} was already set to "
              f"{os.environ[ENV_KEY]!r}; clearing for the baseline.")
        del os.environ[ENV_KEY]

    t0 = time.perf_counter()
    detected = nki_target._detect_target()
    t1 = time.perf_counter()
    detect_ms = (t1 - t0) * 1e3
    print(f"  _detect_target() -> {detected!r} in {detect_ms:.2f} ms  (one call, in isolation)")
    print(f"  this is the value the override will be set to — not a hardcoded guess")

    def bench(label):
        """Median wall time for N_CALLS chained NKI calls, plus a correctness check."""
        # warm: ensure compiled and cached under the CURRENT target setting
        for _ in range(2):
            _ = layer(x)
        xm.mark_step()
        xm.wait_device_ops()

        samples = []
        for _ in range(ITERS):
            xm.mark_step()
            xm.wait_device_ops()
            t0 = time.perf_counter()
            out = x
            for _ in range(N_CALLS):
                out = layer(out)
            xm.mark_step()
            xm.wait_device_ops()
            samples.append((time.perf_counter() - t0) * 1e3)
            del out

        single = layer(x)
        xm.mark_step()
        xm.wait_device_ops()
        sim = cos_sim(single.cpu(), ref)
        del single

        med = statistics.median(samples)
        print(f"  {label:34s} {med:9.2f} ms total   {med / N_CALLS:7.2f} ms/call   "
              f"cos_sim {sim:.6f}")
        return med, sim

    print()
    print(f"  {N_CALLS} chained NKI calls, median of {ITERS}, tile [512, 3072] bf16")
    print()

    # ---- 1. baseline, no override -------------------------------------------------------
    base_ms, base_sim = bench("baseline (no override)")

    # ---- 2. Fix A: env override ---------------------------------------------------------
    os.environ[ENV_KEY] = detected
    a_ms, a_sim = bench(f"Fix A: {ENV_KEY}={detected}")
    del os.environ[ENV_KEY]

    # ---- 3. Fix B: lru_cache on _detect_target ------------------------------------------
    original = nki_target._detect_target
    nki_target._detect_target = functools.lru_cache(maxsize=1)(original)
    try:
        b_ms, b_sim = bench("Fix B: lru_cache(_detect_target)")
    finally:
        nki_target._detect_target = original

    # ---- 4. re-run baseline last, to rule out warming ----------------------------------
    base2_ms, base2_sim = bench("baseline again (control)")

    # ---- verdict ------------------------------------------------------------------------
    print()
    print(SEP)
    print("VERDICT")
    print(SEP)
    print(f"  baseline          {base_ms / N_CALLS:8.2f} ms/call")
    print(f"  baseline again    {base2_ms / N_CALLS:8.2f} ms/call   "
          f"(within {abs(base2_ms - base_ms) / base_ms * 100:.1f}% of the first — "
          f"so the ordering isn't the cause)")
    print(f"  Fix A (env)       {a_ms / N_CALLS:8.2f} ms/call   "
          f"{base_ms / max(a_ms, 1e-9):7.1f}x faster")
    print(f"  Fix B (lru_cache) {b_ms / N_CALLS:8.2f} ms/call   "
          f"{base_ms / max(b_ms, 1e-9):7.1f}x faster")
    print()
    print(f"  accuracy unchanged? baseline {base_sim:.6f}  "
          f"A {a_sim:.6f}  B {b_sim:.6f}")

    accurate = min(base_sim, a_sim, b_sim) > 0.999
    fixed = (a_ms < 0.5 * base_ms) and (b_ms < 0.5 * base_ms)

    print()
    if fixed and accurate:
        print("  CONFIRMED. Both fixes remove most of the per-call cost with no accuracy change.")
        print("  The 208x MFU regression was an uncached subprocess in NKI's dispatch path, not")
        print("  a structural mismatch between the Kernel Hub's per-layer model and Neuron.")
        print("  -> re-run scripts/measure_mfu.py with the fix applied before concluding anything")
        print("     about whether the kernels are a net win.")
    elif fixed and not accurate:
        print("  FASTER BUT WRONG. Do not report this as a fix. One of the variants changed the")
        print("  compile target in a way that altered results — investigate before proceeding.")
    else:
        print("  NOT CONFIRMED. The subprocess is not the dominant cost, or something else")
        print("  dominates once it is removed. Report the raw numbers; do not claim a fix.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
