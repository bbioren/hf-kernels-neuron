"""Is the per-call `create_computation` rebuild cacheable? (Open item B12 — 91% of what's left)

THE FINDING THIS TESTS, which is source-confirmed before it is measured
After Finding #24 removed the ~52 ms `neuron-ls` subprocess, ~0.5 ms/call remained, and cProfile
put it in `create_computation` under `torch_xla`'s op registry, rebuilding the XLA computation and
its HLO protobufs on every invocation — on a warm path where the kernel is already compiled.

Reading the source explains why, and it is Finding #24's shape a second time: **a cache exists and
the code path throws it away.**

`torch_xla/core/xla_op_registry.py` defines `Op`, which holds

    self._computations = dict()          # keyed on pickle.dumps([shapes, kwargs])

and its own docstring says: *"Python based XLA operations should be preferably registered globally,
in order to amortize the lowering cost."*

`nki/framework/_torch_xla.py::TorchXlaKernel.__call__` does this instead:

    @xla_hlo_call                         # -> xla_call -> xla_op_registry.register -> Op(...)
    def nki_custom_call(*tensors):
        ...
    xla_result = nki_custom_call(*input_tensors)

The decorator runs INSIDE `__call__`, so every kernel invocation constructs a brand-new `Op` with an
empty `_computations` dict. The cache is never cold by accident — it is newly created, and therefore
always empty, on every call. #24 was "target resolution runs while building the cache key, so a hit
still pays the subprocess." This is "the computation cache lives on an object that is recreated per
call."

WHY THE KEY IS SAFE, and this is the part that decides whether the fix is legitimate
The lowering closure captures `config = nir.build_config()` — output specs, `backend_config_b64`,
operand/output aliases, `has_collectives`. All of it comes from `nir`, which comes from
`self._cached_compile_to_bir(frontend, converted_inputs, compile_opts)`, which is ALREADY memoised on
`self._generate_cache_key(converted_inputs, compile_opts)`. So the same key implies the same `nir`,
which implies the same `config`, which implies the same closure. Reusing the `Op` under that key is
not a guess about what is safe to share; it is the key NKI already uses for the thing the closure is
built from.

Two guards make that concrete rather than argued:
  - the Op is cached only when NKI's own compile cache is enabled, since with
    `NKI_DISABLE_COMPILE_CACHE` set `nir` is rebuilt per call and could in principle differ
  - a null key (unhashable arguments) falls through to the original uncached path

WHY THIS SCRIPT IS SHAPED LIKE probe_target_override_fix.py
Same reason: a speedup with degraded accuracy is a bug, not a fix. Every variant is checked against a
CPU reference in the same process, the baseline is re-run LAST as a control so ordering cannot
explain the result, and the probe reports cache hit/miss counts so a timing improvement cannot be
credited to the cache without evidence the cache was used.

It also refuses to patch a version of NKI it does not recognise. The patch reimplements
`TorchXlaKernel.__call__`, so it is tied to that function's structure; the landmarks it depends on are
asserted in the installed source first, and the source hash is printed for the record.

Run on trn2:
    python scripts/probe_op_registry_cache.py
    python scripts/probe_op_registry_cache.py --json-out results/raw/op-cache/op_cache.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent))

import torch

# The patch itself lives in nki_dispatch_fixes so that this probe and measure_mfu.py apply the
# SAME code. A copied patch is a patch that will eventually differ between the number and the
# verification of the number.
from nki_dispatch_fixes import LANDMARKS, check_patch_applies, fix_op_registry_cache, \
    fix_target_detection
from nki_test_utils import load_kernel_module, require_neuron

SEP = "=" * 88
N_CALLS = 28
ITERS = 3


def cos_sim(a, b):
    a = a.detach().float().flatten().cpu()
    b = b.detach().float().flatten().cpu()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    require_neuron()
    import torch_xla.core.xla_model as xm

    # Finding #24's fix, applied for EVERY variant. This probe measures the residual that remains
    # after #24, so leaving the subprocess in would bury the whole effect under a 52 ms constant.
    fix_target_detection(verbose=False)

    dev = xm.xla_device()
    mod = load_kernel_module("neuron_silu")
    if not mod._HAS_NKI:
        print("NKI unavailable — refusing to report a result.")
        return 1

    layer = mod.layers.NeuronSiLU().to(dev)
    x_cpu = torch.randn(512, 3072, dtype=torch.bfloat16)
    x = x_cpu.to(dev)
    ref = torch.nn.functional.silu(x_cpu.float())

    print(SEP)
    print("IS THE PER-CALL create_computation REBUILD CACHEABLE?  (open item B12)")
    print(SEP)

    ok, src_hash, missing = check_patch_applies()
    if not ok:
        print("  REFUSING TO PATCH — the installed NKI dispatch path does not match what this")
        print("  patch was written against. Re-read nki/framework/_torch_xla.py and update")
        print("  scripts/nki_dispatch_fixes.py before trusting any number from this probe.")
        return 1
    print(f"  ({len(LANDMARKS)} landmarks, so the patch applies to this version)")
    print()
    print("  The cache being tested already exists, in torch_xla:")
    print("    xla_op_registry.Op.__init__:  self._computations = dict()")
    print("    Op.__call__:                  keyed on pickle.dumps([shapes, kwargs])")
    print("  NKI applies @xla_hlo_call INSIDE __call__, so a fresh Op — with a fresh empty")
    print("  _computations — is constructed on every kernel invocation.")
    print()

    def bench(label):
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
        print(f"  {label:32s} {med:9.2f} ms total   {med / N_CALLS:7.3f} ms/call   "
              f"cos_sim {sim:.6f}")
        return med, sim

    print(f"  {N_CALLS} chained NKI calls, median of {ITERS}, tile [512, 3072] bf16")
    print("  Finding #24's lru_cache applied throughout, so this is the RESIDUAL")
    print()

    base_ms, base_sim = bench("baseline (post-#24)")

    stats, restore = fix_op_registry_cache(verbose=False)
    if stats is None:
        print("  patch refused to apply; aborting")
        return 1
    try:
        fix_ms, fix_sim = bench("with Op registry cached")
    finally:
        restore()

    base2_ms, base2_sim = bench("baseline again (control)")

    print()
    print(SEP)
    print("VERDICT")
    print(SEP)
    drift = abs(base2_ms - base_ms) / base_ms * 100
    speedup = base_ms / max(fix_ms, 1e-9)
    print(f"  baseline            {base_ms / N_CALLS:8.3f} ms/call")
    print(f"  baseline again      {base2_ms / N_CALLS:8.3f} ms/call   (within {drift:.1f}% of the "
          f"first, so ordering is not the cause)")
    print(f"  Op registry cached  {fix_ms / N_CALLS:8.3f} ms/call   {speedup:7.2f}x faster")
    print()
    print(f"  cache behaviour: {stats!r}")
    print(f"  accuracy: baseline {base_sim:.6f}  cached {fix_sim:.6f}  "
          f"control {base2_sim:.6f}")
    print()

    accurate = min(base_sim, fix_sim, base2_sim) > 0.999
    identical = abs(fix_sim - base_sim) < 1e-6
    used = stats["hit"] > 0
    faster = fix_ms < 0.9 * base_ms

    if not used:
        print("  INCONCLUSIVE — the cache was never hit, so any timing difference is noise.")
        print("  Every call produced a distinct key, which would mean the key is not stable")
        print("  across identical invocations. Investigate _generate_cache_key before retrying.")
        rc = 1
    elif faster and accurate and identical:
        print(f"  ANSWERED: YES, it is cacheable. {speedup:.2f}x on the residual, with cosine")
        print(f"  similarity identical to the baseline at 1e-6, and {stats['hit']} cache hits")
        print("  against 1 miss — so the computation really was being rebuilt per call and really")
        print("  is reusable.")
        print()
        print("  This closes open item B12 as a QUESTION. It is not a shipped fix: it is a runtime")
        print("  monkeypatch verified on one kernel and one shape. What it establishes is that the")
        print("  remaining ~91% of the regression is a caching bug of the same kind as #24, in the")
        print("  same dispatch path, and that the upstream change is small — register the lowering")
        print("  once per compile-cache key instead of once per call.")
        rc = 0
    elif faster and not identical:
        print("  FASTER BUT NOT IDENTICAL. Do not report this as a fix. Reusing the Op changed the")
        print("  result, which means the key does not fully determine the lowering. Find out which")
        print("  part of `config` varies under a fixed key before going further.")
        rc = 1
    elif accurate and not faster:
        print("  NO MEANINGFUL SPEEDUP. The Op rebuild is not the dominant residual cost, so")
        print("  cProfile's attribution needs re-reading. Report the raw numbers; claim nothing.")
        rc = 0
    else:
        print("  INCONCLUSIVE. Report the raw numbers and do not claim a fix.")
        rc = 1

    if args.json_out:
        out = {
            "source_sha256_16": src_hash,
            "n_calls": N_CALLS,
            "iters": ITERS,
            "baseline_ms_per_call": round(base_ms / N_CALLS, 4),
            "baseline_control_ms_per_call": round(base2_ms / N_CALLS, 4),
            "op_cached_ms_per_call": round(fix_ms / N_CALLS, 4),
            "speedup": round(speedup, 3),
            "control_drift_pct": round(drift, 2),
            "cache": stats.as_dict(),
            "cos_sim": {"baseline": base_sim, "op_cached": fix_sim, "control": base2_sim},
            "accurate": accurate,
            "identical_to_1e6": identical,
            "verdict": "cacheable" if (faster and accurate and identical and used) else "see stdout",
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\n  wrote {args.json_out}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
