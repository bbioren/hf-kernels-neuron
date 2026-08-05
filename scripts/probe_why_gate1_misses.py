"""Why doesn't Gate 1 block us? Show the two collections it would filter, and that both are empty.

Gate 1 is `hasattr(torch, "neuron")` being False, which makes `_backend()` report CUDA on a Neuron
host. That wrong answer is consumed in exactly two places on the kernel-load path:

    kernels/variants.py::resolve_variant(variants, backend)     picks a build directory
    kernels/utils.py:201  validate_dependencies(name, deps, _backend())   checks the manifest

In both cases the wrong backend is used as a lookup key into a collection. This prints those two
collections for a real kernel of ours. If both are empty, the wrong key is computed and then applied
to nothing, which is the whole reason six weeks of measurements were unaffected by a broken backend
check.

That is worth being explicit about, because "it works" and "it works without touching the part that
breaks on publish" are different claims, and only the second one is true here.

    python scripts/probe_why_gate1_misses.py
    python scripts/probe_why_gate1_misses.py --kernel neuron_rope
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="neuron_rmsnorm")
    args = ap.parse_args()

    from kernels.backends import _backend
    from kernels.variants import get_variants_local, resolve_variant

    p = ROOT / "kernels" / args.kernel
    if not p.exists():
        print(f"missing {p}")
        return 1

    print("=" * 84)
    print(f"WHY GATE 1 MISSES — kernels/{args.kernel}")
    print("=" * 84)
    print(f"  _backend()  = {_backend()!r}   <- WRONG. Gate 1. Should be Neuron().")
    print()

    print("  Consumer 1: resolve_variant(variants, backend) — picks a build directory")
    subdirs = [e.name for e in p.iterdir() if e.is_dir()]
    print(f"    subdirectories present         {subdirs}")
    v_root = get_variants_local(p)
    v_build = get_variants_local(p / "build")
    print(f"    get_variants_local(repo)       {v_root}")
    print(f"    get_variants_local(repo/build) {v_build}")
    print(f"    resolve_variant(...)           {resolve_variant(v_root, None)[0]}")
    print("    -> nothing parses as a build variant, so there is no list to filter. The wrong")
    print("       backend is computed and then applied to an empty collection. get_local_kernel")
    print("       falls through to its explicit-package-path branch and imports the repo root.")
    print()

    print("  Consumer 2: validate_dependencies(name, python_depends, _backend())")
    deps = json.loads((p / "metadata.json").read_text()).get("python-depends")
    print(f"    declared python-depends        {deps}")
    print("    -> the validator's body is `for dependency in dependencies:`, so an empty list")
    print("       means zero iterations. The wrong backend is passed in and never used.")
    print()

    print("=" * 84)
    print("CONCLUSION")
    print("=" * 84)
    empty_variants = not v_root and not v_build
    empty_deps = not deps
    if empty_variants and empty_deps:
        print("  Gate 1 gives a wrong answer to two questions and we ask neither of them.")
        print("  A wrong lookup key is harmless against a zero-length collection.")
        print()
        print("  This is NOT a fix and NOT robustness. It is the flat layout and the empty")
        print("  manifest each dodging one consequence. Both dodges end at publish time:")
        print("    - a Hub repo with real build/<variant>/ directories gives resolve_variant")
        print("      something to filter, and it will filter for the wrong backend")
        print("    - an honest \"python-depends\": [\"nki\"] gives the validator something to look")
        print("      up, and it will look in the cuda table (verified: probe_gate1_fix.py)")
        return 0
    print("  At least one collection is non-empty, so Gate 1 IS reachable in this configuration.")
    print(f"    variants empty: {empty_variants}   deps empty: {empty_deps}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
