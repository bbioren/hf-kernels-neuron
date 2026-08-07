#!/usr/bin/env python3
"""Pre-flight: does the build-variant layout load through the *local* loader?

Purpose. Before uploading anything to the Hub, confirm that the layout
`scripts/build_hub_repo.py` emits is one the `kernels` library actually resolves, and
find out which variant it picks. This exercises the same `_import_from_path` and
`Metadata.read_from_file` code that the Hub path uses (`utils.py:195-218`), so a pass
here means the only untested things left for the Hub are download and trust.

It also answers a question we could not answer from reading the source: with both
`build/torch-neuron/` and `build/torch-universal/` present, which one wins?
`variants.py:526` computes `universal_order = 1 if backend_name == "universal" else 0`,
which *suggests* the specific backend sorts first, but the surrounding tuple ordering is
easier to measure than to reason about.

Run:
    ./scripts/run_native.sh scripts/probe_hub_layout.py
    ./scripts/run_native.sh scripts/probe_hub_layout.py --repo dist/hub/neuron-rmsnorm
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SEP = "=" * 78


def probe(repo_path: Path) -> int:
    import kernels
    from kernels import get_local_kernel

    print(SEP)
    print("Hub layout pre-flight (local loader)")
    print(SEP)
    print(f"  kernels     {kernels.__version__}")
    print(f"  repo path   {repo_path}")

    if not repo_path.is_dir():
        print(f"  FAIL: {repo_path} does not exist. Run scripts/build_hub_repo.py first.")
        return 1

    build_dir = repo_path / "build"
    variants_on_disk = sorted(p.name for p in build_dir.iterdir() if p.is_dir()) if build_dir.is_dir() else []
    print(f"  variants on disk: {variants_on_disk or 'NONE'}")

    # What does the library think of these variants, before we try to load?
    print(f"\n{'-' * 78}\nVariant resolution\n{'-' * 78}")
    try:
        from kernels.backends import _backend
        from kernels.variants import get_variants_local, resolve_variant

        backend = _backend()
        print(f"  system backend       {backend}  (name={getattr(backend, 'name', '?')!r})")

        parsed = get_variants_local(build_dir)
        print(f"  parsed variants      {[v.variant_str for v in parsed]}")

        selected, decisions = resolve_variant(parsed)
        print(f"  SELECTED             {selected.variant_str if selected else 'NONE'}")
        for d in decisions:
            kind = type(d).__name__
            vs = getattr(getattr(d, "variant", None), "variant_str", "?")
            reason = getattr(d, "reason", "")
            print(f"    {kind:16} {vs:20} {reason}")
    except Exception as e:
        print(f"  (introspection unavailable: {type(e).__name__}: {e})")

    # The load itself. This is the part that must work.
    print(f"\n{'-' * 78}\nLoad\n{'-' * 78}")
    try:
        mod = get_local_kernel(repo_path)
    except Exception as e:
        print(f"  FAIL  {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print(f"  loaded module        {mod}")
    print(f"  module file          {getattr(mod, '__file__', '?')}")

    layers = getattr(mod, "layers", None)
    if layers is None:
        print("  FAIL: module has no `layers` namespace, so kernelize() cannot use it.")
        return 1

    exported = [n for n in dir(layers) if not n.startswith("_")]
    print(f"  layers namespace     {exported}")

    # The class must be a real nn.Module subclass with a forward, and must declare
    # the two capability flags kernelize() reads.
    ok = True
    for name in exported:
        cls = getattr(layers, name)
        has_fwd = callable(getattr(cls, "forward", None))
        hb = getattr(cls, "has_backward", "<unset>")
        tc = getattr(cls, "can_torch_compile", "<unset>")
        print(f"    {name}: forward={has_fwd} has_backward={hb} can_torch_compile={tc}")
        if not has_fwd:
            ok = False

    print(f"\n{SEP}")
    if ok:
        variant_used = "?"
        f = getattr(mod, "__file__", "") or ""
        for v in variants_on_disk:
            if f"/build/{v}/" in f:
                variant_used = v
                break
        print(f"PASS — build-variant layout loads. Variant used: {variant_used}")
        print("Layout is valid for the Hub path. Remaining unknowns: download + trust.")
        return 0
    print("FAIL — layout loaded but the layer class is not usable.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="dist/hub/neuron-rmsnorm", help="staged repo dir")
    args = ap.parse_args()
    path = Path(args.repo)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return probe(path)


if __name__ == "__main__":
    sys.exit(main())
