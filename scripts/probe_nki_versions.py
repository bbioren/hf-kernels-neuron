"""Probe: are `nki` and `neuronxcc.nki` two GENERATIONS rather than two variants?

Finding #14 currently says the two NKI import paths have "different capabilities and
neither is a superset", and files it as an open question for the NKI team.

The Native PyTorch beta setup log suggests a simpler and less flattering explanation:

  - `from neuronxcc import nki`  -> the older NKI bundled with the compiler
  - `import nki`                 -> the newer standalone NKI package
  - and in NKI 0.5.0, `nl.arange` is listed as **removed**, replaced by `nl.ds` slicing

If that holds here, Finding #14 is not an upstream mystery — it is version skew, `nl.arange`
is a deprecated idiom, and our RMSNorm and SiLU kernels are written against the old API while
RoPE is written against the current one. That converts an "ask the NKI team" item into our own
tech debt, which is a materially different (and more actionable) conclusion.

Checks:
  1. Version behind each import path
  2. Whether `nl.ds` exists on both, and `nl.arange` on either
  3. Whether the two paths are actually the same module object

Run on trn2:
    python scripts/probe_nki_versions.py
"""

import importlib
import sys

SEP = "=" * 76


def ver(mod):
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if v is not None:
            return str(v)
    return "unknown"


def main():
    print(SEP)
    print("NKI import paths: versions and API surface")
    print(SEP)

    mods = {}
    for name in ("nki", "neuronxcc.nki"):
        try:
            mods[name] = importlib.import_module(name)
        except Exception as e:
            print(f"  {name:16s} import FAILED: {type(e).__name__}: {e}")
            mods[name] = None

    print()
    for name, m in mods.items():
        if m is None:
            continue
        print(f"  {name:16s} version={ver(m):40s}")
        print(f"  {'':16s} file={getattr(m, '__file__', '?')}")

    a, b = mods.get("nki"), mods.get("neuronxcc.nki")
    if a is not None and b is not None:
        print()
        print(f"  same module object? {a is b}")
        print(f"  same version?       {ver(a) == ver(b)}")

    print()
    print(SEP)
    print("Language-level API surface")
    print(SEP)
    lang = {}
    for name in ("nki.language", "neuronxcc.nki.language"):
        try:
            lang[name] = importlib.import_module(name)
        except Exception as e:
            print(f"  {name} import FAILED: {e}")
            lang[name] = None

    symbols = ["arange", "ds", "mgrid", "load", "store", "affine_range",
               "shared_hbm", "sbuf", "silu", "ndarray"]
    header = "  symbol         " + "".join(f"{n.split('.')[0]:>22s}" for n in lang)
    print(header)
    for s in symbols:
        row = f"  {s:14s}"
        for name, m in lang.items():
            if m is None:
                row += f"{'n/a':>22s}"
            else:
                row += f"{('yes' if hasattr(m, s) else 'NO'):>22s}"
        print(row)

    print()
    print(SEP)
    print("INTERPRETATION")
    print(SEP)
    va, vb = (ver(a) if a else "?"), (ver(b) if b else "?")
    if a is not None and b is not None and va != vb:
        print(f"  Two different NKI versions are installed:")
        print(f"    import nki               -> {va}")
        print(f"    from neuronxcc import nki -> {vb}")
        print()
        print("  => Finding #14 is VERSION SKEW, not a capability split between peers.")
        print("     If `nl.arange` is absent from the newer one, it is a removed/deprecated")
        print("     idiom, and our RMSNorm + SiLU kernels are written against the old API.")
        print("     That is our tech debt to pay, not an upstream question to file.")
    else:
        print("  Versions match or could not be determined — Finding #14's framing may")
        print("  stand. Do not revise it on this evidence alone.")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
