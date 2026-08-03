"""Probe: the installed nkilib MLP kernel's exact signature, defaults, and enums.

Prerequisite for the standalone MLP spike. The GitHub `main` signature has 40 parameters
and nki-library's README warns that `main` is not guaranteed compatible with a given
compiler version, so read the *installed* copy rather than trusting the source we fetched.

Run on trn2:
    python scripts/probe_mlp_signature.py
"""

import inspect
import sys

SEP = "=" * 72


def main():
    try:
        from nkilib.core.mlp.mlp import mlp
    except Exception as e:
        print(f"cannot import nkilib.core.mlp.mlp: {type(e).__name__}: {e}")
        return 1

    print(SEP)
    print("Installed nkilib MLP kernel")
    print(SEP)
    target = getattr(mlp, "func", getattr(mlp, "__wrapped__", mlp))
    try:
        sig = inspect.signature(target)
    except Exception as e:
        print(f"  cannot introspect: {e}")
        return 1

    print(f"  parameters: {len(sig.parameters)}")
    print()
    for name, p in sig.parameters.items():
        default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
        print(f"    {name}{default}")

    print()
    print(SEP)
    print("Enums we need")
    print(SEP)
    try:
        from nkilib.core.utils.common_types import (
            ActFnType,
            ComputationMode,
            DtypeMode,
            NormType,
            QuantizationType,
        )

        for cls in (ActFnType, NormType, QuantizationType, ComputationMode, DtypeMode):
            members = [m.name for m in cls]
            print(f"  {cls.__name__:18s} {members}")
    except Exception as e:
        print(f"  could not import enums: {type(e).__name__}: {e}")

    print()
    print(SEP)
    print("Torch reference")
    print(SEP)
    for mod_name, sym in [
        ("nkilib.core.mlp.mlp_torch", "mlp_torch_ref"),
        ("nkilib.core.mlp.mlp_parameters", "is_mlp_tkg"),
    ]:
        try:
            m = __import__(mod_name, fromlist=[sym])
            obj = getattr(m, sym, None)
            print(f"  {mod_name}.{sym}: {'present' if obj is not None else 'MISSING'}")
            if obj is not None and sym == "is_mlp_tkg":
                try:
                    print(f"      signature {inspect.signature(obj)}")
                except Exception:
                    pass
        except Exception as e:
            print(f"  {mod_name}.{sym}: FAIL {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
