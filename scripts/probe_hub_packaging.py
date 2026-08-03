"""Probe: what does Hub packaging require for a Neuron NKI kernel?

Week 3, T7. Finding #7 already established that a flat repo (no `build/<variant>/`
directory) loads via the fallback path while a variant-structured one does not,
because the variant resolver reads torch's build config and gets `cu128` on the
DLAMI.

This probe answers the questions that remain before a Neuron kernel could be
published:

  1. What backend does `kernels._backend()` report here? Everything below hinges
     on it.
  2. `python_depends.json` in kernels 0.15.2 contains a `neuron` backend entry
     whitelisting `nki`. Can a kernel actually DECLARE `python-depends: ["nki"]`
     and load? If `_backend()` reports cuda, the lookup should miss the neuron
     table and raise — meaning the neuron allowlist entry is unreachable, and the
     two bugs compound.
  3. What does `metadata.json` strictly require? Does the `digest` field matter?
  4. Does our real flat kernel layout load cleanly?

Run on trn2:
    python scripts/probe_hub_packaging.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEP = "=" * 72


def hdr(t):
    print()
    print(SEP)
    print(t)
    print(SEP)


def probe_backend():
    hdr("1. Which backend does the kernels library think we're on?")
    from kernels.utils import _backend

    try:
        b = _backend()
        print(f"  _backend()        = {b!r}")
        print(f"  _backend().name   = {b.name!r}")
        return b
    except Exception as e:
        print(f"  _backend() FAILED: {type(e).__name__}: {e}")
        return None


def probe_allowlist():
    hdr("2. Is `nki` declarable as a python-depends?")
    import kernels
    from kernels.deps import _DEPENDENCY_DATA

    print("  python_depends.json contents:")
    print(f"    general  : {sorted(_DEPENDENCY_DATA.general.keys())}")
    for name, deps in sorted(_DEPENDENCY_DATA.backends.items()):
        print(f"    {name:8s} : {sorted(deps.keys())}")

    neuron_deps = _DEPENDENCY_DATA.backends.get("neuron", {})
    print()
    if "nki" in neuron_deps:
        print("  => HF already whitelists `nki` for the neuron backend.")
        print("     (So HF has anticipated NKI kernels — good news.)")
    else:
        print("  => `nki` is NOT whitelisted for neuron.")

    # Now the real question: does declaring it actually work on this host?
    from kernels.utils import _backend
    from kernels.deps import validate_dependencies

    print()
    for dep in ["nki", "nkilib", "einops"]:
        try:
            validate_dependencies("probe", [dep], _backend())
            print(f"  validate_dependencies(['{dep}'])  -> OK")
        except Exception as e:
            print(f"  validate_dependencies(['{dep}'])  -> {type(e).__name__}: {e}")


def probe_declare_nki_in_real_kernel():
    """Copy our real RoPE kernel, declare python-depends: ["nki"], try to load."""
    hdr("3. Can a real kernel repo declare python-depends: ['nki'] and load?")

    from kernels import get_local_kernel

    src = PROJECT_ROOT / "kernels" / "neuron_rope"
    if not src.exists():
        print(f"  missing {src}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "neuron_rope"
        shutil.copytree(src, dst)

        meta_path = dst / "metadata.json"
        meta = json.loads(meta_path.read_text())

        # (a) as-is: python-depends == []
        try:
            get_local_kernel(dst)
            print("  python-depends: []          -> loads OK")
        except Exception as e:
            print(f"  python-depends: []          -> {type(e).__name__}: {e}")

        # (b) declaring nki, which is what an honest Neuron kernel should do
        meta["python-depends"] = ["nki"]
        meta_path.write_text(json.dumps(meta, indent=2))
        # fresh dir to dodge the module cache
        dst2 = Path(tmp) / "neuron_rope_declared"
        shutil.copytree(dst, dst2)
        try:
            get_local_kernel(dst2)
            print("  python-depends: ['nki']     -> loads OK")
        except Exception as e:
            print(f"  python-depends: ['nki']     -> {type(e).__name__}: {e}")
            print()
            print("  => The neuron allowlist entry for `nki` is UNREACHABLE on this")
            print("     host: validate_dependencies looks up the table for the")
            print("     backend that _backend() reports, and that is not neuron.")
            print("     Two bugs compound — a Neuron kernel cannot honestly declare")
            print("     its own dependency, and must lie with python-depends: [].")


def probe_metadata_requirements():
    hdr("4. What does metadata.json strictly require?")
    from kernels.utils import Metadata

    base = {
        "name": "neuron-probe",
        "id": "neuron_probe",
        "version": 0,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": {"type": "neuron"},
        "digest": {"algorithm": "sha256", "files": {}},
    }

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "metadata.json"

        p.write_text(json.dumps(base))
        try:
            m = Metadata.read_from_file(p)
            print(f"  full metadata            -> OK (name={m.name}, id={m.id})")
        except Exception as e:
            print(f"  full metadata            -> {type(e).__name__}: {e}")

        for drop in ["digest", "python-depends", "license", "version", "backend", "id", "name"]:
            trimmed = {k: v for k, v in base.items() if k != drop}
            p.write_text(json.dumps(trimmed))
            try:
                Metadata.read_from_file(p)
                print(f"  without '{drop}'{' ' * max(0, 16 - len(drop))}-> OK (optional)")
            except Exception as e:
                msg = str(e).replace("\n", " ")[:60]
                print(f"  without '{drop}'{' ' * max(0, 16 - len(drop))}-> REQUIRED ({type(e).__name__}: {msg})")


def probe_flat_layout_loads():
    hdr("5. Do our real flat kernel repos load?")
    from kernels import get_local_kernel

    for name in ["neuron_rmsnorm", "neuron_rope"]:
        path = PROJECT_ROOT / "kernels" / name
        if not path.exists():
            print(f"  {name:16s} -> MISSING")
            continue
        files = sorted(p.name for p in path.iterdir() if p.is_file())
        try:
            mod = get_local_kernel(path)
            has_layers = hasattr(mod, "layers")
            print(f"  {name:16s} -> OK   files={files}  layers_ns={has_layers}")
        except Exception as e:
            print(f"  {name:16s} -> FAIL {type(e).__name__}: {e}")


def main():
    print(SEP)
    print("PROBE: Hub packaging requirements for a Neuron NKI kernel")
    print(SEP)
    probe_backend()
    probe_allowlist()
    probe_declare_nki_in_real_kernel()
    probe_metadata_requirements()
    probe_flat_layout_loads()
    print()
    print(SEP)
    print("PROBE COMPLETE")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
