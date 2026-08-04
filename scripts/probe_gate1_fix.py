"""Is `setattr(torch, "neuron", True)` sufficient to fix Gate 1? (never tested until now)

WHY THIS EXISTS
Gate 1 is `hasattr(torch, "neuron")`, checked at `kernels/backends.py:198`. It is False on a Neuron
DLAMI even after `import torch_neuronx`, so `_backend()` returns `CUDA(...)` and two things break:
build-variant resolution, and `python-depends` validation.

We diagnosed that, wrote the proposed fix into `PROPOSED_UPSTREAM_DIFF`, and **never applied it.**
Every measurement in the project routed around it instead — local repositories so there is no Hub
fetch, pure-Python kernels so there is no build variant to resolve, and `"python-depends": []` so
there is nothing to validate. That configuration is unaffected by Gate 1, which is exactly why the
gap went unnoticed.

So the project has been asserting two things it did not check:

  1. that `torch_neuronx` setting the attribute would be sufficient, and
  2. that our kernels could then honestly declare `"python-depends": ["nki"]`

Claim (2) is in the outbound message to HuggingFace, as the reason to queue an `nkilib` allowlist
entry. Asking another team to make a change on the strength of an untested assertion is the failure
mode Finding #9 already caught once: the first proposed fix for Gate 2 was in the wrong file, and only
applying it revealed that.

WHAT IT TESTS
Three things, before and after `setattr(torch, "neuron", True)`:

  A. what `_backend()` reports
  B. whether `validate_dependencies(..., ["nki"], _backend())` passes
  C. whether a real kernel package declaring `"python-depends": ["nki"]` loads

(C) is the one that matters, because it is what a published kernel would do. It uses a temporary copy
of our RoPE kernel with the honest manifest, so nothing in `kernels/` is modified.

Faking the attribute is not a proposal to ship. It reproduces the state the real fix would create, so
the sufficiency claim can be checked before it is sent to anyone.

Run on trn2:
    python scripts/probe_gate1_fix.py
    python scripts/probe_gate1_fix.py --json-out results/raw/gate1/gate1.json
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
SEP = "=" * 84


def report_backend():
    from kernels.backends import _backend

    try:
        b = _backend()
        return {"repr": repr(b), "name": b.name, "error": None}
    except Exception as e:
        return {"repr": None, "name": None, "error": f"{type(e).__name__}: {e}"}


def check_declare(dep: str):
    """Can a kernel declare `dep` right now?"""
    from kernels.backends import _backend
    from kernels.deps import validate_dependencies

    try:
        validate_dependencies("probe_gate1", [dep], _backend())
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def check_load_with_declared_dep(dep: str):
    """Copy our RoPE kernel, give it an HONEST manifest, and try to load it.

    This is the test that matters: it is what a published kernel would look like. Uses a temp copy so
    the committed kernel is untouched.
    """
    src = ROOT / "kernels" / "neuron_rope"
    if not src.exists():
        return {"ok": False, "error": f"missing {src}"}

    tmp = Path(tempfile.mkdtemp(prefix="gate1_"))
    try:
        dst = tmp / "neuron_rope_declared"
        shutil.copytree(src, dst)
        meta = json.loads((dst / "metadata.json").read_text())
        meta["python-depends"] = [dep]
        (dst / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

        from kernels import LocalFuncRepository

        # repo_path must be a Path, not a str — LocalFuncRepository passes it to
        # get_local_kernel(), which does path arithmetic on it. A str produces
        # "TypeError: unsupported operand type(s) for /: 'str' and 'str'", which reads like a
        # library bug and is not one. The first version of this probe made that mistake and the
        # verdict logic then reported the FIX as insufficient, on the strength of my own error.
        repo = LocalFuncRepository(repo_path=dst, func_name="apply_rotary_pos_emb")
        repo.load()      # this is the path that calls validate_dependencies()
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def snapshot(label, dep):
    print(f"  --- {label} ---")
    b = report_backend()
    print(f"    _backend()                     {b['repr'] or b['error']}")
    d = check_declare(dep)
    print(f"    declare python-depends=[{dep!r}]   "
          f"{'PASSES' if d['ok'] else 'REJECTED'}")
    if not d["ok"]:
        print(f"      {d['error']}")
    l = check_load_with_declared_dep(dep)
    print(f"    load a kernel declaring it     {'OK' if l['ok'] else 'FAILED'}")
    if not l["ok"]:
        print(f"      {l['error']}")
    print()
    return {"backend": b, "declare": d, "load": l}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep", default="nki")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import torch

    print(SEP)
    print("GATE 1: is setattr(torch, 'neuron', ...) sufficient?")
    print(SEP)
    print(f"  Gate 1 is `hasattr(torch, \"neuron\")` at kernels/backends.py:198.")
    print(f"  hasattr(torch, 'neuron') = {hasattr(torch, 'neuron')} "
          f"(torch {torch.__version__})")
    try:
        import torch_neuronx  # noqa: F401
        print(f"  after `import torch_neuronx`: {hasattr(torch, 'neuron')}  "
              f"<- this is the bug")
    except Exception as e:
        print(f"  torch_neuronx import failed: {type(e).__name__}: {e}")
    print()

    had_attr = hasattr(torch, "neuron")
    before = snapshot("BEFORE (as shipped)", args.dep)

    # Reproduce the state the real fix would create. NOT a proposal to ship this line.
    torch.neuron = True
    try:
        after = snapshot("AFTER setattr(torch, 'neuron', True)", args.dep)
    finally:
        if not had_attr:
            delattr(torch, "neuron")

    print(SEP)
    print("VERDICT")
    print(SEP)
    fixed_backend = after["backend"]["name"] == "neuron"
    fixed_declare = after["declare"]["ok"] and not before["declare"]["ok"]
    fixed_load = after["load"]["ok"] and not before["load"]["ok"]

    print(f"  _backend() cuda -> neuron        {'YES' if fixed_backend else 'NO'}")
    print(f"  can declare python-depends       {'YES' if fixed_declare else 'NO'}"
          f"{'  (was already passing)' if before['declare']['ok'] else ''}")
    print(f"  kernel with honest manifest loads {'YES' if fixed_load else 'NO'}"
          f"{'  (was already loading)' if before['load']['ok'] else ''}")
    print()

    if fixed_backend and after["declare"]["ok"] and after["load"]["ok"]:
        print("  SUFFICIENT for dependency declaration. One attribute takes `_backend()` from")
        print("  cuda to neuron, which makes the `neuron` section of python_depends.json reachable,")
        print(f"  and a kernel declaring `{args.dep}` then loads. So the ask to torch_neuronx is")
        print("  verified rather than assumed, and our kernels can carry an honest manifest as soon")
        print("  as it lands.")
        print()
        print("  STILL UNTESTED: build-variant resolution. Our kernels are pure Python with no")
        print("  build/ directory, so nothing here exercises whether a repo containing")
        print("  build/torch29-neuron-x86_64-linux/ resolves once the backend reports neuron.")
        rc = 0
    elif fixed_backend:
        print("  PARTIALLY SUFFICIENT. `_backend()` reports neuron, but declaring the dependency")
        print("  still does not work — so something beyond Gate 1 is involved and the message to")
        print("  HuggingFace needs correcting before it goes out.")
        rc = 1
    else:
        print("  NOT SUFFICIENT. Setting the attribute did not change what `_backend()` reports,")
        print("  so the proposed fix is wrong or incomplete. Do NOT send the ask as written.")
        rc = 1

    if args.json_out:
        out = {"dep": args.dep, "torch": torch.__version__,
               "before": before, "after": after,
               "backend_fixed": fixed_backend,
               "declare_works_after": after["declare"]["ok"],
               "load_works_after": after["load"]["ok"],
               "build_variant_resolution": "UNTESTED — our kernels have no build/ directory"}
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\n  wrote {args.json_out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
