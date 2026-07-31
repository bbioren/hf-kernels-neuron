"""Diagnose *why* torch.compile fails here, instead of accepting that it does.

Finding #21 recorded "torch.compile doesn't work on this stack" on the basis of one error
message repeated across backends:

    TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
    call_function <function silu at ...>

That is a fake-tensor propagation failure, which has several fixable causes (missing
torch_xla dynamo bridge registration, version skew between torch and torch_xla, wrong entry
point). Declaring the decisive experiment blocked on the strength of one error was premature.

This script:
  1. Reports torch / torch_xla versions and whether they are a matched pair
  2. Checks whether the openxla dynamo backend is actually registered
  3. Prints the FULL traceback rather than a truncated message
  4. Tries the alternative entry points: torch_xla.compile(), torch_xla.experimental.compile,
     and the dynamo bridge directly
  5. Tries progressively simpler functions (silu -> mul -> add) to find where it breaks
  6. Tries CPU tensors vs XLA tensors, to separate "compile is broken" from
     "compile is broken on XLA tensors"

Run on trn2:
    python scripts/diagnose_torch_compile.py
"""

import sys
import traceback

SEP = "=" * 84


def hdr(t):
    print()
    print(SEP)
    print(t)
    print(SEP)


def versions():
    hdr("1. Versions and pairing")
    import torch

    print(f"  torch            {torch.__version__}")
    try:
        import torch_xla

        print(f"  torch_xla        {getattr(torch_xla, '__version__', '?')}")
    except Exception as e:
        print(f"  torch_xla        IMPORT FAILED: {e}")
        return
    # torch_xla is built against a specific torch; a mismatch breaks the dynamo bridge.
    try:
        import torch_xla.version as v

        print(f"  torch_xla.version {[a for a in dir(v) if not a.startswith('_')]}")
        for a in dir(v):
            if not a.startswith("_"):
                print(f"    {a} = {getattr(v, a)}")
    except Exception:
        pass


def backends():
    hdr("2. Is the openxla dynamo backend registered?")
    import torch

    try:
        from torch._dynamo import list_backends

        bes = sorted(str(b) for b in list_backends())
        print(f"  registered: {bes}")
        print(f"  'openxla' present: {'openxla' in bes}")
    except Exception as e:
        print(f"  list_backends failed: {e}")

    # The bridge registers the backend as a side effect of import on some versions.
    for mod in ["torch_xla.core.dynamo_bridge",
                "torch_xla._dynamo.dynamo_bridge",
                "torch_xla.experimental"]:
        try:
            __import__(mod)
            print(f"  import {mod:38s} OK")
        except Exception as e:
            print(f"  import {mod:38s} {type(e).__name__}: {str(e)[:60]}")

    try:
        from torch._dynamo import list_backends

        bes = sorted(str(b) for b in list_backends())
        print(f"  after bridge imports, 'openxla' present: {'openxla' in bes}")
    except Exception:
        pass


def full_traceback():
    hdr("3. Full traceback for the failing case")
    import torch
    import torch.nn.functional as F

    try:
        import torch_xla.core.xla_model as xm

        dev = xm.xla_device()
    except Exception as e:
        print(f"  no xla device: {e}")
        return

    x = torch.randn(64, 64).to(dev)

    def f(t):
        return F.silu(t)

    try:
        torch._dynamo.reset()
        cf = torch.compile(f, backend="openxla")
        out = cf(x)
        xm.mark_step()
        print(f"  UNEXPECTEDLY SUCCEEDED: {out.sum().item():.4f}")
    except Exception:
        print("  full traceback:")
        traceback.print_exc()


def simpler_ops():
    hdr("4. Where does it break? simplest ops first, CPU vs XLA")
    import torch
    import torch.nn.functional as F

    try:
        import torch_xla.core.xla_model as xm

        dev = xm.xla_device()
    except Exception as e:
        print(f"  no xla device: {e}")
        return

    cases = [
        ("add",  lambda t: t + 1.0),
        ("mul",  lambda t: t * 2.0),
        ("relu", lambda t: F.relu(t)),
        ("silu", lambda t: F.silu(t)),
    ]
    for where, target in (("cpu", "cpu"), ("xla", dev)):
        print(f"  --- tensors on {where} ---")
        for name, fn in cases:
            x = torch.randn(64, 64).to(target)
            try:
                torch._dynamo.reset()
                cf = torch.compile(fn, backend="openxla" if where == "xla" else "inductor")
                out = cf(x)
                if where == "xla":
                    xm.mark_step()
                _ = out.sum().item()
                print(f"    {name:6s} OK")
            except Exception as e:
                print(f"    {name:6s} {type(e).__name__}: {str(e).replace(chr(10),' ')[:90]}")


def alternative_entrypoints():
    hdr("5. Alternative graph-mode entry points")
    import torch
    import torch.nn.functional as F

    try:
        import torch_xla.core.xla_model as xm

        dev = xm.xla_device()
    except Exception as e:
        print(f"  no xla device: {e}")
        return

    x = torch.randn(64, 64).to(dev)

    def f(t):
        return F.silu(t)

    # a) torch_xla.compile (newer torch_xla exposes this)
    try:
        import torch_xla

        if hasattr(torch_xla, "compile"):
            cf = torch_xla.compile(f)
            out = cf(x)
            xm.mark_step()
            print(f"  torch_xla.compile           OK  sum={out.sum().item():.4f}")
        else:
            print("  torch_xla.compile           not present")
    except Exception as e:
        print(f"  torch_xla.compile           {type(e).__name__}: {str(e)[:80]}")

    # b) dynamo bridge's extract_compiled_graph, used by the openxla backend internally
    try:
        from torch_xla.core import dynamo_bridge

        print(f"  dynamo_bridge attrs: "
              f"{[a for a in dir(dynamo_bridge) if 'compil' in a.lower()][:6]}")
    except Exception as e:
        print(f"  dynamo_bridge               {type(e).__name__}: {str(e)[:60]}")

    # c) plain lazy-tensor mode IS graph mode — establish the baseline it provides
    try:
        out = f(x)
        xm.mark_step()
        print(f"  lazy-tensor (mark_step)     OK  sum={out.sum().item():.4f}")
        print("    NOTE: torch-xla is already a graph runtime. Ops accumulate into an HLO")
        print("    graph and compile at mark_step. So 'does graph mode help' is partly")
        print("    answerable without torch.compile at all — see scripts/probe_neff_count.py")
    except Exception as e:
        print(f"  lazy-tensor                 {type(e).__name__}: {str(e)[:80]}")


def main():
    versions()
    backends()
    full_traceback()
    simpler_ops()
    alternative_entrypoints()
    print()
    print(SEP)
    print("DIAGNOSIS COMPLETE")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
