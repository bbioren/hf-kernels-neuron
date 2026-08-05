"""What changes on the Native PyTorch stack? Re-tests every stack-dependent finding in one pass.

WHY
Every measurement in this project ran on torch-xla. Samir Araujo (HF kernels) pointed out that is the
wrong stack — torch-xla is being deprecated, and on the native backend a model on-device reports
device type "neuron", not "xla". That puts several findings in question, and this probe settles which:

  Gate 1  hasattr(torch, "neuron") is False, so kernels._backend() reports CUDA on a Neuron host,
          so build-variant resolution and python-depends validation both read the wrong table.
  Gate 2  model.device.type is "xla", never "neuron", so transformers' kernelize() can never match a
          "neuron" mapping entry — and fails SILENTLY, returning success with every layer unchanged.
  #28     the residual per-call dispatch cost was torch_xla's xla_op_registry rebuilding the XLA
          computation per call. That module is torch_xla; it may not be in the native path at all.
  #24     ~52 ms/call from nki/compiler/target.py::_detect_target forking `neuron-ls`. That is NKI's
          own code in CompileKernel, shared across framework paths, so it probably still applies.

Also checks the thing most likely to bite: the drop ships **NKI 0.6.0b1**, and all three of our
kernels are written against NKI 0.5.0. A prior 0.6.0 alpha was recorded as having a substantially
different API, so the kernels may not import or compile.

Run with the native venv:
    /home/ubuntu/native_venv/bin/python scripts/probe_native_stack.py
    /home/ubuntu/native_venv/bin/python scripts/probe_native_stack.py --json-out results/raw/native/native.json
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
SEP = "=" * 88


def section(t):
    print()
    print(SEP)
    print(t)
    print(SEP)


def try_(label, fn):
    """Run fn, print label -> result, return (ok, value_or_error)."""
    try:
        v = fn()
        print(f"  {label:<38} {v}")
        return True, v
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:150]}"
        print(f"  {label:<38} FAILED  {msg}")
        return False, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    out = {}

    import torch

    section("STACK IDENTITY")
    out["torch"] = torch.__version__
    print(f"  torch                                  {torch.__version__}")
    for mod in ("torch_neuronx", "torch_xla", "nki", "neuronxcc", "kernels", "transformers"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "(no __version__)")
        except Exception as e:
            v = f"NOT IMPORTABLE ({type(e).__name__})"
        out[mod] = str(v)
        print(f"  {mod:<38} {v}")

    section("GATE 1 — hasattr(torch, 'neuron') and what _backend() reports")
    ok, has = try_("hasattr(torch, 'neuron')", lambda: hasattr(torch, "neuron"))
    out["gate1_hasattr"] = has
    try_("privateuse1 backend name", lambda: torch._C._get_privateuse1_backend_name())
    ok, be = try_("kernels._backend()",
                  lambda: repr(__import__("kernels.backends", fromlist=["_backend"])._backend()))
    out["gate1_backend"] = be
    ok, nm = try_("kernels._backend().name",
                  lambda: __import__("kernels.backends", fromlist=["_backend"])._backend().name)
    out["gate1_backend_name"] = nm
    if nm == "neuron":
        print("  -> GATE 1 IS GONE on this stack. python_depends.json's neuron section is now")
        print("     reachable, and build-variant resolution will look for neuron variants.")
    else:
        print(f"  -> GATE 1 STILL PRESENT: _backend() says {nm!r}, not 'neuron'.")

    section("GATE 1b — can a kernel declare its real dependency now?")
    def declare():
        from kernels.backends import _backend
        from kernels.deps import validate_dependencies
        validate_dependencies("probe_native", ["nki"], _backend())
        return "PASSES"
    ok, r = try_("validate_dependencies(['nki'])", declare)
    out["gate1b_declare_nki"] = r

    section("GATE 2 — what device type does a model report?")
    ok, dev = try_("torch.randn(2,2,device='neuron').device",
                   lambda: str(torch.randn(2, 2, device="neuron").device))
    out["gate2_tensor_device"] = dev
    ok, dt = try_("  .device.type", lambda: torch.randn(2, 2, device="neuron").device.type)
    out["gate2_device_type"] = dt
    if dt == "neuron":
        print("  -> GATE 2 IS GONE on this stack. transformers' kernelize() derives the device from")
        print("     model.device.type, which is now 'neuron', so a 'neuron' mapping entry matches")
        print("     without any patch. The silent no-op cannot occur.")
    else:
        print(f"  -> GATE 2 STILL PRESENT: device.type is {dt!r}.")

    section("REAL COMPUTE — is the host runtime adequate without the deb/ packages?")
    def compute():
        a = torch.randn(256, 256, device="neuron", dtype=torch.bfloat16)
        b = torch.randn(256, 256, device="neuron", dtype=torch.bfloat16)
        c = torch.mm(a, b)
        if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
            torch.neuron.synchronize()
        got = c.cpu().float()
        ref = a.cpu().float() @ b.cpu().float()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), ref.flatten(), dim=0).item()
        return f"OK  cos_sim={cos:.6f}"
    ok, comp = try_("bf16 256x256 matmul on device", compute)
    out["compute"] = comp
    if not ok:
        print("  -> compute failed. This is the signal that the deb/ runtime+driver packages are")
        print("     required (the beta guide says the public DLAMI driver is not compatible).")

    section("#28 — is torch_xla's op registry even in this path?")
    ok, r = try_("import torch_xla.core.xla_op_registry",
                 lambda: "importable (torch_xla present)"
                 if __import__("torch_xla.core.xla_op_registry") else "?")
    out["xla_op_registry"] = r
    if not ok:
        print("  -> torch_xla is absent, so Finding #28 (the per-call XLA computation rebuild) does")
        print("     not apply on this stack. The dispatch cost has to be re-measured from scratch.")

    section("#24 — is the neuron-ls subprocess still in NKI's dispatch path?")
    def detect():
        import inspect

        import nki.compiler.target as t
        src = inspect.getsource(t._detect_target)
        forks = "check_output" in src or "subprocess" in src
        return f"_detect_target forks a subprocess: {forks}"
    ok, r = try_("nki.compiler.target._detect_target", detect)
    out["finding24_still_forks"] = r

    section("OUR KERNELS UNDER NKI 0.6.0b1 (written for 0.5.0)")
    sys.path.insert(0, str(ROOT / "tests"))
    for name in ("neuron_rmsnorm", "neuron_rope", "neuron_silu"):
        def load(name=name):
            from nki_test_utils import load_kernel_module
            m = load_kernel_module(name)
            return f"imported, _HAS_NKI={getattr(m, '_HAS_NKI', '?')}"
        ok, r = try_(name, load)
        out[f"kernel_{name}"] = r

    section("SUMMARY")
    g1 = out.get("gate1_backend_name") == "neuron"
    g2 = out.get("gate2_device_type") == "neuron"
    print(f"  Gate 1 (backend detection)   {'RESOLVED by the native stack' if g1 else 'STILL PRESENT'}")
    print(f"  Gate 2 (device type)         {'RESOLVED by the native stack' if g2 else 'STILL PRESENT'}")
    print(f"  compute works                {out.get('compute')}")
    print()
    if g1 and g2:
        print("  Both upstream asks in the Samir message are XLA-path artifacts. They should be")
        print("  withdrawn, not filed. The honest finding is that they are real ON TORCH-XLA and")
        print("  that torch-xla is the wrong stack — which is worth telling him, because the silent")
        print("  no-op will still bite anyone who tries the Kernel Hub on the XLA path.")

    if args.json_out:
        p = ROOT / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
