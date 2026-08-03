"""Probe: does the NKI kernel actually EXECUTE, or are we silently on the fallback?

Motivation (Week 3): the Week 2 accuracy test reported `max_diff = 0.00e+00`
(bit-identical) for every shape. That is the signature of the PyTorch fallback,
not of a real NKI kernel, because `_pytorch_rmsnorm` is mathematically identical
to `Qwen3RMSNorm.forward`.

`NeuronRMSNorm.forward` gates on:
    if _HAS_NKI and hidden_states.device.type != "cpu":

and the test feeds CPU tensors. So the NKI branch is likely never taken.

This probe determines, unambiguously:
  1. Which branch the current kernel takes for CPU tensors
  2. Whether an @nki.jit kernel can be called with CPU tensors at all
  3. Whether it can be called with XLA (Neuron) tensors
  4. Whether the numerical result is correct in the cases that do run

Run on trn2:
    python scripts/probe_nki_execution.py
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
SEP = "=" * 68


def hdr(t):
    print()
    print(SEP)
    print(t)
    print(SEP)


def load_kernel_module():
    p = PROJECT_ROOT / "kernels" / "neuron_rmsnorm" / "__init__.py"
    spec = importlib.util.spec_from_file_location("neuron_rmsnorm", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cos_sim(a, b):
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
    ).item()


def reference_rmsnorm(x, w, eps):
    dt = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (w * x.to(dt))


def probe_1_which_branch():
    """Instrument the module to see which branch actually runs."""
    hdr("1. Which branch does NeuronRMSNorm.forward take for CPU tensors?")

    mod = load_kernel_module()
    print(f"  _HAS_NKI = {mod._HAS_NKI}")

    calls = {"nki": 0, "fallback": 0}

    real_nki = getattr(mod, "_nki_rmsnorm_kernel", None)
    real_fb = mod._pytorch_rmsnorm

    def spy_fb(*a, **k):
        calls["fallback"] += 1
        return real_fb(*a, **k)

    mod._pytorch_rmsnorm = spy_fb

    if real_nki is not None:
        def spy_nki(*a, **k):
            calls["nki"] += 1
            return real_nki(*a, **k)

        mod._nki_rmsnorm_kernel = spy_nki

    import torch.nn as nn

    layer = mod.layers.NeuronRMSNorm()
    layer.weight = nn.Parameter(torch.randn(128) * 0.5 + 1.0)
    layer.variance_epsilon = 1e-6

    x = torch.randn(1, 32, 128)
    print(f"  input device = {x.device}")
    with torch.no_grad():
        out = layer(x)

    print(f"  NKI kernel calls      = {calls['nki']}")
    print(f"  PyTorch fallback calls= {calls['fallback']}")
    if calls["fallback"] > 0 and calls["nki"] == 0:
        print()
        print("  => CONFIRMED: on CPU tensors the NKI kernel is NOT executed.")
        print("     The Week 2 'cos_sim=1.000000, max_diff=0.00e+00' results were")
        print("     measuring the PyTorch fallback against a mathematically")
        print("     identical reference. They do NOT validate the NKI kernel.")
    elif calls["nki"] > 0:
        print("  => NKI kernel executed.")
    return out


def probe_2_nki_on_cpu():
    """Can an @nki.jit kernel be called with CPU tensors at all?"""
    hdr("2. Can the @nki.jit kernel be called directly with CPU tensors?")

    mod = load_kernel_module()
    if not mod._HAS_NKI:
        print("  NKI not available, skipping")
        return

    w = torch.randn(128) * 0.5 + 1.0
    x = torch.randn(32, 128)
    try:
        out = mod._nki_rmsnorm_kernel(x, w, 1e-6)
        ref = reference_rmsnorm(x, w, 1e-6)
        print(f"  called OK. out.device={out.device} out.shape={tuple(out.shape)}")
        print(f"  cos_sim vs reference = {cos_sim(ref, out):.6f}")
        print(f"  max_abs_diff         = {(ref.float()-out.float()).abs().max().item():.3e}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  => @nki.jit requires tensors on a Neuron/XLA device.")


def probe_3_nki_on_xla():
    """Can the @nki.jit kernel run with tensors on the XLA (Neuron) device?"""
    hdr("3. Can the @nki.jit kernel run on XLA (Neuron) device tensors?")

    mod = load_kernel_module()
    if not mod._HAS_NKI:
        print("  NKI not available, skipping")
        return

    try:
        import torch_xla.core.xla_model as xm
    except Exception as e:
        print(f"  torch_xla unavailable: {e}")
        return

    dev = xm.xla_device()
    print(f"  xla device = {dev}")
    try:
        print(f"  device hw  = {xm.xla_device_hw(dev)}")
    except Exception as e:
        print(f"  xla_device_hw failed: {e}")

    hidden = 128
    rows = 32
    w_cpu = torch.randn(hidden) * 0.5 + 1.0
    x_cpu = torch.randn(rows, hidden)
    ref = reference_rmsnorm(x_cpu, w_cpu, 1e-6)

    try:
        x = x_cpu.to(dev)
        w = w_cpu.to(dev)
        out = mod._nki_rmsnorm_kernel(x, w, 1e-6)
        xm.mark_step()
        out_cpu = out.cpu()
        print(f"  called OK. out.shape={tuple(out_cpu.shape)}")
        cs = cos_sim(ref, out_cpu)
        md = (ref.float() - out_cpu.float()).abs().max().item()
        print(f"  cos_sim vs reference = {cs:.6f}")
        print(f"  max_abs_diff         = {md:.3e}")
        if md > 0:
            print("  => non-zero diff: this is REAL NKI hardware execution")
        print(f"  {'PASS' if cs > 0.999 else 'FAIL'} (target cos_sim > 0.999)")
    except Exception as e:
        import traceback

        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


def probe_4_layer_on_xla():
    """Does the full NeuronRMSNorm layer take the NKI branch on XLA tensors?"""
    hdr("4. Does NeuronRMSNorm take the NKI branch with XLA tensors?")

    mod = load_kernel_module()
    if not mod._HAS_NKI:
        print("  NKI not available, skipping")
        return
    try:
        import torch_xla.core.xla_model as xm
    except Exception as e:
        print(f"  torch_xla unavailable: {e}")
        return

    calls = {"nki": 0, "fallback": 0}
    real_nki = mod._nki_rmsnorm_kernel
    real_fb = mod._pytorch_rmsnorm

    def spy_nki(*a, **k):
        calls["nki"] += 1
        return real_nki(*a, **k)

    def spy_fb(*a, **k):
        calls["fallback"] += 1
        return real_fb(*a, **k)

    mod._nki_rmsnorm_kernel = spy_nki
    mod._pytorch_rmsnorm = spy_fb

    import torch.nn as nn

    dev = xm.xla_device()
    hidden = 128
    layer = mod.layers.NeuronRMSNorm()
    w_cpu = torch.randn(hidden) * 0.5 + 1.0
    layer.weight = nn.Parameter(w_cpu.clone())
    layer.variance_epsilon = 1e-6
    layer = layer.to(dev)

    x_cpu = torch.randn(1, 32, hidden)
    ref = reference_rmsnorm(x_cpu, w_cpu, 1e-6)

    try:
        with torch.no_grad():
            out = layer(x_cpu.to(dev))
        xm.mark_step()
        out_cpu = out.cpu()
        print(f"  NKI calls      = {calls['nki']}")
        print(f"  fallback calls = {calls['fallback']}")
        cs = cos_sim(ref, out_cpu)
        md = (ref.float() - out_cpu.float()).abs().max().item()
        print(f"  cos_sim = {cs:.6f}   max_abs_diff = {md:.3e}")
        print(f"  {'PASS' if cs > 0.999 else 'FAIL'} (target cos_sim > 0.999)")
    except Exception as e:
        import traceback

        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


def main():
    print(SEP)
    print("PROBE: is the NKI kernel actually executing?")
    print(SEP)
    probe_1_which_branch()
    probe_2_nki_on_cpu()
    probe_3_nki_on_xla()
    probe_4_layer_on_xla()
    print()
    print(SEP)
    print("PROBE COMPLETE")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
