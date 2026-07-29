"""Probe: which NKI API surface is actually available on this DLAMI?

Week 3, T4 prerequisite. nki-library source imports the top-level `nki` package:

    import nki
    import nki.isa as nisa
    import nki.language as nl

but our Week 2 RMSNorm kernel imports the compiler-bundled path:

    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

If only one of those resolves here, then nki-library source cannot be copied
verbatim — every ported kernel needs its imports rewritten. That is a concrete,
mechanical porting cost worth measuring precisely.

Also checks the specific primitives `rope_hf` depends on.

Run on trn2:
    python scripts/probe_nki_api.py
"""

import importlib
import sys

SEP = "=" * 68


def probe_import_paths():
    print(SEP)
    print("1. Which NKI import paths resolve?")
    print(SEP)
    results = {}
    for name in ["nki", "nki.language", "nki.isa",
                 "neuronxcc.nki", "neuronxcc.nki.language", "neuronxcc.nki.isa"]:
        try:
            m = importlib.import_module(name)
            path = getattr(m, "__file__", "?")
            print(f"  OK    {name:28s} {path}")
            results[name] = m
        except Exception as e:
            print(f"  FAIL  {name:28s} {type(e).__name__}: {e}")
            results[name] = None
    return results


def probe_primitives():
    print()
    print(SEP)
    print("2. Are the primitives rope_hf needs available?")
    print(SEP)
    try:
        import neuronxcc.nki.isa as nisa
        import neuronxcc.nki.language as nl
    except Exception as e:
        print(f"  cannot import neuronxcc.nki: {e}")
        return

    print(f"  nl.tile_size.pmax = {nl.tile_size.pmax}")
    print()
    print("  nki.isa:")
    for a in ["tensor_tensor", "dma_copy", "tensor_scalar", "tensor_copy",
              "memset", "activation"]:
        print(f"    {'yes' if hasattr(nisa, a) else 'NO ':4s} nisa.{a}")
    print()
    print("  nki.language:")
    for a in ["ndarray", "sbuf", "shared_hbm", "psum", "multiply", "subtract",
              "add", "negative", "mgrid", "affine_range", "arange", "load",
              "store", "program_id", "num_programs", "program_ndim"]:
        print(f"    {'yes' if hasattr(nl, a) else 'NO ':4s} nl.{a}")


def probe_tensor_methods():
    """rope_hf uses NkiTensor.reshape_dim() and .permute() on HBM views."""
    print()
    print(SEP)
    print("3. NkiTensor methods used by rope_hf (reshape_dim / permute)")
    print(SEP)
    try:
        import neuronxcc.nki.language as nl
    except Exception as e:
        print(f"  cannot import: {e}")
        return

    tensor_cls = getattr(nl, "NkiTensor", None)
    if tensor_cls is None:
        print("  nl.NkiTensor not exposed; checking via a traced kernel instead")
    else:
        for a in ["reshape_dim", "permute", "slice", "expand_dim", "broadcast",
                  "reshape", "broadcast_to"]:
            print(f"    {'yes' if hasattr(tensor_cls, a) else 'NO ':4s} NkiTensor.{a}")


def probe_multi_output_jit():
    """Can an @nki.jit kernel allocate and return TWO shared_hbm outputs?

    rope_hf uses destination-passing (q_out/k_out passed in). HF's
    apply_rotary_pos_emb returns a tuple. If nki.jit can return two outputs we
    can allocate internally and avoid the destination-passing mismatch entirely.
    """
    print()
    print(SEP)
    print("4. Can @nki.jit return two allocated outputs?")
    print(SEP)
    try:
        import torch
        import torch_xla.core.xla_model as xm
        import neuronxcc.nki as nki
        import neuronxcc.nki.language as nl
    except Exception as e:
        print(f"  cannot set up: {e}")
        return

    @nki.jit
    def _two_out(a):
        o1 = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
        o2 = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
        t = nl.load(a)
        nl.store(o1, value=nl.multiply(t, 2.0))
        nl.store(o2, value=nl.add(t, 1.0))
        return o1, o2

    dev = xm.xla_device()
    x = torch.ones(4, 8).to(dev)
    try:
        r1, r2 = _two_out(x)
        xm.mark_step()
        a, b = r1.cpu(), r2.cpu()
        ok = torch.allclose(a, torch.full((4, 8), 2.0)) and torch.allclose(
            b, torch.full((4, 8), 2.0)
        )
        print(f"  returned two tensors: shapes {tuple(a.shape)}, {tuple(b.shape)}")
        print(f"  o1[0,0]={a[0,0].item()} (expect 2.0), o2[0,0]={b[0,0].item()} (expect 2.0)")
        print(f"  => multi-output @nki.jit WORKS ({'values correct' if ok else 'values unexpected'})")
    except Exception as e:
        import traceback

        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("  => must use destination-passing like nki-library does")


def probe_activations():
    """Which activation primitives exist? Needed for the SiLU kernel."""
    print()
    print(SEP)
    print("5. Activation primitives (for SiLU / GELU kernels)")
    print(SEP)
    try:
        import neuronxcc.nki.isa as nisa
        import neuronxcc.nki.language as nl
    except Exception as e:
        print(f"  cannot import: {e}")
        return

    cands = ["silu", "swish", "sigmoid", "gelu", "gelu_tanh", "gelu_apprx_tanh",
             "exp", "tanh", "relu", "sqrt", "rsqrt", "multiply", "divide",
             "add", "subtract", "negative", "reciprocal"]
    present = [c for c in cands if hasattr(nl, c)]
    missing = [c for c in cands if not hasattr(nl, c)]
    print(f"  nl has     : {present}")
    print(f"  nl missing : {missing}")

    act_like = sorted(
        a for a in dir(nl)
        if any(t in a.lower() for t in ("silu", "sigmoid", "gelu", "swish", "erf"))
    )
    print(f"  activation-ish names in nl: {act_like}")

    if hasattr(nisa, "activation"):
        import inspect

        try:
            print(f"  nisa.activation signature: {inspect.signature(nisa.activation)}")
        except Exception as e:
            print(f"  nisa.activation signature unavailable: {e}")


def main():
    probe_import_paths()
    probe_primitives()
    probe_tensor_methods()
    probe_multi_output_jit()
    probe_activations()
    print()
    print(SEP)
    print("PROBE COMPLETE")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
