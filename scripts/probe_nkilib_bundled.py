"""Probe: is `nkilib` already importable from the bundled neuronx-cc?

nki-library's README states that the Neuron compiler ships a bundled copy of the
package under the `nkilib` namespace, and that `pip install nki-library` overrides it.

If true, it changes the porting recommendation substantially. Every port so far
(RMSNorm, RoPE) assumed we must vendor or reimplement kernel source, because HF kernels
must be self-contained. But if `nkilib` is *already present* wherever neuronx-cc is, a
kernel could import it directly and the wrapper becomes a few dozen lines.

That would matter most for the MLP kernel, whose dependency closure is ~7,250 lines
across 22 files — roughly 480x RoPE's ~15 lines. Vendoring that is not viable; importing
it is trivial.

Two things to establish:
  1. Does `import nkilib` work in the Neuron venv, and from where (bundled vs pip)?
  2. Does the bundled copy's `mlp()` signature match nki-library `main`? The README warns
     that `main` is not guaranteed compatible with a given compiler version, and several
     parameters look recent (`gate_up_w_layout`, `dtype_mode`, `transposed_in/out`).

Note this does NOT resolve whether HF would *allow* the dependency — `python-depends`
whitelists `nki` but not `nkilib`, and the neuron table is unreachable anyway
(Finding #12). It only establishes whether the code is physically available.

Run on trn2:
    python scripts/probe_nkilib_bundled.py
"""

import importlib
import inspect
import sys

SEP = "=" * 72


def hdr(t):
    print()
    print(SEP)
    print(t)
    print(SEP)


def probe_import():
    hdr("1. Is `nkilib` importable, and from where?")
    try:
        import nkilib

        print(f"  import nkilib      -> OK")
        print(f"  __file__           = {getattr(nkilib, '__file__', '?')}")
        print(f"  __version__        = {getattr(nkilib, '__version__', 'n/a')}")
        path = str(getattr(nkilib, "__file__", ""))
        if "neuronxcc" in path:
            print("  => BUNDLED inside neuronx-cc (no pip install needed)")
        else:
            print("  => resolved outside neuronxcc; possibly a pip install")
        return nkilib
    except Exception as e:
        print(f"  import nkilib      -> FAILED: {type(e).__name__}: {e}")
        print("  => the bundled-copy claim does not hold in this venv")
        return None


def probe_mlp_entry():
    hdr("2. Is the MLP kernel entry point importable, and what is its signature?")
    try:
        mod = importlib.import_module("nkilib.core.mlp.mlp")
        print(f"  module file = {getattr(mod, '__file__', '?')}")
    except Exception as e:
        print(f"  import nkilib.core.mlp.mlp -> FAILED: {type(e).__name__}: {e}")
        return None

    fn = getattr(mod, "mlp", None)
    if fn is None:
        print("  no `mlp` attribute found")
        return None

    print("  found `mlp` entry point")
    target = getattr(fn, "func", getattr(fn, "__wrapped__", fn))
    try:
        sig = inspect.signature(target)
        params = list(sig.parameters)
        print(f"  parameter count = {len(params)}")
        print("  first 8 params  =")
        for p in params[:8]:
            print(f"      {p}")
        # Parameters the sub-agent flagged as possibly newer than the bundled copy
        print()
        print("  presence of recent-looking parameters:")
        for p in ["gate_up_w_layout", "dtype_mode", "transposed_in", "transposed_out",
                  "activation_fn", "normalization_type", "quantization_type", "mode"]:
            print(f"      {'yes' if p in params else 'NO ':4s} {p}")
    except Exception as e:
        print(f"  could not introspect signature: {type(e).__name__}: {e}")
    return fn


def probe_other_kernels():
    hdr("3. Which other nkilib kernels are importable from the bundled copy?")
    targets = [
        ("nkilib.core.embeddings.rope_hf", "rope_hf"),
        ("nkilib.core.mlp.mlp", "mlp"),
        ("nkilib.core.moe.moe_cte.moe_cte", "moe_cte"),
        ("nkilib.core.router_topk.router_topk", None),
        ("nkilib.core.rmsnorm.rmsnorm_quant", None),
    ]
    for mod_name, sym in targets:
        try:
            m = importlib.import_module(mod_name)
            if sym:
                ok = hasattr(m, sym)
                print(f"  {mod_name:44s} -> OK (has {sym}: {ok})")
            else:
                fns = [a for a in dir(m) if not a.startswith("_")][:4]
                print(f"  {mod_name:44s} -> OK (exports e.g. {fns})")
        except Exception as e:
            msg = str(e).replace("\n", " ")[:60]
            print(f"  {mod_name:44s} -> FAIL {type(e).__name__}: {msg}")


def probe_rope_comparison():
    """Would importing nkilib have been simpler than our RoPE port?"""
    hdr("4. Could our RoPE kernel have just imported nkilib instead?")
    try:
        m = importlib.import_module("nkilib.core.embeddings.rope_hf")
    except Exception as e:
        print(f"  nkilib.core.embeddings.rope_hf unavailable: {type(e).__name__}: {e}")
        return
    fn = getattr(m, "rope_hf", None)
    if fn is None:
        print("  no rope_hf symbol")
        return
    target = getattr(fn, "func", getattr(fn, "__wrapped__", fn))
    try:
        print(f"  rope_hf signature: {inspect.signature(target)}")
    except Exception as e:
        print(f"  signature unavailable: {e}")
    print()
    print("  If this is importable, a future RoPE kernel could wrap it directly rather")
    print("  than vendoring ~200 lines. Our port remains the right call for the PoC —")
    print("  it documents the porting path and avoids an undeclarable dependency")
    print("  (Finding #12) — but this is the cheaper long-term shape.")


def main():
    print(SEP)
    print("PROBE: is nkilib bundled with neuronx-cc?")
    print(SEP)
    try:
        import neuronxcc

        print(f"  neuronx-cc version = {getattr(neuronxcc, '__version__', '?')}")
    except Exception as e:
        print(f"  neuronxcc unavailable: {e}")

    probe_import()
    probe_mlp_entry()
    probe_other_kernels()
    probe_rope_comparison()
    print()
    print(SEP)
    print("PROBE COMPLETE")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
