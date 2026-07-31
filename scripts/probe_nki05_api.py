"""Probe: what does NKI 0.5.0 (top-level `nki`) offer for tiling and reductions?

Prerequisite for migrating RMSNorm and SiLU off `nl.arange` (removed in 0.5.0) onto
`nl.ds` slicing. Finding #14's correction established that top-level `nki` is 0.5.0 and the
going-forward surface; those two kernels are still written against the older bundled API.

Questions:
  1. Which ops we depend on exist in 0.5.0?
  2. Does `nl.load` still accept `mask=`? (that's how the old kernels handled ragged tails)
  3. Does `nl.ds(start, size)` tile correctly, including a ragged tail?
  4. Does broadcasting a [1, H] weight across a partial tile work?

TWO STRUCTURAL CONSTRAINTS LEARNED WRITING THIS PROBE — both cost a round trip, and both
produce errors that point at the wrong thing:

  * **Kernels must be defined at module level, and their `nl` / `nisa` imports must be
    module globals.** Defining an `@nki.jit` function inside another function makes `nl` a
    closure variable, and the tracer resolves names from module globals only. The error is
    `failed to resolve name 'nl.ndarray'` — which looks like a missing API, not a scoping
    problem. This is the same error text `nl.arange` produces when it genuinely is missing,
    so the two causes are indistinguishable from the message alone.
  * **No inner function definitions inside a kernel.** `NKI does not support inner function
    definitions`. Tile bodies must be inlined or factored into module-level helpers.

Run on trn2:
    python scripts/probe_nki05_api.py
"""

import inspect
import sys

# Module-level imports: required, see the note above.
import nki
import nki.language as nl

SEP = "=" * 76


@nki.jit
def _tiled_scale(a_tensor, g_tensor):
    """out[r, :] = a[r, :] * g[:] — tiled over rows with a ragged tail.

    Exercises exactly the pattern the migrated kernels need: full 128-row tiles via
    `nl.ds`, a static-size partial tile for the remainder (since `nl.load` no longer takes
    a `mask`), and a [1, H] weight broadcast across both.
    """
    out = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    num_rows, num_cols = a_tensor.shape

    g_tile = nl.load(g_tensor.reshape((1, num_cols)))

    num_full = num_rows // 128
    rem = num_rows % 128

    # Use the module function nl.broadcast_to, not the tensor method: in 0.5.0
    # `g_tile.broadcast_to(...)` fails with `failed to resolve name`.
    for i in nl.affine_range(num_full):
        tile = nl.load(a_tensor[nl.ds(i * 128, 128), :])
        g_b = nl.broadcast_to(g_tile, (128, num_cols))
        nl.store(out[nl.ds(i * 128, 128), :], value=nl.multiply(tile, g_b))

    if rem > 0:
        tail = nl.load(a_tensor[nl.ds(num_full * 128, rem), :])
        g_t = nl.broadcast_to(g_tile, (rem, num_cols))
        nl.store(out[nl.ds(num_full * 128, rem), :], value=nl.multiply(tail, g_t))

    return out


@nki.jit
def _tiled_rowmean(a_tensor):
    """out[r, 0] = mean(a[r, :]) — checks reductions survive nl.ds tiling."""
    num_rows, num_cols = a_tensor.shape
    out = nl.ndarray((num_rows, 1), dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    num_full = num_rows // 128
    rem = num_rows % 128

    # `nl.sum(...) / num_cols` is rejected in 0.5.0 ("'div' expected (int, int) or
    # (float, float)") — a tile cannot be divided by a Python scalar with `/`.
    # nl.mean is both available and clearer.
    for i in nl.affine_range(num_full):
        tile = nl.load(a_tensor[nl.ds(i * 128, 128), :])
        nl.store(out[nl.ds(i * 128, 128), :],
                 value=nl.mean(tile, axis=[1], keepdims=True))

    if rem > 0:
        tail = nl.load(a_tensor[nl.ds(num_full * 128, rem), :])
        nl.store(out[nl.ds(num_full * 128, rem), :],
                 value=nl.mean(tail, axis=[1], keepdims=True))

    return out


def report_surface():
    print(SEP)
    print(f"NKI {getattr(nki, '__version__', '?')} — API surface for RMSNorm / SiLU migration")
    print(SEP)

    groups = {
        "tiling / indexing": ["ds", "arange", "mgrid", "affine_range", "sequential_range",
                              "static_range"],
        "memory": ["load", "store", "ndarray", "zeros", "full", "shared_hbm", "sbuf", "psum"],
        "math": ["square", "sum", "mean", "rsqrt", "sqrt", "add", "subtract", "multiply",
                 "divide", "silu", "sigmoid", "negative"],
        "shape": ["broadcast_to", "expand_dims", "reshape", "transpose"],
    }
    for group, names in groups.items():
        print()
        print(f"  {group}:")
        for n in names:
            print(f"    {'yes' if hasattr(nl, n) else 'NO ':4s} nl.{n}")

    print()
    print("  signatures:")
    for name in ["ds", "load", "store", "sum", "broadcast_to"]:
        fn = getattr(nl, name, None)
        if fn is None:
            print(f"    nl.{name}: absent")
            continue
        try:
            print(f"    nl.{name}{inspect.signature(fn)}")
        except Exception:
            print(f"    nl.{name}: signature unavailable")

    try:
        params = list(inspect.signature(nl.load).parameters)
        print()
        print(f"  nl.load params = {params}")
        print(f"  => mask supported: {'mask' in params}")
        if "mask" not in params:
            print("     Ragged tails must use a separate static-size nl.ds tile.")
    except Exception:
        pass


def main():
    report_surface()

    print()
    print(SEP)
    print("Functional test: nl.ds tiling with a ragged tail")
    print(SEP)
    try:
        import torch
        import torch_xla.core.xla_model as xm
    except Exception as e:
        print(f"  torch_xla unavailable: {e}")
        return 0

    dev = xm.xla_device()
    if xm.xla_device_hw(dev) != "NEURON":
        print("  not on Neuron hardware; skipping")
        return 0

    all_ok = True
    print("  scale + broadcast:")
    for rows, cols in [(128, 64), (300, 64), (250, 128), (1, 32)]:
        torch.manual_seed(0)
        a = torch.randn(rows, cols)
        g = torch.randn(cols)
        want = a * g
        try:
            got = _tiled_scale(a.to(dev), g.to(dev))
            xm.mark_step()
            err = (want - got.cpu()).abs().max().item()
            ok = err < 1e-4
            all_ok &= ok
            print(f"    rows={rows:4d} cols={cols:4d}  {'PASS' if ok else 'FAIL'}  "
                  f"max_diff={err:.2e}   (full tiles={rows // 128}, tail={rows % 128})")
        except Exception as e:
            all_ok = False
            msg = str(e).replace("\n", " ")[:140]
            print(f"    rows={rows:4d} cols={cols:4d}  FAILED: {type(e).__name__}: {msg}")

    print("  row-mean reduction:")
    for rows, cols in [(128, 64), (300, 64)]:
        torch.manual_seed(0)
        a = torch.randn(rows, cols)
        want = a.mean(-1, keepdim=True)
        try:
            got = _tiled_rowmean(a.to(dev))
            xm.mark_step()
            err = (want - got.cpu()).abs().max().item()
            ok = err < 1e-4
            all_ok &= ok
            print(f"    rows={rows:4d} cols={cols:4d}  {'PASS' if ok else 'FAIL'}  "
                  f"max_diff={err:.2e}")
        except Exception as e:
            all_ok = False
            msg = str(e).replace("\n", " ")[:140]
            print(f"    rows={rows:4d} cols={cols:4d}  FAILED: {type(e).__name__}: {msg}")

    print()
    print(SEP)
    if all_ok:
        print("nl.ds tiling works, including ragged tails and reductions.")
        print("=> Migration of RMSNorm and SiLU onto NKI 0.5.0 is viable.")
    else:
        print("nl.ds tiling has gaps — see failures above before migrating.")
    print(SEP)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
