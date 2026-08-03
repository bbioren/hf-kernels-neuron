"""Spike: call the production nkilib MLP kernel directly and check it against Qwen3MLP.

Week 4 prerequisite, recommended in deliverables/week-3.md. The fused gate/up/SiLU/down
MLP is where MLP performance actually is (a standalone elementwise SiLU is memory-bound),
and `nkilib.core.mlp.mlp` implements it with quantization and normalization both opt-in.

Two independent questions, and the second one matters more than it looks:

  Q1. Does the production kernel produce Qwen3MLP's output when driven from PyTorch/XLA
      with HF weights? This derisks the highest-value kernel for ~a day of work, and it
      does not depend on the Finding #17 design decision.

  Q2. Does a non-contiguous transposed VIEW work, or must the transpose be materialized?

      Finding #17 asserts the transpose must be materialized, and the whole severity of
      that finding rests on it — if `.t()` views were acceptable there would be no memory
      doubling and no `save_pretrained` hazard, just a cheap view per forward. That claim
      was reasoned from "non-contiguous tensor failures are a known Neuron beta issue,"
      NOT measured. This measures it.

Weight layouts. HF `nn.Linear.weight` is [out_features, in_features]; the kernel wants the
transpose of each:

    gate_proj  HF [I, H]  ->  kernel [H, I]
    up_proj    HF [I, H]  ->  kernel [H, I]
    down_proj  HF [H, I]  ->  kernel [I, H]

Constraints on the non-quant CTE path: H % 128 == 0, and BxS > 96 to stay off the
decode (TKG) path.

Run on trn2:
    python scripts/spike_nkilib_mlp.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEP = "=" * 76


def hdr(t):
    print()
    print(SEP)
    print(t)
    print(SEP)


def cos_sim(a, b):
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
    ).item()


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def hf_reference(x, w_gate, w_up, w_down):
    """Qwen3MLP.forward: down(silu(gate(x)) * up(x)), computed in fp32 on CPU.

    Weights are in HF orientation: w_gate/w_up are [I, H], w_down is [H, I].
    """
    x = x.float()
    g = F.linear(x, w_gate.float())
    u = F.linear(x, w_up.float())
    return F.linear(F.silu(g) * u, w_down.float())


def to_kernel_layout(w_gate, w_up, w_down, contiguous=True):
    """HF [out, in] -> kernel layout. Optionally leave the transpose as a view."""
    g = w_gate.t()
    u = w_up.t()
    d = w_down.t()
    if contiguous:
        g, u, d = g.contiguous(), u.contiguous(), d.contiguous()
    return g, u, d


def transpose_on_device(w_gate, w_up, w_down, dev):
    """Move HF-oriented weights to the device FIRST, then transpose there.

    This is the case that actually matters, and the one the first version of this spike
    got wrong. Transposing on the host and then calling `.to(device)` lets the transfer
    materialize the result, so the kernel receives a contiguous device tensor and the
    test proves nothing about non-contiguous handling.

    It is also the realistic scenario: in a kernelized model the HF weights already live
    on the device, so a wrapper would transpose there.
    """
    wg = w_gate.to(dev)   # [I, H] contiguous on device
    wu = w_up.to(dev)
    wd = w_down.to(dev)
    return wg.t(), wu.t(), wd.t()   # non-contiguous device views


def run_kernel(mlp, x, g, u, d, **kwargs):
    """Call the kernel and normalize the return (it returns a list)."""
    out = mlp(x, g, u, d, **kwargs)
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def case(mlp, xm, dev, label, H, I, B, S, dtype=torch.float32,
         weight_mode="host_contiguous", quiet=False, **kernel_kwargs):
    """One configuration end to end. Returns (ok, detail).

    weight_mode:
      host_contiguous -> transpose on host, .contiguous(), then move to device
      host_view       -> transpose on host as a view, then move (transfer may materialize)
      device_view     -> move first, then transpose ON DEVICE (a real non-contiguous
                         device tensor) — this is the meaningful test
    """
    torch.manual_seed(0)
    # HF-oriented weights
    w_gate = (torch.randn(I, H) * (H ** -0.5)).to(dtype)
    w_up = (torch.randn(I, H) * (H ** -0.5)).to(dtype)
    w_down = (torch.randn(H, I) * (I ** -0.5)).to(dtype)
    x = torch.randn(B, S, H, dtype=dtype)

    golden = hf_reference(x, w_gate, w_up, w_down)

    if not quiet:
        print(f"  {label}")
        print(f"      H={H} I={I} B={B} S={S} BxS={B*S} "
              f"dtype={str(dtype).split('.')[-1]} weights={weight_mode}"
              + (f" kwargs={list(kernel_kwargs)}" if kernel_kwargs else ""))

    try:
        xd = x.to(dev)
        if weight_mode == "device_view":
            gd, ud, dd = transpose_on_device(w_gate, w_up, w_down, dev)
        else:
            g, u, d = to_kernel_layout(
                w_gate, w_up, w_down, contiguous=(weight_mode == "host_contiguous")
            )
            gd, ud, dd = g.to(dev), u.to(dev), d.to(dev)

        if not quiet:
            print(f"      on-device is_contiguous: gate={gd.is_contiguous()} "
                  f"up={ud.is_contiguous()} down={dd.is_contiguous()}")

        t0 = time.time()
        out = run_kernel(mlp, xd, gd, ud, dd, **kernel_kwargs)
        xm.mark_step()
        out_cpu = out.cpu()
        elapsed = time.time() - t0
    except Exception as e:
        msg = str(e).replace("\n", " ")[:160]
        if not quiet:
            print(f"      FAILED: {type(e).__name__}: {msg}")
        return False, f"{type(e).__name__}: {msg[:70]}"

    if tuple(out_cpu.shape) != tuple(golden.shape):
        if not quiet:
            print(f"      shape mismatch: got {tuple(out_cpu.shape)}, "
                  f"expected {tuple(golden.shape)}")
        return False, "shape mismatch"

    cs = cos_sim(golden, out_cpu)
    md = max_diff(golden, out_cpu)
    ok = cs > 0.999
    if not quiet:
        print(f"      cos_sim={cs:.6f}  max_diff={md:.3e}  wall={elapsed:.1f}s  "
              f"{'PASS' if ok else 'FAIL'}")
    return ok, f"cos_sim={cs:.6f} max_diff={md:.1e}"


def main():
    print(SEP)
    print("SPIKE: production nkilib MLP kernel vs Qwen3MLP")
    print(SEP)

    try:
        from nkilib.core.mlp.mlp import mlp
        import nkilib

        print(f"  nkilib: {nkilib.__file__}")
    except Exception as e:
        print(f"  cannot import nkilib MLP: {type(e).__name__}: {e}")
        return 1

    try:
        import torch_xla.core.xla_model as xm

        dev = xm.xla_device()
        if xm.xla_device_hw(dev) != "NEURON":
            print("  not on Neuron hardware; aborting")
            return 1
        print(f"  device: {dev} ({xm.xla_device_hw(dev)})")
    except Exception as e:
        print(f"  torch_xla unavailable: {e}")
        return 1

    results = {}

    # ------------------------------------------------------------------
    hdr("Q1. Correctness with materialized (contiguous) transposed weights")
    # Qwen3-0.6B-like: H=1024, I=3072. BxS=128 > 96 keeps us on the CTE (prefill) path.
    results["small fp32"] = case(
        mlp, xm, dev, "A. Qwen3-0.6B dims, fp32", H=1024, I=3072, B=1, S=128
    )
    results["small bf16"] = case(
        mlp, xm, dev, "B. Qwen3-0.6B dims, bf16", H=1024, I=3072, B=1, S=128,
        dtype=torch.bfloat16,
    )

    # ------------------------------------------------------------------
    hdr("Q2. Does a non-contiguous transposed view work? (Finding #17's premise)")
    print("  METHODOLOGY NOTE. The first version of this spike transposed on the host and")
    print("  then called .to(device), which lets the transfer materialize the result — so")
    print("  it proved nothing. The meaningful test moves the HF weight to the device")
    print("  first and transposes THERE, producing a genuinely non-contiguous device")
    print("  tensor. That is also the realistic case: in a kernelized model the weights")
    print("  already live on the device.")
    print()
    results["host_view"] = case(
        mlp, xm, dev, "C. transpose on host, then move (may materialize)",
        H=1024, I=3072, B=1, S=128, weight_mode="host_view",
    )
    results["device_view"] = case(
        mlp, xm, dev, "D. move to device, then transpose THERE  <-- the real test",
        H=1024, I=3072, B=1, S=128, weight_mode="device_view",
    )

    # ------------------------------------------------------------------
    hdr("Q3. Qwen3-8B dimensions — where is the boundary?")
    print("  H=4096, I=12288 failed with:")
    print("    'floordiv' does not allow division by zero")
    print("    in kernel_helpers.get_ceil_quotient, called from")
    print("    mlp_cte_tile_info.py:236 build_with_subtiling(bxs_dim, ...)")
    print("  i.e. a subtile size computes to 0 for this shape. Mapping the boundary:")
    print()

    from nkilib.core.utils.common_types import ComputationMode

    grid = [
        # (label, H, I, B, S, kwargs)
        ("H=1024  I=3072   S=128", 1024, 3072, 1, 128, {}),
        ("H=2048  I=6144   S=128", 2048, 6144, 1, 128, {}),
        ("H=4096  I=12288  S=128", 4096, 12288, 1, 128, {}),
        ("H=4096  I=12288  S=256", 4096, 12288, 1, 256, {}),
        ("H=4096  I=12288  S=512", 4096, 12288, 1, 512, {}),
        ("H=4096  I=4096   S=128", 4096, 4096, 1, 128, {}),
        ("H=4096  I=8192   S=128", 4096, 8192, 1, 128, {}),
        ("H=4096  I=12288  S=128 force_cte", 4096, 12288, 1, 128,
         {"force_cte_mode": True}),
        ("H=4096  I=12288  S=128 PREFILL", 4096, 12288, 1, 128,
         {"mode": ComputationMode.PREFILL}),
    ]
    for label, H, I, B, S, kw in grid:
        ok, detail = case(
            mlp, xm, dev, label, H=H, I=I, B=B, S=S,
            dtype=torch.bfloat16, quiet=True, **kw
        )
        mark = "pass" if ok else "FAIL"
        print(f"    {mark}  {label:38s} {detail}")
        results[f"grid:{label}"] = (ok, detail)

    # ------------------------------------------------------------------
    hdr("Q4. Pinning the failure boundary")
    print("  Hypothesis from Q3: failures track intermediate_size > 4096, not the I/H")
    print("  ratio. nki-library's CTE sharding heuristic forces shard_on_inter=True when")
    print("  intermediate_size > 4096; we launch single-core (no SPMD grid), so the")
    print("  inter-sharding tile math would divide by a zero worker count.")
    print()
    print("  If true, the fused MLP cannot run single-core at the sizes that matter")
    print("  (Qwen3-8B is I=12288), which is a harder blocker than the weight layout.")
    print()
    boundary = [
        (4096, 4096), (4096, 4224), (4096, 4608), (4096, 5120), (4096, 8192),
        (1024, 3072), (1024, 4096), (1024, 5120),
        (2048, 4096), (2048, 6144),
    ]
    for H, I in boundary:
        ok, detail = case(
            mlp, xm, dev, f"H={H} I={I}", H=H, I=I, B=1, S=128,
            dtype=torch.bfloat16, quiet=True,
        )
        mark = "pass" if ok else "FAIL"
        gate = "I<=4096" if I <= 4096 else "I >4096"
        print(f"    {mark}  H={H:5d} I={I:6d}  ({gate})  "
              f"{detail if ok else 'compile error'}")
        results[f"bound:{H}:{I}"] = (ok, detail)

    bound_items = [(k, v) for k, v in results.items() if k.startswith("bound:")]
    consistent = all(
        ok == (int(k.split(":")[2]) <= 4096) for k, (ok, _) in bound_items
    )
    print()
    print(f"  Hypothesis 'passes iff intermediate_size <= 4096': "
          f"{'CONSISTENT with all data points' if consistent else 'CONTRADICTED'}")

    # ------------------------------------------------------------------
    hdr("VERDICT")

    q1 = results["small fp32"][0] and results["small bf16"][0]
    host_view_ok, host_view_detail = results["host_view"]
    dev_view_ok, dev_view_detail = results["device_view"]

    print("  Q1 — production kernel drivable from PyTorch/XLA with HF weights:")
    print(f"       {'YES' if q1 else 'NO'}")
    for k in ("small fp32", "small bf16"):
        ok, detail = results[k]
        print(f"         {k:12s} {'pass' if ok else 'FAIL'}  {detail}")
    if q1:
        print("       => The fused MLP kernel itself works, called directly with HF")
        print("          weights. No vendoring, no reimplementation.")

    print()
    print("  Q2 — non-contiguous transposed device view accepted:")
    print(f"       transpose-on-host-then-move : {'yes' if host_view_ok else 'NO'}"
          f"  ({host_view_detail})")
    print(f"       transpose-on-device         : {'yes' if dev_view_ok else 'NO'}"
          f"  ({dev_view_detail})")
    if dev_view_ok:
        print("       => Finding #17 needs REVISING. A device-side `.t()` view is accepted,")
        print("          so the memory-doubling and checkpoint-corruption options may both")
        print("          be avoidable. Confirm the perf cost before relaxing the finding —")
        print("          'accepted' is not the same as 'free'.")
    else:
        print("       => Finding #17 CONFIRMED by measurement: the transpose must be")
        print("          materialized, so the weight-lifecycle question is real.")

    print()
    n_grid_pass = sum(1 for k, (ok, _) in results.items() if k.startswith("grid:") and ok)
    n_grid = sum(1 for k in results if k.startswith("grid:"))
    print(f"  Q3 — shape support: {n_grid_pass}/{n_grid} configurations compiled")
    print("       See the table above for which. A divide-by-zero in the kernel's own")
    print("       tile-size computation is an nki-library bug to report, not something")
    print("       a wrapper can work around.")

    print(SEP)
    return 0 if q1 else 1


if __name__ == "__main__":
    sys.exit(main())
