"""Does nkilib's FUSED RMSNorm+MLP beat the compiler? Samir's suggestion, tested.

BACKGROUND
Samir Araujo, after reading the Week 3-6 findings: "there was a fused rmsnorm with mlp that
doesn't have quantisation you can use. The only thing you need to use the layer fusion feature
of HF Kernels for that."

He is right that it exists, and this project had missed it. `nkilib.core.mlp.mlp` takes

    normalization_type: NormType = NormType.NO_NORM        <- we always used the default
    quantization_type: QuantizationType = QuantizationType.NONE

and `NormType.RMS_NORM = 1`. So a non-quantising RMSNorm+MLP is one keyword argument away.
Finding #26 recorded "nkilib has no standalone RMSNorm, it always fuses quantisation", which is
true of `core/rmsnorm/rmsnorm_quant.py` and led to the wrong general conclusion that no
non-quantising RMSNorm path existed anywhere in the library.

WHY THIS IS THE BEST REMAINING CANDIDATE
Finding #25 gave the criterion a kernel must satisfy to win: it has to replace a region the
compiler would not otherwise fuse well, AND contain real arithmetic to restructure. Scoring the
candidates against it:

    RMSNorm / RoPE / SiLU alone   small, memory-bound, already fused -> the swap COSTS a fusion
    fused MLP (NO_NORM)          spans a fusable region, 2 real matmuls, but lost 2.99x
                                 single-core for lack of an SPMD grid (Finding #26)
    fused RMSNorm+MLP (this)     same region PLUS the normalisation

That last row matters for a specific reason. This kernel absorbs one of the very interception
points Finding #25 says loses money — the `post_attention_layernorm` RMSNorm — into the region
that has real compute. So it is simultaneously a bigger fused span (6 torch ops replaced instead
of 4: norm, gate, up, silu, mul, down) and one fewer optimisation barrier in the graph.

WHAT IS MEASURED
  1. Does it compile and is it correct, against a CPU fp32 reference of
     `Qwen3MLP(Qwen3RMSNorm(x))`. A fast wrong answer is not a result.
  2. Wall-clock, N chained blocks, distinct weights per block, both implementations.
  3. Whether Finding #18's `intermediate_size > 4096` single-core compile boundary still holds
     on the NEW compiler. That boundary was measured on neuronx-cc 2.26.6360.0; native ships
     2.0.266551.0a0, and nothing says the heuristic is unchanged.

HONEST LIMIT ON THE TIMING, stated up front because it decides how much the ratio is worth.
This is WALL CLOCK, so it includes dispatch. On torch-xla we could separate the two with
`neuron-explorer` on a NEFF+NTFF pair, and the device-only comparison is what Findings #25-27
rest on. That profiling is not wired up for the native stack, and NKI's per-call dispatch cost
on native is uncharacterised (Finding #28 is a torch_xla fix and does not port). So a ratio here
bounds the answer rather than settling it: the NKI side pays one dispatch per block, the torch
side pays six, which cuts in NKI's favour, while any native equivalent of the #28 lowering cost
cuts against it. Treat as provisional and label it as such anywhere it is quoted.

    ./scripts/run_native.sh scripts/profile_fused_rmsnorm_mlp.py
    ./scripts/run_native.sh scripts/profile_fused_rmsnorm_mlp.py --boundary-sweep
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import faulthandler

import torch
import torch.nn.functional as F

from nki_test_utils import require_neuron, stack, sync

SEP = "=" * 84
HANG_TIMEOUT_S = 2400
EPS = 1e-6


def cos_sim(a, b):
    return F.cosine_similarity(
        a.detach().flatten().float(), b.detach().flatten().float(), dim=0
    ).item()


def torch_rmsnorm(x, w, eps=EPS):
    """Reference RMSNorm, identical to Qwen3RMSNorm.forward."""
    dt = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    return (w * (x * torch.rsqrt(var + eps)).to(dt))


def cpu_reference(x, norm_w, w_gate, w_up, w_down):
    """Qwen3MLP(Qwen3RMSNorm(x)) in fp32 on CPU — the thing the fused kernel must reproduce."""
    h = torch_rmsnorm(x.float(), norm_w.float())
    return F.linear(
        F.silu(F.linear(h, w_gate.float())) * F.linear(h, w_up.float()), w_down.float()
    )


def make_weights(H, I, dtype):
    torch.manual_seed(0)
    scale = 0.02
    return (
        torch.randn(H, dtype=dtype) * 0 + 1.0,             # norm gamma, ones-ish
        torch.randn(I, H, dtype=dtype) * scale,            # gate [I, H]  (HF nn.Linear layout)
        torch.randn(I, H, dtype=dtype) * scale,            # up   [I, H]
        torch.randn(H, I, dtype=dtype) * scale,            # down [H, I]
    )


def time_blocks(fn, x, n_blocks, iters, warmup):
    """Median wall time for n_blocks chained applications. Output consumed to defeat DCE."""
    def one_pass():
        t = x
        for i in range(n_blocks):
            t = fn(t, i)
        return t

    for _ in range(warmup):
        out = one_pass()
        float(out.float().sum().item())
        sync()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = one_pass()
        float(out.float().sum().item())
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return statistics.median(samples), samples[0], samples[-1]


def run_shape(dev, H, I, S, B, n_blocks, iters, warmup, result):
    """Correctness + timing at one (H, I) shape. Returns True if the shape ran at all."""
    from nkilib.core.mlp.mlp import mlp as nki_mlp
    from nkilib.core.utils.common_types import NormType, QuantizationType

    dtype = torch.bfloat16
    print(f"\n{SEP}\nH={H} I={I} S={S} B={B} blocks={n_blocks} bf16\n{SEP}")

    norm_w, w_gate, w_up, w_down = make_weights(H, I, dtype)
    # 3D [batch, seq, hidden]. The RMS_NORM path is NOT interface-compatible with the NO_NORM
    # path we measured in Finding #26: it routes through load_hidden_tensor_tile, which computes
    #   indices.batch_idx * view.shape[1] * view.shape[2] + bxs_offset * view.shape[2]
    # (mlp_cte_tensor_io.py:78-81), so a 2D [B*S, H] hidden tensor fails with
    #   error: index 2 out of bounds for sequence of size 2
    # Our earlier fused-MLP work passed 2D and worked. So enabling normalisation changes the
    # required input rank — a second interface mismatch on this one keyword argument.
    x_cpu = torch.randn(B, S, H, dtype=dtype) * 0.5
    ref = cpu_reference(x_cpu, norm_w, w_gate, w_up, w_down)
    x = x_cpu.to(dev)

    entry = {"H": H, "I": I, "S": S, "B": B, "blocks": n_blocks}

    # Distinct weights per block. Finding #26's harness error was reusing ONE weight set across
    # all blocks, which let the compiler amortise a single weight load over the whole chain and
    # handed torch an advantage a real model never gives it.
    sets = []
    for i in range(n_blocks):
        d = 0.0 if i == 0 else (i + 1) * 1e-3
        sets.append((norm_w + d, w_gate + d, w_up + d, w_down + d))

    # ---- NKI fused RMSNorm+MLP ----------------------------------------------------------
    # Transpose on device: that is the realistic path, since in a kernelized model the HF
    # weights already live there. Finding #17 quantified this transpose as a one-time cost.
    # normalization_weights_tensor must be 2D. Passing HF's native 1D [H] gamma fails with
    #   [NCC_INKI016] Unexpected HBM tensor shape of (1024,). Expected a vector with
    #   shape [1, X] or [X, 1]
    # HF's RMSNorm weight is nn.Parameter(torch.ones(hidden_size)), i.e. 1D, so a wrapper has to
    # reshape. Small, but it is one more interface mismatch on top of the three transposes —
    # the same class of friction as Finding #17, and worth counting when estimating a port.
    nki_w = [
        (nw.to(dev).reshape(1, H), g.to(dev).t(), u.to(dev).t(), dn.to(dev).t())
        for nw, g, u, dn in sets
    ]

    def nki_block(t, i):
        nw, g, u, dn = nki_w[i % len(nki_w)]
        out = nki_mlp(
            t, g, u, dn,
            normalization_weights_tensor=nw,
            normalization_type=NormType.RMS_NORM,
            quantization_type=QuantizationType.NONE,
            eps=EPS,
        )
        return out[0] if isinstance(out, (list, tuple)) else out

    print("  compiling NKI fused RMSNorm+MLP ...", flush=True)
    try:
        t0 = time.time()
        single = nki_block(x, 0)
        sync()
        got = single.cpu()
        print(f"    compiled+ran in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        entry["nki_error"] = f"{type(e).__name__}: {e}"
        short = str(e).replace("\n", " ")[:300]
        print(f"    FAILED: {type(e).__name__}: {short}")
        result["shapes"].append(entry)
        return False

    sim = cos_sim(got, ref)
    entry["nki_cos_sim"] = sim
    print(f"    correctness vs CPU fp32 (1 block): cos_sim = {sim:.6f}")
    if sim < 0.999:
        entry["verdict"] = "FAILED ACCURACY"
        print("    FAILED the accuracy gate — refusing to report timings for a wrong kernel.")
        result["shapes"].append(entry)
        return False
    del single, got

    # ---- torch reference: RMSNorm + MLP as separate ops --------------------------------
    torch_w = [(nw.to(dev), g.to(dev), u.to(dev), dn.to(dev)) for nw, g, u, dn in sets]

    def torch_block(t, i):
        nw, g, u, dn = torch_w[i % len(torch_w)]
        h = torch_rmsnorm(t, nw)
        return F.linear(F.silu(F.linear(h, g)) * F.linear(h, u), dn)

    print("  compiling torch RMSNorm + MLP ...", flush=True)
    t0 = time.time()
    tsingle = torch_block(x, 0)
    sync()
    tgot = tsingle.cpu()
    print(f"    compiled+ran in {time.time() - t0:.1f}s", flush=True)
    tsim = cos_sim(tgot, ref)
    entry["torch_cos_sim"] = tsim
    print(f"    torch path cos_sim = {tsim:.6f}  (sanity check on the reference itself)")
    del tsingle, tgot

    # ---- timing -------------------------------------------------------------------------
    print(f"  timing {n_blocks} chained blocks, {iters} iters ...", flush=True)
    n_med, n_lo, n_hi = time_blocks(nki_block, x, n_blocks, iters, warmup)
    t_med, t_lo, t_hi = time_blocks(torch_block, x, n_blocks, iters, warmup)

    entry["nki_ms"] = n_med * 1e3
    entry["torch_ms"] = t_med * 1e3
    entry["nki_ms_per_block"] = n_med * 1e3 / n_blocks
    entry["torch_ms_per_block"] = t_med * 1e3 / n_blocks
    entry["ratio_nki_over_torch"] = n_med / t_med

    print(f"    NKI fused   {n_med*1e3:9.3f} ms  ({n_med*1e3/n_blocks:7.4f} ms/block)  "
          f"[{n_lo*1e3:.2f}-{n_hi*1e3:.2f}]")
    print(f"    torch       {t_med*1e3:9.3f} ms  ({t_med*1e3/n_blocks:7.4f} ms/block)  "
          f"[{t_lo*1e3:.2f}-{t_hi*1e3:.2f}]")
    r = n_med / t_med
    if r < 1.0:
        entry["verdict"] = f"NKI FASTER by {1/r:.2f}x"
        print(f"    => NKI is FASTER by {1/r:.2f}x  (wall clock, includes dispatch)")
    else:
        entry["verdict"] = f"NKI slower by {r:.2f}x"
        print(f"    => NKI is slower by {r:.2f}x  (wall clock, includes dispatch)")

    result["shapes"].append(entry)
    return True


def boundary_sweep(dev, result):
    """Does Finding #18's single-core I>4096 compile boundary still hold on this compiler?

    Measured originally on neuronx-cc 2.26.6360.0. Native ships 2.0.266551.0a0 and nothing
    says the CTE sharding heuristic is unchanged, so it is worth one cheap re-test rather than
    inheriting the constraint. Only compile+run is checked here, not accuracy or speed.
    """
    from nkilib.core.mlp.mlp import mlp as nki_mlp
    from nkilib.core.utils.common_types import NormType, QuantizationType

    print(f"\n{SEP}\nFinding #18 boundary re-test on the native compiler\n{SEP}")
    print("  original result (neuronx-cc 2.26.6360.0): passes iff intermediate_size <= 4096")

    cases = [(1024, 3072), (1024, 4096), (1024, 5120), (4096, 4096), (4096, 12288)]
    out = []
    for H, I in cases:
        dtype = torch.bfloat16
        norm_w, g, u, dn = make_weights(H, I, dtype)
        x = (torch.randn(1, 128, H, dtype=dtype) * 0.5).to(dev)  # 3D, see run_shape
        try:
            r = nki_mlp(
                x, g.to(dev).t(), u.to(dev).t(), dn.to(dev).t(),
                normalization_weights_tensor=norm_w.to(dev).reshape(1, H),
                normalization_type=NormType.RMS_NORM,
                quantization_type=QuantizationType.NONE,
                eps=EPS,
            )
            v = r[0] if isinstance(r, (list, tuple)) else r
            sync()
            _ = v.cpu()
            print(f"    H={H:5d} I={I:6d}  PASS")
            out.append({"H": H, "I": I, "ok": True})
        except Exception as e:
            short = str(e).replace("\n", " ")[:140]
            print(f"    H={H:5d} I={I:6d}  FAIL  {type(e).__name__}: {short}")
            out.append({"H": H, "I": I, "ok": False, "error": f"{type(e).__name__}: {short}"})
    result["boundary_sweep"] = out

    passing = [c for c in out if c["ok"]]
    over_4096 = [c for c in passing if c["I"] > 4096]
    if over_4096:
        print("\n  CHANGED: at least one I>4096 shape now compiles single-core on this")
        print("  compiler. Finding #18's boundary is compiler-version-specific.")
    else:
        print("\n  UNCHANGED: the I<=4096 single-core boundary still holds. Finding #18 stands")
        print("  on the native compiler too, which supports #26's reading of it as a design")
        print("  boundary (no SPMD grid) rather than a fixable arithmetic bug.")
    result["boundary_changed"] = bool(over_4096)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--boundary-sweep", action="store_true",
                    help="also re-test Finding #18's I>4096 single-core compile boundary")
    ap.add_argument("--json-out", default="results/raw/native/fused_rmsnorm_mlp.json")
    args = ap.parse_args()

    dev = require_neuron()
    result = {"stack": stack(), "torch": torch.__version__, "shapes": []}
    try:
        import neuronxcc

        result["neuronx_cc"] = getattr(neuronxcc, "__version__", "unknown")
    except ImportError:
        pass

    print(SEP)
    print("Fused RMSNorm+MLP (nkilib, NormType.RMS_NORM) vs torch")
    print(SEP)
    print(f"  stack: {result['stack']}   torch {torch.__version__}")
    print("  This is the candidate Finding #25's criterion favours most: it spans a fusable")
    print("  region, contains two real matmuls, AND absorbs an RMSNorm that would otherwise")
    print("  be a separate optimisation barrier. 6 torch ops replaced by 1 kernel call.")
    print("  TIMING IS WALL CLOCK and includes dispatch — see the module docstring.")

    # Qwen3-0.6B MLP shape: every MFU number in this project is on this model, and Finding #18's
    # own data has it passing single-core.
    ran = run_shape(dev, 1024, 3072, args.seq, args.batch,
                    args.blocks, args.iters, args.warmup, result)

    # Largest shape Finding #18 found passing single-core.
    run_shape(dev, 4096, 4096, args.seq, args.batch,
              max(2, args.blocks // 4), args.iters, args.warmup, result)

    if args.boundary_sweep:
        boundary_sweep(dev, result)

    print(f"\n{SEP}\nSUMMARY\n{SEP}")
    for e in result["shapes"]:
        tag = e.get("verdict", e.get("nki_error", "?"))
        print(f"  H={e['H']:5d} I={e['I']:6d}  {tag}")
    wins = [e for e in result["shapes"] if "FASTER" in str(e.get("verdict", ""))]
    result["any_win"] = bool(wins)
    print()
    if wins:
        print("  A fused kernel beat the compiler here. This is the shape Finding #25 predicted")
        print("  would work, and it absorbs an interception point that loses money on its own.")
    elif ran:
        print("  No win at these shapes. Consistent with Finding #26: single-core, these kernels")
        print("  tile for an execution model they were not built for. The fusion span is bigger")
        print("  than the plain MLP's, and it is still not enough.")
    else:
        print("  The kernel did not run. See the error above.")
    print(SEP)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    faulthandler.dump_traceback_later(HANG_TIMEOUT_S, exit=True)
    try:
        sys.exit(main())
    finally:
        faulthandler.cancel_dump_traceback_later()
