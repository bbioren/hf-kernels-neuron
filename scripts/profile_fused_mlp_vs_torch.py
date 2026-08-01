"""Does the FUSED nki-library MLP beat torch's MLP? The one case where a speedup is plausible.

WHY THIS EXPERIMENT SHOULD HAVE HAPPENED FIRST
Findings #24 and #25 measured RMSNorm, RoPE and SiLU and found them slower. Those kernels could
never have been faster, for two reasons that were both visible before the measurement:
  - they are small and memory-bound, so there is no arithmetic to optimise
  - the compiler was already fusing them into neighbours, and an opaque custom call forfeits that

The fused MLP is the opposite shape. `nkilib.core.mlp.mlp` performs
`down(silu(gate(x)) * up(x))` in one kernel, so it:
  - REPLACES a fusable region rather than interrupting one — the kernel does the fusion internally
  - contains two real matmuls, so there is compute to optimise, not just bandwidth

Finding #18 recorded that this kernel fails to compile single-core when `intermediate_size > 4096`,
and that was used to write it off. But #18's own data shows `hidden_size=1024, intermediate_size=3072`
PASSES at cos_sim 0.999995 — which is exactly Qwen3-0.6B's MLP shape, the model every MFU number in
this project was measured on. So the kernel works for the benchmarked model and was never timed.

WHAT THIS MEASURES
Device time (neuron-explorer `total_time`) and wall time for one MLP block, NKI fused vs torch, at
Qwen3-0.6B dimensions. Correctness is checked against a CPU fp32 reference on every run, because a
fast wrong answer is not a result.

Weights are transposed ON DEVICE, which is the realistic path: in a kernelized model the HF weights
already live on the device, so a wrapper would transpose there. (The first version of the Week 4 spike
transposed on the host and then moved, which materialises the result and proves nothing about how the
kernel handles a non-contiguous view.)

Usage — one invocation per implementation, each needs its own NEFF directory:
    python scripts/profile_fused_mlp_vs_torch.py --impl nki   --calls 28
    python scripts/profile_fused_mlp_vs_torch.py --impl torch --calls 28
Then:
    python scripts/summarise_device_profiles.py /tmp/prof_mlp_nki_n28 /tmp/prof_mlp_torch_n28 --calls 28
"""

import argparse
import functools
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--impl", choices=["nki", "torch"], required=True)
ap.add_argument("--calls", type=int, default=28, help="MLP blocks per graph (28 = Qwen3-0.6B depth)")
ap.add_argument("--hidden", type=int, default=1024, help="Qwen3-0.6B hidden_size")
ap.add_argument("--inter", type=int, default=3072, help="Qwen3-0.6B intermediate_size (<=4096, #18)")
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--outdir", default=None)
ap.add_argument("--iters", type=int, default=4)
ap.add_argument("--shared-weights", action="store_true",
                help="reuse ONE weight set across all blocks. Unrealistic — it lets the compiler "
                     "load the weights once and amortise them over N blocks, which a real model "
                     "cannot do. Kept only to reproduce the flawed first version of this "
                     "measurement, where torch showed 12.1 MB/block of traffic against an 18.9 MB "
                     "weight set.")
args = ap.parse_args()

OUTDIR = args.outdir or f"/tmp/prof_mlp_{args.impl}_n{args.calls}"

os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = OUTDIR
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import time

import torch
import torch.nn.functional as F

from nki_test_utils import require_neuron

H, I, S, B = args.hidden, args.inter, args.seq, args.batch


def cos_sim(a, b):
    a = a.detach().float().flatten().cpu()
    b = b.detach().float().flatten().cpu()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    require_neuron()
    import torch_xla.core.xla_model as xm

    # Finding #24 fix, so the run isn't dominated by subprocess forks. Does not affect device time.
    import nki.compiler.target as nki_target

    nki_target._detect_target = functools.lru_cache(maxsize=1)(nki_target._detect_target)

    dev = xm.xla_device()

    torch.manual_seed(0)
    # HF orientation: gate/up are [I, H], down is [H, I]
    w_gate = torch.randn(I, H, dtype=torch.bfloat16) * (H ** -0.5)
    w_up = torch.randn(I, H, dtype=torch.bfloat16) * (H ** -0.5)
    w_down = torch.randn(H, I, dtype=torch.bfloat16) * (I ** -0.5)
    # The kernel expects a 3D [B, S, H] activation, not [B*S, H] — a 2D input fails with
    # "index 2 out of bounds for sequence of size 2" in mlp_cte_tensor_io.
    x_cpu = torch.randn(B, S, H, dtype=torch.bfloat16)

    # DISTINCT WEIGHTS PER BLOCK, and this matters enough to be the default.
    # Reusing one weight set across N chained blocks lets the compiler load the weights once and
    # amortise them over all N. A real model has different weights per layer, so that amortisation
    # does not exist. With shared weights, torch showed 12.1 MB/block of HBM traffic — less than a
    # single 18.9 MB weight set — which is the tell that the comparison was measuring caching rather
    # than the kernels. --shared-weights reproduces that (wrong) configuration for comparison.
    n_sets = 1 if args.shared_weights else args.calls

    # CPU fp32 reference for ONE MLP block, per Qwen3MLP.forward.
    ref = F.linear(
        F.silu(F.linear(x_cpu.float(), w_gate.float())) * F.linear(x_cpu.float(), w_up.float()),
        w_down.float(),
    )

    x = x_cpu.to(dev)

    print(f"impl={args.impl} calls={args.calls}  H={H} I={I} S={S} B={B} bfloat16")
    print(f"  weight sets: {n_sets} "
          f"({'SHARED — unrealistic, see --shared-weights' if args.shared_weights else 'one per block, realistic'})")
    print(f"  weights per block: {3 * I * H * 2 / 1e6:.1f} MB; "
          f"total {n_sets * 3 * I * H * 2 / 1e6:.0f} MB")
    print(f"  inspect dir: {OUTDIR}")
    print(f"  I<=4096 so Finding #18's compile limit does not apply "
          f"(its own data: H=1024/I=3072 passes at cos_sim 0.999995)")

    # Build n_sets independent weight triples, perturbed so nothing can be CSE'd away.
    weights = []
    for i in range(n_sets):
        d = 0.0 if i == 0 else (i + 1) * 1e-3
        weights.append((w_gate + d, w_up + d, w_down + d))

    if args.impl == "nki":
        # nkilib.core.mlp is the module; the kernel is nkilib.core.mlp.mlp.mlp.
        # Defaults are already what an HF MLP wants: activation_fn=SiLU,
        # normalization_type=NO_NORM, quantization_type=NONE.
        from nkilib.core.mlp.mlp import mlp as nki_mlp

        # Move HF-oriented weights to device, THEN transpose there — the realistic path,
        # since in a kernelized model the HF weights already live on the device.
        dev_w = [(g.to(dev).t(), u.to(dev).t(), dn.to(dev).t()) for g, u, dn in weights]

        def block(t, i):
            g, u, dn = dev_w[i % len(dev_w)]
            out = nki_mlp(t, g, u, dn)
            return out[0] if isinstance(out, (list, tuple)) else out
    else:
        dev_w = [(g.to(dev), u.to(dev), dn.to(dev)) for g, u, dn in weights]

        def block(t, i):
            g, u, dn = dev_w[i % len(dev_w)]
            return F.linear(F.silu(F.linear(t, g)) * F.linear(t, u), dn)

    # ---- correctness first: a fast wrong answer is not a result ---------------------------
    # Block 0 uses the unperturbed weights, so it matches the CPU reference exactly.
    single = block(x, 0)
    xm.mark_step()
    xm.wait_device_ops()
    sim = cos_sim(single.cpu(), ref)
    print(f"  correctness (1 block vs CPU fp32): cos_sim = {sim:.6f}")
    if sim < 0.999:
        print("  FAILED accuracy gate — refusing to report timings for a wrong kernel.")
        return 1
    del single

    # ---- timed graph: N chained MLP blocks, each with its own weights ---------------------
    # Chained so shapes stay consistent (H in == H out for an MLP block), but with distinct
    # weights per block so neither implementation can amortise a single weight load.
    def graph():
        out = x
        for i in range(args.calls):
            out = block(out, i)
        return out

    for i in range(args.iters):
        t0 = time.perf_counter()
        out = graph()
        xm.mark_step()
        xm.wait_device_ops()
        ms = (time.perf_counter() - t0) * 1e3
        print(f"  iter {i}: wall {ms:9.2f} ms  ({ms / args.calls:7.3f} ms/block)"
              f"{'  (compile)' if i == 0 else ''}")
        del out

    # FLOPs per MLP block: 3 matmuls, 2 of [S,H]x[H,I] and 1 of [S,I]x[I,H]
    flops = 2 * B * S * H * I * 3
    print(f"\n  FLOPs per block: {flops / 1e9:.2f} GFLOP  "
          f"({args.calls} blocks = {flops * args.calls / 1e9:.1f} GFLOP)")

    print("\n  artifacts:")
    for p in sorted(Path(OUTDIR).rglob("*.neff")):
        print(f"    NEFF {p.name}  ({p.stat().st_size / 1024:.0f} KiB)")
    for p in sorted(Path(OUTDIR).rglob("*.ntff")):
        print(f"    NTFF {p.name}  ({p.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
