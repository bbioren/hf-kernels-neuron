"""Does the nkilib FLASH ATTENTION kernel beat torch eager attention? The real speedup candidate.

WHY THIS IS THE EXPERIMENT THAT SHOULD DECIDE THE RECOMMENDATION
Findings #25 and #26 established a criterion for when a swapped kernel can win, and then found no
candidate that met it:

  A kernel wins when it replaces a region the compiler would NOT otherwise fuse well,
  AND there is real arithmetic to restructure.

  RMSNorm / RoPE / SiLU  fail both halves. Small, memory-bound, and already fused into their
                         neighbours — so an opaque custom call REMOVES an optimisation. Measured:
                         torch's marginal traffic across a 28-op chain is ~0 MB/call because the
                         chain collapses into one pass, while the NKI kernels sit at exactly the
                         unfused floor of 6.29 MB/call. The kernels are optimal and still lose,
                         because you cannot beat not touching memory.
  Fused MLP              passes the second half but loses 2.99x single-core, because nkilib
                         kernels are built for a multi-core SPMD grid and tile far too finely
                         without one (Finding #26, and #18 reframed).

Attention is the first candidate that passes both halves for a reason that is not incidental:

  1. Flash attention is an ALGORITHMIC restructuring, not a fusion. It never materialises the
     [heads, S, S] score matrix, using online softmax with running max/sum instead. A compiler
     does not discover that from the eager formulation — it fuses elementwise chains, it does not
     re-derive the algorithm. So the compiler is NOT already doing this.
  2. There is real arithmetic: two matmuls per head, and with causal masking half of the score
     tiles can be skipped entirely rather than computed and then masked to -inf. `attention_cte`
     does skip them ("Enables compute skipping: skip MM1/MM2 for upper triangle tiles").
  3. `nkilib/core/attention/attention_cte.py` states it "can be invoked with 1D SPMD grid for LNC2
     or without grid", and falls back to single core for sequences under 1024 tokens. That is the
     exact property the fused MLP lacked, and it is why #26's verdict does not automatically carry
     over to here.

WHAT TORCH BASELINE IS FAIR
The eager formulation HF actually runs on Neuron: softmax(scale * q @ k^T + causal) @ v, with the
score matrix materialised. That is what a kernel swap would replace, so that is what it must beat.
Not SDPA — `torch_neuronx` overrides do not give a flash path here, and comparing against a
hypothetical would be comparing against something the model does not run.

WHY DISTINCT K/V PER LAYER
The first version of the fused-MLP experiment shared one weight set across all chained blocks,
which let the compiler load it once and amortise it over N. A real model has different weights per
layer, so that amortisation does not exist, and the shared-weight version measured caching rather
than kernels. Same trap here: each of the N layers gets its own K/V, and only Q flows through the
chain (attention output and Q input share the shape, so chaining is shape-consistent).

Usage — one invocation per implementation, each needs its own NEFF directory:
    python scripts/profile_attention_nki_vs_torch.py --impl nki   --calls 28
    python scripts/profile_attention_nki_vs_torch.py --impl torch --calls 28
Then:
    python scripts/summarise_device_profiles.py results/raw/prof_attn_{nki,torch}_n28
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--impl", choices=["nki", "torch"], required=True)
ap.add_argument("--calls", type=int, default=28, help="attention layers per graph (28 = Qwen3-0.6B)")
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--heads", type=int, default=16, help="Qwen3-0.6B num_attention_heads")
ap.add_argument("--kv-heads", type=int, default=8, help="Qwen3-0.6B num_key_value_heads (GQA)")
ap.add_argument("--head-dim", type=int, default=128, help="Qwen3-0.6B head_dim")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--causal", action="store_true", default=True)
ap.add_argument("--no-causal", dest="causal", action="store_false")
ap.add_argument("--outdir", default=None)
ap.add_argument("--iters", type=int, default=4)
args = ap.parse_args()

OUTDIR = args.outdir or f"results/raw/prof_attn_{args.impl}_n{args.calls}"

os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = OUTDIR
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

# Clear the profile directory: summarise_device_profiles.py SUMS device time over every NEFF it
# finds, so a leftover silently inflates the result (sticking point #20). This run emits two NEFFs
# by design — a 1-layer correctness graph and an args.calls-layer timed graph.
if Path(OUTDIR).exists():
    shutil.rmtree(OUTDIR)
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent))

import time

import torch
import torch.nn.functional as F

from nki_test_utils import require_neuron

B, S, H, HKV, D = args.batch, args.seq, args.heads, args.kv_heads, args.head_dim
SCALE = 1.0 / math.sqrt(D)
# The kernel folds batch and heads into one leading axis, and expresses GQA as
# batch_size_kv < batch_size rather than by replicating K/V.
NQ, NKV = B * H, B * HKV
GROUP = H // HKV


def cos_sim(a, b):
    a = a.detach().float().flatten().cpu()
    b = b.detach().float().flatten().cpu()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def torch_attention(q, k, v, causal):
    """The eager formulation HF runs: materialise scores, mask, softmax, matmul.

    q: (NQ, S, D)   k, v: (NKV, S, D)   ->  (NQ, S, D)
    K/V are expanded to NQ heads here because torch has no native GQA matmul; that expansion is
    part of the cost of the eager path and is therefore inside the measurement, not outside it.
    """
    k = k.repeat_interleave(GROUP, dim=0)
    v = v.repeat_interleave(GROUP, dim=0)
    scores = torch.bmm(q, k.transpose(1, 2)) * SCALE       # (NQ, S, S)  <- materialised
    if causal:
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.bmm(probs, v)


def main():
    require_neuron()
    import torch_xla.core.xla_model as xm

    # Both dispatch fixes, so this measures the kernel rather than framework overhead. Without them
    # the NKI side carries ~52 ms/call of subprocess and per-call computation rebuild, which would
    # swamp any device-side difference.
    from nki_dispatch_fixes import fix_op_registry_cache, fix_target_detection

    fix_target_detection(verbose=False)
    op_stats = None
    if args.impl == "nki":
        op_stats, _ = fix_op_registry_cache(verbose=False)

    dev = xm.xla_device()
    torch.manual_seed(0)

    q0 = torch.randn(NQ, S, D, dtype=torch.bfloat16) * 0.1
    kvs = [(torch.randn(NKV, S, D, dtype=torch.bfloat16) * 0.1,
            torch.randn(NKV, S, D, dtype=torch.bfloat16) * 0.1)
           for _ in range(args.calls)]

    score_mb = NQ * S * S * 2 / 1e6
    print(f"impl={args.impl} layers={args.calls} causal={args.causal}")
    print(f"  B={B} heads={H} kv_heads={HKV} (GQA group {GROUP}) S={S} D={D} bfloat16")
    print(f"  q {tuple(q0.shape)}  k/v {tuple(kvs[0][0].shape)}  scale={SCALE:.8f}")
    print(f"  eager score matrix is {score_mb:.1f} MB per layer, "
          f"{score_mb * args.calls:.0f} MB over {args.calls} layers")
    print(f"  flash attention never materialises it — that is the hypothesis under test")
    print(f"  inspect dir: {OUTDIR}")

    # ---- CPU fp32 reference for ONE layer ------------------------------------------------
    kq, kk = q0.float(), kvs[0][0].float()
    kv = kvs[0][1].float()
    ref = torch_attention(kq, kk, kv, args.causal)

    q = q0.to(dev)
    dev_kv = [(k.to(dev), v.to(dev)) for k, v in kvs]

    if args.impl == "nki":
        from nkilib.core.attention.attention_cte import attention_cte

        def layer(t, i):
            k, v = dev_kv[i]
            # tp_q=True  -> q is (batch, seqlen, d)
            # tp_k=True  -> k is (batch_kv, seqlen, d), matching how HF stores K
            # tp_out=False -> out is (batch, seqlen, d), same layout as q, so it chains
            out = attention_cte(t, k, v, scale=SCALE, causal_mask=args.causal,
                                tp_q=True, tp_k=True, tp_out=False)
            return out[0] if isinstance(out, (list, tuple)) else out
    else:
        def layer(t, i):
            k, v = dev_kv[i]
            return torch_attention(t, k, v, args.causal)

    # ---- correctness first: a fast wrong answer is not a result ---------------------------
    single = layer(q, 0)
    xm.mark_step()
    xm.wait_device_ops()
    sim = cos_sim(single.cpu(), ref)
    max_abs = (single.cpu().float() - ref).abs().max().item()
    print(f"\n  correctness (1 layer vs CPU fp32): cos_sim = {sim:.6f}, max_abs = {max_abs:.5f}")
    if sim < 0.999:
        print("  FAILED accuracy gate — refusing to report timings for a wrong kernel.")
        print(f"  out shape {tuple(single.shape)} vs ref {tuple(ref.shape)}")
        return 1
    del single

    # ---- timed graph: N chained layers, each with its own K/V ----------------------------
    def graph():
        out = q
        for i in range(args.calls):
            out = layer(out, i)
        return out

    for i in range(args.iters):
        t0 = time.perf_counter()
        out = graph()
        xm.mark_step()
        xm.wait_device_ops()
        ms = (time.perf_counter() - t0) * 1e3
        print(f"  iter {i}: wall {ms:9.2f} ms  ({ms / args.calls:7.3f} ms/layer)"
              f"{'  (compile)' if i == 0 else ''}")
        del out

    # Attention FLOPs per layer: 2 matmuls of (S x D) x (D x S) and (S x S) x (S x D), per q head.
    # Causal masking halves the useful work, and the kernel skips it while torch computes it.
    flops_full = 2 * 2 * NQ * S * S * D
    print(f"\n  FLOPs per layer: {flops_full / 1e9:.2f} GFLOP dense"
          f"{f' ({flops_full / 2 / 1e9:.2f} useful under causal masking)' if args.causal else ''}")
    if op_stats is not None:
        print(f"  Op-registry cache: {op_stats!r}")

    print("\n  artifacts:")
    for p in sorted(Path(OUTDIR).rglob("*.neff")):
        print(f"    NEFF {p.name}  ({p.stat().st_size / 1024:.0f} KiB)")
    for p in sorted(Path(OUTDIR).rglob("*.ntff")):
        print(f"    NTFF {p.name}  ({p.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
