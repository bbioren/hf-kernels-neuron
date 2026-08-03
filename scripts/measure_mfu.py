"""MFU measurement: Qwen3 forward pass on Neuron, with and without NKI kernels.

This is the Week 4 deliverable and the number the PoC's recommendation turns on.

WHAT IS MEASURED
  Forward-pass-only MFU in eager mode, single logical NeuronCore, bf16.
  Forward-only because all three kernels declare `has_backward = False`, so training
  mode would silently fall back and measure nothing interesting.

DENOMINATOR — stated explicitly, because this is where MFU numbers usually go wrong.
  Trn2 peak bf16 is quoted two ways:
    * 667 TFLOPS/device  — published figure, includes VectorEngine + ScalarEngine
    * 632 TFLOPS/device  — TensorEngine only (79 TFLOPS x 8), per the Cayman arch doc
  We report against **632** as the primary (matmul-bound work belongs to the TensorEngine)
  and show 667 alongside so either convention can be recovered.

  Second and more important subtlety: this instance is `logical-neuroncore-config: 2`
  (LNC2), so its 4 physical cores present as 2 logical cores. An eager per-layer swap runs
  on ONE logical core = half the device. So:
    * per-device peak  = 632 TFLOPS
    * per-logical-core = 316 TFLOPS   <-- the honest denominator for this run
  Both are reported. Quoting the per-device number for a single-core run would understate
  MFU by 2x; quoting per-core as if it were a device number would overstate it by 2x.

FLOP COUNTING
  Computed explicitly from the config and printed, so the arithmetic is auditable rather
  than a black box. Forward pass, per step:
    QKV proj    2 * T * H * (Hq + 2*Hkv)
    attn QK^T   2 * T * S * Hq
    attn AV     2 * T * S * Hq
    O proj      2 * T * Hq * H
    MLP         2 * T * H * I * 3            (gate, up, down)
    LM head     2 * T * H * V
  where T = batch*seq tokens, H = hidden, I = intermediate, S = seq,
  Hq = q_heads*head_dim, Hkv = kv_heads*head_dim, V = vocab.
  Embedding lookups are not FLOPs and are excluded.

  Note the attention terms assume full (non-causal-masked) score computation, which is what
  an eager implementation actually executes. Halving them for causality would raise MFU;
  we do not, and say so.

ALSO REPORTED: kernel launch count per step. Finding #19 measured ~0.36 ms of host-side
dispatch per NKI call vs ~0.011 ms eager. If step time is launch-bound, MFU alone reads as
a kernel-quality problem when it is an integration-model problem, so the two numbers have
to be presented together.

CAVEAT: `use_kernels=True` cannot route to Neuron (Finding #9), so the kernelized run goes
through `kernelize_for_neuron()`. Same kernels, same swap mechanism, non-standard entry
point. Stated in the output.

Run on trn2:
    python scripts/measure_mfu.py                 # default: Qwen3-0.6B-like, full depth
    python scripts/measure_mfu.py --preset 8b     # Qwen3-8B widths, reduced depth
    python scripts/measure_mfu.py --seq 512
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

from neuron_kernel_registration import kernelize_for_neuron
from nki_test_utils import nki_call_counter, require_neuron

SEP = "=" * 88

# Trn2 peak bf16, TFLOPS per device. See the denominator note above.
PEAK_TFLOPS_DEVICE_TENSORENGINE = 632.0
PEAK_TFLOPS_DEVICE_PUBLISHED = 667.0
LOGICAL_CORES_PER_DEVICE = 2  # LNC2 on this instance
LOGICAL_CORES_USED = 1        # eager per-layer swap runs on one

PRESETS = {
    # Qwen3-0.6B actual config: full depth, so this is a real model end to end.
    "0.6b": dict(hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
                 num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                 vocab_size=151936),
    # Qwen3-8B widths at reduced depth. MFU is a rate, so depth affects it only weakly;
    # width and seq length are what matter. Labelled a proxy in the output.
    "8b": dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=4,
               num_attention_heads=32, num_key_value_heads=8, head_dim=128,
               vocab_size=151936),
}


def forward_flops(cfg, batch, seq):
    """Explicit forward-pass FLOP count. Returns (total, breakdown dict)."""
    T = batch * seq
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    Hq = cfg["num_attention_heads"] * cfg["head_dim"]
    Hkv = cfg["num_key_value_heads"] * cfg["head_dim"]
    V = cfg["vocab_size"]
    S = seq

    per_layer = {
        "qkv_proj": 2 * T * H * (Hq + 2 * Hkv),
        "attn_qk": 2 * T * S * Hq,
        "attn_av": 2 * T * S * Hq,
        "o_proj": 2 * T * Hq * H,
        "mlp": 2 * T * H * I * 3,
    }
    breakdown = {k: v * L for k, v in per_layer.items()}
    breakdown["lm_head"] = 2 * T * H * V
    return sum(breakdown.values()), breakdown


def build_model(cfg, dtype=torch.bfloat16):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        max_position_embeddings=8192,
        use_cache=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).to(dtype)
    model.eval()
    return model, config


def sync():
    import torch_xla.core.xla_model as xm

    xm.mark_step()
    try:
        xm.wait_device_ops()
    except Exception:
        pass


def time_steps(model, ids, iters, warmup):
    """Median / IQR step time in seconds. Output consumed to defeat dead-code elimination."""
    for _ in range(warmup):
        with torch.no_grad():
            out = model(ids).logits
        float(out.float().sum().item())
        sync()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(ids).logits
        float(out.float().sum().item())
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return (statistics.median(samples),
            samples[len(samples) // 4],
            samples[(3 * len(samples)) // 4])


def mfu_from(flops, step_s):
    achieved_tflops = flops / step_s / 1e12
    per_core_peak = PEAK_TFLOPS_DEVICE_TENSORENGINE / LOGICAL_CORES_PER_DEVICE
    return {
        "achieved_tflops": achieved_tflops,
        "mfu_per_core_te": 100.0 * achieved_tflops / per_core_peak,
        "mfu_device_te": 100.0 * achieved_tflops / PEAK_TFLOPS_DEVICE_TENSORENGINE,
        "mfu_per_core_published": 100.0 * achieved_tflops
        / (PEAK_TFLOPS_DEVICE_PUBLISHED / LOGICAL_CORES_PER_DEVICE),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="0.6b", choices=sorted(PRESETS))
    ap.add_argument("--seq", type=int, default=512, help="must be a multiple of 128 for RoPE")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--only", default=None,
        help="comma-separated kernel names to enable (RMSNorm, rotary_pos_emb, SiLU). "
             "Enables per-kernel attribution: run one at a time to find which kernel "
             "carries the cost in situ, since in-isolation microbenchmarks do not predict "
             "in-model cost (see Finding #19).",
    )
    ap.add_argument("--skip-baseline", action="store_true",
                    help="reuse a known baseline step time instead of re-measuring")
    ap.add_argument("--baseline-ms", type=float, default=None,
                    help="baseline step time in ms, for use with --skip-baseline")
    ap.add_argument(
        "--fix-target-detection", action="store_true",
        help="cache nki.compiler.target._detect_target, which otherwise forks `neuron-ls` on "
             "EVERY kernel invocation at ~52 ms a time (Finding #24). Without this flag the "
             "measurement reproduces the original 208x regression; with it, the measurement "
             "reflects what a customer would see once the upstream bug is fixed. Verified "
             "accuracy-neutral by scripts/probe_target_override_fix.py.",
    )
    ap.add_argument(
        "--fix-op-registry", action="store_true",
        help="ALSO register the XLA computation once per compile-cache key instead of once per "
             "kernel call (open item B12). torch_xla's Op class already memoises the built "
             "computation; NKI applies @xla_hlo_call inside __call__, so a fresh Op with an empty "
             "memo is created every time. Removes ~2/3 of the residual that remains after #24: "
             "0.53 -> 0.18 ms/call. Verified accuracy-neutral by "
             "scripts/probe_op_registry_cache.py. Implies --fix-target-detection, since measuring "
             "this residual under a 52 ms constant is pointless.",
    )
    args = ap.parse_args()

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    # Applied before any kernel runs. Patching the module attribute works regardless of how
    # resolve_target imported it, because resolve_target resolves _detect_target from its own
    # module globals at call time.
    op_stats = None
    if args.fix_target_detection or args.fix_op_registry:
        from nki_dispatch_fixes import fix_target_detection

        fix_target_detection()
    if args.fix_op_registry:
        from nki_dispatch_fixes import fix_op_registry_cache

        op_stats, _restore = fix_op_registry_cache()
        if op_stats is None:
            print("  ABORTING: --fix-op-registry was requested but the patch does not apply to "
                  "this NKI version. Refusing to report a number under a fix that is not active.")
            return 1

    device = require_neuron()
    cfg = PRESETS[args.preset]

    print(SEP)
    print(f"MFU: Qwen3 forward pass on Neuron — preset '{args.preset}'")
    print(SEP)
    print(f"  hidden={cfg['hidden_size']} intermediate={cfg['intermediate_size']} "
          f"layers={cfg['num_hidden_layers']} q_heads={cfg['num_attention_heads']} "
          f"kv_heads={cfg['num_key_value_heads']} head_dim={cfg['head_dim']}")
    print(f"  batch={args.batch} seq={args.seq} dtype=bfloat16 forward-only, eager")
    if args.preset == "8b":
        print("  NOTE: Qwen3-8B *widths* at reduced depth (4 layers). MFU is a rate, so")
        print("        depth affects it weakly; width and seq are what matter. Still a proxy.")
    if args.seq % 128 != 0:
        print(f"  WARNING: seq={args.seq} is not a multiple of 128, so the RoPE kernel will")
        print("           fall back. The kernelized run will not include NKI RoPE.")

    flops, breakdown = forward_flops(cfg, args.batch, args.seq)
    print()
    print("  FLOP breakdown (forward, per step):")
    for k, v in breakdown.items():
        print(f"    {k:12s} {v/1e9:12.2f} GFLOP  ({100.0*v/flops:5.1f}%)")
    print(f"    {'TOTAL':12s} {flops/1e9:12.2f} GFLOP")
    print()
    print("  Denominator:")
    print(f"    device peak (TensorEngine only) {PEAK_TFLOPS_DEVICE_TENSORENGINE:6.0f} TFLOPS bf16")
    print(f"    device peak (published)         {PEAK_TFLOPS_DEVICE_PUBLISHED:6.0f} TFLOPS bf16")
    print(f"    LNC config {LOGICAL_CORES_PER_DEVICE} logical cores/device, using "
          f"{LOGICAL_CORES_USED}")
    print(f"    => primary denominator = "
          f"{PEAK_TFLOPS_DEVICE_TENSORENGINE / LOGICAL_CORES_PER_DEVICE:.0f} TFLOPS "
          f"(per logical core, TensorEngine)")

    ids = torch.randint(0, cfg["vocab_size"], (args.batch, args.seq))

    # ---------------- baseline ----------------
    print()
    print(SEP)
    print("BASELINE (no kernels)")
    print(SEP)
    ids_d = ids.to(device)
    if args.skip_baseline and args.baseline_ms:
        base_med = args.baseline_ms / 1e3
        base_q1 = base_q3 = base_med
        print(f"  using supplied baseline {args.baseline_ms:.2f} ms (not re-measured)")
    else:
        model, _ = build_model(cfg)
        model = model.to(device)
        print("  compiling + warming up ...")
        t0 = time.time()
        base_med, base_q1, base_q3 = time_steps(model, ids_d, args.iters, args.warmup)
        print(f"  done in {time.time() - t0:.0f}s")
        del model
    base = mfu_from(flops, base_med)
    print(f"  step time      {base_med*1e3:9.2f} ms   (IQR {base_q1*1e3:.2f}-{base_q3*1e3:.2f})")
    print(f"  throughput     {args.batch*args.seq/base_med:9.0f} tok/s")
    print(f"  achieved       {base['achieved_tflops']:9.2f} TFLOPS")
    print(f"  MFU per-core   {base['mfu_per_core_te']:9.2f} %   (632/2 TensorEngine)")
    print(f"  MFU device     {base['mfu_device_te']:9.2f} %   (632, for reference only)")

    # ---------------- kernelized ----------------
    print()
    print(SEP)
    label = ", ".join(only) if only else "RMSNorm + RoPE + SiLU"
    print(f"KERNELIZED (NKI {label})")
    print(SEP)
    print("  NOTE: via kernelize_for_neuron(), not use_kernels=True — see Finding #9.")
    model2, _ = build_model(cfg)
    model2 = model2.to(device)
    kernelize_for_neuron(model2, only=only)

    from kernels import get_local_kernel

    rms_mod = get_local_kernel(PROJECT_ROOT / "kernels" / "neuron_rmsnorm")
    rope_mod = get_local_kernel(PROJECT_ROOT / "kernels" / "neuron_rope")
    silu_mod = get_local_kernel(PROJECT_ROOT / "kernels" / "neuron_silu")

    # One instrumented step to establish launch counts and confirm the kernels engage.
    print("  confirming kernel engagement (one instrumented step) ...")
    with nki_call_counter(rms_mod, ["_nki_rmsnorm_kernel"], ["_pytorch_rmsnorm"]) as rc:
        with nki_call_counter(rope_mod, ["_nki_rope_hf"], ["_torch_rope"]) as pc:
            with nki_call_counter(silu_mod, ["_nki_silu_kernel"], ["_torch_silu"]) as sc:
                with torch.no_grad():
                    out = model2(ids_d).logits
                float(out.float().sum().item())
                sync()
    L = cfg["num_hidden_layers"]
    expected = {"RMSNorm": 4 * L + 1, "rotary_pos_emb": L, "SiLU": L}
    if only is not None:
        expected = {k: (v if k in only else 0) for k, v in expected.items()}
    print(f"    RMSNorm {rc}  (expect nki={expected['RMSNorm']})")
    print(f"    RoPE    {pc}  (expect nki={expected['rotary_pos_emb']})")
    print(f"    SiLU    {sc}  (expect nki={expected['SiLU']})")
    launches = rc.nki + pc.nki + sc.nki
    counts_ok = (rc.nki == expected["RMSNorm"]
                 and pc.nki == expected["rotary_pos_emb"]
                 and sc.nki == expected["SiLU"])
    no_fallback = rc.fallback == 0 and pc.fallback == 0 and sc.fallback == 0
    engaged = counts_ok and no_fallback
    if not engaged:
        print("    WARNING: launch counts do not match expectation. MFU below may not")
        print("             measure what you think. Check for silent fallbacks.")
    print(f"    NKI launches per step: {launches}")

    print("  timing ...")
    t0 = time.time()
    k_med, k_q1, k_q3 = time_steps(model2, ids_d, args.iters, args.warmup)
    print(f"  done in {time.time() - t0:.0f}s")
    kern = mfu_from(flops, k_med)
    print(f"  step time      {k_med*1e3:9.2f} ms   (IQR {k_q1*1e3:.2f}-{k_q3*1e3:.2f})")
    print(f"  throughput     {args.batch*args.seq/k_med:9.0f} tok/s")
    print(f"  achieved       {kern['achieved_tflops']:9.2f} TFLOPS")
    print(f"  MFU per-core   {kern['mfu_per_core_te']:9.2f} %   (632/2 TensorEngine)")
    print(f"  MFU device     {kern['mfu_device_te']:9.2f} %   (632, for reference only)")

    # ---------------- verdict ----------------
    print()
    print(SEP)
    print("RESULT")
    print(SEP)
    speedup = base_med / k_med
    delta_mfu = kern["mfu_per_core_te"] - base["mfu_per_core_te"]
    overlap = not (k_q3 < base_q1 or base_q3 < k_q1)

    print(f"  baseline   MFU {base['mfu_per_core_te']:6.2f} %   step {base_med*1e3:8.2f} ms")
    print(f"  kernelized MFU {kern['mfu_per_core_te']:6.2f} %   step {k_med*1e3:8.2f} ms")
    print(f"  delta          {delta_mfu:+6.2f} pp  speedup {speedup:.3f}x"
          + ("   (IQRs OVERLAP — not resolvable)" if overlap else ""))
    print()
    if overlap:
        print("  The two distributions overlap, so this run cannot distinguish them.")
    elif speedup > 1.0:
        print(f"  Kernels are FASTER by {speedup:.3f}x.")
    else:
        delta_ms = (k_med - base_med) * 1e3
        per_launch = delta_ms / launches if launches else float("nan")
        host_est = launches * 0.36
        print(f"  Kernels are SLOWER by {1/speedup:.3f}x.")
        print()
        print(f"  Attribution: {launches} NKI launches/step, delta {delta_ms:.0f} ms")
        print(f"    => {per_launch:.1f} ms of added cost per NKI call")
        print(f"    host dispatch alone (Finding #19, ~0.36 ms/call) = ~{host_est:.0f} ms")
        if per_launch > 5 * 0.36:
            print(f"    {per_launch:.1f} ms/call is {per_launch/0.36:.0f}x host dispatch, so this is")
            print("    NOT launch-bound. Something structural is costing per-call time —")
            print("    most likely each @nki.jit call executing as its own NEFF with an HBM")
            print("    round trip, rather than fusing into the surrounding XLA graph.")
            print("    Re-run with --only to attribute the cost to a specific kernel.")
        else:
            print("    That is the same order as the delta, so the result is launch-bound")
            print("    rather than kernel-quality-bound.")
    if op_stats is not None:
        # Print the cache statistics, not only the timing. A faster run with zero cache hits would
        # mean the speedup came from somewhere else, and that should be impossible to overlook.
        print()
        print(f"  Op-registry cache (B12): {op_stats!r}")
        if op_stats.get("hit", 0) == 0:
            print("    WARNING: zero hits. The fix was applied but never used, so it cannot be")
            print("    credited with any part of this result.")
    print()
    print("  Caveats: forward-only; single logical core; eager (not torch.compile);")
    print("  kernelized via kernelize_for_neuron() since use_kernels=True can't route")
    print("  to Neuron; attention FLOPs counted without a causal-mask discount.")
    print(SEP)

    if args.json_out:
        payload = {
            "preset": args.preset, "batch": args.batch, "seq": args.seq,
            "config": cfg, "flops_forward": flops, "flop_breakdown": breakdown,
            "denominator": {
                "device_tflops_tensorengine": PEAK_TFLOPS_DEVICE_TENSORENGINE,
                "device_tflops_published": PEAK_TFLOPS_DEVICE_PUBLISHED,
                "logical_cores_per_device": LOGICAL_CORES_PER_DEVICE,
                "logical_cores_used": LOGICAL_CORES_USED,
            },
            "baseline": {"step_s": base_med, **base},
            "kernelized": {"step_s": k_med, **kern,
                           "nki_launches_per_step": launches,
                           "all_kernels_engaged": engaged},
            "speedup": speedup, "delta_mfu_pp": delta_mfu, "iqr_overlap": overlap,
            "fixes": {
                "target_detection": bool(args.fix_target_detection or args.fix_op_registry),
                "op_registry": bool(args.fix_op_registry),
                "op_registry_cache_stats": op_stats.as_dict() if op_stats else None,
            },
        }
        # mkdir first. Without it a --json-out into a new directory throws AFTER the whole
        # measurement has run, which on a long model run means losing the result to a one-line bug
        # in the last statement. Every other producer in this project does the same.
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
