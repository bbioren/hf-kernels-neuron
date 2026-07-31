"""How much of the in-model slowdown is device time, and how much is dispatch?

THE GAP THIS FILLS
Finding #25 measured NKI 2.5-2.7x slower than torch on device — but from a CHAINED microbenchmark,
28 identical ops back to back. That is the best possible case for the compiler's fusion and therefore
the worst possible case for NKI, so it is an upper bound on the fusion penalty, not an estimate of it.
In a real model these ops are separated by matmuls and less fusion is available.

Finding #24 measured the model-level slowdown in wall clock: 42.04 -> 141.43 ms/step at seq 512.
That conflates dispatch cost with device cost, so it cannot separate the two.

This profiles the real Qwen3 forward, baseline and kernelized, and sums device time across every NEFF
the runtime executes. That decomposes the wall-clock gap:

    wall_kernelized - wall_baseline  =  (device_k - device_b)  +  (dispatch_k - dispatch_b)
                                         ^ fusion loss in situ    ^ per-call dispatch overhead

If device_k - device_b is small, the in-situ fusion penalty is minor and the model-level regression is
almost entirely dispatch — which would mean Fix 7 matters more than #25 suggests. If it is large, the
fusion barrier is real in situ and #25's direction holds at model scale.

Either result is worth having, and neither is currently known.

Usage — one invocation per configuration, each needs its own NEFF directory:
    python scripts/profile_model_device_time.py --mode baseline   --outdir /tmp/prof_model_base
    python scripts/profile_model_device_time.py --mode kernelized --outdir /tmp/prof_model_kern

Then:
    python scripts/summarise_device_profiles.py /tmp/prof_model_base /tmp/prof_model_kern --calls 1
(which sums nothing — use --sum-all below instead, since a full model emits several NEFFs)
"""

import argparse
import functools
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["baseline", "kernelized"], required=True)
ap.add_argument("--outdir", default=None)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--iters", type=int, default=4)
args = ap.parse_args()

OUTDIR = args.outdir or f"/tmp/prof_model_{args.mode}"

os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = OUTDIR
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import time

import torch

from nki_test_utils import require_neuron

# Same config as scripts/measure_mfu.py --preset 0.6b, so numbers are comparable.
CFG = dict(
    hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
    num_attention_heads=16, num_key_value_heads=8, head_dim=128, vocab_size=151936,
)


def main():
    require_neuron()
    import torch_xla.core.xla_model as xm

    # Finding #24 fix. Without it the kernelized run takes minutes of subprocess forking, which
    # does not affect device time but wastes the whole session.
    import nki.compiler.target as nki_target

    nki_target._detect_target = functools.lru_cache(maxsize=1)(nki_target._detect_target)

    from transformers import AutoConfig, AutoModelForCausalLM

    dev = xm.xla_device()

    cfg = AutoConfig.for_model("qwen3", **CFG)
    cfg.torch_dtype = torch.bfloat16
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()

    if args.mode == "kernelized":
        from neuron_kernel_registration import kernelize_for_neuron

        model = model.to(dev)
        kernelize_for_neuron(model)
    else:
        model = model.to(dev)

    ids = torch.randint(0, CFG["vocab_size"], (args.batch, args.seq)).to(dev)

    print(f"mode={args.mode} seq={args.seq} batch={args.batch} layers={CFG['num_hidden_layers']}")
    print(f"  inspect dir: {OUTDIR}")
    print(f"  NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '(unset — compiler defaults)')}")

    # Call counts are already established by scripts/measure_mfu.py (169/step, zero fallbacks),
    # so this run only needs wall time plus the emitted NEFFs. Keeping the counter out avoids
    # patching kernel modules while a profile is being captured.
    with torch.no_grad():
        for i in range(args.iters):
            t0 = time.perf_counter()
            out = model(ids)
            xm.mark_step()
            xm.wait_device_ops()
            ms = (time.perf_counter() - t0) * 1e3
            print(f"  iter {i}: wall {ms:9.2f} ms{'  (compile)' if i == 0 else ''}")
            del out

    neffs = sorted(Path(OUTDIR).rglob("*.neff"))
    ntffs = sorted(Path(OUTDIR).rglob("*.ntff"))
    print(f"\n  {len(neffs)} NEFF(s), {len(ntffs)} NTFF(s) written under {OUTDIR}")
    print("  (a full model emits several; device time must be SUMMED across them)")
    for p in neffs:
        print(f"    {p.name}  ({p.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
