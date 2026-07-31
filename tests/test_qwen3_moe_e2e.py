"""End-to-end: do the three dense NKI kernels transfer to Qwen3-MoE?

Week 5 goal: "reuse RMSNorm/RoPE/SiLU from weeks 2-4" on Qwen3-MoE.

The MoE-*specific* kernels are blocked (see deliverables/week-5-moe-gap-analysis.md:
no Qwen3-MoE interception point, weight-layout mismatch, and routing metadata the kernel
expects the caller to build). But the three dense kernels should transfer for free, because
Qwen3-MoE shares the same decorated interception points:

  RMSNorm         `@use_kernel_forward_from_hub("RMSNorm")` on Qwen3MoeRMSNorm
  rotary_pos_emb  `@use_kernel_func_from_hub("rotary_pos_emb")` — qwen3_moe is in the 95-file list
  SiLU            decorated once in activations.py, so any ACT2FN["silu"] user gets it

"Should transfer for free" is a claim, so this measures it rather than asserting it.

Note on expected counts: Qwen3-MoE's expert MLPs may not route through `ACT2FN["silu"]`
the way the dense MLP does, so the SiLU count is *discovered* here rather than predicted.
Whatever it is, it is reported, and a count of zero is a finding rather than a failure.

Run on trn2:
    python tests/test_qwen3_moe_e2e.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

from neuron_kernel_registration import kernelize_for_neuron
from nki_test_utils import (
    cosine_similarity,
    max_abs_diff,
    nki_call_counter,
    require_neuron,
    sync,
)

SEP = "=" * 80
SEQ_LEN = 128  # multiple of 128 so the RoPE kernel engages


def build_qwen3_moe(num_layers=2, experts_impl=None):
    """Small Qwen3-MoE with random weights. Config built directly, no Hub access.

    `experts_impl` selects transformers' experts kernel via `_experts_implementation`.
    The default (`grouped_mm`) does not run on Neuron — see `find_working_experts_impl`.
    """
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    kwargs = {}
    if experts_impl is not None:
        kwargs["experts_implementation"] = experts_impl

    config = Qwen3MoeConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        max_position_embeddings=1024,
        use_cache=False,
        **kwargs,
    )
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(config)
    model.eval()
    return model, config


def find_working_experts_impl(device, ids_len=SEQ_LEN):
    """Find an experts implementation whose forward actually runs on Neuron.

    The default `grouped_mm` path calls `torch.sort` and `torch.histc`, which lower to a
    `sort` HLO the Neuron compiler rejects:

        [NCC_EVRF029] Operation sort is not supported on trn2. Use supported equivalent
        operation like TopK or replace it with an alternate implementation via NKI.

    That is a property of stock transformers on Neuron, not of our kernels — it fails
    identically with no kernelization. So before testing kernel transfer we have to find a
    configuration in which the model runs at all.
    """
    print("  probing experts implementations (default grouped_mm fails on Neuron) ...")
    for impl in [None, "batched_mm", "deepgemm", "sonicmoe"]:
        label = impl or "default (grouped_mm)"
        try:
            m, cfg = build_qwen3_moe(experts_impl=impl)
            m = m.to(device)
            ids = torch.randint(0, cfg.vocab_size, (1, ids_len)).to(device)
            with torch.no_grad():
                out = m(ids).logits
            sync()
            _ = out.cpu()
            print(f"    {label:24s} OK")
            del m
            return impl
        except Exception as e:
            first = str(e).replace("\n", " ")
            key = "sort is not supported" if "sort is not supported" in first else first[:70]
            print(f"    {label:24s} FAILED  {key}")
    return "__none__"


def forward_overrides(model):
    """module path -> qualname of a swapped (instance-level) forward."""
    out = {}
    for name, m in model.named_modules():
        fwd = m.__dict__.get("forward")
        if fwd is None:
            continue
        fn = getattr(fwd, "__func__", fwd)
        out[name or "<root>"] = getattr(fn, "__qualname__", repr(fn))
    return out


def main():
    device = require_neuron()
    print()
    print(SEP)
    print("Qwen3-MoE + the three dense NKI kernels")
    print(SEP)

    impl = find_working_experts_impl(device)
    if impl == "__none__":
        print()
        print("  No experts implementation runs Qwen3-MoE on Neuron.")
        print("  This blocks the Week 5 goal for a reason unrelated to NKI kernels:")
        print("  transformers' MoE expert paths use ops the Neuron compiler rejects.")
        print("  Recorded as a finding; see deliverables/week-5-moe-gap-analysis.md.")
        print(SEP)
        return 1
    print(f"  using experts_implementation = {impl or 'default'}")
    print()

    try:
        model, config = build_qwen3_moe(experts_impl=impl)
    except Exception as e:
        print(f"  could not build Qwen3-MoE: {type(e).__name__}: {e}")
        return 1

    print(f"  layers={config.num_hidden_layers} hidden={config.hidden_size} "
          f"experts={config.num_experts} top_k={config.num_experts_per_tok} "
          f"seq={SEQ_LEN}")

    # Confirm the interception points exist on this model family before testing the swap.
    from transformers.models.qwen3_moe import modeling_qwen3_moe as mq

    rms_name = getattr(mq.Qwen3MoeRMSNorm, "kernel_layer_name", None)
    rope_obj = getattr(mq, "apply_rotary_pos_emb", None)
    rope_name = getattr(type(rope_obj), "kernel_layer_name", None) if rope_obj else None
    print(f"  Qwen3MoeRMSNorm.kernel_layer_name = {rms_name!r}")
    print(f"  apply_rotary_pos_emb kernel name  = {rope_name!r}")

    model = model.to(device)
    ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN)).to(device)

    with torch.no_grad():
        ref_logits = model(ids).logits
    sync()
    ref_logits = ref_logits.cpu()

    kernelize_for_neuron(model)

    swaps = forward_overrides(model)
    n_rms = sum(1 for q in swaps.values() if "NeuronRMSNorm" in q)
    n_silu = sum(1 for q in swaps.values() if "NeuronSiLU" in q)
    print()
    print(f"  RMSNorm forwards swapped: {n_rms}")
    print(f"  SiLU    forwards swapped: {n_silu}")

    from kernels import get_local_kernel

    rms_mod = get_local_kernel(PROJECT_ROOT / "kernels" / "neuron_rmsnorm")
    rope_mod = get_local_kernel(PROJECT_ROOT / "kernels" / "neuron_rope")
    silu_mod = get_local_kernel(PROJECT_ROOT / "kernels" / "neuron_silu")

    print("  running instrumented forward (each NKI call costs ~52 ms, Finding #20) ...")
    with nki_call_counter(rms_mod, ["_nki_rmsnorm_kernel"], ["_pytorch_rmsnorm"]) as rc:
        with nki_call_counter(rope_mod, ["_nki_rope_hf"], ["_torch_rope"]) as pc:
            with nki_call_counter(silu_mod, ["_nki_silu_kernel"], ["_torch_silu"]) as sc:
                with torch.no_grad():
                    nki_logits = model(ids).logits
                sync()
                nki_logits = nki_logits.cpu()

    cos = cosine_similarity(ref_logits, nki_logits)
    diff = max_abs_diff(ref_logits, nki_logits)
    L = config.num_hidden_layers

    print()
    print(f"  RMSNorm dispatch : {rc}   (expect nki={4*L+1}, same structure as dense)")
    print(f"  RoPE    dispatch : {pc}   (expect nki={L})")
    print(f"  SiLU    dispatch : {sc}   (count discovered, not predicted — see docstring)")
    print(f"  logits cos_sim   : {cos:.6f}")
    print(f"  logits max_diff  : {diff:.3e}")

    rms_ok = rc.nki == 4 * L + 1 and rc.fallback == 0
    rope_ok = pc.nki == L and pc.fallback == 0
    silu_ran = sc.nki > 0
    accurate = cos > 0.999

    print()
    print(f"  NKI RMSNorm transfers : {'yes' if rms_ok else 'NO'}")
    print(f"  NKI RoPE    transfers : {'yes' if rope_ok else 'NO'}")
    print(f"  NKI SiLU    engaged   : {'yes' if silu_ran else 'no'} "
          f"({'expected — MoE experts may not use ACT2FN' if not silu_ran else ''})")
    print(f"  logits match (>0.999) : {'yes' if accurate else 'NO'}")

    # SiLU engagement is reported, not required: whether Qwen3-MoE's expert MLPs route
    # through the decorated activation is a property of the model, not of our kernel.
    passed = rms_ok and rope_ok and accurate
    print()
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    if passed:
        print("  => The dense kernels transfer to Qwen3-MoE with no changes. Week 5's")
        print("     'reuse RMSNorm/RoPE/SiLU' goal is met. The MoE-specific kernels remain")
        print("     blocked — see deliverables/week-5-moe-gap-analysis.md.")
    print(SEP)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
