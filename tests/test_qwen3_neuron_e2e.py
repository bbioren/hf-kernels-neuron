"""End-to-end: NKI RMSNorm + RoPE swapped into a real Qwen3 model on Neuron.

This is the Week 3 integration check. It verifies the whole chain:

  1. Both kernels load through the Kernel Hub mechanism (LocalLayerRepository for
     the RMSNorm layer, LocalFuncRepository for the RoPE function).
  2. `kernelize()` finds and swaps them — including the function kernel, which
     requires the `_hidden_kernels` attach/detach dance.
  3. Both NKI kernels actually execute during a real forward pass (call counters,
     per Finding #8 — not inferred from output values).
  4. Logits still match the unkernelized model.

Then it separately demonstrates that the proposed upstream fix for Finding #9 is
sufficient to make `use_kernels=True` reach these kernels, using an in-process
shim rather than modifying the installed venv.

Run on trn2:
    python tests/test_qwen3_neuron_e2e.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

from neuron_kernel_registration import (
    disable_neuron_device_detection,
    enable_neuron_device_detection,
    kernelize_for_neuron,
    register_with_transformers,
)
from nki_test_utils import (
    cosine_similarity,
    load_kernel_module,
    max_abs_diff,
    nki_call_counter,
    require_neuron,
    sync,
)

SEP = "=" * 76

# seq_len must be a multiple of 128 for the RoPE kernel to engage.
SEQ_LEN = 128


def build_qwen3(num_layers=2, hidden_size=256, head_dim=64):
    """Small Qwen3 with random weights. Config built directly to avoid Hub access."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=1024,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=1024,
        use_cache=False,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model, config


def forward_overrides(model) -> dict[str, str]:
    """Map module path -> qualname of its swapped forward.

    `kernelize()` swaps by assigning an *instance* attribute:
        module.forward = MethodType(kernel_layer.forward, module)
    so a swapped module has "forward" in its own __dict__, while an untouched one
    inherits forward from its class. That instance-attribute presence is the
    reliable signal; the class name is not.
    """
    out: dict[str, str] = {}
    for name, m in model.named_modules():
        fwd = m.__dict__.get("forward")
        if fwd is None:
            continue
        fn = getattr(fwd, "__func__", fwd)
        out[name or "<root>"] = getattr(fn, "__qualname__", repr(fn))
    return out


def count_rmsnorm_swaps(model) -> int:
    return sum(1 for qn in forward_overrides(model).values() if "NeuronRMSNorm" in qn)


def test_kernels_execute_via_repo_objects(device):
    """Swap both kernels into a real Qwen3 forward and prove they execute.

    The instrumentation has to target the module objects the *repositories* loaded,
    not a fresh `load_kernel_module()` copy — those are distinct objects, and
    patching the wrong one yields nki=0 while the kernel is in fact running.
    `get_local_kernel()` caches, so it hands back the same object the repos used.

    This is the authoritative execution proof required by Finding #8.
    """
    print(SEP)
    print("1. Both NKI kernels in a real Qwen3 forward pass (execution-verified)")
    print(SEP)

    from kernels import get_local_kernel

    rms_repo_path = PROJECT_ROOT / "kernels" / "neuron_rmsnorm"
    rope_repo_path = PROJECT_ROOT / "kernels" / "neuron_rope"

    rms_kernel_mod = get_local_kernel(rms_repo_path)
    rope_kernel_mod = get_local_kernel(rope_repo_path)

    model, config = build_qwen3()
    model = model.to(device)
    ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN)).to(device)

    with torch.no_grad():
        ref_logits = model(ids).logits
    sync()
    ref_logits = ref_logits.cpu()

    kernelize_for_neuron(model)

    n_rms_swapped = count_rmsnorm_swaps(model)
    print(f"  RMSNorm forwards swapped: {n_rms_swapped}")
    print("  swapped modules:")
    for path, qn in sorted(forward_overrides(model).items()):
        print(f"      {path or '<root>':38s} -> {qn}")

    with nki_call_counter(rms_kernel_mod, ["_nki_rmsnorm_kernel"], ["_pytorch_rmsnorm"]) as rms_counts:
        with nki_call_counter(rope_kernel_mod, ["_nki_rope_hf"], ["_torch_rope"]) as rope_counts:
            with torch.no_grad():
                nki_logits = model(ids).logits
            sync()
            nki_logits = nki_logits.cpu()

    cos = cosine_similarity(ref_logits, nki_logits)
    diff = max_abs_diff(ref_logits, nki_logits)

    # Expected counts for a 2-layer Qwen3: RMSNorm = 4 per layer
    # (input_layernorm, post_attention_layernorm, q_norm, k_norm) + 1 final norm = 9.
    # RoPE = 1 per layer = 2.
    n_layers = config.num_hidden_layers
    expected_rms = 4 * n_layers + 1
    expected_rope = n_layers

    print()
    print(f"  RMSNorm dispatch : {rms_counts}  (expected nki={expected_rms})")
    print(f"  RoPE    dispatch : {rope_counts}  (expected nki={expected_rope})")
    print(f"  logits cos_sim   : {cos:.6f}")
    print(f"  logits max_diff  : {diff:.3e}")

    rms_ran = rms_counts.nki > 0 and rms_counts.fallback == 0
    rope_ran = rope_counts.nki > 0 and rope_counts.fallback == 0
    counts_match = rms_counts.nki == expected_rms and rope_counts.nki == expected_rope
    accurate = cos > 0.999

    print()
    print(f"  NKI RMSNorm executed  : {'yes' if rms_ran else 'NO'} ({rms_counts})")
    print(f"  NKI RoPE executed     : {'yes' if rope_ran else 'NO'} ({rope_counts})")
    print(f"  call counts as expected: {'yes' if counts_match else 'NO'}")
    print(f"  logits match (>0.999) : {'yes' if accurate else 'NO'}")

    passed = rms_ran and rope_ran and counts_match and accurate
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_use_kernels_with_proposed_fix(device):
    """Does `use_kernels=True` work once the Finding #9 fix is applied?

    Applies the in-process shim (xla-on-Neuron -> Device(type="neuron")), registers
    the neuron entries, and drives the *transformers* entry point rather than the
    kernels library. If the swap fires, the proposed upstream patch is sufficient.
    """
    print(SEP)
    print("2. Would the proposed upstream fix make use_kernels=True work?")
    print(SEP)

    from transformers.integrations import hub_kernels

    model, config = build_qwen3()
    model = model.to(device)
    ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN)).to(device)

    # First: confirm it fails WITHOUT the shim. That is the finding.
    print("  --- stock behaviour (no shim) ---")
    try:
        register_with_transformers()
        hub_kernels.kernelize(model)
        n_rms = count_rmsnorm_swaps(model)
        print(f"    kernelize() returned without error; RMSNorm swapped = {n_rms}")
        stock_swapped = n_rms > 0
        if not stock_swapped:
            print("    => mapping ignored, as predicted (device resolved to xla/cpu)")
    except Exception as e:
        stock_swapped = False
        print(f"    kernelize() raised: {type(e).__name__}: {e}")
        print("    => as predicted by Finding #9")

    # Now with the shim + registered entries.
    print()
    print("  --- with proposed fix applied in-process ---")
    model2, config2 = build_qwen3()
    model2 = model2.to(device)
    ids2 = torch.randint(0, config2.vocab_size, (1, SEQ_LEN)).to(device)

    with torch.no_grad():
        ref_logits = model2(ids2).logits
    sync()
    ref_logits = ref_logits.cpu()

    enable_neuron_device_detection()
    try:
        register_with_transformers()
        hub_kernels.kernelize(model2)
        n_rms = count_rmsnorm_swaps(model2)
        print(f"    RMSNorm layers swapped: {n_rms}")

        with torch.no_grad():
            out_logits = model2(ids2).logits
        sync()
        out_logits = out_logits.cpu()
        cos = cosine_similarity(ref_logits, out_logits)
        print(f"    logits cos_sim after use_kernels path: {cos:.6f}")
        fixed_swapped = n_rms > 0 and cos > 0.999
    except Exception as e:
        import traceback

        fixed_swapped = False
        print(f"    FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        disable_neuron_device_detection()

    print()
    print(f"  stock use_kernels reaches neuron : {'yes' if stock_swapped else 'no'}")
    print(f"  with proposed fix                : {'yes' if fixed_swapped else 'no'}")
    conclusion = (not stock_swapped) and fixed_swapped
    if conclusion:
        print("  => Finding #9 confirmed AND the proposed fix is sufficient.")
    print(f"  {'PASS' if conclusion else 'INCONCLUSIVE'}")
    print()
    return conclusion


def main():
    device = require_neuron()
    print()
    print(SEP)
    print("Qwen3 + NKI RMSNorm + NKI RoPE on Neuron — end to end")
    print(SEP)
    print(f"  seq_len = {SEQ_LEN} (multiple of 128, so the RoPE kernel engages)")
    print()

    results = []
    results.append(("both NKI kernels execute in Qwen3, logits match",
                    test_kernels_execute_via_repo_objects(device)))

    results.append(("proposed fix enables use_kernels=True",
                    test_use_kernels_with_proposed_fix(device)))

    print(SEP)
    print("RESULTS")
    print(SEP)
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print(SEP)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
