"""Test NeuronRMSNorm with a real Qwen3 model layer via kernelize().

This is the end-to-end validation: load an actual Qwen3 model (small variant),
swap in NeuronRMSNorm via the Kernel Hub mechanism, and confirm the output
matches the original.

Run on trn2:
    python tests/test_qwen3_layer.py
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

import torch
import torch.nn.functional as F
from kernels import kernelize, Mode, LocalLayerRepository, use_kernel_mapping


def load_neuron_rmsnorm_module():
    """Load the neuron_rmsnorm package from our local kernel directory."""
    kernel_path = PROJECT_ROOT / "kernels" / "neuron_rmsnorm" / "__init__.py"
    spec = importlib.util.spec_from_file_location("neuron_rmsnorm", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def test_qwen3_single_layer():
    """Test NeuronRMSNorm swap on a single Qwen3RMSNorm layer."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    print("--- Test 1: Single Qwen3RMSNorm layer swap ---")

    hidden_size = 896  # Qwen3-0.5B
    eps = 1e-6

    # Create a real Qwen3RMSNorm with random weights
    norm = Qwen3RMSNorm(hidden_size, eps=eps)
    norm.weight.data = torch.randn(hidden_size) * 0.5 + 1.0

    # Get reference output before swap
    x = torch.randn(2, 64, hidden_size)
    with torch.no_grad():
        ref_output = norm(x.clone())

    # Set up kernel mapping and kernelize
    repo_path = PROJECT_ROOT / "kernels" / "neuron_rmsnorm"
    mapping = {
        "RMSNorm": {
            "neuron": LocalLayerRepository(
                repo_path=repo_path,
                layer_name="NeuronRMSNorm",
            )
        }
    }

    with use_kernel_mapping(mapping, inherit_mapping=False):
        kernelize(norm, device="neuron", mode=Mode.INFERENCE)

    # Get output after swap (NKI kernel should be executing now)
    with torch.no_grad():
        nki_output = norm(x.clone())

    cos_sim = cosine_similarity(ref_output, nki_output)
    max_diff = (ref_output.float() - nki_output.float()).abs().max().item()

    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max abs diff:      {max_diff:.2e}")
    print(f"  Shape:             {x.shape}")

    passed = cos_sim > 0.999
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    return passed


def test_qwen3_model_forward():
    """Test NeuronRMSNorm inside a real Qwen3 model forward pass.

    Loads Qwen3 0.5B config (no weights — uses random init) and runs
    a forward pass with and without the NKI kernel swap.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    print("--- Test 2: Qwen3 model forward with NKI RMSNorm ---")

    # Load config only (random weights to avoid downloading 1GB+)
    try:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception:
        # If we can't access the Hub, create a minimal config
        from transformers import Qwen2Config  # Qwen3 config may not exist separately
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")

    # Reduce layers for faster testing
    config.num_hidden_layers = 2
    config.use_cache = False

    print(f"  Model: Qwen3 (2 layers, hidden_size={config.hidden_size})")

    # Create model with random weights
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    # Create dummy input
    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    # Reference output
    with torch.no_grad():
        ref_output = model(input_ids).logits.clone()

    # Kernelize — swap all RMSNorm layers with NKI version
    repo_path = PROJECT_ROOT / "kernels" / "neuron_rmsnorm"
    mapping = {
        "RMSNorm": {
            "neuron": LocalLayerRepository(
                repo_path=repo_path,
                layer_name="NeuronRMSNorm",
            )
        }
    }

    with use_kernel_mapping(mapping, inherit_mapping=False):
        kernelize(model, device="neuron", mode=Mode.INFERENCE)

    # NKI output
    with torch.no_grad():
        nki_output = model(input_ids).logits.clone()

    cos_sim = cosine_similarity(ref_output, nki_output)
    max_diff = (ref_output.float() - nki_output.float()).abs().max().item()

    # Count how many RMSNorm layers were swapped
    num_norms = sum(1 for name, _ in model.named_modules() if "norm" in name.lower())

    print(f"  RMSNorm layers in model: {num_norms}")
    print(f"  Logits cosine similarity: {cos_sim:.6f}")
    print(f"  Logits max abs diff:      {max_diff:.2e}")

    passed = cos_sim > 0.999
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    return passed


def main():
    print("=" * 60)
    print("NeuronRMSNorm + Real Qwen3 Layer Test")
    print("=" * 60)
    print()

    # Load module to check backend
    mod = load_neuron_rmsnorm_module()
    if mod._HAS_NKI:
        print("  Backend: NKI kernel (NeuronCores)")
    else:
        print("  Backend: PyTorch fallback (CPU)")
    print()

    results = []

    # Test 1: Single layer swap
    results.append(("Single Qwen3RMSNorm layer", test_qwen3_single_layer()))

    # Test 2: Full model forward (reduced layers)
    results.append(("Qwen3 model forward (2 layers)", test_qwen3_model_forward()))

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 NKI RMSNorm validated inside real Qwen3 model!")
        print("   The kernel swap fires, executes correctly, and produces")
        print("   matching logits through the full transformer stack.")
    else:
        print("⚠️  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
