"""Accuracy test: NeuronRMSNorm vs Qwen3RMSNorm.

Validates that our NKI kernel produces the same output as the reference
Qwen3RMSNorm implementation. Target: cosine similarity > 0.999.

Run on trn2:
    python tests/test_rmsnorm_accuracy.py

This test exercises:
1. NKI kernel (on Neuron hardware) or PyTorch fallback (on CPU)
2. Various input shapes (different seq_len, hidden_size, batch)
3. Comparison against the actual Qwen3RMSNorm from transformers
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_neuron_rmsnorm_module():
    """Load the neuron_rmsnorm package from our local kernel directory."""
    kernel_path = PROJECT_ROOT / "kernels" / "neuron_rmsnorm" / "__init__.py"
    spec = importlib.util.spec_from_file_location("neuron_rmsnorm", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute max absolute difference between two tensors."""
    return (a.float() - b.float()).abs().max().item()


def get_qwen3_rmsnorm():
    """Get the Qwen3RMSNorm class from transformers."""
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        return Qwen3RMSNorm
    except ImportError:
        # Fall back to a manual implementation matching Qwen3's
        print("  (Qwen3RMSNorm not available, using manual reference)")

        class ReferenceRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * hidden_states.to(input_dtype)

        return ReferenceRMSNorm


def get_neuron_rmsnorm():
    """Load our NeuronRMSNorm kernel."""
    mod = load_neuron_rmsnorm_module()
    return mod.layers.NeuronRMSNorm


def test_accuracy(hidden_size, seq_len, batch_size=1, eps=1e-6, dtype=torch.float32):
    """Run a single accuracy comparison.

    Creates a Qwen3RMSNorm, applies our NeuronRMSNorm forward to the same
    module (simulating the kernelize swap), and compares outputs.
    """
    Qwen3RMSNorm = get_qwen3_rmsnorm()

    # Create the reference module with random weights
    ref_norm = Qwen3RMSNorm(hidden_size, eps=eps)
    # Randomize the weight (don't leave at ones — too easy)
    ref_norm.weight = nn.Parameter(torch.randn(hidden_size) * 0.5 + 1.0)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    # Reference output
    with torch.no_grad():
        ref_output = ref_norm(x)

    # NKI kernel output — simulate the kernelize swap by calling
    # NeuronRMSNorm.forward with self = ref_norm (the original module)
    mod = load_neuron_rmsnorm_module()
    neuron_forward = mod.layers.NeuronRMSNorm.forward

    with torch.no_grad():
        # Bind the NeuronRMSNorm forward to the ref_norm module
        # This is what kernelize() does — replaces forward, keeps the module
        nki_output = neuron_forward(ref_norm, x)

    # Compute metrics
    cos_sim = cosine_similarity(ref_output, nki_output)
    max_diff = max_abs_diff(ref_output, nki_output)
    allclose = torch.allclose(ref_output, nki_output, atol=1e-5, rtol=1e-3)

    return cos_sim, max_diff, allclose


def main():
    print("=" * 60)
    print("NeuronRMSNorm vs Qwen3RMSNorm Accuracy Test")
    print("=" * 60)
    print()

    # Check if NKI is available
    mod = load_neuron_rmsnorm_module()
    has_nki = mod._HAS_NKI
    if has_nki:
        print("  Backend: NKI kernel (NeuronCores)")
    else:
        print("  Backend: PyTorch fallback (CPU)")
    print()

    # Test configurations: (hidden_size, seq_len, batch_size)
    test_configs = [
        # Small — quick sanity check
        (64, 8, 1),
        (128, 16, 2),
        # Qwen3-like dimensions
        (896, 128, 1),       # Qwen3-0.5B hidden size
        (1536, 128, 1),      # Qwen3-1.7B hidden size
        (2048, 64, 2),       # Qwen3-4B hidden size
        (3584, 32, 1),       # Qwen3-8B hidden size
        # Edge cases
        (256, 1, 1),         # single token
        (512, 250, 1),       # non-power-of-2 seq_len (matches tutorial test)
    ]

    all_passed = True
    results = []

    for hidden_size, seq_len, batch_size in test_configs:
        cos_sim, max_diff, allclose = test_accuracy(
            hidden_size=hidden_size,
            seq_len=seq_len,
            batch_size=batch_size,
        )

        passed = cos_sim > 0.999
        status = "✓" if passed else "✗"
        results.append((hidden_size, seq_len, batch_size, cos_sim, max_diff, passed))

        if not passed:
            all_passed = False

        print(f"  {status} shape=({batch_size}, {seq_len}, {hidden_size})"
              f"  cos_sim={cos_sim:.6f}  max_diff={max_diff:.2e}"
              f"  allclose={allclose}")

    print()
    print("-" * 60)

    if all_passed:
        print("✓ ALL TESTS PASSED (cosine similarity > 0.999)")
    else:
        print("✗ SOME TESTS FAILED")
        for h, s, b, cos, diff, passed in results:
            if not passed:
                print(f"  FAILED: shape=({b}, {s}, {h}) cos_sim={cos:.6f}")

    print()

    # Summary stats
    cos_sims = [r[3] for r in results]
    max_diffs = [r[4] for r in results]
    print(f"  Min cosine similarity: {min(cos_sims):.6f}")
    print(f"  Max absolute diff:     {max(max_diffs):.2e}")
    print()

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
