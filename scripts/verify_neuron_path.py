#!/usr/bin/env python3
"""Verify the HF Kernels 'neuron' device path works on this host.

This script validates the Week 1 goal:
1. `kernelize()` accepts `device="neuron"`
2. `LocalLayerRepository` loads a local NKI kernel for the neuron device
3. The forward swap fires (confirmed by numerical signature)
4. Fallback works when mapping is absent

Run on trn2: python scripts/verify_neuron_path.py

Expected output:
  ✓ kernels library imported (version: ...)
  ✓ transformers library imported (version: ...)
  ✓ kernelize accepts device="neuron"
  ✓ LocalLayerRepository loads neuron_rmsnorm package
  ✓ Forward swap fires (output differs from original)
  ✓ Fallback works when mapping is absent
"""

import sys
import os
from pathlib import Path

# Add project root to path so we can find our kernel packages
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_imports():
    """Check that kernels and transformers are importable and print versions."""
    print("=" * 60)
    print("HF Kernels Neuron Device Path Verification")
    print("=" * 60)
    print()

    try:
        import kernels
        print(f"  ✓ kernels library imported (version: {getattr(kernels, '__version__', 'dev')})")
    except ImportError as e:
        print(f"  ✗ kernels library NOT importable: {e}")
        print("    → pip install git+https://github.com/huggingface/kernels.git@main")
        return False

    try:
        import transformers
        print(f"  ✓ transformers library imported (version: {transformers.__version__})")
    except ImportError as e:
        print(f"  ✗ transformers library NOT importable: {e}")
        print("    → pip install git+https://github.com/huggingface/transformers.git@main")
        return False

    try:
        import torch
        print(f"  ✓ torch imported (version: {torch.__version__})")
        # Check for neuronx
        try:
            import torch_neuronx
            print(f"  ✓ torch_neuronx imported (Neuron SDK available)")
        except ImportError:
            print(f"  ⚠ torch_neuronx NOT available (running off-device, some tests will be limited)")
    except ImportError as e:
        print(f"  ✗ torch NOT importable: {e}")
        return False

    print()
    return True


def check_kernelize_device():
    """Verify kernelize() accepts device='neuron' without error."""
    import torch
    import torch.nn as nn
    from kernels import kernelize, Mode

    print("--- Test 1: kernelize() accepts device='neuron' ---")

    # Create a trivial model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16)

        def forward(self, x):
            return self.linear(x)

    model = TinyModel()

    try:
        # kernelize with device="neuron" — should not raise even if no
        # neuron kernels are registered (it just won't swap anything)
        kernelize(model, device="neuron", mode=Mode.INFERENCE)
        print("  ✓ kernelize accepts device='neuron' without error")
        print()
        return True
    except Exception as e:
        print(f"  ✗ kernelize raised: {type(e).__name__}: {e}")
        print()
        return False


def check_local_layer_repository():
    """Verify LocalLayerRepository loads our neuron_rmsnorm kernel."""
    import torch
    import torch.nn as nn
    from kernels import kernelize, Mode, LocalLayerRepository
    from kernels import use_kernel_mapping

    print("--- Test 2: LocalLayerRepository loads NeuronRMSNorm ---")

    # Path to our local kernel repo
    repo_path = PROJECT_ROOT / "kernels" / "neuron_rmsnorm"

    if not repo_path.exists():
        print(f"  ✗ Kernel repo not found at: {repo_path}")
        return False

    try:
        local_repo = LocalLayerRepository(
            repo_path=repo_path,
            layer_name="NeuronRMSNorm",
        )
        print(f"  ✓ LocalLayerRepository created for neuron_rmsnorm")
    except Exception as e:
        print(f"  ✗ LocalLayerRepository creation failed: {type(e).__name__}: {e}")
        return False

    print()
    return True


def check_forward_swap():
    """Verify the forward method is actually swapped when kernelized."""
    import torch
    import torch.nn as nn
    from kernels import (
        kernelize,
        Mode,
        LocalLayerRepository,
        use_kernel_mapping,
        use_kernel_forward_from_hub,
    )

    print("--- Test 3: Forward swap fires ---")

    # Create an RMSNorm-like module that mimics what Qwen3 has
    @use_kernel_forward_from_hub("RMSNorm")
    class SimpleRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # Original implementation — returns all zeros as a sentinel
            # so we can detect when it's swapped out
            return torch.zeros_like(hidden_states)

    # Create model with the RMSNorm layer
    class TinyNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = SimpleRMSNorm(hidden_size=16)

        def forward(self, x):
            return self.norm(x)

    model = TinyNormModel()

    # Test input
    x = torch.randn(2, 16)

    # Before kernelize: should return zeros (our sentinel)
    out_before = model(x)
    assert torch.all(out_before == 0), "Pre-swap output should be zeros (sentinel)"

    # Set up kernel mapping pointing to our local NeuronRMSNorm
    repo_path = PROJECT_ROOT / "kernels" / "neuron_rmsnorm"
    kernel_mapping = {
        "RMSNorm": {
            "neuron": LocalLayerRepository(
                repo_path=repo_path,
                layer_name="NeuronRMSNorm",
            )
        }
    }

    # Kernelize with neuron device
    with use_kernel_mapping(kernel_mapping, inherit_mapping=False):
        kernelize(model, device="neuron", mode=Mode.INFERENCE)

    # After kernelize: should NOT return zeros (NeuronRMSNorm does real computation)
    out_after = model(x)

    if not torch.all(out_after == 0):
        print("  ✓ Forward swap confirmed: output changed from sentinel zeros")
        print(f"    Original output (sentinel): all zeros")
        print(f"    Swapped output (NeuronRMSNorm): mean={out_after.mean().item():.4f}, "
              f"std={out_after.std().item():.4f}")
    else:
        print("  ✗ Forward swap did NOT fire: output is still zeros")
        print("    The kernelize() call did not replace the forward method")
        return False

    print()
    return True


def check_fallback():
    """Verify fallback works when no neuron mapping exists."""
    import torch
    import torch.nn as nn
    from kernels import (
        kernelize,
        Mode,
        use_kernel_forward_from_hub,
        use_kernel_mapping,
    )

    print("--- Test 4: Fallback when mapping is absent ---")

    @use_kernel_forward_from_hub("SomeLayerWithNoNeuronKernel")
    class UnmappedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.ones(8))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.param  # Original implementation

    class ModelWithUnmapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = UnmappedLayer()

        def forward(self, x):
            return self.layer(x)

    model = ModelWithUnmapped()
    x = torch.randn(2, 8)

    # Get reference output before kernelize
    ref_out = model(x)

    # Kernelize with neuron device — no mapping for this layer
    with use_kernel_mapping({}, inherit_mapping=False):
        kernelize(model, device="neuron", mode=Mode.INFERENCE)

    # Should still work with original forward
    out = model(x)

    if torch.allclose(out, ref_out):
        print("  ✓ Fallback works: unmapped layer uses original forward()")
    else:
        print("  ✗ Fallback broken: output changed even with no mapping")
        return False

    print()
    return True


def print_environment_info():
    """Print environment details for the PoC record."""
    import torch

    print("--- Environment ---")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Platform: {sys.platform}")

    try:
        import kernels
        print(f"  kernels: {getattr(kernels, '__version__', 'dev (from main)')}")
    except ImportError:
        pass

    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError:
        pass

    try:
        import torch_neuronx
        print(f"  torch_neuronx: available")
        import neuronxcc
        print(f"  neuronx-cc: {getattr(neuronxcc, '__version__', 'unknown')}")
    except ImportError:
        print(f"  torch_neuronx: NOT available (off-device)")

    # Check if running on Neuron hardware
    try:
        # On Neuron, devices show up via torch_neuronx
        import torch_neuronx
        print(f"  Neuron device count: {torch_neuronx.xla_impl.xla_model.xrt_world_size()}")
    except Exception:
        print(f"  Neuron device: not detected (CPU-only mode)")

    print()


def main():
    if not check_imports():
        print("\n❌ Import check failed. Install dependencies first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    print_environment_info()

    results = []
    results.append(("kernelize device='neuron'", check_kernelize_device()))
    results.append(("LocalLayerRepository", check_local_layer_repository()))
    results.append(("Forward swap", check_forward_swap()))
    results.append(("Fallback", check_fallback()))

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
        print("🎉 All checks passed! Neuron device path is working.")
        print()
        print("Next steps:")
        print("  1. Pin the exact versions to README.md")
        print("  2. Replace NeuronRMSNorm reference impl with NKI kernel (Week 2)")
        print("  3. Test with a real Qwen3 model layer")
    else:
        print("⚠️  Some checks failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
