"""Test: Can we create a Hub-compatible directory layout and load a Neuron kernel from it?

If this works, it means you can publish an NKI kernel to the Hub by simply
uploading Python files in the right directory structure. No kernel-builder needed.
"""
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from kernels import kernelize, Mode, LocalLayerRepository, use_kernel_mapping, use_kernel_forward_from_hub


def create_hub_layout(base_dir: Path):
    """Create a Hub-compatible kernel layout for a neuron kernel."""
    # Hub layout: build/<variant_str>/<files>
    variant_dir = base_dir / "build" / "torch29-neuron-x86_64-linux"
    variant_dir.mkdir(parents=True)

    # __init__.py with kernel class + layers namespace
    init_content = '''
import torch
import torch.nn as nn

class NeuronRMSNorm(nn.Module):
    has_backward = False
    can_torch_compile = False

    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class layers:
    NeuronRMSNorm = NeuronRMSNorm
'''
    (variant_dir / "__init__.py").write_text(init_content)

    # metadata.json in the variant directory
    metadata = {
        "name": "neuron-rmsnorm",
        "id": "neuron_rmsnorm",
        "version": 0,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": {"type": "neuron"},
        "digest": {"algorithm": "sha256", "files": {}},
    }
    (variant_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Also put metadata.json at repo root (the loader checks here on fallback)
    (base_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return base_dir


def test_load_from_hub_layout():
    """Test loading a kernel from the Hub directory structure."""
    print("=" * 60)
    print("Test: Load NKI kernel from Hub-compatible directory layout")
    print("=" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmp:
        repo_path = create_hub_layout(Path(tmp))
        print(f"Created Hub layout at: {repo_path}")
        print(f"  metadata.json (root)")
        print(f"  build/torch29-neuron-x86_64-linux/__init__.py")
        print(f"  build/torch29-neuron-x86_64-linux/metadata.json")
        print()

        # Try loading via LocalLayerRepository
        try:
            repo = LocalLayerRepository(
                repo_path=repo_path,
                layer_name="NeuronRMSNorm",
            )
            print("  ✓ LocalLayerRepository created from Hub layout")
        except Exception as e:
            print(f"  ✗ Failed to create repo: {type(e).__name__}: {e}")
            return False

        # Try kernelizing with it
        @use_kernel_forward_from_hub("RMSNorm")
        class TestRMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(16))
                self.variance_epsilon = 1e-6

            def forward(self, x):
                return torch.zeros_like(x)  # sentinel

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = TestRMSNorm()

            def forward(self, x):
                return self.norm(x)

        model = Model()
        x = torch.randn(2, 16)

        out_before = model(x)

        mapping = {"RMSNorm": {"neuron": repo}}
        with use_kernel_mapping(mapping, inherit_mapping=False):
            kernelize(model, device="neuron", mode=Mode.INFERENCE)

        out_after = model(x)

        if not torch.allclose(out_before, out_after):
            print("  ✓ Forward swap works from Hub layout!")
            print()
            print("=" * 60)
            print("CONCLUSION: NKI kernels CAN be published to the Hub")
            print("=" * 60)
            print()
            print("What this means:")
            print("  - No kernel-builder compilation step needed")
            print("  - Just upload Python files in this structure:")
            print("    metadata.json (root)")
            print("    build/torch29-neuron-x86_64-linux/__init__.py")
            print("    build/torch29-neuron-x86_64-linux/metadata.json")
            print("  - The kernels library already parses 'neuron' variants")
            print("  - The variant resolver has Neuron backend support")
            print()
            print("Remaining question:")
            print("  Can huggingface_hub upload to a kernel-type repo")
            print("  without kernel-builder? (manual upload via API or git)")
            return True
        else:
            print("  ✗ Forward swap did NOT work")
            return False


if __name__ == "__main__":
    success = test_load_from_hub_layout()
    sys.exit(0 if success else 1)
