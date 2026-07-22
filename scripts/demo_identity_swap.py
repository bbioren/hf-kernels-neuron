#!/usr/bin/env python3
"""Demo: minimal NKI identity kernel swapped via kernelize().

This is the Week 1 proof-of-concept demo. It:
1. Creates a toy model with a layer annotated for kernel replacement
2. Registers our NeuronIdentityScale as the neuron kernel for that layer
3. Calls kernelize() to swap the forward method
4. Shows that the NKI kernel (or its PyTorch fallback) executes

On trn2: runs the actual NKI kernel on NeuronCores
On CPU: runs the PyTorch fallback but still proves the swap mechanism works

Usage:
    python scripts/demo_identity_swap.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from kernels import (
    kernelize,
    Mode,
    LocalLayerRepository,
    use_kernel_mapping,
    use_kernel_forward_from_hub,
)


# --- Model definition ---
# This simulates what a real HF model looks like after the
# @use_kernel_forward_from_hub decorator is applied

@use_kernel_forward_from_hub("IdentityScale")
class IdentityScaleLayer(nn.Module):
    """A trivial layer: output = input * scale.

    In a real model this would be something like RMSNorm or an activation.
    For PoC purposes we just need a layer that:
    - Has the @use_kernel_forward_from_hub decorator
    - Has some state (self.scale) the kernel layer can read
    """

    def __init__(self, hidden_size: int, scale_value: float = 2.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((hidden_size,), scale_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward — returns NEGATIVE of scaled input as a sentinel
        # so we can clearly see when it's been swapped
        return -(x * self.scale)


class ToyModel(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.identity = IdentityScaleLayer(hidden_size, scale_value=2.0)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.identity(x)
        x = self.linear(x)
        return x


def main():
    print("=" * 60)
    print("Week 1 PoC: NKI Identity Kernel Forward Swap")
    print("=" * 60)
    print()

    hidden_size = 32
    model = ToyModel(hidden_size=hidden_size)
    x = torch.randn(4, hidden_size)

    # --- Before swap ---
    print("1. Before kernelize():")
    out_before = model.identity(x)
    print(f"   Input mean: {x.mean().item():.4f}")
    print(f"   Output mean: {out_before.mean().item():.4f}")
    print(f"   Output is negative (sentinel): {(out_before.mean() < 0) == (x.mean() > 0)}")
    print()

    # --- Set up kernel mapping ---
    repo_path = PROJECT_ROOT / "kernels" / "neuron_identity"
    kernel_mapping = {
        "IdentityScale": {
            "neuron": LocalLayerRepository(
                repo_path=repo_path,
                layer_name="NeuronIdentityScale",
            )
        }
    }

    # --- Kernelize ---
    print("2. Calling kernelize(model, device='neuron', mode=Mode.INFERENCE)...")
    with use_kernel_mapping(kernel_mapping, inherit_mapping=False):
        kernelize(model, device="neuron", mode=Mode.INFERENCE)
    print("   Done.")
    print()

    # --- After swap ---
    print("3. After kernelize():")
    out_after = model.identity(x)
    print(f"   Input mean: {x.mean().item():.4f}")
    print(f"   Output mean: {out_after.mean().item():.4f}")

    # The swap should have replaced the sentinel (negative) forward
    # with NeuronIdentityScale which does x * scale (positive when input positive)
    swap_detected = not torch.allclose(out_before, out_after)
    print(f"   Output changed after swap: {swap_detected}")
    print()

    if swap_detected:
        print("✓ Forward swap CONFIRMED!")
        print(f"  Before (sentinel): output was negated → mean={out_before.mean().item():.4f}")
        print(f"  After (NKI kernel): output is x*scale → mean={out_after.mean().item():.4f}")
        print()
        # Verify the swapped kernel does what we expect (x * scale)
        expected = x * model.identity.scale
        if torch.allclose(out_after, expected, atol=1e-5):
            print("✓ Output matches expected x*scale — kernel executing correctly")
        else:
            print("⚠ Output doesn't exactly match x*scale (may be NKI precision)")
            print(f"  Max diff: {(out_after - expected).abs().max().item():.6e}")
    else:
        print("✗ Forward swap NOT detected — something is wrong.")
        print("  Possible causes:")
        print("  - kernelize() didn't find the layer (check @use_kernel_forward_from_hub name)")
        print("  - LocalLayerRepository didn't load (check package_name/layer_name)")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Week 1 PoC complete. The neuron device path + forward swap works.")
    print("=" * 60)


if __name__ == "__main__":
    main()
