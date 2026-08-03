"""Probe: does `use_kernels=True` reach the "neuron" device path on Trainium?

Week 3 investigation. The Week 3 goal is "confirm `use_kernels=True` alone triggers
the swaps on Neuron". Reading the source suggests it cannot, because device
resolution never yields "neuron". This script verifies that empirically.

What we're testing, in order:
  1. Is "neuron" a valid device key in the kernels library?  (source says yes)
  2. What does `model.device.type` actually report on the Neuron DLAMI?
  3. Does `hasattr(torch, "neuron")` fire?  (gates _has_neuron_ops)
  4. Are the Qwen3 kernel decorators actually applied at import time?
  5. What is registered in Qwen3Attention._hidden_kernels?
  6. Does the transformers `kernelize()` wrapper let us reach "neuron"?

Run on trn2:
    python scripts/probe_neuron_device_path.py
"""

import sys

import torch
import torch.nn as nn

SEP = "=" * 68


def hdr(title):
    print()
    print(SEP)
    print(title)
    print(SEP)


def probe_1_device_key_validity():
    """Is 'neuron' accepted as a device by the kernels library?"""
    hdr("1. Is 'neuron' a valid device in the kernels library?")

    from kernels import Device
    from kernels.layer.kernelize import _validate_device_type
    from kernels.layer.repos import DeviceRepos

    try:
        dev = Device(type="neuron")
        print(f"  Device(type='neuron')      -> OK: {dev!r}")
    except Exception as e:
        print(f"  Device(type='neuron')      -> FAILED: {type(e).__name__}: {e}")
        return False

    try:
        _validate_device_type("neuron")
        print("  _validate_device_type      -> OK ('neuron' is supported)")
    except Exception as e:
        print(f"  _validate_device_type      -> FAILED: {e}")
        return False

    try:
        repos = DeviceRepos.from_device(dev)
        print(f"  DeviceRepos.from_device    -> OK: {type(repos).__name__}")
    except Exception as e:
        print(f"  DeviceRepos.from_device    -> FAILED: {e}")
        return False

    return True


def probe_2_model_device_type():
    """What does model.device.type report? This is what transformers uses."""
    hdr("2. What device type does a model actually report?")

    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()

    print(f"  model.device               = {model.device}")
    print(f"  model.device.type          = '{model.device.type}'")
    print(f"  next(params).device.type   = '{next(model.parameters()).device.type}'")

    # This is exactly what kernels' _find_device does
    from kernels.layer.kernelize import _find_device

    resolved = _find_device(model)
    print(f"  kernels._find_device(model)= {resolved!r}")
    print()
    if resolved.type == "neuron":
        print("  => resolves to 'neuron'. The neuron mapping WOULD be selected.")
    else:
        print(f"  => resolves to '{resolved.type}', NOT 'neuron'.")
        print("     A 'neuron' mapping entry would be IGNORED on this path.")

    # Now try moving to an XLA device, which is how eager Neuron actually runs
    print()
    print("  --- after moving to XLA device (how Neuron eager actually runs) ---")
    try:
        import torch_xla.core.xla_model as xm

        xla_dev = xm.xla_device()
        print(f"  xm.xla_device()            = {xla_dev}")
        print(f"  xla_device.type            = '{xla_dev.type}'")
        model_xla = model.to(xla_dev)
        print(f"  model.device.type (on XLA) = '{model_xla.device.type}'")
        try:
            resolved_xla = _find_device(model_xla)
            print(f"  _find_device(model on XLA) = {resolved_xla!r}")
            from kernels.layer.kernelize import _validate_device_type

            try:
                _validate_device_type(resolved_xla.type)
                print(f"  _validate_device_type('{resolved_xla.type}') -> OK")
            except Exception as e:
                print(f"  _validate_device_type('{resolved_xla.type}') -> REJECTED: {e}")
        except Exception as e:
            print(f"  _find_device on XLA failed: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  torch_xla unavailable / failed: {type(e).__name__}: {e}")

    return model


def probe_3_torch_neuron_attr():
    """Does hasattr(torch, 'neuron') fire? Gates _has_neuron_ops()."""
    hdr("3. Does `hasattr(torch, 'neuron')` fire?")

    from kernels.layer.kernelize import _has_neuron_ops

    print(f"  hasattr(torch, 'neuron')   = {hasattr(torch, 'neuron')}")
    print(f"  _has_neuron_ops()          = {_has_neuron_ops()}")
    try:
        import torch_neuronx  # noqa: F401

        print("  torch_neuronx import       = OK")
        print(f"  hasattr(torch,'neuron') after torch_neuronx import = {hasattr(torch, 'neuron')}")
    except Exception as e:
        print(f"  torch_neuronx import       = FAILED: {type(e).__name__}: {e}")


def probe_4_qwen3_decorators():
    """Are the Qwen3 kernel decorators applied at import time?"""
    hdr("4. Are the Qwen3 kernel decorators applied?")

    from transformers.models.qwen3 import modeling_qwen3 as mq

    # RMSNorm: layer decorator sets kernel_layer_name on the CLASS
    rms_name = getattr(mq.Qwen3RMSNorm, "kernel_layer_name", None)
    print(f"  Qwen3RMSNorm.kernel_layer_name       = {rms_name!r}")

    # RoPE: func decorator replaces the module-level name with an nn.Module INSTANCE
    rope = mq.apply_rotary_pos_emb
    print(f"  type(apply_rotary_pos_emb)           = {type(rope).__name__}")
    print(f"  isinstance(..., nn.Module)           = {isinstance(rope, nn.Module)}")
    print(f"  .kernel_layer_name                   = {getattr(rope, 'kernel_layer_name', None)!r}")
    print(f"  type(...).kernel_layer_name          = {getattr(type(rope), 'kernel_layer_name', None)!r}")
    print(f"  .has_backward                        = {getattr(type(rope), 'has_backward', None)!r}")
    print(f"  .can_torch_compile                   = {getattr(type(rope), 'can_torch_compile', None)!r}")
    import inspect

    try:
        print(f"  forward signature                    = {inspect.signature(type(rope).forward)}")
    except Exception as e:
        print(f"  forward signature failed: {e}")


def probe_5_hidden_kernels(model):
    """What is registered in _hidden_kernels on the attention modules?"""
    hdr("5. What lands in Qwen3Attention._hidden_kernels?")

    found = 0
    for name, module in model.named_modules():
        hk = getattr(module, "_hidden_kernels", None)
        if hk:
            found += 1
            if found <= 2:
                print(f"  {name}:")
                for k, v in hk.items():
                    print(f"      '{k}' -> {type(v).__name__} "
                          f"(kernel_layer_name={getattr(type(v), 'kernel_layer_name', None)!r})")
    print(f"  modules with _hidden_kernels: {found}")

    # Confirm identity: is the registered Func the SAME object as the module-level one?
    from transformers.models.qwen3 import modeling_qwen3 as mq

    for _, module in model.named_modules():
        hk = getattr(module, "_hidden_kernels", None)
        if hk:
            for k, v in hk.items():
                same = v is mq.apply_rotary_pos_emb
                print(f"  '{k}' is the module-level apply_rotary_pos_emb object: {same}")
                if same:
                    print("      => the Func instance is PROCESS-GLOBAL, shared by all layers")
                    print("         and all models. An in-place swap is not per-model.")
            break


def probe_6_transformers_kernelize(model):
    """Does the transformers kernelize() wrapper let us reach 'neuron'?"""
    hdr("6. Can the transformers kernelize() wrapper reach 'neuron'?")

    from transformers.integrations import hub_kernels
    import inspect

    sig = inspect.signature(hub_kernels.kernelize)
    print(f"  transformers kernelize signature = {sig}")
    if "device" in sig.parameters:
        print("  => accepts a device override")
    else:
        print("  => NO device parameter. Device comes from model.device.type only.")

    from kernels import kernelize as kernels_kernelize

    sig2 = inspect.signature(kernels_kernelize)
    print(f"  kernels    kernelize signature   = {sig2}")
    print(f"  => kernels lib DOES accept device: {'device' in sig2.parameters}")


def main():
    print(SEP)
    print("PROBE: neuron device path for use_kernels=True")
    print(SEP)
    import kernels
    import transformers

    print(f"  kernels      {kernels.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  torch        {torch.__version__}")

    probe_1_device_key_validity()
    model = probe_2_model_device_type()
    probe_3_torch_neuron_attr()
    probe_4_qwen3_decorators()
    probe_5_hidden_kernels(model)
    probe_6_transformers_kernelize(model)

    print()
    print(SEP)
    print("PROBE COMPLETE")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
