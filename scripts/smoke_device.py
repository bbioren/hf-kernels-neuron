"""Minimal device smoke test: can we get an XLA/Neuron device and run one op?

Used to distinguish "the Neuron runtime is unavailable" from "my kernel is broken".
Worth having as a standalone check, since a core-allocation failure surfaces as a
confusing error deep inside an unrelated stack trace.

Run on trn2:
    python scripts/smoke_device.py
"""

import os
import sys


def main():
    print("env NEURON_RT_* :",
          {k: v for k, v in os.environ.items() if k.startswith("NEURON_RT")} or "(none set)")
    try:
        import torch
        import torch_xla.core.xla_model as xm
    except Exception as e:
        print(f"import failed: {type(e).__name__}: {e}")
        return 1

    try:
        dev = xm.xla_device()
        hw = xm.xla_device_hw(dev)
        print(f"device = {dev}  hw = {hw}")
    except Exception as e:
        print(f"device acquisition FAILED: {type(e).__name__}: {e}")
        return 1

    try:
        t = torch.ones(4, 4).to(dev)
        out = (t * 2).cpu().sum().item()
        print(f"compute ok: sum = {out} (expect 32.0)")
        return 0 if out == 32.0 else 1
    except Exception as e:
        print(f"compute FAILED: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
