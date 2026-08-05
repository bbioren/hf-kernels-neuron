"""Minimal: does real compute work on the native stack with the CURRENT host runtime?

Deliberately does nothing else. The combined probe (probe_native_stack.py) reported
"Failed to initialize Neuron Runtime: status code 1", but `neuron-ls` showed the probe's OWN pid
holding /dev/neuron0 plus a second forked pid from the same run — two Neuron processes contending,
which is a known way to get an init failure that looks like a version incompatibility. An earlier
bare test in setup_native_venv.sh created a device tensor successfully.

So this settles the actual question, which decides whether the deb/ runtime+driver packages have to
be installed on the host (a change that would very likely break the existing XLA venv):

    can one clean process create a device tensor and run a matmul, on the production runtime?

Nothing else imported, nothing forked.

    /home/ubuntu/native_venv/bin/python scripts/probe_native_compute.py
"""

import sys

import torch


def main():
    print(f"torch {torch.__version__}")
    print(f"hasattr(torch, 'neuron') {hasattr(torch, 'neuron')}")
    print(f"privateuse1 {torch._C._get_privateuse1_backend_name()}")

    print("\ncreating device tensor ...", flush=True)
    a = torch.randn(256, 256, dtype=torch.bfloat16)
    b = torch.randn(256, 256, dtype=torch.bfloat16)
    ad = a.to("neuron")
    bd = b.to("neuron")
    print(f"  ad.device        {ad.device}")
    print(f"  ad.device.type   {ad.device.type}   <- this is what transformers' kernelize() reads")

    print("\nmatmul on device ...", flush=True)
    cd = torch.mm(ad, bd)
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()
    got = cd.cpu().float()
    ref = a.float() @ b.float()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    err = (got - ref).abs().max().item()
    print(f"  cos_sim {cos:.6f}   max_abs {err:.4f}")

    print("\nRESULT: compute works on the current host runtime. The deb/ packages are NOT")
    print("        required just to run — so the host Neuron runtime can be left alone and the")
    print("        existing XLA venv stays intact.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        print("\nIf this is a runtime/version error in a CLEAN process, the deb/ packages really are")
        print("required and the host runtime has to be replaced.")
        sys.exit(1)
