"""NKI RMSNorm kernel implementation.

Week 2: Port the RMSNorm kernel from nki_samples here.
This file will contain the actual NKI kernel that runs on NeuronCores.

Reference: aws-neuron/nki-samples repository, rmsnorm kernel.

The kernel signature should be:
    nki_rmsnorm_kernel(hidden_states, weight, epsilon) -> Tensor

It must be a pure function (no state), taking the input tensor,
the scale weight, and epsilon, and returning the normalized output.
"""

# Placeholder — will be filled in Week 2 on trn2 host with:
#
# import neuronxcc.nki as nki
# import neuronxcc.nki.language as nl
#
# @nki.jit
# def nki_rmsnorm_kernel(hidden_states, weight, epsilon):
#     ...


def nki_rmsnorm_kernel(hidden_states, weight, epsilon):
    """Placeholder for NKI RMSNorm kernel. Replace on trn2."""
    raise NotImplementedError(
        "NKI RMSNorm kernel requires Neuron hardware. "
        "Run on trn2 with neuronxcc installed."
    )
