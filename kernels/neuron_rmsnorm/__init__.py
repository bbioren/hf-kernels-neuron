"""NKI RMSNorm kernel for Neuron (Trainium/Inferentia).

This is the local kernel repo for developing and testing the NKI RMSNorm
before publishing to the Hub. Used via LocalLayerRepository.

Layout follows Hub kernel requirements:
  kernels/neuron_rmsnorm/
    __init__.py       <- this file (exports layers module)
    layers.py         <- NeuronRMSNorm layer class
    nki_rmsnorm.py    <- NKI kernel implementation (from nki_samples)
"""

from . import layers

__all__ = ["layers"]
