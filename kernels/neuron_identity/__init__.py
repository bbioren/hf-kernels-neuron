"""Minimal NKI identity kernel for Week 1 proof-of-concept.

This is the simplest possible NKI kernel — an identity-plus-scale operation.
Its purpose is to prove the kernelize() → NKI kernel execution path works
on Trainium before tackling real kernels like RMSNorm.

The kernel multiplies input by a scalar (default 1.0) so the output is
numerically distinguishable from a plain identity when scale != 1.0.
"""

from . import layers

__all__ = ["layers"]
