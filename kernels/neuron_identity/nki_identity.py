"""NKI identity-scale kernel — the simplest possible NKI kernel.

This kernel just multiplies input by a scalar. It exists to validate:
1. nki.jit compilation works
2. Data flows in and out of NeuronCores correctly
3. The kernel integrates with the HF Kernel Hub forward-swap

On trn2, this will compile to a single NKI program that loads a tile,
multiplies by the scalar, and stores the result.
"""

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import torch

    @nki.jit
    def _nki_identity_scale_kernel(x_ptr, out_ptr, scale):
        """Minimal NKI kernel: out = x * scale.

        Processes one tile at a time. For the PoC this is intentionally
        simple — no tiling optimization, no multi-core parallelism.
        """
        # Get input shape
        # Partition dimension (P) is the first axis, free dimension (F) is the rest
        i_p = nl.arange(x_ptr.shape[0])[:, None]
        i_f = nl.arange(x_ptr.shape[1])[None, :]

        # Load tile from HBM to SBUF
        x_tile = nl.load(x_ptr[i_p, i_f])

        # Multiply by scale
        result = nl.multiply(x_tile, scale)

        # Store result back to HBM
        nl.store(out_ptr[i_p, i_f], value=result)

    def nki_identity_scale(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Wrapper that calls the NKI kernel with proper tensor setup.

        Args:
            x: Input tensor (any shape, will be reshaped to 2D for NKI)
            scale: Multiplicative scalar

        Returns:
            x * scale, computed on NeuronCore
        """
        original_shape = x.shape
        # NKI kernels work on 2D tiles: (partition_dim, free_dim)
        # Flatten everything except the last dim
        x_2d = x.reshape(-1, x.shape[-1])

        # Allocate output
        out_2d = torch.empty_like(x_2d)

        # Call NKI kernel
        _nki_identity_scale_kernel(x_2d, out_2d, scale)

        # Reshape back
        return out_2d.reshape(original_shape)

except ImportError:
    # Not on Neuron hardware — provide a stub that raises
    def nki_identity_scale(x, scale=1.0):
        raise RuntimeError(
            "NKI identity kernel requires Neuron hardware (torch_neuronx). "
            "Use the PyTorch fallback path instead."
        )
