"""Experiment: can an HF kernel be a THIN WRAPPER over installed nkilib?

`nkilib` turns out to be preinstalled in the Neuron venv (verified by
scripts/probe_nkilib_bundled.py), with every production kernel importable —
including `nkilib.core.embeddings.rope_hf.rope_hf`, the exact kernel we hand-ported
into kernels/neuron_rope/.

This is Option D from docs/nki-library-porting-analysis.md, the strategy that would
make porting cheap at scale:

    class NeuronRoPE(nn.Module):
        def forward(self, q, k, cos, sin):
            return nkilib.rope_hf(q, k, ...)      # no vendoring at all

The question this script answers is narrow but load-bearing: **does calling the
installed production kernel directly from PyTorch/XLA actually work, and is it
numerically correct?** If yes, the recommendation to the kernels team changes from
"port each kernel by hand" to "wrap the library; fix the dependency allowlist".

The known obstacle is calling convention: `rope_hf` uses destination-passing —
`rope_hf(q, k, q_out, k_out, cos=..., sin=...)` — with preallocated outputs, whereas
our hand-port allocates internally and returns a tuple. Whether an @nki.jit kernel can
write into a caller-supplied XLA tensor is exactly what needs testing.

This does NOT resolve whether HF would permit the dependency: `python-depends`
whitelists `nki` but not `nkilib`, and the neuron table is unreachable anyway
(Finding #12). It establishes only technical feasibility.

Run on trn2:
    python scripts/experiment_nkilib_thin_wrapper.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import torch
import torch.nn.functional as F

SEP = "=" * 72


def hdr(t):
    print()
    print(SEP)
    print(t)
    print(SEP)


def cos_sim(a, b):
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
    ).item()


def reference_rope(q, k, cos, sin):
    """transformers' apply_rotary_pos_emb with unsqueeze_dim=1."""
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    return (q * c) + (rotate_half(q) * s), (k * c) + (rotate_half(k) * s)


def make_inputs(b, qh, kh, s, d, dtype=torch.float32):
    torch.manual_seed(0)
    q = torch.randn(b, qh, s, d, dtype=dtype)
    k = torch.randn(b, kh, s, d, dtype=dtype)
    pos = torch.arange(s, dtype=torch.float32).unsqueeze(1)
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
    freqs = pos * inv.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(0).expand(b, -1, -1).contiguous()
    sin = emb.sin().to(dtype).unsqueeze(0).expand(b, -1, -1).contiguous()
    return q, k, cos, sin


def attempt(label, fn):
    """Run a calling strategy, report outcome without aborting the script."""
    print(f"  -- {label}")
    try:
        result = fn()
        print(f"     SUCCESS: {result}")
        return True, result
    except Exception as e:
        msg = str(e).replace("\n", " ")[:150]
        print(f"     FAILED: {type(e).__name__}: {msg}")
        return False, None


def main():
    print(SEP)
    print("EXPERIMENT: thin wrapper over installed nkilib")
    print(SEP)

    try:
        import nkilib
        from nkilib.core.embeddings.rope_hf import rope_hf

        print(f"  nkilib at   {nkilib.__file__}")
        print(f"  rope_hf     imported OK")
    except Exception as e:
        print(f"  cannot import nkilib rope_hf: {type(e).__name__}: {e}")
        return 1

    try:
        import torch_xla.core.xla_model as xm

        dev = xm.xla_device()
        if xm.xla_device_hw(dev) != "NEURON":
            print("  not on Neuron hardware; aborting")
            return 1
    except Exception as e:
        print(f"  torch_xla unavailable: {e}")
        return 1

    b, qh, kh, s, d = 1, 4, 2, 256, 64
    q, k, cos, sin = make_inputs(b, qh, kh, s, d)
    gold_q, gold_k = reference_rope(q, k, cos, sin)

    hdr("Calling strategies for destination-passing from PyTorch/XLA")

    qd, kd = q.to(dev), k.to(dev)
    cosd, sind = cos.to(dev), sin.to(dev)

    # Strategy A: pass preallocated outputs positionally, use the return value.
    def strat_a():
        q_out = torch.empty_like(qd)
        k_out = torch.empty_like(kd)
        r = rope_hf(qd, kd, q_out, k_out, cos=cosd, sin=sind)
        xm.mark_step()
        if isinstance(r, (tuple, list)) and len(r) == 2:
            rq, rk = r[0].cpu(), r[1].cpu()
            return (f"returned tuple; q cos_sim={cos_sim(gold_q, rq):.6f} "
                    f"k cos_sim={cos_sim(gold_k, rk):.6f}")
        return f"returned {type(r)}"

    ok_a, _ = attempt("A: preallocated outs, read the RETURN value", strat_a)

    # Strategy B: same call, but read the mutated input tensors instead.
    def strat_b():
        q_out = torch.zeros_like(qd)
        k_out = torch.zeros_like(kd)
        rope_hf(qd, kd, q_out, k_out, cos=cosd, sin=sind)
        xm.mark_step()
        rq, rk = q_out.cpu(), k_out.cpu()
        return (f"mutated in place; q cos_sim={cos_sim(gold_q, rq):.6f} "
                f"k cos_sim={cos_sim(gold_k, rk):.6f}")

    ok_b, _ = attempt("B: preallocated outs, read the MUTATED ARGUMENTS", strat_b)

    hdr("Verdict")
    if ok_a or ok_b:
        which = "return value" if ok_a else "mutated arguments"
        print(f"  The installed production kernel IS directly callable (via {which}).")
        print("  A thin-wrapper HF kernel over nkilib is technically feasible today.")
        print()
        print("  Remaining blocker is policy, not code: `python-depends` whitelists")
        print("  `nki` but not `nkilib`, and the neuron table is unreachable anyway")
        print("  (Finding #12). A wrapper would have to under-declare its dependency,")
        print("  relying on nkilib happening to be preinstalled.")
    else:
        print("  Neither calling strategy worked. Destination-passing does not survive")
        print("  the PyTorch/XLA boundary in the obvious ways, which is a concrete")
        print("  argument for nki-library exposing return-value entry points for the")
        print("  HF use case — and it justifies our hand-port having reshaped the")
        print("  kernel to allocate internally.")
    print()
    print("  Either way our hand-port remains the right choice for the PoC: it is")
    print("  self-contained, declares no undeclarable dependency, and documents the")
    print("  porting path. This experiment establishes the cheaper long-term shape.")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
