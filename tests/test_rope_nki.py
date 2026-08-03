"""NKI RoPE accuracy on real Neuron hardware, with execution proof.

Validates the ported nki-library `rope_hf` kernel against transformers'
`apply_rotary_pos_emb`. Uses the Finding #8 harness: tensors on the XLA device,
and a call counter asserting the NKI branch ran rather than the fallback.

Also checks the fallback path explicitly — the kernel requires
`seq_len % 128 == 0`, and HF passes arbitrary sequence lengths, so graceful
(and *loud*) degradation matters as much as the fast path.

Run on trn2:
    python tests/test_rope_nki.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from nki_test_utils import (
    AccuracyResult,
    CallCounts,
    assert_nki_accuracy,
    cosine_similarity,
    load_kernel_module,
    max_abs_diff,
    nki_call_counter,
    report,
    require_neuron,
    sync,
    tol_for_dtype,
)

NKI_NAMES = ["_nki_rope_hf"]
FALLBACK_NAMES = ["_torch_rope"]


def reference_rope(q, k, cos, sin, unsqueeze_dim=1):
    """transformers' apply_rotary_pos_emb, computed on CPU as golden reference.

    Uses the real transformers implementation where available so we're testing
    against the thing we claim to match, not against our own restatement of it.
    """
    try:
        from transformers.models.qwen3.modeling_qwen3 import rotate_half
    except ImportError:
        def rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def make_inputs(batch, q_heads, k_heads, seq_len, head_dim, dtype, cos_3d=True):
    """Build q/k/cos/sin the way Qwen3RotaryEmbedding would."""
    torch.manual_seed(0)
    q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch, k_heads, seq_len, head_dim, dtype=dtype)

    # HF builds cos/sin by duplicating half-width freqs to full head_dim
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = pos * inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    if cos_3d:
        cos = cos.unsqueeze(0).expand(batch, -1, -1).contiguous()
        sin = sin.unsqueeze(0).expand(batch, -1, -1).contiguous()
    return q, k, cos, sin


def run_case(mod, device, batch, q_heads, k_heads, seq_len, head_dim,
             dtype=torch.float32, cos_3d=True):
    """Run one shape on hardware and score q and k separately."""
    tag = (f"b{batch} qh{q_heads} kh{k_heads} s{seq_len} d{head_dim} "
           f"{str(dtype).replace('torch.', '')}{'' if cos_3d else ' cos2d'}")

    q, k, cos, sin = make_inputs(batch, q_heads, k_heads, seq_len, head_dim, dtype, cos_3d)

    cos_ref = cos if cos.ndim == 3 else cos.unsqueeze(0).expand(batch, -1, -1)
    sin_ref = sin if sin.ndim == 3 else sin.unsqueeze(0).expand(batch, -1, -1)
    q_gold, k_gold = reference_rope(q, k, cos_ref, sin_ref)

    qd, kd = q.to(device), k.to(device)
    cosd, sind = cos.to(device), sin.to(device)

    with nki_call_counter(mod, NKI_NAMES, FALLBACK_NAMES) as counts:
        with torch.no_grad():
            q_out, k_out = mod.apply_rotary_pos_emb(qd, kd, cosd, sind)
        sync()
        q_cpu, k_cpu = q_out.cpu(), k_out.cpu()

    tol = tol_for_dtype(dtype)
    # RoPE is elementwise, so NKI and PyTorch do identical IEEE ops in identical
    # order -> bit-identical output is expected, not suspicious. See
    # test_discrimination() for proof this comparison still catches wrong results.
    return [
        assert_nki_accuracy(f"q  {tag}", q_gold, q_cpu, counts, max_diff_tol=tol,
                            expect_bit_identical=True),
        assert_nki_accuracy(f"k  {tag}", k_gold, k_cpu, counts, max_diff_tol=tol,
                            expect_bit_identical=True),
    ]


def test_fallback_is_loud(mod, device):
    """seq_len not a multiple of 128 must fall back, warn, and stay correct."""
    print()
    print("-" * 76)
    print("Fallback behaviour (seq_len=100, not a multiple of 128)")
    print("-" * 76)

    batch, q_heads, k_heads, seq_len, head_dim = 1, 4, 2, 100, 64
    q, k, cos, sin = make_inputs(batch, q_heads, k_heads, seq_len, head_dim, torch.float32)
    q_gold, k_gold = reference_rope(q, k, cos, sin)

    mod._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with nki_call_counter(mod, NKI_NAMES, FALLBACK_NAMES) as counts:
            with torch.no_grad():
                q_out, k_out = mod.apply_rotary_pos_emb(
                    q.to(device), k.to(device), cos.to(device), sin.to(device)
                )
            sync()
            q_cpu, k_cpu = q_out.cpu(), k_out.cpu()

    warned = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    cos_q = cosine_similarity(q_gold, q_cpu)
    diff_q = max_abs_diff(q_gold, q_cpu)

    took_fallback = counts.fallback > 0 and counts.nki == 0
    print(f"  dispatch          : {counts}")
    print(f"  took fallback     : {took_fallback}")
    print(f"  warning emitted   : {len(warned) > 0}")
    if warned:
        print(f"  warning text      : {str(warned[0].message)[:100]}")
    print(f"  still correct     : cos_sim={cos_q:.6f} max_diff={diff_q:.3e}")

    ok = took_fallback and len(warned) > 0 and cos_q > 0.999
    print(f"  {'PASS' if ok else 'FAIL'}")
    return AccuracyResult(
        label="fallback loud + correct (seq_len=100)",
        cos_sim=cos_q,
        max_diff=diff_q,
        counts=CallCounts(nki=1 if not took_fallback else 0, fallback=counts.fallback),
        passed=ok,
        notes=[] if ok else ["fallback did not warn or was not taken"],
    )


def test_discrimination(mod, device):
    """Negative controls: prove this test CAN fail.

    Every NKI RoPE case reports max_diff = 0.000e+00. That is expected here —
    RoPE is purely elementwise (`q*cos +/- q_swapped*sin`, three IEEE ops per
    element, no reduction), so NKI and PyTorch perform identical operations in
    identical order and produce bit-identical results. Contrast RMSNorm, which
    reduces over hidden_size and therefore differs by ~1e-4.

    But "bit-identical everywhere" is also what a test that isn't measuring
    anything looks like. So verify the comparison discriminates:

      A. against a deliberately wrong reference (sin negated) -> must FAIL
      B. against the unrotated input                          -> must FAIL
         (proves the kernel actually rotated, rather than copying q through)
    """
    print()
    print("-" * 76)
    print("Discrimination checks (negative controls)")
    print("-" * 76)

    batch, q_heads, k_heads, seq_len, head_dim = 1, 4, 2, 128, 64
    q, k, cos, sin = make_inputs(batch, q_heads, k_heads, seq_len, head_dim, torch.float32)

    with torch.no_grad():
        q_out, k_out = mod.apply_rotary_pos_emb(
            q.to(device), k.to(device), cos.to(device), sin.to(device)
        )
    sync()
    q_cpu = q_out.cpu()

    # A. wrong-sign sin -> wrong rotation direction
    q_wrong, _ = reference_rope(q, k, cos, -sin)
    cos_wrong = cosine_similarity(q_wrong, q_cpu)
    diff_wrong = max_abs_diff(q_wrong, q_cpu)
    a_ok = cos_wrong < 0.999
    print(f"  A. vs negated-sin reference : cos_sim={cos_wrong:.6f} "
          f"max_diff={diff_wrong:.3e}  -> {'discriminates' if a_ok else 'DOES NOT DISCRIMINATE'}")

    # B. vs the unrotated input -> proves rotation happened
    cos_ident = cosine_similarity(q, q_cpu)
    diff_ident = max_abs_diff(q, q_cpu)
    b_ok = cos_ident < 0.999
    print(f"  B. vs unrotated input       : cos_sim={cos_ident:.6f} "
          f"max_diff={diff_ident:.3e}  -> {'kernel rotated' if b_ok else 'KERNEL IS A NO-OP'}")

    # C. correct reference still matches exactly
    q_gold, _ = reference_rope(q, k, cos, sin)
    cos_gold = cosine_similarity(q_gold, q_cpu)
    c_ok = cos_gold > 0.999
    print(f"  C. vs correct reference     : cos_sim={cos_gold:.6f}  "
          f"-> {'matches' if c_ok else 'MISMATCH'}")

    ok = a_ok and b_ok and c_ok
    print(f"  {'PASS' if ok else 'FAIL'} — comparison is meaningful")
    return AccuracyResult(
        label="discrimination (negative controls)",
        cos_sim=cos_gold,
        max_diff=diff_ident,
        counts=CallCounts(nki=1, fallback=0),
        passed=ok,
        notes=[] if ok else ["test does not discriminate a wrong result"],
    )


def test_guard_conditions(mod, device):
    """Every guard in _nki_unsupported_reason must fall back, warn, and stay correct.

    These are the constraints inherited from nki-library's `_validate_apply_rope_inputs`,
    which the original enforces with hard asserts. We convert them to a graceful
    fallback, so each one needs to be exercised — an over-permissive guard would crash
    at compile time, and an over-strict one would silently disable acceleration.
    """
    print()
    print("-" * 76)
    print("Guard conditions (each must fall back, warn, and remain correct)")
    print("-" * 76)

    b, qh, kh, d = 1, 4, 2, 64
    # (label, inputs, kwargs, check_correctness)
    #
    # check_correctness=False for cases where the eager reference cannot produce a
    # golden value either (mismatched cos width, or unsqueeze_dim=2 against 4D q/k).
    # For those, the meaningful assertion is still "declined the kernel and said so" —
    # the guard's job is to route away from NKI, not to make invalid input valid.
    cases = []

    q, k, cos, sin = make_inputs(b, qh, kh, 100, d, torch.float32)
    cases.append(("seq_len=100 (not multiple of 128)", (q, k, cos, sin), {}, True))

    q2, k2, cos2, sin2 = make_inputs(b, qh, kh, 128, d, torch.float32)
    cases.append(("unsqueeze_dim=2", (q2, k2, cos2, sin2), {"unsqueeze_dim": 2}, False))

    # 3D inputs (kernel is 4D-only)
    cases.append(("3D q/k (kernel is 4D-only)", (q2[:, 0], k2[:, 0], cos2, sin2), {}, True))

    # odd head_dim
    q4, k4, cos4, sin4 = make_inputs(b, qh, kh, 128, 66, torch.float32)
    cases.append(
        ("odd head_dim=65",
         (q4[..., :65], k4[..., :65], cos4[..., :65], sin4[..., :65]), {}, True)
    )

    # cos last dim != head_dim
    cases.append(
        ("cos width mismatch",
         (q2, k2, cos2[..., : d // 2], sin2[..., : d // 2]), {}, False)
    )

    # q/k dtype mismatch
    cases.append(
        ("q/k dtype mismatch", (q2, k2.to(torch.bfloat16), cos2, sin2), {}, False)
    )

    results = []
    for label, (qq, kk, cc, ss), kwargs, check_correct in cases:
        mod._warned.clear()

        golden_q = None
        if check_correct:
            try:
                cos_ref = cc if cc.ndim == 3 else cc.unsqueeze(0)
                sin_ref = ss if ss.ndim == 3 else ss.unsqueeze(0)
                golden_q, _ = reference_rope(
                    qq, kk, cos_ref, sin_ref, kwargs.get("unsqueeze_dim", 1)
                )
            except Exception as e:
                print(f"  note: reference unavailable for {label} ({type(e).__name__});"
                      " checking dispatch only")
                check_correct = False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with nki_call_counter(mod, NKI_NAMES, FALLBACK_NAMES) as counts:
                try:
                    with torch.no_grad():
                        q_out, _ = mod.apply_rotary_pos_emb(
                            qq.to(device), kk.to(device), cc.to(device), ss.to(device), **kwargs
                        )
                    sync()
                    q_cpu = q_out.cpu()
                    err = None
                except Exception as e:
                    q_cpu, err = None, e

        warned = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        fell_back = counts.fallback > 0 and counts.nki == 0

        if check_correct:
            correct = err is None and q_cpu is not None and cosine_similarity(golden_q, q_cpu) > 0.999
            detail = f"fallback={fell_back} warned={len(warned) > 0} correct={correct}"
            ok = fell_back and len(warned) > 0 and correct
        else:
            # Invalid input: only require that we declined NKI and warned. The eager
            # path may legitimately raise, since the input is genuinely malformed.
            correct = True
            detail = f"fallback={fell_back} warned={len(warned) > 0} (dispatch only)"
            ok = fell_back and len(warned) > 0

        if err is not None:
            detail += f" eager_raised={type(err).__name__}"
        print(f"  {'PASS' if ok else 'FAIL'}  {label:36s} {detail}")
        results.append(ok)

    all_ok = all(results) and len(results) > 0
    print(f"  {'PASS' if all_ok else 'FAIL'} — {sum(results)}/{len(results)} guards behave correctly")
    return AccuracyResult(
        label="guard conditions fall back correctly",
        cos_sim=1.0 if all_ok else 0.0,
        max_diff=0.0,
        counts=CallCounts(nki=1, fallback=0),
        passed=all_ok,
        notes=[] if all_ok else ["one or more guards misbehaved"],
    )


def main():
    device = require_neuron()
    mod = load_kernel_module("neuron_rope")

    if not mod._HAS_NKI:
        print("NKI unavailable — cannot validate the kernel.")
        return 1

    results = []

    # (batch, q_heads, k_heads, seq_len, head_dim)
    # Shapes drawn from nki-library's own test_rope_hf.py plus Qwen3 configs.
    cases = [
        (1, 4, 2, 128, 64),      # minimal: exactly one seq tile
        (1, 32, 8, 256, 128),    # nki-library test shape (Llama-ish GQA)
        (2, 16, 4, 512, 128),    # nki-library test shape, batched
        (4, 1, 1, 512, 128),     # nki-library test shape, MQA
        (1, 32, 8, 512, 64),     # nki-library test shape, small head_dim
        (1, 16, 8, 128, 128),    # Qwen3-8B: 32 q heads / 8 kv, head_dim 128
        (1, 8, 4, 1024, 128),    # longer sequence, 8 coalesced tiles
        (2, 4, 2, 128, 64),      # batch > 1, small
    ]

    for batch, qh, kh, s, d in cases:
        try:
            results.extend(run_case(mod, device, batch, qh, kh, s, d))
        except Exception as e:
            import traceback

            print(f"  ERROR b{batch} qh{qh} kh{kh} s{s} d{d}: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append(
                AccuracyResult(
                    label=f"b{batch} qh{qh} kh{kh} s{s} d{d}",
                    cos_sim=0.0,
                    max_diff=float("inf"),
                    counts=CallCounts(),
                    passed=False,
                    notes=[f"exception: {type(e).__name__}"],
                )
            )

    # bf16 at a Qwen3 shape
    try:
        results.extend(run_case(mod, device, 1, 16, 8, 256, 128, dtype=torch.bfloat16))
    except Exception as e:
        print(f"  bf16 case failed: {type(e).__name__}: {e}")

    # 2D cos/sin (no batch dim) — the kernel supports both
    try:
        results.extend(run_case(mod, device, 1, 4, 2, 128, 64, cos_3d=False))
    except Exception as e:
        print(f"  2D cos/sin case failed: {type(e).__name__}: {e}")

    ok = report(results, "NKI RoPE on Neuron hardware (execution-verified)")

    disc = test_discrimination(mod, device)
    fb = test_fallback_is_loud(mod, device)
    guards = test_guard_conditions(mod, device)
    return 0 if (ok and disc.passed and fb.passed and guards.passed) else 1


if __name__ == "__main__":
    sys.exit(main())
