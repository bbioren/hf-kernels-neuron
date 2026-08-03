"""Shared test utilities for validating NKI kernels on real Neuron hardware.

WHY THIS EXISTS
---------------
The Week 2 accuracy tests passed while never executing a single NKI instruction.
`@nki.jit` requires XLA tensors and hard-errors on CPU ones, so our kernels guard
with `if _HAS_NKI and hidden_states.device.type != "cpu"`. The tests fed CPU
tensors, so every one silently took the PyTorch fallback and compared it against a
mathematically identical reference — reporting a flawless `cos_sim = 1.000000` and
`max_diff = 0.00e+00`. See `docs/poc-findings.md` Finding #8.

The lesson: for a hardware kernel, numerical correctness alone cannot distinguish
"the kernel is right" from "the kernel never ran". You have to assert execution.

So every accuracy test in this project must do two things:
  1. Put tensors on the XLA (Neuron) device, so the NKI path is even reachable.
  2. Assert, via a call counter, that the NKI branch actually executed and the
     fallback did not.

`assert_nki_accuracy()` below enforces both, plus the cosine-similarity target.
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent

# Cosine similarity target from the steering doc.
COS_SIM_TARGET = 0.999

# Max-abs-diff tolerances, per dtype.
#
# Cosine similarity is the primary criterion (per the steering doc). max_abs_diff
# is a secondary sanity check, and it has to be calibrated per dtype: bf16 carries
# only ~8 mantissa bits, so absolute errors around 3e-2 are ordinary at activation
# magnitudes, whereas the same error in fp32 would signal a real bug.
DEFAULT_MAX_DIFF_TOL = 1e-2

MAX_DIFF_TOL_BY_DTYPE = {
    torch.float32: 1e-2,
    torch.float64: 1e-2,
    torch.bfloat16: 1e-1,
    torch.float16: 5e-2,
}


def tol_for_dtype(dtype: torch.dtype) -> float:
    """Max-abs-diff tolerance appropriate to a dtype's precision."""
    return MAX_DIFF_TOL_BY_DTYPE.get(dtype, DEFAULT_MAX_DIFF_TOL)


# --------------------------------------------------------------------------
# Kernel loading
# --------------------------------------------------------------------------

def load_kernel_module(package_name: str):
    """Load one of our local kernel packages by path.

    Our `kernels/` directory shadows the `kernels` pip package, so a normal
    import would resolve to the wrong thing. Load by explicit file location.

    The module is registered in `sys.modules` under its own name. That matters for more
    than tidiness: `torch.compile`/Dynamo re-imports a function's defining module by name
    while tracing, and without the registration it fails with
    `ModuleNotFoundError: No module named 'neuron_silu'` — which looks like a NKI/compile
    incompatibility but is purely an artifact of loading by path.
    """
    import sys

    kernel_path = PROJECT_ROOT / "kernels" / package_name / "__init__.py"
    if not kernel_path.exists():
        raise FileNotFoundError(f"No kernel package at {kernel_path}")
    if package_name in sys.modules:
        return sys.modules[package_name]
    spec = importlib.util.spec_from_file_location(package_name, kernel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = mod          # register before exec, so self-refs resolve
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(package_name, None)
        raise
    return mod


# --------------------------------------------------------------------------
# Device handling
# --------------------------------------------------------------------------

def get_xla_device():
    """Return the Neuron XLA device, or None if unavailable.

    Note `device.type` is `"xla"`, not `"neuron"` — which is precisely why
    `use_kernels=True` can't route to the neuron mapping (Finding #9).
    """
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except Exception:
        return None


def xla_hardware() -> str | None:
    """Report the XLA hardware kind, e.g. 'NEURON'. Confirms we're on real hardware."""
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device_hw(xm.xla_device())
    except Exception:
        return None


def sync():
    """Force pending XLA operations to execute."""
    try:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    except Exception:
        pass


def require_neuron() -> torch.device:
    """Return the Neuron XLA device, raising a clear error if we're not on hardware.

    Tests must fail loudly rather than silently degrade to a CPU run — that
    silent degradation is exactly what produced Finding #8.
    """
    dev = get_xla_device()
    if dev is None:
        raise RuntimeError(
            "No XLA device available. These tests must run on Trainium.\n"
            "Run on trn2 with the Neuron venv activated:\n"
            "  source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate"
        )
    hw = xla_hardware()
    if hw != "NEURON":
        raise RuntimeError(
            f"XLA device present but hardware is {hw!r}, expected 'NEURON'. "
            "Refusing to report NKI results from non-Neuron hardware."
        )
    return dev


# --------------------------------------------------------------------------
# Execution-path instrumentation
# --------------------------------------------------------------------------

@dataclass
class CallCounts:
    """Counts of which implementation actually executed."""

    nki: int = 0
    fallback: int = 0
    _names: dict = field(default_factory=dict)

    @property
    def nki_ran(self) -> bool:
        return self.nki > 0 and self.fallback == 0

    def __str__(self) -> str:
        return f"nki={self.nki} fallback={self.fallback}"


@contextmanager
def nki_call_counter(mod, nki_names: list[str], fallback_names: list[str]):
    """Patch a kernel module to count NKI vs fallback invocations.

    The kernel's `forward()` resolves these as module globals, so replacing the
    module attributes intercepts the real dispatch decision rather than guessing
    at it from the outside.

    Args:
        mod: the loaded kernel module
        nki_names: module-level attribute names of the NKI entry point(s)
        fallback_names: module-level attribute names of the PyTorch fallback(s)
    """
    counts = CallCounts()
    originals: dict[str, object] = {}

    def make_spy(name, bucket):
        real = getattr(mod, name)
        originals[name] = real

        def spy(*args, **kwargs):
            setattr(counts, bucket, getattr(counts, bucket) + 1)
            return real(*args, **kwargs)

        return spy

    try:
        for name in nki_names:
            if hasattr(mod, name):
                setattr(mod, name, make_spy(name, "nki"))
        for name in fallback_names:
            if hasattr(mod, name):
                setattr(mod, name, make_spy(name, "fallback"))
        yield counts
    finally:
        for name, real in originals.items():
            setattr(mod, name, real)


# --------------------------------------------------------------------------
# Metrics and assertions
# --------------------------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
    ).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


@dataclass
class AccuracyResult:
    label: str
    cos_sim: float
    max_diff: float
    counts: CallCounts
    passed: bool
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        note = ("  [" + "; ".join(self.notes) + "]") if self.notes else ""
        return (
            f"  {mark}  {self.label}  cos_sim={self.cos_sim:.6f}  "
            f"max_diff={self.max_diff:.3e}  {self.counts}{note}"
        )


def assert_nki_accuracy(
    label: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    counts: CallCounts,
    *,
    cos_sim_target: float = COS_SIM_TARGET,
    max_diff_tol: float = DEFAULT_MAX_DIFF_TOL,
    expect_bit_identical: bool = False,
) -> AccuracyResult:
    """Check a kernel result on all three axes that matter.

    1. The NKI branch executed and the fallback did not  (the Finding #8 guard —
       this is the authoritative check; without it the other two prove nothing).
    2. Cosine similarity clears the target.
    3. Max absolute difference is within tolerance.

    On `expect_bit_identical`: whether a zero diff is suspicious depends on the op.
    - Reduction ops (RMSNorm) sum over an axis, so NKI's reduction order differs
      from PyTorch's and the result should differ by ~1e-4. A zero diff there means
      the kernel probably didn't run — that is exactly how Finding #8 was caught.
    - Elementwise ops (RoPE) apply the same few IEEE operations in the same order
      on both backends, so bit-identical output is the *correct* expectation.
    Pass `expect_bit_identical=True` for the elementwise case to suppress the note.
    Either way the call counter, not the diff, is the authoritative execution proof.
    """
    reference = reference.detach().float().cpu()
    actual = actual.detach().float().cpu()

    cos = cosine_similarity(reference, actual)
    diff = max_abs_diff(reference, actual)

    notes: list[str] = []
    passed = True

    if not counts.nki_ran:
        passed = False
        notes.append(f"NKI DID NOT RUN ({counts})")

    if cos < cos_sim_target:
        passed = False
        notes.append(f"cos_sim below {cos_sim_target}")

    if diff > max_diff_tol:
        passed = False
        notes.append(f"max_diff exceeds {max_diff_tol:.1e}")

    if diff == 0.0 and counts.nki_ran and not expect_bit_identical:
        notes.append("suspicious: bit-identical to reference")

    return AccuracyResult(
        label=label, cos_sim=cos, max_diff=diff, counts=counts, passed=passed, notes=notes
    )


def report(results: list[AccuracyResult], title: str) -> bool:
    """Print a result table. Returns True if everything passed."""
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)
    hw = xla_hardware()
    print(f"  hardware: {hw}")
    print()
    for r in results:
        print(r.line())
    print()
    all_passed = all(r.passed for r in results)
    n_pass = sum(1 for r in results if r.passed)
    print(f"  {n_pass}/{len(results)} passed")
    if all_passed:
        print(f"  ALL PASSED (cos_sim > {COS_SIM_TARGET}, NKI execution confirmed)")
    else:
        print("  FAILURES:")
        for r in results:
            if not r.passed:
                print(f"    - {r.label}: {'; '.join(r.notes)}")
    print("=" * 76)
    return all_passed
