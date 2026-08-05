"""Do the three NKI kernels actually COMPILE AND RUN on the Native PyTorch stack?

Everything measured in this project so far ran on the torch-xla stack with NKI 0.5.0. The native
drop is a different compiler (neuronx-cc 2.0.266551.0a0 vs 2.26.6360.0) and a different NKI
(0.6.0b1 vs 0.5.0). An earlier probe established only that all three kernels IMPORT under 0.6.0b1
with _HAS_NKI=True. Import is not compile: @nki.jit does nothing until traced, and the 0.5.0
migration already broke this project once (nl.arange removed, mask= removed on load/store --
Finding #14). So the kernels have to be executed, not imported.

Three things are checked, in order:

  1. torch.neuron.synchronize(). It hung before, and I wrongly wrote that up as a second,
     independent fork/futex deadlock. strace showed both hangs were the same missing-neuronx-cc-
     on-PATH failure. If it works here, that confirms it and retires the phantom defect.

  2. Each kernel runs on a device tensor and matches a CPU reference.

  3. Whether NKI ACTUALLY EXECUTED, rather than the kernel quietly taking its PyTorch fallback.
     This is the whole point. Every kernel here falls back to eager PyTorch when it cannot run,
     which keeps the numbers correct while delivering no acceleration -- Finding #8, the most
     expensive failure mode on this project, because nothing looks wrong. Each kernel emits a
     RuntimeWarning naming the reason when it declines, so warnings are captured and a cos_sim
     of 1.0 that came from the fallback is reported as a FALLBACK, not a pass.

    ./scripts/run_native.sh scripts/probe_native_kernels.py

Must be run through run_native.sh: invoking the venv python directly leaves neuronx-cc off PATH
and the first device op hangs forever with no diagnostic.
"""

import importlib.util
import json
import pathlib
import sys
import warnings

import faulthandler

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
HANG_TIMEOUT_S = 900

RESULT = {
    "stack": "native",
    "torch": torch.__version__,
    "device_type_seen": None,
    "synchronize": {},
    "kernels": {},
}


def load_kernel(name):
    """Load a local kernel package by path.

    By path because the local kernel directory is itself called `kernels/`, which would
    shadow the HuggingFace `kernels` library on a normal import.
    """
    path = PROJECT_ROOT / "kernels" / name / "__init__.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.detach().flatten().float(), b.detach().flatten().float(), dim=0
    ).item()


def run_checked(label, fn):
    """Run fn() capturing fallback warnings, so a fallback cannot masquerade as a pass."""
    print(f"\n>>> {label}", flush=True)
    entry = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = fn()
            entry["ran"] = True
        except Exception as e:  # noqa: BLE001 - record whatever the compiler threw
            entry["ran"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
            print(f"    ERROR {type(e).__name__}: {e}", flush=True)
            RESULT["kernels"][label] = entry
            return None

    fallbacks = [str(w.message) for w in caught if "falling back" in str(w.message)]
    entry["used_nki"] = not fallbacks
    entry["fallback_reasons"] = fallbacks
    if fallbacks:
        print(f"    FALLBACK (NKI did NOT run): {fallbacks[0]}", flush=True)
    else:
        print("    NKI executed", flush=True)
    RESULT["kernels"][label] = entry
    return out


def main():
    print(f"torch {torch.__version__}")
    try:
        import nki

        nki_ver = getattr(nki, "__version__", "unknown")
    except ImportError:
        nki_ver = "not importable"
    print(f"nki {nki_ver}")
    RESULT["nki"] = nki_ver

    dev = "neuron"
    torch.manual_seed(0)

    # ---- 1. synchronize(), the previously "deadlocked" call -------------------------------
    print("\n>>> torch.neuron.synchronize() after a real op", flush=True)
    try:
        x = torch.randn(128, 128, dtype=torch.bfloat16).to(dev)
        y = torch.mm(x, x)
        torch.neuron.synchronize()
        _ = y.cpu()
        RESULT["synchronize"] = {"ok": True}
        RESULT["device_type_seen"] = x.device.type
        print("    OK -- so the earlier hang was the PATH/neuronx-cc issue, not a", flush=True)
        print("    separate fork/futex deadlock. Phantom defect retired.", flush=True)
    except Exception as e:  # noqa: BLE001
        RESULT["synchronize"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"    FAILED {type(e).__name__}: {e}", flush=True)

    # ---- 2. RMSNorm -----------------------------------------------------------------------
    rms = load_kernel("neuron_rmsnorm")
    hidden, eps = 2048, 1e-6
    x_cpu = torch.randn(256, hidden, dtype=torch.bfloat16)
    w_cpu = torch.randn(hidden, dtype=torch.bfloat16)
    ref = rms._pytorch_rmsnorm(x_cpu, w_cpu, eps)

    class RMS(torch.nn.Module):
        pass

    def _rms():
        layer = rms.NeuronRMSNorm()
        object.__setattr__(layer, "weight", w_cpu.to(dev))
        object.__setattr__(layer, "variance_epsilon", eps)
        return layer.forward(x_cpu.to(dev)).cpu()

    got = run_checked("rmsnorm", _rms)
    if got is not None:
        c = cos_sim(got, ref)
        RESULT["kernels"]["rmsnorm"]["cos_sim"] = c
        RESULT["kernels"]["rmsnorm"]["shape"] = list(x_cpu.shape)
        print(f"    cos_sim {c:.6f}", flush=True)

    # ---- 3. SiLU --------------------------------------------------------------------------
    silu = load_kernel("neuron_silu")
    s_cpu = torch.randn(256, 1024, dtype=torch.bfloat16)
    s_ref = torch.nn.functional.silu(s_cpu)

    def _silu():
        return silu.NeuronSiLU().forward(s_cpu.to(dev)).cpu()

    got = run_checked("silu", _silu)
    if got is not None:
        c = cos_sim(got, s_ref)
        RESULT["kernels"]["silu"]["cos_sim"] = c
        RESULT["kernels"]["silu"]["shape"] = list(s_cpu.shape)
        print(f"    cos_sim {c:.6f}", flush=True)

    # ---- 4. RoPE --------------------------------------------------------------------------
    # seq_len must be a multiple of 128 or the kernel declines by design.
    rope = load_kernel("neuron_rope")
    b, qh, kh, seq, hd = 1, 4, 2, 256, 128
    q_cpu = torch.randn(b, qh, seq, hd, dtype=torch.bfloat16)
    k_cpu = torch.randn(b, kh, seq, hd, dtype=torch.bfloat16)
    # cos/sin are 3D [batch, seq, head_dim]. The kernel accepts 2D as well, but the torch
    # reference does cos.unsqueeze(1), which only broadcasts against [b, heads, seq, hd] when
    # cos is 3D. 3D is also what transformers actually passes, so it is the honest shape to test.
    cos_cpu = torch.randn(b, seq, hd, dtype=torch.bfloat16)
    sin_cpu = torch.randn(b, seq, hd, dtype=torch.bfloat16)
    q_ref, k_ref = rope._torch_rope(q_cpu, k_cpu, cos_cpu, sin_cpu)

    def _rope():
        qo, ko = rope.apply_rotary_pos_emb(
            q_cpu.to(dev), k_cpu.to(dev), cos_cpu.to(dev), sin_cpu.to(dev)
        )
        return qo.cpu(), ko.cpu()

    got = run_checked("rope", _rope)
    if got is not None:
        cq, ck = cos_sim(got[0], q_ref), cos_sim(got[1], k_ref)
        RESULT["kernels"]["rope"]["cos_sim_q"] = cq
        RESULT["kernels"]["rope"]["cos_sim_k"] = ck
        RESULT["kernels"]["rope"]["shape"] = [b, qh, kh, seq, hd]
        print(f"    cos_sim q {cq:.6f}   k {ck:.6f}", flush=True)

    # ---- summary --------------------------------------------------------------------------
    print("\n" + "=" * 74)
    ran_nki, accurate, failed = [], [], []
    for name, e in RESULT["kernels"].items():
        if not e.get("ran"):
            failed.append(name)
            continue
        if e.get("used_nki"):
            ran_nki.append(name)
        cs = [v for k, v in e.items() if k.startswith("cos_sim")]
        if cs and all(v > 0.99 for v in cs):
            accurate.append(name)

    print(f"compiled+ran under NKI {RESULT['nki']}: {ran_nki or 'NONE'}")
    print(f"accurate (cos_sim > 0.99)            : {accurate or 'NONE'}")
    print(f"errored                              : {failed or 'none'}")
    fellback = [n for n, e in RESULT["kernels"].items() if e.get("ran") and not e.get("used_nki")]
    print(f"fell back to eager PyTorch           : {fellback or 'none'}")
    if fellback:
        print("  ^ these are NOT wins. Correct numbers, no acceleration. Finding #8.")
    print("=" * 74)

    RESULT["summary"] = {
        "ran_nki": ran_nki,
        "accurate": accurate,
        "errored": failed,
        "fell_back": fellback,
    }
    return 0 if (not failed and len(ran_nki) == 3) else 1


if __name__ == "__main__":
    out = PROJECT_ROOT / "results/raw/native/native_kernels.json"
    faulthandler.dump_traceback_later(HANG_TIMEOUT_S, exit=True)
    code = 1
    try:
        code = main()
    except BaseException as e:  # noqa: BLE001
        RESULT["fatal"] = f"{type(e).__name__}: {e}"
        import traceback

        traceback.print_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(RESULT, indent=2) + "\n")
        print(f"\nwrote {out}")
    sys.exit(code)
