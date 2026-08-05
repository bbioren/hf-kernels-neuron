"""Does real compute work on the native stack with the CURRENT host Neuron runtime?

Second attempt. The first (probe_native_compute.py) DEADLOCKED, but not where I guessed.

py-spy on both the parent and its forked child showed the same stack:

    _fn (torch/_dynamo/eval_frame.py:1263)
    synchronize (torch_neuronx/__init__.py:477)   -> _C._neuron_synchronize(device_index)
    synchronize (torch_neuronx/neuron.py:286)
    main (probe_native_compute.py:40)             -> torch.neuron.synchronize()

So it was NOT stuck in torch.mm. The matmul is queued lazily and the explicit
torch.neuron.synchronize() is what hung. Kernel state at the time:

    parent 146412  29 threads, main on futex_do_wait, one thread on do_wait_intr_irq (waitpid)
    child  146453   1 thread,  futex_do_wait

CORRECTION. From that state I concluded "classic fork-from-a-multithreaded-process deadlock":
fork() copies only the calling thread, so a lock held by one of the other 28 threads is held
forever in the child. It fit every observation and it was wrong. strace settled it:

    execve("/usr/local/sbin/neuronx-cc", ["neuronx-cc", "compile", "module.mlir", ...]) = -1 ENOENT
    execve("/usr/local/bin/neuronx-cc",  ...) = -1 ENOENT      <- and 5 more, the whole PATH

The child forks to exec the COMPILER, by bare name, resolved through PATH. neuronx-cc lives in
native_venv/bin, and this was launched as /home/ubuntu/native_venv/bin/python <script> -- an
absolute path, which does not put the venv's bin on PATH. So every PATH entry missed, and the
child then hung rather than reporting "neuronx-cc not found". The futex wait was a symptom.

Two consequences. The hang is not version skew, so the drop's deb packages are NOT required to
run; and synchronize() was never a separate defect -- it forces the same compile, so it hung for
the same reason. This probe drops the explicit synchronize() anyway, since .cpu() has to
materialise the result and therefore synchronises.

Guardrails, because the last run hung for four minutes with no output:
  - faulthandler dumps every thread's stack and exits if any step exceeds the timeout,
    so a hang is self-diagnosing rather than needing py-spy attached after the fact
  - each step prints before it starts and flushes, so the log localises the hang
  - nothing is forked here on purpose

    /home/ubuntu/native_venv/bin/python scripts/probe_native_compute2.py
"""

import faulthandler
import json
import os
import pathlib
import sys
import time

# If any single step wedges, dump every thread's Python stack and die rather than hang forever.
HANG_TIMEOUT_S = 420

import torch

RESULT = {
    "torch": torch.__version__,
    "has_torch_neuron_attr": hasattr(torch, "neuron"),
    "privateuse1": torch._C._get_privateuse1_backend_name(),
    "steps": {},
    # Root cause of the hang, established by strace AFTER this probe was first written.
    # The fork/futex story below was wrong; see the CORRECTION note in the module docstring.
    "hang_root_cause": {
        "diagnosis": "neuronx-cc not on PATH",
        "evidence": "strace: child execve()s bare name 'neuronx-cc' and gets ENOENT on all 7 "
        "PATH entries, then hangs instead of reporting the failure",
        "trigger": "any first device op that needs a compile (.cpu() materialisation "
        "or torch.neuron.synchronize() -- both, because both force the compile)",
        "fix": "activate the venv so native_venv/bin is on PATH (scripts/run_native.sh)",
        "not_caused_by": "host runtime / driver version mismatch. The deb packages in the "
        "drop are NOT required to run.",
    },
}


def step(name):
    """Mark a step so a hang or crash is attributable to an exact line."""
    print(f"\n>>> {name} ...", flush=True)
    return time.time()


def done(name, t0, **fields):
    dt = time.time() - t0
    RESULT["steps"][name] = {"ok": True, "seconds": round(dt, 3), **fields}
    detail = "  ".join(f"{k}={v}" for k, v in fields.items())
    print(f"<<< {name} OK in {dt:.2f}s   {detail}", flush=True)


def main():
    print(f"torch {torch.__version__}")
    print(f"hasattr(torch, 'neuron') {hasattr(torch, 'neuron')}")
    print(f"privateuse1 {torch._C._get_privateuse1_backend_name()}")
    print(f"pid {os.getpid()}   hang timeout {HANG_TIMEOUT_S}s")

    torch.manual_seed(0)

    # 1. move tensors to device. Known to work from the earlier bare test.
    t0 = step("h2d transfer")
    a = torch.randn(256, 256, dtype=torch.bfloat16)
    b = torch.randn(256, 256, dtype=torch.bfloat16)
    ad, bd = a.to("neuron"), b.to("neuron")
    done("h2d transfer", t0, device=str(ad.device), device_type=ad.device.type)

    # 2. the real question: does a matmul execute and come back correct?
    #    .cpu() forces materialisation, so this covers compile + execute + d2h
    #    WITHOUT calling the synchronize() that deadlocked.
    t0 = step("matmul + d2h via .cpu() (no explicit synchronize)")
    cd = torch.mm(ad, bd)
    got = cd.cpu().float()
    ref = a.float() @ b.float()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    err = (got - ref).abs().max().item()
    done("matmul + d2h via .cpu() (no explicit synchronize)", t0,
         cos_sim=round(cos, 6), max_abs=round(err, 4))
    RESULT["matmul_cos_sim"] = cos
    RESULT["matmul_max_abs"] = err

    # 3. a couple of elementwise ops, to show it was not a one-off fluke
    t0 = step("elementwise add/mul/silu")
    sd = (ad + bd) * 2.0
    fd = torch.nn.functional.silu(sd)
    got2 = fd.cpu().float()
    ref2 = torch.nn.functional.silu((a.float() + b.float()) * 2.0)
    cos2 = torch.nn.functional.cosine_similarity(got2.flatten(), ref2.flatten(), dim=0).item()
    done("elementwise add/mul/silu", t0, cos_sim=round(cos2, 6))
    RESULT["elementwise_cos_sim"] = cos2

    ok = cos > 0.99 and cos2 > 0.99
    RESULT["compute_works"] = ok
    RESULT["deb_packages_required_to_run"] = not ok

    print("\n" + "=" * 72)
    if ok:
        print("RESULT: compute WORKS on the native stack against the CURRENT host runtime.")
        print("        The deb/ runtime+driver packages are NOT required merely to run, so the")
        print("        host runtime can be left alone and the existing XLA venv stays intact.")
        print("        Separately: torch.neuron.synchronize() deadlocks. Avoid it; .cpu() syncs.")
    else:
        print("RESULT: compute FAILED. The deb/ packages likely ARE required.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    out = pathlib.Path("results/raw/native/native_compute2.json")
    faulthandler.dump_traceback_later(HANG_TIMEOUT_S, exit=True)
    code = 1
    try:
        code = main()
    except BaseException as e:  # noqa: BLE001 - want the reason recorded, whatever it is
        RESULT["error"] = f"{type(e).__name__}: {e}"
        RESULT["compute_works"] = False
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(RESULT, indent=2) + "\n")
        print(f"\nwrote {out}")
    sys.exit(code)
