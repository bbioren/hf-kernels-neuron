"""Re-run every measurement in this project and write raw artifacts into results/raw/.

WHY THIS EXISTS
The first round of measurements wrote everything to /tmp on the trn2 host. That instance expired and
all raw artifacts went with it — JSON, NEFF/NTFF profile pairs, detached run logs. The numbers
survived only because each run's stdout had been pasted into a commit message. This script makes the
raw evidence land in the repo instead, so it survives the host.

Every stage writes:
  results/raw/<stage>/stdout.log     full stdout+stderr
  results/raw/<stage>/*.json         structured output where the script emits it
  results/raw/<stage>/*.neff|.ntff   device profiles where the stage produces them
  results/raw/index.json             stage -> {command, exit code, seconds, artifacts}

Run ON trn2, from the repo root, inside the Neuron venv. Expect roughly 30-60 minutes: several
stages compile full models, and Neuron processes must not run concurrently (they contend for cores),
so stages are strictly sequential.

    python scripts/regenerate_results.py                 # everything
    python scripts/regenerate_results.py --only mfu fix  # named stages
    python scripts/regenerate_results.py --list
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "results" / "raw"

# (stage, argv, [globs of files the stage leaves elsewhere that should be collected])
STAGES = [
    ("versions", [sys.executable, "scripts/probe_nki_versions.py"], []),
    ("smoke", [sys.executable, "scripts/smoke_device.py"], []),
    ("tests", [sys.executable, "scripts/run_all_tests.py"], []),

    # Run the flag control EARLY: if the NKI/torch ratio turns out to depend on
    # NEURON_CC_FLAGS, every device-time measurement below needs re-running under the better
    # setting, and it is better to learn that in minute five than at the end.
    ("compiler-flag-control", [sys.executable, "scripts/probe_compiler_flags.py"], []),

    ("mfu-baseline-and-kernelized-512-fixed",
     [sys.executable, "scripts/measure_mfu.py", "--preset", "0.6b", "--seq", "512",
      "--fix-target-detection", "--json-out", "STAGE/mfu_512_fixed.json"], []),
    ("mfu-kernelized-512-nofix",
     [sys.executable, "scripts/measure_mfu.py", "--preset", "0.6b", "--seq", "512",
      "--json-out", "STAGE/mfu_512_nofix.json"], []),
    ("mfu-2048-fixed",
     [sys.executable, "scripts/measure_mfu.py", "--preset", "0.6b", "--seq", "2048",
      "--fix-target-detection", "--json-out", "STAGE/mfu_2048_fixed.json"], []),

    ("fix-verification", [sys.executable, "scripts/probe_target_override_fix.py"], []),
    ("graph-batching", [sys.executable, "scripts/probe_neff_count.py"], []),
    ("host-vs-device-split", [sys.executable, "scripts/probe_where_is_the_time.py"], []),
    ("cprofile-before", [sys.executable, "scripts/probe_inside_one_call.py"], []),
    ("cprofile-after",
     [sys.executable, "scripts/probe_inside_one_call.py", "--fix-target-detection"], []),
    ("torch-compile-diagnosis", [sys.executable, "scripts/diagnose_torch_compile.py"], []),

    ("device-profile-28-calls",
     [sys.executable, "scripts/profile_nki_call_cost.py", "--calls", "28",
      "--outdir", "STAGE/prof_n28"], []),

    ("fusion-sweep",
     [sys.executable, "scripts/run_device_profile_sweep.py", "--calls", "1", "28"], []),
    ("fusion-analysis", [sys.executable, "scripts/analyse_fusion_barrier.py"], []),

    ("insitu-baseline",
     [sys.executable, "scripts/profile_model_device_time.py", "--mode", "baseline",
      "--outdir", "RAW/prof_model_baseline"], []),
    ("insitu-kernelized",
     [sys.executable, "scripts/profile_model_device_time.py", "--mode", "kernelized",
      "--outdir", "RAW/prof_model_kernelized"], []),
    ("insitu-summary",
     [sys.executable, "scripts/sum_model_device_time.py",
      "RAW/prof_model_baseline", "RAW/prof_model_kernelized",
      "--wall-baseline", "46.65", "--wall-kernelized", "146.65", "--nki-calls", "169"], []),

    ("fused-mlp-nki",
     [sys.executable, "scripts/profile_fused_mlp_vs_torch.py", "--impl", "nki", "--calls", "28",
      "--outdir", "RAW/prof_mlp_nki"], []),
    ("fused-mlp-torch",
     [sys.executable, "scripts/profile_fused_mlp_vs_torch.py", "--impl", "torch", "--calls", "28",
      "--outdir", "RAW/prof_mlp_torch"], []),
    ("fused-mlp-boundary", [sys.executable, "scripts/spike_nkilib_mlp.py"], []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="run only these stages (substring match)")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing stage instead of stopping")
    args = ap.parse_args()

    if args.list:
        for name, argv, _ in STAGES:
            print(f"  {name:42s} {' '.join(argv[1:])}")
        return 0

    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if any(sub in s[0] for sub in args.only)]
        if not stages:
            print(f"no stage matches {args.only}; use --list")
            return 1

    RAW.mkdir(parents=True, exist_ok=True)
    index_path = RAW / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    print(f"regenerating {len(stages)} stage(s) into {RAW.relative_to(ROOT)}")
    print("stages run sequentially — two Neuron processes contend for cores\n")

    failures = []
    for i, (name, argv, extra_globs) in enumerate(stages, 1):
        outdir = RAW / name
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True)

        # Placeholders let a stage's --outdir / --json-out write into the artifact tree rather
        # than /tmp. Two of them, because they mean different things:
        #   STAGE/ -> this stage's own directory. Use for per-stage output, so it is picked up
        #             by the artifact glob below and cannot collide with another stage.
        #   RAW/   -> the shared raw root. Use only when a LATER stage must read the output
        #             (the in-situ profiles are produced by two stages and consumed by a third).
        resolved = [a.replace("STAGE/", f"{outdir}/").replace("RAW/", f"{RAW}/") for a in argv]

        print(f"[{i}/{len(stages)}] {name}")
        print(f"    {' '.join(resolved[1:])}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(resolved, cwd=ROOT, capture_output=True, text=True)
        dt = time.perf_counter() - t0

        (outdir / "stdout.log").write_text(proc.stdout + "\n----- stderr -----\n" + proc.stderr)

        artifacts = sorted(
            str(p.relative_to(ROOT)) for p in outdir.rglob("*") if p.is_file()
        )
        index[name] = {
            "command": " ".join(resolved[1:]),
            "exit_code": proc.returncode,
            "seconds": round(dt, 1),
            "artifacts": artifacts,
        }
        index_path.write_text(json.dumps(index, indent=2) + "\n")

        if proc.returncode == 0:
            print(f"    ok in {dt:.0f}s, {len(artifacts)} artifact(s)\n")
        else:
            failures.append(name)
            tail = proc.stdout.strip().splitlines()[-4:]
            print(f"    FAILED exit {proc.returncode} in {dt:.0f}s")
            for t in tail:
                print(f"      {t}")
            print()
            if not args.keep_going:
                print("stopping. re-run with --keep-going to continue past failures.")
                break

    print("=" * 72)
    print(f"wrote {index_path.relative_to(ROOT)}")
    if failures:
        print(f"{len(failures)} stage(s) failed: {', '.join(failures)}")
    else:
        print("all stages ok")
    print()
    print("NEXT: transcribe any changed numbers into results/measurements.json, flip their")
    print("      status from 'transcribed' to 'in_repo', then run scripts/render_results.py.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
