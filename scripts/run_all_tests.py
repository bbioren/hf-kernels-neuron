"""Run every test suite in one process-per-suite sequence, and summarise.

`make test` shells out per suite, which does not survive the detached-run wrapper
(scripts/run_detached.sh execs `python "$@"`, so it cannot take `make` as its target). This gives
the same coverage in a form the wrapper can launch, which matters because the e2e suites compile
full models and exceed a typical SSH command timeout.

Runs each suite as its own subprocess so a crash in one cannot mask the others, and so each gets a
clean Neuron runtime — two suites in one process would contend for the same cores.

Usage (on trn2):
    ./scripts/run_detached.sh /tmp/regress.log scripts/run_all_tests.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

SUITES = [
    "tests/test_rmsnorm_nki.py",
    "tests/test_rope_nki.py",
    "tests/test_silu_nki.py",
    "tests/test_qwen3_neuron_e2e.py",
    "tests/test_qwen3_moe_e2e.py",
]


def main():
    results = []
    for suite in SUITES:
        path = ROOT / suite
        if not path.exists():
            results.append((suite, "MISSING", 0.0))
            print(f"===== {suite} — MISSING, skipped =====", flush=True)
            continue

        print(f"===== {suite} =====", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
        dt = time.perf_counter() - t0
        results.append((suite, "PASS" if proc.returncode == 0 else
                        f"FAIL (exit {proc.returncode})", dt))
        print(f"----- {suite}: {results[-1][1]} in {dt:.1f}s -----", flush=True)

    print()
    print("=" * 72)
    print("REGRESSION SUMMARY")
    print("=" * 72)
    for suite, status, dt in results:
        print(f"  {status:20s} {dt:7.1f}s  {suite}")

    failed = [s for s, st, _ in results if st != "PASS"]
    print()
    if failed:
        print(f"  {len(failed)} suite(s) not passing: {', '.join(failed)}")
        return 1
    print(f"  all {len(results)} suites pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
