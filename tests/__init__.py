"""Test package marker.

Exists so `tests/` is importable, which lets the suites share `tests/nki_test_utils.py` (the
execution-assertion harness added after Finding #8). The suites are plain scripts rather than
pytest cases — they must run on Neuron hardware and each needs its own process, so
`scripts/run_all_tests.py` invokes them as subprocesses.
"""
