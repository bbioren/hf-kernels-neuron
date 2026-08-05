#!/usr/bin/env bash
# Run a script against the Native PyTorch venv with PATH set up correctly.
#
# Why this exists. Invoking /home/ubuntu/native_venv/bin/python by absolute path does NOT put the
# venv's bin/ on PATH. That matters more than usual here, because on the first device op the native
# runtime forks a child and execve()s `neuronx-cc` by BARE NAME, resolved through PATH. With the
# venv unactivated, neuronx-cc is unreachable (it lives at native_venv/bin/neuronx-cc, while the
# default PATH points at /opt/aws/neuron/bin), and the child gets ENOENT on every PATH entry.
#
# The failure mode is the problem: instead of surfacing "neuronx-cc not found", the child HANGS,
# the parent waits on it, and the process deadlocks forever with no diagnostic. Twice this looked
# like a runtime/version incompatibility that would need the drop's deb packages installed over the
# host runtime. It was PATH. See docs/sticking-points.md.
#
#   ./scripts/run_native.sh scripts/probe_native_compute2.py
#   ./scripts/run_native.sh -m pytest tests/ -x

set -euo pipefail

VENV=/home/ubuntu/native_venv

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "error: no native venv python at $VENV/bin/python" >&2
    exit 1
fi

# Activating is what puts $VENV/bin on PATH, which is the whole point.
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! command -v neuronx-cc > /dev/null; then
    echo "error: neuronx-cc still not on PATH after activating $VENV" >&2
    echo "       the first device op will fork, fail to exec it, and hang forever" >&2
    exit 1
fi

echo "native venv : $VENV"
echo "python      : $(command -v python)  $(python --version 2>&1)"
echo "neuronx-cc  : $(command -v neuronx-cc)  $(neuronx-cc --version 2>&1 | head -1)"
echo "---"

exec python "$@"
