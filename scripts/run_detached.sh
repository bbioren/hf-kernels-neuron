#!/usr/bin/env bash
# Run a project script on trn2 detached, so it survives SSH timeouts.
#
# Full-model compiles take longer than a typical SSH command timeout, and a dropped
# connection sends SIGHUP to the remote process. This wrapper activates the Neuron venv,
# cds into the repo, and execs the target under nohup with stdin closed.
#
# Usage (ON trn2):
#   /tmp/run_detached.sh <logfile> <script> [args...]
#
# Example:
#   ./scripts/run_detached.sh /tmp/mfu.log scripts/measure_mfu.py --preset 0.6b
set -euo pipefail

LOG="$1"
shift

source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd "$HOME/hf-kernels-neuron"

nohup python "$@" > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
