#!/usr/bin/env bash
# Build the Native PyTorch (TorchNeuron) venv from the beta drop, and report what it can do.
#
# CONTEXT
# Every measurement in this project so far ran on the torch-xla stack (the DLAMI venv
# /opt/aws_neuronx_venv_pytorch_2_9, torch_xla 2.9.0, xm.xla_device()). That is the wrong stack:
# torch-xla is being deprecated, and on the native backend a model on-device reports device type
# "neuron" rather than "xla". Two of this project's findings — Gate 1 (hasattr(torch, "neuron") is
# False) and Gate 2 (model.device.type is "xla") — are very likely artifacts of the XLA path, and
# Finding #28 (the torch_xla op-registry cache) may not exist here at all.
#
# Drop: s3://huggingface-aws/pytorch-native/drop_jun_25/  (Samir Araujo allowlisted our AWS account
# on the bucket; fetched by scripts/fetch_native_drop.sh)
#
# THIS SCRIPT IS ADDITIVE AND SAFE. It creates a new venv and installs wheels into it. It does NOT
# install the deb/ packages, which replace the host Neuron runtime and driver and would very likely
# break the existing XLA venv — that is a separate, deliberate step.
#
# Usage, on the instance:
#   ./scripts/setup_native_venv.sh
set -euo pipefail

DROP="${DROP:-/home/ubuntu/native_drop}"
VENV="${VENV:-/home/ubuntu/native_venv}"
PIPDIR="$DROP/pip"

[ -d "$PIPDIR" ] || { echo "missing $PIPDIR — run scripts/fetch_native_drop.sh first"; exit 1; }

echo "=== wheel requirements (before installing anything) ==="
for w in "$PIPDIR"/*.whl; do
    echo "--- $(basename "$w")"
    unzip -p "$w" '*.dist-info/METADATA' 2>/dev/null \
        | grep -E '^(Name|Version|Requires-Python|Requires-Dist):' | head -14 | sed 's/^/    /'
done

echo
echo "=== creating venv at $VENV ==="
if [ -d "$VENV" ]; then
    echo "  exists already; reusing"
else
    python3.12 -m venv "$VENV"
fi
# The steering log records that stock pip 24.0 on Ubuntu 24.04 chokes on NKI's non-standard version
# string with "TypeError: expected string or bytes-like object, got 'NoneType'". Upgrade first.
"$VENV/bin/pip" install -q --upgrade pip
echo "  pip $("$VENV/bin/pip" --version | awk '{print $2}')"

echo
echo "=== installing wheels ==="
# torch_mlir and nki/neuronx_cc first, then torch_neuronx last so its dependency resolution
# (including the torch version it wants) settles on top of an already-consistent set.
for w in torch_mlir nki neuronx_cc torch_neuronx; do
    f=$(ls "$PIPDIR"/${w}-*.whl 2>/dev/null | head -1)
    [ -n "$f" ] || { echo "  SKIP $w (no wheel)"; continue; }
    echo "  installing $(basename "$f")"
    "$VENV/bin/pip" install -q "$f" 2>&1 | tail -3 | sed 's/^/      /'
done

echo
echo "=== installed versions ==="
"$VENV/bin/pip" list 2>/dev/null | grep -iE '^(torch|torch-neuronx|torch-mlir|nki|neuronx-cc|numpy) ' \
    | awk '{printf "  %-18s %s\n", $1, $2}'

echo
echo "=== does the native device exist? ==="
"$VENV/bin/python" - <<'PY' 2>&1 | sed 's/^/  /'
import torch
print("torch                ", torch.__version__)
print("hasattr(torch,neuron)", hasattr(torch, "neuron"))
try:
    print("privateuse1 name     ", torch._C._get_privateuse1_backend_name())
except Exception as e:
    print("privateuse1 name      ERROR", e)
try:
    t = torch.randn(2, 2, device="neuron")
    print("device='neuron'       OK ->", t.device)
except Exception as e:
    print("device='neuron'       FAILED:", type(e).__name__, str(e)[:180])
PY

echo
echo "NEXT"
echo "  If device='neuron' failed with a runtime/driver mismatch, the deb/ packages are required."
echo "  Those replace the host Neuron runtime + driver:"
echo "    host now : dkms 2.29.0.0, runtime-lib 2.33.10.0, collectives 2.33.10.0, tools 2.31.13.0"
echo "    drop has : dkms 2.x.9869.0, runtime-lib 2.x.59853.0, collectives 2.x.60792.0, tools 2.0.27186.0"
echo "  and will very likely break /opt/aws_neuronx_venv_pytorch_2_9 (the XLA venv all existing"
echo "  measurements were taken on). Recoverable from the public apt repo, ~10 min DKMS each way."
