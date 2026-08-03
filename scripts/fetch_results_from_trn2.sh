#!/usr/bin/env bash
# Pull generated measurement artifacts back from trn2 into results/raw/.
#
# WHY THIS EXISTS SEPARATELY FROM sync_to_trn2.sh
# The sync script is deliberately push-only, uses --delete, and excludes *.neff / *.ntff. All three
# are right for pushing source to the host and all three are wrong for bringing results home:
# --delete would remove remote artifacts that do not exist locally, and the excludes would skip
# exactly the device profiles we want. So retrieval needs its own direction and its own filter.
#
# This is the step whose absence lost every artifact from the first round of measurements: they were
# generated on the host, never copied back, and died with the instance.
#
# Usage:
#   ./scripts/fetch_results_from_trn2.sh
#   REMOTE=other-host ./scripts/fetch_results_from_trn2.sh
#
# Safe to re-run. Does NOT delete local files — if a stage failed remotely, the previous local copy
# survives rather than being replaced by nothing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-trn2}"
REMOTE_PATH="${REMOTE_PATH:-hf-kernels-neuron}"

if ! ssh -o ConnectTimeout=20 "$REMOTE" "test -d $REMOTE_PATH/results/raw"; then
  echo "no $REMOTE_PATH/results/raw on $REMOTE — has 'make results' run there yet?" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/results/raw"

# No --delete: a failed remote stage should not wipe a good local artifact.
# NEFF/NTFF are explicitly included here, unlike the push direction.
# --stats rather than --info=stats2: macOS ships rsync 2.6.9, which has no --info flag.
rsync -az --stats \
  "$REMOTE:$REMOTE_PATH/results/raw/" "$REPO_ROOT/results/raw/" | tail -6

echo
echo "=== fetched into results/raw/ ==="
if [ -f "$REPO_ROOT/results/raw/index.json" ]; then
  python3 - "$REPO_ROOT/results/raw/index.json" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))
ok = [k for k, v in idx.items() if v.get("exit_code") == 0]
bad = [k for k, v in idx.items() if v.get("exit_code") != 0]
n_art = sum(len(v.get("artifacts", [])) for v in idx.values())
print(f"  {len(idx)} stage(s), {len(ok)} ok, {len(bad)} failed, {n_art} artifact(s)")
for k in bad:
    print(f"    FAILED  {k}  (exit {idx[k]['exit_code']})")
PY
else
  echo "  (no index.json — regenerate_results.py may not have completed)"
fi

echo
du -sh "$REPO_ROOT/results/raw" 2>/dev/null || true
echo
echo "NEXT:"
echo "  1. transcribe any changed numbers into results/measurements.json"
echo "  2. flip their status from 'transcribed' to 'in_repo'"
echo "  3. make results-render && make check-docs"
echo "  4. review the artifact size before committing — NEFF/NTFF pairs can be large"
