#!/usr/bin/env bash
# Sync the working tree to trn2 for testing.
#
# Tests must run on trn2 (NKI backend confirmed there), but we edit locally.
# rsync avoids committing/pushing just to test — and we never push to a remote.
#
# Usage:
#   ./scripts/sync_to_trn2.sh
#
# Excludes git metadata, caches, and Neuron compile artifacts.
#
# CRITICAL: results/raw/ is EXCLUDED, and that exclusion is load-bearing.
# This script uses --delete, so anything present on the remote but absent locally is removed.
# results/raw/ is an OUTPUT directory — it is written on the remote by `make results` and only
# ever travels remote -> local. Without this exclusion, syncing a code change after a
# measurement run deletes the run's artifacts. That happened once and cost 19 of 22 stages'
# artifacts; it is the same failure as the original loss (results living somewhere something
# else destroys them), just with a different destroyer.
#
# Use scripts/fetch_results_from_trn2.sh for the return direction. Never add results/raw/ here.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-trn2}"
REMOTE_PATH="${REMOTE_PATH:-hf-kernels-neuron}"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.venv/' \
  --exclude 'neuron_cache/' \
  --exclude '.neuron_cache/' \
  --exclude '*.neff' \
  --exclude '*.ntff' \
  --exclude 'results/raw/' \
  "$REPO_ROOT/" "$REMOTE:$REMOTE_PATH/"

echo "synced $REPO_ROOT -> $REMOTE:$REMOTE_PATH  (results/raw/ deliberately untouched)"
