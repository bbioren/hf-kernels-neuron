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
  "$REPO_ROOT/" "$REMOTE:$REMOTE_PATH/"

echo "synced $REPO_ROOT -> $REMOTE:$REMOTE_PATH"
