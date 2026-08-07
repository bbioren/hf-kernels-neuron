#!/usr/bin/env python3
"""Upload a staged kernel repo to the HuggingFace Hub.

This has an external side effect, which nothing else in this project does. Guardrails:

  - PRIVATE by default. Private -> public is one click; public -> private after someone
    has already cloned it is not. `--public` is opt-in.
  - `--dry-run` prints exactly what would be uploaded and exits.
  - Prints the resulting commit SHA, because a kernel loaded with
    `trust_remote_code=True` should be pinned by `revision=<sha>` rather than a mutable
    branch name. Upstream does exactly this for `Atlas-Inference/gdn`:
    `LayerRepository(repo_id="Atlas-Inference/gdn", revision="ef12347f...",
    trust_remote_code=True)` with a TODO to drop it once the org is allow-listed.

Requires a cached write token on the machine:
    /home/ubuntu/native_venv/bin/hf auth login

Run:
    ./scripts/run_native.sh scripts/publish_hub_repo.py --dry-run
    ./scripts/run_native.sh scripts/publish_hub_repo.py
    ./scripts/run_native.sh scripts/publish_hub_repo.py --public
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEP = "=" * 78


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="dist/hub/neuron-rmsnorm", help="staged repo dir")
    ap.add_argument("--owner", default=None, help="Hub namespace. Default: the logged-in user")
    ap.add_argument("--name", default=None, help="repo name. Default: the staged dir name")
    ap.add_argument("--public", action="store_true", help="create PUBLIC (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    folder = Path(args.repo)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    if not folder.is_dir():
        sys.exit(f"ERROR: {folder} does not exist. Run scripts/build_hub_repo.py first.")

    api = HfApi()

    try:
        me = api.whoami()
    except Exception as e:
        sys.exit(
            f"ERROR: not authenticated ({type(e).__name__}: {e})\n"
            "Run: /home/ubuntu/native_venv/bin/hf auth login"
        )

    owner = args.owner or me["name"]
    name = args.name or folder.name
    repo_id = f"{owner}/{name}"
    private = not args.public

    # Loading a kernel in place leaves __pycache__ behind. Never ship bytecode: it is
    # machine- and version-specific, and a stale .pyc next to an edited .py is a
    # genuinely confusing failure for whoever downloads it.
    IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]

    def ignored(p: Path) -> bool:
        return "__pycache__" in p.parts or p.suffix == ".pyc" or p.name == ".DS_Store"

    files = sorted(p for p in folder.rglob("*") if p.is_file() and not ignored(p))
    skipped = sorted(p for p in folder.rglob("*") if p.is_file() and ignored(p))
    total = sum(p.stat().st_size for p in files)

    print(SEP)
    print("Publish kernel to the HuggingFace Hub")
    print(SEP)
    print(f"  authenticated as  {me['name']}")
    print(f"  target repo       {repo_id}")
    print(f"  visibility        {'PUBLIC' if args.public else 'PRIVATE'}")
    print(f"  source            {folder}")
    print(f"  files             {len(files)}  ({total / 1024:.1f} KiB)")
    for p in files:
        print(f"    {p.relative_to(folder)}")
    for p in skipped:
        print(f"    (skipped) {p.relative_to(folder)}")

    if args.dry_run:
        print(f"\n{SEP}\nDRY RUN — nothing uploaded.")
        return 0

    print(f"\n{'-' * 78}\nCreating repo\n{'-' * 78}")
    try:
        url = api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"  {url}")
    except HfHubHTTPError as e:
        sys.exit(f"ERROR creating repo: {e}")

    print(f"\n{'-' * 78}\nUploading\n{'-' * 78}")
    try:
        commit = api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add NKI kernel for Neuron (HF kernels build-variant layout)",
            ignore_patterns=IGNORE,
        )
    except HfHubHTTPError as e:
        sys.exit(f"ERROR uploading: {e}")

    sha = getattr(commit, "oid", None) or getattr(commit, "commit_id", None)
    print(f"  commit  {sha}")
    print(f"  url     {getattr(commit, 'commit_url', '?')}")

    print(f"\n{SEP}")
    print(f"Uploaded {repo_id}  ({'public' if args.public else 'private'})")
    print(f"\nPin this SHA when loading, since a personal namespace is not a trusted publisher:")
    print(f"""
    LayerRepository(
        repo_id="{repo_id}",
        layer_name="NeuronRMSNorm",
        revision="{sha}",
        trust_remote_code=True,
    )
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
