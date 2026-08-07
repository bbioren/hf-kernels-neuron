#!/usr/bin/env python3
"""Generate a Hub-publishable kernel repo from a flat local kernel package.

WHY THIS EXISTS
---------------
Our kernels live flat on disk (`kernels/neuron_rmsnorm/{__init__.py,metadata.json}`)
and load fine through `LocalLayerRepository`. That works because of an asymmetry in
the `kernels` library:

  get_local_kernel  -> _resolve_local_variant_path  -> has a flat fallback
  get_kernel (Hub)  -> install_kernel              -> NO flat fallback

`utils.py:367` builds the path unconditionally:

    variant_path = repo_path / "build" / variant_str
    if not variant_path.exists():
        raise FileNotFoundError(f"Variant path does not exist: `{variant_path}`")

So a flat layout that loads locally cannot load from the Hub. This script emits the
layout `get_kernel` actually wants.

THE LAYOUT (verified against kernels 0.15.2 source, not guessed)
----------------------------------------------------------------
    <repo>/
      LICENSE                                  Apache-2.0 (nki-library derivative)
      NOTICE                                   required by Apache-2.0 4(d)
      README.md                                model card
      build/
        torch-neuron/                          noarch variant: "torch-<backend>"
          metadata.json                        <- INSIDE the variant dir, not at root
          neuron_rmsnorm/
            __init__.py

Two details that are easy to get wrong:

1. `metadata.json` is read from `variant_path / "metadata.json"` (`utils.py:199`),
   i.e. inside `build/<variant>/`. Putting it at the repo root silently fails.

2. The module dir name is `metadata.name.python_name` (`utils.py:200`), so
   `"name": "neuron-rmsnorm"` implies a `neuron_rmsnorm/` directory. `_import_from_path`
   accepts either `build/<variant>/__init__.py` directly or
   `build/<variant>/<python_name>/__init__.py` (`utils.py:203-205`). We emit the
   nested form because that is what real Hub kernels look like.

VARIANT NAMING
--------------
`variants.py:210-235` parses noarch variants as `torch-<backend>`. `variants.py:470-477`
accepts a noarch variant if its backend matches the system backend *or* is `universal`.
`variants.py:526` prefers the specific backend over universal when both are present.

So `torch-neuron` and `torch-universal` should both resolve on Neuron. Which one is
correct for a pure-Python NKI kernel is a real question, so `--variant` takes both and
defaults to emitting both, letting the on-device test decide rather than us.

Usage:
    python scripts/build_hub_repo.py neuron_rmsnorm
    python scripts/build_hub_repo.py neuron_rmsnorm --variant torch-neuron
    python scripts/build_hub_repo.py --all
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
KERNELS_DIR = PROJECT_ROOT / "kernels"
OUT_DIR = PROJECT_ROOT / "dist" / "hub"

# Both candidates. `variants.py:473` accepts either on a Neuron system.
DEFAULT_VARIANTS = ["torch-neuron", "torch-universal"]


def python_name(name: str) -> str:
    """Mirror `metadata.name.python_name`: Hub names are dashed, modules underscored."""
    return name.replace("-", "_")


def read_metadata(kernel_dir: Path) -> dict:
    meta_path = kernel_dir / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"ERROR: no metadata.json in {kernel_dir}")
    with open(meta_path) as f:
        return json.load(f)


def build_readme(meta: dict, variants: list[str], kernel_dir: Path) -> str:
    """Model card.

    The card carries the two things a user cannot discover from the code: that this
    needs Trainium/Inferentia hardware, and that it needs `trust_remote_code=True`
    because a personal namespace has no `trustedKernelPublisher` flag.
    """
    name = meta["name"]
    mod = python_name(name)
    is_rope_derivative = kernel_dir.name == "neuron_rope"

    provenance = (
        "Ported from [`aws-neuron/nki-library`](https://github.com/aws-neuron/nki-library) "
        "(`src/nkilib_src/nkilib/core/embeddings/rope_hf.py`), Apache-2.0. "
        "Modifications are stated in the file header and `NOTICE`."
        if is_rope_derivative
        else "Derived from the NKI tutorials in "
        "[`aws-neuron/nki-samples`](https://github.com/aws-neuron/nki-samples) (MIT-0)."
    )

    variant_lines = "\n".join(f"- `build/{v}/`" for v in variants)

    return f"""---
license: apache-2.0
tags:
- kernel
- neuron
- trainium
- nki
---

# {name}

A [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/) kernel for
AWS Trainium / Inferentia, packaged for the HuggingFace
[`kernels`](https://github.com/huggingface/kernels) library.

**This is a proof-of-concept artifact.** It exists to validate that NKI kernels can be
distributed through the Kernel Hub and swapped into stock `transformers` models on Neuron
hardware. It is not an officially supported AWS release.

## Requirements

- AWS Trainium or Inferentia hardware. **This kernel cannot run on CPU or CUDA.**
- `neuronx-cc` on `PATH` and the Neuron runtime installed.
- `torch` with a Neuron backend (either `torch-neuronx` native or `torch_xla`).

## Usage

```python
from kernels import LayerRepository, Mode, kernelize, register_kernel_mapping

register_kernel_mapping({{
    "{meta.get('layer_name', 'RMSNorm')}": {{
        "neuron": LayerRepository(
            repo_id="bbioren/{name}",
            layer_name="{meta.get('layer_name', 'NeuronRMSNorm')}",
            # Required: a personal namespace is not a `trustedKernelPublisher`.
            # Upstream does the same for `Atlas-Inference/gdn`.
            trust_remote_code=True,
        )
    }}
}})

model = kernelize(model, mode=Mode.INFERENCE)
```

## Build variants

{variant_lines}

Pure Python, so there is nothing architecture-specific to compile — `kernel-builder` is
not involved. The variants are noarch (`torch-<backend>`, per `variants.py:210-235`).

## Provenance

{provenance}

## Known gaps

- `has_backward = False`. Forward only; training is not supported.
- `can_torch_compile = False`. Untested under `torch.compile`, not known to fail.
- `python-depends` is empty in `metadata.json` even though the kernel imports `nki`.
  Deliberate, to keep the first Hub-loading test to one variable.
"""


def build_one(kernel_name: str, variants: list[str], clean: bool = True) -> Path:
    kernel_dir = KERNELS_DIR / kernel_name
    if not kernel_dir.is_dir():
        sys.exit(f"ERROR: {kernel_dir} does not exist")

    meta = read_metadata(kernel_dir)
    repo_name = meta["name"]
    mod = python_name(repo_name)

    repo_root = OUT_DIR / repo_name
    if clean and repo_root.exists():
        shutil.rmtree(repo_root)

    for variant in variants:
        variant_dir = repo_root / "build" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        sources = sorted(kernel_dir.glob("*.py"))

        # The official spec (huggingface.co/docs/kernels/v0.15.2/kernel-requirements)
        # requires the kernel at `build/<variant>/__init__.py` AND, "for compatibility
        # with older versions of the kernels package", a nested directory named after
        # the repo (dashes -> underscores) exporting the same symbols:
        #
        #   build/<variant>/__init__.py          <- primary
        #   build/<variant>/<mod>/__init__.py    <- compat, same symbols
        #
        # kernels 0.15.2 tries the primary first and only falls back to the nested path
        # (utils.py:203-205), so shipping only the nested form works *today* but is not
        # spec-compliant. We emit both. The files are duplicated rather than re-exported
        # via a relative import because these modules are loaded by file path
        # (`spec_from_file_location`), not imported as a package, so `from .. import *`
        # would fail with "attempted relative import beyond top-level package".
        mod_dir = variant_dir / mod
        mod_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            shutil.copy2(src, variant_dir / src.name)
            shutil.copy2(src, mod_dir / src.name)

        # metadata.json goes in the VARIANT dir, not the module dir and not the root.
        # utils.py:199 reads `variant_path / "metadata.json"`.
        shutil.copy2(kernel_dir / "metadata.json", variant_dir / "metadata.json")

    # Apache-2.0 4(a)/4(d): ship the license and the upstream NOTICE.
    for extra in ("LICENSE", "NOTICE"):
        src = PROJECT_ROOT / extra
        if src.exists():
            shutil.copy2(src, repo_root / extra)
        else:
            print(f"  WARNING: {extra} missing at repo root, not shipped")

    (repo_root / "README.md").write_text(build_readme(meta, variants, kernel_dir))

    return repo_root


def describe(repo_root: Path) -> None:
    print(f"\n  {repo_root.relative_to(PROJECT_ROOT)}")
    for p in sorted(repo_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(repo_root)
            print(f"    {rel}  ({p.stat().st_size} B)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel", nargs="?", help="kernel dir under kernels/, e.g. neuron_rmsnorm")
    ap.add_argument("--all", action="store_true", help="build every kernel under kernels/")
    ap.add_argument(
        "--variant",
        action="append",
        default=None,
        help=f"build variant(s). Default: {' and '.join(DEFAULT_VARIANTS)}",
    )
    args = ap.parse_args()

    variants = args.variant or DEFAULT_VARIANTS

    if args.all:
        targets = sorted(p.name for p in KERNELS_DIR.iterdir() if p.is_dir() and (p / "metadata.json").exists())
    elif args.kernel:
        targets = [args.kernel]
    else:
        ap.error("give a kernel name or --all")

    print("=" * 78)
    print("Building Hub repo layout")
    print("=" * 78)
    print(f"  variants: {', '.join(variants)}")
    print(f"  output:   {OUT_DIR.relative_to(PROJECT_ROOT)}")

    for name in targets:
        repo_root = build_one(name, variants)
        describe(repo_root)

    print(f"\n  {len(targets)} repo(s) staged. Nothing uploaded — use scripts/publish_hub_repo.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
