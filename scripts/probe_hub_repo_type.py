#!/usr/bin/env python3
"""What repo_type does a Hub kernel actually live under?

We uploaded with `repo_type="model"` and `get_kernel` 404'd on
`https://huggingface.co/api/kernels/bbioren/neuron-rmsnorm/tree/...`. Every read path in
the `kernels` library hardcodes `repo_type="kernel"` (variants.py:239, utils.py:286/323/
349/393/551, lockfile.py:52/62, status.py:57), but `huggingface_hub.constants.REPO_TYPES`
lists only `[None, "model", "dataset", "space"]`.

So either:
  (a) "kernel" is an alias that resolves to a model repo server-side, and our 404 has
      some other cause, or
  (b) "kernel" is a genuinely distinct repo type we have to create differently.

Rather than guess, interrogate a known-working kernel repo (kernels-community/activation)
both ways and compare against ours.

Run:
    ./scripts/run_native.sh scripts/probe_hub_repo_type.py
"""

from __future__ import annotations

import sys

SEP = "=" * 78
KNOWN_GOOD = "kernels-community/activation"
OURS = "bbioren/neuron-rmsnorm"


def try_tree(api, repo_id: str, repo_type: str, path: str = "build"):
    try:
        tree = list(api.list_repo_tree(repo_id, path_in_repo=path, repo_type=repo_type))
        names = [t.path for t in tree]
        return True, names
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"


def try_info(api, repo_id: str, repo_type: str):
    try:
        info = api.repo_info(repo_id, repo_type=repo_type)
        return True, f"id={info.id} private={getattr(info, 'private', '?')} sha={(getattr(info, 'sha', '') or '')[:8]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"


def main() -> int:
    from huggingface_hub import HfApi
    from huggingface_hub import constants as C

    api = HfApi()

    print(SEP)
    print("Hub repo_type probe")
    print(SEP)
    print(f"  REPO_TYPES              {C.REPO_TYPES}")
    print(f"  REPO_TYPES_URL_PREFIXES {getattr(C, 'REPO_TYPES_URL_PREFIXES', '?')}")
    print(f"  REPO_TYPE_KERNEL        {getattr(C, 'REPO_TYPE_KERNEL', '<absent>')!r}")
    kmap = getattr(C, "REPO_TYPES_MAPPING", None)
    print(f"  REPO_TYPES_MAPPING      {kmap}")

    for repo_id, label in ((KNOWN_GOOD, "KNOWN-GOOD"), (OURS, "OURS")):
        print(f"\n{'-' * 78}\n{label}: {repo_id}\n{'-' * 78}")
        for rt in ("kernel", "model"):
            ok, detail = try_info(api, repo_id, rt)
            print(f"  repo_info(repo_type={rt!r:9}) {'OK  ' if ok else 'FAIL'}  {detail}")
        for rt in ("kernel", "model"):
            ok, detail = try_tree(api, repo_id, rt)
            if ok:
                print(f"  tree(build, {rt!r:9})        OK    {detail}")
            else:
                print(f"  tree(build, {rt!r:9})        FAIL  {detail}")

    # Can we even create a "kernel" repo type?
    print(f"\n{'-' * 78}\nIs repo_type='kernel' creatable?\n{'-' * 78}")
    import inspect

    try:
        src = inspect.getsource(api.create_repo)
        mentions = [ln.strip() for ln in src.splitlines() if "repo_type" in ln][:6]
        print("  create_repo repo_type handling:")
        for m in mentions:
            print(f"    {m}")
    except Exception as e:
        print(f"  (could not introspect create_repo: {e})")

    print(f"\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
