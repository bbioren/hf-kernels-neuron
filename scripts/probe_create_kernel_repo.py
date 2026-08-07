#!/usr/bin/env python3
"""Can we create a `repo_type="kernel"` repo from huggingface_hub 1.26.0?

Established by scripts/probe_hub_repo_type.py:
  - Every read path in `kernels` uses repo_type="kernel" (variants.py:239 et al).
  - `kernels-community/activation` resolves under BOTH "kernel" and "model", with
    different SHAs and different build/ variant sets -- two distinct repos.
  - Our model-type upload is invisible to get_kernel: 404 on api/kernels/...

So a loadable kernel has to live in a kernel-type repo. But
`huggingface_hub.constants.REPO_TYPES` is `[None, "model", "dataset", "space"]` --
"kernel" is absent, even though REPO_TYPE_KERNEL exists and REPO_TYPES_URL_PREFIXES
maps "kernel" -> "kernels/".

This determines whether that is a client-side validation gap or a server-side
capability we lack. Read-only apart from the create attempt, which is idempotent
(exist_ok=True) and targets the user's own namespace.

Run:
    ./scripts/run_native.sh scripts/probe_create_kernel_repo.py
"""

from __future__ import annotations

import sys

SEP = "=" * 78


def main() -> int:
    from huggingface_hub import HfApi
    from huggingface_hub import constants as C

    api = HfApi()
    me = api.whoami()["name"]
    repo_id = f"{me}/neuron-rmsnorm"

    print(SEP)
    print("Can repo_type='kernel' be created?")
    print(SEP)
    print(f"  user                {me}")
    print(f"  REPO_TYPES          {C.REPO_TYPES}")
    print(f"  'kernel' in list?   {'kernel' in [t for t in C.REPO_TYPES if t]}")

    # 1. Does client-side validation reject it outright?
    print(f"\n{'-' * 78}\n1. create_repo(repo_type='kernel')\n{'-' * 78}")
    try:
        url = api.create_repo(repo_id=repo_id, repo_type="kernel", private=True, exist_ok=True)
        print(f"  OK    created/exists: {url}")
        created = True
    except Exception as e:
        print(f"  FAIL  {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        created = False

    # 2. If the client blocks it, is the block purely local? Try the raw HTTP API.
    if not created:
        print(f"\n{'-' * 78}\n2. raw POST /api/repos/create with type=kernel\n{'-' * 78}")
        try:
            import json

            from huggingface_hub.utils import build_hf_headers, get_session

            r = get_session().post(
                f"{C.ENDPOINT}/api/repos/create",
                headers=build_hf_headers(),
                json={"name": "neuron-rmsnorm", "type": "kernel", "private": True},
            )
            print(f"  status {r.status_code}")
            body = r.text[:400]
            print(f"  body   {body}")
        except Exception as e:
            print(f"  FAIL  {type(e).__name__}: {str(e)[:200]}")

    # 3. What does the Hub say a kernel repo even is? Check a known one's URL form.
    print(f"\n{'-' * 78}\n3. URL forms\n{'-' * 78}")
    print(f"  model  {C.ENDPOINT}/{repo_id}")
    print(f"  kernel {C.ENDPOINT}/kernels/{repo_id}")

    print(f"\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
