#!/usr/bin/env python3
"""End-to-end: load a NKI kernel from the HuggingFace Hub and swap it into Qwen3.

Everything up to now loaded kernels from disk via `LocalLayerRepository`. That path has
a flat fallback the Hub path does not have, so "it loads locally" was never evidence that
it would load from the Hub. This closes that gap.

Three things are tested, in increasing strength:

  A. `get_kernel(repo_id, revision=...)` resolves and imports from the Hub at all.
  B. A `LayerRepository` pointing at the Hub repo survives `kernelize()` and actually
     replaces `Qwen3RMSNorm.forward`.
  C. The swapped forward, when run, calls the *NKI* path rather than the PyTorch
     fallback -- and the model's logits still match the unkernelized model.

C matters more than it looks. An earlier version of this project reported a flawless
`max_diff = 0.00e+00` while the kernel had never executed once: the swap had silently
fallen back. So the NKI entry point is spied on directly rather than inferred.

The cache is purged first. Without that, a "successful Hub load" can be a local cache
hit from a previous run, which proves nothing about the Hub.

Run:
    ./scripts/run_native.sh tests/test_hub_kernel_e2e.py --revision <sha>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch

SEP = "=" * 78
REPO_ID = "bbioren/neuron-rmsnorm"
LAYER_NAME = "NeuronRMSNorm"
SEQ_LEN = 128


def purge_cache(repo_id: str) -> None:
    """Delete the local Hub cache for `repo_id` so the next load really downloads."""
    from huggingface_hub.constants import HF_HUB_CACHE

    slug = "models--" + repo_id.replace("/", "--")
    path = Path(HF_HUB_CACHE) / slug
    if path.exists():
        shutil.rmtree(path)
        print(f"  purged cache: {path}")
    else:
        print(f"  no cache to purge at {path}")


def build_qwen3(num_layers=2, hidden_size=256, head_dim=64):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=1024,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=1024,
        use_cache=False,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model, config


def test_a_get_kernel(revision: str | None) -> tuple[bool, object]:
    print(f"\n{'-' * 78}\nA. get_kernel() from the Hub\n{'-' * 78}")
    from kernels import get_kernel

    purge_cache(REPO_ID)

    # trust_remote_code=True is MANDATORY for a personal namespace, and not because we
    # skipped a setup step. `utils.py:80` checks trust via
    # `get_organization_overview(publisher)`, which 404s for a *user* namespace since a
    # user is not an organization. So the check cannot succeed for `bbioren/...` at all
    # -- it can only be bypassed. Same bypass upstream uses for `Atlas-Inference/gdn`.
    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    try:
        mod = get_kernel(REPO_ID, **kwargs)
    except Exception as e:
        print(f"  FAIL  {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None

    print(f"  loaded      {mod}")
    print(f"  from file   {getattr(mod, '__file__', '?')}")
    variant = "?"
    f = getattr(mod, "__file__", "") or ""
    for cand in ("torch-neuron", "torch-universal"):
        if f"/build/{cand}/" in f:
            variant = cand
            break
    print(f"  variant     {variant}")

    layers = getattr(mod, "layers", None)
    if layers is None or not hasattr(layers, LAYER_NAME):
        print(f"  FAIL: no layers.{LAYER_NAME}")
        return False, None
    print(f"  layers.{LAYER_NAME} present")
    return True, mod


def test_b_kernelize(revision: str | None, device) -> tuple[bool, object, object]:
    print(f"\n{'-' * 78}\nB. kernelize() with a Hub LayerRepository\n{'-' * 78}")
    from kernels import LayerRepository, Mode, kernelize, use_kernel_mapping

    model, config = build_qwen3()
    model = model.to(device)

    import transformers.models.qwen3.modeling_qwen3 as mq

    target = mq.Qwen3RMSNorm
    print(f"  target      {target.__name__}")
    print(f"  kernel_layer_name = {getattr(target, 'kernel_layer_name', None)!r}")

    before = {
        name: type(m).forward is not m.forward
        for name, m in model.named_modules()
        if isinstance(m, target)
    }
    n_rms = len(before)
    print(f"  {n_rms} {target.__name__} instances found")

    repo_kwargs = dict(repo_id=REPO_ID, layer_name=LAYER_NAME, trust_remote_code=True)
    if revision:
        repo_kwargs["revision"] = revision
    print(f"  LayerRepository({', '.join(f'{k}={v!r}' for k, v in repo_kwargs.items())})")

    mapping = {"RMSNorm": {"neuron": LayerRepository(**repo_kwargs)}}

    try:
        with use_kernel_mapping(mapping, inherit_mapping=False):
            # use_fallback=True is required, not a weakening of the test. With False,
            # `kernelize` raises `No layer mapping for 'SiLU'` because Qwen3 has other
            # decorated layers (SiLU, rotary_pos_emb) that this single-kernel mapping
            # deliberately does not cover. The swap is asserted below by counting
            # replaced forwards, and its execution by the NKI call counter in step C,
            # so strict mode buys nothing here.
            kernelize(model, device="neuron", mode=Mode.INFERENCE, use_fallback=True)
    except Exception as e:
        print(f"  FAIL  {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None, None

    swapped = [
        name
        for name, m in model.named_modules()
        if isinstance(m, target) and "forward" in m.__dict__
    ]
    print(f"  swapped     {len(swapped)}/{n_rms}")
    for name in swapped[:4]:
        m = dict(model.named_modules())[name]
        print(f"    {name} -> {m.forward.__qualname__}")
    if len(swapped) > 4:
        print(f"    ... and {len(swapped) - 4} more")

    if not swapped:
        print("  FAIL: kernelize() replaced nothing")
        return False, None, None
    return True, model, config


def test_c_executes(mod, model, config, device) -> bool:
    print(f"\n{'-' * 78}\nC. does the NKI path actually run, and is it correct?\n{'-' * 78}")
    from nki_test_utils import cosine_similarity, max_abs_diff, nki_call_counter, sync

    ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN)).to(device)

    # Reference: an identical unkernelized model.
    ref_model, _ = build_qwen3()
    ref_model = ref_model.to(device)
    with torch.no_grad():
        ref_logits = ref_model(ids).logits
    sync()

    with nki_call_counter(mod, ["_nki_rmsnorm_kernel"], ["_pytorch_rmsnorm"]) as counts:
        with torch.no_grad():
            logits = model(ids).logits
        sync()

    print(f"  NKI calls       {counts.nki}")
    print(f"  fallback calls  {counts.fallback}")

    if counts.nki == 0:
        print("  FAIL: the Hub kernel was swapped in but never executed the NKI path.")
        print("        This is exactly the failure mode that produced a misleading")
        print("        max_diff = 0.00e+00 earlier in this project.")
        return False

    cos = cosine_similarity(logits, ref_logits)
    mad = max_abs_diff(logits, ref_logits)
    print(f"  logits cos_sim  {cos:.6f}")
    print(f"  logits max_diff {mad:.3e}")

    ok = cos > 0.999
    if not ok:
        print(f"  FAIL: cos_sim {cos:.6f} below 0.999")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=None, help="pin a commit SHA (recommended)")
    ap.add_argument(
        "--local-override",
        default=None,
        metavar="PATH",
        help=(
            "Set LOCAL_KERNELS to map the repo_id to a local staged repo, stubbing ONLY "
            "the network download. Everything else -- get_kernel, variant resolution, "
            "metadata parsing, LayerRepository, kernelize, NKI execution -- stays real. "
            "Use while kernel-repo creation is access-gated (Finding #35)."
        ),
    )
    args = ap.parse_args()

    if args.local_override:
        import os

        path = Path(args.local_override)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        os.environ["LOCAL_KERNELS"] = f"{REPO_ID}={path}"
        # `LayerRepository.__init__` requires a revision or a version
        # (`layer/layer.py:76`). A pinned SHA is meaningless against a local directory,
        # so use the branch name: it satisfies the constructor and LOCAL_KERNELS
        # short-circuits the lookup before the revision is ever resolved.
        args.revision = args.revision or "main"
        print(f"!! LOCAL_KERNELS={os.environ['LOCAL_KERNELS']}")
        print("!! The network download is STUBBED. This does not prove Hub delivery.")

    from nki_test_utils import require_neuron, stack

    device = require_neuron()

    print(SEP)
    print("Hub kernel end-to-end")
    print(SEP)
    import kernels

    print(f"  stack       {stack()}")
    print(f"  kernels     {kernels.__version__}")
    print(f"  repo        {REPO_ID}")
    print(f"  revision    {args.revision or '(default branch)'}")

    ok_a, mod = test_a_get_kernel(args.revision)
    if not ok_a:
        print(f"\n{SEP}\nFAILED at A: the Hub path does not resolve.")
        return 1

    ok_b, model, config = test_b_kernelize(args.revision, device)
    if not ok_b:
        print(f"\n{SEP}\nFAILED at B: loaded from the Hub but kernelize() did not swap.")
        return 1

    ok_c = test_c_executes(mod, model, config, device)

    suffix = "  (download STUBBED)" if args.local_override else ""
    print(f"\n{SEP}")
    print(f"  A get_kernel by repo_id  {'PASS' if ok_a else 'FAIL'}{suffix}")
    print(f"  B kernelize swap         {'PASS' if ok_b else 'FAIL'}")
    print(f"  C NKI ran + correct      {'PASS' if ok_c else 'FAIL'}")
    if args.local_override:
        print()
        print("  NOT PROVEN: Hub download. LOCAL_KERNELS short-circuited it.")
        print("  Everything else on the get_kernel path is real.")
    print(SEP)
    return 0 if (ok_a and ok_b and ok_c) else 1


if __name__ == "__main__":
    sys.exit(main())
