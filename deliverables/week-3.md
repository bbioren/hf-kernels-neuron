# Week 3 Deliverable — RoPE, Hub packaging, and the neuron device path

**Date:** 2026-07-29
**Branch:** `week-3`
**Hardware:** trn2.3xlarge (1 Neuron device, 4 NeuronCores), `xla_device_hw() == "NEURON"`
**Versions:** `kernels 0.15.2`, `transformers 5.15.0.dev0`, `torch 2.9.1+cu128`, `neuronx-cc 2.26.6360.0+6f180f47`

---

## Headline

Three NKI kernels — RMSNorm, RoPE, SiLU — now execute inside a real Qwen3 forward pass
on Trainium through the HuggingFace Kernel Hub mechanism, with execution verified rather
than assumed. RoPE is a genuine port of a production `nki-library` kernel.

Two findings changed the shape of the project:

1. **Week 2's accuracy results were measuring the PyTorch fallback, not NKI.** The
   kernel was never executed. It turned out to be correct, but it had not been tested.
2. **`use_kernels=True` cannot reach the `"neuron"` device path at all.** It fails as a
   silent no-op. We identified the minimal upstream fix and *verified* it is sufficient.

---

## What was validated

All numbers from trn2 over SSH, with the NKI branch confirmed executing via call
counters. Cosine similarity target is > 0.999.

### Per-kernel accuracy

| Kernel | Cases | Result | max_abs_diff | Source |
|--------|-------|--------|--------------|--------|
| RMSNorm | 11 | **11/11 pass**, all `nki=1 fallback=0` | 1.2e-05 → 3.9e-04 (fp32), 3.1e-02 (bf16) | NKI tutorial (production kernel unusable) |
| RoPE | 20 | **20/20 pass**, all `nki=1 fallback=0` | 0.0 (elementwise — expected) | **`nki-library` `rope_hf`, ported** |
| SiLU | 9 | **9/9 pass**, all `nki=1 fallback=0` | 0.0 → 2.1e-06 | `nl.silu` native |

Coverage: GQA and MQA, batch 1–4, seq_len 128–1024, head_dim 64/128, hidden sizes up to
4096, MLP widths up to 12288 (Qwen3-8B), fp32 and bf16, 2D and 3D cos/sin.

### End-to-end on Qwen3

Real `Qwen3ForCausalLM` (2 layers), seq_len 128, on the Neuron device:

| Kernel | NKI calls | Fallback calls | Expected |
|--------|-----------|----------------|----------|
| RMSNorm | **9** | 0 | 9 (4/layer + final norm) |
| RoPE | **2** | 0 | 2 (1/layer) |
| SiLU | **2** | 0 | 2 (1/layer) |

Logits vs the unkernelized on-device model: **cos_sim 1.000001, max_abs_diff 5.29e-05**.

### Fallback behaviour

Every kernel degrades gracefully *and loudly* when its constraints aren't met
(RoPE at `seq_len % 128 != 0`, SiLU beyond a SBUF-safe width): fallback taken, warning
emitted with the specific reason, output still correct.

---

## Finding #8 — Week 2's results were not measuring NKI [CRITICAL]

The Week 2 test reported `cos_sim = 1.000000, max_diff = 0.00e+00` for all 8 shapes.
Bit-identical output is the wrong answer for a reduction kernel: NKI sums in a different
order than PyTorch and should differ by ~1e-4. Exact zero meant both sides ran the same
code.

They did. `NeuronRMSNorm.forward` gates on `device.type != "cpu"`, and the tests built
inputs with `torch.randn(...)` — CPU tensors. So every case took the PyTorch fallback,
which is mathematically identical to `Qwen3RMSNorm.forward`, and compared it to itself.
The test even printed `"Backend: NKI kernel (NeuronCores)"` — reporting whether NKI was
*importable*, not whether it ran.

Verified with instrumentation: `nki=0, fallback=1` on CPU tensors; `@nki.jit` raises
`RuntimeError: Expected all tensors ... to be XLA tensors` if called with them directly;
and on XLA tensors the kernel runs correctly with `max_diff = 1.7e-04`.

**Why this generalizes.** `@nki.jit` hard-errors on CPU tensors, so *every* HF Neuron
kernel needs a device guard, and the natural way to write that guard produces a silent
fallback. A customer sets `use_kernels=True`, sees no warning and correct numbers, and
concludes they have acceleration. There is no signal anywhere.

**What we changed.** `tests/nki_test_utils.py` now enforces three things per case:
`require_neuron()` refuses to report results off Neuron hardware; `nki_call_counter()`
asserts the NKI branch ran and the fallback did not; and tolerances are dtype-aware with
cosine similarity as the primary gate. Kernels also `warn_once` on fallback with the
reason.

---

## Finding #9 — `use_kernels=True` cannot reach `"neuron"` [HIGH]

The Week 3 goal "confirm `use_kernels=True` alone triggers the swaps" **cannot be met
today**. Two independent causes:

- transformers' `kernelize(model, mode)` has **no `device` parameter**. It derives
  everything from `model.device.type`. The underlying `kernels.kernelize` *does* accept
  a device; transformers doesn't expose it.
- Neuron never reports `"neuron"`. Params on the host give `"cpu"`; moved to the device
  they give `"xla"`. Nothing maps `"xla"` → `"neuron"`, and
  `hasattr(torch, "neuron")` is False even after `import torch_neuronx`.

**The failure is a silent no-op, not an error.** transformers passes a `Device` *object*,
and `kernels.kernelize` only validates device types given as *strings*. So
`Device(type="xla")` passes through unvalidated, matches no mapping entry, and every
layer quietly keeps its original forward while `kernelize()` returns success.

### The fix, and why verifying it mattered

Our first proposal was to patch `kernels._find_device`. **That would not have worked** —
transformers computes the device itself and never calls `_find_device` on this path. The
e2e test caught it. The fix belongs in transformers:

```python
# transformers/integrations/hub_kernels.py::kernelize
device_type = model.device.type
if device_type == "cuda" and is_rocm_platform():
    device_type = "rocm"
elif device_type == "xla" and _is_neuron_xla():     # <- the fix
    device_type = "neuron"
device = Device(type=device_type)
```

`_is_neuron_xla()` checks `xm.xla_device_hw(xm.xla_device()) == "NEURON"`, which we
confirmed returns exactly that on trn2 — reliable, no new dependency.

**Verified sufficient:** applied in-process, this single branch takes Qwen3 from
**0 → 9 swapped RMSNorm layers** via the transformers `use_kernels` path, logits
`cos_sim 1.000001`. The recommendation is demonstrated, not hypothesized.

The same gap exists at two more sites that need the same treatment:
`transformers/utils/kernel_config.py::infer_device()` and `kernels._find_device`.

---

## Porting RoPE — and why it reverses the RMSNorm story

For RMSNorm, the production `nki-library` kernel was unusable: fused with FP8
quantization, no unfused path, heavy internal dependencies. We used a tutorial kernel and
concluded that nki-library kernels are too fused to port.

**RoPE is the opposite case**, and it matters for the recommendation:

- `nki-library/core/embeddings/rope_hf.py` is **already HF-shaped**: 4D
  `[batch, heads, seq, head_dim]`, precomputed cos/sin, tuple return, `rotate_half`
  convention, independent q/k head counts for GQA, and a backward path.
- There is **no** rotary tutorial anywhere in `nki-samples`. nki-library is the only source.
- Only 3 internal symbols to inline (~15 lines), and no `common_types` dependency.

So "nki-library is too fused to port" is **per-kernel, not a general truth**. The
`embeddings/` module already contains HF-friendly code.

Adaptations required: strip SPMD sharding (HF swaps per-layer, single core); convert
destination-passing (`q_out`/`k_out` args) to internal `nl.shared_hbm` allocation
returning a tuple — verified multi-output `@nki.jit` works; drop the `rope_cache` and
`backward` branches; add a Python-level guard so constraint violations fall back instead
of asserting.

**The inherited constraint is the real limitation:** `seq_len % (128 × LNC) == 0`. HF
passes arbitrary sequence lengths, so this is the most likely reason a customer silently
gets no acceleration.

Two NKI idioms worth recording: there is **no concatenation primitive**, so
`rotate_half`'s `torch.cat((-x2, x1), -1)` becomes writes into disjoint slices of a
preallocated destination with the negation folded into `op=nl.subtract`; and
`nki-library`'s own implementation is the best available reference for that pattern.

---

## Finding #12 — HF already whitelists `nki`, but the entry is unreachable [HIGH]

`kernels/python_depends.json` contains a `neuron` backend section whitelisting `nki`.
HuggingFace has already made room for NKI kernels — better than we assumed.

But `validate_dependencies()` consults the table for whatever `_backend()` reports, and
on the DLAMI that is `CUDA(version=12.8)`. Verified against a real copy of our RoPE kernel:

| `python-depends` | Result |
|---|---|
| `[]` | loads |
| `["nki"]` | `ValueError: unsupported kernel dependency: nki` |

So a Neuron kernel must **under-declare its own dependency** to load at all. This is
Finding #7 compounding: the same `hasattr(torch, "neuron")` root cause now breaks both
build-variant resolution and dependency validation. Fixing `_backend()` is the
highest-leverage single change — though it does *not* fix device routing, which needs
Finding #9's separate transformers change.

`nkilib` is not whitelisted. The ask to add it is now concrete: `nki` establishes the
precedent and the exact JSON shape to copy, in the same file.

Also measured: `metadata.json` requires `name`, `id`, `version`, `license`,
`python-depends`, `backend`; `digest` is **optional**. Minimum viable kernel repo is two
files.

---

## Finding #14 — the two NKI import paths are not interchangeable [HIGH]

Both `import nki` and `import neuronxcc.nki` succeed, but they have different
capabilities and **neither is a superset**:

| Idiom | top-level `nki` | `neuronxcc.nki` |
|---|---|---|
| `nl.arange` index tensors | **fails**: `failed to resolve name 'nki.language.arange'` | works |
| `//` on shape values | works | **fails**: `math.trunc() is not supported for scalar` |

So RMSNorm and SiLU require `neuronxcc.nki`, RoPE requires top-level `nki`, and this
repository genuinely needs both. `hasattr(nl, "arange")` returns True under the package
where it cannot be resolved, so there is no import-time feature detection — you find out
at compile time, per kernel. nki-library source uses top-level `nki` while the tutorials
use `neuronxcc.nki`, so any mass-porting effort meets this immediately.

---

## Finding #15 — what the mechanism can actually reach

| Interception point | Registrations | Status |
|---|---|---|
| `RMSNorm` (layer) | **115** | ported ✓ |
| `rotary_pos_emb` (func) | **95** model files | ported ✓ |
| `SiLU` (layer) | 1 decoration in `activations.py`, covering every model using that `ACT2FN` | ported ✓ |
| `GeLU`, `GeluTanh`, `NewGELU`, `FastGELU`, `QuickGELU` | 1 each, same leverage | not ported |
| `MegaBlocksMoeMLP`, `Llama4TextMoe` | 2, 1 | Week 5 candidates |

Because activation decorations live in `activations.py` rather than per-model, **one**
activation kernel covers every model using that entry. This is the strongest form of the
per-kernel-rather-than-per-model investment argument.

**Caveat:** several `_KERNEL_MAPPING` entries are registered by **no model** and are
unreachable via the decorator path — including `SwiGLUMLP`, `GeGLUMLP`, and `Linear`.
`SwiGLUMLP` matters, because a fused gate/up/SiLU/down MLP is where MLP performance
actually is. It requires the separate, more invasive fusion API
(`register_kernel_replacements_and_fusions` / `make_parent_class_for_kernel_fusion`),
not a per-layer forward swap.

**On SiLU and performance honesty:** standalone elementwise SiLU is memory-bandwidth
bound. It is included to prove the activation path works and complete mechanism
coverage, **not** because it should be expected to speed anything up. The profitable
unit is the fused MLP. No performance claim should be made for it without measurement.

---

## Week 3 goals vs. outcome

| Goal (from steering doc) | Status |
|---|---|
| Move from local loading to Hub-style packaged repo | **Partial.** Flat layout validated as loadable; `digest` shown optional; minimum repo is 2 files. No upload (out of scope this session). Blocked on Finding #12 for honest dependency declaration. |
| Add RoPE NKI kernel as a `FuncRepository` entry | **Done.** Ported from production nki-library, 20/20 accuracy, registered via `LocalFuncRepository`. |
| Add `"neuron"` entries to `_KERNEL_MAPPING` for RMSNorm and RoPE | **Done locally** (`scripts/neuron_kernel_registration.py`), plus SiLU. Upstream diff written out. Not submitted — no remote actions this session. |
| Confirm `use_kernels=True` alone triggers the swaps on Neuron | **Cannot be met today — Finding #9.** Root-caused, minimal fix identified *and verified sufficient*. This is the single highest-value upstream ask. |
| Coordinate with Samir on Hub repo home | **Not done** — requires external communication, out of scope. Still open. |
| Stretch: start SiLU | **Done.** 9/9 accuracy, wired into e2e. |

---

## What needs a decision

1. **Hub repo home:** `kernels-community/` vs `aws-neuron/`. Blocks publishing and fixes
   the `repo_id` in the upstream diff. Needs Samir.
2. **Who owns the upstream fixes?** Finding #9 is a transformers change (3 sites);
   Finding #12/#7 is a `torch_neuronx` change; Finding #14 is an NKI-team question. All
   three are small; none are ours to merge.
3. **Is `has_backward=False` acceptable for beta?** All three kernels are
   inference-only, so training mode falls back. nki-library's `rope_hf` *does* have a
   backward path we chose not to wire up.

## Recommended next steps (Week 4)

- MFU measurement with and without the kernels, stating the denominator explicitly.
  Expect RMSNorm and RoPE to help and SiLU not to; measure rather than assume.
- Full-size Qwen3-8B rather than a 2-layer stand-in, to confirm the `seq_len % 128`
  guard doesn't silently disable RoPE at realistic sequence lengths.
- Investigate the fusion API as the route to a fused MLP kernel — likely a bigger win
  than any remaining elementwise op.
