# Upstream fixes: what we need from other teams

Consolidated from `docs/poc-findings.md`. Each entry has the exact code location, a
ready-to-paste patch, the verification status, and who owns it.

**None of these are ours to merge.** All are small. Between them they unblock the entire
`use_kernels=True` experience on Trainium.

Versions these were verified against: `kernels 0.15.2`, `transformers 5.15.0.dev0`
(commit `bb3ffb97`), `torch 2.9.1+cu128`, `neuronx-cc 2.26.6360.0+6f180f47`, trn2.3xlarge.

| # | Fix | Owner | Size | Verified? | Findings |
|---|-----|-------|------|-----------|----------|
| 1 | Route XLA-on-Neuron to `Device(type="neuron")` | transformers (+ kernels) | ~12 lines, 3 sites | **Yes — demonstrated sufficient** | #9 |
| 2 | Make `_backend()` report `neuron` | `torch_neuronx` | 1 attribute | Root-caused, fix not built | #7, #12 |
| 3 | Resolve the `nki` / `neuronxcc.nki` capability split | NKI team | needs a decision | Documented only | #14 |
| 4 | Add `nkilib` to the `python-depends` allowlist | HF `kernels` | ~6 lines | Feasibility verified | #16 |

Fixes 1 and 4 are asks to HuggingFace (raise with Samir). Fix 2 is internal to Neuron.
Fix 3 is an NKI-team decision.

---

## Fix 1 — Route XLA-on-Neuron to the `"neuron"` device [HIGHEST VALUE]

### The problem

`use_kernels=True` never selects a `"neuron"` mapping entry. It fails as a **silent no-op**:
`kernelize()` returns successfully, every layer keeps its original forward, and nothing is
logged.

Two independent causes:

1. transformers' `kernelize(model, mode)` has no `device` parameter. It derives the device
   solely from `model.device.type`.
2. Neuron never reports `"neuron"`. Params on the host give `"cpu"`; moved to the device
   they give `"xla"`. Nothing maps `"xla"` → `"neuron"`.

And the no-op is invisible because transformers passes a `Device` **object**, while
`kernels.kernelize` only validates device types given as **strings**:

```python
# kernels/layer/kernelize.py
if device is None:            device_type = _find_device(model)
elif isinstance(device, str): _validate_device_type(device); device_type = Device(type=device)
else:                         device_type = Device(device.type)      # <- unvalidated
```

So `Device(type="xla")` passes straight through and matches nothing. (Calling the kernels
library directly with the *string* `device="xla"` does raise
`Unsupported device type 'xla'` — which is how we first mis-diagnosed this.)

### Where it goes

We initially proposed patching `kernels._find_device`. **That would not have worked** — the
transformers wrapper computes the device itself and never calls `_find_device` on this path.
Our e2e test caught it. Three sites need the same treatment:

**Site A (required — the `use_kernels=True` path)**
`transformers/integrations/hub_kernels.py::kernelize`

```python
     device_type = model.device.type
     if device_type == "cuda" and is_rocm_platform():
         device_type = "rocm"
+    elif device_type == "xla" and _is_neuron_xla():
+        device_type = "neuron"
     device = Device(type=device_type)
```

**Site B (required — the `KernelConfig` path)**
`transformers/utils/kernel_config.py::infer_device` — same `param.device.type` logic with a
cuda/rocm refinement and no xla handling.

```python
     dev_type = param.device.type
     if dev_type == "cuda":
         ...
+    elif dev_type == "xla" and _is_neuron_xla():
+        return "neuron"
     return dev_type
```

**Site C (recommended — direct kernels-library callers)**
`kernels/layer/kernelize.py::_find_device`, used when `kernelize(model)` is called with no
device.

```python
     dev_type = param.device.type
     if dev_type == "cuda":
         ...
+    elif dev_type == "xla" and _is_neuron_xla():
+        return Device(type="neuron")
     return Device(type=dev_type)
```

**The shared helper**, needed in transformers and in kernels:

```python
def _is_neuron_xla() -> bool:
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device_hw(xm.xla_device()) == "NEURON"
    except Exception:
        return False
```

Verified on trn2: `xm.xla_device_hw(xm.xla_device())` returns exactly `"NEURON"`. Reliable,
no new dependency, and it fails closed.

### Also needed: the mapping entries

Site A alone routes correctly but finds nothing to load until `"neuron"` entries exist.
`"rotary_pos_emb"` already has `cuda` / `rocm` / `xpu` siblings, so this is an addition, not
a restructure. In `transformers/integrations/hub_kernels.py::_build_kernel_mapping`:

```python
     _KERNEL_MAPPING = {
         ...
+        "RMSNorm": {
+            "neuron": LayerRepository(
+                repo_id="<aws-neuron|kernels-community>/rmsnorm",
+                layer_name="NeuronRMSNorm", version=1,
+            )
+        },
+        "SiLU": {
+            "neuron": LayerRepository(
+                repo_id="<...>/silu", layer_name="NeuronSiLU", version=1,
+            )
+        },
     }

     _FUNCTION_KERNEL_MAPPING = {
         "rotary_pos_emb": {
             "cuda": FuncRepository(...), "rocm": {...}, "xpu": {...},
+            "neuron": FuncRepository(
+                repo_id="<...>/rope",
+                func_name="apply_rotary_pos_emb", version=1,
+            ),
         },
     }
```

`repo_id` is blocked on the Hub repo-home decision (Samir).

### Verification status: **demonstrated sufficient**

`enable_neuron_device_detection()` in `scripts/neuron_kernel_registration.py` applies the
Site A patch in-process (it substitutes a faithful copy of the transformers wrapper with the
one branch added — nothing on disk is modified). `tests/test_qwen3_neuron_e2e.py` test 2
then drives the *transformers* entry point:

| | RMSNorm layers swapped | logits cos_sim |
|---|---|---|
| stock | **0** | — |
| with the patch | **9** | 1.000001 |

So this is not a hypothesis. Lead with it when filing.

### What to do

1. Open a transformers issue with the reproduction from
   `scripts/probe_neuron_device_path.py` (prints the device resolution at each step).
2. Offer the patch for Sites A and B plus the mapping entries, once `repo_id` is settled.
3. Open a companion `kernels` issue for Site C.
4. Raise with Samir first — he can say whether HF wants the helper in `kernels` and imported
   by transformers, rather than duplicated.

### Interim workaround (in use now)

`kernelize_for_neuron(model)` in `scripts/neuron_kernel_registration.py`: calls the kernels
library directly with `device="neuron"` and replicates the `_hidden_kernels` attach/detach
that function kernels require.

---

## Fix 2 — Make `_backend()` report `neuron` on Neuron hosts

### The problem

```
kernels.utils._backend()  ->  CUDA(version=Version('12.8'))
```

on a Neuron DLAMI, because the root check is:

```python
# kernels/backends.py:198
if hasattr(torch, "neuron"):
```

and `hasattr(torch, "neuron")` is **False even after `import torch_neuronx`**.

### Why it matters twice

One root cause, two independent breakages:

**(a) Build-variant resolution (#7).** A Hub repo containing
`build/torch29-neuron-x86_64-linux/` won't resolve — the loader looks for a CUDA variant.
Our flat layout works only because variant resolution failing falls back to importing the
repo root. Multi-backend repos (CUDA + Neuron in one package) are impossible until this is
fixed.

**(b) `python-depends` validation (#12).** `kernels/python_depends.json` **already
whitelists `nki` under a `neuron` backend section** — HF anticipated NKI kernels. But
`validate_dependencies()` consults the table for whatever `_backend()` reports, so the neuron
table is never read. Verified against a real copy of our RoPE kernel:

| `python-depends` | Result |
|---|---|
| `[]` | loads |
| `["nki"]` | `ValueError: unsupported kernel dependency: nki` |

**So a Neuron kernel must under-declare its own dependency in order to load at all.** Ours
ship `python-depends: []` while importing `nki`. That works only because the DLAMI happens to
have NKI preinstalled; elsewhere the user gets a bare `ImportError` instead of the intended
`"requires Python dependency nki. Please install with: pip install nki"`.

### The fix

`torch_neuronx` should set a `torch.neuron` attribute at import time, so `hasattr` succeeds.
One line in Neuron's own code — no HF change required, which makes it the cheapest item on
this list.

**Important:** this does **not** fix Fix 1. It changes neither the transformers device
computation nor `_find_device`'s return value. Two distinct problems; don't let them get
conflated into one ticket.

### What to do

1. Confirm with the `torch_neuronx` owners that setting `torch.neuron` is acceptable and what
   it should hold (a bool, a module, a version — HF only checks presence).
2. File against `torch_neuronx` referencing `kernels/backends.py:198` and
   `kernels/layer/kernelize.py:307` (`_has_neuron_ops`).
3. Reproduction: `scripts/probe_hub_packaging.py` (prints `_backend()` and the
   `python-depends` failure).
4. Once landed, change our kernels' `metadata.json` to `"python-depends": ["nki"]` and
   re-run `make probe` to confirm.

---

## Fix 3 — Resolve the `nki` / `neuronxcc.nki` capability split

### The problem

Both import successfully. Neither is a superset. A kernel is pinned to whichever supports its
idiom, discovered only at compile time.

| Idiom | top-level `nki` | `neuronxcc.nki` |
|---|---|---|
| `nl.arange` index tensors + `mask=` | **fails**: `error: failed to resolve name 'nki.language.arange'` | works |
| `//` on tensor shape values | works (shapes are plain ints) | **fails**: `NotImplementedError: math.trunc() is not supported for scalar` |

Consequence in this repo, verified by swapping imports and re-running both suites:

| Kernel | Idiom | Required package |
|--------|-------|------------------|
| `neuron_rmsnorm` | `nl.arange` + mask | `neuronxcc.nki` |
| `neuron_silu` | `nl.arange` + mask | `neuronxcc.nki` |
| `neuron_rope` | slicing + `div_ceil` | top-level `nki` |

We genuinely need both packages in one repository.

The nastier part: `hasattr(nl, "arange")` returns **True** under the package where it cannot
be resolved. There is no import-time feature detection, and the error text never hints that
the sibling package would work. nki-library source uses top-level `nki` while the public
tutorials use `neuronxcc.nki`, so anyone porting at scale meets this immediately.

### What to do

This is a question, not a patch. Ask the NKI team:

1. Which package is the supported long-term surface for kernel authors?
2. Are the capability gaps intentional (deliberate API narrowing) or drift?
3. If top-level `nki` is the future, can `nl.arange` be made to resolve there? If
   `neuronxcc.nki` is, can shape values support integer division?
4. In the meantime, can they publish a compatibility table? The table above is a start.

Lower urgency than 1, 2, and 4 — it's a productivity tax on kernel authors, not a blocker on
the customer experience. But it will bite any mass-porting effort on day one.

---

## Fix 4 — Add `nkilib` to the `python-depends` allowlist

### Why this is worth asking for

`nkilib` is **already installed** in the Neuron venv
(`/opt/.../site-packages/nkilib/`), and its production kernels are **directly callable from
PyTorch/XLA with correct results**. Verified against the installed `rope_hf` — the same
kernel we hand-ported:

| Calling strategy | Result |
|---|---|
| pass preallocated `q_out`/`k_out`, read the **return value** | **q cos_sim 1.000001, k cos_sim 1.000000** |
| pass preallocated outputs, read the **mutated arguments** | cos_sim **0.000000** — never written |

(Destination-passing is vestigial across the XLA boundary: outputs act as shape/dtype
templates, results come back via the return value. nki-library's own tests use
`must_alias_input`, which points a reader at the second strategy and silently gives zeros —
worth reporting to the nki-library team separately.)

So a thin-wrapper HF kernel works today:

```python
class NeuronRoPE(nn.Module):
    def forward(self, q, k, cos, sin, unsqueeze_dim=1):
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        return rope_hf(q, k, q_out, k_out, cos=cos, sin=sin)   # no vendoring
```

**The blocker is policy, not code.** And the scale argument is decisive: RoPE needed ~15
lines inlined; the MLP kernel's dependency closure is **7,249 lines across 22 files**
(~480x). Hand-porting does not reach the kernels that matter for performance.

### The patch

`kernels/python_depends.json` — `nki` is already there as precedent, so this copies its shape:

```json
     "neuron": {
       "nki": {
         "nix": [],
         "python": [{ "pkg": "nki", "import": "nki" }]
       },
+      "nkilib": {
+        "nix": [],
+        "python": [{ "pkg": "nki-library", "import": "nkilib" }]
+      }
     },
```

Note the pip package is `nki-library`, the import name is `nkilib`.

Requires Fix 2 to be useful — otherwise the neuron table still isn't consulted.

### The tradeoff to state honestly

A thin wrapper couples the HF kernel repo to both `nkilib` and `neuronx-cc` versions.
nki-library's README explicitly warns that GitHub `main` is not guaranteed compatible with a
given compiler version. Vendored kernels don't carry that coupling. Version coupling is more
tractable than maintaining hand-ports of 7,000-line kernels, but it is a real cost — don't
present this as free.

### What to do

1. Raise with Samir alongside Fix 1 — same team, same file area.
2. Bring the numbers: 15 lines vs 7,249, and the verified `cos_sim 1.000001`.
3. Ask how HF wants version coupling handled (pin in `python-depends`? a floor? nothing?).
4. Reproduction: `scripts/probe_nkilib_bundled.py` and
   `scripts/experiment_nkilib_thin_wrapper.py`.

---

## Not a fix: Finding #17 (weight layout) is a design decision

See `docs/poc-findings.md` #17 for the full analysis. Summarised here because it will come up
in the same conversations, and because it is the one item where **we should not propose a
patch** — we should ask a question.

Fused kernels want weights in a different layout than `nn.Linear` provides, and `kernelize()`
has no parameter-transformation hook. All three HF MLP weights are transposed relative to
`nkilib.core.mlp.mlp`, the transpose must be materialized, and every workaround is bad:
in-place mutation silently breaks `save_pretrained` round-tripping; a transposed copy roughly
doubles MLP weight memory; lazy-cache adds the same memory plus a first-step stall;
per-forward transposing costs three weight-sized HBM round trips per layer per step and
erases the speedup.

The two questions to put to the HF kernels team:

1. Should `kernels` gain a weight-transformation hook — something like
   `prepare_weights(module)` called once at kernelize time — with a defined contract about
   whether `state_dict()` reflects original or kernel layout?
2. Or is the intended pattern that backend kernels accept framework-native layouts and
   absorb the transpose internally? That's a request to nki-library rather than HF, and is
   arguably the cleaner division of responsibility since the kernel already knows its tiling.

This blocks *every* fused-kernel port on Neuron, not just the MLP, so it gates Week 5 MoE
work. It needs an answer before any fusion-API implementation starts.

---

## Filing order

1. **Fix 1** — biggest customer impact, and we have a verified patch plus a working demo.
   Strongest thing to lead with.
2. **Fix 2** — cheapest, entirely within Neuron's control, unblocks two things at once.
3. **Finding #17 question** — gates Week 5, so the answer is needed early even though it
   isn't a patch.
4. **Fix 4** — changes the shape of the whole program, but only becomes useful after Fix 2.
5. **Fix 3** — real, but a productivity tax rather than a customer blocker.
