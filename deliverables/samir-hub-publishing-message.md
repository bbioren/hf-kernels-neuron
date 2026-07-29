# Draft message to Samir (arsamir) — HF Kernel Hub for Neuron

**Status: DRAFT, NOT SENT.** Review before sending. Slack is probably the right channel; the
long version below works as a doc or email if he wants detail.

Judgement calls I made, change them if you disagree:
- Leads with the repo-home question since that's the actual ask, then the two `kernels`-side
  fixes, then the design question. Rationale: the first is a decision only he can make, the
  next two are small and concrete, the last needs thinking time.
- Recommends `aws-neuron/` for the repo home rather than leaving it open. Easier for him to
  push back on a proposal than to arbitrate an open question.
- Does **not** ask him to fix anything himself — just to point us at the right owner and say
  whether the approach is sound.
- Mentions Finding #8 (our own testing mistake) briefly. It's a bit exposing, but it's the
  most useful thing we learned for other backend authors, and volunteering it buys
  credibility for the rest.

---

## Short version (Slack)

> Hi Samir — Ben here, wrapping up week 3 of the PoC putting NKI kernels on the HF Kernel
> Hub for Trainium. Three NKI kernels (RMSNorm, RoPE, SiLU) now swap into a stock Qwen3 and
> run on Neuron hardware, logits matching at cos_sim 1.000001. RoPE is a real port of our
> production kernel library, and it goes in as a `FuncRepository` against the existing
> `rotary_pos_emb` name.
>
> Four things I'd like your read on, roughly in order of how much they block us:
>
> **1. Where should Neuron kernels live on the Hub — `kernels-community/` or `aws-neuron/`?**
> This is the one blocking us from publishing. My lean is `aws-neuron/`, so we own versioning
> and can ship fixes against specific `neuronx-cc` releases without going through
> kernels-community review. But if you'd rather they sit in `kernels-community/` for
> discoverability, that's fine and I'll follow your call. It also decides the `repo_id` in the
> `_KERNEL_MAPPING` entries we want to PR.
>
> **2. `use_kernels=True` can't reach a `"neuron"` mapping entry today, and it fails
> silently.** transformers' `kernelize()` derives the device from `model.device.type`, which
> on Neuron is `"cpu"` or `"xla"`, never `"neuron"`. Because a `Device` object is passed rather
> than a string, it skips validation, matches nothing, and returns success with every layer
> unchanged. I've got a ~3-line fix (map `xla` → `neuron` when
> `xla_device_hw() == "NEURON"`) and I verified it's sufficient — it takes Qwen3 from 0 to 9
> swapped layers via the transformers path. Happy to open the issue/PR; mainly want to know
> whether the helper belongs in `kernels` and gets imported by transformers, or is duplicated
> in both.
>
> **3. Could `nkilib` be added to `python_depends.json` under the `neuron` backend?**
> You already whitelist `nki` there, which was a nice surprise. `nkilib` (our production
> kernel library) turns out to be preinstalled on the Neuron DLAMI and its kernels are
> directly callable — I validated our production RoPE kernel through it at cos_sim 1.000001.
> That would let a Neuron kernel be a ~40-line wrapper instead of a hand-port. For scale:
> the kernel I did port needed ~15 lines of deps inlined; the MLP kernel's closure is ~7,250
> lines across 22 files, so hand-porting doesn't reach the kernels that actually matter for
> perf.
>
> **4. A design question I don't think we should answer unilaterally.** Fused kernels want
> weights in a different layout than `nn.Linear` gives (all three of our MLP weights are
> transposed vs what our kernel wants), and `kernelize()` only rewrites `forward` — there's
> no parameter-transformation hook. Mutating params in place breaks `save_pretrained`
> round-tripping; keeping a transposed copy roughly doubles MLP weight memory; transposing
> per-forward erases the speedup. Is a `prepare_weights(module)`-style hook something you'd
> want, or is the expectation that backend kernels absorb the layout difference internally?
> This blocks any fused-kernel work for us, so I'd rather ask than guess.
>
> Happy to write any of these up properly or jump on a call. Full findings doc is coming at
> the end of week 6 — I can share the current draft if useful.

---

## Long version (email / doc, if he wants detail)

Subject: **HF Kernel Hub on Trainium — repo home decision + 2 small `kernels` fixes + 1 design question**

Hi Samir,

Ben Bioren, AWS Neuron. I'm three weeks into a PoC packaging NKI (Neuron Kernel Interface)
kernels for the HuggingFace `kernels` library, to see whether Neuron should invest in
first-class Kernel Hub support. Wanted to share where it's landed and ask you four things.

### Where it is

Three NKI kernels swap into a stock Qwen3 and execute on Trainium:

| Kernel | Interception point | Registrations in transformers | Accuracy |
|--------|-------------------|-------------------------------|----------|
| RMSNorm | `@use_kernel_forward_from_hub("RMSNorm")` | 115 | 11/11 cases |
| RoPE | `@use_kernel_func_from_hub("rotary_pos_emb")` | 95 model files | 20/20 + 6/6 guards |
| SiLU | `@use_kernel_forward_from_hub("SiLU")` | 1 decoration, covers all `ACT2FN["silu"]` users | 9/9 cases |

End-to-end on Qwen3: RMSNorm executes 9× per forward, RoPE 2×, SiLU 2×, zero fallbacks,
logits `cos_sim 1.000001` against the unkernelized model.

The mechanism itself works well. The interception points already exist upstream, Qwen3
already opts into all three, and the forward-swap model is the right shape for eager-mode
Neuron. Nothing we hit needs architectural change.

One thing worth flagging for other backend authors, since we lost a week to it: our first
round of accuracy tests all passed while never executing a single NKI instruction. `@nki.jit`
requires XLA tensors, so kernels need a device guard, and the natural guard silently routes
host tensors to the PyTorch fallback — which is numerically correct. Our tests fed CPU
tensors and compared the fallback against a mathematically identical reference, reporting a
flawless `max_diff = 0.00e+00`. The perfect score *was* the bug. We now assert via a call
counter that the kernel actually ran. Which leads into a suggestion at the bottom.

### 1. Repo home — the decision blocking us

Should Neuron kernels live under `kernels-community/` or `aws-neuron/`?

**My recommendation: `aws-neuron/`.** NKI kernels are validated against specific
`neuronx-cc` compiler versions, and we'd want to ship fixes on our own cadence without
routing through kernels-community review. Discoverability is the tradeoff; if you think that
outweighs it, I'm happy to go the other way.

Either way it decides the `repo_id` in the `_KERNEL_MAPPING` entries we'd like to PR:

```python
"RMSNorm":        { "neuron": LayerRepository(repo_id="<...>/rmsnorm", layer_name="NeuronRMSNorm", version=1) }
"SiLU":           { "neuron": LayerRepository(repo_id="<...>/silu",    layer_name="NeuronSiLU",    version=1) }
"rotary_pos_emb": { "neuron": FuncRepository(repo_id="<...>/rope", func_name="apply_rotary_pos_emb", version=1) }
```

On packaging mechanics, in case it saves you answering: we don't need `kernel-builder`. NKI
kernels are pure Python (`@nki.jit`), so there's no compile step and a flat repo works —
`__init__.py` + `metadata.json` at the root, no `build/<variant>/` directory. We confirmed
`digest` is optional and the other six metadata fields are required. The variant-structured
layout does *not* work for us yet, for the reason in item 2b below.

### 2. Two small `kernels`-side fixes

**2a. Device routing (blocks the whole user-facing experience).**

`use_kernels=True` can never select a `"neuron"` mapping entry. transformers'
`kernelize(model, mode)` has no `device` parameter and derives everything from
`model.device.type`, which on Neuron is `"cpu"` (params on host) or `"xla"` (on device) —
never `"neuron"`.

Worse, it's silent rather than an error. `kernelize` only calls `_validate_device_type` when
it receives a device *string*; transformers passes a `Device` object, so `Device(type="xla")`
passes through unvalidated, matches nothing, and returns success with every layer untouched.

The fix is one branch, in `hub_kernels.py::kernelize`:

```python
     device_type = model.device.type
     if device_type == "cuda" and is_rocm_platform():
         device_type = "rocm"
+    elif device_type == "xla" and _is_neuron_xla():
+        device_type = "neuron"
     device = Device(type=device_type)
```

where `_is_neuron_xla()` checks `xm.xla_device_hw(xm.xla_device()) == "NEURON"` — verified
to return exactly that on trn2, no new dependency, fails closed.

**I verified this is sufficient** rather than just proposing it: applied in-process, it takes
Qwen3 from 0 → 9 swapped RMSNorm layers through the transformers path, logits `cos_sim
1.000001`. (We initially proposed patching `kernels._find_device` instead — that would have
done nothing, since transformers never calls it on this path. The test caught it.)

Two other sites want the same treatment: `kernel_config.py::infer_device` (the `KernelConfig`
path) and `kernels/layer/kernelize.py::_find_device` (direct kernels-library callers).

Question for you: should `_is_neuron_xla()` live in `kernels` and be imported by
transformers, or be duplicated? Happy to open the issues and PRs either way.

**2b. `nkilib` on the `python-depends` allowlist.**

Nice surprise: `python_depends.json` already has a `neuron` backend section whitelisting
`nki`. Two asks around it.

First, a bug: that entry is currently unreachable. `validate_dependencies()` looks up the
table for whatever `_backend()` reports, and on the Neuron DLAMI `_backend()` returns
`CUDA(version=12.8)` — because the root check is `hasattr(torch, "neuron")`, which is False
even after `import torch_neuronx`. So declaring `python-depends: ["nki"]` fails with
`unsupported kernel dependency: nki`, and a Neuron kernel has to ship `python-depends: []`
while importing `nki`. That's on us to fix in `torch_neuronx` (set the attribute) — flagging
it only because the same root cause is why variant-structured Hub layouts don't resolve for
us either, which is why we're proposing flat repos in item 1.

Second, the ask: could `nkilib` be whitelisted alongside `nki`?

```json
     "neuron": {
       "nki": { "nix": [], "python": [{ "pkg": "nki", "import": "nki" }] },
+      "nkilib": { "nix": [], "python": [{ "pkg": "nki-library", "import": "nkilib" }] }
     },
```

`nkilib` is our production kernel library. It turns out to be preinstalled on the Neuron
DLAMI, and its kernels are directly callable from PyTorch/XLA — I validated our production
RoPE kernel through it at `cos_sim 1.000001`. That makes a Neuron Hub kernel a ~40-line
wrapper rather than a hand-port.

The scale argument is what makes this matter. The RoPE kernel I hand-ported needed ~15 lines
of internal deps inlined. The MLP kernel's dependency closure is ~7,250 lines across 22
files. Hand-porting doesn't reach the kernels that actually drive performance; wrapping does.

I'll be upfront about the tradeoff: a wrapper couples the kernel repo to both `nkilib` and
`neuronx-cc` versions, and our own README warns that library `main` isn't guaranteed
compatible with a given compiler. I think that's more tractable than maintaining hand-ports
of 7,000-line kernels, but it's a real cost. If you have a view on how you'd want version
coupling expressed, I'd rather design to it than around it.

### 3. A design question: weight layouts for fused kernels

This is the one I don't think we should answer on our own.

The per-layer forward swap works cleanly for weightless ops (RoPE, SiLU) and for ops that
read weights as-is (RMSNorm). It breaks down when a kernel wants weights in a different
layout — which is exactly the fused, matmul-heavy kernels where the performance is.

Concretely: all three of Qwen3's MLP weights are transposed relative to what our fused MLP
kernel wants (`gate_proj`/`up_proj` are `[I,H]` and it wants `[H,I]`; `down_proj` the
reverse). `.t()` is a free view in torch, but the kernel DMAs from HBM assuming row-major, so
it has to be materialized. And `kernelize()` only rewrites `forward` — it never touches
parameters, and there's no hook where a one-time transform could live.

Every workaround is bad in a different way:

| Approach | Problem |
|---|---|
| Mutate params in place at kernelize time | `state_dict()` / `save_pretrained()` now emit weights a stock `Qwen3MLP` can't load — silent checkpoint corruption |
| Hold a second transposed copy | ~2x MLP weight memory; MLP dominates an 8B model, so close to 2x model memory |
| Transpose lazily, cache | same memory cost plus a first-step stall |
| Transpose every forward | three weight-sized HBM round trips per layer per step — erases the speedup |

So: is a `prepare_weights(module)`-style hook (called once at kernelize time, with a defined
contract about whether `state_dict()` reflects original or kernel layout) something you'd
want in `kernels`? Or is the expectation that backend kernels accept framework-native layouts
and absorb the transpose internally — in which case this is a request to our kernel library,
not to you, and arguably the cleaner split since the kernel knows its own tiling.

This blocks all fused-kernel work for us, so an early read would help a lot even if the
answer is "we haven't decided".

### 4. One suggestion, take it or leave it

Given how we lost a week: would `kernels` consider exposing which implementation is live per
layer — something like `model.get_kernel_report()`?

The issue isn't specific to us, but it's sharper on Neuron: a fallback is numerically
correct, so a user cannot distinguish "accelerated" from "no-op" by looking at outputs.
Combined with a silent fallback, there's no signal at all. A `use_fallback=False` strict mode
exists in `kernels` but transformers doesn't expose it, and even that only helps if you know
to ask.

### What I'm not asking for

Nothing here needs `kernel-builder` work, and I'm not asking you to write any of the fixes —
just for the repo-home call, a sanity check on the routing approach, a yes/no on `nkilib`,
and a read on the weight-layout question. Happy to do the issues and PRs.

Full PoC writeup lands end of week 6, with MFU numbers. Glad to share the current draft or
walk through any of this live.

Thanks,
Ben

---

## Follow-ups to track after sending

- [ ] Repo home decided → fill `repo_id` in `scripts/neuron_kernel_registration.py::PROPOSED_UPSTREAM_DIFF`
- [ ] Routing fix: confirm helper location, then open transformers issue + `kernels` companion
- [ ] `nkilib` allowlist: yes/no; if yes, also needs our `torch_neuronx` fix to be useful
- [ ] Weight-layout question: answer, or an owner and a timeline
- [ ] Kernel-report suggestion: interest level
- [ ] Ask separately whether HF wants the Finding #8 silent-fallback writeup as a docs
      contribution — it generalizes to any accelerator whose kernels need a device guard
