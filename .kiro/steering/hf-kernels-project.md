# HuggingFace Kernels on Neuron — Project Context

## What this project is

A 6-week internship PoC: take NKI kernels from `aws-neuron/nki-library` (the production kernel library), package them for the HuggingFace `kernels` library (Kernel Hub), and validate that a stock HuggingFace model (Qwen3 dense) runs end-to-end on Trainium with `use_kernels=True` swapping in NKI-accelerated layers. If the dense flow lands cleanly, extend to Qwen3-MoE (router, top_k, blockwise MoE MLP). Final deliverable is a PoC document for the kernels team that captures what worked, what the Kernel Hub integration required, measured MFU impact, and a recommendation on whether Neuron should invest in first-class HF Kernel Hub support.

Ben's first half (the attention tutorial) already ramped him on NKI, the Neuron profiler, and the Native PyTorch beta setup. Week 1 is HF Kernels architecture ramp-up, not Trainium ramp-up.

## Why this project

HuggingFace's `kernels` library is a runtime kernel replacement system that swaps `nn.Module.forward()` methods with optimized implementations pulled from the Hub. It is now merged to transformers mainline. Adding a `"neuron"` device path to the kernel mapping gives every HuggingFace model with RMSNorm (87 model files), rotary embeddings (66 model files), and standard activations access to NKI kernels automatically, in eager mode, with graceful fallback when no Neuron kernel exists.

This is the highest-leverage HF ecosystem integration point for Neuron: per-kernel work that scales to the entire model zoo rather than per-model work. An intern PoC is the right vehicle to prove the mechanism end-to-end and hand the kernels team a validated path.

## Key concepts

- **KERNEL_MAPPING**: dict of `(layer_class_name, device) → kernel_impl`. Adding `"neuron"` entries gives all HF models with RMSNorm/RoPE/SiLU access to NKI kernels automatically.
- **Neuron device path**: the routing branch in `kernels` lib that selects `_NeuronRepos` when on Neuron hardware. Already merged to transformers mainline.
- **kernelize() flow**: walks model tree, matches layer names against KERNEL_MAPPING for current device, hot-swaps `forward()` method pointers. Module weights stay in place.
- **LocalLayerRepository**: local on-disk kernel repo for development without Hub publishing. Requires `__init__.py` + `metadata.json`.
- **KernelConfig(use_local_kernel=True)**: transformers-side API for local kernels. Format: `{"RMSNorm": "path/to/kernel:ClassName"}`.
- **Stateless kernel**: pure `nn.Module` subclass that reads weights from the adopting module via `self`. No `__init__`, only `forward()`. Declares `has_backward` / `can_torch_compile`.
- **Single-file kernel pattern (PR #46754)**: kernel class + `class layers:` namespace in one `__init__.py` file. This is the correct authoring pattern for Python-only kernels (NKI).
- **nki-library**: `aws-neuron/nki-library` — the production NKI kernel library. Source of kernels to port. Kernels are fused, have internal deps, and use different calling conventions than HF expects.

## Source of NKI kernels

**Use `aws-neuron/nki-library` (production library), NOT `nki-samples` (tutorial code).**

The PoC's value is documenting how to port production kernels at scale.

**Revised after Week 3 — the earlier blanket claim was drawn from one kernel.** "nki-library
kernels are fused, so they can't be ported" is true of RMSNorm and false of RoPE and MLP:

| Kernel | Standalone? | Fusion forced? | Verdict |
|--------|-------------|----------------|---------|
| `rmsnorm/rmsnorm_quant.py` | no | **yes** — always quantizes, `QuantizationType.NONE` unsupported | must reimplement |
| `embeddings/rope_hf.py` | **yes** | no | **already HF-shaped**; ported ✓ |
| `mlp/mlp.py` | **yes** | no — quant *and* norm both opt-in | feasible; blocked on weight layout (#17) |

RMSNorm is the outlier, not the archetype. Per-kernel friction actually encountered:
- Calling conventions differ (explicit args + dataclass vs reading `self.weight`)
- SPMD multi-core assumptions don't fit the per-layer swap model (strip to single-core)
- Destination-passing (`q_out`/`k_out` args) vs HF's return-value convention
- Dependency inlining cost varies enormously: RoPE needed **~15 lines**, the MLP kernel's
  closure is **7,249 lines across 22 files** (~480x) — hand-porting does not scale to it
- **NKI has no concatenation primitive**; `torch.cat` becomes writes into disjoint slices
  of a preallocated destination
- The two NKI import paths (`nki` vs `neuronxcc.nki`) have **different capabilities and
  neither is a superset**, so kernels are pinned per-idiom (#14)

**And the strategy may not need hand-porting at all: `nkilib` is already installed** in the
Neuron venv, and its production kernels are directly callable from PyTorch/XLA with correct
results (verified: installed `rope_hf` → `cos_sim 1.000001`). A thin wrapper is a few dozen
lines. The blocker is that `python-depends` whitelists `nki` but not `nkilib` — policy, not
code. See Finding #16 and `docs/upstream-fixes.md`.

See `docs/nki-library-porting-analysis.md` for full per-kernel analysis (RoPE, MLP, MoE).

## Kernel authoring pattern

Single-file, per PR #46754:
```python
# kernels/neuron_rmsnorm/__init__.py
class NeuronRMSNorm(nn.Module):
    has_backward = False
    can_torch_compile = False
    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states):
        # NKI kernel call here
        ...

class layers:
    NeuronRMSNorm = NeuronRMSNorm
```

Plus `metadata.json` with `{"backend": {"type": "neuron"}}`.

## Target model: Qwen3 dense

Qwen3 already opts into all three interception points upstream — no transformers-side model
changes needed. Registration counts are higher than originally estimated:

- **RMSNorm**: `@use_kernel_forward_from_hub("RMSNorm")` on `Qwen3RMSNorm`. Reads
  `self.weight` and `self.variance_epsilon`. **115 registrations** across transformers
  (est. was 87). Ported ✓
- **RoPE**: `@use_kernel_func_from_hub("rotary_pos_emb")` on
  `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)`, plus
  `@use_kernelized_func(apply_rotary_pos_emb)` on `Qwen3Attention`. **95 model files**
  (est. was 66). Needs `LocalFuncRepository`. Ported ✓
- **SiLU**: `@use_kernel_forward_from_hub("SiLU")` on `SiLUActivation` in
  `transformers/activations.py`. **One** decoration covers every model using
  `ACT2FN["silu"]` — the strongest form of the per-kernel leverage argument. Ported ✓

Two asymmetries that are easy to get wrong:
- **Layer** repos resolve `kernel.layers.<name>`; **function** repos resolve `<name>` at
  module top level. A function kernel placed inside `layers` will not be found.
- `has_backward` defaults to **True** for function kernels and **False** for layers. Set it
  explicitly.

Function kernels are also **process-global**: `@use_kernel_func_from_hub` creates a single
`Func` instance shared by every layer and every model in the process, so kernelizing one
model changes RoPE for all of them (#10).

`_KERNEL_MAPPING` also contains entries **no model registers** — `SwiGLUMLP`, `GeGLUMLP`,
`Linear` — so they are unreachable via the decorator path. `SwiGLUMLP` matters because fused
MLP is where the real MLP performance is; it needs the separate fusion API
(`register_kernel_replacements_and_fusions`), not a per-layer swap (#15, #17).

## Accuracy targets

- Cosine similarity > 0.999 against reference layer output
- For e2e: logits parity vs CPU/CUDA golden reference

### Non-negotiable: prove the kernel actually ran

Cosine similarity alone is **not** sufficient evidence. `@nki.jit` requires XLA tensors, so
every kernel needs a device guard, and the natural guard silently routes CPU tensors to the
PyTorch fallback — which is numerically correct. In Week 2 this made a whole suite pass
without executing a single NKI instruction (Finding #8).

Every accuracy test must therefore:
1. Run on Trainium via `require_neuron()`, which refuses to report results unless
   `xla_device_hw() == "NEURON"`
2. Place tensors on the XLA device, not the host
3. Assert via `nki_call_counter()` that the NKI branch ran and the fallback did **not**
4. Include a negative control proving the comparison can fail

Interpreting `max_diff`, which is op-dependent:
- **Reduction ops** (RMSNorm): expect ~1e-4. A diff of exactly `0.0` means the kernel
  did not run.
- **Elementwise ops** (RoPE, SiLU): bit-identical is *correct* — both backends do the same
  IEEE ops in the same order. Pass `expect_bit_identical=True`.

Helpers live in `tests/nki_test_utils.py`. Kernels must also `warn_once` on fallback with a
specific reason — never fall back silently.

## Upstream blockers (as of Week 3)

None of these are ours to merge, and none require architectural change. Full detail,
including exact code locations and ready-to-paste patches, in **`docs/upstream-fixes.md`**.

| # | Blocker | Owner | Size | Status |
|---|---------|-------|------|--------|
| 1 | `use_kernels=True` can't select the `"neuron"` device — silent no-op | transformers (+kernels) | ~12 lines, 3 sites | **fix verified sufficient** (0 → 9 swapped layers) |
| 2 | `_backend()` reports `cuda` on Neuron hosts | `torch_neuronx` | 1 attribute | root-caused; unblocks #7 *and* #12 |
| 3 | `nki` vs `neuronxcc.nki` capability split | NKI team | needs a decision | documented, no fix proposed |
| 4 | `nkilib` not on the `python-depends` allowlist | HF `kernels` | ~6 lines | `nki` already there as precedent |

**Practical consequence for Weeks 4-6:** `use_kernels=True` will not route to Neuron until
blocker 1 lands. Use `kernelize_for_neuron(model)` from
`scripts/neuron_kernel_registration.py`, which calls the kernels library directly with
`device="neuron"` and handles the `_hidden_kernels` attach/detach that function kernels need.

| 5 | Fused MLP divides by zero single-core when `intermediate_size > 4096` | nki-library | bug fix | **boundary measured, 10 data points**; no wrapper workaround |

**Fused-kernel work is blocked, not merely expensive.** Two independent gates, found by the
Week 4 derisking spike (`scripts/spike_nkilib_mlp.py`), which was worth running precisely
because it surfaced them before 2-3 weeks went into the integration:

- **Finding #18 (blocker 5):** `nkilib.core.mlp.mlp` fails to compile single-core above
  `intermediate_size = 4096`. Sharp boundary across 10 configs. Excludes Qwen3-8B (I=12288),
  Llama-3-8B and Mistral-7B (I=14336). A wrapper cannot work around it. Resolve this **first** —
  Finding #17 is moot until a kernel that compiles at useful sizes exists.
- **Finding #17:** `kernelize()` has no parameter-transformation hook for the weight-layout
  difference. A design decision for the HF kernels team. Note its premise was partly corrected
  by measurement — read the CORRECTION block in `docs/poc-findings.md` before quoting it.

The positive half of that spike: the production fused MLP kernel **is** drivable directly from
PyTorch/XLA with HF weights (cos_sim 0.999979-0.999995), which further supports the
thin-wrapper thesis in Finding #16.

**Performance measurement caveat (Finding #19).** Eager NKI dispatch costs ~0.36 ms of host
time per call vs ~0.011 ms for eager PyTorch — roughly 25x. At 217 kernel calls per Qwen3-8B
forward that is ~76 ms of host-side overhead per step. Two consequences:
- **Per-layer microbenchmarking cannot resolve kernel quality here** — overhead is 90%+ of
  measured latency at realistic shapes. Do not quote per-layer NKI-vs-eager ratios. Week 4
  full-model MFU is the right instrument, and should report launch count alongside MFU.
- This makes fusion *more* valuable, since one fused call replaces several dispatches.

## Week-by-week plan

### Week 1: HF Kernels architecture ramp-up and neuron-path verification ✓ DONE
- Verified `kernelize(device="neuron")` works on trn2
- Confirmed `LocalLayerRepository` loads local kernel packages
- Proved forward swap fires + fallback works
- Confirmed `KernelConfig(use_local_kernel=True)` accepts neuron mapping

### Week 2: RMSNorm NKI kernel, local validation on Qwen3 dense ✓ DONE (results superseded)
- Ported NKI RMSNorm kernel (tutorial-derived, production analysis documented)
- Documented nki-library porting friction (fusion, deps, interface mismatch)
- **The Week 2 accuracy numbers were invalid** and have been replaced. They reported
  `cos_sim = 1.000000, max_diff = 0.00e+00` because the tests fed CPU tensors, so the
  kernel silently took its PyTorch fallback and was compared against a mathematically
  identical reference. The NKI branch never executed. See Finding #8.
- Re-validated in Week 3 on the XLA device with execution asserted: **11/11 pass**,
  fp32 max_diff 1.2e-05 → 3.9e-04, bf16 3.1e-02 (cos_sim 0.999993). The kernel was
  correct all along; it had simply never been run.

### Week 3: Package RMSNorm for Hub, add RoPE, register neuron entries ✓ DONE (1 goal blocked upstream)
- **RoPE kernel: done.** Ported from *production* `nki-library/core/embeddings/rope_hf.py`
  (not a tutorial — none exists for RoPE). 20/20 accuracy + 6/6 guard cases, NKI execution
  asserted. Registered via `LocalFuncRepository`.
- **`"neuron"` mapping entries: done locally** for RMSNorm, `rotary_pos_emb`, and SiLU in
  `scripts/neuron_kernel_registration.py`. Upstream diff written as
  `PROPOSED_UPSTREAM_DIFF`; not submitted.
- **Hub packaging: partial.** Flat layout confirmed loadable; `digest` is optional;
  minimum repo is `__init__.py` + `metadata.json`. No upload — needs the repo-home
  decision (see Samir item) and Finding #12 for an honest dependency declaration.
- **`use_kernels=True` alone: BLOCKED UPSTREAM, cannot be met today.** Two independent
  causes; fails as a *silent no-op*. Root-caused, and the minimal fix is identified **and
  verified sufficient** (0 → 9 swapped layers). See Finding #9 and `docs/upstream-fixes.md`.
- **Samir coordination: still open.** Draft message ready at
  `deliverables/samir-hub-publishing-message.md`.
- **Stretch SiLU: done early.** 9/9 accuracy, wired into e2e.
- New findings #8-#17 in `docs/poc-findings.md`. The two that change planning:
  **#16** `nkilib` is already installed and its production kernels are directly callable,
  so thin-wrapper porting works today (blocker is the `python-depends` allowlist, i.e.
  policy not code); **#17** the fused MLP is blocked by weight layout, not by the kernel.

### Week 4: Full Qwen3 dense end-to-end, MFU measurement
- ~~SiLU NKI activation kernel + registration~~ — **done in Week 3**
- Full Qwen3 dense (full-size 8B, not the 2-layer stand-in) with NKI RMSNorm + RoPE + SiLU
- Confirm correctness (logits parity)
- Measure MFU with and without the kernels, **stating the denominator explicitly**
- Expect RMSNorm and RoPE to help and SiLU **not** to — standalone elementwise activations
  are memory-bandwidth bound. Measure; be willing to report neutral or negative.
- Confirm the RoPE `seq_len % 128` guard doesn't silently disable the kernel at realistic
  sequence lengths. This is the most likely way a customer gets no acceleration unnoticed.
- 1-2 day spike: call `nkilib.core.mlp.mlp` directly and validate against its own
  `mlp_torch_ref`, to derisk the highest-value kernel before any fusion-API work
- Note `use_kernels=True` will still not route to neuron unless the upstream fix lands;
  use `kernelize_for_neuron()` from `scripts/neuron_kernel_registration.py`
- Deliberately **not** planned: more elementwise activations (GeLU family). Hours each and
  broad coverage, but memory-bound — adds surface area without adding evidence.

### Week 5: Qwen3-MoE (stretch)
- Map Qwen3-MoE forward to Kernel Hub layer names
- Reuse RMSNorm/RoPE/SiLU from weeks 2-4 (all three already done)
- At least one MoE-specific NKI kernel swapped and validated, or gap analysis
- **Gated on Finding #17** — MoE kernels are fused and weight-layout-sensitive, so the
  weight-transformation question has to be answered before implementation starts
- Week 3 recon (see `docs/nki-library-porting-analysis.md`): `core/moe/moe_cte/moe_cte.py`
  weight layout is a **free reshape** for `Llama4TextExperts` but needs transposes for
  `Qwen3MoeExperts` — the reverse of the dense case. The real work is **routing metadata**
  (`token_position_to_id`, `block_to_expert`, `expert_affinities_masked`, a `[T+1, H]`
  hidden tensor with a padding row) which the kernel expects the *caller* to build, whereas
  the megablocks path HF wraps builds it internally. That gap is the port, and the
  thin-wrapper strategy does not shortcut it.
- Naming: `"Llama4TextMoe"` is **commented out** in `_KERNEL_MAPPING` ("no longer
  maintained"). The only live MoE layer name is `"MegaBlocksMoeMLP"`.
- Gap analysis is a legitimate outcome here — say so rather than forcing a thin result.

### Week 6: PoC document, review, and ship
- Kernel Hub mechanism and why forward-swap is the correct interception point
- **And where forward-swap runs out**: it works for weightless ops and ops reading weights
  as-is, and breaks for fused kernels wanting a different weight layout (#17). That boundary
  is a central finding, not a footnote.
- Upstream state: neuron device path merged but **unreachable** via `use_kernels=True` (#9);
  `_backend()` misreports (#7/#12); `kernel-builder` is a non-issue since NKI is pure Python
- What was validated: kernels, models, accuracy (with execution asserted), MFU delta
- **How we validated it**, and the Week 2 silent-fallback mistake (#8). This is the most
  transferable lesson in the PoC and generalizes to any accelerator whose kernels need a
  device guard. Do not bury it.
- Porting strategy recommendation: hand-port vs thin wrapper over `nkilib` (#16), with the
  15-lines-vs-7,249-lines scale argument and the version-coupling tradeoff stated honestly
- What is not done: backward kernels, torch.compile, Hub upload, MoE gaps, fused MLP
- Recommendation: is first-class HF Kernel Hub support worth engineering investment?
  Current lean, to be tested against Week 4 MFU: **yes, but the investment is 4 small
  upstream fixes plus one design decision, not a kernel-porting program.**

## Definition of done

**Floor (must hit):**
- ✓ `"neuron"` device support working locally with forward-swap proven on Trainium
  (via the kernels library with `device="neuron"`; the `use_kernels=True` entry point is
  blocked upstream — blocker 1)
- ✓ At least NKI RMSNorm and RoPE packaged and validated e2e on Qwen3 dense
  (both, plus SiLU; execution asserted, logits `cos_sim 1.000001`)
- ☐ Measured MFU delta with denominator stated — **Week 4**
- ☐ PoC document delivered to the kernels team — **Week 6**

**Ceiling (stretch):**
- ✓ SiLU activation kernel added (Week 3, early)
- ☐ MLP kernel — feasible but gated on Finding #17 (weight layout). Do the standalone
  `nkilib.mlp()` spike first; do not start the fusion integration until #17 is decided.
- ☐ Hub publishing working for a Neuron kernel — layout validated, blocked on the repo-home
  decision (Samir) and blocker 4 for an honest dependency declaration
- ☐ At least one Qwen3-MoE kernel swapped and validated. Note the MoE bottleneck is
  *routing metadata* built outside the kernel, not the matmul — the thin-wrapper strategy
  does not help here. A gap analysis may be the honest outcome.

## Environment (re-verified 2026-07-29)

| Package | Version |
|---------|---------|
| kernels | 0.15.2 (PyPI) |
| transformers | 5.15.0.dev0 (commit bb3ffb97) |
| torch | 2.9.1+cu128 (Neuron DLAMI) |
| neuronx-cc | 2.26.6360.0+6f180f47 |
| **nkilib** | **preinstalled in the venv** — every production kernel importable (#16) |
| Instance | trn2.3xlarge (1 device, 4 NeuronCores, 96 GB HBM), `xla_device_hw() == "NEURON"` |
| Neuron venv | `/opt/aws_neuronx_venv_pytorch_2_9` |
| SSH | `ssh trn2` (16.26.235.50, ubuntu, ben-ssh.pem) |

Setup on a fresh instance: clone, then
`source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && pip install -r requirements.txt`.
Use `make sync` (rsync) to iterate from a local checkout — tests must run on trn2.
`make test-nki`, `make test-e2e`, `make probe`, `make registration`.

Two environment gotchas that cost time:
- Our `kernels/` directory shadows the `kernels` pip package. Load local kernels via
  `importlib.util.spec_from_file_location` (see `tests/nki_test_utils.py::load_kernel_module`).
- `import kernels.layer.kernelize as kz` gives you the *function*, not the module —
  `kernels/layer/__init__.py` re-exports a same-named function. Use `importlib.import_module`.

## Documentation sources

| Source | What it covers |
|--------|---------------|
| [kernels docs — Layers](https://github.com/huggingface/kernels/blob/main/docs/source/layers.md) | kernelize(), use_kernel_mapping, LocalLayerRepository |
| [kernels docs — Requirements](https://huggingface.co/docs/kernels/kernel-requirements) | metadata.json schema, backend types, build variants |
| [transformers PR #46754](https://github.com/huggingface/transformers/pull/46754/files) | "Writing kernels" doc — single-file pattern, KernelConfig |
| [NKI Tutorial — RMSNorm](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/general/nki/tutorials/rmsnorm.html) | Reference NKI kernel implementation. Note: no RoPE tutorial exists anywhere in nki-samples. |
| [nki-library GitHub](https://github.com/aws-neuron/nki-library) | Production kernel source. Key paths: `core/embeddings/rope_hf.py` (HF-shaped RoPE, **absent from the public API reference**), `core/mlp/mlp.py`, `core/moe/moe_cte/`, `core/router_topk/`. |
| [nki.language API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.language.html) | `nl.silu`, `nl.gelu`, `nl.silu_dx` etc. exist natively. **No concatenation primitive** — `torch.cat` becomes writes into disjoint destination slices. |

**nki-library doc bugs found in Week 3, worth reporting upstream:** `rope_hf` is missing from
the public API reference entirely; the reference cites `nkilib.core.rope.RoPE` but the real
path is `nkilib.core.embeddings.rope.RoPE`; and `mlp()`'s mode assert message has PREFILL and
DECODE labelled backwards.

## Coordination

- **John**: **internship mentor.** Wrote the project guide and the week-by-week schedule in this
  document. The person to take *schedule* and *scope* decisions to — whether Week 5 becomes a gap
  analysis, whether Week 4 is still best spent on MFU, whether to contact HF-side people
  directly. Not a routing queue for bugs; bring him decisions, not tasks. Check with him before
  reaching out to external contacts. Draft check-in at `deliverables/john-mentor-checkin.md`.
- **Samir (arsamir)**: HF kernels team contact, Hub repo home decision.
  **Open as of Week 3.** Draft message ready at
  `deliverables/samir-hub-publishing-message.md` — covers the repo-home question plus the
  two `kernels`-side asks (device routing, `nkilib` allowlist) and the Finding #17 design
  question. Recommendation on repo home: `aws-neuron/`, so Neuron owns versioning.
- **Pinak (panpinak)**: SA team reviewer
- **Hanbo Wang / Karthick Gopalswamy**: kernels team (PoC recipients)
- **Matt (mmcclean)**: final deliverable recipient

## Tracking Documents — UPDATE THESE THROUGHOUT

These docs accumulate findings that become the final PoC. Update them as you work, not just at the end.

| Document | Purpose | When to update |
|----------|---------|----------------|
| `docs/sticking-points.md` | Running log of things that blocked or slowed progress | Every time something takes >10 min to debug or is harder than expected |
| `docs/customer-experience.md` | What a customer would struggle with today | When you hit setup friction, unclear errors, missing docs, or workflow gaps |
| `docs/porting-recommendations.md` | How the engineering team should port kernels at scale | When you learn something about nki-library structure, HF requirements, or automation opportunities |
| `docs/poc-findings.md` | Technical findings with severity ratings | When you discover a gap, API issue, or architectural mismatch |
| `docs/nki-library-porting-analysis.md` | Deep analysis of nki-library kernel structure | When you investigate a new kernel from nki-library |
| `docs/upstream-fixes.md` | The asks we're making of other teams, with exact locations + patches | When a blocker is found, root-caused, verified, or filed |
| `deliverables/week-N.md` | Weekly deliverable writeup | End of each week |

## What to Always Be Tracking

1. **Sticking points**: anything that took longer than expected, would trip up a customer, or reveals a systemic gap. Log it with time lost + who it affects.
2. **Customer experience**: imagine someone just did `pip install transformers` and wants NKI kernels. What's missing? What errors do they hit? What's underdocumented?
3. **Porting friction**: for each nki-library kernel you look at, note: is it fused? what deps does it pull? does the interface match HF? what would automation need?
4. **Recommendations**: concrete suggestions for the engineering team. Not just "this is hard" but "here's what to build/change to make it easy."
5. **Accuracy results**: always record cosine similarity, max abs diff, shapes tested, and whether NKI or fallback was used.
