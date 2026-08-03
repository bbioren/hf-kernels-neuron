# Week 5 — Qwen3-MoE gap analysis

**Status of the Week 5 goal.** The plan was "map Qwen3-MoE forward to Kernel Hub layer names,
reuse RMSNorm/RoPE/SiLU, and get at least one MoE-specific NKI kernel swapped and validated,
or a gap analysis." This is the gap analysis, and the reasoning for why it is the honest
outcome rather than a fallback.

**Summary.** The three dense kernels transfer to Qwen3-MoE for free — RMSNorm, RoPE, and SiLU
all sit on interception points the MoE model shares. The MoE-*specific* work does not, and it
is blocked by three separate things, only one of which is about the kernel. Attempting an
implementation would have produced either a toy that doesn't run at real expert widths, or a
reimplementation of routing logic that already exists on the framework side.

---

## What transfers for free — now MEASURED, not assumed

Verified on trn2 (`tests/test_qwen3_moe_e2e.py`), Qwen3-MoE, 2 layers, 4 experts, top-k 2,
seq 128:

| Kernel | Interception point | Dispatch | Result |
|--------|-------------------|----------|--------|
| RMSNorm | `@use_kernel_forward_from_hub("RMSNorm")` on `Qwen3MoeRMSNorm` | `nki=9 fallback=0` | transfers |
| RoPE | `@use_kernel_func_from_hub("rotary_pos_emb")` | `nki=2 fallback=0` | transfers |
| SiLU | decorated once in `activations.py` | `nki=2 fallback=0` | transfers |

Logits `cos_sim = 1.000002` against the unkernelized model. **Zero code changes to the
kernels.** Week 5's "reuse RMSNorm/RoPE/SiLU" goal is met.

This is the load-bearing evidence for the "per-kernel, not per-model" thesis: the same three
kernels now work across two model architectures with no modification, because the interception
points are shared.

### But first you have to make Qwen3-MoE run on Neuron at all

**The default experts implementation does not work on Neuron.** With no kernelization
whatsoever, a plain forward fails:

```
[NCC_EVRF029] Operation sort is not supported on trn2. Use supported equivalent operation
like TopK or replace it with an alternate implementation via Neuron Kernel Interface (NKI).
```

`grouped_mm_experts_forward` (the default) calls `torch.sort` and `torch.histc`; `histc`
lowers to an unsupported `sort` HLO. Fix:

```python
config = Qwen3MoeConfig(..., experts_implementation="batched_mm")
```

`batched_mm` has no `sort`/`histc`/`nonzero`/`unique`/`bincount` in its path and runs fine.
Nothing documents this, and the error names an HLO op with no hint that a config flag resolves
it. See Finding #22.

**This changes the recommendation below about which MoE kernel to write.** The thing actually
blocking Qwen3-MoE on Neuron is a routing histogram, not the expert matmul — and the compiler
error itself points at NKI as the remedy. A small NKI kernel for the `sort`/`histc` step (or
wiring up `nkilib/core/router_topk` and `core/topk`) would unblock the *default* MoE path
entirely, and unlike the expert matmul it is not blocked by Findings #17 or #18.

What it does *not* cover is the part that makes MoE expensive — expert routing and the
blockwise grouped matmul.

---

## Blocker 1 — the only live MoE layer name is `MegaBlocksMoeMLP`

Counting decorator registrations in transformers `5.15.0.dev0`:

| Kernel name | Registrations | Status |
|---|---|---|
| `MegaBlocksMoeMLP` | 2 | live |
| `Llama4TextMoe` | 1 | **commented out** in `_KERNEL_MAPPING` — "NOTE: No longer maintained" |

`Qwen3MoeSparseMoeBlock` / `Qwen3MoeExperts` carry **no** kernel decorator at all. So there is
no Qwen3-MoE-specific interception point to target. Targeting MoE through this mechanism means
either:

- conforming to the `MegaBlocksMoeMLP` interface, which is a different model family's module
  shape, or
- getting a `Qwen3MoeExperts` decoration added upstream, or
- going through the fusion API (`register_kernel_replacements_and_fusions`), which is the more
  invasive integration this PoC has not validated.

This is the same structural issue as `SwiGLUMLP` in the dense case (Finding #15): the mapping
contains names no model registers, and the names models *do* register don't line up with the
kernels we have.

---

## Blocker 2 — weight layout, and it cuts the opposite way from dense

`nkilib/core/moe/moe_cte/moe_cte.py` expects:

```
gate_up_proj_weight   [E, H, 2, I_TP]
down_proj_weight      [E, I_TP, H]
```

Against what transformers holds:

| Model | `gate_up_proj` | `down_proj` | Transform needed |
|-------|----------------|-------------|------------------|
| `Llama4TextExperts` | `[E, H, 2I]` (gate-first `chunk(2,-1)`) | `[E, I, H]` | **free reshape** |
| `Qwen3MoeExperts` | `[E, 2I, H]` | `[E, H, I]` | **transposes** |

So the kernel is a good fit for Llama4 and a poor one for Qwen3-MoE — exactly inverted from
the dense MLP case, where HF's `nn.Linear` layout needed transposing. That inversion is worth
noting on its own: **weight-layout compatibility is per-model-family, not per-backend**, so
"does the NKI kernel match HF's layout" has no single answer.

And the transform has nowhere to live, for the same reason as Finding #17: `kernelize()`
rewrites `forward` and never touches parameters.

---

## Blocker 3 — routing metadata is the actual work, and no dependency change helps

This is the one that makes an implementation attempt unwise rather than merely awkward.

`moe_cte` requires the **caller** to supply:

| Input | What it is |
|-------|-----------|
| `token_position_to_id [N*B]` | token-to-slot mapping after sorting by expert |
| `block_to_expert [N,1]` | which expert each block belongs to |
| `expert_affinities_masked [(T+1)*E, 1]` | per-token per-expert gate weights, masked |
| hidden `[T+1, H]` | with a **padding row** |

Token sorting, block assignment, and the padding convention all live *outside* the kernel. The
megablocks path HuggingFace wraps builds that metadata internally, so an HF-shaped kernel would
have to reproduce it — in PyTorch, on device, per step.

That is the port. It is not a kernel-writing exercise; it is reimplementing a scheduling layer
and matching another implementation's conventions exactly. And note it is the one part of the
MoE story that the Finding #16 thin-wrapper strategy does **not** shortcut: wrapping
`nkilib` gets you the matmul, not the metadata.

`nkilib/core/router_topk/router_topk.py` maps cleanly onto `Qwen3MoeTopKRouter`
(x@w+bias → softmax/sigmoid → top-k → scatter affinities; constraints T≤2048, E≤512,
`H % 128 == 0`, K≤8) but writes into caller-allocated mutable outputs rather than returning, so
it needs an allocation wrapper. It is also the only file in this area importing
`neuronxcc.nki` (for `nki.typing`), which given Finding #14 means it targets the older API.

`nkilib/core/moe_block/moe_block_tkg.py` fuses rmsnorm + router + MoE but is **decode-only**,
so it is not applicable to a training or prefill measurement.

---

## Why the fused-MLP result decides this

The dense fused MLP spike (Finding #18) found that `nkilib.core.mlp.mlp` cannot compile
single-core above `intermediate_size = 4096`, because the CTE sharding heuristic forces
inter-dimension sharding above that width and there is no SPMD grid to shard across.

MoE expert widths are in the same regime — Qwen3-MoE-30B-A3B has `moe_intermediate_size` in
the thousands per expert, and the aggregate is far larger. There is no reason to expect the MoE
CTE path to behave differently, and it shares the same tiling and sharding utilities
(`tile_info.py`, `kernel_helpers.py`) that produced the divide-by-zero.

So the likely outcome of a Week 5 implementation attempt is the same wall, reached three weeks
later and with a routing-metadata reimplementation in between. **Verifying that hypothesis is
cheap** (one call to `moe_cte` at realistic expert width, single-core) and is listed as a
next step below rather than assumed.

---

## The structural conclusion, which is the real deliverable

This is worth separating from the Qwen3-MoE specifics, because it generalizes and it belongs
in the Week 6 recommendation.

**The HF Kernel Hub's per-layer forward swap and nki-library's fused kernels are built for
different execution models, and MoE is where the mismatch is total.**

| | Kernel Hub assumes | nki-library fused kernels assume |
|---|---|---|
| Granularity | one layer, one `forward()` | a fused multi-op region |
| Cores | whatever the model is on (one, in eager) | SPMD across logical cores at useful widths |
| Weights | as `nn.Module` holds them | a layout chosen for the kernel's tiling |
| Scheduling metadata | built by the framework | supplied by the caller |

Our three dense kernels work precisely because they sit on the benign side of every row:
single op, single core, weights read as-is, no metadata. RMSNorm, RoPE and SiLU are the
*easy* cases, and the mechanism handling them well should not be read as evidence it will
handle MoE.

That is not an argument against investing — it is an argument about *where* to invest. The
per-kernel work that scales across the model zoo (norms, rotary, activations) is exactly what
the Kernel Hub is good at. The fused, sharded, metadata-heavy kernels need either a different
integration point (the fusion API, plus a weight-lifecycle contract) or a different mechanism
entirely.

---

## What would need to be true for MoE to work here

Ordered, with dependencies:

1. **Finding #18 resolved** — fused CTE kernels must compile single-core at realistic widths,
   or the Kernel Hub must be able to launch SPMD from a per-layer swap. Owner: nki-library, then
   HF. Without this, nothing below matters.
2. **A weight-lifecycle contract** (Finding #17) — somewhere for a one-time layout transform to
   live, with a defined `state_dict()` story. Owner: HF kernels team.
3. **A Qwen3-MoE interception point**, or agreement to target `MegaBlocksMoeMLP`'s module shape.
   Owner: transformers.
4. **A routing-metadata story** — either nki-library exposes an entry point that builds its own
   metadata from `(hidden, router_logits, top_k)`, or the HF wrapper reimplements it. The former
   is a much better division of responsibility and is the concrete ask.
5. **`nkilib` on the `python-depends` allowlist** (Finding #16) — otherwise the wrapper carries
   an undeclarable dependency. Owner: HF.

Items 1 and 4 are the load-bearing ones. Item 4 in particular is worth raising with the
nki-library team as a design request, not a bug: *for a framework-integration use case, kernels
that require caller-built scheduling metadata are much harder to adopt than kernels that build
it internally.*

---

## Revised recommendation: write the routing kernel, not the expert matmul

Finding #22 changes the priority order. The highest-value MoE NKI work is no longer the
blockwise expert matmul:

1. **A NKI kernel for the routing histogram / sort step.** This is the one that unblocks the
   *default* Qwen3-MoE path on Neuron, the compiler explicitly recommends NKI for it, and it is
   small. Critically, it is **not** blocked by Findings #17 (weight layout — a histogram has no
   weights) or #18 (single-core width limits — it is not a fused matmul). Of everything surveyed
   in this project, this is the MoE work most likely to succeed.
   - Building blocks exist: `nkilib/core/topk/` and `nkilib/core/router_topk/`.
   - Interception is the open question: there is no decorated hook for it, so it may need a
     transformers-side change or the fusion API.
2. **`router_topk` wrapped standalone**, validated against `Qwen3MoeTopKRouter`. Clean
   interface, needs only an allocation wrapper (it writes into caller-allocated mutable
   outputs). Would establish whether any MoE-specific NKI kernel can be driven from
   PyTorch/XLA.
3. **The expert matmul (`moe_cte`)** drops to last. It is gated on #17 and #18, needs the
   routing metadata reimplemented, and per Finding #20 would not be a speedup in eager mode
   anyway.

### Still-cheap diagnostics

- **Confirm the width hypothesis for `moe_cte`.** Call it single-core at a realistic expert
  width and see whether it hits the same `floordiv` divide-by-zero as the dense MLP. Turns
  "likely the same wall" into a measured fact.
- ~~Validate the dense three on Qwen3-MoE~~ — **done**, see above.

### And the Finding #20 caveat over all of it

Per-invocation cost is ~53 ms in eager mode, so **no** MoE NKI kernel will be a speedup on the
eager per-layer path. That does not make the routing kernel pointless — it would make
Qwen3-MoE *run* on Neuron with the default configuration, which is a correctness/coverage win
independent of performance. But it should be pitched as unblocking, not accelerating, until the
graph-mode question (Finding #21) is settled.
