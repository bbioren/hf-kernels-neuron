# Draft message to Samir (arsamir) — HF Kernel Hub for Neuron

**Status: DRAFT, NOT SENT.** Review before sending. Slack for the short version; the detail below
works as a doc or a thread reply if he asks.

## Version history — worth reading, because three earlier drafts should not have gone out

- **v1** told Samir the Kernel Hub's per-layer granularity might be structurally wrong for Neuron, on
  the strength of a 208x slowdown. **Wrong, and the cause was ours** — an uncached `neuron-ls`
  subprocess in NKI's dispatch path. Would have handed the HF team a false problem statement about
  their own design.
- **v2** corrected that and narrowed the ask to our residual ~0.59 ms/call, framed as ours to fix.
- **v3** led with a real finding: with dispatch excluded, our kernels are 2.5–2.7x slower on device in
  a chained microbenchmark, because a NKI custom call is opaque to the compiler and each swap forfeits
  a fusion. Correct as far as it went, and it implied a conclusion that has since turned out to be
  **incomplete**: that per-layer swapping has a structural ceiling on a fusing-compiler backend.
- **v4 (this one)** has the result that changes the conclusion. Pointed at the right op — flash
  attention rather than RMSNorm/RoPE/SiLU — per-layer swapping **wins by 2.11x**. So the mechanism is
  sound; what we had wrong was which ops to aim it at. That is a more useful thing to tell a
  kernels-library maintainer than a complaint, and it generalises past Neuron to any backend with a
  fusing compiler.

Also new since v3: a **second** dispatch caching bug, which takes the model-level cost from 3.31x to
1.62x. Both bugs were ours, both were one line.

### Judgement calls I made — change them if you disagree

- Leads with the finding, not the asks. He is a maintainer; the interesting content is what we learned
  about where his library pays off on a new backend.
- States the ops we picked first were the wrong ones. That is embarrassing and it is the substance.
- Ask 3 explicitly says **don't merge it yet**, because it depends on a fix on our side landing first.
  Better than having him ship something inert.
- Cut the Slack version from ~1,300 words to ~825, and the whole file from ~3,760 to ~1,970. v3's
  "short version" was not a Slack message. 825 is still long, and it is the floor I could reach while
  keeping the two tables — those are the substance, and paraphrasing them would just invite him to ask
  for the numbers.

---

## Short version (Slack)

> Hi Samir — Ben here, wrapping up a PoC putting NKI kernels on the HF Kernel Hub for Trainium. Three
> kernels (RMSNorm, RoPE, SiLU) swap into stock Qwen3 and stock Qwen3-MoE and run on Neuron hardware,
> logits matching at `cos_sim 1.000001` / `1.000002`, zero code changes between the two model families.
> RoPE is a real port of our production kernel library, as a `FuncRepository` against the existing
> `rotary_pos_emb` name.
>
> **One finding I think is useful to you, then four asks.**
>
> On a backend with a fusing compiler, the ops the Hub intercepts most widely are the ones with the
> least to gain. Device time, each NKI kernel vs the torch op it replaces, 28 chained applications,
> dispatch excluded:
>
> ```
>                  device ms   marginal HBM traffic per call
>   NKI SiLU          0.609    6.29 MB = 1.00x the unfused floor
>   torch SiLU        0.224    ~0.00 MB
>   NKI RMSNorm       1.626    6.29 MB = 1.00x the floor
>   torch RMSNorm     0.636    ~0.00 MB
> ```
>
> Our kernels move exactly the theoretical minimum for an op that can't fuse — one tile in, one tile
> out — so this isn't kernel quality. Torch's marginal traffic is ~0 because the compiler collapses the
> chain into a single pass. **You can't beat not touching memory.**
>
> But that's about which ops, not about the mechanism. Pointed at the op the analysis actually favours —
> our flash attention kernel, an *algorithmic* restructuring rather than a fusion the compiler already
> does — per-layer swapping wins:
>
> ```
>   seq     NKI     torch   result
>    512   0.246    0.123   2.01x slower
>   2048   1.144    1.690   1.48x FASTER
>   3072   1.848    3.906   2.11x FASTER
>   4096   2.830    1.578   1.79x slower
> ```
>
> A window, not a threshold: below ~1.5K the compiler keeps the S×S score matrix resident and there's
> nothing to win; above it the matrix stops fitting. The 4096 edge is the *compiler* improving — its HBM
> traffic drops 47% there while the score matrix grows — not the kernel degrading. It also dropped into
> the Hub's calling convention first try, GQA and all, `cos_sim 1.000010`.
>
> **The rule, if it's useful:** a Hub kernel wins when it replaces a region the compiler wouldn't fuse
> anyway *and* there's real arithmetic to restructure. Attention qualifies; norms and activations can't.
> I'd expect that to hold for any XLA-style backend, not just Neuron.
>
> Four asks, in order of how much they block us:
>
> **1. Where should Neuron kernels live — `kernels-community/` or `aws-neuron/`?** The only one actually
> blocking publishing. My lean is `aws-neuron/` so we can ship against specific `neuronx-cc` releases,
> but I'll follow your call. It also decides the `repo_id` in the `_KERNEL_MAPPING` entries we'd PR.
>
> **2. `use_kernels=True` can't reach a `"neuron"` entry today, and it fails silently.** `kernelize()`
> derives the device from `model.device.type`, which on Neuron is `"cpu"` or `"xla"`, never `"neuron"`.
> A `Device` object is passed rather than a string, so validation is skipped, nothing matches, and it
> returns success with every layer unchanged. I have a ~4-line fix — map `xla` → `neuron` when
> `xla_device_hw() == "NEURON"`, mirroring the existing ROCm branch — verified sufficient: 0 → 9 swapped
> layers through the transformers path, logits unchanged. Happy to open the PR; mainly want to know
> whether the helper belongs in `kernels` or transformers.
>
> **3. `nkilib` in `python_depends.json` under the `neuron` backend — queued, not merged yet.** You
> already whitelist `nki` there, which was a nice surprise. But `_backend()` reports `cuda` on a Neuron
> host (`hasattr(torch, "neuron")` is False), so the `neuron` section is never read and our kernels ship
> `"python-depends": []` while importing `nki`. **That's ours to fix** — `torch_neuronx` should set the
> attribute — and until it lands an `nkilib` entry would be inert. Once it works, a Neuron kernel can be
> a ~40-line wrapper instead of a hand-port; the MLP kernel's dependency closure is ~7,250 lines across
> 22 files, so hand-porting doesn't reach the kernels that matter for perf.
>
> **4. Weight layouts for fused kernels — a design question I don't want to answer unilaterally.**
> Fused kernels want a different layout than `nn.Linear` gives (all three of our MLP weights are
> transposed relative to what the kernel wants), and `kernelize()` only rewrites `forward`, so there's
> no parameter-transformation hook. Mutating params breaks `save_pretrained` round-tripping; a
> transposed copy roughly doubles MLP weight memory; transposing per-forward erases the gain. Is a
> `prepare_weights(module)`-style hook something you'd want, or should backend kernels absorb the layout
> difference internally? This blocks fused-kernel work for us.
>
> Happy to send the full writeup. Nothing needs a fast answer except maybe (1).

---

## Detail, if he asks

### Where it is

| | status |
|---|---|
| Kernels | RMSNorm, RoPE, SiLU — single-file, `metadata.json` with `{"backend": {"type": "neuron"}}` |
| Models | Stock **unmodified** transformers architectures — `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`. No fork, no patched model code. Configs are constructed in code with random weights, **not pretrained checkpoints**: dense MFU uses Qwen3-0.6B's real shape (28 layers, hidden 1024, intermediate 3072); correctness runs use 2-layer configs (MoE: 4 experts, top-k 2, hidden 256) |
| Accuracy | isolated layers `cos_sim 1.000000`; e2e logits `1.000001` (dense) / `1.000002` (MoE) |
| Provenance | RoPE ported from `nkilib.core.embeddings.rope_hf`. RMSNorm and SiLU are tutorial-derived **because no standalone versions exist in nkilib** — `rmsnorm_quant.py` always quantises and there is no activations module |
| Loading | flat layout, `LocalLayerRepository`. Works, but only because build-variant resolution fails and falls back to importing the repo root |
| Published | **nothing.** No external side effects were permitted for this PoC |

### The two dispatch bugs, both ours

Worth telling him because the second one lives in `torch_xla`'s op registry, which he may see bite
other XLA backends.

| | the cache that exists | how it was defeated | effect |
|---|---|---|---|
| 1 | `func._nki_compile_cache` | target resolution runs while building the cache *key*, so a hit still forks `neuron-ls` | 52.25 → 0.605 ms/call |
| 2 | `xla_op_registry.Op._computations` | NKI applies `@xla_hlo_call` *inside* `__call__`, so a fresh `Op` with an empty memo is built per call | 0.605 → 0.162 ms/call |

**322x** off the per-call overhead. Model-level: 206x slower → 3.31x → **1.62x** at seq 512, **1.37x**
at seq 2048. `torch_xla`'s own docstring asks callers to register ops globally "in order to amortize
the lowering cost", which is exactly what NKI wasn't doing.

Both verified accuracy-neutral with the baseline re-run last as a control, and cosine similarity
bit-identical to 16 digits. Neither is shipped — they're runtime patches with a source fingerprint, so
they refuse to apply to an NKI version they weren't written against.

### Why the small-op result is trustworthy

Don't divide total traffic by N. A small NEFF carries fixed setup traffic that dominates at N=1, and
doing so gave us a false "the kernels spill an fp32 intermediate" reading. Solving
`traffic(N) = FIXED + N × MARGINAL` across N=1 and N=28 is what shows NKI's marginal traffic is exactly
the 6.29 MB floor and torch's is ~0.

Two caveats we put in front of our own numbers:

- The 2.5–2.7x is a **chained microbenchmark** — 28 identical ops back to back — which maximises the
  compiler's fusion opportunity and is therefore our worst case. In a real forward pass the device term
  is only **8.4–8.9%** of the regression; the rest was our dispatch overhead.
- We also ruled out a compiler-flag artifact rather than assuming: across
  `{unset, --target trn2, ±--lnc 1/2, -O2}`, NKI's device time varies 1.05x and its marginal traffic
  is pinned at the floor under every setting. So there's no headroom a flag could find.

### Interception coverage, for context on ask 1

115 RMSNorm registrations, 95 RoPE model files, and one `ACT2FN` decoration covering every activation
user. Wider than we assumed — which is why the "reach and benefit are inversely correlated" result is
the interesting one rather than a footnote.

### What I'm not asking for

- Not asking him to fix our dispatch bugs. Both were ours, both are diagnosed.
- Not asking for a `kernel-builder` Neuron target. It would make Neuron a first-class backend rather
  than a fallback path, but NKI kernels are pure Python so nothing we need is blocked on it. Mention
  only if he asks what a complete integration would need.
- Not asking him to endorse the attention result. It's a 4-layer microbenchmark, not in a model, and
  the kernel isn't wired through the Hub yet — it's called directly.

---

## Follow-ups to track after sending

1. Repo home decision → unblocks publishing and fixes the `repo_id` in the mapping PR.
2. Whether the `xla` → `neuron` helper lives in `kernels` or transformers → decides where our PR goes.
3. `torch.neuron` attribute with the `torch_neuronx` owners → must land before the `nkilib` allowlist
   entry does anything.
4. His answer on the weight-layout hook → gates any fused-kernel work.
5. If he's interested in the attention finding, the obvious next step is wiring it through
   transformers' attention interface rather than calling it directly. That's the thing that would turn
   a microbenchmark into a real integration result.
