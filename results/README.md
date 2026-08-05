# Results

**GENERATED FILE — do not edit.** Source of truth is [`measurements.json`](measurements.json); regenerate with `python scripts/render_results.py`.

Rendered 2026-08-05 18:28 UTC from commit `7af0eb5`.

## Read this before quoting any number

### Provenance

The trn2 instance used for the ORIGINAL measurements expired on 2026-08-02. Every raw artifact lived in /tmp on that host: measure_mfu --json-out files, neuron-explorer summary-json output, NEFF/NTFF profile pairs, and detached run logs. All are gone, permanently — a re-run produces new artifacts, not the lost ones. The original numbers survive because each run's stdout was pasted into the commit message at the time, and every producing script is committed. Every measurement has since been re-run on a replacement instance and those artifacts ARE committed (see _reproduction), so each number is now backed by a raw file plus an independent confirmation. Logged as sticking point #17: results should never have lived only on an ephemeral host, and the fix was one default path per script, not new code.

Every number below carries a `status`:

- **`transcribed`** (3 of 23) — Number is recorded here from the run's stdout, which was captured in the git commit message at the listed SHA. The raw artifact (JSON/log/NEFF) is NOT in the repo — see _artifact_loss.
- **`reproduced`** (15 of 23) — Original number is transcribed AND an independent re-run on the replacement instance is committed under results/raw/. The committed artifact carries the re-run's value, which is recorded alongside. See _reproduction for the agreement.
- **`in_repo`** (5 of 23) — Raw artifact is committed under results/raw/ and is the source of the number quoted here.

### The whole set was re-run on a second instance

The original host expired before any raw artifact was committed. Re-running everything on a replacement is the only way to turn transcribed numbers back into evidence, and it doubles as an independent check: a number that reproduces on different physical hardware is not a fluke of one instance.

- **When** 2026-07-29
- **Where** 16.51.184.34 (i-0b05f044388db8080), trn2.3xlarge, 1 device, 4 NeuronCores, LNC2 — same instance type and core config as the expired host, so figures are comparable
- **Command** `make results  (scripts/regenerate_results.py, 23 sequential stages)`
- **Outcome** All 23 stages exited 0. Absolute step times run up to ~8% higher on this physical instance; every RATIO and every conclusion reproduces, several to 3 significant figures.

| quantity | original | re-run | delta |
|---|---|---|---|
| baseline seq512 step | 42.04 ms | 45.15 ms | +7.4% absolute |
| kernelized seq512 fixed step | 141.43 ms | 151.81 ms | +7.3% absolute |
| kernelized/baseline slowdown, fixed | 3.36x | 3.36x | exact |
| kernelized/baseline slowdown, no fix | 208x | 198x | -5%, same order |
| Finding #24 fix speedup | 102.8x / 105.5x | 103.0x / 106.3x | +0.2% / +0.8% |
| host-issue share of wall time | 99.9% | 99.9% | exact |
| NKI marginal HBM traffic vs unfused floor | 1.00x both ops | 1.00x both ops | exact |
| device NKI/torch, silu and rmsnorm | 2.70x / 2.55x | 2.70x / 2.55x | exact |
| in-situ device share of the regression | 8.4% | 8.6% | +0.2pp |
| projected slowdown with dispatch fixed | 1.18x | 1.18x | exact |

The re-run is a reproduction, not a recovery. Where a measurement's committed artifact and its transcribed number differ, the artifact is the re-run and the transcribed number is the original; both are stated and neither is silently replaced.

Versions were checked before trusting any of it: Matched the recorded environment exactly before trusting any re-run: torch 2.9.1+cu128, torch_xla 2.9.0, neuronx-cc 2.26.6360.0+6f180f47 (DLAMI); kernels 0.15.2 and transformers 5.15.0.dev0 installed from requirements.txt, which pins transformers to commit bb3ffb97 rather than tracking main. nki reports 0.5.0+28631259367.ga768afa6 where the original run recorded plain 0.5.0 — same version, build suffix now captured.

### The number to lead with

Kernelizing Qwen3-0.6B costs **100 ms/step**, and that splits:

| term | ms | share |
|---|---|---|
| dispatch (framework overhead) | 91.608 | **91.6%** |
| device (forfeited compiler fusion) | 8.392 | 8.4% |

So the slowdown is overwhelmingly a **framework bug, not a property of the approach**. With dispatch fixed the projection was ~55 ms/step, about **1.18x** slower — PROJECTION, not measured.

**That projection has since been partly realised.** Both dispatch caches are now identified and fixed, and the slowdown is measured rather than projected:

| stage | seq | baseline ms | kernelized ms | slowdown | MFU | added ms/call |
|---|---|---|---|---|---|---|
| no fixes | 512 | 43.06 | 8873.67 | 206.07x | 0.02% | 52.2522 |
| #24 only | 512 | 44.36 | 146.67 | 3.31x | 1.45% | 0.6054 |
| #24 + B12 | 512 | 43.94 | 71.32 | 1.62x | 2.98% | 0.162 |
| #24 only | 2048 | 109.64 | 226.16 | 2.06x | 4.76% | 0.6894 |
| #24 + B12 | 2048 | 117.78 | 161.04 | 1.37x | 6.69% | 0.256 |
| device floor | | | | | | 0.0495 |

**52.2522 -> 0.6054 -> 0.162 ms per call.** 86.3x from the first fix, 3.7x from the second, **322.5x together**. Now within 3.3x of the device floor, and 69% of what remains is still dispatch.

The 1.15-1.18x figure recorded elsewhere is a PROJECTION that assumed the dispatch gap went to zero. 1.62x at seq 512 and 1.37x at seq 2048 are MEASURED, and they sit between the old 3.31x and that projection — which is what the projection predicted, since B12 removes about two thirds of the dispatch term rather than all of it.

The split has now been computed from FOUR independent wall-time pairs across two physical instances: 46.65/146.65, 47.52/144.19, 50.138/144.65 and 54.783/153.43. Device share lands at 8.4%, 8.6%, 8.9% and 8.5%; the projection at 1.18x, 1.18x, 1.17x and 1.15x. The conclusion does not depend on which pair is used.

Two figures elsewhere in this project are easy to quote out of context:

- **208x slower** — real, but that is *before* the one-line fix in Finding #24.
- **2.5–2.7x slower on device** — real, but from a chained microbenchmark that maximises the compiler's fusion advantage and so is NKI's worst case. In situ the device term is 8.4% of the regression.

## Control: is any of this a compiler-flag artifact?

Asked first, because a bad compiler default would be the cheapest possible explanation for the whole slowdown, and because it is the most plausible technical form of the objection that there should not be a slowdown at all.

28 chained SiLU applications, tile [512,3072] bf16, wall clock. One subprocess and one isolated compile cache per setting, so no setting inherits another's cache. Absolute times are not comparable across settings; the ratio is the measurement.

| `NEURON_CC_FLAGS` | NKI ms | torch ms | ratio |
|---|---|---|---|
| `(unset — project default)` | 14.096 | 0.728 | 19.366x |
| `--target trn2` | 14.061 | 0.764 | 18.393x |
| `--target trn2 --lnc 1` | 13.821 | 1.708 | 8.094x |
| `--target trn2 --lnc 2` | 14.019 | 0.726 | 19.321x |
| `--target trn2 -O2` | 14.152 | 1.023 | 13.832x |

Spread across settings: ratio **2.39x**, NKI **1.02x**, torch **2.35x**.

**NKI is INVARIANT across compiler settings — 13.82 to 14.15 ms, 1.02x spread. No setting rescues the NKI path, so the gap is not a compiler-flag artifact.** The ratio spread is driven entirely by torch moving: `--lnc 1` makes *torch* slower, which flatters the ratio without helping NKI.

*Scope limit.* Measures WALL time. 13.82 ms / 28 calls = 0.494 ms/call, which is the post-fix dispatch floor, so this run is ~97% dispatch. It establishes that the DISPATCH cost is flag-invariant and does NOT by itself settle whether the DEVICE-time gap in Findings #25/#26 depends on flags.

So the device half was measured separately. SiLU profiled at N=1 and N=28 under each of the five flag settings, one isolated compile cache per setting so no setting can serve another's NEFF. Reports device time at N=28 and, more diagnostically, marginal HBM traffic per call solved from the two call counts.

| `NEURON_CC_FLAGS` | NKI ms | torch ms | ratio | NKI MB/call | vs unfused floor |
|---|---|---|---|---|---|
| `(unset)` | 0.608 | 0.224 | 2.72x | 6.29 | 1.00x |
| `--target trn2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |
| `--target trn2 --lnc 1` | 0.58 | 0.429 | 1.35x | 6.29 | 1.00x |
| `--target trn2 --lnc 2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |
| `--target trn2 -O2` | 0.608 | 0.224 | 2.71x | 6.29 | 1.00x |

**No setting makes NKI faster than torch on device, and NKI's marginal traffic stays pinned at exactly 6.29 MB/call — the unfused floor — under every setting. The device gap is STRUCTURAL: an opaque custom call cannot be fused into its neighbours and no compiler flag reaches that.**

The result is not 'we tried five settings and none was better', which would leave a sixth setting open. It is that the quantity a better setting would have to move is already at its theoretical minimum. NKI moves one tile in and one tile out per call, which is the least an unfusable op can move. There is no headroom for a flag to find.

*On the 1.35x row.* The 1.35x at --lnc 1 is the best ratio in the table and is NOT an improvement: NKI barely moves (0.608 -> 0.580) while torch gets 91% slower (0.224 -> 0.429). Reading it as progress is the same mistake the first version of the wall-clock control made.

## MFU

Denominator stated explicitly: **316 TFLOPS** = 632 TFLOPS/device (TensorEngine bf16) / 2 for LNC2, 1 logical core used. (667 is the published figure; it includes VectorE and ScalarE.)

| configuration | step ms | MFU | NKI calls | vs baseline | re-run step ms |
|---|---|---|---|---|---|
| baseline, seq 512 | 42.04 | 5.05% | 0 | — | 44.36 |
| NKI SiLU only, seq 512, no fix | 1495.54 | 0.14% | 28 | — | — |
| all 3 kernels, seq 512, **no fix** | 8753.65 | 0.02% | 169 | 208x | 8873.67 |
| all 3 kernels, seq 512, **with fix** | 141.43 | 1.5% | 169 | 3.36x | 146.67 |
| baseline, seq 2048 | 108.76 | 9.9% | 0 | — | 109.65 |
| all 3 kernels, seq 2048, with fix | 223.99 | 4.81% | 169 | 2.06x | 226.16 |

The re-run column is the same configuration on a second physical instance. Step times run a few percent higher there; the slowdown ratios are what reproduce.

FLOPs per step: 670.42 GFLOP, computed explicitly rather than estimated.

## The fix (Finding #24)

| variant | ms/call | speedup | cos_sim |
|---|---|---|---|
| baseline (no override) | 51.74 | — | 0.999938 |
| NEURON_PLATFORM_TARGET_OVERRIDE=trn2 | 0.5 | 102.8x | 0.999938 |
| lru_cache(_detect_target) | 0.49 | 105.5x | 0.999938 |
| baseline again (control) | 51.43 | — | 0.999938 |

The override is set to whatever _detect_target() returns on the host, never hardcoded, because a wrong target would compile for the wrong hardware and could be silently wrong. cos_sim identical to 6 dp across all four, so neither fix changes what gets compiled. Baseline is re-run LAST as a control: 52.11 then 52.07 ms/call, within 0.1%, which rules out ordering or warm-up as the cause of the speedup.

## How the root cause was localised

| step | instrument | result | ruled out |
|---|---|---|---|
| 1 | torch-xla `ExecuteTime` counter | 28 NKI calls -> **1** device execution, 196-node graph | graph batching as the lever |
| 2 | neuron-explorer on that NEFF | device `total_time` **0.609 ms**, 43.0% MBU, 95.0% active | every device-side explanation |
| 3 | wall-clock split | **99.9%** of 1459.28 ms spent before `mark_step` | anything after dispatch |
| 4 | cProfile of one call | 51 of 52 ms in `select.poll` under `subprocess.check_output` | everything else |

Step 2 vs step 3 is the decisive comparison: 1459 ms wall against 0.609 ms device is a ~2396x ratio, which eliminates every device-side explanation simultaneously.

## The speedup: flash attention, seq 2048-3072

Findings #25/#26 produced a criterion — a kernel wins when it replaces a region the compiler would NOT otherwise fuse well AND there is real arithmetic to restructure — and then found no candidate meeting it. Attention is the first that passes both halves non-incidentally: flash attention is an ALGORITHMIC restructuring (online softmax, never materialising the [heads,S,S] score matrix), which a compiler does not derive because it fuses elementwise chains rather than re-deriving algorithms; and there are two matmuls per head with causal compute-skipping. Critically, attention_cte states it runs 'with 1D SPMD grid for LNC2 or without grid' — the exact property the fused MLP lacked, so #26's verdict does not carry over.

Qwen3-0.6B head geometry: 16 q heads, 8 kv heads (GQA group 2), head_dim 128, bf16, causal, batch 1, single logical core. 4 layers per graph with DISTINCT K/V per layer so neither side can amortise one weight load (the trap the first fused-MLP measurement fell into).

| seq | NKI ms/layer | torch ms/layer | NKI/torch | NKI HBM MB | torch HBM MB | score matrix MB |
|---|---|---|---|---|---|---|
| 512 | 0.2463 | 0.1225 | 2.01x slower | 8.39 | 3.16 | 8.4 |
| 1024 | 0.4939 | 0.4269 | 1.16x slower | 16.78 | 13.12 | 33.6 |
| 2048 | 1.1438 | 1.6902 | **1.47x FASTER** | 33.55 | 279.86 | 134.2 |
| 3072 | 1.8484 | 3.9062 | **2.13x FASTER** | 50.33 | 748.7 | 302.0 |
| 4096 | 2.8295 | 1.5784 | 1.79x slower | 67.11 | 395.05 | 536.9 |

**A speedup exists: up to 2.11x at seq 3072.** A WINDOW, not a threshold. Same kernel is 2.01x slower at seq 512 and 2.11x faster at seq 3072.

*Accuracy.* cos_sim 1.000010 (seq 512) to 1.001040 (seq 4096) against a CPU fp32 reference, on the FIRST run. max_abs 0.0006-0.0010. torch shows the same pattern at each length, so the comparison is fair.

*Why there is a lower edge.* At seq 512 torch moves 3.16 MB/layer, BELOW the 6.29 MB it costs merely to read q,k,v and write the output once. That is only possible if the compiler fused the whole chain and kept the score matrix resident — so at short sequences the compiler already achieves flash attention's central advantage, and the kernel pays an HBM round-trip at its custom-call boundary to buy something it does not get. Same fusion-barrier mechanism as Finding #25, from the opposite direction.

*Why there is an upper edge, which I first got backwards.* The 4096 reversal is on the TORCH side and my first hypothesis was backwards. I read it as the NKI kernel running out of single-core SBUF (K and V are 8.4 MB each; attention_cte only sections K/V above 10K tokens) — plausible, matched Finding #26, and blamed the kernel. The traffic column contradicts it: torch's HBM per layer goes 279.86 -> 748.70 -> 395.05 MB across seq 2048 -> 3072 -> 4096, DROPPING 47% while the score matrix grows from 302 to 537 MB. At 3072 it moves 2.5x the score matrix; at 4096 it moves 0.74x, less than one copy. Meanwhile NKI is exactly linear at every point (16.78 MB per 1024 tokens) and exactly on its time trend. Nothing degraded on the kernel side — the COMPILER got better, presumably switching attention strategy above a threshold. Checkable by diffing HLO either side of it; not done, so stated as what the traffic supports rather than confirmed.

*Reproduction.* The 2048-4096 region was measured TWICE in separate runs, with 3072 added the second time. It reproduces to four significant figures (NKI 1.1420/1.1438 at 2048, 2.8299/2.8295 at 4096), so the reversal is not noise. The wrong SBUF explanation survived only until the traffic column was read — a one-number benchmark would have produced a confident wrong answer.

*Porting cost.* Worked FIRST TRY against the HF-native layout: tp_q=True, tp_k=True, tp_out=False maps straight onto (batch*heads, seq, head_dim), and GQA is expressed natively as batch_size_kv < batch_size with no K/V replication. This is the first nkilib kernel in the project that dropped into the Kernel Hub's calling convention without a fight — a data point about porting cost as well as performance. Contrast Finding #13 (RoPE, undocumented) and #17/#18 (MLP, wrong weight layout plus a compile boundary).

*Dependency.* Both dispatch fixes are applied for this measurement. At 0.53 ms/call of overhead the seq-2048 win (1.14 ms/layer device) would have been invisible, so Findings #24 and #28 are what make an attention swap worth doing at all.

*Not done.* The kernel is called directly, NOT wired into the Kernel Hub. transformers has an attention interface that is the right interception point. Also: 4 chained layers is a microbenchmark, not a model — in situ attention sits between QKV and O projections that force HBM boundaries anyway, so the boundary should cost LESS than it does here, but that is unmeasured.

## Are the kernels any good? (Finding #25)

Device time only, dispatch excluded by construction. **Chained microbenchmark — NKI's worst case, see the caveat at the top.**

| op | impl | calls | device ms | HBM MB | MBU |
|---|---|---|---|---|---|
| silu | nki | 28 | 0.607 | 188.7 | 43.2% |
| silu | torch | 28 | 0.224 | 6.3 | 3.9% |
| rmsnorm | nki | 28 | 1.625 | 188.8 | 16.2% |
| rmsnorm | torch | 28 | 0.637 | 6.4 | 1.4% |
| silu | nki | 1 | 0.052 | 18.87 | 47.0% |
| silu | torch | 1 | 0.026 | 6.29 | 33.4% |
| rmsnorm | nki | 1 | 0.091 | 18.88 | 27.5% |
| rmsnorm | torch | 1 | 0.039 | 7.88 | 28.4% |

Solving `traffic(N) = FIXED + N x MARGINAL` across the N=1 and N=28 points: NKI marginal traffic is **6.29 MB/call = 1.00x the unfused floor** for both ops, and torch's is ~0 MB. So **the kernels are optimal** — they move the theoretical minimum for an op that cannot fuse — and the gap is the fusion the swap forfeits, not kernel quality.

Do not divide total traffic by N: a small NEFF carries fixed setup traffic that dominates at N=1, and doing so produced a false 'the kernels spill an fp32 intermediate' reading.

## The fused MLP — the one kernel that could have won (Finding #26)

| shape | blocks | impl | device ms | per block | HBM MB | MBU | cos_sim |
|---|---|---|---|---|---|---|---|
| H=1024 I=3072 | 28 | nki | 8.321 | 0.2972 | 2172.6 | 36.5% | 0.999979 |
| H=1024 I=3072 | 28 | torch | 2.782 | 0.0993 | 1059.1 | 53.2% | 0.999979 |
| H=4096 I=4096 | 8 | nki | 11.625 | 1.4532 | 3288.3 | 39.5% | 0.999977 |
| H=4096 I=4096 | 8 | torch | 4.18 | 0.5225 | 1619.1 | 54.1% | 0.999977 |

NKI/torch = **2.99x** at Qwen3-0.6B's MLP shape and **2.78x** at the largest shape it runs single-core. The gap barely narrows with scale, so it is not a shape artifact. Interpretation: nkilib kernels need a multi-core SPMD grid; single-core they tile far more finely than designed.

Weight-layout cost (Finding #17) quantified for the first time: the on-device transpose is 3.533 ms / 1172.3 MB at H=1024/I=3072. One-time at load, not per step.

Compile boundary, 10 data points (4 pass): **passes iff intermediate_size <= 4096; boundary is between 4096 and 4224; not a ratio effect**. REFRAMED by Finding #26: this is a design boundary, not a bug. A floordiv by zero on a shard-count calculation is what happens with no SPMD shard grid. Originally filed as an nki-library bug to fix. The reframe matters for the recommendation: 'the library has a bug at I>4096' invites a bug report, whereas 'the library assumes a multi-core shard grid we are not giving it' invites the correct question, which is whether per-layer swapping is the right integration model for these kernels at all.

## Correctness

| suite | result | seconds | cases |
|---|---|---|---|
| `tests/test_rmsnorm_nki.py` | PASS | 13.0 | 11/11 |
| `tests/test_rope_nki.py` | PASS | 16.9 | 20/20 + 6/6 guards |
| `tests/test_silu_nki.py` | PASS | 15.6 | 9/9 |
| `tests/test_qwen3_neuron_e2e.py` | PASS | 16.5 | — |
| `tests/test_qwen3_moe_e2e.py` | PASS | 16.1 | — |

End-to-end logits: Qwen3 dense `cos_sim 1.000001`, Qwen3-MoE `cos_sim 1.000002`. Every case asserts via a call counter that the NKI branch actually ran — a silent fallback is numerically correct and would otherwise pass.

Upstream coverage: 115 RMSNorm registrations, 95 RoPE model files, 1 SiLU decoration (one decoration covers every ACT2FN['silu'] user).

## Open items

- **[closed]** CLOSED — is any of this a compiler-flag artifact? — Was the top open item, on the theory that a bad compiler default could be the entire slowdown. Closed in both halves. Dispatch: NKI wall time is invariant across {unset, --target trn2, +--lnc 1, +--lnc 2, +-O2} at 1.02x spread. Device: NKI device time is invariant at 1.05x spread and NKI marginal HBM traffic is 6.29 MB/call — exactly the unfused floor — under every setting, so there is no headroom for a flag to find. Both probes run as harness stages, so a future stack change re-tests it automatically rather than requiring someone to remember.
- **[closed]** CLOSED — is the per-call create_computation rebuild cacheable? — Yes. It was 91% of what remained after Finding #24, and it is the same bug as #24: torch_xla's Op class already memoises the built computation, and NKI applies @xla_hlo_call inside TorchXlaKernel.__call__ so a fresh Op with an empty memo is created per call. Registering once per compile-cache key gives 0.528 -> 0.183 ms/call with bit-identical output, and takes the model-level slowdown from 3.31x to 1.62x at seq 512 and 1.37x at seq 2048. See the op-registry-cache and mfu-both-fixes measurements. What remains open is the UPSTREAM change: this is verified as a runtime monkeypatch, not shipped, and it needs an owner in the same way the #24 fix does.
- **[medium]** Can a NKI custom call participate in compiler fusion? — Decides whether the last ~18% after the dispatch fix is recoverable.
- **[medium]** Does a kernel spanning a fused region beat the compiler on that region? — The fused MLP answers this for single-core (no, by ~3x). Unmeasured multi-core with an SPMD grid, which is the configuration the kernel was built for.

## Environment

trn2.3xlarge, 1 Neuron device, 4 NeuronCores, LNC2 (2 logical cores), single logical core used unless stated.

**NEURON_CC_FLAGS unset for every measurement — compiler defaults throughout. This was the top open item, on the theory that a bad default could be the whole slowdown. Now CLOSED in both halves by two controls that sweep {unset, --target trn2, +--lnc 1, +--lnc 2, +-O2}. Wall clock: NKI is invariant at 1.02x spread (13.82-14.15 ms). Device: NKI is invariant at 1.05x spread (0.580-0.608 ms) and its marginal HBM traffic is 6.29 MB/call under every setting, which is exactly the unfused floor — one tile in, one tile out — so no flag has anywhere to go. See the compiler-flag-control and device-time-under-flags measurements.**

| package | version |
|---|---|
| `kernels` | 0.15.2 |
| `transformers` | 5.15.0.dev0 (commit bb3ffb97) |
| `torch` | 2.9.1+cu128 |
| `torch_xla` | 2.9.0 |
| `neuronx-cc` | 2.26.6360.0+6f180f47 |
| `nki` | 0.5.0+28631259367.ga768afa6 |

## Regenerating

On a fresh trn2 with the repo synced:

```bash
make results      # re-runs every measurement, writes raw artifacts to results/raw/
```

Individual measurements list their own command in `measurements.json`.

