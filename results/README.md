# Results

**GENERATED FILE — do not edit.** Source of truth is [`measurements.json`](measurements.json); regenerate with `python scripts/render_results.py`.

Rendered 2026-08-03 16:55 UTC from commit `b68a706`.

## Read this before quoting any number

### Raw artifacts are missing

The trn2 instance used for all measurements expired on 2026-08-02. Every raw artifact lived in /tmp on that host: measure_mfu --json-out files, neuron-explorer summary-json output, NEFF/NTFF profile pairs, and detached run logs. All are gone. The numbers survive because each run's stdout was pasted into the commit message at the time, and every producing script is committed. Nothing here is unreproducible, but nothing here is currently backed by a raw file either. `make results` regenerates the whole set into results/raw/ on a fresh instance. Logged as a sticking point: results should never have lived only on an ephemeral host.

Every number below is marked `transcribed` (from the run's stdout, captured in the listed commit message) or `in_repo` (raw artifact under `results/raw/`). At time of writing all are `transcribed`.

### The number to lead with

Kernelizing Qwen3-0.6B costs **100 ms/step**, and that splits:

| term | ms | share |
|---|---|---|
| dispatch (framework overhead) | 91.608 | **91.6%** |
| device (forfeited compiler fusion) | 8.392 | 8.4% |

So the slowdown is overwhelmingly a **framework bug, not a property of the approach**. With dispatch fixed the projection is ~55 ms/step, about **1.18x** slower — PROJECTION, not measured.

Two figures elsewhere in this project are easy to quote out of context:

- **208x slower** — real, but that is *before* the one-line fix in Finding #24.
- **2.5–2.7x slower on device** — real, but from a chained microbenchmark that maximises the compiler's fusion advantage and so is NKI's worst case. In situ the device term is 8.4% of the regression.

## MFU

Denominator stated explicitly: **316 TFLOPS** = 632 TFLOPS/device (TensorEngine bf16) / 2 for LNC2, 1 logical core used. (667 is the published figure; it includes VectorE and ScalarE.)

| configuration | step ms | MFU | NKI calls | vs baseline |
|---|---|---|---|---|
| baseline, seq 512 | 42.04 | 5.05% | 0 | — |
| NKI SiLU only, seq 512, no fix | 1495.54 | 0.14% | 28 | — |
| all 3 kernels, seq 512, **no fix** | 8753.65 | 0.02% | 169 | 208x |
| all 3 kernels, seq 512, **with fix** | 141.43 | 1.5% | 169 | 3.36x |
| baseline, seq 2048 | 108.76 | 9.9% | 0 | — |
| all 3 kernels, seq 2048, with fix | 223.99 | 4.81% | 169 | 2.06x |

FLOPs per step: 670.42 GFLOP, computed explicitly rather than estimated.

## The fix (Finding #24)

| variant | ms/call | speedup | cos_sim |
|---|---|---|---|
| baseline (no override) | 51.74 | — | 0.999938 |
| NEURON_PLATFORM_TARGET_OVERRIDE=trn2 | 0.5 | 102.8x | 0.999938 |
| lru_cache(_detect_target) | 0.49 | 105.5x | 0.999938 |
| baseline again (control) | 51.43 | — | 0.999938 |

The override is set to whatever _detect_target() returns on the host, never hardcoded, because a wrong target would compile for the wrong hardware and could be silently wrong. cos_sim identical to 6 dp across all four, so neither fix changes what gets compiled.

## How the root cause was localised

| step | instrument | result | ruled out |
|---|---|---|---|
| 1 | torch-xla `ExecuteTime` counter | 28 NKI calls -> **1** device execution, 196-node graph | graph batching as the lever |
| 2 | neuron-explorer on that NEFF | device `total_time` **0.609 ms**, 43.0% MBU, 95.0% active | every device-side explanation |
| 3 | wall-clock split | **99.9%** of 1459.28 ms spent before `mark_step` | anything after dispatch |
| 4 | cProfile of one call | 51 of 52 ms in `select.poll` under `subprocess.check_output` | everything else |

Step 2 vs step 3 is the decisive comparison: 1459 ms wall against 0.609 ms device is a ~2396x ratio, which eliminates every device-side explanation simultaneously.

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

Compile boundary, 10 data points (4 pass): **passes iff intermediate_size <= 4096; boundary is between 4096 and 4224; not a ratio effect**. REFRAMED by Finding #26: this is a design boundary, not a bug. A floordiv by zero on a shard-count calculation is what happens with no SPMD shard grid. Originally filed as an nki-library bug to fix.

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

- **[high]** No run has confirmed the results are independent of compiler flags — NEURON_CC_FLAGS was unset for every measurement. A config artifact is the one thing that could invalidate the device-time comparisons in Findings #25 and #26, and it is also the most plausible technical form of the reviewers' objection that there should not be a slowdown. Now INSTRUMENTED rather than merely noted: scripts/probe_compiler_flags.py sweeps {unset, --target trn2, +--lnc 1, +--lnc 2, +-O2}, one subprocess and one isolated compile cache per setting, and reports whether the NKI/torch RATIO moves. It runs as the fourth stage of `make results`, so a fresh instance answers this before spending an hour on measurements that a bad default would invalidate. Verdict threshold: ratio spread <1.25x closes the item.
- **[high]** Is the per-call create_computation rebuild cacheable? — 91.6% of the remaining regression. Not attempted: inside torch_xla's op-registry path, and a wrong guess could be silently incorrect rather than error.
- **[medium]** Can a NKI custom call participate in compiler fusion? — Decides whether the last ~18% after the dispatch fix is recoverable.
- **[medium]** Does a kernel spanning a fused region beat the compiler on that region? — The fused MLP answers this for single-core (no, by ~3x). Unmeasured multi-core with an SPMD grid, which is the configuration the kernel was built for.

## Environment

trn2.3xlarge, 1 Neuron device, 4 NeuronCores, LNC2 (2 logical cores), single logical core used unless stated.

**NEURON_CC_FLAGS unset for every measurement — compiler defaults throughout. This is a known open item: no run has confirmed the results are independent of default target/LNC selection.**

| package | version |
|---|---|
| `kernels` | 0.15.2 |
| `transformers` | 5.15.0.dev0 (commit bb3ffb97) |
| `torch` | 2.9.1+cu128 |
| `torch_xla` | 2.9.0 |
| `neuronx-cc` | 2.26.6360.0+6f180f47 |
| `nki` | 0.5.0 |

## Regenerating

On a fresh trn2 with the repo synced:

```bash
make results      # re-runs every measurement, writes raw artifacts to results/raw/
```

Individual measurements list their own command in `measurements.json`.

