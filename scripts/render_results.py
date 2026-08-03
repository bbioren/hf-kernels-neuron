"""Render results/README.md from results/measurements.json.

measurements.json is the single source of truth. This script generates the human-readable summary
so a number can never disagree between the two. Run it after editing the JSON:

    python scripts/render_results.py

Runs anywhere — no Neuron hardware needed.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "results" / "measurements.json"
OUT = ROOT / "results" / "README.md"


def git(*args, default="unknown"):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return default


def by_id(ms):
    return {m["id"]: m for m in ms}


def main():
    d = json.loads(SRC.read_text())
    about = d["_about"]
    ms = by_id(d["measurements"])
    den = d["denominator"]
    L = []

    L.append("# Results")
    L.append("")
    L.append("**GENERATED FILE — do not edit.** Source of truth is "
             "[`measurements.json`](measurements.json); regenerate with "
             "`python scripts/render_results.py`.")
    L.append("")
    L.append(f"Rendered {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} "
             f"from commit `{git('rev-parse', '--short', 'HEAD')}`.")
    L.append("")

    # --- the honest bit, first -----------------------------------------------------------
    L.append("## Read this before quoting any number")
    L.append("")
    L.append("### Raw artifacts are missing")
    L.append("")
    L.append(about["_artifact_loss"])
    L.append("")
    L.append("Every number below is marked `transcribed` (from the run's stdout, captured in the "
             "listed commit message) or `in_repo` (raw artifact under `results/raw/`). "
             "At time of writing all are `transcribed`.")
    L.append("")
    L.append("### The number to lead with")
    L.append("")
    ins = ms["in-situ-device-vs-dispatch"]
    L.append(f"Kernelizing Qwen3-0.6B costs **{ins['wall_gap_ms']:.0f} ms/step**, and that splits:")
    L.append("")
    L.append("| term | ms | share |")
    L.append("|---|---|---|")
    L.append(f"| dispatch (framework overhead) | {ins['dispatch_gap_ms']:.3f} | "
             f"**{ins['dispatch_gap_pct']}%** |")
    L.append(f"| device (forfeited compiler fusion) | {ins['device_gap_ms']:.3f} | "
             f"{ins['device_gap_pct']}% |")
    L.append("")
    proj = ins["projected_with_dispatch_fixed"]
    L.append(f"So the slowdown is overwhelmingly a **framework bug, not a property of the "
             f"approach**. With dispatch fixed the projection is ~{proj['step_ms']:.0f} ms/step, "
             f"about **{proj['slowdown_vs_baseline']}x** slower — {proj['status']}.")
    L.append("")
    L.append("Two figures elsewhere in this project are easy to quote out of context:")
    L.append("")
    L.append("- **208x slower** — real, but that is *before* the one-line fix in Finding #24.")
    L.append("- **2.5–2.7x slower on device** — real, but from a chained microbenchmark that "
             "maximises the compiler's fusion advantage and so is NKI's worst case. In situ the "
             f"device term is {ins['device_gap_pct']}% of the regression.")
    L.append("")

    # --- MFU ------------------------------------------------------------------------------
    L.append("## MFU")
    L.append("")
    L.append(f"Denominator stated explicitly: **{den['per_core_denominator_tflops']} TFLOPS** "
             f"= {den['device_tflops_tensorengine_bf16']} TFLOPS/device (TensorEngine bf16) "
             f"/ {den['logical_cores_per_device']} for LNC2, "
             f"{den['logical_cores_used']} logical core used. "
             f"({den['device_tflops_published_bf16']} is the published figure; it includes "
             f"VectorE and ScalarE.)")
    L.append("")
    L.append("| configuration | step ms | MFU | NKI calls | vs baseline |")
    L.append("|---|---|---|---|---|")
    order = [
        ("mfu-baseline-512", "baseline, seq 512"),
        ("mfu-silu-only-512", "NKI SiLU only, seq 512, no fix"),
        ("mfu-kernelized-512-nofix", "all 3 kernels, seq 512, **no fix**"),
        ("mfu-kernelized-512-fixed", "all 3 kernels, seq 512, **with fix**"),
        ("mfu-baseline-2048", "baseline, seq 2048"),
        ("mfu-kernelized-2048-fixed", "all 3 kernels, seq 2048, with fix"),
    ]
    for k, label in order:
        m = ms[k]
        sd = m.get("slowdown_vs_baseline")
        L.append(f"| {label} | {m['step_ms']} | {m['mfu_per_core_pct']}% | "
                 f"{m.get('nki_calls_per_step', 0)} | {str(sd) + 'x' if sd else '—'} |")
    L.append("")
    L.append(f"FLOPs per step: {ms['mfu-baseline-512']['flops_per_step_gflop']} GFLOP, "
             f"computed explicitly rather than estimated.")
    L.append("")

    # --- the fix --------------------------------------------------------------------------
    fx = ms["fix-verification"]
    L.append("## The fix (Finding #24)")
    L.append("")
    L.append("| variant | ms/call | speedup | cos_sim |")
    L.append("|---|---|---|---|")
    for v in fx["variants"]:
        sp = f"{v['speedup']}x" if v["speedup"] else "—"
        L.append(f"| {v['variant']} | {v['ms_per_call']} | {sp} | {v['cos_sim']} |")
    L.append("")
    L.append(fx["notes"])
    L.append("")

    # --- localisation chain ---------------------------------------------------------------
    L.append("## How the root cause was localised")
    L.append("")
    gb = ms["graph-batching"]
    hv = ms["host-vs-device-split"]
    dp = ms["device-profile-28-nki-calls"]
    L.append("| step | instrument | result | ruled out |")
    L.append("|---|---|---|---|")
    L.append(f"| 1 | torch-xla `ExecuteTime` counter | 28 NKI calls -> "
             f"**{gb['variants'][0]['device_executions']}** device execution, "
             f"{gb['graph_nodes_28_calls']}-node graph | graph batching as the lever |")
    L.append(f"| 2 | neuron-explorer on that NEFF | device `total_time` "
             f"**{dp['device_total_time_ms']} ms**, {dp['mbu_pct']}% MBU, "
             f"{dp['total_active_time_pct']}% active | every device-side explanation |")
    L.append(f"| 3 | wall-clock split | **{hv['host_issue_pct']}%** of "
             f"{hv['wall_ms']} ms spent before `mark_step` | anything after dispatch |")
    L.append(f"| 4 | cProfile of one call | 51 of 52 ms in `select.poll` under "
             f"`subprocess.check_output` | everything else |")
    L.append("")
    L.append(f"Step 2 vs step 3 is the decisive comparison: "
             f"{hv['wall_ms']:.0f} ms wall against {dp['device_total_time_ms']} ms device is a "
             f"~{hv['wall_ms'] / dp['device_total_time_ms']:.0f}x ratio, which eliminates every "
             f"device-side explanation simultaneously.")
    L.append("")

    # --- kernel quality -------------------------------------------------------------------
    nvt = ms["nki-vs-torch-device"]
    L.append("## Are the kernels any good? (Finding #25)")
    L.append("")
    L.append("Device time only, dispatch excluded by construction. **Chained microbenchmark — "
             "NKI's worst case, see the caveat at the top.**")
    L.append("")
    L.append("| op | impl | calls | device ms | HBM MB | MBU |")
    L.append("|---|---|---|---|---|---|")
    for r in nvt["rows"]:
        L.append(f"| {r['op']} | {r['impl']} | {r['calls']} | {r['device_ms']} | "
                 f"{r['hbm_mb']} | {r['mbu_pct']}% |")
    L.append("")
    mt = nvt["marginal_traffic_regression"]
    L.append(f"Solving `traffic(N) = FIXED + N x MARGINAL` across the N=1 and N=28 points: "
             f"NKI marginal traffic is **{mt['silu_nki_marginal_mb']} MB/call = "
             f"{mt['silu_nki_vs_floor']:.2f}x the unfused floor** for both ops, and torch's is "
             f"~{mt['silu_torch_marginal_mb']:.0f} MB. So **the kernels are optimal** — they move "
             f"the theoretical minimum for an op that cannot fuse — and the gap is the fusion the "
             f"swap forfeits, not kernel quality.")
    L.append("")
    L.append("Do not divide total traffic by N: a small NEFF carries fixed setup traffic that "
             "dominates at N=1, and doing so produced a false 'the kernels spill an fp32 "
             "intermediate' reading.")
    L.append("")

    # --- fused MLP ------------------------------------------------------------------------
    fm = ms["fused-mlp-vs-torch"]
    L.append("## The fused MLP — the one kernel that could have won (Finding #26)")
    L.append("")
    L.append("| shape | blocks | impl | device ms | per block | HBM MB | MBU | cos_sim |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in fm["rows"]:
        L.append(f"| {r['shape']} | {r['blocks']} | {r['impl']} | {r['device_ms']} | "
                 f"{r['per_block_ms']} | {r['hbm_mb']} | {r['mbu_pct']}% | {r['cos_sim']} |")
    L.append("")
    rt = fm["ratios"]
    L.append(f"NKI/torch = **{rt['H1024_I3072']}x** at Qwen3-0.6B's MLP shape and "
             f"**{rt['H4096_I4096']}x** at the largest shape it runs single-core. The gap barely "
             f"narrows with scale, so it is not a shape artifact. Interpretation: nkilib kernels "
             f"need a multi-core SPMD grid; single-core they tile far more finely than designed.")
    L.append("")
    wt = fm["weight_transpose_cost"]
    L.append(f"Weight-layout cost (Finding #17) quantified for the first time: the on-device "
             f"transpose is {wt['H1024_I3072']['device_ms']} ms / "
             f"{wt['H1024_I3072']['hbm_mb']} MB at H=1024/I=3072. One-time at load, not per step.")
    L.append("")
    fb = ms["fused-mlp-compile-boundary"]
    npass = sum(1 for p in fb["data_points"] if p["result"] == "pass")
    L.append(f"Compile boundary, {len(fb['data_points'])} data points "
             f"({npass} pass): **{fb['conclusion']}**. {fb['notes']}")
    L.append("")

    # --- correctness ----------------------------------------------------------------------
    ac = ms["accuracy-suites"]
    L.append("## Correctness")
    L.append("")
    L.append("| suite | result | seconds | cases |")
    L.append("|---|---|---|---|")
    for s in ac["suites"]:
        L.append(f"| `{s['suite']}` | {s['result']} | {s['seconds']} | {s.get('cases', '—')} |")
    L.append("")
    e = ac["e2e_logits_cos_sim"]
    L.append(f"End-to-end logits: Qwen3 dense `cos_sim {e['qwen3_dense']}`, "
             f"Qwen3-MoE `cos_sim {e['qwen3_moe']}`. Every case asserts via a call counter that "
             f"the NKI branch actually ran — a silent fallback is numerically correct and would "
             f"otherwise pass.")
    L.append("")
    cov = ms["interception-coverage"]
    L.append(f"Upstream coverage: {cov['rmsnorm_registrations']} RMSNorm registrations, "
             f"{cov['rope_model_files']} RoPE model files, {cov['silu_decorations']} SiLU "
             f"decoration ({cov['silu_note']}).")
    L.append("")

    # --- open items -----------------------------------------------------------------------
    L.append("## Open items")
    L.append("")
    for o in d["open_items"]:
        L.append(f"- **[{o['priority']}]** {o['item']} — {o['detail']}")
    L.append("")

    # --- environment ----------------------------------------------------------------------
    L.append("## Environment")
    L.append("")
    L.append(f"{about['hardware']}.")
    L.append("")
    L.append(f"**{about['compiler_flags']}**")
    L.append("")
    L.append("| package | version |")
    L.append("|---|---|")
    for k, v in about["versions"].items():
        L.append(f"| `{k}` | {v} |")
    L.append("")
    L.append("## Regenerating")
    L.append("")
    L.append("On a fresh trn2 with the repo synced:")
    L.append("")
    L.append("```bash")
    L.append("make results      # re-runs every measurement, writes raw artifacts to results/raw/")
    L.append("```")
    L.append("")
    L.append("Individual measurements list their own command in `measurements.json`.")
    L.append("")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(L)} lines) "
          f"from {len(d['measurements'])} measurements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
