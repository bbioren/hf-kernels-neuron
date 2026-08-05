# Customer Experience Report

What would a customer struggle with if they tried to use NKI kernels via the HF Kernel Hub today? This doc tracks friction from the perspective of someone who just wants `use_kernels=True` to work on Trainium.

---

## Setup Friction

| Issue | Severity | Notes |
|-------|----------|-------|
| Torch lives in `/opt/` venv, not system-wide | Medium | Must `source activate` the right venv. No `pip install torch` on DLAMI. |
| Ubuntu 24.04 blocks system pip (PEP 668) | Low | Venv required, but expected for modern Python |
| `torch_neuronx` import triggers runtime init | Medium | Fails if helper binaries not on PATH. Must use full DLAMI venv, not .pth hack. |
| ~15 min from fresh instance to working | Low | Acceptable for devs, but worse than GPU (just `pip install` and go) |
| No HF_TOKEN configured by default on DLAMI | Low | Rate-limited Hub access, warning messages |

## API / Integration Friction

| Issue | Severity | Notes |
|-------|----------|-------|
| `use_kernels=True` cannot select the neuron path at all | **Critical** | transformers' `kernelize(model, mode)` has no `device` arg; it reads `model.device.type`, which is `"cpu"` or `"xla"` on Neuron — never `"neuron"`. See Finding #9. |
| `"xla"` is not a supported device type in `kernels` | **Critical** | Kernelizing a model that has been moved to a Neuron device raises `Unsupported device type 'xla'`. So the correct way to run *breaks*, and the incorrect way (params on host) silently no-ops. |
| Customer must call the `kernels` library directly, bypassing transformers | High | `kernelize(model, device="neuron", mode=Mode.INFERENCE)` works, but it is not the documented transformers entry point and skips `KernelConfig` handling. |
| No way to ask "is my kernel actually active?" | High | Nothing reports which implementation is live. Combined with silent fallback, a customer cannot tell acceleration from no-op. |
| Function kernels swap process-globally | Medium | Kernelizing one model changes `apply_rotary_pos_emb` for every model in the process. Surprising for multi-model serving. |
| `@nki.jit` hard-errors on CPU tensors | Medium | Forces every kernel to carry a device guard, which is what creates the silent-fallback trap. |

## Silent Failure Modes (highest-risk category)

These are the issues where the customer gets **no error and no warning**, and would
reasonably believe things are working.

| Failure | What the customer sees | What's actually happening |
|---------|------------------------|---------------------------|
| Kernel falls back on host tensors | Correct numbers, no warning. Our own test even printed "Backend: NKI kernel" | Eager PyTorch. Zero NKI execution. Cost us a week of false confidence — see Finding #8. |
| `"neuron"` mapping ignored on a `cpu`-device model | `use_kernels=True` returns successfully | Mapping lookup misses; original forward retained |
| Accuracy test passes with `max_diff = 0.00e+00` | "Bit-identical, great" | Both sides ran the same PyTorch code. For a hardware kernel, a perfect match means the kernel didn't run. |
| Benchmark shows plausible latency, output discarded | "My kernel is 8x slower" (or faster) | XLA is lazy; with no live output the computation is eliminated. You timed an empty graph — Finding #19. |
| Kernel appears slow in isolation | "The NKI kernel is bad" | ~0.36 ms/call of host-side dispatch overhead dominates at per-layer granularity. It's the integration model, not the kernel. |

**The pattern worth generalizing:** on a lazy-execution accelerator backend, *both* correctness
and performance measurements fail silently by default. A fallback is numerically correct, and
an eliminated computation is fast. Every measurement needs an independent check that it
exercised the thing being measured — a call counter for correctness, a size-scaling check for
performance. Neither is standard practice. Both cost us a cycle.

**The general lesson for the PoC:** on Neuron, the dangerous outcome is not a crash,
it's a no-op that looks like success. Any customer-facing story for NKI kernels on the
Hub needs an affirmative "this kernel is live on this layer" signal. Numerical
correctness alone cannot distinguish acceleration from fallback, because the fallback is
*also* numerically correct.

## Documentation Gaps

| Gap | Impact | Notes |
|-----|--------|-------|
| No "Hello World kernel" tutorial for authors | High | Had to piece together from PR, spec, and trial-and-error |
| `LocalLayerRepository` docs show removed `package_name` arg | Medium | TypeError on first try |
| `metadata.json` fields underdocumented for local dev | Medium | Required even for local testing, not obvious |
| No docs on Neuron-specific kernel authoring | High | Nothing tells you how to do this for Neuron specifically |
| `kernelize()` docstring omits `neuron` from supported devices | Medium | Lists "cuda", "mps", "npu", "rocm", "xpu". Both `neuron` and `cpu` are supported in code. A reader would conclude Neuron isn't. |
| No documented difference between `nki` and `neuronxcc.nki` | **High** | They have different capabilities and neither is a superset (Finding #14). Nothing says which is supported, or that `hasattr` lies about `nl.arange`. |
| `nki-library`'s `rope_hf` absent from the public API reference | Medium | The best HF-shaped kernel in the library is source-only. The reference also cites a non-existent import path (`nkilib.core.rope` vs real `nkilib.core.embeddings.rope`). |
| No guidance that layer vs function kernels resolve differently | Medium | Layer repos look in `kernel.layers.<name>`; func repos look at module top level. Getting it wrong yields a confusing "not found". |

## Runtime Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| `@nki.jit` requires XLA tensors, errors on CPU | Medium | `RuntimeError: Expected all tensors ... to be XLA tensors`. Forces a device guard in every kernel, which is what creates the silent-fallback trap. |
| A Neuron kernel cannot declare `python-depends: ["nki"]` | High | HF whitelists `nki` for the neuron backend, but `_backend()` reports cuda on the DLAMI so the entry is unreachable. Kernels must under-declare to load (Finding #12). |
| Kernel constraints silently disable acceleration | High | RoPE needs `seq_len % 128 == 0`; HF passes arbitrary lengths. Without an explicit warning the customer just gets eager speed. We added `warn_once`; upstream kernels generally don't. |
| Per-kernel NKI import path pinning | High | Some kernels only compile under `neuronxcc.nki`, others only under top-level `nki`. A multi-kernel repo needs both, discovered at compile time. |

## What a Customer Would Need to Do Today

1. Install `kernels` from PyPI (pinned to a minor: `>=0.15,<0.16`)
2. Install `transformers` from main (neuron path not in a tagged release yet)
3. Activate the DLAMI Neuron venv (`source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate`)
4. Clone the kernel repo locally (no Hub publishing yet)
5. **Bypass transformers entirely.** `use_kernels=True` cannot reach the neuron path
   (Finding #9), so call the `kernels` library directly with an explicit device:
   ```python
   from kernels import kernelize, Mode, use_kernel_mapping
   with use_kernel_mapping(mapping, inherit_mapping=False):
       kernelize(model, device="neuron", mode=Mode.INFERENCE)
   ```
6. **Manually attach function kernels.** For RoPE, replicate the `_hidden_kernels`
   attach/detach that transformers' wrapper does, or the function swap won't be found.
7. Move the model to the Neuron device *before* kernelizing, and know that leaving it on
   the host means the kernels silently don't run.
8. Verify the kernels actually ran — nothing reports it, and correct output does not
   imply acceleration.

Steps 5–8 are all consequences of gaps we found this week. None of them are documented
anywhere, and step 8 has no supported mechanism at all.

## What "Just Works" Would Look Like

```python
from transformers import AutoModelForCausalLM

# This is the dream. No config, no local files, no device override.
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    use_kernels=True,
    device_map="neuron",
)
# NKI kernels downloaded from Hub, swapped in, model runs accelerated.
```

## Gaps Between Current State and "Just Works"

Ordered by how much they block the "just works" experience.

| # | Gap | Owner | Effort | Notes |
|---|-----|-------|--------|-------|
| 1 | `use_kernels=True` can't select the neuron device | **transformers** | ~5 lines, 3 sites | Map `xla`→`neuron` when `xla_device_hw()=="NEURON"`. **Verified sufficient**: takes Qwen3 from 0→9 swapped layers. Sites: `hub_kernels.kernelize`, `kernel_config.infer_device`, `kernels._find_device`. |
| 2 | `_backend()` reports cuda on Neuron hosts | **torch_neuronx** | 1 attribute | Set `torch.neuron`. Unblocks build-variant resolution *and* `python-depends: ["nki"]` at once (Findings #7, #12). Does NOT fix #1. |
| 3 | No way to verify a kernel is live | HF `kernels` | small | Silent fallback + no reporting means acceleration is indistinguishable from a no-op. Biggest trust problem. |
| 4 | Neuron kernels not published on Hub | Neuron/HF | — | Flat layout works (no kernel-builder needed). Blocked on repo-home decision + gap 2 for honest deps. |
| 5 | `nkilib` not on the `python-depends` allowlist | HF `kernels` | 4 lines | Precedent and exact JSON shape already there for `nki`. Prerequisite for thin-wrapper porting. |
| 6 | `nki` vs `neuronxcc.nki` capability split | NKI team | needs a decision | Neither is a superset; kernels are pinned per-idiom (Finding #14). |
| 7 | `device_map="neuron"` doesn't exist | transformers | larger | Would make the "dream" snippet work as written. |
| 8 | transformers neuron path not in a tagged release | HF releases | — | Requires install from main. |

**The good news, and it is real:** gaps 1 and 2 are both small, well-understood, and
between them unblock most of the experience. The interception points already exist
upstream, Qwen3 already opts into them, and coverage is large (115 RMSNorm, 95 RoPE).
Nothing here requires architectural change — which is the central input to the
"is this worth investing in" question.

---

## The worst customer-experience issue in this project is invisible and costs 102x

Added after Finding #24, and it outranks everything above.

**A customer calling NKI kernels per-layer from eager PyTorch today pays ~52 ms per kernel call to
fork `neuron-ls`.** No error, no warning, no log line. It presents as "NKI kernels are slow", which
is the single most misleading possible symptom because it points at the kernel — the one thing that
is not the problem.

What a customer would actually experience:

1. Write or adopt a NKI kernel. It is numerically correct.
2. Benchmark it. It is dramatically slower than the PyTorch op it replaced.
3. Conclude the kernel is bad, or that NKI is not worth it, and stop.

Nothing in that loop points at process spawning. The profile that reveals it is not one a customer
would think to run: you need to compare device time against wall time, notice a ~2400x gap, and then
cProfile a single call. We only did it after four framework-level experiments failed to close the
story, and we had a strong prior that something was wrong. A customer with a deadline stops at
step 3.

**Severity: highest in this document.** It is a one-decorator fix (`functools.lru_cache` on
`_detect_target`) worth 102x per call, and the current state actively teaches customers a false
lesson about NKI.

### What would have surfaced it

| Mitigation | Cost | Effect |
|---|---|---|
| Cache `_detect_target()` | one decorator | removes the problem |
| Warn once if target detection runs more than N times per process | a few lines | makes it self-diagnosing |
| Document `NEURON_PLATFORM_TARGET_OVERRIDE` as a perf-relevant setting | doc change | gives customers a lever, but only if they know to look |
| A "why is my kernel slow" playbook that starts with device-time vs wall-time | doc change | generalises past this bug |

The last one is the most broadly useful. The technique that found this — compare device time to wall
time before forming any hypothesis — would find *any* host-side overhead problem, and it is two
numbers. That belongs in the NKI performance documentation as step one.

### A second, related paper cut

`_detect_target()` falls back to `"trn3"` when `neuron-ls` is not on `PATH`:

```python
if shutil.which("neuron-ls") is None:
    return "trn3"
```

On a trn1 or trn2 host without `neuron-ls` installed, this silently compiles for the wrong
generation. Same failure shape as everything else in this document: a wrong result rather than an
error. Worth fixing in the same change.

### And a third: `torch.compile` fails on most transformers, confusingly

Covered in Finding #23. `torch_neuronx` overrides `gelu`, `silu`, `randn`, `CrossEntropyLoss`,
`Dropout`, `Embedding`, `clip_grad_norm_`, `argmax`, `Softmax`, `topk` and `upsample_nearest2d` with
XLA user computations that are not fake-tensor safe. A customer running `torch.compile` on any model
containing an embedding or a softmax gets:

```
Dynamo failed to run FX node with fake tensors: call_function <function silu ...>
got RuntimeError('Expected all tensors ... Got: XLAFloatType')
```

That message names `silu` and mentions XLA tensor types, which invites the conclusion that
`torch.compile` is unsupported on Neuron generally. We drew exactly that conclusion and recorded it
as a finding before checking. `add`/`mul`/`relu` compile fine. The workaround
(`torch_xla.compile()`) exists and is undocumented in this context.

---

## Native PyTorch changes the customer story substantially — and adds one severe new trap

Everything above was written against the torch-xla stack. On the Native PyTorch drop
(`torch 2.11.0`, `torch-neuronx 0.1.0`, NKI 0.6.0b1) the two worst *integration* problems in this
document simply do not exist, and one new *operational* problem is worse than any of them.

### What gets better

| | torch-xla | Native PyTorch |
|---|---|---|
| `model.device.type` | `"xla"` | **`"neuron"`** |
| `kernels._backend()` | `CUDA(version=12.8)` | **`Neuron()`** |
| `hasattr(torch, "neuron")` | False | **True** |
| declaring `"python-depends": ["nki"]` | rejected — validates against the cuda table | **accepted** |
| stock `use_kernels=True` reaching a Neuron kernel | never | **works, unpatched** |

So the two monkeypatches this project carried for weeks are not needed. A customer on the supported
stack does not hit either. That is a materially better story than the one in the sections above, and
it is worth stating plainly rather than burying: **the device-routing and dependency-declaration
friction was an artifact of using the wrong stack, and it was our mistake, not HuggingFace's.**

What a customer *still* cannot do is `use_kernels=True` and get NKI kernels, because transformers ships
no `"neuron"` entries in `_KERNEL_MAPPING`. That is the one real remaining blocker, and it is a small,
well-understood addition rather than an architectural gap.

### The new trap, and it is severe

**If `neuronx-cc` is not on `PATH`, the first real operation hangs forever with no output.**

Not an error. Not a timeout. The process blocks indefinitely, at the first op that needs a compile,
having printed nothing useful. Underneath, the runtime forks a child and `execve`s `neuronx-cc` by bare
name; all seven `PATH` entries return `ENOENT`; the child then blocks before it can report anything and
the parent waits on the child.

The reason a customer will hit this is subtle and unlucky:

```bash
/home/ubuntu/native_venv/bin/python train.py     # hangs forever
source /home/ubuntu/native_venv/bin/activate && python train.py   # works
```

For essentially all Python tooling those two are equivalent, and calling `venv/bin/python` directly is
a widely taught habit — it is what most Makefiles, systemd units, cron jobs, CI configs and IDE
interpreter settings do. It stops being equivalent here precisely because a **subprocess** resolves a
binary through `PATH`, and `PATH` is what activation sets.

Severity is high for three compounding reasons:

1. **It presents as a hardware or version problem.** First op, no output, driver right there in
   `neuron-ls`. Nothing points at `PATH`.
2. **The obvious next step is destructive.** Our drop ships driver/runtime debs at build numbers newer
   than the host's, so "version mismatch, install the matching debs" is the natural hypothesis. That
   replaces the host Neuron driver and runtime. We came one step from doing it. A customer with less
   reason to be cautious will do it, and will then be debugging a driver install that was never the
   problem.
3. **Diagnosing it needs `strace`.** Process state (`/proc/*/wchan`, thread counts, py-spy) is
   *consistent with a genuine fork deadlock* and will lead a careful person to the wrong conclusion.

Cost to us with prior Neuron experience and both tools to hand: ~50 minutes. Cost to a customer cold:
plausibly hours, with a real chance of a broken driver at the end.

**The fix is small and entirely on the Neuron side.** The runtime already knows it wanted `neuronx-cc`
and already knows every `execve` returned `ENOENT`. Raising there — naming the binary and the `PATH`
searched — turns a multi-hour mystery into a one-line message. This is the single highest
value-per-effort customer-experience fix identified in this project.

Until then: document that the venv must be **activated**, not merely invoked by path, and say why.
Our `scripts/run_native.sh` does it and asserts the compiler resolves before running anything.

### Setup friction for the native drop

For the record, and unchanged in character from the DLAMI notes above:

- The drop is an S3 prefix, not a package index. Needs allowlisting per AWS account.
- The bucket lives in one region while the CLI may default to another; `s3 ls` follows the redirect
  but `presign` bakes the endpoint in and fails with `IllegalLocationConstraintException`. Region has
  to be passed explicitly.
- An EC2 instance role will not usually have S3 access to it, so the fetch needs credentials from
  elsewhere. Presigned URLs keep those credentials off the instance, which is the right pattern.
- Python 3.12 venv built from scratch, then `pip install` the wheels. ~926 MB of artifacts.
- The `deb/` packages turned out **not** to be required — compute works against the production host
  runtime — but nothing in the drop says so, and the instruction accompanying it says to install them.
  Worth confirming with the drop's owners which parts are actually mandatory, because installing them
  is the step most likely to break an existing working environment.
