# Sticking Points Log

Running log of things that were harder than expected, took extra time, or blocked progress. Each entry notes the date, what happened, how long it took to resolve, and whether it would affect customers or engineering at scale.

---

## Format

```
### [DATE] Short title
**Time lost:** X min/hours
**Would affect:** customers / engineering / both
**Resolution:** what fixed it
**Takeaway:** what should be different
```

---

### [2026-07-22] `kernels` not installable from GitHub source
**Time lost:** 15 min
**Would affect:** engineering (anyone trying to test unreleased features)
**Resolution:** Install from PyPI instead. Library is Rust/Python hybrid, needs maturin to build from source.
**Takeaway:** Document that devs must use PyPI releases. If Neuron patches land before a release, building from source requires Rust toolchain.

### [2026-07-22] `LocalLayerRepository` API changed — docs show removed param
**Time lost:** 20 min
**Would affect:** both (any kernel author following docs)
**Resolution:** Checked actual constructor signature via `help()` on trn2. v0.15.2 only takes `(repo_path, *, layer_name)`.
**Takeaway:** Pre-1.0 library, APIs shift. Always verify against installed version, not docs. Pin minor version.

### [2026-07-22] `metadata.json` required for local dev (underdocumented)
**Time lost:** 30 min (two round-trips to trn2 to figure out required fields)
**Would affect:** both
**Resolution:** Added `python-depends: []` and `digest: {"algorithm": "sha256", "files": {}}` — both required but not in the "local dev" docs.
**Takeaway:** HF should either simplify LocalLayerRepository to not need metadata, or document the minimum fields clearly.

### [2026-07-22] Neuron DLAMI venv structure — torch not system-wide
**Time lost:** 45 min (tried .pth hack, then standalone venv, finally just activated DLAMI venv)
**Would affect:** customers
**Resolution:** `source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate` — use the DLAMI venv directly.
**Takeaway:** The DLAMI doesn't match the "just pip install" developer experience. Must document the activation step prominently.

### [2026-07-22] `torch_neuronx` import crashes without bin/ on PATH
**Time lost:** 15 min
**Would affect:** customers
**Resolution:** The `.pth` hack (adding site-packages to a local venv) doesn't expose the neuron venv's `bin/` directory. Must fully activate the venv.
**Takeaway:** `torch_neuronx` has a hard dependency on `libneuronpjrt-path` binary being on PATH at import time. This is unusual for a Python package.

### [2026-07-22] Our `kernels/` directory conflicts with the `kernels` pip package
**Time lost:** 20 min
**Would affect:** engineering
**Resolution:** Used `importlib.util.spec_from_file_location()` to load our local kernel by explicit path instead of normal import.
**Takeaway:** Don't name your project directory the same as a pip package you depend on. Or use a different project structure.

### [2026-07-22] Variant resolver detects CUDA, not Neuron
**Time lost:** 1 hour (investigating, writing test scripts, reading source)
**Would affect:** both (blocks Hub publishing with variant structure)
**Resolution:** `hasattr(torch, "neuron")` returns False on current DLAMI. Flat structure (no build/ dir) works via fallback path.
**Takeaway:** `torch_neuronx` should set `torch.neuron` attribute. File bug or PR.

### [2026-07-22] nki-library has no standalone RMSNorm
**Time lost:** 30 min reading source + writing analysis doc
**Would affect:** engineering (blocks mass porting)
**Resolution:** Used tutorial-derived kernel for PoC. Documented that production kernels need unfused entry points.
**Takeaway:** nki-library is designed for the NxDI inference pipeline (always fused with quant). Needs a simple `nkilib.ops.*` API for the HF use case.

### [2026-07-29] Week 2 accuracy results were measuring the PyTorch fallback, not NKI
**Time lost:** ~1 hour (spotting it, writing the instrumented probe, re-validating)
**Would affect:** both — and this is the worst kind of problem because it fails silently
**Resolution:** `@nki.jit` requires XLA tensors and hard-errors on CPU ones. Our kernel
guards with `device.type != "cpu"`, so CPU-tensor tests took the fallback branch every
time. Fixed by adding `tests/nki_test_utils.py`, which places tensors on the XLA device
and asserts via a call counter that the NKI branch actually ran.
**Takeaway:** The tell was `max_diff = 0.00e+00`. For a hardware kernel, a *perfect*
match is evidence of failure, not success — real NKI reductions differ from PyTorch by
~1e-4. Never accept exact-zero diff as a pass. Always assert the kernel executed, not
just that the numbers look right.

### [2026-07-29] `use_kernels=True` can't reach the neuron device path — two independent gaps
**Time lost:** ~45 min investigating (across kernels + transformers source, then a probe)
**Would affect:** customers (this is the headline user-facing gap)
**Resolution:** No fix available locally. transformers' `kernelize(model, mode)` has no
`device` parameter and derives everything from `model.device.type`; Neuron reports
`"cpu"` (mapping ignored) or `"xla"` (rejected as unsupported). Worked around in tests by
calling the `kernels` library directly with `device="neuron"`.
**Takeaway:** Documented the minimal upstream patch — map `"xla"` → `"neuron"` in
`kernels._find_device` using `xm.xla_device_hw()`, which we confirmed returns `"NEURON"`.
Filed as Finding #9. This is the single highest-value upstream change for the project.

### [2026-07-29] `nki` and `neuronxcc.nki` are not interchangeable — neither is a superset
**Time lost:** ~50 min (SiLU failed on all 9 shapes; then a wrong "standardise on one
package" attempt broke all 20 RoPE cases before I reverted)
**Would affect:** both — anyone porting more than one nki-library kernel hits this
**Resolution:** Pinned each kernel to the package its idiom needs. RMSNorm and SiLU use
`nl.arange` index tensors, which only resolve under `neuronxcc.nki`. RoPE uses slicing
plus `//` on shape values, which only works under top-level `nki` (`neuronxcc.nki`
treats shapes as symbolic scalars and raises
`NotImplementedError: math.trunc() is not supported for scalar`). Our repo genuinely
needs both packages.
**Takeaway:** `hasattr(nl, "arange")` is True under the top-level package even though
the name cannot be resolved at trace time, so there is no import-time feature detection
— you find out at compile time, per kernel. And the error text never hints that the
sibling package would work. nki-library source uses top-level `nki` while the tutorials
use `neuronxcc.nki`, so a mass-porting effort meets this immediately. Needs a supported
compatibility table from the NKI team. Finding #14.

### [2026-07-29] My own test instrumentation gave a false negative
**Time lost:** ~20 min
**Would affect:** engineering (anyone writing kernel tests)
**Resolution:** In the e2e test I patched a freshly `load_kernel_module()`-ed copy of the
kernel, but `LocalLayerRepository` had loaded its *own* module object, so the counters
read nki=0 while the kernel was demonstrably running (logits had changed). Fixed by
instrumenting via `get_local_kernel()`, which caches and returns the same object the
repository used.
**Takeaway:** Ironic and instructive: this is a false negative of exactly the shape
Finding #8 is a false positive of. Whenever you assert on "did the kernel run", confirm
you are observing the same module object the framework loaded — Python module identity
is easy to get wrong when a package is loaded by path.

---

## Summary Statistics

| Category | Count | Total Time Lost |
|----------|-------|----------------|
| Documentation gaps | 3 | ~65 min |
| Environment/setup | 3 | ~75 min |
| API instability | 2 | ~70 min |
| Architecture mismatch | 3 | ~2.25 hours |
| Silent-failure / test methodology | 2 | ~80 min |
| **Total** | **13** | **~7 hours** |

### Where the time actually goes

Two categories dominate, and they are not the ones you would guess from Week 1:

- **Silent failures and test methodology (~80 min).** Nothing crashed. The kernels
  produced correct numbers while not running at all. This cost a week of false
  confidence in Week 2 and was only caught by noticing an implausibly *perfect* result.
- **Undocumented capability splits (~70 min).** Two NKI packages that both import, both
  pass `hasattr`, and fail differently at compile time.

Neither shows up as an error message a customer could search for. That is the through-line
of this PoC: the Neuron + HF Kernel Hub integration mostly fails *quietly*.

---

## 14. Chasing a performance regression to the wrong layer [~6 hours, the largest single item]

**What happened.** Kernelizing Qwen3 made it 208x slower. Root-causing that consumed most of two
sessions and the conclusion was wrong for most of it.

**Time breakdown, because the shape of it is the lesson:**

| activity | time | outcome |
|---|---|---|
| four framework-level experiments (interleaving, data volume, recompilation, our-vs-production kernels) | ~3 h | all consistent with a wrong hypothesis |
| writing up the graph-transition explanation, twice | ~1 h | had to be corrected twice |
| chasing `torch.compile` as the decisive test | ~1 h | wrong instrument entirely |
| device profile + Python profile | **~35 min** | **found it** |
| verifying the fix and re-measuring | ~30 min | 102x per call, 62x at model level |

The two measurements that actually resolved it took 35 minutes. Everything before them was
elaboration within a framing that could not be falsified by the instrument in use.

**Why it was slow.** Every one of the four experiments measured wall-clock time at the framework
level. A fixed per-call cost independent of problem size is genuinely the signature of
graph-transition overhead, so each experiment came back consistent and increased confidence in a
wrong answer. The hypothesis was never tested against a device profile, which would have killed it
immediately: 0.609 ms of device time against 1459 ms of wall time.

**What would have saved the time.** Measuring device time against wall time *first*. It is one
number from `neuron-explorer` and one from `time.perf_counter()`, their ratio was 2400x, and it
invalidates every device-side explanation at once. Total cost maybe 15 minutes, and it should be
the first thing done on any accelerator performance question, before any hypothesis is formed.

**Who else this affects.** Anyone debugging NKI performance from eager PyTorch. The `neuron-ls`
subprocess costs ~52 ms per kernel invocation on any workload, and it presents as "NKI kernels are
slow" rather than as anything pointing at process spawning. A customer would have no reason to
suspect it and no easy way to find it — it took a cProfile of a single call to see.

---

## 15. `pgrep -af <pattern>` over SSH matches its own command line [~10 min, twice]

`ssh trn2 'pgrep -af neuronx-cc || echo free'` always reports a match, because the `bash -c`
wrapper carrying the pattern is itself a running process containing that pattern. First time it
looked like a stale compiler process was holding the Neuron cores; second time I recognised it.

Use `pgrep -af neuronx-cc | grep -v pgrep`, or check for the actual artifact (`model.neff`)
instead of the process. Minor, but it produces a false "cores busy" reading, which on this box
looks identical to the real and fairly common stale-lock situation.

---

## 16. torch-xla metric accumulators are nanoseconds, not seconds [~15 min, nearly a published error]

`torch_xla.debug.metrics.metric_data(name)` returns `(count, accumulator, samples)`. The
accumulator is in **nanoseconds**, while `metrics_report()` prints it formatted as `us`/`ms`. I
read it as seconds and printed `ExecuteTime 919108000.00 ms` — a nine-digit millisecond figure in
a table next to a 1459 ms wall time, which is what made it obviously wrong.

Had the scale been closer to plausible it would have gone into a finding. Worth stating as a
general rule: when a derived number is impossible, the units are the first thing to check, and a
sanity range on any computed timing catches this class of error for free. Cross-check against
`metrics_report()`, which formats the same values with explicit units.

---

## 17. Every raw measurement artifact was lost when the instance expired [unrecoverable, ~2 h to mitigate]

**What happened.** The trn2 instance used for all measurements expired. Every raw artifact lived in
`/tmp` on that host and went with it:

- `measure_mfu.py --json-out` result files
- `neuron-explorer view --output-format summary-json` output
- NEFF / NTFF device-profile pairs (the evidence behind Findings #25 and #26)
- detached run logs from `scripts/run_detached.sh`

**Why it did not lose the project.** Each run's stdout had been pasted into the git commit message at
the time, and every producing script is committed. So the numbers survive and are reproducible. What
does *not* survive is auditability: a reviewer cannot open a file and check a figure, they have to
take a commit message's word for it or re-run an hour of measurements.

**Cost.** Roughly two hours to build the mitigation — `results/measurements.json` with per-number
provenance, `scripts/render_results.py` to generate the human-readable summary from it, and
`scripts/regenerate_results.py` to rebuild the raw tree in-repo next time. None of that work would
have been needed had the artifacts been written to the repo in the first place, which costs nothing.

**Why it was missed, and this is the annoying part.** The project explicitly tracked the risk that
"47 commits exist only on one laptop." It did not notice the sharper version of the same risk one
layer down: **the evidence lived on a machine with a shorter lifetime than the laptop.** Ephemerality
was being reasoned about at the wrong level.

**What would have prevented it.** Writing artifacts under `results/raw/` from the first measurement
instead of `/tmp`. Every script already took an `--outdir` or `--json-out`; the default was simply
pointed at the wrong place. One line per script.

**Generalisable rule:** *an output path is a durability decision.* `/tmp` on a rented host is the
least durable location available, and it is also the default that every tutorial and profiling guide
uses, so it takes deliberate thought to avoid. On borrowed hardware, treat the repo as the only real
filesystem.

**Who else this affects.** Anyone profiling on ephemeral Neuron instances. The profiling workflow's
own documented example writes to `./output` in the working directory, which on a cloud instance is
usually as ephemeral as `/tmp`. Worth a line in the profiling guide.

---

## 18. Reporting the most dramatic true number rather than the most representative one [cost: reviewer trust]

Not a debugging sticking point, but it cost more than most of the ones above and is the failure a
reviewer noticed before I did.

**What happened.** Three separate figures for the same phenomenon were all true and all
differently misleading:

| figure | true of | misleading because |
|---|---|---|
| 208x slower | the pre-fix state | a one-line bug caused it; it says nothing about the approach |
| 2.5–2.7x slower on device | a chained microbenchmark | that benchmark is deliberately NKI's worst case |
| 8.4% device / 91.6% dispatch | a real forward pass | this is the representative one, and it arrived last |

Each time I led with the newest and most dramatic, and each time the framing was corrected by a
later measurement. The reviewers' pushback — "there shouldn't be a slowdown" — was correct, and it
was correct against a number I had put in front of them.

**What would have prevented it.** Asking, before quoting any ratio, *what configuration is this
true of, and is that the configuration anyone cares about?* The chained microbenchmark answer was
available immediately: 28 identical ops back to back is not a model.

**Mitigation now in place.** `results/README.md` leads with the in-situ split and explicitly names
the two figures most likely to be quoted out of context, with why. `results/measurements.json` tags
the microbenchmark rows with a note saying not to cite them without the in-situ figure. That is a
partial fix — a doc cannot stop a number travelling — but it at least means the correction ships
alongside the claim.

---

## 19. The sync script deleted the artifacts it was supposed to be unrelated to [cost: ~40 min, 19 stages of artifacts]

**What happened.** After the first full regeneration run finished on the replacement instance, I
pushed a small code change with `./scripts/sync_to_trn2.sh` and then went to fetch the results. 19 of
22 stages' artifacts were gone from the remote.

`sync_to_trn2.sh` uses `rsync --delete`, which is right for pushing source: it means a file deleted
locally goes away remotely instead of lingering. But `results/raw/` existed locally containing only
`README.md`, so `--delete` did exactly what it was told and removed everything under the remote
`results/raw/` that had no local counterpart. Which was all of it.

**Why it took a while to see.** The sync printed a normal success line. The loss was only visible on
the next `ls`, and my first theory was that a stage had failed rather than that a later, unrelated
command had deleted the output of stages that had already succeeded.

**What it is really an instance of.** Sticking point #17 was "results lived somewhere something else
destroys them", and the destroyer was an expiring instance. This is the same failure with a different
destroyer, discovered *while fixing the first one*. Moving artifacts out of `/tmp` and into the repo
tree put them directly in the path of a tool whose whole job is making the remote tree match the
local one.

**Fix.** `results/raw/` is excluded from the push, with a comment saying the exclusion is
load-bearing and why. That comment matters more than the exclusion: the line looks removable —
"why would we not sync results?" — and removing it re-arms the bug.

**Generalisable rule:** *a directory that is written remotely and read locally must be excluded from
any bidirectional sync, and the exclusion needs a comment explaining what breaks if you remove it.*
The push and the fetch are different directions with different correct flags, so they are two
scripts, not one script with a flag.

---

## 20. A summing consumer plus a non-clearing producer, and no error [cost: ~30 min, one wrong headline number]

**What happened.** `sum_model_device_time.py` sums device time across every NEFF/NTFF pair it finds
in a directory, because a full model forward can compile to several NEFFs. `profile_model_device_time.py`
wrote into its output directory without clearing it. Re-running the profile into a directory that
already had a NEFF therefore produced exactly double the device time — and the number it produced,
16.9% device instead of 8.4%, is plausible. Nothing errored, nothing warned. It would have gone into
the design doc if the "2 NEFF(s)" in the output line had not looked odd.

Doubling is the worst possible corruption here, because it is *almost* right. A 10x error announces
itself; a 2x error looks like a different but reasonable measurement.

**Fix, in two places, because either alone is insufficient.** The producer now clears its own output
directory. The consumer gained `--expect-neffs` (default 1) and prints a loud warning naming each
file when the count is wrong. Clearing alone would still leave the consumer silently summing whatever
it finds if some other producer writes there; the assertion alone would still let a stale file in.

**Two more instances of the same shape, found by looking for it deliberately.** Once the pattern was
named, `profile_nki_call_cost.py` and `profile_fused_mlp_vs_torch.py` turned out not to clear their
directories either. The fused-MLP case is the interesting one: it legitimately emits *two* NEFFs, a
1-block correctness graph and an N-block timed graph, so "more than one NEFF" is not by itself the
signal. They are told apart by instruction count.

**Generalisable rule:** *if a consumer aggregates over "everything in this directory", the producer
must own that directory exclusively and clear it, and the consumer must assert the count it expects.*
Aggregating with `sum` over a glob is a silent-corruption interface — `max`, or reading a manifest the
producer wrote, would have failed loudly instead.

---

## 21. A string placeholder that only matched its prefix form [cost: ~25 min, 8 profile directories in the wrong place]

**What happened.** `regenerate_results.py` substitutes `STAGE/` and `RAW/` in each stage's argv so
output lands in the artifact tree. The substitution was `a.replace("RAW/", ...)`, which handles
`RAW/prof_model_baseline` but not a bare `RAW` — and `--outdir-base` takes a bare directory. So the
token passed through literally, the sweep created `./RAW/` in the repo root, wrote 8 profile
directories there, and the *consuming* stage read from the same literal path and looked fine. Only
the final extraction stage, which globs `results/raw/`, noticed — and its complaint was eight lines of
`skip ... (missing)`, which reads like a stage that had not run yet.

**Why the design invited it.** A placeholder that is sometimes a prefix and sometimes a whole value
has two forms, and a `str.replace` only sees one. The fix handles the exact token as a separate case.

**The deeper problem was the same class as #19 and #20:** an output path that no one verified. Three
separate path bugs in one harness, all of the same shape — *a file was written somewhere nobody
checked* — and all invisible because every stage exited 0. A harness that reports "all stages ok"
while writing to the wrong directory is worse than one that fails, because it produces confidence.

**Fix, beyond the substitution.** `scripts/check_measurement_provenance.py` now verifies that every
path named in a `measurements.json` `artifact` field actually exists. That converts "the artifact is
somewhere" from a belief into a check, and it caught its own first run: fifteen artifacts recorded as
committed while they were still only on the instance.

**Generalisable rule:** *a harness must verify its outputs exist where it claims, not just that its
subprocesses exited 0.* Exit code 0 means the command ran, not that it did the thing.

---

## 22. A blanket `*.log` in `.gitignore` would have made ten provenance claims false [cost: ~15 min, would have cost reviewer trust]

**What happened.** With artifacts finally fetched into `results/raw/`, `measurements.json` was updated
to say 17 of 20 measurements were file-backed, naming the exact artifact for each. Ten of those
artifacts are a stage's `stdout.log`. The repo root `.gitignore` has a blanket `*.log`.

So the repository would have claimed, in writing and per-measurement, that ten numbers were backed by
committed files that git was silently declining to commit. A reviewer cloning the repo would have
found the claim and not the file. That is worse than the original artifact loss, because the original
loss was *disclosed* and this would have been an assertion that was false.

**How it was caught.** Not by reading the `.gitignore` — I had read it earlier in the session and it
looked fine, because `*.log` is a sensible rule and the problem only exists in combination with a
decision made later. It was caught because `check_measurement_provenance.py` had been written to test
that named artifacts exist, and extending it from "exists on disk" to "git would actually commit it"
was an obvious next question once the file paths were in a machine-readable field.

**A second bug inside the fix.** The first implementation used `git check-ignore -v`, which reports a
match for *negation* patterns too and exits 0 for them — so after adding `!*.log` to rescue the logs,
every rescued file still looked ignored. The check now asks the question directly: a path is
committable iff it appears in `git ls-files` or `git ls-files --others --exclude-standard`. Membership
in what git would commit is the actual question; interpreting ignore rules was a detour that
introduced its own error.

**Verified with a negative control**, because a check that has never failed is not known to work:
removing `!*.log` makes the check fail on all ten artifacts and name `.gitignore:36:*.log` as the
cause. Restoring it makes the check pass.

**Generalisable rule:** *"the file exists" and "the file is in the repository" are different claims,
and a provenance status asserts the second.* Any project that records where its evidence lives should
verify that record against what version control would actually preserve — otherwise the record is a
belief about the filesystem, held on one machine.

This is the fourth bug this session in one family: an output written somewhere nobody verified
(#19 deleted by a sync, #20 double-counted from a dirty directory, #21 written to a literal `RAW/`,
#22 excluded by an ignore rule). All four were invisible to the exit code.

---

## 23. A link check that could not fail [cost: ~10 min, one silently broken anchor]

**What happened.** `check_docs_consistency.py` verifies that every markdown link resolves. Its
implementation split each target on `#` and tested that the *path* part exists. For a link into
another file that is correct. For a same-document anchor — `[label](#some-heading)`, with nothing
before the `#` — the path part is the empty string, so it tested whether the containing directory
exists, which it always does. **Every anchor link in the project passed, unconditionally, including a
broken one.**

(Writing that sentence broke the check a second time: the example link above is inside backticks, and
the checker was matching link syntax inside code spans. It now blanks fenced blocks and inline code
before scanning, preserving line numbers. A checker for documentation has to tolerate documentation
about itself.)

The broken one was mine, written minutes earlier: I linked to a shortened form of a long heading. The
real GitHub slug includes the whole heading, brackets and all, so my anchor was a 68-character prefix
of a 150-character slug. It looks correct in the source and does not resolve.

**Why the shape is worth naming.** This is the same failure as sticking point #22 one level up. There,
`check_measurement_provenance.py` verified that artifacts existed on disk, which is not the claim a
provenance status makes. Here, the link checker verified something true by construction rather than the
thing it was written to check. Both passed. Both were reassuring and empty.

**Fix.** The checker now parses every heading, reproduces GitHub's slug rules, and validates the
fragment against them. When a fragment is a near-miss it prints the correct slug, which is what
actually made this a 30-second fix rather than a hunt.

**Generalisable rule:** *for every check, construct the failure it is supposed to catch and confirm it
catches it.* Both of this session's two new checks were verified against a deliberate break — removing
the `!*.log` negation, and the prefix anchor — and both times the negative control was the only reason
I knew the check worked. A green check is evidence only if red is reachable.

---

## 24. A four-minute silent hang that was `PATH`, and that I nearly "fixed" by replacing the host driver [cost: ~50 min, and a near-miss on breaking the whole environment]

First contact with the Native PyTorch drop. Ran a minimal probe: create two device tensors, matmul,
compare. It printed up to `matmul on device ...` and then produced nothing for four minutes.

What was visible at that point all pointed one way:

```
parent  29 threads, main on futex_do_wait, one thread on do_wait_intr_irq (waitpid)
child    1 thread,  futex_do_wait
```

Neither process using CPU. No `neuronx-cc` process. py-spy showing both stopped at the same Python
line with no frames beneath it. That is a textbook **fork-from-a-multithreaded-process deadlock** —
`fork()` copies only the calling thread, so a lock held by any of the other 28 is held forever in the
child, which blocks before `exec` while the parent waits on it.

**The dangerous part was that this diagnosis came with a plausible and expensive remedy.** The drop
ships its own driver, runtime, collectives and tools debs at internal build numbers
(`runtime-lib 2.x.59853.0`, `dkms 2.x.9869.0`) against the host's public `2.33.10.0` / `2.29.0.0`, and
Samir's instruction was to install them. "Native pip packages on a mismatched production runtime hangs
on first execution" is a completely credible story. Installing those debs would have replaced the host
Neuron driver and runtime, which would very likely have broken the XLA venv that every measurement in
this project was taken on — a ~10 minute DKMS rebuild each way to undo, assuming it undid cleanly.

`strace -f -e trace=clone,execve` took about a minute and gave the actual answer:

```
execve("/usr/local/sbin/neuronx-cc", ["neuronx-cc", "compile", "module.mlir", ...]) = -1 ENOENT
execve("/usr/local/bin/neuronx-cc",  ...) = -1 ENOENT     ... and 5 more, the whole PATH
```

On the first op needing a compile, the runtime forks and `execve`s **`neuronx-cc` by bare name**
through `PATH`. It lives in `native_venv/bin/`, and I had launched
`/home/ubuntu/native_venv/bin/python <script>` — an absolute path, which does **not** put the venv's
`bin/` on `PATH`. Every entry missed, and the child then hung rather than reporting it.

Activating the venv fixed it entirely. The same matmul completes in 1.21 s at `cos_sim 1.000002`, and
the deb packages turned out not to be needed at all.

**Three things to carry forward.**

*Invoking `venv/bin/python` directly is not equivalent to activating.* For almost all Python tooling
it is. It stops being true the moment a **subprocess** resolves a binary through `PATH` — which is
exactly what this compiler invocation does. Fixed permanently with `scripts/run_native.sh`, which
activates, asserts `neuronx-cc` resolves, prints the compiler version, and refuses to run otherwise.

*Changing instrument beat refining the diagnosis.* More process inspection would never have falsified
the fork-deadlock theory, because every process-state observation was genuinely consistent with it.
`strace` answered it in one shot. This is the fifth time in this project (#8, #19, #21→#24, #29's SBUF
story, this) that a well-fitting hypothesis was wrong and only a *different kind* of measurement
killed it.

*Following an explicit instruction immediately would have been the expensive choice.* Samir said to
install the debs, and that was reasonable advice for the situation as described to him. Diagnosing
before executing is what kept the environment intact — and the diagnosis showed the instruction
addressed a problem that did not exist. Worth noting because deferring an instruction from the person
helping you feels like the wrong call in the moment.

---

## 25. A structural counter contradicted an execution counter, and the verdict followed the wrong one [cost: ~20 min, one false negative on the project's decisive test]

The decisive Native PyTorch test — does stock `use_kernels=True` reach Neuron without any patch —
printed this:

```
    RoPE    swapped : 0  (expected 2)
    ...
    RoPE    dispatch : nki=2 fallback=0
```

and then concluded **"Gate 2 NOT cleared."** Two lines of the same output directly contradict each
other, and the pass criterion happened to be wired to the wrong one.

The dispatch counter is right: the NKI RoPE kernel ran twice, once per layer. The structural counter
walks `named_modules()` looking for instance-level `forward` overrides, and function kernels are
invisible to it for two independent reasons — stock `kernelize()` ends with
`finally: model.apply(detach_hidden_kernels)`, which `delattr`s the submodule alias, and the swap
mutates `fn.forward` rather than adding an attribute to any model submodule.

Two lessons, and the first is embarrassing because the project already knew it.

**Finding #8 established that execution counters are authoritative and structural inspection is
not.** That is the single most-repeated methodological point in this work. The probe collected the
right evidence and then adjudicated on the weaker signal anyway. Having the correct principle written
down is not the same as wiring it into the pass condition.

**And the obvious fix would also have failed.** My first repair matched qualnames inside
`_hidden_kernels` — which still returned 0, because the post-swap qualname is *identical* to the
pre-swap one: `_create_func_module.<locals>.Func.forward` both times, since the kernels library wraps
our function in a freshly generated `Func` module. Only object identity distinguishes them. Snapshot
`id(fn.forward)` before `kernelize()` and compare after: 2/2.

Had this run once and been believed, the conclusion would have been "Gate 2 survives on native" —
the exact opposite of the truth, on the question the whole native investigation existed to answer.
