# Raw artifacts

**Currently empty.** This is a known gap, not an oversight in progress.

Every measurement in this project was run against a trn2 instance that has since expired, and every
raw artifact lived in `/tmp` on that host:

- `measure_mfu.py --json-out` result files
- `neuron-explorer view --output-format summary-json` output
- NEFF / NTFF device-profile pairs
- detached run logs from `scripts/run_detached.sh`

All of it is gone. The numbers survived because each run's stdout was pasted into the git commit
message at the time, and every producing script is committed — so the results are reproducible, but
they are not currently *auditable* against a file. `results/measurements.json` marks each number
`transcribed` for exactly this reason.

## To populate this directory

On a fresh trn2 with the repo synced and the Neuron venv active:

```bash
make results
```

That runs `scripts/regenerate_results.py`, which re-runs every measurement sequentially — stages
cannot overlap, since two Neuron processes contend for the same cores — and writes into
`results/raw/<stage>/` plus an `index.json` mapping stage to command, exit code, duration and
artifact list.

Then transcribe any changed numbers into `results/measurements.json`, flip their `status` from
`transcribed` to `in_repo`, and re-render:

```bash
python scripts/render_results.py
```

## The lesson

Results should never have lived only on an ephemeral host. The project tracked the risk of "commits
exist only on one laptop" and missed the sharper version of the same risk one layer down: the
*evidence* was on a machine with a shorter lifetime than the laptop. Writing artifacts into the repo
costs nothing and is the difference between a reviewable result and a remembered one.
