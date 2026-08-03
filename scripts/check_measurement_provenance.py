"""Check that every measurement's provenance claim is internally consistent.

measurements.json is the source of truth for every number in the project, and its `status` field is
a claim about evidence: `transcribed` means the raw artifact is gone, `reproduced` and `in_repo` mean
a file backs the number. A wrong status is worse than a missing one, because it invites a reviewer to
look for a file that is not there.

This checks the claims against each other and against the filesystem:

  1. JSON parses, and every measurement has id / what / script / status
  2. status is one of the documented values in _about.provenance_status
  3. any status other than `transcribed` names an `artifact`
  4. every path in an `artifact` field exists, allowing {a,b} brace sets and trailing /
  5. every named artifact is actually COMMITTABLE — not excluded by a .gitignore
  6. every `script` path exists
  7. a `reproduced` block, if present, carries a `delta` saying how far the re-run moved

Check 5 is the one that earned its place. The repo root .gitignore carries a blanket `*.log`, which
silently excluded every harness stage's stdout.log — the named artifact for ten measurements. They
existed on disk, so an existence check passed, and they would have been recorded as committed while
being invisible to anyone who cloned the repo. "Exists on my machine" and "is in the repo" are
different claims and only the second one is what a provenance status asserts.

Runs anywhere — no hardware, no network. Needs git only for check 5, which is skipped if git is
unavailable rather than failing.

    python scripts/check_measurement_provenance.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "results" / "measurements.json"

REQUIRED = ("id", "what", "script", "status")


def split_top_level(s: str):
    """Split on commas that are NOT inside braces.

    An artifact field lists several paths comma-separated and also uses brace sets, so a naive
    split on ',' cuts 'prof_{silu,rmsnorm}' in half and then reports both halves as missing files.
    """
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def expand(spec: str):
    """Expand 'a/prof_{x,y}/f' into concrete paths. Artifact fields use brace sets to keep
    closely-related profile directories on one line instead of six."""
    m = re.search(r"\{([^}]*)\}", spec)
    if not m:
        return [spec]
    out = []
    for alt in m.group(1).split(","):
        out += expand(spec[:m.start()] + alt.strip() + spec[m.end():])
    return out


def _git(*args):
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return p


def committable_set():
    """Every path git would put in a commit: tracked, plus untracked-and-not-ignored.

    Membership in this set is the question a provenance status actually depends on, and asking it
    directly avoids interpreting ignore rules. `git check-ignore -v` is not usable here: it reports
    a match for NEGATION patterns too and exits 0 for them, so a file rescued by `!*.log` looks
    identical to one excluded by `*.log`. That produced a false failure on every stdout.log the
    negation had just fixed.

    Returns None if git is unavailable, so the caller can skip rather than fail.
    """
    tracked = _git("ls-files")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    if tracked is None or tracked.returncode != 0 or untracked is None:
        return None
    return set(tracked.stdout.split("\n")) | set(untracked.stdout.split("\n"))


def why_ignored(path: str) -> str:
    """The .gitignore rule responsible, for the error message. Best effort."""
    p = _git("check-ignore", "-v", "--", path)
    if p and p.returncode == 0 and p.stdout.strip():
        rule, _, _ = p.stdout.splitlines()[0].rpartition("\t")
        return rule
    return "an unknown rule"


def files_under(p: Path):
    """The files a named artifact covers: itself if a file, everything beneath it if a directory."""
    if p.is_file():
        return [p]
    if p.is_dir():
        return [f for f in p.rglob("*") if f.is_file()]
    return []


def main():
    try:
        d = json.loads(SRC.read_text())
    except json.JSONDecodeError as e:
        print(f"FAIL  {SRC.name} does not parse: {e}")
        return 1

    known = set(d["_about"]["provenance_status"])
    ms = d["measurements"]
    problems = []
    committable = committable_set()
    if committable is None:
        print("NOTE  git unavailable; skipping the committability check\n")

    print(f"{len(ms)} measurements in {SRC.relative_to(ROOT)}")
    print()

    counts = {}
    for m in ms:
        mid = m.get("id", "<no id>")
        for f in REQUIRED:
            if f not in m:
                problems.append(f"{mid}: missing required field '{f}'")

        st = m.get("status", "")
        counts[st] = counts.get(st, 0) + 1
        if st and st not in known:
            problems.append(f"{mid}: status '{st}' is not in _about.provenance_status {sorted(known)}")

        if st and st != "transcribed" and "artifact" not in m:
            problems.append(f"{mid}: status '{st}' claims a file backs it but names no 'artifact'")

        for spec in split_top_level(str(m.get("artifact", ""))):
            # Drop a parenthetical aside like "(profile_dir prof_n28)"
            spec = re.sub(r"\s*\(.*?\)\s*", "", spec).strip()
            if not spec:
                continue
            for p in expand(spec):
                target = ROOT / p.rstrip("/")
                if not target.exists():
                    problems.append(f"{mid}: artifact '{p}' does not exist")
                    continue
                # An artifact that exists but is git-ignored is not evidence: it is invisible to
                # anyone who clones the repo, which is the audience a provenance status is for.
                # Binaries are exempt — they are deliberately ignored and the extracted JSON is
                # what the numbers rest on.
                if committable is None:
                    continue
                for f in files_under(target):
                    if f.suffix in (".ntff", ".neff", ".pb"):
                        continue
                    rel = str(f.relative_to(ROOT))
                    if rel not in committable:
                        problems.append(
                            f"{mid}: artifact '{rel}' EXISTS but git would not commit it "
                            f"({why_ignored(rel)}) — so the '{st}' status would be false for "
                            f"anyone who cloned the repo")

        for s in split_top_level(str(m.get("script", "")).replace("+", ",")):
            if s.endswith(".py") and not (ROOT / s).exists():
                problems.append(f"{mid}: script '{s}' does not exist")

        rep = m.get("reproduced")
        if isinstance(rep, dict) and "delta" not in rep and "note" not in rep:
            problems.append(f"{mid}: has a 'reproduced' block with no 'delta' — a reproduction "
                            f"without a stated distance from the original is not a check")

    for st in sorted(counts):
        print(f"  {counts[st]:2d}  {st}")
    print()

    if problems:
        print(f"FAIL  {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("PASS  every provenance claim is consistent and every named file exists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
