"""Consistency checks across every review-facing document. Runs anywhere, no hardware needed.

Four things this project got wrong by hand at least once, so they are checked mechanically:

1. **Broken links.** Docs cross-reference heavily and files were renamed mid-project.
2. **The 2.5-2.7x figure quoted without a qualifier.** That number comes from a CHAINED
   microbenchmark, which is deliberately NKI's worst case; in a real forward pass the device term is
   8.4% of the regression. It travelled without that context and drew a (correct) reviewer objection.
   Any occurrence must have a qualifier within ~700 characters.
3. **results/README.md out of sync with measurements.json.** The README is generated; hand-editing it
   or forgetting to re-render lets the human-readable and machine-readable numbers diverge.
4. **A measurement naming a script that does not exist.** Provenance is the point of
   measurements.json; a dangling script reference makes it worthless.

Usage:
    python scripts/check_docs_consistency.py
    make check-docs
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# A qualifier that makes the microbenchmark figure honest when it appears.
QUALIFIERS = (
    "microbenchmark", "chained", "worst case", "in situ", "in-situ", "8.4",
    "upper bound", "do not quote", "over-claim", "overstat", "superseded",
)
WINDOW = 700


def md_files():
    return sorted(p for p in ROOT.rglob("*.md") if ".git" not in str(p))


def check_links():
    print("=== links ===")
    bad = 0
    for doc in md_files():
        for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", doc.read_text()):
            if target.startswith(("http", "#", "mailto")):
                continue
            if not (doc.parent / target.split("#")[0]).exists():
                print(f"  BROKEN {doc.relative_to(ROOT)}: [{label}]({target})")
                bad += 1
    print("  ok" if not bad else f"  {bad} broken")
    return bad


def check_microbenchmark_qualifier():
    print("\n=== 2.5-2.7x qualifier ===")
    bad = 0
    for doc in md_files():
        text = doc.read_text()
        for m in re.finditer(r"2\.5[-–]2\.7x", text):
            window = text[max(0, m.start() - WINDOW): m.end() + WINDOW].lower()
            if not any(q in window for q in QUALIFIERS):
                line = text[: m.start()].count("\n") + 1
                print(f"  UNQUALIFIED {doc.relative_to(ROOT)}:{line}")
                bad += 1
    print("  ok — every occurrence carries a qualifier" if not bad else
          f"  {bad} unqualified occurrence(s) — add 'chained microbenchmark' or the in-situ figure")
    return bad


def check_results_in_sync():
    print("\n=== results/README.md in sync with measurements.json ===")
    readme = ROOT / "results" / "README.md"
    before = readme.read_text() if readme.exists() else ""
    subprocess.run([sys.executable, "scripts/render_results.py"], cwd=ROOT, capture_output=True)
    after = readme.read_text()

    def strip(s):  # the render timestamp always differs
        return "\n".join(l for l in s.splitlines() if not l.startswith("Rendered "))

    if strip(before) == strip(after):
        print("  ok")
        return 0
    print("  STALE — was out of sync, has been regenerated. Commit the change.")
    return 1


def check_measurement_provenance():
    print("\n=== measurement provenance ===")
    d = json.loads((ROOT / "results" / "measurements.json").read_text())
    bad = 0
    for m in d["measurements"]:
        for s in re.findall(r"scripts/[A-Za-z0-9_]+\.py", m.get("script", "")):
            if not (ROOT / s).exists():
                print(f"  MISSING {m['id']} -> {s}")
                bad += 1
    if not bad:
        print(f"  ok — all {len(d['measurements'])} measurements name existing scripts")
    return bad


def main():
    fail = (check_links()
            + check_microbenchmark_qualifier()
            + check_results_in_sync()
            + check_measurement_provenance())
    print("\n" + "=" * 58)
    print("ALL CHECKS PASS" if not fail else f"{fail} FAILURE(S)")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
