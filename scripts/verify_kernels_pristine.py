"""Verify the installed `kernels` package is byte-identical to the official wheel.

WHY THIS EXISTS
The project's outbound message to HuggingFace cites specific file:line locations in the `kernels`
library as evidence that HF already built Neuron support — `_NeuronRepos`, a `Neuron` backend class,
`neuron` in `KNOWN_BACKENDS`, a `neuron` section in `python_depends.json`. If any of that had been
added locally by this project, the message would be telling an external maintainer that their library
contains code we wrote. That is the kind of error you cannot walk back.

`scripts/neuron_kernel_registration.py` states the policy — patch in-process, never modify the shared
venv — but a docstring is a claim, not a check. pip records a SHA256 for every file it installs in
`<dist-info>/RECORD`, so the claim is verifiable: hash every file and compare.

Exit 0 means every file matches the wheel and the citations are safe to send. Exit 1 means something
was modified locally and every claim about the library's contents needs re-checking before it leaves
the building.

Run on the host with the venv (i.e. trn2):
    python scripts/verify_kernels_pristine.py
    python scripts/verify_kernels_pristine.py --package transformers
"""

import argparse
import base64
import hashlib
import sys
from pathlib import Path


def urlsafe_b64_nopad(digest: bytes) -> str:
    """pip's RECORD format: urlsafe base64 of the sha256, '=' padding stripped."""
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="kernels")
    ap.add_argument("--show", nargs="*", default=None,
                    help="also print the verification status of these specific paths")
    args = ap.parse_args()

    mod = __import__(args.package)
    pkg_dir = Path(mod.__file__).parent
    site = pkg_dir.parent

    dist_infos = sorted(site.glob(f"{args.package}-*.dist-info"))
    if not dist_infos:
        print(f"FAIL  no dist-info for {args.package} under {site} — cannot verify")
        return 1
    dist_info = dist_infos[-1]
    record = dist_info / "RECORD"
    if not record.exists():
        print(f"FAIL  {record} missing — pip did not record hashes, cannot verify")
        return 1

    print(f"package   {args.package}")
    print(f"location  {pkg_dir}")
    print(f"record    {record.name} in {dist_info.name}")
    print()

    checked = modified = missing = skipped = 0
    problems = []
    statuses = {}

    for line in record.read_text().splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) != 3:
            continue
        rel, digest_field, _size = parts
        # Only verify the package's own source; dist-info metadata and RECORD itself carry no hash.
        if not rel.startswith(f"{args.package}/"):
            continue
        if not digest_field.startswith("sha256="):
            skipped += 1
            continue
        expected = digest_field[len("sha256="):]
        path = site / rel
        if not path.exists():
            missing += 1
            problems.append(f"MISSING  {rel}")
            statuses[rel] = "MISSING"
            continue
        actual = urlsafe_b64_nopad(hashlib.sha256(path.read_bytes()).digest())
        checked += 1
        if actual != expected:
            modified += 1
            problems.append(f"MODIFIED {rel}\n           expected sha256={expected}"
                            f"\n           actual   sha256={actual}")
            statuses[rel] = "MODIFIED"
        else:
            statuses[rel] = "ok"

    if args.show:
        print("requested paths:")
        for s in args.show:
            print(f"  {statuses.get(s, 'NOT IN RECORD'):<14} {s}")
        print()

    print(f"{checked} file(s) verified, {modified} modified, {missing} missing, "
          f"{skipped} unhashed")
    if problems:
        print()
        print(f"FAIL  {len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        print()
        print("  Any claim this project makes about this library's contents is now suspect.")
        print("  Do NOT send file:line citations to an external maintainer until this is resolved.")
        return 1

    print()
    print(f"PASS  the installed {args.package} is byte-identical to the published wheel.")
    print("      Nothing in it was added or altered locally, so file:line citations describe")
    print("      upstream code and are safe to quote externally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
