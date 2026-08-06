"""Sync a local Markdown file into an EXISTING Quip document, in place.

WHY IN PLACE MATTERS
The obvious approach is to import the Markdown as a new Quip doc each time. The internal
"Markdown to Quip" Tampermonkey userscript does exactly that and does it well
(w.amazon.com/bin/view/Digital/Spree/AI-Initiatives/Tools/MarkdownToQuip/). But a new doc per
update means a new URL, and it throws away every comment your reviewers left. For a standing
status doc that Pinak and John are meant to read repeatedly and comment on, the thread has to
survive. So this updates one thread rather than creating threads.

HOW
Quip's Automation API renders Markdown natively — headings, lists, TABLES, code blocks and
blockquotes all come through, per the feature table on the MarkdownToQuip wiki page. So there is
no HTML conversion step and no lossy round trip; the .md in this repo stays the source of truth.

Quip has no "replace whole document" operation, so `--mode replace` does it in three steps:
  1. GET the thread and record the existing section IDs
  2. APPEND the new content
  3. DELETE each previously-recorded section
Append-then-delete rather than delete-then-append, deliberately: if the script dies midway you
are left with a doc containing both versions, which is recoverable by hand. The other order can
leave an empty document.

WHAT THIS COSTS YOU: inline comments anchored to specific paragraphs are attached to sections,
and those sections are replaced, so anchored comments will be orphaned. Thread-level comments
survive. If your reviewers comment inline a lot, use `--mode append` on a doc whose top section
you maintain by hand instead.

TOKEN — read this
Get one at https://quip-amazon.com/dev/token and pass it via the QUIP_API_TOKEN environment
variable. Never commit it, never paste it in Slack or email, and never use someone else's: a
Quip token grants full read/write as you. This script only ever reads it from the environment,
and refuses to accept it as a command-line argument so it cannot end up in your shell history.

    export QUIP_API_TOKEN='...'                     # not in the repo, not in a dotfile you commit
    python scripts/sync_quip.py --thread <ID> --file deliverables/status-and-questions.md
    python scripts/sync_quip.py --thread <ID> --file ... --apply

DRY RUN IS THE DEFAULT. Nothing is written without `--apply`. This script has not been executed
against a live document — I have no token and would not use one that is not mine — so treat the
first real run as a test: point it at a scratch Quip doc before your real one.

The thread ID is the token in the doc URL: https://quip-amazon.com/AbCdEfGhIjKl/My-Doc -> AbCdEfGhIjKl
"""

from __future__ import annotations  # so `str | None` works on Python < 3.10

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

# Amazon's Quip tenant. The public SaaS endpoint is platform.quip.com; override with --base-url
# if you are pointing at something else.
DEFAULT_BASE_URL = "https://platform.quip-amazon.com"

# Quip's edit-document `location` enum.
LOC_APPEND = 0
LOC_PREPEND = 1
LOC_AFTER_SECTION = 2
LOC_BEFORE_SECTION = 3
LOC_REPLACE_SECTION = 4
LOC_DELETE_SECTION = 5


class QuipError(RuntimeError):
    pass


def _request(base_url, path, token, data=None, method=None):
    url = f"{base_url.rstrip('/')}/1/{path.lstrip('/')}"
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method or ("POST" if data else "GET"))
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        # Deliberately does not echo the token, and 401/403 get an actionable hint rather than
        # a bare status code, since an expired token is the most likely failure here.
        hint = ""
        if e.code in (401, 403):
            hint = ("\n  The token is missing, expired, or lacks access to this thread. "
                    "Get a fresh one at https://quip-amazon.com/dev/token")
        elif e.code == 404:
            hint = "\n  Thread not found. Check the ID from the doc URL."
        elif e.code == 429:
            hint = "\n  Rate limited by Quip. Wait a minute and retry."
        raise QuipError(f"HTTP {e.code} on {path}: {detail}{hint}") from e
    except urllib.error.URLError as e:
        raise QuipError(
            f"Could not reach {url}: {e.reason}\n"
            "  Are you on the corporate network / VPN?"
        ) from e


def get_thread(base_url, token, thread_id):
    return _request(base_url, f"threads/{thread_id}", token)


def section_ids(thread) -> list[str]:
    """Every section id in the document, in document order.

    Quip returns the document as HTML with an `id` attribute on each section element, which is
    the documented way to address sections. Parsing them out with a regex rather than an HTML
    parser is adequate because we only need the id attributes, not structure — but it is the
    fragile part of this script, so the caller prints the count for eyeballing before applying.
    """
    html = thread.get("html") or ""
    return re.findall(r"<[a-zA-Z0-9]+[^>]*\bid=['\"]([^'\"]+)['\"]", html)


def edit(base_url, token, thread_id, *, content=None, location=LOC_APPEND,
         section_id=None, fmt="markdown"):
    data = {"thread_id": thread_id, "location": str(location), "format": fmt}
    if content is not None:
        data["content"] = content
    if section_id is not None:
        data["section_id"] = section_id
    return _request(base_url, "threads/edit-document", token, data=data)


def strip_leading_h1(md: str) -> tuple[str, str | None]:
    """Split off a leading `# Title` line.

    Quip treats the document's first line as its title, so pushing an H1 in as body content gives
    you the title twice. Returns (body, title-or-None).
    """
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if line.startswith("# "):
            return "\n".join(lines[i + 1:]).lstrip("\n"), line[2:].strip()
        break
    return md, None


def main():
    ap = argparse.ArgumentParser(
        description="Sync a Markdown file into an existing Quip document, in place.")
    ap.add_argument("--thread", required=True,
                    help="Quip thread ID, the token in the doc URL")
    ap.add_argument("--file", required=True, help="path to the .md file")
    ap.add_argument("--mode", choices=("replace", "append"), default="replace",
                    help="replace: swap the whole body (default). append: add to the end.")
    ap.add_argument("--format", dest="fmt", choices=("markdown", "html"), default="markdown",
                    help="Quip renders markdown natively including tables. Fall back to html "
                         "only if something specific fails to convert.")
    ap.add_argument("--keep-h1", action="store_true",
                    help="keep a leading '# Title' line in the body. Off by default, since Quip "
                         "already shows the document title and you would see it twice.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--apply", action="store_true",
                    help="actually write. Without this it is a dry run.")
    args = ap.parse_args()

    token = os.environ.get("QUIP_API_TOKEN")
    if not token:
        print("error: QUIP_API_TOKEN is not set.\n"
              "  Get a token at https://quip-amazon.com/dev/token, then:\n"
              "    export QUIP_API_TOKEN='...'\n"
              "  It is read from the environment on purpose — passing it as an argument would "
              "put it in your shell history.", file=sys.stderr)
        return 2

    md = open(args.file, encoding="utf-8").read()
    body, title = (md, None) if args.keep_h1 else strip_leading_h1(md)
    if not body.strip():
        print(f"error: {args.file} has no content to sync", file=sys.stderr)
        return 2

    print(f"file          {args.file}  ({len(body)} chars, {len(body.splitlines())} lines)")
    if title:
        print(f"stripped H1   {title!r}  (Quip shows the doc title already; --keep-h1 to keep it)")
    print(f"thread        {args.thread}")
    print(f"mode          {args.mode}")
    print(f"format        {args.fmt}")

    thread = get_thread(args.base_url, token, args.thread)
    meta = thread.get("thread", {})
    print(f"quip title    {meta.get('title')!r}")
    print(f"quip url      {meta.get('link')}")

    old = section_ids(thread)
    print(f"existing      {len(old)} section(s)")

    if not args.apply:
        print()
        print("DRY RUN — nothing written. Re-run with --apply.")
        if args.mode == "replace":
            print(f"Would append the new body, then delete {len(old)} old section(s).")
            print("Note: inline comments anchored to those sections will be orphaned. "
                  "Thread-level comments survive.")
        else:
            print("Would append the new body, leaving existing content alone.")
        return 0

    if args.mode == "append":
        edit(args.base_url, token, args.thread,
             content=body, location=LOC_APPEND, fmt=args.fmt)
        print("\nappended.")
        print(f"open: {meta.get('link')}")
        return 0

    # replace: append first, then delete the old sections. This order means a mid-run failure
    # leaves both versions present (recoverable) rather than an empty document.
    print("\nappending new content ...")
    edit(args.base_url, token, args.thread, content=body, location=LOC_APPEND, fmt=args.fmt)

    print(f"deleting {len(old)} old section(s) ...")
    failed = []
    for i, sid in enumerate(old, 1):
        try:
            edit(args.base_url, token, args.thread,
                 location=LOC_DELETE_SECTION, section_id=sid, fmt=args.fmt)
        except QuipError as e:
            # Keep going. A section can legitimately fail to delete if it was already removed
            # as part of a parent, and aborting here would leave a half-cleaned document.
            failed.append((sid, str(e).splitlines()[0]))
        if i % 25 == 0:
            print(f"  {i}/{len(old)}")

    print(f"\nsynced. {len(old) - len(failed)}/{len(old)} old sections removed.")
    if failed:
        print(f"{len(failed)} section(s) would not delete — check the doc and tidy by hand:")
        for sid, msg in failed[:5]:
            print(f"  {sid}: {msg}")
    print(f"open: {meta.get('link')}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QuipError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
