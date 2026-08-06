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
import sys
from html.parser import HTMLParser
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


def new_document(base_url, token, content, *, title=None, fmt="markdown"):
    """Create a new Quip document and return the thread payload.

    Used once, to bring the doc into existence. Every update after that goes through
    edit-document against the returned thread id, so the URL and the comment thread persist.
    """
    data = {"content": content, "format": fmt}
    if title:
        data["title"] = title
    return _request(base_url, "threads/new-document", token, data=data)


# Void elements never get a closing tag, so they must not change nesting depth.
_VOID_TAGS = {"br", "img", "hr", "input", "meta", "link", "col", "area", "base",
              "embed", "source", "track", "wbr"}


class _TopLevelSections(HTMLParser):
    """Collect `id` attributes of TOP-LEVEL elements only.

    This has to be depth-aware, and the reason is worth recording. Quip assigns a section id to
    every addressable element, which for a table means one per CELL. A regex over all `id=`
    occurrences reported 938 sections for a 213-line document with 8 tables — and since
    edit-document deletes one section per HTTP call, that is 938 requests, which is slow and
    walks straight into Quip's rate limit.

    Deleting the top-level element removes everything nested inside it, so the top-level ids are
    both sufficient and two orders of magnitude cheaper. Quip's document HTML has no wrapper
    element (it starts directly with `<h1 id=...>`), so depth 0 is exactly the set we want.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.ids: list[str] = []
        # (id, tag, text) per top-level section. The text is needed to locate a marker
        # heading for --mode section, so a sync can target part of a document.
        self.sections: list[tuple[str, str, str]] = []
        self._cur: list | None = None

    def _maybe_record(self, tag, attrs):
        if self.depth == 0:
            sid = dict(attrs).get("id")
            if sid:
                self.ids.append(sid)
                return [sid, tag.lower(), []]
        return None

    def handle_starttag(self, tag, attrs):
        opened = self._maybe_record(tag, attrs)
        if opened is not None and self._cur is None:
            self._cur = opened
        if tag.lower() not in _VOID_TAGS:
            self.depth += 1

    def handle_startendtag(self, tag, attrs):
        opened = self._maybe_record(tag, attrs)  # self-closing: no depth change
        if opened is not None:
            self.sections.append((opened[0], opened[1], ""))

    def handle_endtag(self, tag):
        if tag.lower() not in _VOID_TAGS:
            self.depth = max(0, self.depth - 1)
        if self.depth == 0 and self._cur is not None:
            sid, t, parts = self._cur
            self.sections.append((sid, t, "".join(parts).strip()))
            self._cur = None

    def handle_data(self, data):
        if self._cur is not None:
            self._cur[2].append(data)


def _parse(thread) -> _TopLevelSections:
    parser = _TopLevelSections()
    parser.feed(thread.get("html") or "")
    parser.close()
    return parser


def section_ids(thread) -> list[str]:
    """Top-level section ids, in document order."""
    return _parse(thread).ids


def sections_after_marker(thread, marker: str):
    """Return (marker_id, [ids after it]) for the first top-level section containing `marker`.

    This is what makes it safe to sync into a document somebody else owns the top of. The
    project plan lives above a "Progress Tracking:" heading; everything below that heading is
    ours to replace on every sync, and everything above it is never touched.

    Matching is on the section's visible text, case-insensitively, and prefers a heading over a
    paragraph so a passing mention in prose cannot be mistaken for the marker.
    """
    secs = _parse(thread).sections
    needle = marker.strip().lower()

    idx = None
    for i, (_sid, tag, text) in enumerate(secs):
        if needle in text.lower() and tag.startswith("h"):
            idx = i
            break
    if idx is None:  # fall back to any section type
        for i, (_sid, _tag, text) in enumerate(secs):
            if needle in text.lower():
                idx = i
                break
    if idx is None:
        raise QuipError(
            f"marker {marker!r} not found among {len(secs)} top-level sections.\n"
            "  Add a heading with that text to the Quip doc, or pass --after-heading."
        )
    return secs[idx][0], [sid for sid, _t, _x in secs[idx + 1:]]


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
    ap.add_argument("--thread",
                    help="Quip thread ID, the token in the doc URL. Omit with --create.")
    ap.add_argument("--create", action="store_true",
                    help="create a NEW document instead of updating one, and print its thread "
                         "ID. Run this once; use --thread for every update after that, so the "
                         "URL and the comment thread survive.")
    ap.add_argument("--file", required=True, help="path to the .md file")
    ap.add_argument("--mode", choices=("replace", "append", "section"), default="section",
                    help="section (default): replace only the content BELOW --after-heading, "
                         "leaving everything above it untouched. Use this when the doc has "
                         "content you do not own. replace: swap the whole body. append: add to "
                         "the end without removing anything.")
    ap.add_argument("--after-heading", default="Progress Tracking",
                    help="marker heading for --mode section. Everything below it is replaced on "
                         "each sync; everything above it is never touched.")
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

    if not args.create and not args.thread:
        print("error: pass --thread <ID> to update a doc, or --create to make a new one.",
              file=sys.stderr)
        return 2

    print(f"file          {args.file}  ({len(body)} chars, {len(body.splitlines())} lines)")
    if title:
        print(f"stripped H1   {title!r}  (used as the Quip doc title)")

    if args.create:
        print(f"mode          CREATE a new document")
        print(f"format        {args.fmt}")
        if not args.apply:
            print("\nDRY RUN — nothing written. Re-run with --apply to create it.")
            return 0
        res = new_document(args.base_url, token, body, title=title, fmt=args.fmt)
        meta = res.get("thread", {})
        print(f"\ncreated.")
        print(f"  thread id   {meta.get('id')}")
        print(f"  url         {meta.get('link')}")
        print(f"  title       {meta.get('title')!r}")
        print(f"\nFor every update from now on:")
        print(f"  make quip-status QUIP_THREAD={meta.get('id')} APPLY=1")
        return 0

    print(f"thread        {args.thread}")
    print(f"mode          {args.mode}")
    print(f"format        {args.fmt}")

    thread = get_thread(args.base_url, token, args.thread)
    meta = thread.get("thread", {})
    print(f"quip title    {meta.get('title')!r}")
    print(f"quip url      {meta.get('link')}")

    if args.mode == "section":
        marker_id, old = sections_after_marker(thread, args.after_heading)
        total = len(section_ids(thread))
        print(f"marker        {args.after_heading!r} -> {marker_id}")
        print(f"sections      {total} total, {total - len(old) - 1} preserved above the marker, "
              f"{len(old)} below it will be replaced")
    else:
        old = section_ids(thread)
        print(f"existing      {len(old)} section(s)")

    if not args.apply:
        print()
        print("DRY RUN — nothing written. Re-run with --apply.")
        if args.mode == "section":
            print(f"Would append the new body, then delete the {len(old)} section(s) currently "
                  f"below {args.after_heading!r}.")
            print("Everything above the marker is left alone.")
        elif args.mode == "replace":
            print(f"Would append the new body, then delete ALL {len(old)} section(s).")
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

    # replace / section: append first, then delete the old sections. This order means a mid-run
    # failure leaves both versions present (recoverable) rather than an empty document.
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
