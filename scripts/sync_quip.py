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
        # (id, tag, text) per top-level block. Text is needed to locate a marker heading, so
        # --mode section can target part of a document.
        self.sections: list[tuple[str, str, str]] = []
        self._cur: list | None = None   # [id, tag, text-parts] for the block being walked
        self._need_id = False           # inside a depth-0 block that had no id of its own

    def _open_block(self, tag, attrs):
        """Record the outermost id-bearing element of each top-level block.

        Descending matters. Quip wraps a table in an id-LESS div:
            <div data-section-style='13'><table id='...'>...</table></div>
        so a strict depth-0 rule finds no id for that block and the table becomes unaddressable.
        That is not cosmetic: DELETE_SECTION was never being called on tables, they survived
        every sync, and the tracking section accumulated a duplicate set of tables each time —
        silently, because deleting the paragraphs around them succeeded.
        """
        sid = dict(attrs).get("id")
        if sid:
            self.ids.append(sid)
            self._cur = [sid, tag.lower(), []]
            self._need_id = False
        else:
            self._need_id = True        # look for an id on the way down

    def handle_starttag(self, tag, attrs):
        if self.depth == 0:
            self._open_block(tag, attrs)
        elif self._need_id:
            sid = dict(attrs).get("id")
            if sid:
                self.ids.append(sid)
                self._cur = [sid, tag.lower(), []]
                self._need_id = False
        if tag.lower() not in _VOID_TAGS:
            self.depth += 1

    def handle_startendtag(self, tag, attrs):
        # Self-closing, so no depth change. Only meaningful as a whole block at depth 0.
        if self.depth == 0:
            sid = dict(attrs).get("id")
            if sid:
                self.ids.append(sid)
                self.sections.append((sid, tag.lower(), ""))
        elif self._need_id:
            sid = dict(attrs).get("id")
            if sid:
                self.ids.append(sid)
                self._cur = [sid, tag.lower(), []]
                self._need_id = False

    def handle_endtag(self, tag):
        if tag.lower() not in _VOID_TAGS:
            self.depth = max(0, self.depth - 1)
        if self.depth == 0:
            if self._cur is not None:
                sid, t, parts = self._cur
                self.sections.append((sid, t, "".join(parts).strip()))
                self._cur = None
            self._need_id = False

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


# TABLE WIDTHS CANNOT BE SET THROUGH THIS API. Recorded here so nobody spends the afternoon I
# spent on it.
#
# Quip hardcodes every imported table column to style='width: 6em' -- roughly 11 characters --
# regardless of content. A column holding "status" and a column holding a 200-character sentence
# get the same 6em, so wide-content tables come out unreadably narrow. Quip emits that attribute
# in its own output, which made it look like an input it would honour. It is not. Five encodings
# were tested on throwaway documents and all five were stripped:
#
#   style="width: 30em" on <th>            -> 6em
#   width="480" attribute on <th>          -> 6em
#   <colgroup><col style="width: 30em">    -> 6em
#   percentage widths + width:100% table   -> 6em
#   post-import REPLACE_SECTION on the <th> with a width -> API error
#
# Quip owns table geometry. The width machinery below is therefore INERT for its original
# purpose; it is kept because md_to_html remains a working markdown->HTML path and the width
# computation is harmless, but --format markdown is the default and loses nothing.
#
# The real fix is editorial: keep prose out of tables, since Quip renders paragraphs and lists at
# full document width. deliverables/progress-tracking.md is written that way.
TABLE_BUDGET_EM = 46.0   # roughly the usable width of a Quip document body
MIN_COL_EM = 4.0
EM_PER_CHAR = 0.5
# Allocation is proportional to sqrt(desired width), not to desired width itself. Straight
# proportionality lets one very long column starve the others: in the open-questions table a
# 120-character "why it matters" column pushed "Team" down to 3.4em, too narrow for the word
# "frameworks". Dampening keeps the ordering (wide content still gets wide columns) while
# stopping the extremes from crowding everything else out.
WIDTH_DAMPEN = 0.5


def _cell_texts(table_html: str) -> list[list[str]]:
    """Rows of cell text from a markdown-generated table. Regex is defensible here because the
    input is the `markdown` library's own predictable output, not arbitrary HTML."""
    rows = []
    for row in re.findall(r"<tr>(.*?)</tr>", table_html, re.S):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)
        rows.append([re.sub(r"<[^>]+>", "", c).strip() for c in cells])
    return rows


def _column_widths(rows: list[list[str]]) -> list[float]:
    """Allocate TABLE_BUDGET_EM across columns in proportion to the longest cell in each.

    Proportional rather than absolute: a table whose columns all want 30em cannot have them, so
    the budget is divided by relative need. Columns are floored at MIN_COL_EM so a narrow column
    stays clickable, and the result is rescaled so the total still matches the budget.
    """
    ncols = max((len(r) for r in rows), default=0)
    if not ncols:
        return []
    # A fixed budget divided eight ways leaves every column too narrow to hold a word like
    # "Consulted", so wide tables get a bigger budget. Capped, because past a point a wide table
    # just becomes horizontal scrolling, which reads worse than wrapping.
    budget = min(TABLE_BUDGET_EM + max(0, ncols - 4) * 5.0, 76.0)
    want = []
    for c in range(ncols):
        longest = max((len(r[c]) for r in rows if c < len(r)), default=1)
        want.append(max(MIN_COL_EM, min(longest * EM_PER_CHAR, budget)))
    damped = [w ** WIDTH_DAMPEN for w in want]
    total = sum(damped)
    if total <= 0:
        return [budget / ncols] * ncols
    scaled = [max(MIN_COL_EM, w * budget / total) for w in damped]
    # Rescale once more, since clamping to the minimum can push the sum back over budget.
    s = sum(scaled)
    return [round(w * budget / s, 2) for w in scaled]


def _widen_tables(html: str) -> tuple[str, int]:
    """Inject explicit per-column widths into every table. Returns (html, tables_widened)."""
    count = 0

    def fix(match):
        nonlocal count
        table = match.group(0)
        widths = _column_widths(_cell_texts(table))
        if not widths:
            return table
        count += 1

        col = iter(range(len(widths)))

        def add_width(cell_match):
            i = next(col, None)
            if i is None or i >= len(widths):
                return cell_match.group(0)
            tag, attrs = cell_match.group(1), cell_match.group(2)
            return f"<{tag}{attrs} style=\"width: {widths[i]}em\">"

        # Only the header row carries widths; Quip applies them down the column.
        head = re.search(r"<thead>.*?</thead>", table, re.S)
        if head:
            fixed_head = re.sub(r"<(th)((?:\s[^>]*)?)>", add_width, head.group(0))
            table = table[:head.start()] + fixed_head + table[head.end():]
        total = round(sum(widths), 2)
        return table.replace("<table>", f'<table style="width: {total}em">', 1)

    return re.sub(r"<table>.*?</table>", fix, html, flags=re.S), count


def md_to_html(md: str) -> tuple[str, int]:
    """Markdown -> HTML with table widths set from content. Returns (html, tables_widened)."""
    try:
        import markdown as _md
    except ImportError:
        raise QuipError(
            "--format html needs the `markdown` package:\n"
            "    python3 -m pip install --user markdown\n"
            "  Or use --format markdown, which needs no dependency but renders tables narrow."
        )
    html = _md.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    return _widen_tables(html)


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

    # --format html means "convert, then send HTML". Previously it passed raw markdown through
    # with format=html, which would have rendered as literal pipe characters — a latent bug,
    # since nothing exercised that path.
    widened = 0
    if args.fmt == "html":
        body, widened = md_to_html(body)

    if not args.create and not args.thread:
        print("error: pass --thread <ID> to update a doc, or --create to make a new one.",
              file=sys.stderr)
        return 2

    print(f"file          {args.file}  ({len(body)} chars, {len(body.splitlines())} lines)")
    if title:
        print(f"stripped H1   {title!r}  (used as the Quip doc title)")
    if args.fmt == "html":
        print(f"converted     markdown -> HTML, {widened} table(s) given explicit column widths "
              f"(budget {TABLE_BUDGET_EM:g}em)")

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
