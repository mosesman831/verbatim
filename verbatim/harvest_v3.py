"""V3 harvester: structure-aware segmentation for v3 source envelopes.

SPEC_V3 §15 — harvesting v3: segmentation, language, references, predicates,
and time. This module owns the segmentation and candidate-shape half;
``harvest_v3_refs`` owns structured reference extraction (V3-15.05/§15.09).

Design contract:

- ``harvest_v3(text, *, envelope_kind, ...)`` returns a list of candidate
  dicts ``{"text", "kind_hint", "refs", "start_byte", "end_byte"}`` — the
  same ``text``-keyed dict shape the v2 ``context_groups`` write path
  consumes downstream (V3-15.12: deterministic under
  ``HARVESTER_V3_VERSION``).
- ``kind_hint`` is one of ``command`` | ``path`` | ``error`` | ``test`` |
  ``prose`` | ``hunk`` — retrieval lanes read it to route candidates.
- Per-kind segmentation (SPEC_V3 §12 envelope table):

  * ``tool_call`` / ``tool_result``: tool-call boundaries, command lines,
    file-path lines, error blocks — atomic tool interactions stay whole.
  * ``file_diff``: one candidate per ``@@`` hunk with the file-header
    context line preserved (``+++ b/path`` preferred) plus the file path in
    ``candidate["file"]`` and ``refs["paths"]``.
  * ``test_result`` / ``verification``: test-name lines (``FAILED x::y``,
    TAP ``ok``/``not ok``, go ``--- FAIL:``), assertion-failure blocks,
    summary lines.
  * ``error`` / ``recovery``: stack-frame blocks and exception lines —
    one candidate per traceback/exception run.
  * ``plan`` / ``subgoal`` / ``decision`` / ``agent_note`` / ``lesson`` and
    all other kinds: v2-compatible prose path — delegated to
    ``core.harvest.harvest`` so punctuation-light chat, CJK, list items,
    and acknowledgement filtering behave identically (V3-15.01, V3-15.02).

- Language tolerance: prose kinds inherit v2's Unicode-aware boundaries
  (CJK terminators, no-whitespace text is never skipped). Structured kinds
  segment on line structure, not punctuation, so scripts without sentence
  punctuation still yield candidates.
- Verbatim: every emitted ``text`` is an exact substring of the input
  except diff hunks, which prepend the file-context line; the hunk body
  remains an exact substring and its ``start_byte``/``end_byte`` cover the
  body. Nothing is truncated mid-token to fit ``max_len`` silently —
  oversized lines hard-cut at character boundaries (still verbatim
  substrings) rather than being dropped (V3-12.06 spirit: no silent loss).
- Pathological input (``None``, non-text, empty, nonsensical bounds)
  returns ``[]`` — harvesting never crashes the ingest path. Undecodable
  bytes are NOT pathological: they raise a typed ``VerbatimError``
  (``VALIDATION``) so a decode failure can never masquerade as clean-but-
  empty content (V4-13.11/13.12).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .core.harvest import _byte_offsets, harvest as _harvest_v2
from .core.types import ErrorCode, VerbatimError
from .core.types_v3 import EnvelopeKind
from .harvest_v3_refs import extract_refs

HARVESTER_V3_VERSION = "harvest-v3-1"

#: Candidate ``kind_hint`` vocabulary consumed by retrieval lanes.
KIND_HINTS = frozenset({"command", "path", "error", "test", "prose", "hunk"})

_TOOL_KINDS = frozenset({EnvelopeKind.TOOL_CALL, EnvelopeKind.TOOL_RESULT})
_ERROR_KINDS = frozenset({EnvelopeKind.ERROR, EnvelopeKind.RECOVERY})
_DIFF_KINDS = frozenset({EnvelopeKind.FILE_DIFF})
_TEST_KINDS = frozenset({EnvelopeKind.TEST_RESULT, EnvelopeKind.VERIFICATION})
# Everything else — user/assistant messages, document/import/connector,
# plan/subgoal/decision, agent_note/lesson, handoff/delegation, refs and
# system events — uses the v2 prose path (V3-15.01 parity).

# ---------------------------------------------------------------------------
# Versioned structural patterns
# ---------------------------------------------------------------------------

#: Pure separator rules (``===``, ``---``, ``***``, ``___``) and banner lines
#: like ``==== short test summary info ====``. Never matches ``--- a/file``
#: (has non-separator text) or diff body lines.
_SEP_RE = re.compile(
    r"^\s*(?:[-=~_*#]{3,}|={3,}[^\n=]*={3,}|_{2,}[^\n_]*_{2,})\s*$"
)

#: Error-block openers: python/java tracebacks, fatal/panic markers, log
#: severities, causal-chain connectors.
_ERR_START_RE = re.compile(
    r"^\s*(?:Traceback \(most recent call last\)|panic:|fatal:|FATAL\b"
    r"|ERROR\b|CRITICAL\b|Error\b|Exception\b|Caused by:|Suppressed:"
    r"|\w+\s+ERR!(?=\s|$)"
    r"|During handling of the above exception"
    r"|The above exception was the direct cause)",
    re.IGNORECASE,
)

#: Exception-class lines (``ValueError: bad``, bare ``ValueError``) and
#: ``file:line: Error`` tails (``x.py:10: AssertionError``).
_EXC_LINE_RE = re.compile(
    r"^\s*(?:[A-Za-z_][\w.]*(?:Error|Exception|Fault|Failure|Warning)\s*(?::|$)"
    r"|[^\s:]+:\d+:\s*[A-Za-z_][\w.]*(?:Error|Exception|Fault|Failure)\b)"
)

#: Stack-frame and continuation lines inside an error block.
_FRAME_RE = re.compile(
    r"^\s*(?:File \"[^\"]+\", line \d+|at\s+[\w.$]+\s*\(|\.{3}\s|\^+\s*$"
    r"|~+\s*\^+\s*$)"
)

#: Command lines: shell prompts, call-shaped tool uses (``EditFile(...)``),
#: and shell-verb-led invocations (``pytest tests/ -x``). Call/verb forms
#: require column 0 so indented source lines inside tracebacks (``    f()``)
#: stay block continuations, not commands.
_CMD_PROMPT_RE = re.compile(r"^\s*[$#>]\s*\S")
_CMD_CALL_RE = re.compile(r"^[A-Za-z_][\w.]*\s*\(")
_CMD_VERB_RE = re.compile(
    r"^(?:sudo\s+)?(?:git|npm|pnpm|yarn|bun|deno|node|python\d?|pip\d?"
    r"|pytest|tox|nox|cargo|go|make|cmake|ninja|docker|kubectl|helm"
    r"|terraform|ansible|apt(?:-get)?|brew|systemctl|service|curl|wget"
    r"|ssh|rsync|tar|grep|rg|find|sed|awk|cat|ls|cd|mkdir|rm|cp|mv|chmod"
    r"|chown|gcc|g\+\+|clang|javac|java|mvn|gradle|dotnet|ruby|gem|php"
    r"|composer|perl|tsc|eslint|prettier|jest|mocha|vitest|playwright"
    r"|sqlite3|psql|mysql|aws|gcloud|az|gh|pandoc)\s+\S"
)

#: Test-result lines: pytest short-summary (``FAILED x::y - msg``), jest
#: PASS/FAIL, go ``--- FAIL:``, TAP ``ok 1``/``not ok 2``, check marks.
_TEST_LINE_RE = re.compile(
    r"^\s*(?:(?:FAILED|PASSED|XFAILED|XPASSED|SKIPPED|ERROR)\s+\S"
    r"|-{1,3}\s*(?:FAIL|PASS|SKIP|BENCH)\s*:?"
    r"|(?:not ok|ok)\s+\d"
    r"|[✓✔✗✘×]\s*\S"
    r"|(?:PASS|FAIL)\s+\S)"
)

#: A line that is only a file path (optionally with ``:line`` / ``(line N)``).
_PATH_LINE_RE = re.compile(
    r"^\s*(?:[A-Za-z]:[\\/]|~[\\/]|\.{1,2}[\\/]|/)?"
    r"[\w.@~+\-]+(?:[\\/][\w.@~+\-]+)+[\\/]?"
    r"\s*(?::\d+(?::\d+)?|\((?:line\s+)?\d+(?:,\s*\d+)?\))?\s*$"
)

#: Pytest failure-detail section headers: ``______ test_y ______``. The name
#: may itself contain underscores, so the middle is any non-empty text.
_FAILHDR_RE = re.compile(r"^\s*_{2,}\s*\S.*?\s*_{2,}\s*$")

#: Unified-diff hunk header.
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


# ---------------------------------------------------------------------------
# Line machinery
# ---------------------------------------------------------------------------


def _line_spans(text: str) -> list[tuple[int, int]]:
    """Char span of each line, excluding the newline itself."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def _run(classes: list[str], start: int, cont: frozenset | set) -> int:
    """Extend a block while classes are in ``cont``; blank lines ride along
    only when a continuing line follows (trailing blanks never merge)."""
    n = len(classes)
    j = start
    while j < n:
        c = classes[j]
        if c in cont:
            j += 1
            continue
        if c == "blank":
            k = j
            while k < n and classes[k] == "blank":
                k += 1
            if k < n and classes[k] in cont:
                j = k
                continue
        return j
    return j


# ---------------------------------------------------------------------------
# Tool-call / tool-result / error / recovery segmentation
# ---------------------------------------------------------------------------

_ERR_CONT = frozenset({"err", "exc", "frame", "ind"})
_PROSE_CONT = frozenset({"prose", "ind"})


def _cls_tool(line: str) -> str:
    st = line.strip()
    if not st:
        return "blank"
    if _SEP_RE.match(line):
        return "sep"
    if _ERR_START_RE.match(line):
        return "err"
    if _EXC_LINE_RE.match(line):
        return "exc"
    if _FRAME_RE.match(line):
        return "frame"
    if _TEST_LINE_RE.match(line):
        return "test"
    if (
        _CMD_PROMPT_RE.match(line)
        or _CMD_CALL_RE.match(line)
        or _CMD_VERB_RE.match(line)
    ):
        return "cmd"
    if _PATH_LINE_RE.match(line):
        return "path"
    if line[:1] in (" ", "\t"):
        return "ind"
    return "prose"


def _segment_tool(text: str, spans: list[tuple[int, int]]) -> list[dict]:
    """Line-structured segmentation for tool_call/tool_result/error/recovery.

    Error blocks absorb frames, connectors, and indented continuation lines;
    command/path/test lines stay atomic; other lines merge into prose runs.
    Backslash line-continuations keep a wrapped command in one candidate.
    """
    lines = [text[a:b] for a, b in spans]
    classes = [_cls_tool(l) for l in lines]
    out: list[dict] = []
    n = len(lines)
    i = 0
    while i < n:
        c = classes[i]
        if c in ("blank", "sep"):
            i += 1
            continue
        if c in ("err", "exc", "frame"):
            j = _run(classes, i + 1, _ERR_CONT)
            out.append({"hint": "error", "i": i, "j": j})
            i = j
            continue
        if c == "cmd":
            j = i + 1
            while (
                j < n
                and classes[j] in ("prose", "ind", "cmd")
                and lines[j - 1].rstrip().endswith("\\")
            ):
                j += 1
            out.append({"hint": "command", "i": i, "j": j})
            i = j
            continue
        if c == "test":
            # Consecutive result lines are one summary block; every line
            # stays verbatim and each node id lands in refs["tests"].
            j = _run(classes, i + 1, {"test"})
            out.append({"hint": "test", "i": i, "j": j})
            i = j
            continue
        if c == "path":
            # A path listing (e.g. ``ls`` output) is one candidate, not one
            # per entry — keeps output bounded and preserves all names.
            j = _run(classes, i + 1, {"path"})
            out.append({"hint": "path", "i": i, "j": j})
            i = j
            continue
        j = _run(classes, i + 1, _PROSE_CONT)
        out.append({"hint": "prose", "i": i, "j": j})
        i = j
    return out


# ---------------------------------------------------------------------------
# test_result / verification segmentation
# ---------------------------------------------------------------------------


def _cls_test(line: str) -> str:
    st = line.strip()
    if not st:
        return "blank"
    if _FAILHDR_RE.match(line):
        return "failhdr"
    if _SEP_RE.match(line):
        return "sep"
    if _TEST_LINE_RE.match(line):
        return "test"
    if _ERR_START_RE.match(line) or _EXC_LINE_RE.match(line):
        return "err"
    if _FRAME_RE.match(line):
        return "frame"
    if line[:1] in (" ", "\t"):
        return "ind"
    return "prose"


def _segment_test(text: str, spans: list[tuple[int, int]]) -> list[dict]:
    """test_result segmentation: test-name lines stay atomic, ``___ test ___``
    failure headers open blocks holding the assertion detail (``>``/``E``
    lines, ``file:line: Error``), banner separators are skipped, and summary
    lines land in prose runs."""
    lines = [text[a:b] for a, b in spans]
    classes = [_cls_test(l) for l in lines]
    out: list[dict] = []
    n = len(lines)
    i = 0
    while i < n:
        c = classes[i]
        if c in ("blank", "sep"):
            i += 1
            continue
        if c == "failhdr":
            j = i + 1
            while j < n and classes[j] not in ("failhdr", "sep", "test"):
                j += 1
            while j > i + 1 and classes[j - 1] == "blank":
                j -= 1
            out.append({"hint": "error", "i": i, "j": j})
            i = j
            continue
        if c in ("err", "frame"):
            j = _run(classes, i + 1, _ERR_CONT)
            out.append({"hint": "error", "i": i, "j": j})
            i = j
            continue
        if c == "test":
            j = _run(classes, i + 1, {"test"})
            out.append({"hint": "test", "i": i, "j": j})
            i = j
            continue
        j = _run(classes, i + 1, _PROSE_CONT)
        out.append({"hint": "prose", "i": i, "j": j})
        i = j
    return out


# ---------------------------------------------------------------------------
# file_diff segmentation
# ---------------------------------------------------------------------------


def _norm_diff_path(raw: str) -> Optional[str]:
    """Normalize a ``---``/``+++`` header argument or ``diff --git`` path to a
    repo-relative path; ``/dev/null`` and empties yield ``None``."""
    p = raw.split("\t")[0].strip().strip('"')
    if not p or p == "/dev/null":
        return None
    for pre in ("a/", "b/"):
        if p.startswith(pre):
            return p[2:]
    return p


def _segment_diff(text: str, spans: list[tuple[int, int]]) -> list[dict]:
    """One candidate per ``@@`` hunk, file-context line preserved.

    Sections split at ``diff --git`` lines (or ``--- `` + ``+++ `` pairs in
    plain unified diffs). Inside a hunk, ``--- x`` is a removed line unless
    the next line is a ``+++`` header. Header-only sections (mode changes,
    binary notices, renames) emit one prose candidate instead of dropping.
    """
    lines = [text[a:b] for a, b in spans]
    n = len(lines)
    out: list[dict] = []

    # Section boundaries: 'diff --git' lines; bare '--- '/ '+++ ' pairs open
    # sections only when no git headers exist at all.
    git_starts = [i for i, l in enumerate(lines) if l.startswith("diff --git ")]
    if git_starts:
        sections = [
            (s, git_starts[k + 1] if k + 1 < len(git_starts) else n)
            for k, s in enumerate(git_starts)
        ]
        if git_starts[0] > 0:
            sections.insert(0, (0, git_starts[0]))
    else:
        sections = [(0, n)]

    for s, e in sections:
        ctx_line: Optional[str] = None
        old_path: Optional[str] = None
        new_path: Optional[str] = None
        headers: list[tuple[int, int]] = []  # (i, j) line ranges
        # Each hunk snapshots the file context in force when it closes, so a
        # plain multi-file diff never attributes hunks to the wrong file.
        hunks: list[tuple[int, int, Optional[str], Optional[str]]] = []
        i = s
        while i < e:
            l = lines[i]
            if _HUNK_RE.match(l):
                j = i + 1
                while j < e:
                    l2 = lines[j]
                    if l2.startswith("@@ ") or l2.startswith("diff --git "):
                        break
                    if (
                        l2.startswith("--- ")
                        and j + 1 < e
                        and lines[j + 1].startswith("+++ ")
                    ):
                        break
                    j += 1
                # trim trailing blank lines out of the hunk
                while j > i + 1 and not lines[j - 1].strip():
                    j -= 1
                hunks.append((i, j, ctx_line, new_path or old_path))
                i = j
                continue
            if l.startswith("--- "):
                old_path = _norm_diff_path(l[4:])
                new_path = None  # a fresh --- header starts a new file pair
                if ctx_line is None:
                    ctx_line = l
                headers.append((i, i + 1))
            elif l.startswith("+++ "):
                new_path = _norm_diff_path(l[4:])
                ctx_line = l
                headers.append((i, i + 1))
            elif l.startswith("diff --git "):
                m = re.match(r"diff --git (\S+) (\S+)", l)
                if m:
                    old_path = old_path or _norm_diff_path(m.group(1))
                    new_path = new_path or _norm_diff_path(m.group(2))
                if ctx_line is None:
                    ctx_line = l
                headers.append((i, i + 1))
            else:
                headers.append((i, i + 1))
            i += 1

        file_path = new_path or old_path
        if hunks:
            for hi, hj, hctx, hfile in hunks:
                out.append(
                    {
                        "hint": "hunk",
                        "i": hi,
                        "j": hj,
                        "ctx": hctx,
                        "file": hfile,
                    }
                )
        else:
            # Header-only section: keep binary/mode/rename facts rather than
            # dropping them (V3-15.13 state facts). Pure banner noise with no
            # non-blank content emits nothing.
            content = [k for k, (a, b) in enumerate(headers) if lines[a].strip()]
            if content:
                out.append(
                    {
                        "hint": "prose",
                        "i": headers[content[0]][0],
                        "j": headers[content[-1]][1],
                        "file": file_path,
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Candidate emission
# ---------------------------------------------------------------------------


def _pack_lines(
    text: str,
    spans: list[tuple[int, int]],
    i: int,
    j: int,
    budget: int,
) -> list[tuple[int, int]]:
    """Pack line spans ``[i, j)`` into char ranges of at most ``budget`` chars
    (newline joins count). A single line over budget is split at whitespace;
    an unbreakable token is hard-cut at a character boundary — emitted text
    is always a verbatim substring, never dropped silently."""
    out: list[tuple[int, int]] = []
    cur_a: Optional[int] = None
    cur_b: Optional[int] = None
    for k in range(i, j):
        a, b = spans[k]
        if b - a > budget:
            if cur_a is not None:
                out.append((cur_a, cur_b))
                cur_a = cur_b = None
            pos = a
            while pos < b:
                raw_end = min(pos + budget, b)
                end = raw_end
                if end < b:
                    cut = end
                    while cut > pos and not text[cut - 1].isspace():
                        cut -= 1
                    if cut == pos:
                        cut = end  # unbreakable token: hard char-boundary cut
                    end = cut
                e2 = end
                while e2 > pos and text[e2 - 1].isspace():
                    e2 -= 1
                p2 = pos
                while p2 < e2 and text[p2].isspace():
                    p2 += 1
                if p2 < e2:
                    out.append((p2, e2))
                nxt = max(e2, p2)
                while nxt < b and text[nxt].isspace():
                    nxt += 1
                if nxt <= pos:
                    nxt = raw_end  # progress guard: never loop in place
                pos = nxt
            continue
        cost = (b - a) + (1 if cur_a is not None else 0)
        if cur_a is not None and (cur_b - cur_a) + cost > budget:
            out.append((cur_a, cur_b))
            cur_a = cur_b = None
        if cur_a is None:
            cur_a, cur_b = a, b
        else:
            cur_b = b
    if cur_a is not None:
        out.append((cur_a, cur_b))
    return out


def _mk(
    text: str,
    byte_off: list[int],
    hint: str,
    a: int,
    b: int,
    *,
    ctx: Optional[str] = None,
    file: Optional[str] = None,
) -> dict:
    piece = text[a:b]
    full = f"{ctx}\n{piece}" if ctx else piece
    d: dict[str, Any] = {
        "text": full,
        "kind_hint": hint,
        "start_byte": byte_off[a],
        "end_byte": byte_off[b],
        "refs": extract_refs(full),
    }
    if file:
        d["file"] = file
        if file not in d["refs"]["paths"]:
            d["refs"]["paths"].insert(0, file)
    return d


def _emit(
    text: str,
    spans: list[tuple[int, int]],
    byte_off: list[int],
    blocks: list[dict],
    max_candidates: int,
    max_len: int,
) -> list[dict]:
    out: list[dict] = []
    for blk in blocks:
        if len(out) >= max_candidates:
            break
        ctx = blk.get("ctx")
        ctx_cost = len(ctx) + 1 if ctx else 0
        budget = max_len - ctx_cost
        if budget < 8:
            ctx = None
            budget = max_len
        for a, b in _pack_lines(text, spans, blk["i"], blk["j"], budget):
            if len(out) >= max_candidates:
                break
            out.append(
                _mk(
                    text,
                    byte_off,
                    blk["hint"],
                    a,
                    b,
                    ctx=ctx,
                    file=blk.get("file"),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Prose path (v2 delegate) — V3-15.01/§15.02 parity
# ---------------------------------------------------------------------------


def _prose_candidates(
    text: str, max_candidates: int, min_len: int, max_len: int
) -> list[dict]:
    """Delegate to the v2 harvester so punctuation-light chat, CJK, list
    items, abbreviations, and acknowledgement filtering behave identically."""
    payload = text.encode("utf-8")
    res = _harvest_v2(
        payload,
        max_candidates=max_candidates,
        min_len=min_len,
        max_len=max_len,
    )
    out: list[dict] = []
    for c in res.candidates:
        piece = payload[c.start_byte : c.end_byte].decode("utf-8")
        out.append(
            {
                "text": piece,
                "kind_hint": "prose",
                "segment_kind": c.kind,
                "start_byte": c.start_byte,
                "end_byte": c.end_byte,
                "refs": extract_refs(piece),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _norm_kind(envelope_kind: Any) -> Optional[EnvelopeKind]:
    if envelope_kind is None:
        return None
    if isinstance(envelope_kind, EnvelopeKind):
        return envelope_kind
    try:
        return EnvelopeKind(str(envelope_kind))
    except ValueError:
        return None


def harvest_v3(
    text: Any,
    *,
    envelope_kind: Any = None,
    max_candidates: int = 32,
    min_len: int = 32,
    max_len: int = 1200,
) -> list[dict]:
    """Segment ``text`` into v3 harvest candidates for ``envelope_kind``.

    Returns a bounded list of dicts ``{"text", "kind_hint", "refs",
    "start_byte", "end_byte"}`` (plus ``file`` on diff candidates and
    ``segment_kind`` on v2-delegated prose). ``envelope_kind`` accepts an
    ``EnvelopeKind``, its string value, or ``None``; unrecognized kinds fall
    back to the v2 prose path. Pathological input returns ``[]`` rather than
    raising — harvest failure must never crash the ingest worker.

    Malformed UTF-8 bytes raise ``VerbatimError(VALIDATION)``: a decode
    failure is not "no candidates" — treating it as empty would let
    undecodable content read as clean (V4-13.11/13.12, F4-19).
    """
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "harvest_v3 input is not valid UTF-8 — decode failure "
                "cannot be reported as an empty candidate set",
            ) from exc
    if not isinstance(text, str) or not text.strip():
        return []
    if max_candidates < 1 or min_len < 1 or max_len < min_len:
        return []

    kind = _norm_kind(envelope_kind)
    if kind in _DIFF_KINDS or kind in _TEST_KINDS or kind in _TOOL_KINDS or kind in _ERROR_KINDS:
        spans = _line_spans(text)
        byte_off = _byte_offsets(text)
        if kind in _DIFF_KINDS:
            blocks = _segment_diff(text, spans)
        elif kind in _TEST_KINDS:
            blocks = _segment_test(text, spans)
        else:
            blocks = _segment_tool(text, spans)
        return _emit(text, spans, byte_off, blocks, max_candidates, max_len)
    return _prose_candidates(text, max_candidates, min_len, max_len)
