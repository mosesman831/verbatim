"""Deterministic Markdown rendering for file projections (SPEC_V4 §47).

This module is pure: it takes a collected ``ScopePlan`` and returns exact
output bytes. No database access, no clock, no randomness — identical
inputs render byte-identical files (V4-47.01's stable references make
the output diffable across generations).

File anatomy (one file per rendered scope — §47's "readable, editable
memory" surface; SQLite remains canonical authority):

    <!-- verbatim-projection
    {"format":1,"kind":"scope","scope_id":"sA", ...}
    -->
    # Scope "sA"
    ...
    ### claim "clA" — revision 1
    <!-- verbatim:entry {...} -->
    ...metadata lines...
    <!-- verbatim:quote {"sha256":"...","bytes":42} -->
    exact excerpt text
    <!-- /verbatim:quote -->

The HTML-comment markers carry the machine-readable provenance envelope
(stable object id, revision, view kind, digests, permitted edit
semantics) while the Markdown body stays human-readable. ``quote``
blocks are emitted only when the build authority held the ``quote``
verb — a read-only authority renders metadata + references and an
explicit withheld marker, never the excerpt bytes (V4-08.02, §47).

``content_sha256`` in the header hashes the file body *after* the header
comment (the header is the envelope; the body is the content), so a file
can be re-verified without self-referential hashing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from ..core.types import json_dumps

#: Renderer identity recorded in every manifest (V4-43.01).
RENDERER_NAME = "markdown-files"
RENDERER_VERSION = "1.0.0"

#: Projection file format tag + version (``manifest.json`` and headers).
FORMAT_TAG = "verbatim-projection"
FORMAT_VERSION = 1

#: Marker strings — parsed by ``edits.py``/``bundle.py`` on the way back
#: in. They live in HTML comments so the Markdown stays readable.
HEADER_MARK = "verbatim-projection"
ENTRY_MARK = "verbatim:entry"
QUOTE_MARK = "verbatim:quote"
QUOTE_CLOSE = "/verbatim:quote"
SOURCE_MARK = "verbatim:source"
WITHHELD_MARK = "verbatim:withheld"

#: Claim head states rendered into a projection. Matches the kernel's
#: visible-revision set (``_CLAIM_VISIBLE_STATES``): pending/rejected/
#: archived/erased heads are never published; superseded heads are not
#: current revisions and don't reach the head query at all.
RENDERABLE_STATES = ("active", "disputed")


@dataclass(frozen=True)
class EvidenceEntry:
    """One claim_evidence citation rendered as a stable reference."""

    span_id: str
    source_id: str
    source_revision: int
    role: str
    start_byte: int
    end_byte: int
    quote: Optional[str] = None          # exact excerpt text (quote mode)
    quote_sha256: Optional[str] = None   # sha256 over the excerpt utf-8 bytes

    def meta(self) -> dict[str, Any]:
        m: dict[str, Any] = {
            "span_id": self.span_id,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "role": self.role,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
        }
        if self.quote_sha256 is not None:
            m["quote_sha256"] = self.quote_sha256
        else:
            # Locator binding only — bytes were not released to this file.
            m["locator_sha256"] = hashlib.sha256(
                json_dumps(
                    {
                        "span_id": self.span_id,
                        "revision": self.source_revision,
                        "start": self.start_byte,
                        "end": self.end_byte,
                    }
                ).encode("utf-8")
            ).hexdigest()
        return m


@dataclass(frozen=True)
class ClaimEntry:
    """One claim head revision inside a scope document."""

    claim_id: str
    revision: int
    state: str
    recorded_from: int
    recorded_until: Optional[int]
    subject_id: Optional[str]
    predicate: Optional[str]
    obj: Any                            # parsed object_json (may be None)
    evidence: tuple[EvidenceEntry, ...] = ()

    def entry_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "kind": "claim",
            "claim_id": self.claim_id,
            "revision": self.revision,
            "state": self.state,
            "recorded_from": self.recorded_from,
            "recorded_until": self.recorded_until,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "evidence": [e.meta() for e in self.evidence],
            # V4-47.01: permitted edit semantics — a projection file is
            # never a mutation channel; edits become reviewed proposals.
            "edit": "proposal_only",
        }
        if self.obj is not None:
            # Binds the claim's structured content so an importer can
            # verify this projected row against a local copy.
            meta["object_sha256"] = hashlib.sha256(
                json_dumps(self.obj).encode("utf-8")
            ).hexdigest()
        return meta

    def entry_sha256(self) -> str:
        return hashlib.sha256(
            json_dumps(self.entry_meta()).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ScopePlan:
    """Everything needed to render one scope file — no I/O attached."""

    scope_id: str
    quote: bool                         # authority held ``quote``
    claims: tuple[ClaimEntry, ...]
    withheld: int                       # objects held out by suppression/quarantine
    inactive: int                       # heads not in a renderable state
    sources: tuple[dict[str, Any], ...]  # referenced source metadata
    inputs_sha256: str                  # digest over (id, rev) input set


def _jid(value: Any) -> str:
    """JSON-quoted id — unambiguous + deterministic for arbitrary ids."""
    return json.dumps(str(value), ensure_ascii=False)


def _header(meta: dict[str, Any]) -> str:
    return f"<!-- {HEADER_MARK}\n{json_dumps(meta)}\n-->\n"


def render_scope_document(
    plan: ScopePlan,
    *,
    store_id: Optional[str],
    generation: int,
) -> bytes:
    """Render one scope's Markdown file; byte-deterministic for the plan.

    The header binds the *eligibility digest* (``inputs_sha256`` — the
    (id, revision, verdict) set the bytes were rendered from), not the
    event watermark: two builds over identical content produce identical
    bytes even when unrelated store events moved ``event_seq``. The
    snapshot watermark lives in ``manifest.json``/``index.md`` (coverage
    bookkeeping, V4-43.07), not in evidence-carrying files.
    """
    body: list[str] = []
    body.append(f"# Scope {_jid(plan.scope_id)}\n")
    body.append(
        "\n_A verbatim file projection — a derived, reviewable view. The "
        "store is canonical; editing this file does not mutate evidence "
        "(propose edits through verbatim — see `edit` in entry markers)._\n"
    )
    if not plan.quote:
        body.append(
            "\n_Rendered under read authority only: exact excerpt bytes "
            "are withheld; entries carry references and digests._\n"
        )
    if plan.withheld:
        body.append(
            f"\n<!-- {WITHHELD_MARK} "
            f"{json_dumps({'count': plan.withheld})} -->\n"
            f"_{plan.withheld} object(s) withheld under suppression or "
            "quarantine._\n"
        )
    if plan.inactive:
        body.append(
            f"\n_{plan.inactive} head revision(s) not in a publishable "
            "state (pending/rejected/archived/erased) are omitted._\n"
        )

    body.append("\n## Claims\n")
    if not plan.claims:
        body.append("\n_(no publishable claims)_\n")
    for c in plan.claims:
        body.append(f"\n### claim {_jid(c.claim_id)} — revision {c.revision}\n\n")
        body.append(f"<!-- {ENTRY_MARK} {json_dumps(c.entry_meta())} -->\n")
        body.append(f"- **state**: {c.state}\n")
        body.append(
            f"- **recorded**: from seq {c.recorded_from}"
            + (
                f", until seq {c.recorded_until}\n"
                if c.recorded_until is not None
                else " (current)\n"
            )
        )
        if c.subject_id is not None:
            body.append(f"- **subject**: {_jid(c.subject_id)}\n")
        if c.predicate is not None:
            body.append(f"- **predicate**: {_jid(c.predicate)}\n")
        if c.obj is not None:
            body.append(
                f"- **object**: `{json_dumps(c.obj)}`\n"
            )
        body.append(f"- **entry sha256**: `{c.entry_sha256()}`\n")
        if c.evidence:
            body.append("\n#### evidence\n")
        for e in c.evidence:
            body.append(
                f"\n- `span` {_jid(e.span_id)} — source {_jid(e.source_id)}"
                f" revision {e.source_revision} (role: {e.role},"
                f" bytes {e.start_byte}–{e.end_byte})\n"
            )
            if e.quote_sha256 is not None:
                body.append(f"  - excerpt `sha256:{e.quote_sha256}`\n")
            if e.quote is not None:
                # Raw excerpt between explicit markers. The declared
                # digest binds the exact bytes; on import a mismatch is
                # reported, never silently trusted.
                digest = e.quote_sha256 or hashlib.sha256(
                    e.quote.encode("utf-8")
                ).hexdigest()
                nbytes = len(e.quote.encode("utf-8"))
                body.append(
                    f"<!-- {QUOTE_MARK} "
                    f"{json_dumps({'sha256': digest, 'bytes': nbytes})} -->\n"
                )
                # Exactly one joining LF follows the excerpt regardless
                # of whether it already ends with one — the importer
                # strips precisely that byte before re-hashing.
                body.append(e.quote)
                body.append("\n")
                body.append(f"<!-- {QUOTE_CLOSE} -->\n")
            elif e.quote_sha256 is None:
                body.append(
                    "  - _excerpt withheld — `quote` authority absent_\n"
                )

    body.append("\n## Sources\n")
    if not plan.sources:
        body.append("\n_(no cited sources)_\n")
    for s in plan.sources:
        smeta = {
            "kind": "source",
            "source_id": s["source_id"],
            "origin": s["origin"],
            "source_kind": s["source_kind"],
            "speaker_id": s["speaker_id"],
            "revisions": s["revisions"],
        }
        revs = ", ".join(
            f"r{r['revision']} "
            + (
                f"sha256:{r['payload_sha256'][:16]}…"
                if r.get("payload_sha256")
                else f"bytes:{r.get('byte_length', 0)}"
            )
            for r in s["revisions"]
        )
        body.append(
            f"\n<!-- {SOURCE_MARK} {json_dumps(smeta)} -->\n"
            f"- source {_jid(s['source_id'])} — origin {_jid(s['origin'])},"
            f" kind {s['source_kind']}, revisions: {revs}\n"
        )

    body_text = "".join(body)
    content_sha = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
    header = _header(
        {
            "format": FORMAT_VERSION,
            "kind": "scope",
            "scope_id": plan.scope_id,
            "store": store_id,
            "projection_generation": generation,
            "inputs_sha256": plan.inputs_sha256,
            "builder": f"{RENDERER_NAME}/{RENDERER_VERSION}",
            "quote": bool(plan.quote),
            "claims": len(plan.claims),
            "withheld": plan.withheld,
            "inactive": plan.inactive,
            "content_sha256": content_sha,
        }
    )
    return (header + "\n" + body_text).encode("utf-8")


def render_index(
    scopes: list[dict[str, Any]],
    *,
    store_id: Optional[str],
    generation: int,
    snapshot_seq: int,
) -> bytes:
    """Top-level index: one row per projected scope file."""
    body: list[str] = []
    body.append("# Verbatim projection\n")
    body.append(
        f"\n- store: {_jid(store_id)}\n- projection generation:"
        f" {generation}\n- snapshot seq: {snapshot_seq}\n"
        f"- builder: {RENDERER_NAME}/{RENDERER_VERSION}\n"
    )
    body.append(
        "\n| scope | file | claims | withheld | quote | sha256 |\n"
        "| --- | --- | ---: | ---: | --- | --- |\n"
    )
    for s in scopes:
        body.append(
            f"| {_jid(s['scope_id'])} | `{s['file']}` | {s['claims']}"
            f" | {s['withheld']} | {s['quote']} | `{s['sha256'][:16]}…` |\n"
        )
    body_text = "".join(body)
    header = _header(
        {
            "format": FORMAT_VERSION,
            "kind": "index",
            "store": store_id,
            "projection_generation": generation,
            "snapshot_seq": snapshot_seq,
            "builder": f"{RENDERER_NAME}/{RENDERER_VERSION}",
            "scopes": [s["scope_id"] for s in scopes],
            "content_sha256": hashlib.sha256(
                body_text.encode("utf-8")
            ).hexdigest(),
        }
    )
    return (header + "\n" + body_text).encode("utf-8")


def scope_filename(scope_id: str) -> str:
    """Filesystem-safe, collision-free, deterministic scope file name.

    The readable prefix is best-effort (sanitized, dot-runs collapsed);
    the 12-hex sha256 suffix guarantees uniqueness regardless of how
    hostile the scope id is — ``"../x"`` and ``"x"`` never collide.
    """
    safe = "".join(
        ch if ch.isalnum() or ch in "_-" else "-" for ch in scope_id
    ).strip("-.")
    if len(safe) > 40:
        safe = safe[:40]
    if not safe:
        safe = "scope"
    digest = hashlib.sha256(scope_id.encode("utf-8")).hexdigest()[:12]
    return f"scope-{safe}-{digest}.md"


def parse_header(text: str) -> Optional[dict[str, Any]]:
    """Extract the ``verbatim-projection`` header envelope, if present."""
    prefix = f"<!-- {HEADER_MARK}"
    if not text.startswith(prefix):
        return None
    end = text.find("\n-->", len(prefix))
    if end < 0:
        return None
    try:
        meta = json.loads(text[len(prefix):end])
    except (ValueError, TypeError):
        return None
    return meta if isinstance(meta, dict) else None


#: ``### claim "id" — revision N`` — the block boundary this format owns.
#: The id is a JSON string (``_jid``); the pattern is end-anchored so an
#: id containing `` — revision `` can never misparse.
_CLAIM_HEADING = re.compile(
    r'^### claim ("(?:[^"\\]|\\.)*") — revision (\d+)\s*$'
)


def iter_claim_blocks(
    text: str,
) -> Iterator[tuple[Optional[str], Optional[dict[str, Any]], int, int]]:
    """Yield ``(claim_id, entry_meta, start, end)`` per claim block.

    ``start``/``end`` are character offsets: a block runs from its
    ``### claim`` heading to the next claim heading, ``## Sources``, or
    end of text. ``entry_meta`` is the parsed ``verbatim:entry`` marker
    (``None`` when absent or malformed — the heading alone still bounds
    the block).
    """
    lines = text.split("\n")
    offs: list[int] = []
    pos = 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    heads: list[int] = []
    for i, ln in enumerate(lines):
        if _CLAIM_HEADING.match(ln):
            heads.append(i)
    # The Sources section follows the last claim. A quote excerpt can
    # itself contain a "## Sources" line, so only the first occurrence
    # *after the final claim heading* counts as the section boundary —
    # anything earlier is block content, not structure.
    sources_line = len(lines)
    if heads:
        for n in range(heads[-1] + 1, len(lines)):
            if lines[n] == "## Sources":
                sources_line = n
                break
    for k, i in enumerate(heads):
        # block ends at the next claim heading, the Sources section, or EOF
        j = heads[k + 1] if k + 1 < len(heads) else len(lines)
        if sources_line < j:
            j = sources_line
        m = _CLAIM_HEADING.match(lines[i])
        cid: Optional[str] = None
        if m is not None:
            try:
                parsed = json.loads(m.group(1))
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, str):
                cid = parsed
        meta: Optional[dict[str, Any]] = None
        block_text = "\n".join(lines[i:j])
        em = re.search(
            re.escape(f"<!-- {ENTRY_MARK} ") + r"(\{.*?\}) -->",
            block_text,
            flags=re.DOTALL,
        )
        if em is not None:
            try:
                parsed_meta = json.loads(em.group(1))
            except (ValueError, TypeError):
                parsed_meta = None
            if isinstance(parsed_meta, dict):
                meta = parsed_meta
        start = offs[i]
        end = offs[j] if j < len(lines) else len(text)
        yield cid, meta, start, end


def excise_claims(
    text: str,
    *,
    claim_ids: frozenset = frozenset(),
    span_ids: frozenset = frozenset(),
    source_ids: frozenset = frozenset(),
) -> tuple[Optional[str], list[str]]:
    """Surgically remove rendered claim blocks (V4-43.09, active gens).

    A claim is *attributable* when its own id is suppressed or its entry
    marker cites a suppressed span/source — the whole block is replaced
    by a ``verbatim:withheld`` marker, never silently dropped. The file
    header is re-emitted with corrected counts and a fresh
    ``content_sha256`` so the document stays self-consistent.

    Returns ``(new_text, removed_claim_ids)``; ``(None, [])`` when the
    input is not a scope document or nothing in it is attributable.
    """
    meta = parse_header(text)
    if meta is None or meta.get("kind") != "scope":
        return None, []
    hdr_end = text.find("\n-->")
    if hdr_end < 0 or text[hdr_end + 4:hdr_end + 6] != "\n\n":
        return None, []
    body_start = hdr_end + 6
    drops: list[tuple[int, int, str]] = []
    removed: list[str] = []
    quote_shas: set[str] = set()
    for cid, entry_meta, start, end in iter_claim_blocks(text):
        if start < body_start:
            continue
        hit = cid is not None and cid in claim_ids
        if not hit:
            if entry_meta is None:
                # Unparseable marker: we cannot prove this block does
                # not cite a suppressed span/source — fail closed.
                hit = bool(span_ids or source_ids)
            else:
                for ev in entry_meta.get("evidence") or []:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("span_id") in span_ids or (
                        ev.get("source_id") in source_ids
                    ):
                        hit = True
                        break
        if not hit or cid is None:
            continue
        drops.append((start, end, cid))
        removed.append(cid)
        if isinstance(entry_meta, dict):
            for ev in entry_meta.get("evidence") or []:
                if isinstance(ev, dict) and ev.get("quote_sha256"):
                    quote_shas.add(str(ev["quote_sha256"]))
    if not drops:
        return None, []
    body = text
    for start, end, cid in reversed(drops):
        repl = (
            f"<!-- {WITHHELD_MARK} "
            + json_dumps({"claims": [cid], "reason": "suppressed"})
            + f" -->\n_claim {_jid(cid)} suppressed by deletion policy._\n\n"
        )
        body = body[:start] + repl + body[end:]
    new_body = body[body_start:]
    # Post-check: no removed claim block may survive, and no live
    # ``verbatim:quote`` block may still declare one of the removed
    # claims' excerpt digests (a forged "## Sources"/"### claim" line
    # inside excerpt bytes could have truncated a block mid-quote).
    # A *source* marker's payload digest is a reference, not bytes —
    # it is allowed to remain.
    residual = any(
        cid in removed
        for cid, _m, _s, _e in iter_claim_blocks(new_body)
    )
    if not residual:
        for m in re.finditer(
            re.escape(f"<!-- {QUOTE_MARK} ") + r"(\{.*?\}) -->",
            new_body,
            flags=re.DOTALL,
        ):
            try:
                qmeta = json.loads(m.group(1))
            except (ValueError, TypeError):
                continue
            if (
                isinstance(qmeta, dict)
                and qmeta.get("sha256") in quote_shas
            ):
                residual = True
                break
    if residual:
        # Over-remove rather than leak: the whole scope document
        # collapses to a suppression tombstone (rebuild restores the
        # surviving claims).
        scope_id = str(meta.get("scope_id") or "?")
        new_body = (
            f"# Scope {_jid(scope_id)}\n\n<!-- {WITHHELD_MARK} "
            + json_dumps({"claims": sorted(removed), "reason": "suppressed"})
            + " -->\n_this scope projection was suppressed in place; "
            "rebuild to restore the surviving claims._\n"
        )
        meta["claims"] = 0
        meta["withheld"] = int(meta.get("withheld") or 0) + len(removed)
    else:
        meta["claims"] = max(0, int(meta.get("claims") or 0) - len(removed))
        meta["withheld"] = int(meta.get("withheld") or 0) + len(removed)
    meta["content_sha256"] = hashlib.sha256(
        new_body.encode("utf-8")
    ).hexdigest()
    new_text = (
        f"<!-- {HEADER_MARK}\n{json_dumps(meta)}\n-->\n\n" + new_body
    )
    return new_text, removed


def iter_quote_blocks(text: str):
    """Yield (declared_sha256, declared_bytes, quote_text) for each
    ``verbatim:quote`` block in a rendered file.

    Framing is length-prefixed, not marker-scanned: the declared
    ``bytes`` count says exactly how much utf-8 the excerpt occupies —
    an excerpt containing the close marker text still parses correctly,
    and a truncated/forged block yields a digest mismatch downstream.
    """
    open_tag = f"<!-- {QUOTE_MARK} "
    close_tag = f"<!-- {QUOTE_CLOSE} -->"
    pos = 0
    while True:
        start = text.find(open_tag, pos)
        if start < 0:
            return
        hdr_end = text.find(" -->", start)
        if hdr_end < 0:
            return
        try:
            meta = json.loads(text[start + len(open_tag):hdr_end])
        except (ValueError, TypeError):
            meta = {}
        if not isinstance(meta, dict):
            return
        nbytes = meta.get("bytes")
        declared_sha = meta.get("sha256")
        body_start = hdr_end + len(" -->")
        if text[body_start:body_start + 1] == "\n":
            body_start += 1
        if not isinstance(nbytes, int) or nbytes < 0:
            return
        # Slice exactly nbytes of utf-8; the next bytes must be the
        # joining LF + close marker or the block is malformed.
        raw = text.encode("utf-8")
        byte_start = len(text[:body_start].encode("utf-8"))
        quote_bytes = raw[byte_start:byte_start + nbytes]
        after = raw[byte_start + nbytes:]
        trailer = ("\n" + close_tag).encode("utf-8")
        if not after.startswith(trailer):
            return  # malformed block — importer treats as tamper
        try:
            quote = quote_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return
        yield (declared_sha, nbytes, quote)
        pos = body_start + len(quote) + 1


__all__ = [
    "ClaimEntry",
    "ENTRY_MARK",
    "EvidenceEntry",
    "FORMAT_TAG",
    "FORMAT_VERSION",
    "HEADER_MARK",
    "QUOTE_CLOSE",
    "QUOTE_MARK",
    "RENDERABLE_STATES",
    "RENDERER_NAME",
    "RENDERER_VERSION",
    "SOURCE_MARK",
    "ScopePlan",
    "WITHHELD_MARK",
    "excise_claims",
    "iter_claim_blocks",
    "iter_quote_blocks",
    "parse_header",
    "render_index",
    "render_scope_document",
    "scope_filename",
]
