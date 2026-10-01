"""Retirement-signal supersession proposals (§17 derivations; G3 bounds
the false-supersession rate this detector is measured against).

When a claim's verbatim evidence explicitly declares an identifier —
command, flag, path, endpoint — retired/deprecated/removed, and another
active claim in the same scope instructs or asserts that same identifier
over a shared topic, the pair is a *supersession candidate*. Like every
relation proposal in the system (SPEC_V2 §19), it lands on the operator
review queue with pinned expected versions — detection never applies a
transition itself.

Precision is the design constraint (G3): the retirement marker must sit
in the asserting claim's own text adjacent to the identifier, and the
counterparty must share topic vocabulary beyond the identifier. A bare
mention of an identifier never supersedes.
"""

from __future__ import annotations

import hmac
import re
import sqlite3
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError
from ..storage.repos import EdgesRepo, ReviewsRepo


#: Identifier-ish token: commands, flags, paths, endpoints, versions.
_IDENT = r"[A-Za-z0-9][A-Za-z0-9_./-]{1,60}"

#: Words that look identifier-shaped but carry no referent — excluded from
#: both subject and object capture to keep transitive forms honest
#: ("removed the flag" never names "the"; "retired in June" never names
#: "in").
_NON_IDENT = frozenset({
    "the", "a", "an", "it", "this", "that", "our", "your", "its", "his",
    "her", "their", "old", "new", "all", "any", "some", "no", "now",
    "was", "is", "has", "been", "were", "got", "became", "be", "are",
    "in", "on", "at", "of", "for", "to", "and", "or", "by", "with",
    "from", "as", "into", "we", "you", "they", "i", "use", "used",
    "using", "run", "make", "do", "did", "does", "not",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
})

#: Retirement verbs — subject form "X was retired" and object form
#: "we retired X" both assert X's end-of-life.
_RETIRE_VERBS = (
    "retired", "deprecated", "removed", "replaced", "superseded",
    "decommissioned", "obsoleted", "sunset", "sunsetted", "eol",
)

_SUBJECT_RE = re.compile(
    rf"(?P<ident>{_IDENT})[`'\")]?\s+"
    rf"(?P<aux>was|is|has been|were|became|got)\s+"
    rf"(?:{'|'.join(_RETIRE_VERBS)})\b",
    re.IGNORECASE,
)
_OBJECT_RE = re.compile(
    rf"(?:retire|retired|deprecate|deprecated|remove|removed|replace|"
    rf"replaced|supersede|superseded|decommission|decommissioned|"
    rf"obsoleted?|sunset(?:ted)?)\s+[`'\"]?"
    rf"(?:the\s+|our\s+|old\s+)?[`'\"]?(?P<ident>{_IDENT})\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "to", "in", "on", "at", "of", "for",
    "with", "by", "is", "was", "are", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "from", "as", "into", "use", "used",
    "using", "run", "make", "old", "new", "we", "you", "they", "i",
    "june", "january", "february", "march", "april", "may", "july",
    "august", "september", "october", "november", "december",
})

_PAIR_BUDGET = 8

#: Hedging adverbs — these hedge wherever they appear in the clause
#: ("maybe X was retired", "X was reportedly removed").
_HEDGE_ADV_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|reportedly|rumou?red(?:ly)?"
    r"|alleged(?:ly)?|apparently|supposedly|seemingly|arguably|unsure"
    r"|uncertain)\b",
    re.IGNORECASE,
)

#: Modal verbs/complementizers — hedge only when followed by an explicit
#: complementizer ("check whether X was retired", "confirm that X was
#: removed"). A bare verb never hedges: ``ruff check`` is a command, not
#: a hedge. ``whether`` alone always hedges (it only ever introduces an
#: indirect question); ``if``/``that``/``about`` need the verb.
_MODAL_VERB_RE = re.compile(
    r"\b(?:wonder(?:ing|ed)?|check(?:ing|ed)?|confirm(?:ing|ed)?"
    r"|verif(?:y|ying|ied)|ask(?:ing|ed)?|discuss(?:ing|ed)?"
    r"|question(?:ing|ed)?|consider(?:ing|ed)?|unsure|curious)\b",
    re.IGNORECASE,
)
_COMPLEMENT_RE = re.compile(r"\b(?:whether|if|that|about)\b", re.IGNORECASE)

#: Conditional subordinators — "if X is retired" conditions on the event;
#: it never asserts it. ``when`` is handled separately: "when X is
#: deprecated" is hypothetical, but "when X was deprecated" asserts a
#: past event, so it hedges only with a present-tense auxiliary.
_SUBORDINATOR_RE = re.compile(
    r"\b(?:if|unless|whenever|in case)\b", re.IGNORECASE
)
_WHEN_RE = re.compile(r"\bwhen\b", re.IGNORECASE)
_PRESENT_AUX = frozenset({"is", "are", "gets"})

#: Hearsay attribution — "rumor says X was removed" reports a claim; it
#: doesn't assert one (the adverbial forms live in _HEDGE_ADV_RE).
_HEARSAY_RE = re.compile(
    r"\b(?:rumou?rs?\s+(?:says|has it)|word is|they say|i heard|"
    r"people say|report has it)\b",
    re.IGNORECASE,
)

#: Intent markers on the object verb — an infinitive, modal, or request
#: prefix expresses intent, not a completed retirement: "to deprecate X",
#: "should remove X", "please retire X", "let's replace X". Anchored at
#: the window's end so the marker must lead the matched verb (a couple of
#: adverbs may intervene: "should probably remove").
_INTENT_RE = re.compile(
    r"(?:\bto|\bwill\b|\bwould\b|\bshall\b|\bshould\b|\bmay\b|\bmight\b"
    r"|\bcould\b|\bmust\b|\bcan\b|\bplease\b|\blet'?s\b|\blets\b)"
    r"\s+(?:[a-z]+\s+){0,2}$",
    re.IGNORECASE,
)

#: Cancellation markers AFTER the match — "X was retired ... but
#: reinstated" revokes the retirement inside the same sentence.
_CANCEL_RE = re.compile(
    r"\b(?:reinstat\w*|restor\w*|brought back|resurrect\w*|unretir\w*|"
    r"un-?deprecat\w*|back in service|rolled back|re-?enabled)\b",
    re.IGNORECASE,
)

#: Successor endorsement — "X was retired; X-ng is the successor"
#: affirms a counterparty already using the successor ident.
_ENDORSE_RE = re.compile(
    r"\b(?:successor|replacement|handles that|in place of"
    r"|switch(?:ed)?\s+to|migrat\w+\s+to|instead)\b",
    re.IGNORECASE,
)

#: Sentence boundary — a modal marker only hedges the clause it shares.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?;\n:]")


def _hedged(text: str, start: int, aux: Optional[str] = None) -> bool:
    """True when the match sits in a hedged clause — the last 48 chars
    before ``start``, truncated at the most recent sentence boundary.
    ``aux`` is the subject-form auxiliary when known: ``when`` hedges
    only hypothetically (present tense), never a past assertion.
    Known bound: a bare modal verb without a complementizer ("please
    verify X was removed") is treated as an assertion."""
    window = text[max(0, start - 48):start]
    last = None
    for m in _SENTENCE_BOUNDARY_RE.finditer(window):
        last = m
    if last is not None:
        window = window[last.end():]
    if _HEDGE_ADV_RE.search(window):
        return True
    if _HEARSAY_RE.search(window):
        return True
    if re.search(r"\bwhether\b", window, re.IGNORECASE):
        return True
    if _SUBORDINATOR_RE.search(window):
        return True
    if (
        aux is not None
        and aux.lower() in _PRESENT_AUX
        and _WHEN_RE.search(window)
    ):
        return True
    if _INTENT_RE.search(window):
        return True
    for m in _MODAL_VERB_RE.finditer(window):
        if _COMPLEMENT_RE.search(window, m.end()):
            return True
    return False


def _cancelled(text: str, end: int) -> bool:
    """True when the retirement is revoked later in the same sentence —
    "X was retired ... but reinstated" asserts continuity, not an end."""
    seg = text[end:end + 96]
    boundary = _SENTENCE_BOUNDARY_RE.search(seg)
    if boundary is not None:
        seg = seg[:boundary.start()]
    return bool(_CANCEL_RE.search(seg))


def _retired_idents(text: str) -> list[str]:
    """Identifiers the text explicitly declares end-of-life.

    Both directions: ``swagger-gen was retired`` (subject) and ``we
    retired swagger-gen`` (object). Backticks stripped; function words and
    auxiliaries can never capture; hedged, conditional, intended,
    hearsay, or cancelled mentions ("whether X was retired", "if X is
    removed", "please retire X", "X was retired but reinstated") are not
    assertions.
    """
    out: list[str] = []
    for m in _SUBJECT_RE.finditer(text):
        ident = m.group("ident").strip("`'\"").lower()
        if (
            ident
            and ident not in _NON_IDENT
            and not _hedged(text, m.start(), m.group("aux"))
            and not _cancelled(text, m.end())
        ):
            out.append(ident)
    for m in _OBJECT_RE.finditer(text):
        ident = m.group("ident").strip("`'\"").lower()
        if (
            ident
            and ident not in _NON_IDENT
            and not _hedged(text, m.start())
            and not _cancelled(text, m.end())
        ):
            out.append(ident)
    return list(dict.fromkeys(out))


def _content_tokens(text: str) -> frozenset:
    return frozenset(
        t for t in re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())
        if t not in _STOPWORDS and len(t) > 1
    )


def _mentions_ident(text: str, ident: str) -> bool:
    """Word-boundary identifier mention — backticked, bare, or flag-form.

    Leading dashes are absorbed (``force`` matches ``--force``), but a
    word character before the dashes stays a non-match (``x-force``), and
    identifier continuation on the right (``swagger-gen2``) is rejected —
    all enforced procedurally because fixed-width lookbehind cannot walk
    back over optional dashes.
    """
    for m in re.finditer(re.escape(ident), text, re.IGNORECASE):
        end = m.end()
        if end < len(text):
            nxt = text[end]
            # word char or dash continues the identifier; a dot does only
            # when more word chars follow (``x.foo``) — a bare period is
            # sentence punctuation, not a boundary violation.
            if re.match(r"[\w-]", nxt) or (
                nxt == "." and end + 1 < len(text)
                and re.match(r"\w", text[end + 1])
            ):
                continue
        start = m.start()
        while start > 0 and text[start - 1] == "-":
            start -= 1
        if start > 0 and re.match(r"[\w.]", text[start - 1]):
            continue
        return True
    return False


def _claim_text(
    store: Any, conn: sqlite3.Connection, claim_id: str, revision: int
) -> Optional[str]:
    """Exact quotation reconstructed from span + source bytes (v2 form).

    Each slice is re-verified against ``spans.excerpt_hmac`` before use —
    the same consumption-time check ``SpansRepo.text`` performs, so a
    tampered excerpt cannot silently feed supersession detection. A NULL
    digest marks a legacy row and is skipped; an emptied (purged) payload
    contributes nothing.
    """
    rows = conn.execute(
        "SELECT s.start_byte, s.end_byte, sr.payload, s.excerpt_hmac"
        " FROM claim_evidence ce"
        " JOIN spans s ON s.span_id = ce.span_id"
        " JOIN source_revisions sr"
        "   ON sr.source_id = s.source_id AND sr.revision = s.revision"
        " WHERE ce.claim_id = ? AND ce.revision = ? AND ce.evidence_role = 'primary'"
        " ORDER BY s.start_byte",
        (claim_id, revision),
    ).fetchall()
    parts = []
    for start, end, payload, excerpt_hmac in rows:
        payload = bytes(payload)
        if not payload:
            continue  # purged revision: the excerpt bytes no longer exist
        excerpt = payload[start:end]
        if excerpt_hmac is not None and not hmac.compare_digest(
            store.hmac(excerpt), bytes(excerpt_hmac)
        ):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"claim {claim_id} evidence fails integrity check",
            )
        try:
            parts.append(excerpt.decode("utf-8"))
        except UnicodeDecodeError:
            return None
    return " ".join(parts) if parts else None


def propose_retirement_supersessions(
    store: Any, claim_id: str, scope_id: str
) -> list[str]:
    """Propose supersessions for claims retired by ``claim_id``'s evidence.

    Scans the new claim's verbatim text for explicit retirement markers,
    then — for each retired identifier — binds counterparty claims that
    (a) mention the identifier, (b) share topic vocabulary with the new
    claim, and (c) do not themselves assert the same retirement. Each pair
    produces a ``conflicts_with`` edge plus an open ``supersede`` review
    carrying pinned expected versions (SPEC_V2 §19.07 fence). Returns the
    created edge/review ids; ``[]`` when no marker exists — the common
    case costs one text read and no writes.
    """
    with store.read() as conn:
        head = conn.execute(
            "SELECT r.claim_id, r.revision FROM claims c"
            " JOIN claim_revisions r ON r.claim_id = c.claim_id"
            " WHERE c.claim_id = ? AND r.recorded_until IS NULL"
            " ORDER BY r.revision DESC LIMIT 1",
            (claim_id,),
        ).fetchone()
        if head is None:
            return []
        new_text = _claim_text(store, conn, head[0], head[1])
        if not new_text:
            return []
        idents = _retired_idents(new_text)
        if not idents:
            return []
        new_topics = _content_tokens(new_text)
        cands = conn.execute(
            "SELECT c.claim_id, r.revision FROM claims c"
            " JOIN claim_revisions r ON r.claim_id = c.claim_id"
            "   AND r.recorded_until IS NULL"
            " WHERE c.scope_id = ? AND c.claim_id <> ?"
            "   AND r.state IN ('active','disputed','pending')"
            " LIMIT ?",
            (scope_id, claim_id, _PAIR_BUDGET * 4),
        ).fetchall()
        texts = {cid: _claim_text(store, conn, cid, rev) for cid, rev in cands}

    pairs: list[tuple[str, int, str]] = []
    for cid, rev in cands:
        text = texts.get(cid)
        if not text or len(pairs) >= _PAIR_BUDGET:
            continue
        for ident in idents:
            if not _mentions_ident(text, ident):
                continue
            # A counterparty that itself asserts the same retirement is
            # corroborating, not contradicted.
            if ident in _retired_idents(text):
                continue
            # A counterparty already on a successor identifier
            # ("force-ng" while "force" was retired) is affirmed by a
            # retirement+endorsement text, not contradicted by it.
            if (
                re.search(
                    re.escape(ident) + r"[-./][a-z0-9]", text, re.IGNORECASE
                )
                and _ENDORSE_RE.search(new_text)
            ):
                continue
            shared = (new_topics - {ident}) & (_content_tokens(text) - {ident})
            if not shared:
                continue
            pairs.append((cid, rev, ident))
            break

    if not pairs:
        return []

    out: list[str] = []
    with store.tx() as conn:
        edges = EdgesRepo(store)
        reviews = ReviewsRepo(store)
        for cid, rev, ident in pairs:
            a, b = sorted((claim_id, cid))
            edge_id = edges.add(
                conn, scope_id, "claim", a, "claim", b, "conflicts_with"
            )
            out.append(edge_id)
            out.append(
                reviews.create(
                    conn,
                    scope_id,
                    {
                        "effect": "supersede",
                        "predecessor_id": cid,
                        "successor_id": claim_id,
                        "pair_label": "incompatible",
                        "change_signal": "retirement",
                        "reason": (
                            f"successor's evidence declares '{ident}' retired"
                        ),
                    },
                    {cid: rev, claim_id: head[1]},
                    dedup=True,
                )
            )
    return out
