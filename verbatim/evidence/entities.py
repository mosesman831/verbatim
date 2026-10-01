"""Entity and reference resolution (SPEC_V4 §17, V4-17.01–17.08).

Entity ids are scope-bound (V4-17.01): every lookup intersects the
authorized scope set and never crosses a partition boundary. Resolution
follows the precedence the spec requires — exact identifiers first, then
exact normalized labels/aliases — and *never* collapses distinct entities
on name similarity alone (V4-17.02): when a name maps to several entities
the result is ``ambiguous`` with every competing candidate recorded, not
a silently-chosen winner (scenario C38).

Pronouns and deictic references (V4-17.03) resolve ONLY inside a declared
context window — the entity ids the caller supplies plus the entities
linked to the caller's context claims. With no declared window, or no
unique candidate inside it, the reference stays ``unresolved`` or
``ambiguous``; it is never guessed.

User, assistant, quoted third parties, hypothetical subjects, and
reported speakers remain distinct objects (V4-17.04): this module only
ever links entity ids the caller explicitly created/linked (``entities``,
``entity_aliases``, ``claim_entities``); it never fuses identities, and
candidates keep their declared ``kind`` and ``claim_entities`` roles.

Original-language text is never rewritten (V4-17.05): normalization here
is a matching projection only (NFKD + casefold); stored bytes are
untouched and translations, where present, stay separate objects with
their own provenance.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Optional

STATUS_RESOLVED = "resolved"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNRESOLVED = "unresolved"

# Referential expressions that only resolve inside a declared context
# window (V4-17.03). English-centric but explicit — a token outside these
# sets is resolved by identifier/alias evidence, never by guessing.
PRONOUNS = frozenset({
    "he", "she", "they", "it", "him", "her", "them", "his", "hers",
    "their", "theirs", "its", "i", "me", "we", "us", "you",
    "someone", "somebody", "anyone", "anybody",
})
DEICTICS = frozenset({
    "this", "that", "these", "those", "here", "there",
    "the former", "the latter",
})

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_][^\W_']*", re.UNICODE)


def normalize_entity_text(text: str) -> str:
    """Canonical matching projection for labels/aliases: NFKD, combining
    marks dropped, casefolded, whitespace collapsed. Matching only — the
    stored label is never rewritten (V4-17.05)."""
    nfkd = unicodedata.normalize("NFKD", str(text or ""))
    stripped = "".join(
        c for c in nfkd if not unicodedata.combining(c)
    )
    return _WS_RE.sub(" ", stripped).casefold().strip()


def _ph(n: int) -> str:
    return ",".join("?" * n)


@dataclass(frozen=True)
class EntityCandidate:
    """One competing entity for a name/reference — ambiguity is recorded,
    never collapsed (V4-17.02/17.03)."""

    entity_id: str
    label: str
    kind: Optional[str]
    via: str            # entity_id | label | alias | label+alias | context
    confidence: float   # grounded confidence, never a similarity guess
    roles: tuple = ()   # claim_entities roles observed in the window
    scope_id: str = ""


@dataclass(frozen=True)
class EntityResolution:
    """Outcome of resolving one name to an entity (V4-17.02)."""

    query: str
    status: str                     # resolved | ambiguous | unresolved
    entity_id: Optional[str]        # set ONLY when status == resolved
    candidates: tuple = ()          # competing EntityCandidate records
    reason: str = ""


@dataclass(frozen=True)
class ReferenceResolution:
    """Outcome of resolving one referential token (V4-17.03)."""

    reference: str
    status: str                     # resolved | ambiguous | unresolved
    entity_id: Optional[str]
    candidates: tuple = ()
    confidence: float = 0.0
    reason: str = ""


def _entity_rows(conn: sqlite3.Connection, scope_ids, where: str,
                 params) -> dict:
    """Authorized-scope entity rows keyed by entity_id (V4-17.01)."""
    ids = list(dict.fromkeys(scope_ids or ()))
    if not ids:
        return {}
    rows = conn.execute(
        "SELECT entity_id, scope_id, kind, label FROM entities"
        f" WHERE scope_id IN ({_ph(len(ids))}) AND {where}",
        [*ids, *params],
    ).fetchall()
    return {str(r[0]): r for r in rows}


def _roles_for(conn: sqlite3.Connection, entity_ids, claim_ids) -> dict:
    """claim_entities roles each entity holds inside the context window."""
    if not entity_ids or not claim_ids:
        return {}
    eph, cph = _ph(len(entity_ids)), _ph(len(claim_ids))
    rows = conn.execute(
        "SELECT entity_id, role FROM claim_entities"
        f" WHERE entity_id IN ({eph}) AND claim_id IN ({cph})",
        [*entity_ids, *claim_ids],
    ).fetchall()
    out: dict = {}
    for eid, role in rows:
        out.setdefault(str(eid), set()).add(str(role))
    return {k: tuple(sorted(v)) for k, v in out.items()}


def _context_entity_ids(conn: sqlite3.Connection, scope_ids,
                        context_claim_ids, context_entity_ids) -> dict:
    """Entities inside the declared context window — caller-declared ids
    plus entities linked to the window's claims — each mapped to the
    claim_entities roles seen inside the window (V4-17.03)."""
    ids = list(dict.fromkeys(scope_ids or ()))
    if not ids:
        return {}
    out: dict[str, set] = {}
    for eid in context_entity_ids or ():
        out.setdefault(str(eid), set())
    claims = [str(c) for c in (context_claim_ids or ())]
    if claims:
        rows = conn.execute(
            "SELECT ce.entity_id, ce.role FROM claim_entities ce"
            " JOIN entities e ON e.entity_id = ce.entity_id"
            f"  AND e.scope_id IN ({_ph(len(ids))})"
            " JOIN claims c ON c.claim_id = ce.claim_id"
            f"  AND c.scope_id IN ({_ph(len(ids))})"
            f" WHERE ce.claim_id IN ({_ph(len(claims))})",
            [*ids, *ids, *claims],
        ).fetchall()
        for eid, role in rows:
            out.setdefault(str(eid), set()).add(str(role))
    # Only entities that exist inside the authorized scopes count.
    if out:
        live = _entity_rows(conn, ids,
                            f"entity_id IN ({_ph(len(out))})",
                            list(out))
        out = {eid: out[eid] for eid in live}
    return out


def resolve_entity(conn: sqlite3.Connection, scope_ids, name: str,
                   *, kind: Optional[str] = None,
                   context_entity_ids=(), context_claim_ids=(),
                   limit: int = 16) -> EntityResolution:
    """Resolve ``name`` to a scoped entity — or to explicit ambiguity.

    Precedence (V4-17.02): an exact ``entity_id`` match wins outright;
    otherwise exact normalized label/alias matches compete. A ``kind``
    filter and a declared context window are *evidence* — each may narrow
    the candidate set — but two surviving candidates are reported
    ``ambiguous``, never merged by similarity.
    """
    name = str(name or "").strip()
    ids = list(dict.fromkeys(scope_ids or ()))
    if not name:
        return EntityResolution(name, STATUS_UNRESOLVED, None, (),
                                "empty_reference")
    if not ids:
        return EntityResolution(name, STATUS_UNRESOLVED, None, (),
                                "no_authorized_scope")

    # 1) exact identifier — highest precedence (V4-17.02).
    hit = _entity_rows(conn, ids, "entity_id = ?", (name,))
    if hit:
        r = hit[name]
        cand = EntityCandidate(
            entity_id=name, label=str(r[3]), kind=r[2],
            via="entity_id", confidence=1.0, scope_id=str(r[1]),
        )
        return EntityResolution(name, STATUS_RESOLVED, name, (cand,),
                                "exact_entity_id")

    # 2) exact normalized label/alias matches — all of them.
    norm = normalize_entity_text(name)
    label_hits = _entity_rows(
        conn, ids,
        "LOWER(TRIM(label)) = LOWER(TRIM(?))", (name,),
    )
    # Normalized-label match (NFKD+casefold) beyond SQL LOWER: fold in
    # Python over the same-scope rows so 'José' and 'Jose' *compete* —
    # they still never merge.
    all_scope = _entity_rows(conn, ids, "1 = 1", ())
    for eid, r in all_scope.items():
        if eid not in label_hits and normalize_entity_text(r[3]) == norm:
            label_hits[eid] = r
    alias_hits: dict = {}
    if norm:
        rows = conn.execute(
            "SELECT ea.entity_id, ea.normalized_alias FROM entity_aliases ea"
            " JOIN entities e ON e.entity_id = ea.entity_id"
            f"  AND e.scope_id IN ({_ph(len(ids))})"
            " WHERE ea.normalized_alias = ?",
            [*ids, norm],
        ).fetchall()
        for eid, _na in rows:
            if str(eid) in all_scope:
                alias_hits[str(eid)] = all_scope[str(eid)]

    cands: dict[str, EntityCandidate] = {}
    for eid, r in label_hits.items():
        cands[eid] = EntityCandidate(
            entity_id=eid, label=str(r[3]), kind=r[2],
            via="label", confidence=0.95, scope_id=str(r[1]),
        )
    for eid, r in alias_hits.items():
        if eid in cands:
            c = cands[eid]
            cands[eid] = EntityCandidate(
                entity_id=eid, label=c.label, kind=c.kind,
                via="label+alias", confidence=0.97, scope_id=c.scope_id,
            )
        else:
            cands[eid] = EntityCandidate(
                entity_id=eid, label=str(r[3]), kind=r[2],
                via="alias", confidence=0.9, scope_id=str(r[1]),
            )

    if kind is not None:
        cands = {e: c for e, c in cands.items() if c.kind == kind}

    window = set(str(e) for e in (context_entity_ids or ()))
    if context_claim_ids:
        window |= set(_context_entity_ids(
            conn, ids, context_claim_ids, ()))
    roles = _roles_for(conn, list(cands), list(context_claim_ids or ()))
    for eid in list(cands):
        c = cands[eid]
        in_window = eid in window
        cands[eid] = EntityCandidate(
            entity_id=eid, label=c.label, kind=c.kind, via=c.via,
            confidence=min(1.0, c.confidence + (0.05 if in_window else 0)),
            roles=roles.get(eid, ()), scope_id=c.scope_id,
        )
    ordered = tuple(
        sorted(cands.values(),
               key=lambda c: (-c.confidence, c.entity_id))
    )[: max(1, int(limit))]

    if not ordered:
        return EntityResolution(name, STATUS_UNRESOLVED, None, (),
                                "no_exact_match")
    if len(ordered) == 1:
        return EntityResolution(name, STATUS_RESOLVED,
                                ordered[0].entity_id, ordered,
                                f"unique_{ordered[0].via}")
    # Context evidence may narrow to a unique survivor — recorded, still
    # never a similarity guess.
    in_window = [c for c in ordered if c.entity_id in window]
    if len(in_window) == 1:
        winner = in_window[0]
        return EntityResolution(name, STATUS_RESOLVED, winner.entity_id,
                                ordered, "context_disambiguated")
    return EntityResolution(name, STATUS_AMBIGUOUS, None, ordered,
                            "multiple_exact_matches")


def resolve_reference(conn: sqlite3.Connection, scope_ids,
                      reference: str, *,
                      context_claim_ids=(), context_entity_ids=(),
                      limit: int = 16) -> ReferenceResolution:
    """Resolve one referential token (V4-17.03).

    Pronouns/deictics resolve only against the declared context window —
    ``context_entity_ids`` plus the entities of ``context_claim_ids``.
    With no window or no unique survivor the reference stays unresolved
    or ambiguous; it is NEVER guessed from recency or salience.
    """
    ref = str(reference or "").strip()
    if not ref:
        return ReferenceResolution(ref, STATUS_UNRESOLVED, None, (),
                                   0.0, "empty_reference")
    lowered = ref.casefold()

    if lowered in PRONOUNS or lowered in DEICTICS:
        if not context_claim_ids and not context_entity_ids:
            return ReferenceResolution(
                ref, STATUS_UNRESOLVED, None, (), 0.0,
                "no_context_window")
        window = _context_entity_ids(
            conn, scope_ids, context_claim_ids, context_entity_ids)
        cands: list = []
        rows = _entity_rows(
            conn, scope_ids,
            f"entity_id IN ({_ph(len(window))})" if window else "0",
            list(window),
        )
        for eid, r in rows.items():
            cands.append(EntityCandidate(
                entity_id=eid, label=str(r[3]), kind=r[2],
                via="context_window", confidence=0.5,
                roles=tuple(sorted(window.get(eid, ()))),
                scope_id=str(r[1]),
            ))
        cands.sort(key=lambda c: c.entity_id)
        cands = cands[: max(1, int(limit))]
        if not cands:
            return ReferenceResolution(ref, STATUS_UNRESOLVED, None, (),
                                       0.0, "no_entities_in_window")
        if len(cands) == 1:
            return ReferenceResolution(
                ref, STATUS_RESOLVED, cands[0].entity_id, tuple(cands),
                cands[0].confidence, "unique_in_context_window")
        return ReferenceResolution(
            ref, STATUS_AMBIGUOUS, None, tuple(cands), 0.0,
            "competing_entities_in_window")

    res = resolve_entity(
        conn, scope_ids, ref,
        context_entity_ids=context_entity_ids,
        context_claim_ids=context_claim_ids,
        limit=limit,
    )
    conf = res.candidates[0].confidence if (
        res.status == STATUS_RESOLVED and res.candidates) else 0.0
    return ReferenceResolution(
        ref, res.status, res.entity_id, res.candidates, conf, res.reason)


def resolve_text_references(conn: sqlite3.Connection, scope_ids,
                            text: str, *,
                            context_claim_ids=(), context_entity_ids=(),
                            limit: int = 16) -> tuple:
    """Resolve every referential token in ``text`` (V4-17.03).

    Pronouns/deictics are always reported (resolved, ambiguous, or
    unresolved). Other tokens are reported only when entity evidence
    exists for them — a token that matches nothing is not a reference and
    is omitted rather than reported unresolved noise.
    """
    out: list = []
    seen: set = set()
    for m in _TOKEN_RE.finditer(str(text or "")):
        token = m.group(0)
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        if key in PRONOUNS or key in DEICTICS:
            out.append(resolve_reference(
                conn, scope_ids, token,
                context_claim_ids=context_claim_ids,
                context_entity_ids=context_entity_ids,
                limit=limit))
            continue
        res = resolve_entity(
            conn, scope_ids, token,
            context_entity_ids=context_entity_ids,
            context_claim_ids=context_claim_ids,
            limit=limit,
        )
        if res.status != STATUS_UNRESOLVED:
            out.append(ReferenceResolution(
                token, res.status, res.entity_id, res.candidates,
                res.candidates[0].confidence
                if res.status == STATUS_RESOLVED and res.candidates
                else 0.0, res.reason))
    return tuple(out)
