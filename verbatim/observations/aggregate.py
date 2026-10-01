"""``slot_aggregate_v1`` producer (SPEC_V3 §23, V3-23.01..23.10).

A deterministic *rules* producer — no generation models, no network. It
groups admitted structured claim revisions by
(scope, perspective, subject, predicate, condition) into slots; each
distinct normalized value inside a slot becomes one observation candidate:

- ``proof_count`` counts DISTINCT evidence families only
  (``family_members`` → ``evidence_families``; a claim with no family row
  is its own singleton family). Same-family copies never count twice
  (V3-17.05, §18 family dedup).
- Rival values in the same slot are linked as ``contradicts`` evidence on
  each other's observations — never merged into one number (V3-23.05).
- Observation text renders from a fixed template over structured slot
  values — derived metadata, never a quotation of source bytes (V3-17.04).
- Every observation records ``derivations`` edges to each input claim
  revision, ``producer_kind='slot_aggregate_v1'``,
  ``producer_id='obs:'+observation_id`` (V3-17.02).

``record_observation`` is the manual recording path for producers other
than slot aggregation (a host-attested observation still lands as a
derived object with evidence links, never as a quotation — V3-17.07).
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from ..storage import repos_v3
from ..storage.repos import has_table as _has_table
from .freshness import next_seq

SLOT_AGGREGATE_V1 = "slot_aggregate_v1"
#: An observation is consolidated from multiple admitted facts across
#: distinct evidence families (V3-23.01): below two families there is no
#: independent corroboration and nothing to consolidate (V3-23.02
#: recurrence gate).
DEFAULT_MIN_PROOF = 2
#: Admitted head states that count as consolidation input. ``disputed``
#: claims remain evidence — they are still admitted, only contested.
ELIGIBLE_STATES = ("active", "disputed")
#: Only asserted propositions support a belief; hypothetical/habitual/
#: uncertain readings stay exact evidence, not consolidation input.
ELIGIBLE_MODALITIES = ("asserted",)

#: Purge row states that tombstone an object (mirrors
#: ``retrieval.candidates._SUPPRESSING_STATES`` — a suppressing purge hides
#: the object from producers exactly as it hides it from recall).
_SUPPRESSING_STATES = ("suppressed", "purging", "completed")
#: Quarantine states that hide an object (mirrors
#: ``security.quarantine.EXCLUDING_STATES``; re-declared so the fallback
#: path never imports a parallel module).
_EXCLUDING_QUARANTINE = frozenset({"pending", "suppressed"})

_EvidenceRef = Tuple[str, str, int]  # (object_kind, object_id, object_revision)


@dataclass(frozen=True)
class ClaimInput:
    """One admitted head claim revision visible to consolidation."""

    claim_id: str
    revision: int
    subject_id: str
    predicate: str
    value_text: str
    polarity: str
    modality: str
    condition_key: str
    perspective_id: Optional[str]
    freshness: Optional[str]
    family_id: Optional[str]


@dataclass(frozen=True)
class SlotCandidate:
    """One value inside one slot — the unit an observation consolidates."""

    observation_id: str
    scope_id: str
    subject_id: str
    predicate: str
    perspective_id: Optional[str]
    condition_key: str
    value_key: str
    value_text: str
    polarity: str
    text: str
    proof_count: int
    family_keys: Tuple[str, ...]
    supports: Tuple[_EvidenceRef, ...]
    contradicts: Tuple[_EvidenceRef, ...]
    freshness: str


# ---------------------------------------------------------------------------
# input reads (v1/v2 tables via direct SELECT — never in repos_v3._COLUMNS)
# ---------------------------------------------------------------------------


def _family_map(conn: sqlite3.Connection, claim_ids: list[str]) -> dict[str, str]:
    """claim_id → evidence family id, from family_members + claim_evidence."""
    fams: dict[str, str] = {}
    if not claim_ids:
        return fams
    ph = ",".join("?" for _ in claim_ids)
    for cid, fid in conn.execute(
        f"SELECT object_id, family_id FROM family_members"
        f" WHERE object_kind = 'claim' AND object_id IN ({ph})",
        tuple(claim_ids),
    ).fetchall():
        fams.setdefault(cid, fid)
    for cid, fid in conn.execute(
        f"SELECT DISTINCT claim_id, family_id FROM claim_evidence"
        f" WHERE claim_id IN ({ph}) AND family_id IS NOT NULL",
        tuple(claim_ids),
    ).fetchall():
        fams.setdefault(cid, fid)
    return fams


# ---------------------------------------------------------------------------
# held-object / suppression cascade (V3-14.10)
#
# Consolidation is a *producer* reading admitted claims: it must observe the
# same invisibility rule retrieval does. A claim under a quarantine hold or a
# suppressing purge — or whose evidence chain is held (the cited span, its
# source revision, or the covering source envelope) — contributes no text,
# no family count, and no contradiction to a new observation. These helpers
# mirror ``retrieval/v3/union.py::_should_exclude`` /
# ``_span_suppressed_claims`` and ``retrieval/candidates.py::_suppressed``
# on the caller's own ``conn`` so tombstones committed in the same
# transaction are visible here.
# ---------------------------------------------------------------------------


def _suppressed_ids(
    conn: sqlite3.Connection, object_kind: str, object_ids: Iterable[str]
) -> set:
    """Purge-tombstoned ids of one kind (``candidates._suppressed`` style).

    Only purges that passed preview suppress anything; evaluated on the
    caller's conn so same-transaction tombstones are seen.
    """
    ids = list(dict.fromkeys(object_ids))
    if not ids or not _has_table(conn, "purge_targets"):
        return set()
    ph = ",".join("?" for _ in ids)
    sph = ",".join("?" for _ in _SUPPRESSING_STATES)
    rows = conn.execute(
        "SELECT DISTINCT pt.object_id FROM purge_targets pt"
        " JOIN purges p ON p.purge_id = pt.purge_id"
        " WHERE pt.object_kind = ?"
        f" AND p.state IN ({sph}) AND pt.object_id IN ({ph})",
        (object_kind, *_SUPPRESSING_STATES, *ids),
    ).fetchall()
    return {r[0] for r in rows}


def _held(conn: sqlite3.Connection, kind: str, oid: str, rev: int) -> bool:
    """Quarantine hold check — ``security.should_exclude`` when the
    parallel module is provisioned, the local quarantine-table read
    otherwise (mirrors ``union._should_exclude``).

    Fail closed (F4-20, V4-05.12): a hold check that cannot be completed
    raises ``EVIDENCE_UNAVAILABLE`` — never an exception-based release of
    held content into producer input, and never a silent all-held result
    that would let ``consolidate`` retire sound observations on a
    transient read error (the raised error aborts the pass and rolls the
    transaction back instead). The only degrade-open case is the
    security module being genuinely unprovisioned — then the local read
    is the authoritative check, and it too fails closed.
    """
    try:
        from .. import security as _security  # type: ignore
    except Exception:
        _security = None
    if _security is not None:
        try:
            return bool(_security.should_exclude(conn, kind, oid, rev))
        except VerbatimError:
            raise
        except Exception as exc:
            raise VerbatimError(
                ErrorCode.EVIDENCE_UNAVAILABLE,
                f"quarantine hold check failed for {kind}:{oid}@{rev}",
            ) from exc
    try:
        row = conn.execute(
            "SELECT state FROM quarantine"
            " WHERE object_kind = ? AND object_id = ? AND revision = ?",
            (kind, oid, rev),
        ).fetchone()
    except sqlite3.Error as exc:
        raise VerbatimError(
            ErrorCode.EVIDENCE_UNAVAILABLE,
            f"quarantine hold check failed for {kind}:{oid}@{rev}",
        ) from exc
    return row is not None and row[0] in _EXCLUDING_QUARANTINE


def _evidence_held_claims(
    conn: sqlite3.Connection, pairs: list[Tuple[str, int]]
) -> set:
    """Claim ids whose evidence chain is held or suppressed (V3-14.10).

    Walks ``claim_evidence`` → ``spans`` → ``source_envelopes`` exactly as
    ``union._span_suppressed_claims`` does: a quarantine hold or purge
    tombstone on a cited span, its ``(source_id, revision)``, or the
    covering source envelope taints every dependent claim.
    """
    if not pairs:
        return set()
    where = " OR ".join("(claim_id=? AND revision=?)" for _ in pairs)
    flat = [v for pair in pairs for v in pair]
    rows = conn.execute(
        f"SELECT claim_id, span_id FROM claim_evidence WHERE {where}", flat
    ).fetchall()
    spans_of: dict[str, list[str]] = {}
    all_spans: list[str] = []
    for cid, sid in rows:
        spans_of.setdefault(cid, []).append(sid)
        all_spans.append(sid)
    all_spans = list(dict.fromkeys(all_spans))
    if not all_spans:
        return set()
    held: set = set(_suppressed_ids(conn, "span", all_spans))

    span_meta = {
        sid: (src, rev)
        for sid, src, rev in conn.execute(
            "SELECT span_id, source_id, revision FROM spans"
            f" WHERE span_id IN ({','.join('?' for _ in all_spans)})",
            all_spans,
        ).fetchall()
    }
    src_revs = set(span_meta.values())
    if src_revs:
        srcs = [s for s, _r in src_revs]
        held_sources = _suppressed_ids(conn, "source", srcs) | _suppressed_ids(
            conn, "source_revision", [f"{s}:{r}" for s, r in src_revs]
        )
    else:
        held_sources = set()
    spans_by_src: dict[tuple, list[str]] = {}
    for sid, meta in span_meta.items():
        spans_by_src.setdefault(meta, []).append(sid)
    if _has_table(conn, "quarantine"):
        for sid, (src, rev) in span_meta.items():
            if sid not in held and _held(conn, "span", sid, rev):
                held.add(sid)
        env_of: dict[tuple, list[str]] = {}
        if src_revs and _has_table(conn, "source_envelopes"):
            env_where = " OR ".join(
                "(source_id=? AND revision=?)" for _ in src_revs
            )
            env_rows = conn.execute(
                "SELECT envelope_id, source_id, revision"
                f" FROM source_envelopes WHERE {env_where}",
                [v for pair_ in src_revs for v in pair_],
            ).fetchall()
            for eid, sid2, rev2 in env_rows:
                env_of.setdefault((sid2, rev2), []).append(eid)
        for (src, rev), sids in spans_by_src.items():
            tainted = (
                src in held_sources
                or f"{src}:{rev}" in held_sources
                or _held(conn, "source", src, rev)
                or any(
                    _held(conn, "source_envelope", eid, rev)
                    for eid in env_of.get((src, rev), ())
                )
            )
            if tainted:
                held.update(sids)
    else:
        for (src, rev), sids in spans_by_src.items():
            if src in held_sources or f"{src}:{rev}" in held_sources:
                held.update(sids)
    if not held:
        return set()
    return {cid for cid, sids in spans_of.items() if held & set(sids)}


def _held_claim_ids(
    conn: sqlite3.Connection, pairs: list[Tuple[str, int]]
) -> set:
    """Claim ids invisible to producers: direct quarantine/purge holds on
    the claim itself, plus the evidence-chain cascade (V3-14.10)."""
    ids = [cid for cid, _rev in pairs]
    held = set(_suppressed_ids(conn, "claim", ids))
    held |= _evidence_held_claims(conn, pairs)
    if _has_table(conn, "quarantine"):
        for cid, rev in pairs:
            if cid not in held and _held(conn, "claim", cid, rev):
                held.add(cid)
    return held


def fetch_claim_inputs(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    states: Iterable[str] = ELIGIBLE_STATES,
    modalities: Iterable[str] = ELIGIBLE_MODALITIES,
) -> list[ClaimInput]:
    """Admitted structured head revisions in one scope (V3-23.10).

    Eligible = current head (``recorded_until IS NULL``), admitted state,
    asserted modality, non-null ``predicate`` and ``subject_id`` — unknown
    slots are never consolidated.
    """
    require_id(scope_id, "scope_id")
    state_list = list(states)
    mod_list = list(modalities)
    if not state_list or not mod_list:
        return []
    sph = ",".join("?" for _ in state_list)
    mph = ",".join("?" for _ in mod_list)
    rows = conn.execute(
        "SELECT c.claim_id, c.subject_id, c.predicate,"
        "       r.revision, r.object_json, r.polarity, r.modality,"
        "       r.condition_json, r.perspective_id, r.freshness"
        " FROM claims c"
        " JOIN claim_revisions r ON r.claim_id = c.claim_id"
        f" WHERE c.scope_id = ? AND r.recorded_until IS NULL"
        f"   AND r.state IN ({sph}) AND r.modality IN ({mph})"
        "   AND c.predicate IS NOT NULL AND c.subject_id IS NOT NULL"
        " ORDER BY c.claim_id, r.revision",
        (scope_id, *state_list, *mod_list),
    ).fetchall()
    claim_ids = [r[0] for r in rows]
    fams = _family_map(conn, claim_ids)
    # A held claim — quarantined, purge-suppressed, or standing on held
    # evidence — is invisible to producers (V3-14.10): it contributes no
    # text, family count, or contradiction to any new observation.
    held = _held_claim_ids(conn, [(r[0], int(r[3])) for r in rows])
    inputs: list[ClaimInput] = []
    for r in rows:
        if r[0] in held:
            continue
        obj = safe_json_loads(r[4]) if r[4] else None
        value_text = obj.get("text") if isinstance(obj, dict) else None
        if not isinstance(value_text, str) or not value_text.strip():
            # A structured claim without an object value is an unknown
            # slot — never consolidated (V3-23.10).
            continue
        cond = safe_json_loads(r[7]) if r[7] else None
        condition_key = json_dumps(cond) if cond is not None else ""
        inputs.append(
            ClaimInput(
                claim_id=r[0],
                revision=int(r[3]),
                subject_id=r[1],
                predicate=r[2],
                value_text=value_text.strip(),
                polarity=r[5] or "affirmative",
                modality=r[6] or "asserted",
                condition_key=condition_key,
                perspective_id=r[8],
                freshness=r[9],
                family_id=fams.get(r[0]),
            )
        )
    return inputs


# ---------------------------------------------------------------------------
# deterministic rendering + identity
# ---------------------------------------------------------------------------


def _norm_value(text: str) -> str:
    return " ".join(text.split()).lower()


def value_key_for(polarity: str, value_text: str) -> str:
    """Canonical alternative identity inside a slot: polarity + value."""
    return f"{polarity}:{_norm_value(value_text)}"


def observation_id_for(
    scope_id: str,
    subject_id: str,
    predicate: str,
    perspective_id: Optional[str],
    condition_key: str,
    value_key: str,
    producer: str = SLOT_AGGREGATE_V1,
) -> str:
    """Deterministic observation id for one (slot, value) pair.

    Identity derives from the aggregation key, so a later consolidation
    over the same alternatives converges on the same observation and
    evolves by revision — never by silent overwrite (V3-23.04).
    """
    raw = "|".join(
        [
            producer,
            scope_id,
            subject_id,
            predicate,
            perspective_id or "",
            condition_key or "",
            value_key,
        ]
    )
    return "obs_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def render_text(
    subject_id: str,
    predicate: str,
    value_text: str,
    polarity: str,
    proof_count: int,
) -> str:
    """Fixed-template observation text (V3-17.04, V3-23.10).

    Rendered from structured slot values only — this string is derived
    metadata and is never a quotation of source bytes.
    """
    value = value_text if polarity == "affirmative" else f"not {value_text}"
    noun = "family" if proof_count == 1 else "families"
    return (
        f"{subject_id} {predicate}={value} supported by "
        f"{proof_count} independent evidence {noun}"
    )


def _merged_freshness(inputs: Iterable[ClaimInput]) -> str:
    """Weakest freshness across inputs — a belief is no fresher than the
    least durable evidence it stands on (§17.08 merge)."""
    order = ("volatile", "revalidate_after", "unknown", "stable")
    classes = {(c.freshness or "unknown") for c in inputs}
    for klass in order:
        if klass in classes:
            return klass
    return "unknown"


def aggregate_candidates(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    producer: str = SLOT_AGGREGATE_V1,
    states: Iterable[str] = ELIGIBLE_STATES,
    modalities: Iterable[str] = ELIGIBLE_MODALITIES,
    inputs: Optional[list[ClaimInput]] = None,
) -> list[SlotCandidate]:
    """Group eligible revisions into slot candidates (V3-23.10).

    One candidate per (subject, predicate, perspective, condition, value).
    ``supports`` are the revisions asserting the value; ``contradicts``
    are the same-slot revisions asserting rival values — each side is
    rendered on the other side's observation, per V3-23.05.

    ``inputs`` overrides the input set (windowed passes pre-filter to the
    touched region — V4-24.01); when omitted the whole scope is fetched.
    """
    if inputs is None:
        inputs = fetch_claim_inputs(
            conn, scope_id, states=states, modalities=modalities
        )
    slots: dict[tuple, list[ClaimInput]] = {}
    for ci in inputs:
        key = (ci.subject_id, ci.predicate, ci.perspective_id or "", ci.condition_key)
        slots.setdefault(key, []).append(ci)

    candidates: list[SlotCandidate] = []
    for (subject_id, predicate, perspective, cond_key), members in slots.items():
        values: dict[str, list[ClaimInput]] = {}
        for ci in members:
            values.setdefault(value_key_for(ci.polarity, ci.value_text), []).append(ci)
        for vkey, supporters in values.items():
            rivals = [ci for ci in members if value_key_for(ci.polarity, ci.value_text) != vkey]
            # Family dedup (V3-17.05): a claim with no family row is its
            # own singleton family — it is independent evidence.
            fam_keys = {
                ci.family_id if ci.family_id else f"claim:{ci.claim_id}"
                for ci in supporters
            }
            proof = len(fam_keys)
            rep = supporters[0]
            perspective_id = rep.perspective_id
            obs_id = observation_id_for(
                scope_id, subject_id, predicate, perspective_id, cond_key, vkey, producer
            )
            supports = tuple(
                sorted(("claim", ci.claim_id, ci.revision) for ci in supporters)
            )
            contradicts = tuple(
                sorted(("claim", ci.claim_id, ci.revision) for ci in rivals)
            )
            candidates.append(
                SlotCandidate(
                    observation_id=obs_id,
                    scope_id=scope_id,
                    subject_id=subject_id,
                    predicate=predicate,
                    perspective_id=perspective_id,
                    condition_key=cond_key,
                    value_key=vkey,
                    value_text=rep.value_text,
                    polarity=rep.polarity,
                    text=render_text(subject_id, predicate, rep.value_text, rep.polarity, proof),
                    proof_count=proof,
                    family_keys=tuple(sorted(fam_keys)),
                    supports=supports,
                    contradicts=contradicts,
                    freshness=_merged_freshness([*supporters, *rivals]),
                )
            )
    candidates.sort(key=lambda c: (c.subject_id, c.predicate, c.value_key))
    return candidates


# ---------------------------------------------------------------------------
# observation persistence (shared by consolidate + record_observation)
# ---------------------------------------------------------------------------


def _evidence_set(
    conn: sqlite3.Connection, observation_id: str, revision: int
) -> set[tuple]:
    rows = repos_v3.query(
        conn,
        "observation_evidence",
        {"observation_id": observation_id, "revision": revision},
    )
    return {
        (r["role"], r["object_kind"], r["object_id"], int(r["object_revision"]))
        for r in rows
    }


def _resolve_perspective(
    conn: sqlite3.Connection, perspective_id: Optional[str]
) -> Optional[str]:
    """FK-safe perspective write: v2 claims may name a perspective that was
    never persisted; a missing row degrades to NULL, never an FK crash."""
    if not perspective_id:
        return None
    row = repos_v3.get(conn, "perspectives", {"perspective_id": perspective_id})
    return perspective_id if row is not None else None


@dataclass(frozen=True)
class PersistResult:
    """Outcome of one ``persist_observation`` call.

    ``wrote`` is ``False`` when the head already carried the identical
    text/proof/evidence — an idempotent no-op (V3-23.04, V4-24.08).
    When it is ``True``, ``added``/``removed`` name exactly which input
    refs entered and left the support/contradiction sets relative to the
    previous revision — the durable "why" behind the new derivative
    revision (V4-24.07); callers record it on the pass event.
    """

    wrote: bool
    edges_written: int
    revision: int
    previous_revision: Optional[int]
    reactivated: bool
    added: Tuple[_EvidenceRef, ...]
    removed: Tuple[_EvidenceRef, ...]


def persist_observation(
    conn: sqlite3.Connection,
    *,
    scope_id: str,
    observation_id: str,
    text: str,
    proof_count: int,
    perspective_id: Optional[str],
    freshness: str,
    producer: str,
    supports: Iterable[_EvidenceRef],
    contradicts: Iterable[_EvidenceRef],
    seq: Optional[int] = None,
) -> PersistResult:
    """Write or evolve one observation head row.

    Idempotent (V3-23.04): when the stored head already carries the same
    text, proof count, and evidence sets, nothing is written and no new
    revision appears. When the support/contradiction set changed, a new
    revision is appended — the prior belief stays visible through the
    immutable ``observation_evidence``/``derivations`` rows at earlier
    revisions.

    Returns a :class:`PersistResult` (was a ``(wrote, edges, revision)``
    tuple); ``added``/``removed`` diff the prior revision's evidence so a
    refinement can say what changed (V4-24.07).
    """
    supports_set = {tuple(s) for s in supports}
    contradicts_set = {tuple(c) for c in contradicts} - supports_set
    if seq is None:
        seq = next_seq(conn, scope_id)
    desired = {
        ("supports", k, i, r) for (k, i, r) in supports_set
    } | {("contradicts", k, i, r) for (k, i, r) in contradicts_set}

    head = repos_v3.get(conn, "observations", {"observation_id": observation_id})
    persp = _resolve_perspective(conn, perspective_id)
    previous_revision: Optional[int] = None
    reactivated = False
    prior: set[tuple] = set()
    if head is not None and head["recorded_until"] is None:
        cur_rev = int(head["revision"])
        prior = _evidence_set(conn, observation_id, cur_rev)
        if (
            prior == desired
            and head["text"] == text
            and int(head["proof_count"]) == int(proof_count)
        ):
            return PersistResult(
                wrote=False,
                edges_written=0,
                revision=cur_rev,
                previous_revision=cur_rev,
                reactivated=False,
                added=(),
                removed=(),
            )
        previous_revision = cur_rev
        revision = cur_rev + 1
        repos_v3.update(
            conn,
            "observations",
            {
                "revision": revision,
                "text": text,
                "proof_count": int(proof_count),
                "perspective_id": persp,
                "freshness": freshness,
                "stale_since_seq": None,
                "producer": producer,
                "recorded_from": seq,
                "recorded_until": None,
            },
            {"observation_id": observation_id},
        )
    elif head is None:
        revision = 1
        repos_v3.insert(
            conn,
            "observations",
            {
                "observation_id": observation_id,
                "scope_id": scope_id,
                "revision": revision,
                "text": text,
                "proof_count": int(proof_count),
                "perspective_id": persp,
                "freshness": freshness,
                "stale_since_seq": None,
                "producer": producer,
                "recorded_from": seq,
                "recorded_until": None,
            },
        )
    else:
        # Reactivation: the observation was retired when its supports
        # vanished; re-derived evidence appends a fresh revision.
        previous_revision = int(head["revision"])
        prior = _evidence_set(conn, observation_id, previous_revision)
        reactivated = True
        revision = previous_revision + 1
        repos_v3.update(
            conn,
            "observations",
            {
                "revision": revision,
                "text": text,
                "proof_count": int(proof_count),
                "perspective_id": persp,
                "freshness": freshness,
                "stale_since_seq": None,
                "producer": producer,
                "recorded_from": seq,
                "recorded_until": None,
            },
            {"observation_id": observation_id},
        )

    for role, kind, oid, orev in sorted(desired):
        repos_v3.insert(
            conn,
            "observation_evidence",
            {
                "observation_id": observation_id,
                "revision": revision,
                "role": role,
                "object_kind": kind,
                "object_id": oid,
                "object_revision": int(orev),
            },
        )

    inputs = sorted(supports_set | contradicts_set)
    for kind, oid, orev in inputs:
        repos_v3.insert(
            conn,
            "derivations",
            {
                "child_kind": "observation",
                "child_id": observation_id,
                "child_revision": revision,
                "parent_kind": kind,
                "parent_id": oid,
                "parent_revision": int(orev),
                "producer_kind": producer,
                "producer_id": f"obs:{observation_id}",
                "seq": seq,
                "scope_id": scope_id,
            },
        )
    return PersistResult(
        wrote=True,
        edges_written=len(inputs),
        revision=revision,
        previous_revision=previous_revision,
        reactivated=reactivated,
        added=tuple(sorted(desired - prior)),
        removed=tuple(sorted(prior - desired)),
    )


def record_observation(
    conn: sqlite3.Connection,
    *,
    scope_id: str,
    text: str,
    proof_count: int = 0,
    supports: Iterable[Any] = (),
    contradicts: Iterable[Any] = (),
    perspective_id: Optional[str] = None,
    freshness: str = "unknown",
    producer: str = "manual",
    observation_id: Optional[str] = None,
    seq: Optional[int] = None,
) -> str:
    """Record one observation from a producer other than consolidation.

    ``supports``/``contradicts`` accept ``(object_kind, object_id,
    revision)`` triples or ``(object_id, revision)`` pairs (treated as
    claims). The observation lands as a derived object with derivation
    edges to every cited input — a producer decision with its receipt
    (V3-17.02, V3-17.07), and ``text`` is caller-supplied derived metadata,
    never a quotation of source bytes.
    """
    require_id(scope_id, "scope_id")
    if not isinstance(text, str) or not text.strip():
        raise VerbatimError(ErrorCode.VALIDATION, "observation text required")
    require_id(producer, "producer")

    def _norm(refs: Iterable[Any]) -> list[_EvidenceRef]:
        out: list[_EvidenceRef] = []
        for ref in refs:
            if len(ref) == 2:
                out.append(("claim", ref[0], int(ref[1])))
            else:
                out.append((ref[0], ref[1], int(ref[2])))
        return out

    sup = _norm(supports)
    con = _norm(contradicts)
    if observation_id is None:
        observation_id = "obs_" + hashlib.sha256(
            "|".join([producer, scope_id, text]).encode("utf-8")
        ).hexdigest()[:32]
    else:
        require_id(observation_id, "observation_id")
    persist_observation(
        conn,
        scope_id=scope_id,
        observation_id=observation_id,
        text=text,
        proof_count=proof_count,
        perspective_id=perspective_id,
        freshness=freshness,
        producer=producer,
        supports=sup,
        contradicts=con,
        seq=seq if seq is not None else next_seq(conn, scope_id),
    )
    return observation_id


# ---------------------------------------------------------------------------
# pin liveness — grounded-consolidation invariants (SPEC_V6 §03.2, V6-03.13)
#
# An observation's evidence rows pin exact object revisions; the pins must
# resolve to independently retrievable source bytes — the rendered text is
# derived metadata, never the sole trace of the fact it summarizes.
# ``resolve_pins`` reports per-pin liveness (resolvable / held / erased /
# superseded / byte-trace); ``never_sole_trace`` is the strict invariant:
# every pin must chain to a live, non-empty ``source_revisions`` payload.
# ---------------------------------------------------------------------------


#: ``source_state`` dispositions that mean the bytes were erased — mirrors
#: ``dedup.links._DEAD_DISPOSITIONS`` (re-declared so this module never
#: imports the dedup lane for a lifecycle constant).
_DEAD_SOURCE_DISPOSITIONS = frozenset({"erased"})
#: Retained-but-superseded source dispositions — mirrors
#: ``dedup.links._DEPREFERRED_DISPOSITIONS``.
_DEPREFERRED_SOURCE_DISPOSITIONS = frozenset(
    {"superseded", "corrected", "retracted", "archived"}
)
#: Claim revision states whose revision is closed historical evidence.
_CLOSED_CLAIM_STATES = frozenset({"superseded", "rejected", "archived"})


def _source_revision_facts(
    conn: sqlite3.Connection, source_id: str, revision: int
) -> dict:
    """Liveness facts for one ``(source_id, revision)`` endpoint.

    ``held`` follows the V3-14.10 cascade surface — a quarantine hold on
    the source revision or on any covering ``source_envelopes`` row.
    ``erased`` covers every byte-destruction signal the closure paths
    write: suppressing/purging purge tombstones (``source`` or
    ``source_revision`` object kinds), an emptied payload, or a dead
    ``source_state`` disposition.
    """
    facts = {
        "exists": False,
        "payload_bytes": 0,
        "held": False,
        "erased": False,
        "superseded": False,
        "envelopes": [],
    }
    srow = conn.execute(
        "SELECT 1 FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    rrow = conn.execute(
        "SELECT length(payload) FROM source_revisions"
        " WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    ).fetchone()
    facts["exists"] = srow is not None and rrow is not None
    facts["payload_bytes"] = int(rrow[0]) if rrow is not None else 0
    if srow is None:
        return facts
    if rrow is not None and int(rrow[0]) == 0:
        # Emptied revision bytes are the closure erasure signal — the row
        # persists as tombstone metadata (``dedup.links._member_info``
        # honors the same convention).
        facts["erased"] = True
    if source_id in _suppressed_ids(conn, "source", [source_id]):
        facts["erased"] = True
    sr_key = f"{source_id}:{revision}"
    if sr_key in _suppressed_ids(conn, "source_revision", [sr_key]):
        facts["erased"] = True
    if _has_table(conn, "source_state"):
        st = conn.execute(
            "SELECT disposition FROM source_state WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if st is not None:
            if st[0] in _DEAD_SOURCE_DISPOSITIONS:
                facts["erased"] = True
            elif st[0] in _DEPREFERRED_SOURCE_DISPOSITIONS:
                facts["superseded"] = True
    if _has_table(conn, "quarantine"):
        if _held(conn, "source", source_id, revision):
            facts["held"] = True
        env_ids: list = []
        if _has_table(conn, "source_envelopes"):
            env_ids = [
                str(r[0])
                for r in conn.execute(
                    "SELECT envelope_id FROM source_envelopes"
                    " WHERE source_id = ? AND revision = ?",
                    (source_id, revision),
                ).fetchall()
            ]
        facts["envelopes"] = env_ids
        for eid in env_ids:
            if _held(conn, "source_envelope", eid, revision):
                facts["held"] = True
    return facts


def _pin_facts(
    conn: sqlite3.Connection,
    role: str,
    object_kind: str,
    object_id: str,
    revision: int,
) -> dict:
    """Liveness of one observation evidence pin.

    ``resolvable`` means the pinned revision row still exists — the ref
    is not dangling. ``held``/``erased``/``superseded`` are independent
    lifecycle flags over the pin *and* its evidence chain (a claim pin
    inherits the posture of the spans/sources it cites — V3-14.10).
    ``byte_trace`` is the strict invariant: the pin resolves to a live,
    non-empty source payload that is neither held nor erased — the only
    honest answer to "could a reader independently retrieve the bytes".
    ``source_refs`` lists the ``(source_id, revision)`` endpoints the pin
    reaches through ``claim_evidence`` → ``spans``.
    """
    out = {
        "role": role,
        "object_kind": object_kind,
        "object_id": object_id,
        "revision": int(revision),
        "resolvable": False,
        "held": False,
        "erased": False,
        "superseded": False,
        "byte_trace": False,
        "source_refs": [],
    }
    if object_kind == "source":
        facts = _source_revision_facts(conn, object_id, int(revision))
        out["resolvable"] = facts["exists"]
        out["held"] = facts["held"]
        out["erased"] = facts["erased"]
        out["superseded"] = facts["superseded"]
        out["source_refs"] = [[object_id, int(revision)]]
        out["byte_trace"] = (
            facts["exists"]
            and facts["payload_bytes"] > 0
            and not facts["held"]
            and not facts["erased"]
        )
        return out
    if object_kind != "claim":
        # An unknown pin kind can never prove byte grounding — fail the
        # trace rather than assume it (V6-03.13(a)).
        return out
    row = conn.execute(
        "SELECT state, recorded_until FROM claim_revisions"
        " WHERE claim_id = ? AND revision = ?",
        (object_id, int(revision)),
    ).fetchone()
    if row is None:
        return out
    out["resolvable"] = True
    state = str(row[0] or "")
    if row[1] is not None or state in _CLOSED_CLAIM_STATES:
        out["superseded"] = True
    if state == "erased":
        out["erased"] = True
    if object_id in _suppressed_ids(conn, "claim", [object_id]):
        out["erased"] = True
    if _has_table(conn, "quarantine") and _held(
        conn, "claim", object_id, int(revision)
    ):
        out["held"] = True
    # Evidence-chain cascade (mirrors ``_evidence_held_claims``): holds
    # and tombstones on a cited span, its source revision, or the
    # covering source envelope are the pin's own posture.
    span_rows = conn.execute(
        "SELECT span_id FROM claim_evidence"
        " WHERE claim_id = ? AND revision = ?",
        (object_id, int(revision)),
    ).fetchall()
    span_ids = [str(r[0]) for r in span_rows]
    if span_ids:
        if _suppressed_ids(conn, "span", span_ids) & set(span_ids):
            out["erased"] = True
        span_meta = {
            str(r[0]): (str(r[1]), int(r[2]))
            for r in conn.execute(
                "SELECT span_id, source_id, revision FROM spans"
                f" WHERE span_id IN ({','.join('?' for _ in span_ids)})",
                span_ids,
            ).fetchall()
        }
        held_spans: set = set()
        for sid, (src, srev) in span_meta.items():
            out["source_refs"].append([src, srev])
            facts = _source_revision_facts(conn, src, srev)
            if facts["held"]:
                held_spans.add(sid)
            if facts["erased"]:
                out["erased"] = True
            if facts["superseded"]:
                out["superseded"] = True
            if (
                facts["exists"]
                and facts["payload_bytes"] > 0
                and not facts["held"]
                and not facts["erased"]
            ):
                out["byte_trace"] = True
        if held_spans:
            out["held"] = True
        if _has_table(conn, "quarantine"):
            for sid, (_src, srev) in span_meta.items():
                if _held(conn, "span", sid, srev):
                    out["held"] = True
    return out


def resolve_pins(
    conn: sqlite3.Connection,
    observation_id: str,
    *,
    revision: Optional[int] = None,
) -> dict:
    """Per-pin liveness report for one observation (V6-03.13(a)/(b)).

    Resolves the observation's head revision (or an explicit ``revision``)
    and walks every ``observation_evidence`` row to its pinned object —
    and, for claim pins, onward through ``claim_evidence`` → ``spans`` to
    the source revisions carrying the bytes. Unknown observation ids
    return ``found=False`` rather than raising — the audit caller decides
    how to score an absent object.
    """
    require_id(observation_id, "observation_id")
    head = repos_v3.get(
        conn, "observations", {"observation_id": observation_id}
    )
    if head is None:
        return {
            "observation_id": observation_id,
            "found": False,
            "revision": None,
            "recorded_until": None,
            "pins": [],
            "summary": {
                "total": 0,
                "resolvable": 0,
                "held": 0,
                "erased": 0,
                "superseded": 0,
                "byte_trace": 0,
            },
        }
    rev = int(revision) if revision is not None else int(head["revision"])
    rows = repos_v3.query(
        conn,
        "observation_evidence",
        {"observation_id": observation_id, "revision": rev},
    )
    pins = [
        _pin_facts(
            conn,
            str(r["role"]),
            str(r["object_kind"]),
            str(r["object_id"]),
            int(r["object_revision"]),
        )
        for r in rows
    ]
    pins.sort(key=lambda p: (p["role"], p["object_kind"], p["object_id"]))
    return {
        "observation_id": observation_id,
        "found": True,
        "revision": rev,
        "recorded_until": head["recorded_until"],
        "pins": pins,
        "summary": {
            "total": len(pins),
            "resolvable": sum(1 for p in pins if p["resolvable"]),
            "held": sum(1 for p in pins if p["held"]),
            "erased": sum(1 for p in pins if p["erased"]),
            "superseded": sum(1 for p in pins if p["superseded"]),
            "byte_trace": sum(1 for p in pins if p["byte_trace"]),
        },
    }


def never_sole_trace(
    conn: sqlite3.Connection,
    observation_id: str,
    *,
    revision: Optional[int] = None,
) -> bool:
    """The summary-is-never-the-sole-trace invariant (V6-03.13(a)).

    True only when the observation exists, cites at least one pin, and
    *every* pin resolves to independently retrievable source bytes: the
    pin's chain reaches a ``source_revisions`` row with a non-empty
    payload, unheld and unerased. A pinned object that cannot produce
    bytes — dangling, emptied, purged, or under hold — fails closed:
    the derived text must never be the only readable evidence.
    """
    report = resolve_pins(conn, observation_id, revision=revision)
    if not report.get("found"):
        return False
    pins = report["pins"]
    if not pins:
        return False
    return all(p["resolvable"] and p["byte_trace"] for p in pins)
