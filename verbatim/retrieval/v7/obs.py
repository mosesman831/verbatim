"""Observations lane for the V7 read path (V7-14, L-obs).

L-obs in the §04.2 pipeline: surface the *consolidated* artifacts —
``observations_v7`` (grounded beliefs over ≥ 2 support units about one
subject slot), ``profiles_v7`` (per-subject slot materializations), and
``standing_queries`` (stored answer packs rebuilt in the background) — as
fusion candidates backed by their pinned support units.  The lane never
replaces raw evidence: it participates in fusion like any lane
(V7-14.09), and it abstains entirely on the intents that belong to other
lanes (identifier lookups → L-exact/L-lex; ``TEMPORAL_*`` → L-time).

Honesty rules honored here (V7-04.03, LaneV7 protocol):

- consolidated artifacts are closure members (V7-14.08): every emitted
  candidate is backed by a *support unit* resolved through the
  ``(scope_id, generation)`` fence and the caller's eligible set — an
  observation whose supports are all ineligible emits nothing and is
  counted in ``stats["ineligible"]``;
- ``stale`` observations still emit (freshness is information, V7-14.03)
  but are flagged ``signals["stale"]=1`` and ranked below fresh rows;
- ``dirty`` standing-query packs still emit flagged
  ``signals["standing_dirty"]=1`` — a pending rebuild is honest state,
  not a failure;
- scoring is evidence-weighted but trivially deterministic:
  ``raw_score = 1.0 + 0.2*ln(1+proof_count)`` (Hindsight's proof-count
  idea, V7-25.05) — profiles and standing seeds score the 1.0 base;
- a missing artifact table-set reports ``unavailable`` /
  ``reason="no_obs_tables"``; individually missing tables degrade that
  phase only and are named in stats.

Matching (deterministic, no models):

- observations match when a folded query term occurs in the folded
  ``slot`` (substring / normalized-equality) **or** a query entity/speaker
  canon occurs in the folded ``text``/``support_refs_json``;
- profiles match on ``subject_canon IN query.entity_canons`` (folded);
- standing queries match on exact equality between the folded stored
  ``query`` and the incoming normalized query text — no fuzzy match,
  no fabrication.

formula_status: ``provisional/v7-r0``.

The lane is stdlib + sqlite3 only and codes against the §30 column
contract (``observations_v7``/``profiles_v7``/``standing_queries`` and
the ``units`` eligibility join) plus the frozen ``types_v7`` API.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ...core.types_v7 import (
    CandidateV7,
    IntentClass,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    LaneV7,
    QueryViewV7,
)

LANE_NAME = LaneName.OBS.value  # "obs" — stable coverage key
LANE_VERSION = "obs_lane/v7-r0"  # provisional constants (V7-32.01)

OBS_TABLE = "observations_v7"
PROFILES_TABLE = "profiles_v7"
SQ_TABLE = "standing_queries"
UNITS_TABLE = "units"

OBS_ROW_LIMIT = 4096     # bounded scans; hitting it marks partial/scan_bound
PROFILE_ROW_LIMIT = 4096
SQ_ROW_LIMIT = 1024
SQ_SEED_LIMIT = 256      # max units seeded from one stored pack
IN_CHUNK = 200
DEADLINE_CHECK_ROWS = 64

W_PROOF = 0.2  # raw_score = 1.0 + W_PROOF * ln(1 + proof_count)

# Intents the lane declines (V7-14.09: L-obs never stands in for the lanes
# that own them).  Identifier-only queries are exact-id business; the
# temporal family is L-time's domain.
_IDENTIFIER_ONLY = frozenset({IntentClass.IDENTIFIER})
_TEMPORAL_INTENTS = frozenset(
    {
        IntentClass.TEMPORAL_POINT,
        IntentClass.TEMPORAL_ORDER,
        IntentClass.DURATION,
        IntentClass.COUNT_AGGREGATE,
        IntentClass.TEMPORAL_RANGE,
        IntentClass.HISTORY_OF,
        IntentClass.CURRENT_VALUE,
    }
)

_UNIT_COLS = (
    "u.rowid, u.unit_id, u.source_id, u.revision, u.kind, u.session_id,"
    " u.speaker_canon, u.recorded_at_us, u.occurred_start_us,"
    " u.occurred_end_us, u.occurred_precision, u.occurred_source"
)

# JSON keys treated as unit references inside support_refs / pack blobs.
_REF_KEYS = frozenset({"unit_id", "unit", "unit_ref", "ref", "uid"})


# ---------------------------------------------------------------------------
# small pure helpers (deterministic; local so the lane never imports a
# concurrent worker's module)
# ---------------------------------------------------------------------------


def _fold(text: str) -> str:
    """``norm/v2``-equivalent matching fold (NFKC + casefold + diacritic
    strip); idempotent over already-folded input."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? AND type IN ('table','view')",
        (name,),
    ).fetchone()
    return row is not None


def _resolve_conn(ctx: LaneContextV7) -> Optional[sqlite3.Connection]:
    """Find the caller's pinned read snapshot. Lanes never open a new
    transaction (LaneV7 protocol), so a bare Store without a pinned conn
    resolves to None -> ``unavailable`` rather than a fresh snapshot."""
    conn = getattr(ctx, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    store = getattr(ctx, "store", None)
    if isinstance(store, sqlite3.Connection):
        return store
    conn = getattr(store, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    return None


def _eligibility(elig: Any) -> Optional[Callable[[dict], bool]]:
    """Normalize ``ctx.eligible`` into ``row -> bool`` (same accepted forms
    as the other lanes: callable(unit_row|unit_id), ``is_eligible(row)``,
    or a set-like container of unit_ids). Anything else returns None so
    the lane fails closed."""
    if elig is None:
        return None
    if callable(elig):

        def _call(row: dict) -> bool:
            try:
                return bool(elig(row))
            except TypeError:
                return bool(elig(row["unit_id"]))

        return _call
    if hasattr(elig, "is_eligible"):
        return lambda row: bool(elig.is_eligible(row))
    if hasattr(elig, "__contains__"):
        return lambda row: row["unit_id"] in elig
    return None


def _query_terms(qv: QueryViewV7) -> tuple[str, ...]:
    """Folded query terms (text + stem channels), deduped, ordered."""
    seen: list[str] = []
    norm = getattr(qv, "norm", None)
    for t in (norm.terms if norm is not None else ()) or ():
        if t.channel not in ("text", "stem"):
            continue
        term = _fold(t.term)
        if term and term not in seen:
            seen.append(term)
    return tuple(seen)


def _entity_canons(qv: QueryViewV7) -> frozenset:
    """Folded entity canons — the profile match set (V7-14.06)."""
    cans = {_fold(c) for c in (qv.entity_canons or ()) if c}
    cans.discard("")
    return frozenset(cans)


def _subject_canons(qv: QueryViewV7) -> frozenset:
    """Entity canons plus the query speaker — the observation match set
    (same convention as L-time: "what do *I* usually eat")."""
    cans = set(_entity_canons(qv))
    if qv.speaker_canon:
        cans.add(_fold(qv.speaker_canon))
    cans.discard("")
    return frozenset(cans)


def _norm_query_text(qv: QueryViewV7) -> str:
    """The incoming normalized query text for the standing-query exact
    match: ``norm.text`` when the analyzer supplied a folded projection,
    else the raw query folded here (fold is idempotent)."""
    norm = getattr(qv, "norm", None)
    text = getattr(norm, "text", "") if norm is not None else ""
    return _fold(text or getattr(qv, "query", "") or "")


# ---------------------------------------------------------------------------
# ref / pack-blob parsing — deterministic JSON walks, never fabricated
# ---------------------------------------------------------------------------


def _collect_refs(node: Any, out: list[str]) -> None:
    """Walk decoded JSON collecting unit references in document order.

    Accepted shapes: a bare string (a single unit id), a list of strings,
    a list of pin objects, or nested dicts/lists where unit ids sit under
    ``unit_id``/``unit``/``unit_ref``/``ref``/``uid`` keys. Other string
    values (quotes, slots, labels) are never mistaken for unit ids.
    """
    if isinstance(node, str):
        out.append(node)
        return
    if isinstance(node, dict):
        for key, val in node.items():
            if isinstance(val, str) and str(key).lower() in _REF_KEYS:
                out.append(val)
            elif isinstance(val, (dict, list)):
                _collect_refs(val, out)
        return
    if isinstance(node, list):
        for item in node:
            _collect_refs(item, out)


def _parse_refs(refs_json: Any) -> tuple[str, ...]:
    """Ordered, deduped unit ids from a ``support_refs_json`` /
    ``contradict_refs_json`` value. Unparseable content yields ``()`` —
    the row then simply has nothing eligible to stand on."""
    if not refs_json:
        return ()
    text = refs_json.decode("utf-8", "strict") if isinstance(refs_json, bytes) else refs_json
    if not isinstance(text, str):
        return ()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return ()
    out: list[str] = []
    _collect_refs(data, out)
    seen: dict[str, None] = {}
    for ref in out:
        ref = ref.strip()
        if ref and ref not in seen:
            seen[ref] = None
    return tuple(seen)


def _pack_unit_ids(blob: Any) -> tuple[str, ...]:
    """Unit ids seeded from a standing query's stored ``pack_blob``.

    The blob is decoded as UTF-8 JSON and walked for unit references;
    anything else (opaque bytes, non-JSON) seeds nothing — the lane emits
    only what it can pin to real units."""
    if blob is None:
        return ()
    if isinstance(blob, bytes):
        try:
            blob = blob.decode("utf-8", "strict")
        except UnicodeDecodeError:
            return ()
    if not isinstance(blob, str):
        return ()
    try:
        data = json.loads(blob)
    except (ValueError, TypeError):
        return ()
    out: list[str] = []
    _collect_refs(data, out)
    seen: dict[str, None] = {}
    for ref in out:
        ref = ref.strip()
        if ref and ref not in seen:
            seen[ref] = None
    return tuple(seen)


# ---------------------------------------------------------------------------
# candidate accumulator
# ---------------------------------------------------------------------------


@dataclass
class _ObsContribution:
    obs_id: str
    slot: str
    proof: int
    stale: int
    n_support: int
    n_contradict: int
    producer: str


@dataclass
class _ProfileContribution:
    subject: str
    slot: str
    value: Any
    status: Any


@dataclass
class _StandingContribution:
    sq_id: str
    dirty: int


@dataclass
class _Cand:
    """One pooled candidate keyed by backing unit_id. Contributions from
    the three artifact kinds merge onto the same unit; the candidate is
    stale only when *every* contribution is a stale observation."""

    unit_id: str
    source_id: str = ""
    revision: int = 0
    score: float = 1.0
    obs: list[_ObsContribution] = field(default_factory=list)
    profiles: list[_ProfileContribution] = field(default_factory=list)
    standing: list[_StandingContribution] = field(default_factory=list)

    @property
    def stale(self) -> int:
        if not self.obs:
            return 0
        if self.profiles or self.standing:
            return 0  # a fresh artifact path backs this unit too
        return 1 if all(o.stale for o in self.obs) else 0

    def order_key(self) -> tuple:
        # fresh before stale; then proof-weighted score desc; unit_id asc.
        return (self.stale, -self.score, self.unit_id)


def _unit_row_dict(row: tuple) -> dict:
    """Eligibility view of a unit row: the §30 minimum columns."""
    return {
        "unit_id": row[1],
        "source_id": row[2],
        "revision": row[3],
        "kind": row[4],
        "session_id": row[5],
        "speaker_canon": row[6],
        "recorded_at_us": row[7],
        "occurred_start_us": row[8],
        "occurred_end_us": row[9],
        "occurred_precision": row[10],
        "occurred_source": row[11],
    }


def _load_units(
    conn: sqlite3.Connection, scope_id: str, generation: int, unit_ids: list[str]
) -> dict[str, tuple]:
    """Batch-resolve support unit ids through the (scope, generation)
    fence. Ids absent from ``units`` resolve to nothing — an artifact
    pinned to a missing/out-of-generation unit has no eligible backing.

    V7-30.02: the fence is ``generation <= pinned`` and the newest row
    per unit_id wins (ORDER BY generation DESC, first-seen kept)."""
    out: dict[str, tuple] = {}
    ids = sorted(set(unit_ids))
    for i in range(0, len(ids), IN_CHUNK):
        chunk = ids[i : i + IN_CHUNK]
        rows = conn.execute(
            f"SELECT {_UNIT_COLS} FROM units u"
            " WHERE u.scope_id = ? AND u.generation <= ?"
            f" AND u.unit_id IN ({_ph(len(chunk))})"
            " ORDER BY u.unit_id, u.generation DESC",
            [scope_id, generation, *chunk],
        ).fetchall()
        for r in rows:
            out.setdefault(r[1], r)
    return out


def _obs_matches(
    slot: Any, text: Any, refs_json: Any, terms: tuple[str, ...], canons: frozenset
) -> bool:
    """Deterministic observation match: folded query term occurs in the
    folded slot (LIKE '%term%' / normalized equality — substring subsumes
    both), or a query canon occurs in the folded text / support refs."""
    slot_f = _fold(slot or "")
    if terms and slot_f:
        for t in terms:
            if t and t in slot_f:
                return True
    if canons:
        text_f = _fold(text or "")
        refs_f = _fold(refs_json if isinstance(refs_json, str) else "")
        for c in canons:
            if c and (c in text_f or c in refs_f):
                return True
    return False


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def lane_obs(ctx: LaneContextV7, qv: QueryViewV7, slice: LaneSlice) -> LaneOutput:
    """L-obs (V7-14): consolidated observations / profiles / standing
    queries as fusion candidates backed by eligible support units."""
    out = LaneOutput(lane=LANE_NAME, status=LaneStatus.OK)
    stats = out.stats
    stats["lane_version"] = LANE_VERSION
    stats["formula_status"] = "provisional/v7-r0"

    t0 = time.monotonic()
    deadline_s = max(0.0, float(getattr(slice, "deadline_ms", 0.0) or 0.0)) / 1000.0

    def expired() -> bool:
        return (time.monotonic() - t0) > deadline_s

    conn = _resolve_conn(ctx)
    if conn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_read_snapshot"
        return out
    if ctx.generation is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "generation_unpinned"
        return out

    have_obs = _has_table(conn, OBS_TABLE)
    have_prof = _has_table(conn, PROFILES_TABLE)
    have_sq = _has_table(conn, SQ_TABLE)
    stats["observations_table"] = "ok" if have_obs else "missing"
    stats["profiles_table"] = "ok" if have_prof else "missing"
    stats["standing_table"] = "ok" if have_sq else "missing"
    if not (have_obs or have_prof or have_sq):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_obs_tables"
        return out
    if not _has_table(conn, UNITS_TABLE):
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "units_table_missing"
        return out
    eligible_fn = _eligibility(getattr(ctx, "eligible", None))
    if eligible_fn is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "eligibility_handle_missing"
        return out

    # -- intent gate (V7-14.09) ------------------------------------------
    intent = getattr(qv, "intent", None)
    primary = getattr(intent, "primary", None)
    classes = set(getattr(intent, "classes", ()) or ())
    if primary is not None:
        classes.add(primary)
    if classes and classes <= _IDENTIFIER_ONLY:
        out.status = LaneStatus.SKIPPED
        out.reason = "intent_identifier"
        return out
    if primary in _TEMPORAL_INTENTS or (classes and classes <= _TEMPORAL_INTENTS):
        out.status = LaneStatus.SKIPPED
        out.reason = "intent_temporal"
        return out

    cap = max(0, int(getattr(slice, "cap", 0) or 0))
    if cap == 0:
        stats["mode"] = "capped"
        return out
    if expired():
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
        stats["mode"] = "none"
        return out

    terms = _query_terms(qv)
    entity_canons = _entity_canons(qv)
    obs_canons = _subject_canons(qv)
    norm_q = _norm_query_text(qv)
    pool: dict[str, _Cand] = {}
    truncated = False
    deadline_hit = False
    modes: list[str] = []
    stats["ineligible"] = 0
    stats["no_support"] = 0

    def _pool_add(
        unit_id: str,
        urow: tuple,
        *,
        obs: Optional[_ObsContribution] = None,
        profile: Optional[_ProfileContribution] = None,
        standing: Optional[_StandingContribution] = None,
        score: float = 1.0,
    ) -> None:
        cand = pool.get(unit_id)
        if cand is None:
            cand = _Cand(
                unit_id=unit_id,
                source_id=urow[2] or "",
                revision=int(urow[3] or 0),
            )
            pool[unit_id] = cand
        if obs is not None:
            cand.obs.append(obs)
        if profile is not None:
            cand.profiles.append(profile)
        if standing is not None:
            cand.standing.append(standing)
        if score > cand.score:
            cand.score = score

    # -- phase 1: observations --------------------------------------------
    if have_obs:
        modes.append("obs")
        matched: list[tuple[tuple, tuple[str, ...]]] = []
        n = 0
        cur = conn.execute(
            "SELECT obs_id, slot, text, producer, proof_count,"
            " support_refs_json, contradict_refs_json, stale"
            " FROM observations_v7"
            # obs_id is minted once (Rule B): rows stay valid at/below
            # the pinned generation.
            " WHERE scope_id = ? AND generation <= ? ORDER BY obs_id",
            (ctx.scope_id, ctx.generation),
        )
        for row in cur:
            n += 1
            out.examined += 1
            if _obs_matches(row[1], row[2], row[5], terms, obs_canons):
                matched.append((row, _parse_refs(row[5])))
            if n % DEADLINE_CHECK_ROWS == 0 and expired():
                deadline_hit = True
                break
            if n >= OBS_ROW_LIMIT:
                truncated = True
                break
        stats["obs_matched"] = len(matched)

        # resolve every referenced support unit through the fence, then
        # apply eligibility inside production (V7-05.08): the observation
        # emits iff at least one support unit is eligible — closure
        # membership made executable (V7-14.08).
        if matched and not deadline_hit:
            want = sorted({ref for _row, refs in matched for ref in refs})
            urows = _load_units(conn, ctx.scope_id, ctx.generation, want)
            out.examined += len(urows)
            for i, (row, refs) in enumerate(matched):
                if not refs:
                    stats["no_support"] += 1
                    continue
                backing = None
                for ref in refs:  # first eligible support, in pin order
                    urow = urows.get(ref)
                    if urow is None:
                        continue
                    if eligible_fn(_unit_row_dict(urow)):
                        backing = urow
                        break
                if backing is None:
                    stats["ineligible"] += 1
                    continue
                obs_id, slot, text, producer, proof, _refs, contra, stale = row
                proof_i = max(0, int(proof or 0))
                contrib = _ObsContribution(
                    obs_id=obs_id,
                    slot=slot or "",
                    proof=proof_i,
                    stale=1 if stale else 0,
                    n_support=len(refs),
                    n_contradict=len(_parse_refs(contra)),
                    producer=producer or "",
                )
                _pool_add(
                    backing[1],
                    backing,
                    obs=contrib,
                    score=1.0 + W_PROOF * math.log1p(proof_i),
                )
                if i % DEADLINE_CHECK_ROWS == 0 and expired():
                    deadline_hit = True
                    break

    # -- phase 2: profiles (subject_canon IN entity_canons) ---------------
    # The match is on *folded* canons, so the scan is bounded by the scope's
    # profile rows rather than a byte-exact IN list — canons are canonical
    # (already folded) in a well-formed store, and the Python fold check
    # stays correct even when a stored row is not.
    if have_prof and entity_canons and not deadline_hit:
        modes.append("profiles")
        prof_rows: list[tuple] = []
        n = 0
        cur = conn.execute(
            "SELECT subject_canon, slot, value, status, support_refs_json"
            " FROM profiles_v7"
            # V7-30.02: ``generation <= pinned``; profiles are versioned
            # (scope, subject_canon, slot, generation) — the newest row
            # per slot is authoritative (ORDER BY generation DESC,
            # first-seen per (subject, slot) kept).
            " WHERE scope_id = ? AND generation <= ?"
            " ORDER BY subject_canon, slot, generation DESC",
            (ctx.scope_id, ctx.generation),
        )
        seen_slots: set = set()
        for row in cur:
            n += 1
            out.examined += 1
            key = (row[0], row[1])
            if key in seen_slots:
                continue  # older-generation projection of the same slot
            seen_slots.add(key)
            if _fold(row[0]) in entity_canons:
                prof_rows.append(row)
            if n % DEADLINE_CHECK_ROWS == 0 and expired():
                deadline_hit = True
                break
            if n >= PROFILE_ROW_LIMIT:
                truncated = True
                break
        stats["profiles_matched"] = len(prof_rows)

        if prof_rows and not deadline_hit:
            parsed = [(row, _parse_refs(row[4])) for row in prof_rows]
            want = sorted({refs[0] for _row, refs in parsed if refs})
            urows = _load_units(conn, ctx.scope_id, ctx.generation, want)
            out.examined += len(urows)
            for i, (row, refs) in enumerate(parsed):
                if not refs:
                    stats["no_support"] += 1
                    continue
                # the first support pin is the candidate's unit (§30
                # contract); if it is ineligible the row emits nothing —
                # the artifact stands on its first pin or not at all.
                urow = urows.get(refs[0])
                if urow is None or not eligible_fn(_unit_row_dict(urow)):
                    stats["ineligible"] += 1
                    continue
                _pool_add(
                    refs[0],
                    urow,
                    profile=_ProfileContribution(
                        subject=row[0] or "",
                        slot=row[1] or "",
                        value=row[2],
                        status=row[3],
                    ),
                    score=1.0,
                )
                if i % DEADLINE_CHECK_ROWS == 0 and expired():
                    deadline_hit = True
                    break

    # -- phase 3: standing queries (exact normalized-query equality) ------
    if have_sq and norm_q and not deadline_hit:
        modes.append("standing")
        n = 0
        seeded = 0
        cur = conn.execute(
            "SELECT sq_id, query, dirty, pack_blob FROM standing_queries"
            # sq_id is minted once (Rule B) — ``<=`` at/below the pin.
            " WHERE scope_id = ? AND generation <= ? ORDER BY sq_id",
            (ctx.scope_id, ctx.generation),
        )
        sq_rows: list[tuple] = []
        for row in cur:
            n += 1
            out.examined += 1
            if _fold(row[1] or "") == norm_q:
                sq_rows.append(row)
            if n % DEADLINE_CHECK_ROWS == 0 and expired():
                deadline_hit = True
                break
            if n >= SQ_ROW_LIMIT:
                truncated = True
                break
        stats["standing_matched"] = len(sq_rows)

        if sq_rows and not deadline_hit:
            want: list[str] = []
            per_sq: list[tuple[str, int, tuple[str, ...]]] = []
            for sq_id, _q, dirty, blob in sq_rows:
                ids = _pack_unit_ids(blob)[:SQ_SEED_LIMIT]
                per_sq.append((sq_id, 1 if dirty else 0, ids))
                want.extend(ids)
            urows = _load_units(conn, ctx.scope_id, ctx.generation, want)
            out.examined += len(urows)
            n_seed = 0
            for sq_id, dirty, ids in per_sq:
                for uid in ids:
                    urow = urows.get(uid)
                    if urow is None or not eligible_fn(_unit_row_dict(urow)):
                        continue
                    _pool_add(
                        uid,
                        urow,
                        standing=_StandingContribution(sq_id=sq_id, dirty=dirty),
                        score=1.0,
                    )
                    seeded += 1
                    n_seed += 1
                    if n_seed % DEADLINE_CHECK_ROWS == 0 and expired():
                        deadline_hit = True
                        break
                if deadline_hit:
                    break
            stats["standing_seeded"] = seeded

    stats["mode"] = "+".join(modes) if modes else "none"
    out.eligible = len(pool)

    # -- selection: fresh before stale, score desc, unit_id asc -----------
    final = sorted(pool.values(), key=lambda c: c.order_key())[:cap]
    stats["overflow"] = max(0, len(pool) - len(final))
    stats["scan_truncated"] = truncated
    if expired():
        deadline_hit = True

    for rank, cand in enumerate(final, start=1):
        signals: dict[str, Any] = {"stale": cand.stale}
        if cand.obs:
            primary = sorted(cand.obs, key=lambda o: (-o.proof, o.obs_id))[0]
            signals["obs_id"] = primary.obs_id
            signals["slot"] = primary.slot
            signals["proof_count"] = primary.proof
            signals["supports"] = primary.n_support
            signals["contradicts"] = primary.n_contradict
            if primary.producer:
                signals["producer"] = primary.producer
            if len(cand.obs) > 1:
                signals["obs_ids"] = sorted(o.obs_id for o in cand.obs)
        if cand.profiles:
            profs = sorted(cand.profiles, key=lambda p: (p.subject, p.slot))
            signals["profile_slot"] = profs[0].slot
            signals["value"] = profs[0].value
            signals["status"] = profs[0].status
            signals["profile_subject"] = profs[0].subject
            if len(profs) > 1:
                signals["profile_slots"] = [p.slot for p in profs]
        if cand.standing:
            sqs = sorted(cand.standing, key=lambda s: s.sq_id)
            signals["standing"] = sqs[0].sq_id
            if any(s.dirty for s in sqs):
                signals["standing_dirty"] = 1
            if len(sqs) > 1:
                signals["standing_ids"] = [s.sq_id for s in sqs]
        out.candidates.append(
            CandidateV7(
                unit_id=cand.unit_id,
                source_id=cand.source_id,
                revision=cand.revision,
                lane=LANE_NAME,
                rank=rank,
                raw_score=cand.score,
                signals=signals,
            )
        )

    if deadline_hit:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
    elif truncated:
        out.status = LaneStatus.PARTIAL
        out.reason = "scan_bound"
    return out


class ObsLane(LaneV7):
    """LaneV7 protocol wrapper for registry wiring (V7-05.01)."""

    name = LANE_NAME

    def run(self, ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice) -> LaneOutput:
        return lane_obs(ctx, query, slice)


# Lane modules self-register at import (lanes_base contract); the pipeline
# never imports this module itself — the registrar/importer seam is owned
# by the main-session integration.
from .lanes_base import register_lane  # noqa: E402

register_lane(LaneName.OBS, lane_obs)


__all__ = [
    "LANE_NAME",
    "LANE_VERSION",
    "OBS_TABLE",
    "PROFILES_TABLE",
    "SQ_TABLE",
    "W_PROOF",
    "ObsLane",
    "lane_obs",
]
