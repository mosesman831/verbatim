"""Candidate union: eligibility-first filtering + dependency-preserving dedup
(SPEC_V3 §28.01–28.02, §29.07).

The union merges per-lane hits into at most ``candidate_cap`` unique
candidates — but ONLY after every admission-time eligibility predicate has
run: authorization scope, quarantine, purge suppression, lifecycle state,
known-at cutoff, freshness requirement, and perspective visibility.
Oversampling then filtering is never claimed equivalent (V3-28.01).

Mandatory dependencies — conflict members, required context members, and
failure-evidence edges — expand *outside* the candidate cap and are never
pruned as ordinary duplicates (V3-28.02, V3-26.13): a claim in conflict
keeps its conflict member. Dependency objects pass the same eligibility
gates; a dependency that fails marks its parent group incomplete so the
pack stage omits it whole rather than shipping a stripped claim.

Provenance-family dedup (V3-29.07): objects sharing an evidence family
(``family_members`` / ``claim_evidence.family_id``) collapse to their
best-ranked member; independent support and contradictory interpretations
are retained — dependencies are never deduped.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from ...core.types import Condition, safe_json_loads
from ...core.types_v3 import QueryClass
from .. import candidates as _cand
from ..package import _context_items  # noqa: F401  (re-exported for pack.py)

# Claim lifecycle states admitted per query class (§28 eligibility;
# mirrors v2 _MODE_STATES but keyed by QueryClass).
CLASS_STATES: dict[QueryClass, frozenset] = {
    QueryClass.EXACT_LOOKUP: frozenset({"active", "disputed"}),
    QueryClass.CURRENT_STATE: frozenset({"active", "disputed"}),
    QueryClass.RELATIONSHIP: frozenset({"active", "disputed"}),
    QueryClass.CAUSE: frozenset({"active", "disputed"}),
    QueryClass.PROCEDURE: frozenset({"active", "disputed"}),
    QueryClass.FAILURE: frozenset({"active", "disputed"}),
    QueryClass.EXPLORATORY: frozenset({"active", "disputed"}),
    QueryClass.PAST_STATE: frozenset(
        {"active", "disputed", "superseded", "archived"}
    ),
    QueryClass.TIMELINE: frozenset(
        {"active", "disputed", "superseded", "archived"}
    ),
    QueryClass.ARCHIVE: frozenset(
        {"pending", "active", "disputed", "superseded", "rejected", "archived"}
    ),
}

_LENIENT_CLASSES = frozenset(
    {QueryClass.PAST_STATE, QueryClass.TIMELINE, QueryClass.ARCHIVE}
)

# Object kinds the pack stage can serialize.
_PACKAGEABLE = frozenset({
    "claim", "procedure", "episode", "observation", "prospective",
    "working_item", "environment_state", "transition",
})

# Quarantine states that hide an object (mirrors security.EXCLUDING_STATES;
# re-declared so the fallback path never imports a parallel module).
_EXCLUDING_QUARANTINE = frozenset({"pending", "suppressed"})


@dataclass
class UnionHit:
    """One admitted candidate or mandatory dependency."""

    object_kind: str
    object_id: str
    revision: int = 0
    recorded_from: int = 0
    state: str = "active"
    lane_ranks: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    is_dependency: bool = False
    family_id: Optional[str] = None
    freshness: str = "unknown"
    stale: bool = False
    applicability: Optional[str] = None
    cond_keys: tuple = ()
    time_compat: Optional[str] = None
    perspective_id: Optional[str] = None
    security_label_id: Optional[str] = None
    proof_count: int = 0
    # object-scope the row lives in (authorization partition)
    scope_id: str = ""

    @property
    def key(self) -> tuple:
        return (self.object_kind, self.object_id)


class CandidateUnion(dict):
    """``dict[(kind, id), UnionHit]`` plus union-stage diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.overflow = 0
        self.family_deduped = 0
        self.eligibility_dropped = 0
        self.warnings: list = []
        # parent key -> dependency object keys admitted into the union
        self.deps: dict = {}
        # parent claim key -> dependency object keys that failed eligibility
        self.incomplete: dict = {}
        # claim key -> required context span ids (pack resolves them)
        self.context_deps: dict = {}
        # claim key -> open conflict group id (pack assembles the closure)
        self.conflict_groups: dict = {}
        # group ids whose closure is incomplete -> omit whole (§30.06)
        self.broken_groups: set = set()


def _ph(n: int) -> str:
    return ",".join("?" * n)


# ---------------------------------------------------------------------------
# defensive security import (quarantine exclusion)
# ---------------------------------------------------------------------------


def _should_exclude(conn: sqlite3.Connection, kind: str, oid: str,
                    rev: int) -> bool:
    """Quarantine exclusion: ``security.should_exclude`` when the parallel
    module is present; the local quarantine-table check otherwise
    (V3-14.10). Fail closed on unexpected errors."""
    try:
        from ... import security as _security  # type: ignore
    except Exception:
        _security = None
    if _security is not None:
        try:
            return bool(_security.should_exclude(conn, kind, oid, rev))
        except Exception:
            pass  # fall back to the local table check below
    try:
        row = conn.execute(
            "SELECT state FROM quarantine"
            " WHERE object_kind = ? AND object_id = ? AND revision = ?",
            (kind, oid, rev),
        ).fetchone()
    except sqlite3.Error:
        return True  # no readable quarantine state — fail closed
    return row is not None and row[0] in _EXCLUDING_QUARANTINE


# Chunk bound for OR-chained ref/pair sets — keeps each statement far
# below SQLITE_MAX_VARIABLE_NUMBER (3 bound params per triple).
_REF_CHUNK = 150


def _excluded_refs(conn: sqlite3.Connection, refs) -> frozenset:
    """Batched ``_should_exclude``: the held subset of ``refs``.

    Prefers ``security.quarantine.excluded_refs``; mirrors the per-ref
    contract — any backend error or unreadable hold set treats every
    probed ref as held (fail closed); an empty hold table fast-path is
    the only early exit, matching the per-ref code.
    """
    triples = {(k, o, int(r)) for k, o, r in refs}
    if not triples:
        return frozenset()
    try:
        from ...security import quarantine as _quarantine  # type: ignore
    except Exception:
        _quarantine = None
    fn = getattr(_quarantine, "excluded_refs", None) if _quarantine else None
    if fn is not None:
        try:
            return frozenset(fn(conn, triples))
        except Exception:
            pass  # fall back to the local table check below
    try:
        live = conn.execute(
            "SELECT 1 FROM quarantine"
            " WHERE state IN ('pending','suppressed') LIMIT 1"
        ).fetchone()
        if live is None:
            return frozenset()
        held: set = set()
        ordered = sorted(triples)
        for i in range(0, len(ordered), _REF_CHUNK):
            part = ordered[i:i + _REF_CHUNK]
            where = " OR ".join(
                "(object_kind = ? AND object_id = ? AND revision = ?)"
                for _ in part
            )
            held.update(
                (r[0], r[1], int(r[2]))
                for r in conn.execute(
                    "SELECT object_kind, object_id, revision"
                    " FROM quarantine"
                    " WHERE state IN ('pending','suppressed')"
                    " AND (" + where + ")",
                    [v for t in part for v in t],
                ).fetchall()
            )
        return frozenset(held)
    except sqlite3.Error:
        return frozenset(triples)  # no readable holds — fail closed


class SnapshotCache:
    """Per-read-snapshot memo for the eligibility data plane.

    One recall runs inside one ``store.read()`` snapshot; the rows this
    memoizes cannot change under it, and the cache is bound to the
    LaneContext of that single recall — never carried into a later
    snapshot — so a mid-sequence revoke/quarantine/purge committed
    between recalls is always re-read. Per-query caching only: a caller
    reusing a LaneContext across writes must drop ``ctx.snap``.

    ``admit`` pins per-object admission verdicts — the decisive
    deduplication across lane paging, dependency pulls, and the final
    union pass, all of which ask the same question about the same
    snapshot.
    """

    __slots__ = (
        "conn", "_tables", "_cols", "admit", "_held",
        "_persp", "_persp_subj", "_claim_scope",
    )

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._tables: Optional[frozenset] = None
        self._cols: dict = {}
        self.admit: dict = {}
        self._held: dict = {}
        self._persp: dict = {}
        self._persp_subj: dict = {}
        self._claim_scope: dict = {}

    # -- schema probes (frozen while the snapshot is open) --------------
    def has_table(self, name: str) -> bool:
        """One ``sqlite_master`` scan per snapshot instead of a probe per
        call site — identical answers while the read snapshot is open."""
        if self._tables is None:
            self._tables = frozenset(
                r[0]
                for r in self.conn.execute(
                    "SELECT name FROM sqlite_master"
                    " WHERE type IN ('table','view')"
                )
            )
        return name in self._tables

    def columns(self, table: str) -> set:
        cols = self._cols.get(table)
        if cols is None:
            cols = _cand._columns(self.conn, table)
            self._cols[table] = cols
        return cols

    # -- quarantine holds -------------------------------------------------
    def held(self, kind: str, oid: str, rev: int) -> bool:
        ref = (kind, oid, int(rev))
        v = self._held.get(ref)
        if v is None:
            v = _should_exclude(self.conn, kind, oid, rev)
            self._held[ref] = v
        return v

    def held_refs(self, refs) -> frozenset:
        """Held subset of ``refs``; one batch pass fills the memo."""
        want = {(k, o, int(r)) for k, o, r in refs}
        missing = [r for r in want if r not in self._held]
        if missing:
            held = _excluded_refs(self.conn, missing)
            for r in missing:
                self._held[r] = r in held
        return frozenset(r for r in want if self._held[r])

    # -- perspectives -------------------------------------------------------
    def perspective_row(self, pid):
        """``(asserter, observer, audience_json)`` row or None."""
        if pid is None:
            return None
        if pid not in self._persp:
            self._persp[pid] = self.conn.execute(
                "SELECT asserter, observer, audience_json"
                " FROM perspectives WHERE perspective_id = ?",
                (pid,),
            ).fetchone()
        return self._persp[pid]

    def perspective_subjects(self, pid) -> set:
        if pid not in self._persp_subj:
            self._persp_subj[pid] = {
                r[0]
                for r in self.conn.execute(
                    "SELECT subject_id FROM perspective_subjects"
                    " WHERE perspective_id = ?",
                    (pid,),
                ).fetchall()
            }
        return self._persp_subj[pid]

    def prefetch_perspectives(self, pids) -> None:
        """One ``IN`` pass for the perspective rows a batch verdict will
        need; subject lists stay lazy (read only when ``rp.subjects``)."""
        missing = [
            p for p in dict.fromkeys(pids) if p and p not in self._persp
        ]
        for part in _cand._chunks(missing, _cand._IN_CHUNK):
            for row in self.conn.execute(
                "SELECT perspective_id, asserter, observer, audience_json"
                " FROM perspectives"
                f" WHERE perspective_id IN ({_ph(len(part))})",
                part,
            ).fetchall():
                self._persp[row[0]] = (row[1], row[2], row[3])
        for pid in missing:
            self._persp.setdefault(pid, None)

    # -- claims ----------------------------------------------------------------
    def claim_scopes(self, claim_ids) -> dict:
        """claim_id -> scope_id (None when absent) — batched + memoized."""
        missing = [
            c for c in dict.fromkeys(claim_ids)
            if c not in self._claim_scope
        ]
        for part in _cand._chunks(missing, _cand._IN_CHUNK):
            for cid, sid in self.conn.execute(
                "SELECT claim_id, scope_id FROM claims"
                f" WHERE claim_id IN ({_ph(len(part))})",
                part,
            ).fetchall():
                self._claim_scope[cid] = sid
        for cid in missing:
            self._claim_scope.setdefault(cid, None)
        return self._claim_scope


def snapshot_for(ctx: Any) -> SnapshotCache:
    """The LaneContext's per-snapshot cache, created on first use and
    bound to ``ctx.conn``. A LaneContext is built once per recall, so the
    memo can never answer a later query from a stale snapshot."""
    snap = getattr(ctx, "snap", None)
    if snap is None or getattr(snap, "conn", None) is not ctx.conn:
        snap = SnapshotCache(ctx.conn)
        try:
            ctx.snap = snap
        except Exception:
            pass
    return snap


# ---------------------------------------------------------------------------
# object-scope and revision resolution
# ---------------------------------------------------------------------------

_OBJECT_TABLES: dict[str, tuple] = {
    # kind: (table, id_col, rev_col_or_None, scope_col, has_recorded_bounds)
    "claim": ("claims", "claim_id", None, "scope_id", False),
    "procedure": ("procedures", "procedure_id", "revision", "scope_id", True),
    "episode": ("episodes", "episode_id", "revision", "scope_id", True),
    "observation": ("observations", "observation_id", "revision", "scope_id", True),
    "prospective": ("prospective_records", "record_id", "revision", "scope_id", True),
    "environment_state": ("environment_state", "key", None, "scope_id", False),
    "transition": ("transitions", "transition_id", None, "scope_id", False),
}


def _object_meta(conn: sqlite3.Connection, kind: str, oid: str,
                 known: Optional[int], scopes: set) -> Optional[tuple]:
    """``(scope_id, revision, recorded_from, state)`` at the known cutoff.

    ``environment_state`` is keyed (scope_id, key) — the same key may exist
    in several scopes — so rows outside the authorized set are discarded
    before picking a winner rather than after.
    """
    spec = _OBJECT_TABLES.get(kind)
    if spec is None:
        return None
    table, id_col, rev_col, scope_col, has_bounds = spec
    if not _cand._has_table(conn, table):
        return None
    cols = f"{scope_col}"
    if rev_col:
        cols += f", {rev_col}"
    if has_bounds:
        cols += ", recorded_from, recorded_until"
    state_col = {"procedure": "state", "prospective": "status"}.get(kind)
    if state_col:
        cols += f", {state_col}"
    rows = conn.execute(
        f"SELECT {cols} FROM {table} WHERE {id_col} = ?",
        (oid,),
    ).fetchall()
    if not rows:
        return None
    best: Optional[tuple] = None
    for row in rows:
        idx = 0
        scope_id = row[idx]; idx += 1
        if scope_id not in scopes:
            continue  # never leak a same-named object from another scope
        rev = int(row[idx]) if rev_col else 1
        if rev_col:
            idx += 1
        rf = ru = None
        if has_bounds:
            rf, ru = row[idx], row[idx + 1]; idx += 2
        state = row[idx] if state_col else "active"
        if has_bounds:
            if known is None:
                if ru is not None:
                    continue
            elif not (rf <= known and (ru is None or ru > known)):
                continue
        meta = (scope_id, rev, rf or 0, state)
        if best is None or rev > best[1]:
            best = meta
    return best


def _working_item_meta(conn: sqlite3.Connection, item_id: str,
                       now: int) -> Optional[tuple]:
    row = conn.execute(
        "SELECT ws.scope_id, ws.expires_us FROM working_set_items i"
        " JOIN working_sets ws ON ws.set_id = i.set_id"
        " WHERE i.item_id = ?",
        (item_id,),
    ).fetchone()
    if row is None or row[1] <= now:
        return None
    return (row[0], 1, 0, "active")


def _object_meta_many(conn: sqlite3.Connection, snap: SnapshotCache,
                      kind: str, oids: list, known: Optional[int],
                      scopes: set) -> dict:
    """``{(kind, oid): (scope_id, revision, recorded_from, state)}`` — the
    ``_object_meta`` batch form. Per-oid winner logic is identical: rows
    outside the authorized scope set are discarded before picking the
    highest-revision survivor."""
    spec = _OBJECT_TABLES.get(kind)
    if spec is None:
        return {}
    table, id_col, rev_col, scope_col, has_bounds = spec
    if not snap.has_table(table):
        return {}
    cols = f"{id_col}, {scope_col}"
    if rev_col:
        cols += f", {rev_col}"
    if has_bounds:
        cols += ", recorded_from, recorded_until"
    state_col = {"procedure": "state", "prospective": "status"}.get(kind)
    if state_col:
        cols += f", {state_col}"
    out: dict = {}
    for part in _cand._chunks(list(dict.fromkeys(oids)), _cand._IN_CHUNK):
        rows = conn.execute(
            f"SELECT {cols} FROM {table} WHERE {id_col} IN ({_ph(len(part))})",
            part,
        ).fetchall()
        for row in rows:
            idx = 0
            oid = row[idx]; idx += 1
            scope_id = row[idx]; idx += 1
            if scope_id not in scopes:
                continue  # never leak a same-named object from another scope
            rev = int(row[idx]) if rev_col else 1
            if rev_col:
                idx += 1
            rf = ru = None
            if has_bounds:
                rf, ru = row[idx], row[idx + 1]; idx += 2
            state = row[idx] if state_col else "active"
            if has_bounds:
                if known is None:
                    if ru is not None:
                        continue
                elif not (rf <= known and (ru is None or ru > known)):
                    continue
            meta = (scope_id, rev, rf or 0, state)
            key = (kind, oid)
            if key not in out or rev > out[key][1]:
                out[key] = meta
    return out


def _working_items_meta(conn: sqlite3.Connection, item_ids: list,
                        now: int) -> dict:
    """item_id -> (scope_id, revision, recorded_from, state) — batch form
    of ``_working_item_meta``; first matching live row wins per item."""
    if not item_ids:
        return {}
    out: dict = {}
    for part in _cand._chunks(list(dict.fromkeys(item_ids)), _cand._IN_CHUNK):
        for oid, scope_id, expires in conn.execute(
            "SELECT i.item_id, ws.scope_id, ws.expires_us"
            " FROM working_set_items i"
            " JOIN working_sets ws ON ws.set_id = i.set_id"
            f" WHERE i.item_id IN ({_ph(len(part))})",
            part,
        ).fetchall():
            if expires <= now:
                continue
            out.setdefault(oid, (scope_id, 1, 0, "active"))
    return out


def _observation_meta(conn: sqlite3.Connection, oids: list) -> dict:
    """observation_id -> (proof_count, perspective_id) — one pass for the
    two per-oid probes the admit path used to run."""
    if not oids:
        return {}
    out: dict = {}
    for part in _cand._chunks(list(dict.fromkeys(oids)), _cand._IN_CHUNK):
        for oid, pc, pid in conn.execute(
            "SELECT observation_id, proof_count, perspective_id"
            f" FROM observations WHERE observation_id IN ({_ph(len(part))})",
            part,
        ).fetchall():
            out[oid] = (pc, pid)
    return out


# ---------------------------------------------------------------------------
# claim eligibility (mirrors v2 _apply_eligibility, generalized)
# ---------------------------------------------------------------------------


def _resolve_claims(conn: sqlite3.Connection, claim_ids: list,
                    known: Optional[int]) -> dict:
    return _cand._latest_known_revisions(conn, claim_ids, known)


def _claim_states_ok(state: str, query_class: QueryClass) -> bool:
    return state in CLASS_STATES.get(
        query_class, CLASS_STATES[QueryClass.CURRENT_STATE]
    )


def _span_suppressed_claims(conn: sqlite3.Connection, snap: SnapshotCache,
                            store: Any, resolved: dict) -> set:
    """Claims whose evidence chain is suppressed or quarantined — withhold
    whole (§14, V3-14.10). Walks ``claim_evidence`` → ``spans`` →
    ``source_envelopes``: a hold on the span itself, its source revision,
    or the covering source envelope taints every dependent claim.

    The cascade's quarantine checks resolve through one batched
    ``held_refs`` pass per call instead of a row probe per span/source/
    envelope — verdicts are identical, the snapshot pins the reads."""
    pairs = [(cid, meta[0]) for cid, meta in resolved.items()]
    if not pairs:
        return set()
    where = " OR ".join("(claim_id=? AND revision=?)" for _ in pairs)
    flat = [v for pair in pairs for v in pair]
    rows = conn.execute(
        f"SELECT claim_id, span_id FROM claim_evidence WHERE {where}", flat
    ).fetchall()
    spans_of: dict = {}
    all_spans: list = []
    for cid, sid in rows:
        spans_of.setdefault(cid, []).append(sid)
        all_spans.append(sid)
    all_spans = list(dict.fromkeys(all_spans))
    if not all_spans:
        return set()
    held: set = set(_cand._suppressed(store, conn, "span", all_spans))

    span_meta = {
        sid: (src, rev)
        for sid, src, rev in conn.execute(
            f"SELECT span_id, source_id, revision FROM spans"
            f" WHERE span_id IN ({_ph(len(all_spans))})",
            all_spans,
        ).fetchall()
    }
    if snap.has_table("quarantine"):
        src_revs = set(span_meta.values())
        env_of: dict = {}
        if src_revs and snap.has_table("source_envelopes"):
            env_where = " OR ".join(
                "(source_id=? AND revision=?)" for _ in src_revs
            )
            env_rows = conn.execute(
                f"SELECT envelope_id, source_id, revision"
                f" FROM source_envelopes WHERE {env_where}",
                [v for pair_ in src_revs for v in pair_],
            ).fetchall()
            for eid, sid2, rev2 in env_rows:
                env_of.setdefault((sid2, rev2), []).append(eid)
        # one batched quarantine pass over span + source + envelope refs
        refs = {
            ("span", sid, rev) for sid, (src, rev) in span_meta.items()
        }
        refs |= {("source", src, rev) for (src, rev) in src_revs}
        for (src, rev), eids in env_of.items():
            refs |= {
                ("source_envelope", eid, rev) for eid in eids
            }
        qheld = snap.held_refs(refs)
        for sid, (src, rev) in span_meta.items():
            if sid not in held and ("span", sid, rev) in qheld:
                held.add(sid)
        spans_by_src: dict = {}
        for sid, meta in span_meta.items():
            spans_by_src.setdefault(meta, []).append(sid)
        for (src, rev), sids in spans_by_src.items():
            tainted = ("source", src, rev) in qheld or any(
                ("source_envelope", eid, rev) in qheld
                for eid in env_of.get((src, rev), ())
            )
            if tainted:
                held.update(sids)
    if not held:
        return set()
    return {
        cid for cid, sids in spans_of.items() if held & set(sids)
    }


def _claim_freshness(conn: sqlite3.Connection, pairs: list,
                    snap: Optional[SnapshotCache] = None) -> dict:
    """claim_id -> (freshness_class, stale) for resolved revisions."""
    if not pairs:
        return {}
    out: dict = {}
    cols = (
        snap.columns("claim_revisions") if snap is not None
        else _cand._columns(conn, "claim_revisions")
    )
    if "freshness" in cols:
        for part in _cand._chunks(list(pairs), _cand._PAIR_CHUNK):
            where = " OR ".join(
                "(claim_id=? AND revision=?)" for _ in part
            )
            flat = [v for pair in part for v in pair]
            for cid, freshness in conn.execute(
                "SELECT claim_id, freshness FROM claim_revisions"
                f" WHERE {where}",
                flat,
            ).fetchall():
                out[cid] = [freshness or "unknown", False]
    else:
        for cid, _r in pairs:
            out[cid] = ["unknown", False]
    has_fresh = (
        snap.has_table("freshness") if snap is not None
        else _cand._has_table(conn, "freshness")
    )
    if has_fresh:
        ledger: dict = {}
        for part in _cand._chunks(list(pairs), _cand._PAIR_CHUNK):
            where = " OR ".join(
                "(object_id=? AND revision=?)" for _ in part
            )
            flat = [v for pair in part for v in pair]
            for cid, rev, cls, anchors_json in conn.execute(
                "SELECT object_id, revision, class, anchor_refs_json"
                " FROM freshness"
                " WHERE object_kind='claim' AND (" + where + ")",
                flat,
            ).fetchall():
                ledger[(cid, rev)] = (cls, anchors_json)
        for cid, rev in pairs:
            row = ledger.get((cid, rev))
            if row is None:
                continue
            cls, anchors = row[0], safe_json_loads(row[1]) or []
            stale = any(
                isinstance(a, dict) and a.get("stale_since_seq") is not None
                for a in anchors
            )
            # a ledger row outranks the column default
            out[cid] = [cls or out.get(cid, ["unknown"])[0], stale]
    return {cid: tuple(v) for cid, v in out.items()}


def _freshness_ok(cls: str, stale: bool, required: bool, now: int,
                  revalidate_after: Optional[int] = None) -> bool:
    """freshness_required semantics: drop known-bad freshness.

    ``unknown`` passes — absent metadata is not evidence of staleness; a
    caller needing guaranteed freshness uses the verify route instead.
    """
    if not required:
        return True
    if stale or cls == "volatile":
        return False
    if cls == "revalidate_after" and revalidate_after is not None:
        return revalidate_after >= now
    return True


def _perspective_ok(snap: SnapshotCache, perspective_id: Optional[str],
                    request_perspective: Any) -> bool:
    """Explicit-role perspective visibility (§04.09, §27.01).

    Every specified role must match the recorded perspective; an
    unrecorded role fails an explicit filter (None = not recorded, never
    "everyone"). No filter requested → visible. Perspective rows and
    subject lists memoize on the per-snapshot cache.
    """
    rp = request_perspective
    if rp is None:
        return True
    if perspective_id is None:
        return False
    row = snap.perspective_row(perspective_id)
    if row is None:
        return False
    rec_asserter, rec_observer, rec_audience = (
        row[0], row[1], safe_json_loads(row[2]) or []
    )
    if rp.asserter is not None and rec_asserter != rp.asserter:
        return False
    if rp.observer is not None and rec_observer != rp.observer:
        return False
    if rp.subjects:
        subs = snap.perspective_subjects(perspective_id)
        if not (subs & set(rp.subjects)):
            return False
    if rp.audience:
        if not (set(rec_audience) & set(rp.audience)):
            return False
    return True


def _claim_perspective_ids(conn: sqlite3.Connection, pairs: list,
                           snap: Optional[SnapshotCache] = None) -> dict:
    if not pairs:
        return {}
    cols = (
        snap.columns("claim_revisions") if snap is not None
        else _cand._columns(conn, "claim_revisions")
    )
    if "perspective_id" not in cols:
        return {}
    out: dict = {}
    for part in _cand._chunks(list(pairs), _cand._PAIR_CHUNK):
        where = " OR ".join("(claim_id=? AND revision=?)" for _ in part)
        flat = [v for pair in part for v in pair]
        out.update(
            conn.execute(
                "SELECT claim_id, perspective_id FROM claim_revisions"
                f" WHERE {where}",
                flat,
            ).fetchall()
        )
    return out


def _claim_label_ids(conn: sqlite3.Connection, pairs: list,
                     snap: Optional[SnapshotCache] = None) -> dict:
    if not pairs:
        return {}
    cols = (
        snap.columns("claim_revisions") if snap is not None
        else _cand._columns(conn, "claim_revisions")
    )
    if "security_label_id" not in cols:
        return {}
    out: dict = {}
    for part in _cand._chunks(list(pairs), _cand._PAIR_CHUNK):
        where = " OR ".join("(claim_id=? AND revision=?)" for _ in part)
        flat = [v for pair in part for v in pair]
        out.update(
            conn.execute(
                "SELECT claim_id, security_label_id FROM claim_revisions"
                f" WHERE {where}",
                flat,
            ).fetchall()
        )
    return out


def _eval_conditions(conn: sqlite3.Connection, hits: dict,
                     context: dict) -> None:
    """Three-valued condition verdicts on resolved revisions (V2-17)."""
    pairs = [
        (cid, h.revision) for cid, h in hits.items() if h.object_kind == "claim"
    ]
    if not pairs:
        return
    where = " OR ".join("(claim_id=? AND revision=?)" for _ in pairs)
    flat = [v for pair in pairs for v in pair]
    rows = conn.execute(
        f"SELECT claim_id, condition_json FROM claim_revisions WHERE {where}",
        flat,
    ).fetchall()
    for cid, cond_json in rows:
        hit = hits[cid]
        verdict = "applies"
        if cond_json:
            try:
                cond = Condition.from_json(safe_json_loads(cond_json))
                hit.cond_keys = cond.required_keys()
                value = cond.evaluate(context)
            except Exception:
                cond = None
                value = None
                verdict = "unknown"
            if cond is not None:
                verdict = (
                    "applies" if value is True
                    else "does_not_apply" if value is False
                    else "unknown"
                )
        hit.applicability = verdict


# ---------------------------------------------------------------------------
# dependency expansion (V3-28.02)
# ---------------------------------------------------------------------------


def _conflict_deps(conn: sqlite3.Connection, snap: SnapshotCache,
                   claim_ids: list) -> dict:
    """claim_id -> (group_id, [member claim_ids]) over OPEN groups."""
    if not claim_ids or not (
        snap.has_table("conflict_members")
        and snap.has_table("conflict_groups")
    ):
        return {}
    rows = conn.execute(
        "SELECT cm.claim_id, cm.group_id FROM conflict_members cm"
        " JOIN conflict_groups g ON g.group_id = cm.group_id"
        f" WHERE cm.claim_id IN ({_ph(len(claim_ids))}) AND g.status='open'",
        claim_ids,
    ).fetchall()
    touched: dict = {}
    for cid, gid in rows:
        touched.setdefault(cid, gid)
    if not touched:
        return {}
    gids = sorted(set(touched.values()))
    members: dict = {}
    for gid, cid in conn.execute(
        "SELECT group_id, claim_id FROM conflict_members"
        f" WHERE group_id IN ({_ph(len(gids))})",
        gids,
    ).fetchall():
        members.setdefault(gid, []).append(cid)
    return {
        cid: (gid, members.get(gid, [])) for cid, gid in touched.items()
    }


def _terminal_dep_states(conn: sqlite3.Connection, snap: SnapshotCache,
                         dep_keys: dict) -> dict:
    """dep key -> True when the counterparty's head is terminal —
    superseded/rejected/archived. Terminal deps are resolved history:
    their absence is the correct outcome of a lifecycle decision, not a
    dependency failure. Missing heads and live states are not terminal."""
    claim_ids = sorted({
        oid for deps in dep_keys.values()
        for k, oid in deps if k == "claim"
    })
    if not claim_ids or not snap.has_table("claim_revisions"):
        return {}
    rows = conn.execute(
        "SELECT r.claim_id, r.state FROM claim_revisions r"
        " WHERE r.revision = ("
        "   SELECT MAX(revision) FROM claim_revisions"
        "   WHERE claim_id = r.claim_id)"
        f" AND r.claim_id IN ({_ph(len(claim_ids))})",
        claim_ids,
    ).fetchall()
    terminal = {"superseded", "rejected", "archived"}
    return {
        ("claim", cid): state in terminal for cid, state in rows
    }


def _context_deps(conn: sqlite3.Connection, snap: SnapshotCache,
                  pairs: list) -> dict:
    """claim_id -> required context span ids (pack resolves quoting)."""
    if not pairs or not (
        snap.has_table("context_members")
        and snap.has_table("context_groups")
    ):
        return {}
    where = " OR ".join("(claim_id=? AND revision=?)" for _ in pairs)
    flat = [v for pair in pairs for v in pair]
    span_rows = conn.execute(
        f"SELECT claim_id, span_id FROM claim_evidence WHERE {where}", flat
    ).fetchall()
    spans_of: dict = {}
    all_spans: list = []
    for cid, sid in span_rows:
        spans_of.setdefault(cid, []).append(sid)
        all_spans.append(sid)
    if not all_spans:
        return {}
    group_rows = conn.execute(
        "SELECT DISTINCT group_id, span_id FROM context_members"
        f" WHERE span_id IN ({_ph(len(all_spans))})",
        all_spans,
    ).fetchall()
    span_groups: dict = {}
    for gid, sid in group_rows:
        span_groups.setdefault(sid, []).append(gid)
    gids = sorted({g for gs in span_groups.values() for g in gs})
    if not gids:
        return {}
    required_spans = conn.execute(
        "SELECT group_id, span_id FROM context_members"
        f" WHERE group_id IN ({_ph(len(gids))}) AND required = 1",
        gids,
    ).fetchall()
    req_by_group: dict = {}
    for gid, sid in required_spans:
        req_by_group.setdefault(gid, set()).add(sid)
    deps: dict = {}
    for cid, sids in spans_of.items():
        needed: set = set()
        for sid in sids:
            for gid in span_groups.get(sid, ()):
                needed |= req_by_group.get(gid, set())
        needed -= set(sids)
        if needed:
            deps[cid] = needed
    return deps


def _failure_deps(conn: sqlite3.Connection, snap: SnapshotCache,
                  claim_ids: list, scope_ids: list,
                  edge_types: tuple = ("conflicts_with", "context_of",
                                       "corrects")) -> dict:
    """Evidence edges that must travel with a claim (§29: conflict /
    context / failure-evidence edges are not ordinary duplicates).

    ``supersedes`` is deliberately NOT in the default set: the edge exists
    only after the transition applied, so its target is terminal by
    construction — requiring it would mark every successor's group
    incomplete (a resolved conflict is not a failed dependency). The
    caller passes it through the optional-deps path instead.
    """
    if not claim_ids or not snap.has_table("edges"):
        return {}
    rows = conn.execute(
        "SELECT source_id, target_id, source_kind, target_kind FROM edges"
        f" WHERE edge_type IN ({_ph(len(edge_types))})"
        " AND retired_event IS NULL"
        f" AND scope_id IN ({_ph(len(scope_ids))})"
        f" AND ((source_kind='claim' AND source_id IN ({_ph(len(claim_ids))}))"
        f"  OR (target_kind='claim' AND target_id IN ({_ph(len(claim_ids))})))"
        " ORDER BY created_event, edge_id LIMIT 400",
        [*edge_types, *scope_ids, *claim_ids, *claim_ids],
    ).fetchall()
    deps: dict = {}
    claims = set(claim_ids)
    for source_id, target_id, sk, tk in rows:
        if sk == "claim" and source_id in claims and tk == "claim":
            deps.setdefault(source_id, set()).add(("claim", target_id))
        if tk == "claim" and target_id in claims and sk == "claim":
            deps.setdefault(target_id, set()).add(("claim", source_id))
    return deps


def _procedure_failure_refs(conn: sqlite3.Connection, snap: SnapshotCache,
                            proc_ids: list, scope_ids: list) -> dict:
    """procedure_id -> failure-mode evidence refs (§21.05, §29)."""
    if not proc_ids or not snap.has_table("procedures"):
        return {}
    out: dict = {}
    rows = conn.execute(
        "SELECT procedure_id, failure_modes_json FROM procedures"
        f" WHERE procedure_id IN ({_ph(len(proc_ids))})",
        proc_ids,
    ).fetchall()
    claim_ids: set = set()
    refs: dict = {}
    for pid, fm_json in rows:
        modes = safe_json_loads(fm_json or "[]") or []
        if isinstance(modes, dict):
            # compiler payload shape: {"modes": [...], "status": ...}
            modes = modes.get("modes") or []
        seen: set = set()
        for mode in modes:
            if not isinstance(mode, dict):
                continue
            for ref in (mode.get("evidence_refs") or []):
                if not isinstance(ref, str):
                    continue
                kind, _, oid = ref.partition(":")
                if kind == "claim" and oid:
                    seen.add(("claim", oid))
                    claim_ids.add(oid)
                elif ref in proc_ids:
                    continue
                else:
                    seen.add(("claim", ref))
                    claim_ids.add(ref)
        refs[pid] = seen
    # keep only refs that resolve to real authorized claims
    if claim_ids:
        real = {
            r[0]
            for r in conn.execute(
                "SELECT claim_id FROM claims"
                f" WHERE claim_id IN ({_ph(len(claim_ids))})"
                f" AND scope_id IN ({_ph(len(scope_ids))})",
                [*claim_ids, *scope_ids],
            ).fetchall()
        }
        for pid in refs:
            refs[pid] = {k for k in refs[pid] if k[1] in real}
    return {pid: v for pid, v in refs.items() if v}


# ---------------------------------------------------------------------------
# the union itself
# ---------------------------------------------------------------------------


def _prelim_score(hit: UnionHit, weights: dict) -> float:
    return sum(
        weights.get(lane, 0.0) / (60 + rank)
        for lane, rank in hit.lane_ranks.items()
        if rank > 0
    )


# ---------------------------------------------------------------------------
# admission eligibility — used by lanes BEFORE rank truncation (V3-28.01)
# ---------------------------------------------------------------------------


def admit_keys(ctx: Any, keys: list) -> dict:
    """Batched eligibility verdicts: ``{key: meta}`` for ADMITTED keys.

    Every admission predicate runs here: authorization scope, lifecycle
    state at the query class, quarantine exclusion, claim+span purge
    suppression, the ``known_at_seq`` cutoff, ``freshness_required``,
    perspective-role visibility, and three-valued condition applicability.

    Lanes call this BEFORE assigning ranks so an ineligible candidate can
    never consume rank capacity (V3-28.01); ``build_union`` re-verifies
    the merged set authoritatively. ``meta`` carries the resolved
    revision/state/recorded_from/scope/freshness/perspective/label fields
    the downstream stages need — no second resolution pass.

    Each stage resolves in ONE batched pass over the pending keys (scope
    fetch, resolution, suppression, quarantine, freshness, perspective,
    conditions, time) rather than a row probe per candidate. Verdicts
    memoize on ``ctx.snap`` — bound to this recall's read snapshot — so
    later pages/lanes/dependency pulls re-asking the same key cost a
    dict hit, and nothing ever answers a later snapshot from this memo.
    """
    conn = ctx.conn
    store = ctx.store
    snap = snapshot_for(ctx)
    scopes = set(ctx.scope_ids)
    known = ctx.known_at_seq
    now = _now()
    out: dict = {}
    if not keys:
        return out

    request_perspective = getattr(ctx.request, "perspective", None)
    freshness_required = bool(
        getattr(ctx.request, "freshness_required", False)
    )
    allowed_states = CLASS_STATES.get(
        ctx.query_class, CLASS_STATES[QueryClass.CURRENT_STATE]
    )
    lenient = ctx.query_class in _LENIENT_CLASSES

    # cached verdicts from earlier pages/lanes of THIS recall — the same
    # snapshot answers them identically.
    pending = list(dict.fromkeys(k for k in keys if k not in snap.admit))

    if pending:
        claim_ids = [oid for (k, oid) in pending if k == "claim"]
        work_ids = [oid for (k, oid) in pending if k == "working_item"]
        obj_ids: dict = {}
        for k, oid in pending:
            if k not in ("claim", "working_item"):
                obj_ids.setdefault(k, []).append(oid)

        # ---- claims ---------------------------------------------------
        resolved_claims = _resolve_claims(conn=conn, claim_ids=claim_ids,
                                        known=known)
        claim_scopes = snap.claim_scopes(claim_ids)
        suppressed_claims = _cand._suppressed(store, conn, "claim",
                                              claim_ids)
        resolvable = {
            cid: meta for cid, meta in resolved_claims.items()
            if meta[1] in allowed_states
            and cid not in suppressed_claims
            and claim_scopes.get(cid) in scopes
        }
        span_blocked = _span_suppressed_claims(conn, snap, store, resolvable)
        resolvable = {
            cid: meta for cid, meta in resolvable.items()
            if cid not in span_blocked
        }

        # ---- non-claim objects ----------------------------------------
        work_meta = _working_items_meta(conn, work_ids, now)
        work_meta = {
            oid: m for oid, m in work_meta.items() if m[0] in scopes
        }
        obj_meta: dict = {}
        for kind, oids in obj_ids.items():
            obj_meta.update(
                _object_meta_many(conn, snap, kind, oids, known, scopes)
            )
        obj_live = {
            key: om for key, om in obj_meta.items()
            if not (
                om[3] in ("retired", "deprecated", "cancelled")
                and not lenient
            )
        }

        # ---- one batched quarantine pass over every surviving ref -----
        refs = {
            ("claim", cid, m[0]) for cid, m in resolvable.items()
        }
        refs |= {("working_item", oid, 1) for oid in work_meta}
        refs |= {
            (k, oid, om[1]) for (k, oid), om in obj_live.items()
        }
        held = snap.held_refs(refs)
        resolvable = {
            cid: m for cid, m in resolvable.items()
            if ("claim", cid, m[0]) not in held
        }
        work_meta = {
            oid: m for oid, m in work_meta.items()
            if ("working_item", oid, 1) not in held
        }
        obj_live = {
            key: om for key, om in obj_live.items()
            if (key[0], key[1], om[1]) not in held
        }

        # purge suppression for surviving non-claim objects, batched per
        # kind (claims were handled above).
        obj_supp: dict = {}
        for kind in {k for k, _o in obj_live}:
            oids = [oid for (k, oid) in obj_live if k == kind]
            obj_supp[kind] = _cand._suppressed(store, conn, kind, oids)

        # ---- claim-side batch maps over the survivors -----------------
        pairs = [(cid, m[0]) for cid, m in resolvable.items()]
        fresh_map = _claim_freshness(conn, pairs, snap)
        persp_map = _claim_perspective_ids(conn, pairs, snap)
        label_map = _claim_label_ids(conn, pairs, snap)
        verdicts = _condition_verdicts(conn, pairs,
                                     _request_context(ctx.request))

        # valid-time eligibility (V2-26.03 semantics): provably-disjoint
        # intervals drop for strict classes; unknown stays as uncertain
        # evidence; lenient (past/timeline/archive) keeps it labeled.
        point = getattr(ctx.plan, "valid_at_us", None)
        until = getattr(ctx.plan, "valid_until_us", None)
        time_map: dict = {}
        if point is not None or until is not None:
            intervals_of = _cand._interval_rows(conn, pairs)
            for cid, _m in pairs:
                ivs = intervals_of.get(cid, [])
                if until is not None:
                    verdicts_t = [
                        _cand._range_applicability(iv, point, until)
                        for iv in ivs
                    ]
                else:
                    verdicts_t = [iv.applicability_at(point) for iv in ivs]
                if any(v is True for v in verdicts_t):
                    time_map[cid] = "compatible"
                elif not ivs or any(v is None for v in verdicts_t):
                    time_map[cid] = "unknown"
                else:
                    time_map[cid] = "inapplicable"

        # ---- observations: the proof_count + perspective columns ------
        obs_meta = _observation_meta(
            conn, [oid for (k, oid) in obj_live if k == "observation"]
        )

        # ---- object freshness, one batched pass ------------------------
        obj_fresh = _object_freshness_many(
            conn, snap,
            [(k, oid, om[1]) for (k, oid), om in obj_live.items()],
        )

        # ---- perspective prefetch when a filter is active --------------
        if request_perspective is not None:
            snap.prefetch_perspectives(
                [p for p in persp_map.values() if p]
                + [m[1] for m in obs_meta.values() if m and m[1]]
            )

        # ---- per-key verdict assembly (all in-memory) ------------------
        for key in pending:
            kind, oid = key
            meta: dict = {
                "revision": 0, "state": "active", "recorded_from": 0,
                "scope_id": "", "freshness": "unknown", "stale": False,
                "perspective_id": None, "security_label_id": None,
                "proof_count": 0, "applicability": None, "cond_keys": (),
                "time_compat": None,
            }
            verdict_meta: Optional[dict] = None
            if kind == "claim":
                resolved = resolvable.get(oid)
                if resolved is None:
                    snap.admit[key] = None
                    continue
                rev, state, rf = resolved
                cls, stale = fresh_map.get(oid, ("unknown", False))
                if not _freshness_ok(cls, stale, freshness_required, now):
                    snap.admit[key] = None
                    continue
                pid = persp_map.get(oid)
                if not _perspective_ok(snap, pid, request_perspective):
                    snap.admit[key] = None
                    continue
                verdict, cond_keys = verdicts.get(oid, ("applies", ()))
                if verdict == "does_not_apply" and not lenient:
                    snap.admit[key] = None
                    continue
                tverdict = time_map.get(oid)
                if tverdict == "inapplicable" and not lenient:
                    snap.admit[key] = None
                    continue
                meta.update(
                    revision=rev, state=state, recorded_from=rf,
                    scope_id=claim_scopes[oid], freshness=cls, stale=stale,
                    perspective_id=pid,
                    security_label_id=label_map.get(oid),
                    applicability=verdict if cond_keys else None,
                    cond_keys=cond_keys,
                    time_compat=tverdict,
                )
                verdict_meta = meta
            elif kind == "working_item":
                wm = work_meta.get(oid)
                if wm is not None:
                    meta.update(scope_id=wm[0], revision=wm[1],
                                recorded_from=wm[2], state=wm[3])
                    verdict_meta = meta
            else:
                om = obj_live.get(key)
                if om is None:
                    snap.admit[key] = None
                    continue
                scope_id, rev, rf, state = om
                if oid in obj_supp.get(kind, ()):
                    snap.admit[key] = None
                    continue
                if request_perspective is not None:
                    if kind != "observation":
                        # no recorded roles to satisfy an explicit filter
                        snap.admit[key] = None
                        continue
                    prow = obs_meta.get(oid)
                    if not _perspective_ok(
                        snap, prow[1] if prow else None,
                        request_perspective
                    ):
                        snap.admit[key] = None
                        continue
                    meta["perspective_id"] = prow[1] if prow else None
                fcls, fstale = obj_fresh.get(key, ("unknown", False))
                if not _freshness_ok(fcls, fstale, freshness_required, now):
                    snap.admit[key] = None
                    continue
                meta.update(
                    scope_id=scope_id, revision=rev, recorded_from=rf,
                    state=state, freshness=fcls, stale=fstale,
                )
                if kind == "observation":
                    orow = obs_meta.get(oid)
                    meta["proof_count"] = int(orow[0]) if orow else 0
                verdict_meta = meta
            snap.admit[key] = verdict_meta

    for key in keys:
        meta = snap.admit.get(key)
        if meta is not None:
            out[key] = meta
    return out


def _condition_verdicts(conn: sqlite3.Connection, pairs: list,
                        context: dict) -> dict:
    """claim_id -> (verdict, cond_keys) over resolved revisions."""
    if not pairs:
        return {}
    where = " OR ".join("(claim_id=? AND revision=?)" for _ in pairs)
    flat = [v for pair in pairs for v in pair]
    rows = conn.execute(
        f"SELECT claim_id, condition_json FROM claim_revisions WHERE {where}",
        flat,
    ).fetchall()
    out: dict = {}
    for cid, cond_json in rows:
        verdict = "applies"
        keys: tuple = ()
        if cond_json:
            try:
                cond = Condition.from_json(safe_json_loads(cond_json))
                keys = cond.required_keys()
                value = cond.evaluate(context)
            except Exception:
                cond = None
                value = None
                verdict = "unknown"
            if cond is not None:
                verdict = (
                    "applies" if value is True
                    else "does_not_apply" if value is False
                    else "unknown"
                )
        out[cid] = (verdict, keys)
    return out


def build_union(
    ctx: Any,
    lane_results: list,
    candidate_cap: int,
    lane_weights: dict,
) -> CandidateUnion:
    """Merge lane hits → eligible, deduped, dependency-expanded union.

    Order of operations is contractual (V3-28.01): merge → admit
    eligibility → dependency expansion → family dedup → cap. The cap
    applies only to ordinary candidates; mandatory dependencies expand
    outside it.
    """
    union = CandidateUnion()
    now = _now()
    scopes = set(ctx.scope_ids)
    known = ctx.known_at_seq
    snap = snapshot_for(ctx)

    # ---- 1. merge lane hits -------------------------------------------
    for res in lane_results:
        if res.status in ("unavailable", "skipped"):
            continue
        for key, rank in res.hits.items():
            kind, oid = key
            hit = union.get(key)
            if hit is None:
                hit = UnionHit(kind, oid)
                union[key] = hit
            prev = hit.lane_ranks.get(res.lane)
            if prev is None or rank < prev:
                hit.lane_ranks[res.lane] = rank
    if not union:
        return union

    # ---- 2. admission eligibility (BEFORE cap) -------------------------
    admitted = admit_keys(ctx, list(union.keys()))
    for key in list(union.keys()):
        meta = admitted.get(key)
        if meta is None:
            union.eligibility_dropped += 1
            del union[key]
            continue
        hit = union[key]
        hit.revision = meta["revision"]
        hit.state = meta["state"]
        hit.recorded_from = meta["recorded_from"]
        hit.scope_id = meta["scope_id"]
        hit.freshness = meta["freshness"]
        hit.stale = meta["stale"]
        hit.perspective_id = meta["perspective_id"]
        hit.security_label_id = meta["security_label_id"]
        hit.proof_count = meta["proof_count"]
        hit.applicability = meta["applicability"]
        hit.cond_keys = meta["cond_keys"]
        hit.time_compat = meta.get("time_compat")

    if not union:
        return union

    # ---- 4. mandatory dependencies -------------------------------------
    admitted_claims = [oid for (k, oid) in union if k == "claim"]
    conflict = _conflict_deps(ctx.conn, snap, admitted_claims)
    for cid, (gid, members) in conflict.items():
        union.conflict_groups[("claim", cid)] = gid
    failure = _failure_deps(ctx.conn, snap, admitted_claims, list(scopes))
    # Resolved supersession rides as OPTIONAL dependency evidence: the
    # predecessor is terminal by construction, so it can enrich a pack
    # (e.g. historical context on a PAST_STATE answer) but can never mark
    # its successor's group incomplete.
    supersession = _failure_deps(
        ctx.conn, snap, admitted_claims, list(scopes), ("supersedes",)
    )
    proc_failure = _procedure_failure_refs(
        ctx.conn, snap,
        [oid for (k, oid) in union if k == "procedure"],
        list(scopes),
    )
    pairs = [
        (oid, union[("claim", oid)].revision)
        for oid in admitted_claims if ("claim", oid) in union
    ]
    union.context_deps = _context_deps(ctx.conn, snap, pairs)

    dep_keys: dict = {}  # parent key -> set(dep keys)
    for cid, (gid, members) in conflict.items():
        dep_keys.setdefault(("claim", cid), set()).update(
            ("claim", m) for m in members if m != cid
        )
    for cid, deps in failure.items():
        dep_keys.setdefault(("claim", cid), set()).update(deps)
    for pid, deps in proc_failure.items():
        dep_keys.setdefault(("procedure", pid), set()).update(deps)

    # admit dependencies through the SAME eligibility gates (V3-28.02):
    # a dependency that fails marks its parent incomplete rather than
    # silently dropping.
    all_dep_keys = sorted({
        d for deps in dep_keys.values() for d in deps
        if d[0] in _PACKAGEABLE and d not in union
    })
    dep_meta = admit_keys(ctx, all_dep_keys)
    for key, meta in dep_meta.items():
        kind, oid = key
        union[key] = UnionHit(
            kind, oid,
            revision=meta["revision"],
            recorded_from=meta["recorded_from"],
            state=meta["state"],
            is_dependency=True,
            scope_id=meta["scope_id"],
            freshness=meta["freshness"],
            stale=meta["stale"],
            perspective_id=meta["perspective_id"],
            security_label_id=meta["security_label_id"],
            proof_count=meta["proof_count"],
        )

    # Terminal counterparties are resolved history, not missing deps: a
    # claim that was superseded/rejected/archived legitimately fails
    # admission, and its absence must not mark the survivor's group
    # incomplete (§30.06 guards *live* required evidence). Only deps that
    # are missing or unauthorized — head absent or still live but gated —
    # break the group.
    terminal_dep = _terminal_dep_states(ctx.conn, snap, dep_keys)
    for parent, deps in dep_keys.items():
        if parent not in union:
            continue
        failed = {
            d for d in deps
            if d[0] in _PACKAGEABLE and d not in union
            and not terminal_dep.get(d)
        }
        if failed:
            union.incomplete[parent] = failed
            gid = union.conflict_groups.get(parent)
            if gid is not None:
                union.broken_groups.add(gid)
        # dependencies that resolved join the union flagged is_dependency
        admitted = set()
        for d in deps:
            dh = union.get(d)
            if dh is not None:
                dh.is_dependency = True
                admitted.add(d)
        if admitted:
            union.deps[parent] = admitted

    # Optional supersession deps: admitted through the same gates, shipped
    # when they resolve, never counted toward group completeness.
    opt_dep_keys = sorted({
        d for deps in supersession.values() for d in deps
        if d[0] in _PACKAGEABLE and d not in union
    })
    for key, meta in admit_keys(ctx, opt_dep_keys).items():
        kind, oid = key
        union[key] = UnionHit(
            kind, oid,
            revision=meta["revision"],
            recorded_from=meta["recorded_from"],
            state=meta["state"],
            is_dependency=True,
            scope_id=meta["scope_id"],
            freshness=meta["freshness"],
            stale=meta["stale"],
            perspective_id=meta["perspective_id"],
            security_label_id=meta["security_label_id"],
            proof_count=meta["proof_count"],
        )
    for parent, deps in supersession.items():
        if parent not in union:
            continue
        admitted = set()
        for d in deps:
            dh = union.get(d)
            if dh is not None:
                dh.is_dependency = True
                admitted.add(d)
        if admitted:
            union.deps.setdefault(parent, set()).update(admitted)

    # ---- 5. provenance-family dedup ------------------------------------
    fam_of = _families(ctx.conn, snap, union)
    families: dict = {}
    for key, hit in union.items():
        fid = fam_of.get(key)
        if fid:
            hit.family_id = fid
            if not hit.is_dependency:
                families.setdefault(fid, []).append(key)
    for fid, members in families.items():
        if len(members) <= 1:
            continue
        members.sort(
            key=lambda k: (
                -_prelim_score(union[k], lane_weights), k[0], k[1]
            )
        )
        for key in members[1:]:
            union.family_deduped += 1
            del union[key]

    # ---- 6. cap (non-dependency candidates only) -----------------------
    ordinary = [k for k, h in union.items() if not h.is_dependency]
    if len(ordinary) > candidate_cap:
        ordinary.sort(
            key=lambda k: (
                -_prelim_score(union[k], lane_weights),
                -union[k].recorded_from,
                k[0], k[1],
            )
        )
        keep = set(ordinary[:candidate_cap])
        union.overflow = len(ordinary) - candidate_cap
        union.warnings.append("candidate_overflow")
        for key in list(union.keys()):
            if key not in keep and not union[key].is_dependency:
                del union[key]
    return union


def _object_freshness(conn: sqlite3.Connection, kind: str, oid: str,
                      rev: int) -> tuple:
    """(class, stale) for a non-claim object (freshness ledger + columns)."""
    cls = "unknown"
    if kind in ("procedure", "observation") and _cand._has_table(
        conn, {"procedure": "procedures", "observation": "observations"}[kind]
    ):
        table = {"procedure": "procedures", "observation": "observations"}[kind]
        col = {"procedure": "procedure_id", "observation": "observation_id"}[kind]
        if "freshness" in _cand._columns(conn, table):
            row = conn.execute(
                f"SELECT freshness FROM {table} WHERE {col} = ?", (oid,)
            ).fetchone()
            cls = (row[0] if row else None) or "unknown"
    stale = False
    if _cand._has_table(conn, "freshness"):
        row = conn.execute(
            "SELECT class, revalidate_after_us, anchor_refs_json"
            " FROM freshness WHERE object_kind = ? AND object_id = ?"
            " AND revision = ?",
            (kind, oid, rev),
        ).fetchone()
        if row is not None:
            cls = row[0] or cls
            anchors = safe_json_loads(row[2]) or []
            stale = any(
                isinstance(a, dict) and a.get("stale_since_seq") is not None
                for a in anchors
            )
            if (
                cls == "revalidate_after"
                and row[1] is not None
                and row[1] < _now()
            ):
                stale = True
    return cls, stale


def _object_freshness_many(conn: sqlite3.Connection, snap: SnapshotCache,
                           triples: list) -> dict:
    """``{(kind, oid): (class, stale)}`` — the ``_object_freshness`` batch
    form: column freshness for procedures/observations in one pass, then
    one chunked freshness-ledger pass over every ``(kind, oid, rev)``."""
    out: dict = {}
    if not triples:
        return out
    cols_of = {
        "procedure": ("procedures", "procedure_id"),
        "observation": ("observations", "observation_id"),
    }
    base: dict = {}
    by_kind: dict = {}
    for kind, oid, rev in triples:
        by_kind.setdefault(kind, []).append(oid)
    for kind, oids in by_kind.items():
        spec = cols_of.get(kind)
        if spec is None:
            continue
        table, col = spec
        if not (snap.has_table(table) and "freshness" in snap.columns(table)):
            continue
        for part in _cand._chunks(oids, _cand._IN_CHUNK):
            for oid, fr in conn.execute(
                f"SELECT {col}, freshness FROM {table}"
                f" WHERE {col} IN ({_ph(len(part))})",
                part,
            ).fetchall():
                base[(kind, oid)] = fr
    ledger: dict = {}
    if snap.has_table("freshness"):
        ordered = sorted(set(triples))
        for i in range(0, len(ordered), _REF_CHUNK):
            part = ordered[i:i + _REF_CHUNK]
            where = " OR ".join(
                "(object_kind=? AND object_id=? AND revision=?)"
                for _ in part
            )
            for okind, oid, rev, cls, ra, anchors_json in conn.execute(
                "SELECT object_kind, object_id, revision, class,"
                " revalidate_after_us, anchor_refs_json"
                " FROM freshness WHERE " + where,
                [v for t in part for v in t],
            ).fetchall():
                ledger[(okind, oid, int(rev))] = (cls, ra, anchors_json)
    now = _now()
    for kind, oid, rev in triples:
        cls = base.get((kind, oid)) or "unknown"
        stale = False
        row = ledger.get((kind, oid, int(rev)))
        if row is not None:
            cls = row[0] or cls
            anchors = safe_json_loads(row[2]) or []
            stale = any(
                isinstance(a, dict) and a.get("stale_since_seq") is not None
                for a in anchors
            )
            if (
                cls == "revalidate_after"
                and row[1] is not None
                and row[1] < now
            ):
                stale = True
        out[(kind, oid)] = (cls, stale)
    return out


def _families(conn: sqlite3.Connection, snap: SnapshotCache,
              union: CandidateUnion) -> dict:
    """(kind, id) -> family_id for evidence-family dedup (V3-29.07).

    Batched: one grouped pass over ``family_members`` per object kind and
    one ``claim_evidence`` pass for uncovered claims — first row per key
    wins, matching the prior ``LIMIT 1`` probes."""
    out: dict = {}
    has_fm = snap.has_table("family_members")
    fam_col = "family_id" in snap.columns("claim_evidence")
    if not has_fm and not fam_col:
        return out
    # family_members covers object kinds uniformly
    if has_fm:
        by_kind: dict = {}
        for kind, oid in union.keys():
            by_kind.setdefault(kind, []).append(oid)
        for kind, oids in by_kind.items():
            for part in _cand._chunks(
                sorted(set(oids)), _cand._IN_CHUNK
            ):
                for oid, fid in conn.execute(
                    "SELECT object_id, family_id FROM family_members"
                    f" WHERE object_kind = ? AND object_id IN ({_ph(len(part))})",
                    [kind, *part],
                ).fetchall():
                    out.setdefault((kind, oid), fid)
    if fam_col:
        claim_ids = sorted({
            oid for (k, oid), h in union.items()
            if k == "claim" and (k, oid) not in out
        })
        for part in _cand._chunks(claim_ids, _cand._IN_CHUNK):
            for cid, fid in conn.execute(
                "SELECT claim_id, family_id FROM claim_evidence"
                f" WHERE claim_id IN ({_ph(len(part))})"
                " AND family_id IS NOT NULL",
                part,
            ).fetchall():
                out.setdefault(("claim", cid), fid)
    return out


def _request_context(request: Any) -> dict:
    """Typed condition context: explicit request context wins; the v3 task
    environment fingerprint contributes platform/repo keys."""
    ctx = getattr(request, "context", None)
    if isinstance(ctx, dict):
        return ctx
    task = getattr(request, "task", None)
    env = getattr(task, "environment", None)
    out: dict = {}
    if env is not None:
        if getattr(env, "platform", None):
            out["platform"] = env.platform
        if getattr(env, "repo_id", None):
            out["repo_id"] = env.repo_id
        for name, ver in getattr(env, "runtime_versions", ()):
            out[f"runtime:{name}"] = ver
        for name, ver in getattr(env, "tool_schema_versions", ()):
            out[f"tool:{name}"] = ver
    if getattr(request, "context", None):
        out.update(request.context)
    return out


def _now() -> int:
    from ...core.time import now_us

    return now_us()
