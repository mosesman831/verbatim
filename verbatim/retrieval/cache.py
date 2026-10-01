"""Final-pack recall cache (SPEC_V4 §33, V4-33.01–33.08).

An in-process, per-store, capacity-bounded LRU over delivered
``RecallResultV3`` packs. It is a performance layer ONLY — it is never an
authorization decision, never a delivery permit, and never persists:

* **Key** (V4-33.01): canonicalized request (normalized query, scope,
  caller, purpose, modes/kinds, temporal and eligibility fields, budgets,
  wire version) + the authorized scope-set digest + the per-scope
  ``quote``-verb outcome vector + the resolved detail tier + the
  controller/lane configuration bits that shape the route set.
* **Fingerprint** (checked on every probe): the per-scope authorization
  epoch vector, store policy/erasure epochs, projection generation,
  schema version, a split content/journal watermark (``COUNT(*)`` +
  ``MAX(rowid)`` per table — journal tables are the excluded
  delivery/ledger set the recall itself writes), ``PRAGMA
  data_version``, and a cache-local ``mutation_epoch``.
* **Journal restamp**: commits to journal tables (the recall's own
  write phase among them) move ``data_version`` + the journal watermark
  without touching the content part — the entry is re-stamped at the
  current fingerprint rather than invalidated, so delivery traffic does
  not self-invalidate. A ``data_version`` bump with NO watermark moving
  is an in-place UPDATE somewhere: unattributable → ``mutation_epoch``
  bumps and every stored entry invalidates on its next probe. Honest
  bound: an in-place UPDATE co-committed in the same transaction as a
  journal-table insert cannot be attributed either — the per-ref
  revalidation below (suppression/lifecycle/revision) is the guard for
  delivered objects in that window.
* **Hit discipline** (V4-33.02): a matching key+fingerprint re-validates
  every delivered object reference against the CURRENT snapshot —
  quarantine holds, purge suppression, the span→source→envelope cascade,
  and claim head-revision drift — before any cached byte ships. Failing
  any check invalidates the entry (counted, evicted).
* **Expiry bound**: time-decaying inputs (working-set expiry, and
  ``revalidate_after`` freshness when the request requires it) pin a
  ``not_after_us`` on the entry; past it the entry is a miss.
* **Isolation** (V4-33.08): the caller id, purpose, verb outcomes, and
  scope set are all key material — callers never cross-serve.
* **Honesty** (V4-33.07): ``stats()`` reports real hits, misses,
  invalidations, expirations, ref-revalidation failures, stores, and LRU
  evictions. Abstentions and degraded/empty results are never cached.

Only exact-normalized requests share entries (V4-33.04): whitespace and
case are canonicalized, nothing more — there is no fuzzy/semantic query
equivalence, so current-state, negated, and identifier queries can never
reuse a different question's answer.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, safe_json_loads

from . import candidates as _cand


# ---------------------------------------------------------------------------
# content watermark
# ---------------------------------------------------------------------------

# Tables whose writes can never alter delivered recall content: the job
# queue, write-phase journals (the recall's own decision/influence rows
# would otherwise self-invalidate), receipt/ledger surfaces, projection-
# build machinery (the generation fingerprint covers its effects), and
# capability-token tables the read path never consults. Everything else —
# every content, authorization, suppression, and index table — is
# watermarked, so an unanticipated mutation invalidates rather than
# silently serving stale bytes.
_WATERMARK_EXCLUDED = frozenset({
    "jobs", "job_events",
    "routing_decisions", "routing_stats", "decision_inputs", "decisions",
    "influence", "propagations", "disclosures", "feedback",
    # ``source_exposure`` is the consumer search's own delivery ledger
    # (docs/v6_contracts §8) — the source-side twin of ``influence``;
    # without it every cached consumer result would self-invalidate on
    # the emission that follows the delivery. ``policy_artifact_``
    # ``attestations`` is created by the same additive ensure and is an
    # attestation journal the read path never consults (same class as
    # ``policy_artifacts`` below).
    "source_exposure", "policy_artifact_attestations",
    "operation_receipts", "outcome_receipts", "budget_ledger",
    "readiness_obligations", "ingest_batches",
    "migration_history", "schema_operations", "sqlite_sequence",
    "projection_outbox", "projection_builds", "active_projections",
    "closure_runs", "closure_frontier",
    "replay_runs", "learning_snapshots", "usage_aggregates",
    "action_tickets", "ticket_objects", "value_handles",
    "vault_entries", "vault_refs",
    "dispatch_permits", "delivery_permits", "capture_authorizations",
    "connector_cursors", "producer_manifests", "encoder_manifests",
    "artifacts", "artifact_links", "policy_artifacts", "embedding_inputs",
})


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _watermarks(conn: sqlite3.Connection) -> tuple:
    """``(content_digest, journal_digest)`` over (count, max rowid) per
    table.

    * *content* covers every table NOT in ``_WATERMARK_EXCLUDED`` — the
      recall-visible surface (claims, revisions, spans, sources, scopes,
      grants, quarantine, purges, indexes, …).
    * *journal* covers the excluded tables — the delivery/ledger surface
      the recall itself writes (``influence``, ``routing_decisions``, job
      queue, receipts, …).

    ``COUNT(*)`` catches deletes; ``MAX(rowid)`` catches inserts. An
    in-place UPDATE moves neither — that is what the ``data_version``
    fingerprint element plus the update-alert path in ``probe()`` are
    for. A table that rejects the probes (e.g. WITHOUT ROWID) still
    contributes its count; probe errors contribute a fixed marker.

    Two UNION ALL statements replace the ~100 per-table probes — the
    digest is identical; WITHOUT-ROWID tables (sniffed from the stored
    DDL) contribute count-only rows in the second statement. Any
    statement error falls back to the original per-table loop, so a
    schema shape the sniffer missed can never produce a wrong digest.
    """
    try:
        master = conn.execute(
            "SELECT name, COALESCE(sql, '') FROM sqlite_master"
            " WHERE type='table'"
        ).fetchall()
    except sqlite3.Error:
        return "watermark-unavailable", "watermark-unavailable"
    rowid_tables = sorted(
        n for n, sql in master if "WITHOUT ROWID" not in sql.upper()
    )
    norowid_tables = sorted(
        n for n, sql in master if "WITHOUT ROWID" in sql.upper()
    )
    rows: list = []
    try:
        if rowid_tables:
            rows.extend(conn.execute(
                " UNION ALL ".join(
                    f"SELECT '{name}' AS t, COUNT(*) AS c,"
                    f" COALESCE(MAX(rowid), 0) AS m FROM \"{name}\""
                    for name in rowid_tables
                )
            ).fetchall())
        if norowid_tables:
            rows.extend(conn.execute(
                " UNION ALL ".join(
                    f"SELECT '{name}' AS t, COUNT(*) AS c, -1 AS m"
                    f' FROM "{name}"'
                    for name in norowid_tables
                )
            ).fetchall())
    except sqlite3.Error:
        rows = _watermarks_per_table(conn, master)
    h_c = hashlib.sha256()
    h_j = hashlib.sha256()
    for name, cnt, mx in sorted(rows, key=lambda r: r[0]):
        token = f"{name}:{cnt}:{mx};".encode("ascii")
        (h_j if name in _WATERMARK_EXCLUDED else h_c).update(token)
    return h_c.hexdigest(), h_j.hexdigest()


def _watermarks_per_table(conn: sqlite3.Connection,
                          master: list) -> list:
    """Per-table fallback for ``_watermarks`` — identical digests, one
    query per table, per-table errors degrade to the (-1, -1) marker."""
    out: list = []
    for name, _sql in master:
        try:
            cnt, mx = conn.execute(
                f'SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM "{name}"'
            ).fetchone()
        except sqlite3.Error:
            try:
                cnt = conn.execute(
                    f'SELECT COUNT(*) FROM "{name}"'
                ).fetchone()[0]
                mx = -1
            except sqlite3.Error:
                cnt, mx = -1, -1
        out.append((name, cnt, mx))
    return out


def _meta_int(conn: sqlite3.Connection, key: str) -> Optional[int]:
    try:
        row = conn.execute(
            "SELECT value_json FROM meta WHERE key = ?", (key,)
        ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    v = safe_json_loads(row[0])
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _data_version(conn: sqlite3.Connection) -> Optional[int]:
    """``PRAGMA data_version`` — bumps on every commit from another
    connection (the real store's writer); constant on single-conn test
    doubles, where the watermark carries the load instead."""
    try:
        return int(conn.execute("PRAGMA data_version").fetchone()[0])
    except sqlite3.Error:
        return None


def _epoch_vector(conn: sqlite3.Connection, scope_ids) -> tuple:
    """Per-scope authorization epochs — revocation/grant edits bump the
    scope's ``authz_revision`` (V4-33.01 dependency epochs)."""
    sids = sorted(set(scope_ids))
    if not sids:
        return ()
    rows = conn.execute(
        f"SELECT scope_id, authz_revision FROM scopes"
        f" WHERE scope_id IN ({_ph(len(sids))}) ORDER BY scope_id",
        sids,
    ).fetchall()
    return tuple((r[0], int(r[1] or 0)) for r in rows)


def _quote_vector(conn: sqlite3.Connection, request: Any,
                  scope_ids) -> Optional[tuple]:
    """Per-scope ``quote`` outcome for the caller at the request purpose.

    The vector is part of the KEY: a grant change that flips quote on any
    contributing scope yields a different key — the L2 entry is then
    unreachable, never re-served under altered authority.
    """
    try:
        from .. import governance  # type: ignore
    except ImportError:
        return None  # shim stores have no verb model — scope set suffices
    task = getattr(request, "task", None)
    caller = governance.CallerV3(
        principal_id=request.caller_id,
        session_id=(getattr(task, "task_id", None) or "") if task else "",
    )
    out: list = []
    for sid in sorted(set(scope_ids)):
        try:
            governance.authorize(
                conn, caller, sid, "quote", purpose=request.purpose
            )
            out.append((sid, True))
        except VerbatimError as exc:
            if exc.code in (
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                ErrorCode.STALE_EPOCH,
            ):
                out.append((sid, False))
            else:
                raise
    return tuple(out)


def _not_after_us(conn: sqlite3.Connection, request: Any,
                  snap_tables: Optional[frozenset]) -> Optional[int]:
    """Time-decay bound: the earliest moment the stored result could go
    stale WITHOUT a write — working-set item expiry, plus
    ``revalidate_after`` freshness deadlines when the request demands
    fresh material. ``None`` = no passive expiry known."""
    candidates: list = []
    if snap_tables is None or "working_sets" in snap_tables:
        try:
            row = conn.execute(
                "SELECT MIN(expires_us) FROM working_sets"
            ).fetchone()
            if row and row[0] is not None:
                candidates.append(int(row[0]))
        except sqlite3.Error:
            pass
    if getattr(request, "freshness_required", False):
        try:
            row = conn.execute(
                "SELECT MIN(revalidate_after_us) FROM freshness"
                " WHERE class = 'revalidate_after'"
                " AND revalidate_after_us IS NOT NULL"
            ).fetchone()
            if row and row[0] is not None:
                candidates.append(int(row[0]))
        except sqlite3.Error:
            pass
    return min(candidates) if candidates else None


# ---------------------------------------------------------------------------
# canonical request key
# ---------------------------------------------------------------------------


def _canon(value: Any) -> Any:
    """Canonical key material: deterministic, total, field-complete."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (f.name, _canon(getattr(value, f.name)))
            for f in dataclasses.fields(value)
        )
    if isinstance(value, dict):
        return tuple(
            sorted((str(k), _canon(v)) for k, v in value.items())
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(
            _canon(v)
            for v in sorted(value, key=lambda x: repr(x))
        )
    return ("repr", repr(value))


def _normalized_query(query: str) -> str:
    """Exact-normalized form only (V4-33.04): case-fold + whitespace
    collapse. There is deliberately no semantic/fuzzy equivalence — two
    different questions can never share an entry."""
    return " ".join(str(query).split()).lower()


def request_key(request: Any, detail: Any, scope_ids,
                quote_vector, cfg_bits: tuple) -> str:
    """SHA-256 key material for the whole request surface (V4-33.01)."""
    fields = (
        (f.name, _canon(getattr(request, f.name)))
        for f in dataclasses.fields(request)
        if f.name != "query"  # normalized separately — raw casing/
        # whitespace must not split an otherwise-identical request
    )
    material = {
        "request": tuple(sorted(fields)),
        "normalized_query": _normalized_query(request.query),
        "scope_ids": tuple(sorted(scope_ids)),
        "quote_vector": quote_vector,
        "detail": getattr(detail, "value", detail),
        "cfg": cfg_bits,
    }
    return hashlib.sha256(repr(material).encode("utf-8")).hexdigest()


def consumer_request_key(
    *,
    query: str,
    limit: int,
    filters: Any,
    consistency: str,
    namespace: str,
    caller_id: str,
    encoder_id: Optional[str],
    frontier,
) -> str:
    """SHA-256 key for the consumer ``Memory.search`` result cache
    (V6-02.13, docs/v6_contracts §7.5).

    The key carries every request input that shapes the assembled
    result — the exact-normalized query (never fuzzy), the bounded
    limit, validated filters, the consistency mode, the bound
    namespace + caller (authority material, never crossed), the
    encoder identity (the source lane's query vector + the verdict's
    calibrated floor are pinned to it), and the resolved barrier
    frontier (the receipt set the session/causal barrier waited on —
    a different causal context is a different request). Timing knobs
    (``timeout_ms``, ``ready_timeout_ms``) are deliberately absent:
    they bound how long the caller waits, never what a completed
    READY result contains.
    """
    material = {
        "kind": "consumer_search/v1",
        "normalized_query": _normalized_query(query),
        "limit": int(limit),
        "filters": _canon(filters or {}),
        "consistency": str(consistency),
        "namespace": str(namespace),
        "caller_id": str(caller_id),
        "encoder_id": str(encoder_id or ""),
        "frontier": tuple(
            sorted({str(r) for r in (frontier or ()) if r is not None})
        ),
    }
    return hashlib.sha256(repr(material).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    """A stored final-pack result plus the validity fingerprint that must
    still hold to serve it."""

    packs: tuple            # tuple[ContextPack] — frozen, shareable
    omitted: int
    warnings: tuple         # final warning list from the original recall
    capabilities: dict      # lanes/degraded reports (deep-copied)
    routeset: Any           # for the hit-path decision log (replay parity)
    state: Any
    detail: str
    fingerprint: tuple
    not_after_us: Optional[int]
    refs: frozenset         # (object_kind, object_id, revision) delivered
    created_us: int


@dataclass
class _ResultEntry:
    """A stored consumer ``Memory.search`` result (V6-02.13).

    The same validity discipline as ``_Entry`` — a probe-time
    fingerprint the store must still match plus per-delivered-ref
    revalidation — over the assembled consumer result rather than
    v3 packs. ``items`` are the serialized hit payloads (the
    ``Hit`` dataclass fields plus the ``pins``/``signals`` extras a
    delivered hit can carry); nothing beyond what a normal hit
    already ships is stored.
    """

    items: tuple            # tuple[dict] — serialized Hit payloads
    status: str             # always SearchStatus.READY on a stored entry
    warnings: tuple         # final warning list from the original search
    coverage: dict          # lanes/omitted/route/support (deep-frozen)
    fingerprint: tuple
    not_after_us: Optional[int]
    refs: frozenset         # (object_kind, object_id, revision) delivered
    source_expect: dict     # source_id -> (control_version, disposition,
    #                       mutation_head) row snapshot, or None for
    #                       sources with no source_state row at store
    #                       time — the hit-time equality check catches
    #                       supersede/correct/retract/erase transitions
    #                       that move neither watermark counter
    created_us: int


@dataclass
class CacheProbe:
    """probe() result — ``entry`` is served only when ``hit`` is set.

    ``fingerprint`` is the probe-snapshot fingerprint when one was
    computed (always for ``probe_result``, hit-only for ``probe``);
    callers store fresh results under it so the entry is valid only
    while the store still matches the snapshot the request was
    evaluated under.
    """

    key: str
    entry: Optional[Any] = None
    fingerprint: Optional[tuple] = None


class RecallCache:
    """Per-store LRU of final packs. In-process only — nothing here
    survives ``Store.close()`` or crosses a process boundary (V4-33)."""

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = max(1, int(capacity))
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._mutation_epoch = 0
        self._stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "invalidations": 0,
            "expired": 0,
            "ref_denied": 0,
            "evictions": 0,
            "restamps": 0,
            "update_alerts": 0,
            # Consumer-result (Memory.search) attribution — the totals
            # above already include these; the split keeps the envelope
            # report honest about which surface produced the counters.
            "result_lookups": 0,
            "result_hits": 0,
            "result_stores": 0,
        }

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        """Honest counters (V4-33.07) — hits beside invalidation and
        re-validation failures, never a fabricated hit rate."""
        out = dict(self._stats)
        out["capacity"] = self.capacity
        out["entries"] = len(self._entries)
        lookups = out["lookups"] or 0
        out["hit_rate"] = (
            round(out["hits"] / lookups, 6) if lookups else 0.0
        )
        return out

    # -- fingerprint ---------------------------------------------------------

    # Fingerprint layout (V4-33.02):
    #   [0] data_version  — bumps on ANY commit; the only in-place-UPDATE
    #       detector SQLite gives us.
    #   [1] mutation_epoch — cache-local counter bumped when a probe sees
    #       data_version move with NO table's watermark moving: an
    #       in-place UPDATE somewhere. Because the update cannot be
    #       attributed to a table, the bump invalidates EVERY stored
    #       entry on its next probe (folded into the fingerprint).
    #   [2:8] the *content part* — scope epoch vector, policy/erasure
    #       epochs, projection generation, schema, content watermark.
    #       Any difference here is a real content/authority change.
    #   [8] journal watermark — excluded-table inserts/deletes. Movement
    #       here (with the content part unchanged) means the delta was
    #       delivery/ledger traffic — including this recall's own write
    #       phase — never delivered-content state.
    _DV = 0
    _CONTENT = slice(1, 8)
    _JOURNAL = 8

    def _fingerprint(self, conn: sqlite3.Connection, store: Any,
                     scope_ids, generation: int) -> tuple:
        wm_content, wm_journal = _watermarks(conn)
        return (
            _data_version(conn),
            self._mutation_epoch,
            _epoch_vector(conn, scope_ids),
            _meta_int(conn, "policy_epoch"),
            _meta_int(conn, "erasure_epoch"),
            int(generation),
            int(getattr(store, "schema_version", 0) or 0),
            wm_content,
            wm_journal,
        )

    # -- probe / store -------------------------------------------------------

    def probe(self, conn: sqlite3.Connection, store: Any, request: Any,
              *, scope_ids, generation: int, detail: Any,
              cfg_bits: tuple, now: Optional[int] = None) -> CacheProbe:
        """Look up the request; on a hit the entry is *revalidated*, not
        merely found (V4-33.02)."""
        self._stats["lookups"] += 1
        now = now_us() if now is None else now
        qv = _quote_vector(conn, request, scope_ids)
        key = request_key(request, detail, scope_ids, qv, cfg_bits)
        return self._probe_entry(
            conn, store, key, scope_ids, generation, now
        )

    def _probe_entry(
        self,
        conn: sqlite3.Connection,
        store: Any,
        key: str,
        scope_ids,
        generation: int,
        now: int,
        *,
        fingerprint: Optional[tuple] = None,
    ) -> CacheProbe:
        """Post-key probe body shared by the pack and consumer paths.

        Fingerprint check → expiry → per-ref revalidation, on the
        caller's snapshot. ``fingerprint`` may be precomputed (the
        consumer probe always needs it for the store side); when it is
        ``None`` it is computed only when an entry exists, keeping the
        miss path cheap.
        """
        entry = self._entries.get(key)
        if fingerprint is None and entry is not None:
            fingerprint = self._fingerprint(
                conn, store, scope_ids, generation
            )
        if entry is None:
            self._stats["misses"] += 1
            return CacheProbe(key=key, fingerprint=fingerprint)
        fp = fingerprint
        if fp != entry.fingerprint:
            if fp[self._CONTENT] != entry.fingerprint[self._CONTENT]:
                # Real content/authority drift (or a past update-alert
                # epoch bump) — the stored state no longer describes the
                # snapshot the probe is running on.
                self._evict(key)
                self._stats["invalidations"] += 1
                self._stats["misses"] += 1
                return CacheProbe(key=key, fingerprint=fp)
            if (
                fp[self._DV] != entry.fingerprint[self._DV]
                and fp[self._JOURNAL] == entry.fingerprint[self._JOURNAL]
            ):
                # data_version moved but NO table's rowid/count moved —
                # an in-place UPDATE committed somewhere. It cannot be
                # attributed to a table, so the only honest response is
                # a cache-wide invalidation epoch bump (this entry plus
                # every other stored entry mismatches on next probe).
                self._mutation_epoch += 1
                self._stats["update_alerts"] += 1
                self._evict(key)
                self._stats["invalidations"] += 1
                self._stats["misses"] += 1
                return CacheProbe(key=key, fingerprint=fp)
            # Content part identical, journal watermark moved: only
            # delivery/ledger traffic committed (a recall's own write
            # phase writes exactly those tables). Restamp the entry at
            # the current fingerprint and continue to ref revalidation —
            # the hit path's per-ref checks still run below.
            entry.fingerprint = fp
            self._stats["restamps"] += 1
        if entry.not_after_us is not None and now >= entry.not_after_us:
            self._evict(key)
            self._stats["expired"] += 1
            self._stats["misses"] += 1
            return CacheProbe(key=key, fingerprint=fp)
        if not self._refs_deliverable(
            conn, store, entry.refs,
            source_expect=getattr(entry, "source_expect", None),
        ):
            self._evict(key)
            self._stats["ref_denied"] += 1
            self._stats["misses"] += 1
            return CacheProbe(key=key, fingerprint=fp)
        self._stats["hits"] += 1
        self._entries.move_to_end(key)
        return CacheProbe(key=key, entry=entry, fingerprint=fp)

    # -- consumer result cache (V6-02.13) -------------------------------------

    #: Bound on a stored consumer result's passive lifetime. The
    #: fingerprint + per-ref revalidation already cover every committed
    #: change; the TTL caps whatever no write can signal (a
    #: validity-window expiry the read path evaluates lazily, for
    #: example) and bounds every entry deterministically.
    _RESULT_TTL_US = 300 * 1_000_000

    def probe_result(
        self,
        conn: sqlite3.Connection,
        store: Any,
        *,
        key: str,
        scope_ids,
        generation: int,
        now: Optional[int] = None,
    ) -> CacheProbe:
        """Consumer ``Memory.search`` probe (V6-02.13).

        Unlike the pack probe the fingerprint is computed on EVERY
        lookup — the caller stores a fresh result under the
        probe-snapshot fingerprint, so the entry can only ever be
        served while the store still matches the snapshot the request
        was evaluated under.
        """
        self._stats["lookups"] += 1
        self._stats["result_lookups"] += 1
        now = now_us() if now is None else now
        fp = self._fingerprint(conn, store, scope_ids, generation)
        probe = self._probe_entry(
            conn, store, key, scope_ids, generation, now, fingerprint=fp
        )
        if probe.entry is not None:
            self._stats["result_hits"] += 1
        return probe

    def store_result(
        self,
        key: str,
        *,
        fingerprint: tuple,
        items,
        status: str,
        warnings,
        coverage: dict,
        refs,
        source_expect: dict,
        ttl_us: Optional[int] = None,
    ) -> None:
        """Record an assembled consumer result under ``key``.

        ``fingerprint`` is the probe-snapshot fingerprint — NOT a fresh
        post-delivery one: the entry is valid only while the store
        still matches the snapshot the search was evaluated under, so
        any commit that raced the pipeline invalidates on next probe
        rather than blessing a result computed across a write boundary.
        """
        ttl = self._RESULT_TTL_US if ttl_us is None else int(ttl_us)
        entry = _ResultEntry(
            items=tuple(items),
            status=str(status),
            warnings=tuple(warnings),
            coverage=_deep_freeze(dict(coverage or {})),
            fingerprint=tuple(fingerprint),
            not_after_us=(now_us() + ttl) if ttl else None,
            refs=frozenset(refs),
            source_expect=dict(source_expect or {}),
            created_us=now_us(),
        )
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._stats["stores"] += 1
        self._stats["result_stores"] += 1
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self._stats["evictions"] += 1

    def store(self, conn: sqlite3.Connection, store_obj: Any, key: str,
              *, request: Any, scope_ids, generation: int, detail: Any,
              packs, omitted: int, warnings, capabilities: dict,
              routeset: Any, state: Any) -> None:
        """Record a delivered result under ``key`` (LRU-bounded)."""
        refs = frozenset(
            (i.handle.object_kind, i.handle.object_id, i.handle.revision)
            for p in packs
            for i in p.items
        )
        entry = _Entry(
            packs=tuple(packs),
            omitted=int(omitted),
            warnings=tuple(warnings),
            capabilities=_deep_freeze(capabilities),
            routeset=routeset,
            state=state,
            detail=getattr(detail, "value", detail),
            fingerprint=self._fingerprint(
                conn, store_obj, scope_ids, generation
            ),
            not_after_us=_not_after_us(conn, request, None),
            refs=refs,
            created_us=now_us(),
        )
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._stats["stores"] += 1
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self._stats["evictions"] += 1

    def settle(self, conn: sqlite3.Connection, store: Any, key: str,
               *, scope_ids, generation: int) -> bool:
        """Re-stamp the stored fingerprint after the recall's own write
        phase commits (V4-33.02).

        ``PRAGMA data_version`` and the journal watermark move on the
        recall's own journal commit (``influence``/``routing_decisions``).
        Left unstamped, the entry's render-time fingerprint could never
        match a post-write probe and the cache would miss forever.
        Called on a fresh reader snapshot after the write phase:

        * content part unchanged, journal/dv moved → the window's commits
          were ledger traffic → adopt the settled fingerprint so later
          probes hit (journal-only deltas are re-stamped there anyway).
        * content part unchanged but dv moved with the journal watermark
          unmoved → an in-place UPDATE landed inside the window →
          update-alert: bump ``mutation_epoch`` and evict.
        * content part changed → a foreign write landed between render
          and settle — the entry may predate it → evict; the next probe
          recomputes (conservative, honest).

        Returns ``True`` when the entry survived the stamp.
        """
        entry = self._entries.get(key)
        if entry is None:
            return False
        try:
            fp2 = self._fingerprint(conn, store, scope_ids, generation)
        except Exception:
            return False  # unreadable state → keep render-time fp; it
            # can only ever mismatch a probe → conservative miss
        if fp2[self._CONTENT] != entry.fingerprint[self._CONTENT]:
            self._evict(key)
            return False
        if (
            fp2[self._DV] != entry.fingerprint[self._DV]
            and fp2[self._JOURNAL] == entry.fingerprint[self._JOURNAL]
        ):
            self._mutation_epoch += 1
            self._stats["update_alerts"] += 1
            self._evict(key)
            return False
        entry.fingerprint = fp2
        return True

    def _evict(self, key: str) -> None:
        self._entries.pop(key, None)

    # -- hit-time revalidation (V4-33.02) -------------------------------------

    def _refs_deliverable(self, conn: sqlite3.Connection, store: Any,
                          refs: frozenset,
                          source_expect: Optional[dict] = None) -> bool:
        """Fresh kernel validation of every delivered reference.

        Quarantine holds (span/source/envelope cascade included), purge
        suppression, and claim head-revision drift are re-checked against
        the CURRENT snapshot — a cached pack can never serve an object
        that has since been held, erased, or corrected (V4-33.03).

        ``source_expect`` — consumer-result entries only: the
        ``source_state`` row snapshot each delivered source carried at
        store time. Its exact-equality check catches supersede/correct/
        retract/erase transitions — in-place UPDATEs that move neither
        watermark counter — before any stored byte ships (V6-02.13).
        """
        if not refs:
            return True
        from .v3 import union as _union  # deferred: no import cycle

        try:
            held = _union._excluded_refs(conn, refs)
        except Exception:
            return False  # fail closed on unreadable hold state
        if held:
            return False
        by_kind: dict = {}
        for k, o, _r in refs:
            by_kind.setdefault(k, []).append(o)
        for kind, ids in by_kind.items():
            try:
                if _cand._suppressed(store, conn, kind, ids):
                    return False
            except Exception:
                return False
        source_refs = {
            o: r for k, o, r in refs if k == "source"
        }
        if source_refs:
            # Source + covering-source-envelope holds — the same
            # cascade the facade's _held_source_ids applies at delivery
            # time (a revision-exact ref probe misses envelope holds).
            try:
                if _source_held(conn, sorted(source_refs)):
                    return False
            except Exception:
                return False
            if source_expect is not None:
                try:
                    if not _source_states_match(
                        conn, source_refs, source_expect
                    ):
                        return False
                except Exception:
                    return False
        claim_refs = {
            o: r for k, o, r in refs if k == "claim"
        }
        if claim_refs:
            ids = sorted(claim_refs)
            heads: dict = {}
            try:
                for part in _cand._chunks(ids, _cand._IN_CHUNK):
                    for cid, rev, state in conn.execute(
                        "SELECT claim_id, revision, state"
                        " FROM claim_revisions"
                        " WHERE recorded_until IS NULL"
                        f" AND claim_id IN ({_ph(len(part))})",
                        part,
                    ).fetchall():
                        heads[cid] = (rev, state)
            except sqlite3.Error:
                return False
            snap = _union.SnapshotCache(conn)
            for cid, rev in claim_refs.items():
                head = heads.get(cid)
                if head is None or int(head[0]) != int(rev):
                    return False  # corrected / tombstoned since delivery
                if head[1] not in ("active", "disputed"):
                    return False  # lifecycle moved past visibility
                try:
                    if _union._span_suppressed_claims(
                        conn, snap, store, {cid: (rev, head[1], 0)}
                    ):
                        return False
                except Exception:
                    return False
        return True


def _deep_freeze(value: Any) -> Any:
    """Copy a capabilities dict into immutable-by-convention form."""
    if isinstance(value, dict):
        return {k: _deep_freeze(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    return value


def _source_held(conn: sqlite3.Connection, source_ids: list) -> set:
    """Source ids under a live quarantine hold, envelope cascade
    included — the same withheld set the facade's ``_held_source_ids_in``
    computes on its delivery snapshot.

    A ``(source, id, rev)``-exact ref probe misses holds recorded on the
    covering ``source_envelope`` row and holds whose revision token
    differs from the delivered revision, so this check is by source id
    alone — matching the delivery-time contract (V3-14.10 / V6 §7).
    Fail-closed: an unreadable hold table treats every probed id as
    held rather than leaking.
    """
    ids = sorted(set(source_ids))
    if not ids:
        return set()
    live = conn.execute(
        "SELECT 1 FROM quarantine"
        " WHERE state IN ('pending','suppressed') LIMIT 1"
    ).fetchone()
    if live is None:
        return set()
    ph = _ph(len(ids))
    held = {
        str(r[0])
        for r in conn.execute(
            "SELECT object_id FROM quarantine"
            f" WHERE object_kind = 'source' AND object_id IN ({ph})"
            " AND state IN ('pending','suppressed')",
            ids,
        )
    }
    held |= {
        str(r[0])
        for r in conn.execute(
            "SELECT se.source_id FROM quarantine q"
            " JOIN source_envelopes se"
            "  ON se.envelope_id = q.object_id"
            f" WHERE q.object_kind = 'source_envelope'"
            f"  AND se.source_id IN ({ph})"
            "  AND q.state IN ('pending','suppressed')",
            ids,
        )
    }
    return held


def _source_states_match(
    conn: sqlite3.Connection, source_refs: dict, source_expect: dict
) -> bool:
    """Every delivered source's ``source_state`` row must equal the row
    snapshot recorded at store time.

    ``(control_version, disposition, mutation_head)`` equality catches
    lifecycle transitions — supersede/correct/retract/archive/erase and
    head-revision drift — that commit as in-place UPDATEs (no watermark
    counter moves) and would otherwise serve a stale lifecycle label or
    a superseded answer. A source absent from ``source_expect`` is a
    malformed entry → deny; a row that is absent now and was absent at
    store time stays servable (append-only ``source_revisions`` changes
    move the content watermark instead).
    """
    try:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master"
            " WHERE type = 'table' AND name = 'source_state'"
        ).fetchone()
    except sqlite3.Error:
        return False
    if table is None:
        # Legacy store without the control plane — nothing recorded
        # could have transitioned; an expectation row would be
        # inconsistent with the schema the entry was stored under.
        return not any(v is not None for v in source_expect.values())
    ids = sorted(source_refs)
    rows: dict = {}
    for i in range(0, len(ids), _cand._IN_CHUNK):
        part = ids[i : i + _cand._IN_CHUNK]
        try:
            for r in conn.execute(
                "SELECT source_id, control_version, disposition,"
                f" mutation_head FROM source_state"
                f" WHERE source_id IN ({_ph(len(part))})",
                part,
            ).fetchall():
                rows[str(r[0])] = (int(r[1]), str(r[2]), str(r[3]))
        except sqlite3.Error:
            return False
    for sid in ids:
        if sid not in source_expect:
            return False
        cur = rows.get(sid)
        cur_t = tuple(cur) if cur is not None else None
        if cur_t != source_expect.get(sid):
            return False
    return True


def get_recall_cache(store: Any, capacity: int = 256) -> RecallCache:
    """The store's lazily-attached cache — per-store, per-process."""
    cache = getattr(store, "_recall_cache_v4", None)
    if cache is None:
        cache = RecallCache(capacity=capacity)
        try:
            setattr(store, "_recall_cache_v4", cache)
        except Exception:
            pass  # a store that refuses attributes just runs uncached
    return cache


def configured(store: Any, cfg_root: Any) -> Optional[RecallCache]:
    """Resolve ``retrieval.cache.enabled`` → the store's cache or None.

    The flag is OFF by default: the invalidation model is conservative
    (content watermark + data_version + epochs + ref revalidation), but
    it is a performance layer operators opt into, not a silent default —
    tests enable it explicitly.
    """
    retrieval = getattr(cfg_root, "retrieval", None)
    ccfg = getattr(retrieval, "cache", None)
    if ccfg is None:
        return None
    if not bool(getattr(ccfg, "enabled", False)):
        return None
    return get_recall_cache(
        store, capacity=int(getattr(ccfg, "capacity", 256) or 256)
    )


def stats_for(store: Any, cfg_root: Any) -> dict:
    """Status-surface helper: honest cache stats (V4-33.07) for
    ``Engine.status`` / capabilities reporting."""
    retrieval = getattr(cfg_root, "retrieval", None)
    ccfg = getattr(retrieval, "cache", None)
    enabled = bool(getattr(ccfg, "enabled", False))
    cache = getattr(store, "_recall_cache_v4", None)
    if cache is None:
        return {
            "enabled": enabled,
            "state": "configured" if enabled else "implemented",
            "degraded_reason": (
                None if enabled else
                "retrieval.cache.enabled is off (default-off safety gate)"
            ),
            "note": "in-process per-store LRU; counters appear after the "
                    "first recall_v3 while enabled",
        }
    out = cache.stats()
    out["enabled"] = enabled
    out["state"] = "healthy" if enabled else "configured"
    out["degraded_reason"] = (
        None if enabled else "retrieval.cache.enabled is off"
    )
    return out


__all__ = [
    "RecallCache",
    "CacheProbe",
    "configured",
    "consumer_request_key",
    "get_recall_cache",
    "stats_for",
    "request_key",
]
