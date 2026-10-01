"""V7 eligibility adapter — the ``ctx.eligible`` seam every unit lane fences
through (V7-05.08, docs/v7_contracts.md).

``make_eligible(conn, store, ...)`` resolves the caller's authority ONCE at
construction — ``governance.authorize`` on the caller-pinned read snapshot —
then preloads the security universe in bounded, scope-fenced batches and
returns a predicate that never touches SQL again.  The predicate accepts a
unit-row mapping, a bare ``unit_id`` string, or any attribute-bearing
candidate (``CandidateV7`` at the pipeline S3 re-check), and answers one
question:

    may this caller see the content this candidate names?

The verdict is a pure function of the pinned snapshot:

* **scope + generation fence** — the unit row must live in ``scope_id`` at
  ``generation <= pinned``; a row asserting a foreign scope or a newer
  generation fails closed.
* **source-chain integrity** — the unit's ``(source_id, revision)`` must
  resolve to real ``sources`` + ``source_revisions`` rows whose namespace
  authority (``source_state.namespace`` when present, else
  ``sources.scope_id`` — the same authority rule
  ``jobs/source_jobs._resolve_namespace`` applies at write time) is the
  requested scope, with non-empty retained payload (an emptied/purged
  payload reads as absent, matching ``projections/_withhold.py``).
* **quarantine cascade (V3-14.10)** — a live hold (``pending`` /
  ``suppressed``) on the ``source`` revision, on any ``span`` carved from
  it, or on a covering ``source_envelope`` withholds every dependent unit.
  Unit-granular holds (``object_kind='unit'``) deny the unit itself.
* **purge tombstones** — purges in a suppressing state
  (``suppressed``/``purging``/``completed``; ``previewed`` never hides)
  targeting the ``source``, the exact ``source_revision`` key
  (``"<source_id>:<revision>"``), or any of its spans withhold the unit.
  A span purge empties the whole parent revision's bytes
  (``purge._erase_span``), so the cascade is revision-granular by
  construction — identical to ``_withhold.source_revision_withheld``.
* **unrecognized input** — anything that is not a resolvable unit or a
  verifiable ``(source_id, revision)`` pair returns ``False``.

Degradation is honest and never widens access:

* authorization failure propagates the typed ``VerbatimError``
  (``NOT_FOUND_OR_UNAUTHORIZED`` / ``STALE_EPOCH`` / ``VALIDATION``) —
  identical to every other governed surface (§09.09, §10.05): absent and
  forbidden are publicly indistinguishable;
* a store lacking the unit/source plane (``units``, ``sources``,
  ``source_revisions``), an over-bound or unreadable hold/purge
  inventory, a ``purges`` row in a suppressing state whose
  ``purge_targets`` cannot be read, or an unpinned/invalid generation
  yields an all-false predicate reporting ``stats["mode"] ==
  "unavailable"`` with a ``reason`` — the documented unavailable mode;
* optional machinery absent — ``quarantine``, ``spans``,
  ``source_envelopes``, ``source_state``, or the purge pair when no
  suppressing purge exists — means that gate has no rows to apply (a
  hold cannot exist in a table that does not exist);
* any exception inside the predicate returns ``False`` — never ``True``.

Snapshot cache (V8-14.01, §23 ``elig.cache_size`` prior 32, scenarios
K79/K80).  ``_load`` costs one bounded scan set (~18 ms p50, D8-12); the
materialized inputs it produces are a pure function of a small
watermark vector, so ``make_eligible`` keeps a process-local bounded
LRU of immutable snapshot entries keyed by the FULL input identity:

    (db identity, scope_id, principal_id, purpose, pinned generation,
     scopes.authz_revision [grant epoch], grants_v3 watermark,
     delegations watermark [delegation epoch], quarantine watermark,
     purges + purge_targets watermarks, source_state watermark,
     units/sources/source_revisions/spans append watermarks,
     source_envelopes + principals watermarks)

Every component is read through the caller's pinned ``conn``, so the
key is exactly what the pinned snapshot would rebuild from — a pinned
older transaction legitimately HITS an entry built under the same
watermarks (its snapshot genuinely sees that state), while any
governance write (grant create/revoke, delegation, quarantine decision,
purge plan/suppress/execute/lift, source-state transition, principal
retire) moves a component and makes the stale key unreachable — the
spec's wholesale invalidation.  Unreadable components fail closed:
no lookup is served and nothing is stored (``stats["cache_note"]``
records the bypass).  Watermark coverage is complete for every table
``_load``/``authorize`` reads; append tables are guarded by
``MAX(rowid)`` (a documented residual: a same-connection
non-max-rowid ``DELETE`` on an append table outside the
generation/purge paths is the only mutation invisible to the vector —
no production writer does this).  Entries carry ``not_after_us`` —
the earliest instant a verdict could flip WITHOUT a write (live
grant/delegation ``expires_us`` bounds, future ``valid_from`` on
``active`` source_state rows); a hit attempted past it misses and
rebuilds.  Entries hold verdict data only — no payload bytes — and
are immutable: a hit mints a fresh ``_Eligible`` (fresh memos, fresh
stats) so per-call state never crosses searches.

The adapter consumes only the caller's pinned snapshot ``conn`` — it
never opens a transaction.  ``store`` is accepted for the frozen
signature and only consulted to *resolve* a snapshot when ``conn`` is
not itself a connection (the same fallback chain the lanes use); the
predicate closes over loaded sets/maps, not the connection.

``eligible.stats`` is a plain dict, updated by the predicate::

    mode               "ok" | "unavailable"
    reason             None | why the adapter is unavailable
    checked_sources    distinct source_ids whose revisions were gated
    held_sources       distinct source_ids with >=1 withheld pair
    checked_pairs      (source_id, revision) pairs evaluated
    withheld_pairs     pairs failing any gate (missing/foreign included)
    units_fenced       in-scope unit rows at/below the generation pin
    unit_holds         live unit-kind hold ids observed in the snapshot
    calls              predicate invocations
    missing_unit_rows  calls whose unit_id resolved to no snapshot row
    denied             calls returning False
    predicate_errors   calls that raised internally (all return False)
    cache              "hit" | "miss" — snapshot-cache outcome (V8-14.01)
    cache_t_ms         ms spent computing the cache key + probing the LRU
    cache_size         resolved ``elig.cache_size`` arm (0 disables)
    cache_keyed        a complete key vector was readable on the snapshot
    cache_note         bypass/diagnostic detail (unreadable key, expired)
    t_ms               ms inside ``make_eligible`` construction itself
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Optional

from ...core.time import now_us
from ...core.types import (
    ErrorCode,
    VerbatimError,
    require_id,
    safe_json_loads,
)
from ...core.types_v3 import Verb
from ...governance import CallerV3, authorize
from ...storage.repos import has_table

__all__ = [
    "elig_cache_clear",
    "elig_cache_stats",
    "make_eligible",
]

_FIELDS = ("unit_id", "source_id", "revision", "scope_id", "generation")

#: Quarantine states that hide content from ordinary retrieval
#: (``security.quarantine.EXCLUDING_STATES`` — pending + suppressed;
#: released restores visibility, purged is privacy's own tombstone).
_LIVE_HOLDS = ("pending", "suppressed")
#: Purge states that suppress retrieval (``previewed`` never hides —
#: mirrors ``retrieval/candidates._SUPPRESSING_STATES`` and
#: ``PurgesRepo._SUPPRESSED``).
_SUPPRESSING_PURGE = ("suppressed", "purging", "completed")

#: ``source_state`` dispositions that can never be any query's answer
#: (V5-14.09): ``recorded`` is pre-admission registration; the rest
#: carry no current head.  ``superseded`` and closed-window ``active``
#: rows are deliberately absent — the intent-aware currency stage
#: downstream decides drop-vs-labeled delivery (V7-09.11); only states
#: that can answer NO query are withheld before rank here.
_NEVER_CURRENT = frozenset(
    {"recorded", "corrected", "retracted", "archived", "erased"}
)


def _rfc3339_ts(value: Any) -> Optional[float]:
    """RFC3339/ISO-8601 → epoch seconds; ``None`` when unparseable.

    Same rule as ``source._rfc3339_ts`` — duplicated here so the
    eligibility adapter never imports a lane module (lane modules
    self-register on import)."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


def _never_answerable(disp: str, eff: Any, vf: Any, now_ts: float) -> bool:
    """``True`` when the disposition can answer NO query at ``now_ts`` —
    the context-free half of V7-05.08's lifecycle leg, evaluated before
    rank on every lane.  ``superseded`` with a parseable ``effective_at``
    and ``active`` past ``valid_to`` stay eligible: they are history's
    answers (V7-09.11) and the pipeline currency stage filters them for
    current intents."""

    if disp in _NEVER_CURRENT:
        return True
    if disp == "superseded":
        # A superseded row whose boundary is unreadable can never open
        # its answer window — the successor owns the answer (V5-14.12).
        return _rfc3339_ts(eff) is None
    if disp == "active":
        # Not yet valid — it answers nothing now and is not yet history.
        vf_ts = _rfc3339_ts(vf)
        return vf_ts is not None and now_ts < vf_ts
    return False


def _currency_verdict(
    disp: str, eff: Any, vf: Any, vt: Any, now_ts: float, history: bool
) -> tuple:
    """``(deliverable, label|None)`` — the intent-dependent half of the
    V5-14.12 currency window, applied post-rank where intent is known:

    * ``superseded`` answers until ``effective_at`` — always labeled;
      past the boundary it answers ``history_of`` intents only.
    * ``active`` answers inside ``[valid_from, valid_to)``; a closed
      window is history (``history_of`` only, labeled ``historical``).
    * never-answerable dispositions never deliver, any intent.
    * unknown/missing dispositions are admissible unlabeled — an
      unrecognized lifecycle state is not an assertion of suppression
      (V5-14.16)."""

    if not disp:
        return True, None
    if disp in _NEVER_CURRENT:
        return False, None
    if disp == "superseded":
        boundary = _rfc3339_ts(eff)
        if boundary is None:
            return False, None
        if now_ts < boundary:
            return True, "superseded"
        return (True, "superseded") if history else (False, None)
    if disp == "active":
        start = _rfc3339_ts(vf)
        if start is not None and now_ts < start:
            return False, None
        end = _rfc3339_ts(vt)
        if end is not None and now_ts >= end:
            return (True, "historical") if history else (False, None)
        return True, None
    return True, None

_UNIT_SCAN_CAP = 262_144   # in-fence unit rows preloaded
_PAIR_SCAN_CAP = 131_072   # authority-gated (source, revision) pairs
_HOLD_SCAN_CAP = 65_536    # live quarantine rows scanned
_PURGE_SCAN_CAP = 65_536   # suppressing purge-target rows scanned
_IN_CHUNK = 400            # < SQLITE_MAX_VARIABLE_NUMBER


class _InventoryBound(Exception):
    """A security inventory (holds/purge targets) exceeded its scan bound —
    an incomplete withheld set can never be trusted, so construction
    degrades to the all-false unavailable mode instead of guessing."""


def _ph(n: int) -> str:
    return ",".join("?" * n)


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _new_stats() -> dict:
    return {
        "mode": "ok",
        "reason": None,
        "checked_sources": 0,
        "held_sources": 0,
        "checked_pairs": 0,
        "withheld_pairs": 0,
        "units_fenced": 0,
        "unit_holds": 0,
        "calls": 0,
        "missing_unit_rows": 0,
        "denied": 0,
        "denied_scope": 0,
        "denied_generation": 0,
        "denied_source": 0,
        "denied_held": 0,
        "unrecognized": 0,
        "predicate_errors": 0,
        # V8-14.01 snapshot-cache block — ``cache`` reports the honest
        # lookup outcome ("miss" whenever no entry was served, for any
        # reason); ``cache_note`` carries the bypass detail.
        "cache": "miss",
        "cache_t_ms": 0.0,
        "cache_size": 0,
        "cache_keyed": False,
        "cache_note": None,
        "t_ms": 0.0,
    }


# ---------------------------------------------------------------------------
# V8-14.01 — process-local eligibility snapshot cache
# ---------------------------------------------------------------------------
#
# ``_load`` costs one bounded scan set per query (~18 ms p50, D8-12).
# The verdict data it materializes is a pure function of the pinned
# snapshot's (scope, generation, security inventory, authority rows) —
# so the snapshot is cached under a key made of cheap watermark
# components read through the caller's own pinned ``conn``.  A
# component that cannot be read makes the key unreadable: the lookup
# misses and a fresh snapshot is built — fail closed, never a guess.

#: §23 arm name (cross-worker contract) and its prior — the LRU bound.
ELIG_CACHE_SIZE_ARM = "elig.cache_size"
ELIG_CACHE_SIZE_DEFAULT = 32
#: Hard ceiling for the arm — a memory bound, not a quality knob.
_ELIG_CACHE_SIZE_MAX = 65_536

_SNAP_LOCK = threading.Lock()
_SNAP_LRU: "OrderedDict[tuple, _SnapEntry]" = OrderedDict()
#: Process-local serving epoch — bumped on every mutation of the LRU
#: (insert, evict, passive-expire delete, wholesale clear).  The §19
#: closure verifier compares it before/after a purge: post-closure the
#: epoch must have moved past the purge's write (``cache epoch > purge
#: epoch after closure``).
_SNAP_EPOCH = [0]
_SNAP_STATS: dict = {
    "hits": 0,
    "misses": 0,
    "inserts": 0,
    "evictions": 0,
    "expired": 0,
    "bypassed": 0,
    "clears": 0,
}


class _SnapEntry:
    """One immutable cached snapshot — verdict data only, no payloads.

    Shared ``units``/``ok_pairs``/``unit_holds``/``purged``/``unit_ids``
    structures are closed over by every ``_Eligible`` minted from this
    entry; none of them are ever mutated after construction, so a hit
    is byte-identical to a fresh ``_load`` of the same snapshot.
    ``stats_seed`` is the construction-time stats snapshot — copied per
    hit so per-call counters never cross searches.
    """

    __slots__ = (
        "units", "ok_pairs", "unit_holds", "purged", "unit_ids",
        "stats_seed", "not_after_us",
    )

    def __init__(
        self,
        *,
        units: dict,
        ok_pairs: frozenset,
        unit_holds: frozenset,
        purged: frozenset,
        unit_ids: frozenset,
        stats_seed: dict,
        not_after_us: Optional[int],
    ) -> None:
        self.units = units
        self.ok_pairs = ok_pairs
        self.unit_holds = unit_holds
        self.purged = purged
        self.unit_ids = unit_ids
        self.stats_seed = stats_seed
        self.not_after_us = not_after_us


def _unit_id_set(units: dict, ok_pairs: frozenset,
                 unit_holds: frozenset, purged: frozenset) -> frozenset:
    """The eligible ``unit_id`` set for one materialized snapshot — the
    formula ``_Eligible.unit_ids`` materializes lazily, extracted so the
    cache entry precomputes it once (V8-14.01)."""
    return frozenset(
        uid
        for uid, gens in units.items()
        if uid not in unit_holds
        and ("unit", uid) not in purged
        and gens[max(gens)] in ok_pairs
    )


def _resolve_cache_size(
    explicit: Any, params: Any, policy: Any, store: Any
) -> int:
    """Effective ``elig.cache_size`` (§23, prior 32).

    Same resolution convention as the §23 lane arms
    (``entity._per_canon_quota``): the explicit kwarg first, then
    params-style mappings (``params``/``arms``/``overrides`` attributes
    or mappings themselves) keyed by the dotted arm name or its
    ``cache_size`` leaf on ``params``, ``policy``, then ``store``, and
    finally a direct attribute.  Absent everywhere → the prior.  ``0``
    disables caching; mistyped or out-of-range values raise
    ``VALIDATION`` — a mistyped arm fails loudly, never silently
    reconfigures the cache.
    """
    leaf = ELIG_CACHE_SIZE_ARM.split(".", 1)[1]
    raw: Any = None
    seen = False
    if explicit is not None:
        raw, seen = explicit, True
    else:
        for holder in (policy, params, store):
            if holder is None:
                continue
            if isinstance(holder, Mapping):
                if ELIG_CACHE_SIZE_ARM in holder:
                    raw, seen = holder[ELIG_CACHE_SIZE_ARM], True
                elif leaf in holder:
                    raw, seen = holder[leaf], True
                if seen:
                    break
                continue
            for attr in ("params", "arms", "overrides"):
                m = getattr(holder, attr, None)
                if isinstance(m, Mapping):
                    if ELIG_CACHE_SIZE_ARM in m:
                        raw, seen = m[ELIG_CACHE_SIZE_ARM], True
                        break
                    if leaf in m:
                        raw, seen = m[leaf], True
                        break
            if seen:
                break
            val = getattr(holder, "elig_cache_size", None)
            if val is None:
                val = getattr(holder, leaf, None)
            if val is not None:
                raw, seen = val, True
                break
    if not seen or raw is None:
        return ELIG_CACHE_SIZE_DEFAULT

    def _bad() -> VerbatimError:
        return VerbatimError(
            ErrorCode.VALIDATION,
            f"{ELIG_CACHE_SIZE_ARM} must be null or an integer"
            f" in [0, {_ELIG_CACHE_SIZE_MAX}], got {raw!r}",
        )

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _bad()
    if isinstance(raw, float) and not raw.is_integer():
        raise _bad()
    value = int(raw)
    if not 0 <= value <= _ELIG_CACHE_SIZE_MAX:
        raise _bad()
    return value


def _db_identity(conn: sqlite3.Connection, tables: frozenset) -> Any:
    """Which database this snapshot belongs to — the cache is module-
    global, so the key must not let store A serve store B's eligible
    set.  ``meta.db_id`` (minted at ``Store.create``) is the durable
    identity; the file path covers pre-meta file stores; a bare
    ``:memory:``/shim conn falls back to its own id (each in-memory
    database IS its connection)."""
    if "meta" in tables:
        row = conn.execute(
            "SELECT value_json FROM meta WHERE key = 'db_id'"
        ).fetchone()
        if row is not None:
            v = safe_json_loads(row[0])
            return ("db", v if isinstance(v, str) and v else repr(v))
    path: Any = None
    try:
        for _seq, name, file in conn.execute("PRAGMA database_list"):
            if name == "main":
                path = file
                break
    except sqlite3.Error:
        path = None
    if isinstance(path, str) and path not in ("", ":memory:"):
        return ("file", path)
    return ("conn", id(conn))


#: Tables the key vector probes — the complete read surface of
#: ``_load`` + ``authorize`` plus ``meta`` (db identity).  One
#: ``sqlite_master`` probe resolves the present subset.
_KEY_TABLES = (
    "scopes", "grants_v3", "delegations", "quarantine", "purges",
    "purge_targets", "source_state", "units", "sources",
    "source_revisions", "spans", "source_envelopes", "principals",
    "meta",
)


def _cache_key(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    principal_id: str,
    purpose: str,
) -> tuple:
    """``(key, not_after_us)`` — the full eligibility input identity.

    One cheap watermark per table the snapshot reads; every component
    changes whenever verdict-relevant content does:

    * ``scopes.authz_revision`` — the grant epoch: revocation, consent
      and principal retirement bump it in the write transaction.
    * ``grants_v3`` count / ``SUM(revoked_us)`` / ``MAX(issued_us)`` /
      ``MAX(rowid)`` — grant creation never moves the epoch, so the
      table carries its own watermark; scoped to ``scope_id`` exactly
      like ``authorize``'s query.
    * ``delegations`` count / ``SUM(revoked_us)`` / ``MAX(rowid)`` —
      the delegation epoch (edges carry no scope column).
    * ``quarantine`` count / ``SUM(decided_event)`` / ``MAX(opened_event)``
      / ``MAX(rowid)`` — every hold insert moves count+opened+rowid;
      every decision (release/suppress/mark_purged, including the
      purge-side rewrite) writes a fresh ``decided_event`` — the live-
      hold set can never mutate invisibly.
    * ``purges`` count / ``SUM(approved_us)`` / ``SUM(completed_us)`` /
      ``MAX(rowid)`` — previewed→suppressed is the only verdict-moving
      transition without a row change and it always stamps
      ``approved_us``; ``purging``/``completed`` stamp
      ``completed_us``; ``lift_suppression`` deletes rows.
    * ``purge_targets`` count / ``MAX(rowid)``.
    * ``source_state`` count / ``SUM(control_version)`` / ``MAX(rowid)``
      — global on purpose: the namespace-override join in ``_load``
      means a row registered under ANY namespace can flip this scope's
      pair authority, and every transition bumps ``control_version``.
    * ``units`` / ``sources`` / ``source_revisions`` / ``spans``
      ``MAX(rowid)`` — append-only outside generation-bump and purge
      paths (both already keyed); inserts and projection rewrites
      always mint fresh rowids.
    * ``source_envelopes`` count / ``MAX(rowid)`` / ``SUM(revision)`` —
      privacy paths delete envelope rows.
    * ``principals`` count / ``SUM(retired)`` / ``MAX(rowid)`` —
      registration and retirement (retirement also bumps the epoch).

    ``not_after_us`` is the earliest instant a verdict could flip
    without any write — live grant/delegation ``expires_us`` bounds
    (expired rows are dead already, so only FUTURE expiries bind);
    the source-state ``valid_from`` futures are folded in by
    ``_load``.  ``None`` bounds never expire passively.
    """
    now = now_us()
    tables = frozenset(
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master"
            f" WHERE name IN ({_ph(len(_KEY_TABLES))})",
            _KEY_TABLES,
        ).fetchall()
    )
    parts: list = [
        _db_identity(conn, tables),
        scope_id,
        principal_id,
        purpose,
        generation,
    ]
    bounds: list = []

    if "scopes" in tables:
        row = conn.execute(
            "SELECT authz_revision FROM scopes WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()
        parts.append(int(row[0]) if row else 0)
    else:
        parts.append(-1)

    if "grants_v3" in tables:
        r = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(revoked_us), 0),"
            " COALESCE(MAX(issued_us), 0), COALESCE(MAX(rowid), 0),"
            " MIN(CASE WHEN revoked_us IS NULL AND expires_us > ?"
            "       THEN expires_us END)"
            " FROM grants_v3 WHERE scope_id = ?",
            (now, scope_id),
        ).fetchone()
        parts += [int(r[0]), int(r[1]), int(r[2]), int(r[3])]
        bounds.append(r[4])
    else:
        parts += [-1, -1, -1, -1]

    if "delegations" in tables:
        r = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(revoked_us), 0),"
            " COALESCE(MAX(rowid), 0),"
            " MIN(CASE WHEN revoked_us IS NULL AND expires_us > ?"
            "       THEN expires_us END)"
            " FROM delegations",
            (now,),
        ).fetchone()
        parts += [int(r[0]), int(r[1]), int(r[2])]
        bounds.append(r[3])
    else:
        parts += [-1, -1, -1]

    if "quarantine" in tables:
        r = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(decided_event), 0),"
            " COALESCE(MAX(opened_event), 0), COALESCE(MAX(rowid), 0)"
            " FROM quarantine"
        ).fetchone()
        parts += [int(r[0]), int(r[1]), int(r[2]), int(r[3])]
    else:
        parts += [-1, -1, -1, -1]

    if "purges" in tables:
        r = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(approved_us), 0),"
            " COALESCE(SUM(completed_us), 0), COALESCE(MAX(rowid), 0)"
            " FROM purges"
        ).fetchone()
        parts += [int(r[0]), int(r[1]), int(r[2]), int(r[3])]
    else:
        parts += [-1, -1, -1, -1]

    if "purge_targets" in tables:
        r = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(rowid), 0)"
            " FROM purge_targets"
        ).fetchone()
        parts += [int(r[0]), int(r[1])]
    else:
        parts += [-1, -1]

    if "source_state" in tables:
        r = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(control_version), 0),"
            " COALESCE(MAX(rowid), 0) FROM source_state"
        ).fetchone()
        parts += [int(r[0]), int(r[1]), int(r[2])]
    else:
        parts += [-1, -1, -1]

    # Append-plane guards — one round-trip of scalar subqueries; absent
    # tables pin ``-1`` (distinct from every real watermark).
    exprs = [
        f"(SELECT COALESCE(MAX(rowid), 0) FROM {t})" if t in tables else "-1"
        for t in ("units", "sources", "source_revisions", "spans")
    ]
    parts += [
        int(v) for v in conn.execute("SELECT " + ", ".join(exprs)).fetchone()
    ]

    if "source_envelopes" in tables:
        r = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(rowid), 0),"
            " COALESCE(SUM(revision), 0) FROM source_envelopes"
        ).fetchone()
        parts += [int(r[0]), int(r[1]), int(r[2])]
    else:
        parts += [-1, -1, -1]

    if "principals" in tables:
        r = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(retired), 0),"
            " COALESCE(MAX(rowid), 0) FROM principals"
        ).fetchone()
        parts += [int(r[0]), int(r[1]), int(r[2])]
    else:
        parts += [-1, -1, -1]

    not_after: Optional[int] = None
    for b in bounds:
        if b is not None:
            b = int(b)
            not_after = b if not_after is None else min(not_after, b)
    return tuple(parts), not_after


def _snap_lookup(key: tuple) -> Optional[_SnapEntry]:
    """LRU probe — a hit past its ``not_after_us`` bound is evicted and
    reported as a miss (the earliest passive-flip instant arrived)."""
    with _SNAP_LOCK:
        ent = _SNAP_LRU.get(key)
        if ent is None:
            _SNAP_STATS["misses"] += 1
            return None
        bound = ent.not_after_us
        if bound is not None and now_us() >= bound:
            try:
                del _SNAP_LRU[key]
            except KeyError:
                pass
            _SNAP_EPOCH[0] += 1
            _SNAP_STATS["expired"] += 1
            _SNAP_STATS["misses"] += 1
            return None
        _SNAP_LRU.move_to_end(key)
        _SNAP_STATS["hits"] += 1
        return ent


def _snap_store(key: tuple, ent: _SnapEntry, size: int) -> None:
    """Insert + enforce the LRU bound (most-recently-used tail)."""
    if size <= 0:
        return
    with _SNAP_LOCK:
        _SNAP_LRU[key] = ent
        _SNAP_LRU.move_to_end(key)
        _SNAP_EPOCH[0] += 1
        _SNAP_STATS["inserts"] += 1
        while len(_SNAP_LRU) > size:
            _SNAP_LRU.popitem(last=False)
            _SNAP_EPOCH[0] += 1
            _SNAP_STATS["evictions"] += 1


def elig_cache_clear() -> int:
    """Drop every cached snapshot — the wholesale-invalidation hook.

    Keyed watermarks already make stale entries unreachable; this is
    the explicit primitive for tests, verifiers, and any future
    in-process governance-write signal.  Returns the entry count
    dropped."""
    with _SNAP_LOCK:
        n = len(_SNAP_LRU)
        _SNAP_LRU.clear()
        _SNAP_EPOCH[0] += 1
        _SNAP_STATS["clears"] += 1
        return n


def elig_cache_stats() -> dict:
    """Cache telemetry for the §23 arm's hit-rate ledger and the §19
    closure verifier — counters, the serving ``cache_epoch``, and live
    occupancy.  Read-only."""
    with _SNAP_LOCK:
        return {
            "entries": len(_SNAP_LRU),
            "cache_epoch": _SNAP_EPOCH[0],
            **_SNAP_STATS,
        }


def _resolve_conn(conn: Any, store: Any) -> Optional[sqlite3.Connection]:
    """Resolve the caller-pinned read connection — explicit ``conn``,
    then ``store`` itself when it is a connection, ``store.conn``, the
    Store's thread-local reader, or a duck-typed ``execute`` handle.
    Mirrors the lane-side convention (``source._conn``); the adapter
    never opens a transaction of its own (V7-05.06)."""
    if isinstance(conn, sqlite3.Connection):
        return conn
    if isinstance(store, sqlite3.Connection):
        return store
    if hasattr(conn, "execute"):
        return conn  # the caller's snapshot handle wins over store probes
    inner = getattr(store, "conn", None)
    if isinstance(inner, sqlite3.Connection):
        return inner
    reader = getattr(store, "_reader", None)
    if callable(reader):
        try:
            cand = reader()
        except Exception:
            cand = None
        if isinstance(cand, sqlite3.Connection):
            return cand
    if hasattr(store, "execute"):
        return store  # duck-typed connection-like (test shims)
    return None


def _norm_id(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _norm_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _extract(obj: Any) -> Optional[tuple]:
    """``(unit_id, source_id, revision, scope_id, generation)`` — the
    identity a candidate claims.  ``None`` when the input shape is
    unrecognized (the caller treats it as ineligible)."""
    if isinstance(obj, str):
        return (obj, None, None, None, None)
    if type(obj) is dict:
        # Hot path: the overwhelming input is a plain unit-row dict.
        # Inline the ``_norm_*`` fast cases — a value already of the
        # target type (or None) passes through untouched.
        get = obj.get
        v = get("unit_id")
        uid = v if v is None or type(v) is str else _norm_id(v)
        v = get("source_id")
        sid = v if v is None or type(v) is str else _norm_id(v)
        v = get("revision")
        rev = v if v is None or type(v) is int else _norm_int(v)
        v = get("scope_id")
        rscope = v if v is None or type(v) is str else _norm_id(v)
        v = get("generation")
        rgen = v if v is None or type(v) is int else _norm_int(v)
        return (uid, sid, rev, rscope, rgen)
    if isinstance(obj, Mapping):
        get = obj.get
    elif hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        # sqlite3.Row and other mapping-likes without .get / Mapping
        # registration.
        try:
            keys = set(obj.keys())
        except Exception:
            return None

        def get(k: str, _o=obj, _ks=keys) -> Any:
            return _o[k] if k in _ks else None
    else:
        vals = tuple(getattr(obj, k, None) for k in _FIELDS)
        if all(v is None for v in vals):
            return None
        return (
            _norm_id(vals[0]), _norm_id(vals[1]), _norm_int(vals[2]),
            _norm_id(vals[3]), _norm_int(vals[4]),
        )
    return (
        _norm_id(get("unit_id")), _norm_id(get("source_id")),
        _norm_int(get("revision")), _norm_id(get("scope_id")),
        _norm_int(get("generation")),
    )


class _Eligible:
    """Snapshot-pure eligibility predicate — the ``ctx.eligible`` handle.

    Immutable verdict data was assembled at construction; a call is a
    handful of dict/set lookups and never issues SQL.  Every failure
    mode returns ``False``.
    """

    __slots__ = (
        "_scope", "_gen", "_units", "_ok", "_unit_holds", "_purged",
        "_memo", "_id_memo", "_unit_ids", "stats",
    )

    def __init__(
        self,
        *,
        scope_id: str,
        generation: int,
        units: dict,
        ok_pairs: frozenset,
        unit_holds: frozenset,
        purged: frozenset,
        stats: dict,
        unit_ids: Optional[frozenset] = None,
    ) -> None:
        self._scope = scope_id
        self._gen = generation
        self._units = units          # {unit_id: {generation: (sid, rev)}}
        self._ok = ok_pairs          # {(source_id, revision)} verified visible
        self._unit_holds = unit_holds  # unit ids under a live unit-kind hold
        self._purged = purged        # {(object_kind, object_id)}
        # {(uid, sid, rev, rscope, rgen)|None: (verdict, stat_key)} — the
        # verdict is a pure function of the extracted identity and the
        # pinned snapshot, and lanes re-resolve the same units across
        # scans; the memo replays the per-path stat increment on every
        # hit so counters stay exactly call-accurate.
        self._memo: dict = {}
        # {id(obj): (obj, verdict, stat_key)} — lanes re-pass the SAME
        # row-dict objects (the shared universe rows, fetched unit rows)
        # across facets/peers; keying on identity skips re-extraction.
        # The entry holds the object so its id can never be recycled
        # while cached; the memo dies with this per-search predicate, so
        # an object mutated between searches can never alias.
        self._id_memo: dict = {}
        # Lazily materialized eligible-unit_id set (``unit_ids`` below) —
        # preseeded by the snapshot cache on a hit or by ``make_eligible``
        # on a caching build; None until a set-mode consumer asks for it
        # on a non-cached construction.
        self._unit_ids: Optional[frozenset] = unit_ids
        self.stats = stats

    @property
    def unit_ids(self) -> frozenset:
        """The eligible ``unit_id`` set, materialized once from the pinned
        snapshot — zero SQL per lookup.

        Membership reproduces the bare-``unit_id`` verdict exactly:
        latest in-fence generation's ``(source_id, revision)`` must be a
        verified-visible pair, and the unit itself must sit under no live
        hold or suppressing purge.  Lane consumers that fence to the
        newest row per ``unit_id`` (``units`` scans are ``generation
        DESC`` first-seen everywhere) see verdicts identical to the
        per-row callable path; set-mode consumers never re-run the
        row-level ``units`` fetch the callable path would need.
        """
        ids = self._unit_ids
        if ids is None:
            ids = _unit_id_set(
                self._units, self._ok, self._unit_holds, self._purged
            )
            self._unit_ids = ids
            self.stats["unit_ids_materialized"] = len(ids)
        return ids

    # -- protocol forms -------------------------------------------------
    def __call__(self, obj: Any) -> bool:
        try:
            self.stats["calls"] += 1
            verdict = bool(self._decide(obj))
        except Exception:
            # The security property is eligibility-before-rank: a
            # predicate fault is a deny, never a pass (V7-05.08).
            self.stats["predicate_errors"] += 1
            verdict = False
        if not verdict:
            self.stats["denied"] += 1
        return verdict

    def __contains__(self, unit_id: Any) -> bool:
        """Container probe (``unit_id in eligible``) — same verdict as a
        bare-string call so set-style consumers stay consistent."""
        return self(unit_id)

    def is_eligible(self, obj: Any) -> bool:
        """``eligible.is_eligible(row)`` protocol form."""
        return self(obj)

    def eligible(self, obj: Any) -> bool:
        """``eligible.eligible(row)`` protocol form."""
        return self(obj)

    def contains(self, unit_id: Any) -> bool:
        """``eligible.contains(unit_id)`` protocol form."""
        return self(unit_id)

    # -- the verdict ----------------------------------------------------
    def _decide(self, obj: Any) -> bool:
        ient = self._id_memo.get(id(obj))
        if ient is not None and ient[0] is obj:
            verdict, stat = ient[1], ient[2]
            if stat is not None:
                self.stats[stat] += 1
            return verdict
        fields = _extract(obj)
        ent = self._memo.get(fields)
        if ent is None:
            ent = self._verdict(fields)
            self._memo[fields] = ent
        verdict, stat = ent
        if obj is not None and not isinstance(obj, (str, int, float, bool)):
            self._id_memo[id(obj)] = (obj, verdict, stat)
        if stat is not None:
            self.stats[stat] += 1
        return verdict

    def _verdict(self, fields: Optional[tuple]) -> tuple:
        """``(verdict, stat_key)`` for one extracted identity — a pure
        function of the pinned snapshot (memoized by ``_decide``; the
        caller replays ``stats[stat_key] += 1`` on every call so the
        counters stay identical to an unmemoized evaluation)."""
        if fields is None:
            return False, "unrecognized"
        uid, sid, rev, rscope, rgen = fields

        # Asserted scope/generation can only shrink the fence, never
        # widen it — a row claiming a foreign scope or a generation
        # above the pinned snapshot is denied outright.
        if rscope is not None and rscope != self._scope:
            return False, "denied_scope"
        if rgen is not None and rgen > self._gen:
            return False, "denied_generation"

        gens = self._units.get(uid) if uid is not None else None
        if gens:
            # The store row is the identity authority: a claimed
            # (source_id, revision) never overrides what the snapshot
            # says the unit derives from.  A real claimed generation
            # selects that row; anything else resolves to the latest
            # in-fence row (stale-generation nomination, V7-30.02).
            gen_u = rgen if rgen in gens else max(gens)
            usid, urev = gens[gen_u]
            if uid in self._unit_holds or ("unit", uid) in self._purged:
                return False, "denied_held"
            if (usid, urev) in self._ok:
                return True, None
            return False, "denied_source"

        # Unresolved unit_id — the source-level path (the source lane's
        # synthesized ``"{source_id}:{revision}"`` rows and S3-rechecked
        # source-granular candidates): the claimed pair must itself be a
        # verified-visible source revision of this scope.
        if sid is not None and rev is not None:
            if (sid, rev) in self._ok:
                return True, None
            return False, "denied_source"

        if uid is not None:
            return False, "missing_unit_rows"
        return False, None


def _unavailable(
    stats: dict, reason: str, t0: Optional[float] = None
) -> "_Eligible":
    """The honest unavailable mode: an all-false predicate carrying the
    reason.  Never widens — every gate that could not be evaluated
    denies."""
    stats["mode"] = "unavailable"
    stats["reason"] = reason
    if t0 is not None:
        stats["t_ms"] = round((time.monotonic() - t0) * 1000.0, 3)
    return _Eligible(
        scope_id="",
        generation=-1,
        units={},
        ok_pairs=frozenset(),
        unit_holds=frozenset(),
        purged=frozenset(),
        stats=stats,
    )


def _load(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: int,
    stats: dict,
) -> dict:
    """The bounded, scope-fenced snapshot read — runs once inside the
    caller's pinned transaction.  Returns the immutable verdict data the
    predicate closes over."""
    # ---- units slice: scope + generation fence (latest-row resolution is
    # per unit_id; several stamped generations coexist under V7-30.02) ----
    unit_rows: dict[str, dict[int, tuple]] = {}
    unit_pairs: set = set()
    rows = conn.execute(
        "SELECT unit_id, source_id, revision, generation FROM units"
        " WHERE scope_id = ? AND generation <= ?"
        " ORDER BY unit_id, generation LIMIT ?",
        (scope_id, generation, _UNIT_SCAN_CAP + 1),
    ).fetchall()
    if len(rows) > _UNIT_SCAN_CAP:
        stats["units_truncated"] = True
        del rows[_UNIT_SCAN_CAP:]
    for uid, sid, rev, gen in rows:
        sid_s, rev_i, gen_i = str(sid), int(rev), int(gen)
        unit_rows.setdefault(str(uid), {})[gen_i] = (sid_s, rev_i)
        unit_pairs.add((sid_s, rev_i))

    # ---- authority-gated source revisions -----------------------------
    # A pair is *known* only when the source row exists, the revision row
    # exists, and the source's namespace authority is the requested scope
    # (``source_state.namespace`` wins over the ``sources.scope_id``
    # partition — the write-side rule, mirrored).  Missing/foreign rows
    # simply never enter the eligible set.
    if has_table(conn, "source_state"):
        rev_sql = (
            "SELECT sr.source_id, sr.revision,"
            " (sr.payload IS NOT NULL AND length(sr.payload) > 0)"
            " FROM source_revisions sr"
            " JOIN sources s ON s.source_id = sr.source_id"
            " LEFT JOIN source_state ss ON ss.source_id = sr.source_id"
            " WHERE COALESCE(ss.namespace, s.scope_id) = ?"
            " ORDER BY sr.source_id, sr.revision LIMIT ?"
        )
    else:
        rev_sql = (
            "SELECT sr.source_id, sr.revision,"
            " (sr.payload IS NOT NULL AND length(sr.payload) > 0)"
            " FROM source_revisions sr"
            " JOIN sources s ON s.source_id = sr.source_id"
            " WHERE s.scope_id = ?"
            " ORDER BY sr.source_id, sr.revision LIMIT ?"
        )
    rev_rows = conn.execute(
        rev_sql, (scope_id, _PAIR_SCAN_CAP + 1)
    ).fetchall()
    if len(rev_rows) > _PAIR_SCAN_CAP:
        stats["pairs_truncated"] = True
        del rev_rows[_PAIR_SCAN_CAP:]
    pair_live: dict = {}  # (sid, rev) -> payload non-empty
    for sid, rev, nonempty in rev_rows:
        pair_live[(str(sid), int(rev))] = bool(nonempty)

    # ---- lifecycle: never-answerable dispositions (V7-05.08) ----------
    # ``superseded`` past ``effective_at`` and closed-window ``active``
    # rows STAY eligible — they are ``history_of``'s answers (V7-09.11)
    # and the pipeline currency stage applies the intent-dependent
    # window; only states that can answer NO query are withheld before
    # rank here.  Bounded scan, fail-closed on truncation like the rest
    # of the security inventory.
    never_sources: set = set()
    next_flip: Optional[float] = None  # earliest passive eligibility flip (s)
    if has_table(conn, "source_state"):
        now_ts = datetime.now(timezone.utc).timestamp()
        srows = conn.execute(
            "SELECT source_id, disposition, effective_at, valid_from"
            " FROM source_state WHERE namespace = ? LIMIT ?",
            (scope_id, _PAIR_SCAN_CAP + 1),
        ).fetchall()
        if len(srows) > _PAIR_SCAN_CAP:
            raise _InventoryBound("lifecycle_scan_bound")
        for sid, disp, eff, vf in srows:
            if _never_answerable(str(disp or ""), eff, vf, now_ts):
                never_sources.add(str(sid))
            # An ``active`` row withheld only because its window has not
            # opened flips passively at ``valid_from`` — no write marks
            # the boundary, so the cache's ``not_after_us`` tracks the
            # earliest one (V8-14.01).
            if str(disp or "") == "active":
                vf_ts = _rfc3339_ts(vf)
                if vf_ts is not None and vf_ts > now_ts:
                    next_flip = (
                        vf_ts
                        if next_flip is None
                        else min(next_flip, vf_ts)
                    )

    # ---- live quarantine holds (V3-14.10) ------------------------------
    # One bounded scan of the live-hold set; holds are rare so the whole
    # inventory fits far under the cap.  Truncation or a read fault is
    # fail-closed: a partially-known hold set can never be trusted.
    held_refs: set = set()  # (object_kind, object_id, revision)
    if has_table(conn, "quarantine"):
        hrows = conn.execute(
            "SELECT object_kind, object_id, revision FROM quarantine"
            f" WHERE state IN ({_ph(len(_LIVE_HOLDS))})"
            " ORDER BY object_kind, object_id, revision LIMIT ?",
            [*_LIVE_HOLDS, _HOLD_SCAN_CAP + 1],
        ).fetchall()
        if len(hrows) > _HOLD_SCAN_CAP:
            raise _InventoryBound("hold_scan_bound")
        held_refs = {(str(k), str(o), int(r)) for k, o, r in hrows}

    # ---- suppressing purge targets (§15/§40, V2-41.06) -----------------
    purged: set = set()  # (object_kind, object_id)
    if has_table(conn, "purges"):
        live = conn.execute(
            f"SELECT 1 FROM purges p"
            f" WHERE p.state IN ({_ph(len(_SUPPRESSING_PURGE))}) LIMIT 1",
            list(_SUPPRESSING_PURGE),
        ).fetchone()
        if live is not None:
            if not has_table(conn, "purge_targets"):
                # Suppression exists but its target set is unreadable —
                # fail closed rather than guess what was erased.
                raise _InventoryBound("purge_targets_absent")
            prows = conn.execute(
                "SELECT DISTINCT pt.object_kind, pt.object_id"
                " FROM purge_targets pt"
                " JOIN purges p ON p.purge_id = pt.purge_id"
                f" WHERE p.state IN ({_ph(len(_SUPPRESSING_PURGE))})"
                " ORDER BY pt.object_kind, pt.object_id LIMIT ?",
                [*_SUPPRESSING_PURGE, _PURGE_SCAN_CAP + 1],
            ).fetchall()
            if len(prows) > _PURGE_SCAN_CAP:
                raise _InventoryBound("purge_scan_bound")
            purged = {(str(k), str(o)) for k, o in prows}

    # ---- resolve held/purged span + envelope ids to their pairs --------
    # Only the security inventory is resolved — bounded by the number of
    # held/purged objects, never by corpus size.
    span_taint: set = set()
    span_ids = sorted(
        {oid for k, oid, _r in held_refs if k == "span"}
        | {oid for k, oid in purged if k == "span"}
    )
    if span_ids and has_table(conn, "spans"):
        for part in _chunks(span_ids, _IN_CHUNK):
            for oid, sid, rev in conn.execute(
                "SELECT span_id, source_id, revision FROM spans"
                f" WHERE span_id IN ({_ph(len(part))})",
                list(part),
            ).fetchall():
                span_taint.add((str(sid), int(rev)))

    env_taint: set = set()
    env_ids = sorted(
        {oid for k, oid, _r in held_refs if k == "source_envelope"}
        | {oid for k, oid in purged if k == "source_envelope"}
    )
    if env_ids and has_table(conn, "source_envelopes"):
        for part in _chunks(env_ids, _IN_CHUNK):
            for eid, sid, rev in conn.execute(
                "SELECT envelope_id, source_id, revision"
                " FROM source_envelopes"
                f" WHERE envelope_id IN ({_ph(len(part))})",
                list(part),
            ).fetchall():
                env_taint.add((str(sid), int(rev)))

    # ---- pair-level verdicts --------------------------------------------
    ok_pairs: set = set()
    for (sid, rev), nonempty in pair_live.items():
        bad = (
            not nonempty
            or sid in never_sources
            or ("source", sid) in purged
            or ("source_revision", f"{sid}:{rev}") in purged
            or (sid, rev) in span_taint
            or ("source", sid, rev) in held_refs
            or ("source_revision", sid, rev) in held_refs
            or (sid, rev) in env_taint
        )
        if not bad:
            ok_pairs.add((sid, rev))

    unit_holds = {oid for k, oid, _r in held_refs if k == "unit"}

    evaluated = set(pair_live) | unit_pairs
    withheld = evaluated - ok_pairs
    stats["checked_sources"] = len({sid for sid, _r in evaluated})
    stats["checked_pairs"] = len(evaluated)
    stats["held_sources"] = len({sid for sid, _r in withheld})
    stats["withheld_pairs"] = len(withheld)
    stats["lifecycle_blocked"] = len(never_sources)
    stats["units_fenced"] = sum(len(g) for g in unit_rows.values())
    stats["unit_holds"] = len(unit_holds)

    return {
        "units": unit_rows,
        "ok_pairs": frozenset(ok_pairs),
        "unit_holds": frozenset(unit_holds),
        "purged": frozenset(purged),
        # Passive-flip bound in µs (``now_us`` clock) — popped before the
        # ``_Eligible`` kwargs fan-out; feeds the cache entry's
        # ``not_after_us``.
        "not_after_us": (
            int(next_flip * 1_000_000) if next_flip is not None else None
        ),
    }


def make_eligible(
    conn: Any,
    store: Any,
    *,
    scope_id: str,
    generation: int,
    principal_id: str,
    purpose: str = "recall",
    cache_size: Any = None,
    policy: Any = None,
    params: Any = None,
):
    """Build the V7 eligibility predicate — the ``ctx.eligible`` seam.

    Resolves purpose-aware authorization ONCE inside the caller's pinned
    read snapshot (``conn`` — never a new transaction), preloads the
    security universe in bounded scope-fenced queries, and returns a
    callable accepting a unit-row mapping, a bare ``unit_id`` string, or
    an attribute-bearing candidate object.

    ``cache_size``/``policy``/``params`` carry the §23
    ``elig.cache_size`` arm (prior 32; ``0`` disables) — see
    ``_resolve_cache_size`` for the resolution convention.  A cache hit
    returns a predicate minted from an immutable snapshot entry and
    skips both authorization and the scan set — legal because the key
    covers every input either of them reads (authorization cannot have
    flipped: a revocation moves the epoch or the grant watermark, an
    expiry moves ``not_after_us``).  ``stats["cache"]`` reports
    ``"hit"``/``"miss"`` for ``coverage.eligibility``.

    Failure modes (all fail closed):

    * ``VerbatimError`` propagates for argument validation
    (``VALIDATION``) and for construction-time authorization denial
    (``NOT_FOUND_OR_UNAUTHORIZED`` / ``STALE_EPOCH``) — the caller's
    typed-error contract, identical to every governed surface;
    * ``stats["mode"] == "unavailable"`` + ``reason`` for degraded
    snapshots (missing unit/source plane, unreadable or over-bound
    security inventory, unpinned generation) — the returned predicate
    denies everything;
    * any exception inside the predicate returns ``False``.
    """
    stats = _new_stats()
    t0 = time.monotonic()

    scope_id = require_id(scope_id, "scope_id")
    principal_id = require_id(principal_id, "principal_id")
    if not isinstance(purpose, str) or not purpose:
        raise VerbatimError(
            ErrorCode.VALIDATION, "purpose must be a non-empty string"
        )
    size = _resolve_cache_size(cache_size, params, policy, store)
    stats["cache_size"] = size
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        return _unavailable(stats, "generation_unpinned", t0)

    snap = _resolve_conn(conn, store)
    if snap is None:
        return _unavailable(stats, "no_read_snapshot", t0)

    # ---- V8-14.01 keyed lookup --------------------------------------
    # The key is computed through the caller's pinned conn, so it names
    # exactly what this snapshot would rebuild.  An unreadable component
    # bypasses the cache entirely — fail closed, never a partial key.
    key: Optional[tuple] = None
    key_not_after: Optional[int] = None
    kt0 = time.monotonic()
    if size > 0:
        try:
            key, key_not_after = _cache_key(
                snap, scope_id, generation, principal_id, purpose
            )
            stats["cache_keyed"] = True
        except Exception as exc:
            key = None
            stats["cache_note"] = (
                f"key_unreadable:{type(exc).__name__}"
            )
            with _SNAP_LOCK:
                _SNAP_STATS["bypassed"] += 1
    else:
        stats["cache_note"] = "disabled"
    stats["cache_t_ms"] = round((time.monotonic() - kt0) * 1000.0, 3)

    if key is not None:
        ent = _snap_lookup(key)
        if ent is not None:
            # Hit — mint a fresh predicate from the immutable entry:
            # fresh memos (the per-search aliasing argument in
            # ``_Eligible`` still holds), fresh stats seeded from the
            # entry's construction-time snapshot.
            hstats = dict(ent.stats_seed)
            hstats["cache"] = "hit"
            hstats["cache_t_ms"] = stats["cache_t_ms"]
            hstats["t_ms"] = round((time.monotonic() - t0) * 1000.0, 3)
            return _Eligible(
                scope_id=scope_id,
                generation=generation,
                units=ent.units,
                ok_pairs=ent.ok_pairs,
                unit_holds=ent.unit_holds,
                purged=ent.purged,
                stats=hstats,
                unit_ids=ent.unit_ids,
            )

    # Purpose-aware, epoch-checked authorization — once, at construction.
    # Typed denials propagate (§09.09/§10.05); an absent governance plane
    # cannot authorize anything, so the predicate is honestly unavailable.
    try:
        authorize(
            snap,
            CallerV3(principal_id=principal_id),
            scope_id,
            Verb.READ.value,
            purpose=purpose,
        )
    except VerbatimError:
        raise
    except Exception:
        return _unavailable(stats, "governance_unavailable", t0)

    try:
        missing = next(
            (
                t
                for t in ("units", "sources", "source_revisions")
                if not has_table(snap, t)
            ),
            None,
        )
        if missing is not None:
            return _unavailable(stats, f"missing_table:{missing}", t0)
        loaded = _load(snap, scope_id, generation, stats)
    except _InventoryBound as exc:
        return _unavailable(stats, str(exc), t0)
    except VerbatimError:
        raise
    except Exception as exc:
        return _unavailable(
            stats, f"preload_failed:{type(exc).__name__}", t0
        )

    load_not_after = loaded.pop("not_after_us", None)
    not_after = key_not_after
    if load_not_after is not None:
        not_after = (
            load_not_after
            if not_after is None
            else min(not_after, load_not_after)
        )

    if key is not None and size > 0:
        # Materialize the eligible set once — the entry ships it per the
        # V8-14.01 snapshot contract; the returned predicate is
        # preseeded with the same frozenset.
        ids = _unit_id_set(
            loaded["units"], loaded["ok_pairs"],
            loaded["unit_holds"], loaded["purged"],
        )
        stats["unit_ids_materialized"] = len(ids)
        stats["t_ms"] = round((time.monotonic() - t0) * 1000.0, 3)
        _snap_store(
            key,
            _SnapEntry(
                units=loaded["units"],
                ok_pairs=loaded["ok_pairs"],
                unit_holds=loaded["unit_holds"],
                purged=loaded["purged"],
                unit_ids=ids,
                stats_seed=dict(stats),
                not_after_us=not_after,
            ),
            size,
        )
        return _Eligible(
            scope_id=scope_id,
            generation=generation,
            stats=stats,
            unit_ids=ids,
            **loaded,
        )

    stats["t_ms"] = round((time.monotonic() - t0) * 1000.0, 3)
    return _Eligible(
        scope_id=scope_id,
        generation=generation,
        stats=stats,
        **loaded,
    )
