"""Bounded personalized-PageRank lane over ``graph_edges`` (V7-08.08–14,
§32.6 ``graph_ppr/v1``), V8-batched (V8-05.02–05.09, §21.3).

Traversal is over the typed unit graph written by
``verbatim/jobs/graph_jobs.py``. Every constant is the provisional
``graph_ppr/v1`` policy; artifacts record ``formula_status =
provisional/v7-r0`` (V7-32.01).

Seed provenance (V8-05.07): ``S =`` the fused S2a top ``M_seed`` plus the
rarest-term seed of V75-04.04, all drawn from the eligible set. The S2a
union arrives via ``ctx.manifest["seeds"]`` (the pipeline's propagated
core-lane tops); standalone callers may pass explicit unit ids through
the ``seeds`` argument. When neither supplies seeds the lane derives its
own from the caller's snapshot: entity postings for ``qv.entity_canons``
(IDF-weighted, top-10) plus a ``unit_fts`` MATCH over the query's text
terms (top-10, best-effort). With no eligible seeds the lane reports
``skipped`` — it never widens the caller's eligible set to manufacture
starting mass.

Eligibility (V8-05.02 — the security property, V7-08.11 carried):
traversal consults ONLY the request's eligible-unit set — ``_Eligible
.unit_ids`` in production, a set/frozenset/mapping of unit_ids, or any
``__contains__`` probe over unit_ids. Membership is zero-SQL: a unit
absent from the set is ineligible, so it is never added to ``visited``,
its edges are never followed, and no neighbour is surfaced through it.
A missing or invalid set denies the whole lane with ``status="error"``
— the lane never traverses unfiltered (fail-closed; scenario K22).

Batched traversal (§21.3, V8-05.02–05.04):

- Frontier expansion fetches edge rows in ``IN (…)`` chunks bounded by
  ``_EDGE_CHUNK`` (256), capped in SQL at ``K_fan`` edges per frontier
  node per direction (dedupe to the latest generation per logical edge
  first, then ``weight DESC, dst_unit ASC`` — window-function ordered,
  bit-deterministic), never an unbounded ``fetchall`` (V8-05.03).
- Unit rows are fetched in ONE ``unit_id IN (…) AND generation <= ?``
  statement per 400-chunk of newly surfaced nodes — seeds once, each
  round's admitted-candidate set once — with the latest visible
  generation chosen in Python (identical verdict to the retired
  per-node ``ORDER BY generation DESC LIMIT 1``). Traversal of 400
  nodes costs ≤ 8 ``units`` statements (scenario K21).
- The deadline is checked before every SQL statement and at least every
  64 expanded nodes; a cut lane returns its best-so-far ranking as
  ``partial``/``deadline`` with ``nodes_expanded``,
  ``rounds_completed``, ``edges_read`` in stats (V8-05.04, K24).

Traversal (§32.6): frontier = seeds; up to 3 expansion rounds over edges
of the enabled types (``EDGE_WEIGHTS_V1`` — ``causal_candidate`` is
deliberately absent: V7-08.14 bars model-proposed causal rows from
traversal until their ablation passes); expansion stops when
``|visited| ≥ 400``. Edges are treated as a bidirectional relevance
graph — stored direction carries semantics (e.g. causal cause→effect),
not traversal permission.

Scoring (``graph.hop_policy`` arm, V8-05.08): the ``bfs3+power`` default
runs ``r ← p`` then three power iterations ``r ← 0.5·p + 0.5·Wᵀr`` with
``W`` row-normalized over the visited subgraph; the ``onehop_additive``
arm applies Hindsight's one-hop form ``tanh(0.5·shared) + max(link)``
where ``shared`` is the summed effective edge weight between the
candidate and the seed set and ``link`` the strongest single edge.
Candidates are ``visited \\ S`` ranked by score (ties on unit_id for
determinism). Each candidate's score decomposes into the V7-08.09
components — entity overlap / kNN / temporal / causal (plus ``session``
and ``update`` so nothing is hidden) — reported under
``signals["components"]`` as the share of walk mass arriving via each
edge family, with ``seed_mass`` separate. ``signals["path"]`` carries
the highest-weight seed→unit path (product of effective edge weights,
widest-path search) for ``explain``.

Honest degradation: missing tables → ``unavailable``; no eligible seeds →
``skipped``; a deadline mid-expansion or mid-iteration → ``partial`` with
``reason="deadline"`` and honest counts. All iteration order is sorted —
identical inputs give bit-identical scores.

Q6 ablation instrumentation (V7-08.12 prep, SPEC_V7_5 §05 row Q6): the
lane records a first-discovery-depth histogram — ``stats["hop_hist"]``
maps expansion depth → visited-unit count (seeds are depth 0 and are
never emitted as candidates) — and ``stats["hop2_only"]``, the number of
*emitted* candidates first discovered at depth ≥ 2 (i.e. unreachable at
``max_hops=1``). Each candidate already carries its first-discovery
depth in ``signals["hops"]``.

Hop-limit knob: ``ctx.manifest["graph_max_hops"]`` — the same
request-scoped channel as ``query_encoder``/``encoder_id``/``limit``/
``seeds`` (``LaneContextV7`` is frozen; lanes never read ``ctx.policy``
*internals* — arms are read through the resolved-params surface below).
``None``/absent → the declared ``EXPANSION_ROUNDS_V1`` bound, unchanged;
``0`` → the lane still resolves and reports its seeds but emits nothing,
``skipped`` with an honest disabled reason; ``1``/``2`` → expansion
stops after that many rounds. Values above the declared bound clamp to
it (the enforced cap is what ``stats["hop_cap"]`` reports); anything
non-integral or negative is a validation error — a mistyped arm must
fail loudly, never silently reconfigure traversal.

§23 ``graph.*`` arms (V8): ``graph.K_fan`` (default 8 — the write-side
band of ``jobs/graph_jobs.py``), ``graph.M_seed`` (default 20),
``graph.contain_N_g`` (default 20; associative-lane containment —
candidates carry ``signals["contain_N_g"]`` so fusion can bar graph-only
items ranked beyond it, V8-05.06/V8-11.04), ``graph.edge_types``
(default all weighted types; per-type traversal on/off, V8-05.09),
``graph.hop_policy`` (default ``bfs3+power``; V8-05.08), ``graph.gate``
(default armed; V8-05.05 need-gating). Resolution order: request
``ctx.manifest["graph.<name>"]`` → declared policy (``ctx.policy``
``params`` mapping or ``graph_<name>`` attribute) → the §23 prior.
The need-gate *decision* itself is plumbed by the S2b scheduler via
``ctx.manifest["graph.gate_decision"] = {"needed": bool, "inputs": {…}}``
— the lane honors a plumbed ``needed=False`` with
``skipped(not_needed)`` and logs the gate inputs in stats for explain;
with no decision plumbed the lane runs (the gate is an optimization,
never an authorization boundary).
"""

from __future__ import annotations

import heapq
import math
import sqlite3
import time
from collections.abc import Mapping
from typing import Any, Iterable, Optional

from ...core.types import ErrorCode, VerbatimError
from ...core.types_v7 import (
    CandidateV7,
    EdgeType,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    QueryViewV7,
)
from ...storage.repos import has_table

GRAPH_PPR_VERSION = "graph_ppr/v1"
GRAPH_ONEHOP_VERSION = "graph_onehop_add/v1"
FORMULA_STATUS = "provisional/v7-r0"

# §32.6 edge-type weights (pre-tuning). ``contradicts``/``refines`` are
# listed as edge types in V7-08.07 but unweighted in the §32.6 table —
# they carry the ``supersedes`` weight provisionally until the formula
# search (V7-32.02) assigns them.
EDGE_WEIGHTS_V1: dict[EdgeType, float] = {
    EdgeType.CO_MENTION: 1.0,  # stored factor is already 1/df-scaled
    EdgeType.ADJACENT_TURN: 0.8,
    EdgeType.SAME_SESSION: 0.3,
    EdgeType.TEMPORAL_NEAR: 0.5,
    EdgeType.SEMANTIC_KNN: 0.7,
    EdgeType.SUPERSEDES: 0.6,
    EdgeType.CONTRADICTS: 0.6,  # provisional — mirrors supersedes
    EdgeType.REFINES: 0.6,  # provisional — mirrors supersedes
    EdgeType.CAUSAL: 0.9,
    # EdgeType.CAUSAL_CANDIDATE deliberately absent (V7-08.14): never a
    # traversal or eligibility input until its Track R ablation passes.
}
DECLARED_EDGE_TYPES: tuple[str, ...] = tuple(
    sorted(t.value for t in EDGE_WEIGHTS_V1)
)

FRONTIER_CAP_V1 = 400        # |visited| bound (V7-08.08)
EXPANSION_ROUNDS_V1 = 3      # frontier expansion rounds
POWER_ITERS_V1 = 3           # r ← 0.5·p + 0.5·Wᵀr iterations
DAMPING_V1 = 0.5             # personalization damping
SEED_TOP_V1 = 10             # per-source seed cap (top-10 ent ∪ top-10 lex)
SEED_CAP_V1 = 40             # absolute bound on provided seeds
PATH_HOP_CAP_V1 = 8          # widest-path explain bound

#: Q6 ablation knob (V7-08.12 prep): the ``ctx.manifest`` key carrying the
#: hop limit — see the module docstring. Manifest, not ``ctx.policy``,
#: because ``LaneContextV7`` is frozen and per-request lane parameters
#: already travel this channel.
GRAPH_MAX_HOPS_KEY = "graph_max_hops"

#: §23 ``graph.*`` arm names verbatim (V8-05.02–05.09). Each resolves
#: request-manifest → declared-policy → §23 prior via :func:`_arm`.
ARM_K_FAN = "graph.K_fan"
ARM_M_SEED = "graph.M_seed"
ARM_CONTAIN_N_G = "graph.contain_N_g"
ARM_GATE = "graph.gate"
ARM_HOP_POLICY = "graph.hop_policy"
ARM_EDGE_TYPES = "graph.edge_types"

#: Scheduler-plumbed need-gate decision (V8-05.05): the S2b scheduler
#: writes ``ctx.manifest["graph.gate_decision"] = {"needed": bool,
#: "inputs": {…}}`` — the lane honors ``needed=False`` and logs inputs.
GATE_DECISION_KEY = "graph.gate_decision"

# §23 priors (pre-tuning starting points, not selected constants).
K_FAN_V8 = 8                 # mirrors the write-side fan-out band
M_SEED_V8 = 20               # fused S2a top-M_seed (Hindsight seed count)
CONTAIN_N_G_V8 = 20          # associative containment top-N (V75-04.03)
GRAPH_GATE_V8 = True         # need-gated (§23 prior; §05 exit record)
HOP_POLICY_V8 = "bfs3+power"
HOP_POLICIES: tuple[str, ...] = ("bfs3+power", "onehop_additive")

_EDGE_CHUNK = 256            # frontier ids per edge-fetch IN() (§21.3)
_UNIT_CHUNK = 400            # unit ids per row-fetch IN() (< 999 vars)
_DEADLINE_CHECK_EVERY = 64   # expanded nodes between deadline checks

# Edge type → score component family (V7-08.09 decomposition).
_FAMILY: dict[EdgeType, str] = {
    EdgeType.CO_MENTION: "entity",
    EdgeType.SEMANTIC_KNN: "knn",
    EdgeType.TEMPORAL_NEAR: "temporal",
    EdgeType.CAUSAL: "causal",
    EdgeType.SAME_SESSION: "session",
    EdgeType.ADJACENT_TURN: "session",
    EdgeType.SUPERSEDES: "update",
    EdgeType.CONTRADICTS: "update",
    EdgeType.REFINES: "update",
}
_COMPONENT_KEYS = ("entity", "knn", "temporal", "causal", "session", "update")

_UNIT_COLS = (
    "unit_id",
    "source_id",
    "revision",
    "scope_id",
    "session_id",
    "seq",
    "speaker_canon",
    "recorded_at_us",
    "occurred_start_us",
    "occurred_end_us",
    "generation",
)
_UNIT_SQL = (
    "SELECT unit_id, source_id, revision, scope_id, session_id, seq,"
    " speaker_canon, recorded_at_us, occurred_start_us, occurred_end_us,"
    " generation FROM units WHERE unit_id IN ({}) AND generation<=?"
)


class _LaneStatusError(str):
    """``status="error"`` for the V8-05.02 fail-closed path.

    ``LaneStatus`` is a contract-frozen enum (``docs/v7_contracts.md``)
    with no ERROR member, so — like the extension-lane trick in
    ``policy.py`` — this module mints the honesty value locally. A str
    subclass keeps every consumer working: ``out.status.value`` renders
    ``"error"``, ``json.dumps`` serializes it, and ``in``/``is`` checks
    against real members stay False (an errored lane contributes no
    candidates to fusion, exactly like ``unavailable``).
    """

    @property
    def value(self) -> str:  # ``coverage.lane`` / explain serialization
        return str(self)

    @property
    def name(self) -> str:
        return "ERROR"


#: The ``status="error"`` honesty value (V8-05.02, scenario K22).
LANE_STATUS_ERROR: Any = _LaneStatusError("error")


def _monotonic() -> float:
    """Indirection so tests drive the cooperative clock deterministically."""
    return time.monotonic()


def _chunks(seq: Iterable[Any], n: int) -> Iterable[list]:
    """Contiguous ``n``-sized chunks of ``seq`` (deterministic order)."""
    buf: list = []
    for item in seq:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _resolve_hop_cap(ctx: Any) -> Optional[int]:
    """Effective expansion-depth cap from ``ctx.manifest[GRAPH_MAX_HOPS_KEY]``.

    Returns ``None`` (uncapped → ``EXPANSION_ROUNDS_V1``) or the enforced
    cap in ``[0, EXPANSION_ROUNDS_V1]``. Integral floats are accepted
    (JSON manifests); bools, strings, non-integral or negative values
    raise ``VerbatimError(VALIDATION)`` — a mistyped ablation arm must
    fail loudly rather than silently run the wrong traversal.
    """

    manifest = getattr(ctx, "manifest", None) or {}
    raw = manifest.get(GRAPH_MAX_HOPS_KEY)
    if raw is None:
        return None

    def _bad() -> VerbatimError:
        return VerbatimError(
            ErrorCode.VALIDATION,
            f"{GRAPH_MAX_HOPS_KEY} must be null or a non-negative integer,"
            f" got {raw!r}",
        )

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _bad()
    if isinstance(raw, float) and (
        not math.isfinite(raw) or not raw.is_integer()
    ):
        raise _bad()
    v = int(raw)
    if v < 0:
        raise _bad()
    return min(v, EXPANSION_ROUNDS_V1)


# ---------------------------------------------------------------------------
# §23 graph.* arms — manifest → policy → prior
# ---------------------------------------------------------------------------

_MISSING = object()


def _arm(ctx: Any, name: str) -> Any:
    """Resolve a §23 ``graph.*`` arm value, or ``_MISSING`` when undeclared.

    Order: ``ctx.manifest[name]`` (per-request ablation — the same channel
    as ``graph_max_hops``) → ``ctx.policy.params[name]`` (a params mapping
    on the policy object) → ``ctx.policy.graph_<name>`` (a dedicated
    attribute). The declared §23 default is applied by the caller.
    """

    manifest = getattr(ctx, "manifest", None) or {}
    if name in manifest:
        return manifest[name]
    pol = getattr(ctx, "policy", None)
    params = getattr(pol, "params", None)
    if isinstance(params, Mapping) and name in params:
        return params[name]
    val = getattr(pol, name.replace(".", "_"), None)
    return _MISSING if val is None else val


def _bad_arm(name: str, raw: Any, want: str) -> VerbatimError:
    return VerbatimError(
        ErrorCode.VALIDATION, f"{name} must be {want}, got {raw!r}"
    )


def _arm_int(
    ctx: Any, name: str, default: int, *, lo: int, hi: int
) -> int:
    """Positive-integral arm (K_fan/M_seed): ``None`` → default; integral
    floats tolerated (JSON); anything else or out-of-range → VALIDATION."""
    raw = _arm(ctx, name)
    if raw is _MISSING or raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _bad_arm(name, raw, f"an integer in [{lo}, {hi}]")
    if isinstance(raw, float) and (
        not math.isfinite(raw) or not raw.is_integer()
    ):
        raise _bad_arm(name, raw, f"an integer in [{lo}, {hi}]")
    v = int(raw)
    if v < lo or v > hi:
        raise _bad_arm(name, raw, f"an integer in [{lo}, {hi}]")
    return v


def _arm_bool(ctx: Any, name: str, default: bool) -> bool:
    raw = _arm(ctx, name)
    if raw is _MISSING or raw is None:
        return default
    if not isinstance(raw, bool):
        raise _bad_arm(name, raw, "a boolean")
    return raw


def _arm_contain(ctx: Any) -> Optional[int]:
    """``graph.contain_N_g``: int ≥ 0 caps graph-only fusion entry;
    ``"off"``/``False`` disarms containment (→ ``None``)."""
    raw = _arm(ctx, ARM_CONTAIN_N_G)
    if raw is _MISSING or raw is None:
        return CONTAIN_N_G_V8
    if raw is False or raw == "off":
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _bad_arm(ARM_CONTAIN_N_G, raw, "a non-negative integer or \"off\"")
    if isinstance(raw, float) and (
        not math.isfinite(raw) or not raw.is_integer()
    ):
        raise _bad_arm(ARM_CONTAIN_N_G, raw, "a non-negative integer or \"off\"")
    v = int(raw)
    if v < 0:
        raise _bad_arm(ARM_CONTAIN_N_G, raw, "a non-negative integer or \"off\"")
    return v


def _arm_hop_policy(ctx: Any) -> str:
    raw = _arm(ctx, ARM_HOP_POLICY)
    if raw is _MISSING or raw is None:
        return HOP_POLICY_V8
    if not isinstance(raw, str) or raw not in HOP_POLICIES:
        raise _bad_arm(ARM_HOP_POLICY, raw, f"one of {HOP_POLICIES}")
    return raw


def _arm_edge_types(ctx: Any) -> tuple[str, ...]:
    """``graph.edge_types`` — the V8-05.09 per-type traversal arm.

    Absent/``"all"`` → every weighted type. A list of type names is
    validated against ``EdgeType`` and ``EDGE_WEIGHTS_V1``: unknown names
    and weightless types — including ``causal_candidate``, which is a
    valid enum member but barred from traversal by V7-08.14 — raise
    VALIDATION rather than silently dropping a declared arm.
    """
    raw = _arm(ctx, ARM_EDGE_TYPES)
    if raw is _MISSING or raw is None or raw == "all":
        return DECLARED_EDGE_TYPES
    if isinstance(raw, str) or not isinstance(raw, (list, tuple, set, frozenset)):
        raise _bad_arm(ARM_EDGE_TYPES, raw, "\"all\" or a list of edge types")
    out: list[str] = []
    for item in raw:
        try:
            et = EdgeType(str(item))
        except ValueError:
            raise _bad_arm(ARM_EDGE_TYPES, item, "a known edge type")
        if et not in EDGE_WEIGHTS_V1:
            raise _bad_arm(
                ARM_EDGE_TYPES, item, "a traversal-weighted edge type"
            )
        if et.value not in out:
            out.append(et.value)
    return tuple(out)


def _gate_decision(raw: Any) -> Optional[dict]:
    """Normalize the scheduler's plumbed ``graph.gate_decision`` value.

    ``{"needed": bool, "inputs": {...}}`` is the contract; a bare bool is
    accepted as shorthand. ``needed=None`` (decision object without a
    verdict) runs the lane — the gate is an optimization, never an
    authorization boundary, so a malformed/missing verdict fails open
    with an honest record.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return {"needed": raw, "inputs": {}}
    if isinstance(raw, Mapping):
        needed = raw.get("needed")
        inputs = raw.get("inputs")
        return {
            "needed": None if needed is None else bool(needed),
            "inputs": dict(inputs) if isinstance(inputs, Mapping) else {},
        }
    return {"needed": None, "inputs": {}, "malformed": type(raw).__name__}


class _Deadline:
    def __init__(self, deadline_ms: Optional[float]) -> None:
        self._end = (
            None
            if deadline_ms is None or math.isinf(deadline_ms)
            else _monotonic() + deadline_ms / 1000.0
        )

    def expired(self) -> bool:
        return self._end is not None and _monotonic() >= self._end


def _latest_edges(
    rows: Iterable[tuple],
) -> list[tuple[str, str, str, float]]:
    """Resolve multi-generation edge rows to the latest row per logical
    key ``(src_unit, type, dst_unit)`` at/below the snapshot (the PK ends
    in ``generation`` so rebuild coexistence can surface several). Input
    rows are ``(src, dst, type, weight, generation)`` — the traversal
    fetch only ever feeds it K_fan-capped rows (V8-05.03)."""
    best: dict[tuple[str, str, str], tuple[int, float]] = {}
    for src, dst, type_v, w, gen in rows:
        key = (src, type_v, dst)
        cur = best.get(key)
        if cur is None or gen > cur[0]:
            best[key] = (gen, float(w or 0.0))
    return [
        (src, dst, type_v, w)
        for (src, type_v, dst), (_g, w) in sorted(best.items())
    ]


def _conn_of(ctx: Any) -> Optional[sqlite3.Connection]:
    """The caller's pinned read snapshot. ``ctx.store`` may be a raw
    ``sqlite3.Connection`` or an object exposing ``.conn`` (Store-style).
    A lane never opens a fresh transaction."""
    store = getattr(ctx, "store", None)
    if isinstance(store, sqlite3.Connection):
        return store
    conn = getattr(store, "conn", None)
    return conn if isinstance(conn, sqlite3.Connection) else None


class _ContainsSet:
    """Membership-set facade over a ``__contains__`` probe — the
    ``unit_id in eligible`` verdict with zero SQL and no enumeration."""

    __slots__ = ("_probe",)

    def __init__(self, probe: Any) -> None:
        self._probe = probe

    def __contains__(self, uid: Any) -> bool:
        try:
            return uid in self._probe
        except Exception:
            return False  # a probing fault denies (fail closed)


def _eligibility_set(ctx: Any) -> Optional[Any]:
    """V8-05.02: the request's eligible-``unit_id`` set — the only
    eligibility input traversal consults (zero SQL per lookup).

    Resolves ``_Eligible.unit_ids`` first (the production adapter's
    materialized view), then set/frozenset/dict/list/tuple containers,
    then any ``__contains__`` probe. ``None`` — missing or invalid —
    denies the whole lane with ``status="error"`` (K22): the lane never
    traverses unfiltered. A bare ``callable`` is NOT a set — row-level
    predicates are precisely the per-node fetch pattern this change
    removes, so they are invalid input here, not a fallback.
    """

    elig = getattr(ctx, "eligible", None)
    if elig is None:
        return None
    try:
        ids = getattr(elig, "unit_ids", None)
    except Exception:
        return None
    if ids is not None:
        try:
            if callable(ids):
                ids = ids()
            return frozenset(str(u) for u in ids)
        except Exception:
            return None
    if isinstance(elig, (set, frozenset, dict, list, tuple)):
        try:
            return frozenset(str(u) for u in elig)
        except Exception:
            return None
    if hasattr(elig, "__contains__"):
        return _ContainsSet(elig)
    return None


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def _seed_id(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item
    uid = getattr(item, "unit_id", None)
    if isinstance(uid, str):
        return uid
    if isinstance(item, Mapping):
        uid = item.get("unit_id")
        return uid if isinstance(uid, str) else None
    return None


def _derive_seeds(
    ctx: Any, conn: sqlite3.Connection, qv: QueryViewV7, cut: Any
) -> list[str]:
    """Self-seeding fallback per §32.6 (top-10 L-ent ∪ top-10 L-lex) for
    callers without pipeline union seeds. Best-effort: absent tables just
    contribute nothing. Every statement is deadline-gated (V8-05.04) and
    the FTS rowid→unit_id map is one batched ``IN (…)`` read (V8-05.02).
    ``cut()`` reports deadline expiry so the caller can mark partial."""
    seeds: list[str] = []
    canons = [c for c in (qv.entity_canons or ()) if c]
    if canons and has_table(conn, "entity_mentions") and has_table(conn, "units"):
        n_units = 1
        if not cut():
            n_units = conn.execute(
                "SELECT COUNT(DISTINCT unit_id) FROM units"
                " WHERE scope_id=? AND generation<=?",
                (ctx.scope_id, ctx.generation),
            ).fetchone()[0] or 1
        df_map: dict[str, int] = {}
        if has_table(conn, "entity_canon"):
            ph = ",".join("?" for _ in canons)
            # (scope, canon, generation) PK — ascending scan ⇒ last row per
            # canon is its latest visible entry.
            if not cut():
                for c, df in conn.execute(
                    f"SELECT canon, df_units FROM entity_canon"
                    f" WHERE scope_id=? AND generation<=? AND canon IN ({ph})"
                    f" ORDER BY generation",
                    (ctx.scope_id, ctx.generation, *canons),
                ):
                    df_map[c] = df or 0
        ph = ",".join("?" for _ in canons)
        scores: dict[str, float] = {}
        mention_rows: list[tuple] = []
        if not cut():
            mention_rows = conn.execute(
                f"SELECT DISTINCT unit_id, canon FROM entity_mentions"
                f" WHERE scope_id=? AND generation<=? AND canon IN ({ph})",
                (ctx.scope_id, ctx.generation, *canons),
            ).fetchall()
        for uid, canon in mention_rows:
            df = df_map.get(canon)
            if df is None:
                if cut():
                    break
                df = conn.execute(
                    "SELECT COUNT(DISTINCT unit_id) FROM entity_mentions"
                    " WHERE scope_id=? AND generation<=? AND canon=?",
                    (ctx.scope_id, ctx.generation, canon),
                ).fetchone()[0]
                df_map[canon] = df
            idf = math.log(1.0 + (n_units - df + 0.5) / (df + 0.5))
            scores[uid] = scores.get(uid, 0.0) + max(idf, 0.0)
        seeds.extend(
            u for u, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[
                :SEED_TOP_V1
            ]
        )
    terms = [t.term for t in getattr(qv.norm, "terms", ()) if t.channel == "text"]
    if terms and has_table(conn, "unit_fts") and has_table(conn, "units"):
        try:
            match = " OR ".join(f'"{t}"' for t in terms[:16])
            rowids: list[int] = []
            if not cut():
                rowids = [
                    r[0]
                    for r in conn.execute(
                        "SELECT rowid FROM unit_fts WHERE unit_fts MATCH ?"
                        " ORDER BY rank LIMIT ?",
                        (match, SEED_TOP_V1),
                    ).fetchall()
                ]
            # unit_fts.rowid == units.rowid (units_jobs write contract) —
            # one batched IN() replaces the per-rowid lookup; the fence
            # keeps only rows visible at the pinned generation, and a
            # multi-generation unit keeps its newest visible row.
            for chunk in _chunks(rowids, _UNIT_CHUNK):
                if cut():
                    break
                ph = ",".join("?" for _ in chunk)
                best: dict[str, int] = {}
                for rid, uid, gen in conn.execute(
                    f"SELECT rowid, unit_id, generation FROM units"
                    f" WHERE rowid IN ({ph}) AND scope_id=?"
                    f" AND generation<=?",
                    (*chunk, ctx.scope_id, ctx.generation),
                ):
                    cur = best.get(uid)
                    if cur is None or gen > cur:
                        best[uid] = gen
                seeds.extend(sorted(best))
        except sqlite3.Error:
            pass  # FTS absent/different tokenizer → entity seeds only
    seen: set[str] = set()
    out = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:SEED_CAP_V1]


def _term_df_map(
    conn: sqlite3.Connection, scope_id: str, generation: int, terms: list[str]
) -> dict[str, int]:
    """``term -> max df`` across fields from maintained ``lex_df`` — the
    same generation-fenced cumulative-snapshot resolution lexical uses
    (latest row per (term, field[, stats_version]) at/below the pin)."""
    out: dict[str, int] = {}
    if not terms or not has_table(conn, "lex_df"):
        return out
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(lex_df)")}
    except sqlite3.Error:
        return out
    has_sver = "stats_version" in cols
    sel = (
        "term, field, stats_version, generation, df"
        if has_sver
        else "term, field, generation, df"
    )
    order = (
        "term, field, stats_version, generation DESC"
        if has_sver
        else "term, field, generation DESC"
    )
    for chunk in _chunks(list(terms), _UNIT_CHUNK):
        ph = ",".join("?" for _ in chunk)
        try:
            rows = conn.execute(
                f"SELECT {sel} FROM lex_df WHERE scope_id=?"
                f" AND generation<=? AND term IN ({ph}) ORDER BY {order}",
                (scope_id, generation, *chunk),
            ).fetchall()
        except sqlite3.Error:
            return out
        seen: set = set()
        for row in rows:
            if has_sver:
                term, field, sver, _gen, df = row
            else:
                term, field, _gen, df = row
                sver = ""
            key = (str(term), str(field), str(sver))
            if key in seen:
                continue  # older snapshot of the same df key
            seen.add(key)
            t, d = str(term), int(df)
            if t not in out or d > out[t]:
                out[t] = d
    return out


def _rarest_term_seed(
    ctx: Any,
    conn: sqlite3.Connection,
    qv: QueryViewV7,
    elig_set: Any,
    cut: Any,
    room: int,
) -> tuple[list[str], Optional[dict]]:
    """The V75-04.04 rarest-term seed (V8-05.07): eligible postings of
    the query's lowest-df content term (O-Mem clue ``argmax 1/df``),
    bounded by the seed cap.

    df comes from maintained ``lex_df`` (the cheap honest proxy — the
    eligible-df postings scan belongs to the lexical lane); postings come
    from a single ``unit_fts`` MATCH filtered through the eligible set.
    Absent tables or an unevaluable term contribute nothing and are
    reported honestly — never fabricated seeds.
    """
    terms = [
        t.term
        for t in getattr(qv.norm, "terms", ())
        if t.channel == "text" and t.term
    ]
    if not terms or room <= 0:
        return [], None
    if not (has_table(conn, "lex_df") and has_table(conn, "unit_fts")
            and has_table(conn, "units")):
        return [], {"skipped": "tables_absent"}
    df_map = _term_df_map(conn, ctx.scope_id, ctx.generation, sorted(set(terms)))
    cands = [(d, t) for t, d in df_map.items() if d > 0]
    if not cands:
        return [], {"skipped": "no_df"}
    df, term = min(cands)  # lowest df; term asc breaks ties
    try:
        if cut():
            return [], {"term": term, "df": df, "skipped": "deadline"}
        rowids = [
            r[0]
            for r in conn.execute(
                "SELECT rowid FROM unit_fts WHERE unit_fts MATCH ?"
                " ORDER BY rank LIMIT ?",
                (f'"{term}"', SEED_CAP_V1 * 2),
            ).fetchall()
        ]
        ids: list[str] = []
        seen: set[str] = set()
        for chunk in _chunks(rowids, _UNIT_CHUNK):
            if cut():
                break
            ph = ",".join("?" for _ in chunk)
            for rid, uid, gen in sorted(
                conn.execute(
                    f"SELECT rowid, unit_id, generation FROM units"
                    f" WHERE rowid IN ({ph}) AND scope_id=?"
                    f" AND generation<=?",
                    (*chunk, ctx.scope_id, ctx.generation),
                ).fetchall(),
                key=lambda r: (str(r[1]), -(r[2] or 0)),
            ):
                if uid in seen or uid not in elig_set:
                    continue
                seen.add(uid)
                ids.append(uid)
                if len(ids) >= room:
                    break
            if len(ids) >= room:
                break
        return ids, {"term": term, "df": df, "seeds": len(ids)}
    except sqlite3.Error:
        return [], {"term": term, "df": df, "skipped": "fts_error"}


# ---------------------------------------------------------------------------
# The lane
# ---------------------------------------------------------------------------


def lane_graph(
    ctx: Any,
    qv: QueryViewV7,
    slice: LaneSlice,
    seeds: Optional[Iterable[Any]] = None,
) -> LaneOutput:
    out = LaneOutput(lane=LaneName.GRAPH.value, status=LaneStatus.OK)
    stats = out.stats
    # Q6 ablation knob (V7-08.12 prep) — resolved first so every exit path
    # reports the enforced cap. None = the declared 3-round bound.
    hop_cap = _resolve_hop_cap(ctx)
    stats["hop_cap"] = hop_cap

    # ---- §23 graph.* arms (V8) — resolved first, reported on every exit.
    k_fan = _arm_int(ctx, ARM_K_FAN, K_FAN_V8, lo=1, hi=1024)
    m_seed = _arm_int(ctx, ARM_M_SEED, M_SEED_V8, lo=1, hi=SEED_CAP_V1)
    contain_ng = _arm_contain(ctx)
    gate_armed = _arm_bool(ctx, ARM_GATE, GRAPH_GATE_V8)
    hop_policy = _arm_hop_policy(ctx)
    edge_types = _arm_edge_types(ctx)
    stats["formula"] = (
        GRAPH_ONEHOP_VERSION if hop_policy == "onehop_additive"
        else GRAPH_PPR_VERSION
    )
    stats["formula_status"] = FORMULA_STATUS
    stats["K_fan"] = k_fan
    stats["M_seed"] = m_seed
    stats["contained_N_g"] = contain_ng
    stats["hop_policy"] = hop_policy
    stats["edge_types"] = list(edge_types)

    expanded = 0
    rounds_completed = 0
    edges_read = 0
    edges_by_round: dict[int, int] = {}

    def _graph_block() -> None:
        # V8-20.03 coverage.graph — emitted on every exit path (zero/
        # null where the traversal never ran).
        stats["graph"] = {
            "nodes_expanded": expanded,
            "rounds_completed": rounds_completed,
            "edges_read": edges_read,
            "contained_N_g": contain_ng,
        }
        stats["nodes_expanded"] = expanded
        stats["rounds_completed"] = rounds_completed
        stats["edges_read"] = edges_read

    # ---- need-gate (V8-05.05): the scheduler-plumbed decision hook -----
    decision = _gate_decision(
        (getattr(ctx, "manifest", None) or {}).get(GATE_DECISION_KEY)
    )
    stats["gate"] = {
        "armed": gate_armed,
        "decision": decision,
        "ran": True,
    }
    if gate_armed and decision is not None and decision.get("needed") is False:
        stats["gate"]["ran"] = False
        _graph_block()
        out.status = LaneStatus.SKIPPED
        out.reason = "not_needed"
        return out

    conn = _conn_of(ctx)
    if conn is None or not has_table(conn, "graph_edges") or not has_table(
        conn, "units"
    ):
        _graph_block()
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "units/graph_edges tables absent"
        return out

    deadline = _Deadline(slice.deadline_ms)

    def cut() -> bool:
        """The V8-05.04 deadline gate — consult before every SQL
        statement and at least every 64 expanded nodes; no statement may
        start after a failed check (K24)."""
        if deadline.expired():
            stats["deadline"] = True
            return True
        return False

    if cut():
        _graph_block()
        out.status = LaneStatus.DEADLINE
        out.reason = "deadline"
        return out

    # ---- eligibility set (V8-05.02) — before any traversal work -------
    # A missing/invalid set denies the lane outright: traversal filtered
    # by anything less than the request's set is traversal unfiltered.
    elig_set = _eligibility_set(ctx)
    stats["eligibility"] = "set" if elig_set is not None else "unavailable"
    if elig_set is None:
        _graph_block()
        out.status = LANE_STATUS_ERROR
        out.reason = "eligibility_set_unavailable"
        return out

    # ---- seeds (V8-05.07) ----------------------------------------------
    manifest = getattr(ctx, "manifest", None) or {}
    manifest_seeds = manifest.get("seeds") or None
    if seeds is not None:
        stats["seed_source"] = "provided"
        raw_seeds = list(seeds)
    elif manifest_seeds:
        stats["seed_source"] = "manifest"
        raw_seeds = list(manifest_seeds)
    else:
        stats["seed_source"] = "derived"
        raw_seeds = _derive_seeds(ctx, conn, qv, cut)
    seed_ids: list[str] = []
    seen: set[str] = set()
    for item in raw_seeds:
        uid = _seed_id(item)
        if uid and uid not in seen:
            seen.add(uid)
            seed_ids.append(uid)
        if len(seed_ids) >= m_seed:
            break
    # V75-04.04 rarest-term seed — extends the fused top-M_seed, drawn
    # from the eligible set, bounded by the absolute seed cap.
    stats["rarest_seed"] = None
    if len(seed_ids) < SEED_CAP_V1:
        extra, rarest = _rarest_term_seed(
            ctx, conn, qv, elig_set, cut, SEED_CAP_V1 - len(seed_ids)
        )
        stats["rarest_seed"] = rarest
        for uid in extra:
            if uid not in seen:
                seen.add(uid)
                seed_ids.append(uid)
    stats["seeds"] = len(seed_ids)

    # ---- unit rows, batched (V8-05.02) ---------------------------------
    # Latest-visible-generation row per unit, fetched in one IN() per
    # 400-chunk; eligibility during traversal is the set membership above
    # plus this row's visibility+scope verdict — the same batched rows,
    # never a per-node SELECT (the D8-01 N+1).
    unit_cache: dict[str, Optional[dict]] = {}

    def fill_rows(uids: Iterable[str]) -> None:
        missing = [u for u in dict.fromkeys(uids) if u not in unit_cache]
        for chunk in _chunks(missing, _UNIT_CHUNK):
            if cut():
                return  # no statement starts after a failed check
            ph = ",".join("?" for _ in chunk)
            best: dict[str, dict] = {}
            for vals in conn.execute(
                _UNIT_SQL.format(ph), (*chunk, ctx.generation)
            ):
                d = dict(zip(_UNIT_COLS, vals))
                cur = best.get(d["unit_id"])
                if cur is None or (d["generation"] or 0) > (
                    cur["generation"] or 0
                ):
                    best[d["unit_id"]] = d
            for u in chunk:
                unit_cache[u] = best.get(u)

    def visible(uid: str) -> bool:
        """Row-level fence on the batched row: the unit must materialize
        at/below the pinned generation in this scope. The eligible set is
        the authorization half; this is the projection half."""
        row = unit_cache.get(uid)
        if row is None:
            return False
        if row.get("scope_id") != ctx.scope_id:
            return False
        gen = row.get("generation")
        if isinstance(gen, int) and gen > ctx.generation:
            return False
        return True

    def admitted(uid: str) -> bool:
        return uid in elig_set and visible(uid)

    fill_rows(seed_ids)
    seeds_elig = [s for s in seed_ids if admitted(s)]
    stats["eligible_seeds"] = len(seeds_elig)
    if not seeds_elig:
        stats["hop_hist"] = {}
        stats["hop2_only"] = 0
        _graph_block()
        out.status = LaneStatus.SKIPPED
        out.reason = "no_eligible_seeds"
        return out

    # hop_cap == 0 is the Q6 graph-off arm: the seed phase ran (its stats
    # are reported above) but expansion is disabled — emit nothing with an
    # honest skipped status rather than pretending the traversal found
    # nothing. An empty edge_types arm disables traversal identically.
    if hop_cap == 0 or not edge_types:
        stats["hop_hist"] = {0: len(seeds_elig)}
        stats["hop2_only"] = 0
        _graph_block()
        out.status = LaneStatus.SKIPPED
        out.reason = (
            f"disabled:{GRAPH_MAX_HOPS_KEY}=0"
            if hop_cap == 0
            else f"disabled:{ARM_EDGE_TYPES}=[]"
        )
        return out

    # ---- bounded expansion (§21.3) --------------------------------------
    max_rounds = (
        1
        if hop_policy == "onehop_additive"
        else (EXPANSION_ROUNDS_V1 if hop_cap is None else hop_cap)
    )
    visited: dict[str, int] = {s: 0 for s in seeds_elig}  # uid -> round found
    frontier = list(seeds_elig)
    truncated = False
    # Every deduped traversal edge ever read — the onehop scorer's input.
    trav_map: dict[tuple[str, str, str], tuple[str, str, str, float]] = {}
    types_sql = ",".join("?" for _ in edge_types)
    # K_fan per frontier node per direction, applied IN SQL after the
    # latest-generation dedupe: ROW_NUMBER windows resolve each logical
    # edge to its newest visible row, then keep the node's top-K_fan by
    # (weight DESC, peer ASC, type ASC) — a total order, so the read set
    # is bit-identical across runs (V8-05.03, K23).
    edge_sql = {
        "src": (
            "SELECT src_unit, dst_unit, type, weight, generation FROM ("
            " SELECT src_unit, dst_unit, type, weight, generation,"
            " ROW_NUMBER() OVER (PARTITION BY src_unit"
            " ORDER BY weight DESC, dst_unit ASC, type ASC) AS fan"
            " FROM ("
            "  SELECT src_unit, dst_unit, type, weight, generation,"
            "  ROW_NUMBER() OVER (PARTITION BY src_unit, type, dst_unit"
            "   ORDER BY generation DESC) AS gen_rn"
            "  FROM graph_edges WHERE scope_id=? AND generation<=?"
            f"  AND type IN ({types_sql}) AND src_unit IN ({{}})"
            " ) WHERE gen_rn = 1"
            ") WHERE fan <= ? ORDER BY src_unit, type, dst_unit"
        ),
        "dst": (
            "SELECT src_unit, dst_unit, type, weight, generation FROM ("
            " SELECT src_unit, dst_unit, type, weight, generation,"
            " ROW_NUMBER() OVER (PARTITION BY dst_unit"
            " ORDER BY weight DESC, src_unit ASC, type ASC) AS fan"
            " FROM ("
            "  SELECT src_unit, dst_unit, type, weight, generation,"
            "  ROW_NUMBER() OVER (PARTITION BY src_unit, type, dst_unit"
            "   ORDER BY generation DESC) AS gen_rn"
            "  FROM graph_edges WHERE scope_id=? AND generation<=?"
            f"  AND type IN ({types_sql}) AND dst_unit IN ({{}})"
            " ) WHERE gen_rn = 1"
            ") WHERE fan <= ? ORDER BY dst_unit, type, src_unit"
        ),
    }

    for rnd in range(1, max_rounds + 1):
        if not frontier or truncated:
            break
        if cut():
            break
        # -- capped, chunked edge fetch (V8-05.03); every statement gated.
        raw: list[tuple] = []
        for chunk in _chunks(frontier, _EDGE_CHUNK):
            if cut():
                break
            ph = ",".join("?" for _ in chunk)
            raw += conn.execute(
                edge_sql["src"].format(ph),
                (ctx.scope_id, ctx.generation, *edge_types, *chunk, k_fan),
            ).fetchall()
            if cut():
                break
            raw += conn.execute(
                edge_sql["dst"].format(ph),
                (ctx.scope_id, ctx.generation, *edge_types, *chunk, k_fan),
            ).fetchall()
        # Dedupe across the two direction reads (an intra-frontier edge
        # lands in both) — consumes only the capped rows.
        rows = _latest_edges(raw)
        edges_read += len(rows)
        edges_by_round[rnd] = edges_by_round.get(rnd, 0) + len(rows)
        for e in rows:
            trav_map[(e[0], e[2], e[1])] = e
        # -- next frontier: set-membership filter first (zero SQL), then
        #    the round's batched row fetch, then admission in edge order.
        cand: list[str] = []
        seen_c: set[str] = set()
        for src, dst, _t, _w in rows:
            other = dst if src in visited else src
            if other in visited or other in seen_c or other not in elig_set:
                continue
            seen_c.add(other)
            cand.append(other)
        fill_rows(cand)
        nxt: list[str] = []
        for other in cand:
            if (expanded & (_DEADLINE_CHECK_EVERY - 1)) == (
                _DEADLINE_CHECK_EVERY - 1
            ) and cut():
                truncated = True
                break
            if len(visited) >= FRONTIER_CAP_V1:
                truncated = True
                break
            if not visible(other):  # held/purged/foreign/future-gen units
                continue             # are never traversed (V7-08.11)
            visited[other] = rnd
            nxt.append(other)
            expanded += 1
        if not stats.get("deadline"):
            rounds_completed = rnd
        frontier = nxt
    stats["rounds_run"] = max(visited.values(), default=0)
    stats["visited"] = len(visited)
    stats["frontier_truncated"] = truncated
    # Q6 instrumentation: first-discovery depth over the visited set —
    # seeds sit at depth 0 and are never emitted as candidates.
    hop_hist: dict[int, int] = {}
    for depth in visited.values():
        hop_hist[depth] = hop_hist.get(depth, 0) + 1
    stats["hop_hist"] = {d: hop_hist[d] for d in sorted(hop_hist)}
    stats["edges_by_round"] = {
        r: edges_by_round[r] for r in sorted(edges_by_round)
    }
    out.examined = edges_read
    out.eligible = len(visited)
    trav_rows = sorted(trav_map.values())

    seed_set = set(seeds_elig)
    # Personalization (bfs3+power) — declared before the policy branch so
    # signals compute identically on both arms.
    p = {s: 1.0 / len(seeds_elig) for s in seeds_elig}
    shared: dict[str, float] = {}
    maxlink: dict[str, float] = {}

    if hop_policy == "onehop_additive":
        # ---- V8-05.08 arm: tanh(0.5·shared) + max(link) -----------------
        # One expansion round ran above; scoring is additive over the
        # traversal edges already read — no induced-subgraph query, no
        # power iterations. ``shared`` is the summed effective weight
        # between the candidate and the seed set, ``link`` the strongest
        # single seed edge.
        link_edge: dict[str, tuple[str, str, float]] = {}
        fam_mass: dict[str, dict[str, float]] = {}
        adj = {}
        fam = {}
        for src, dst, type_v, w in trav_rows:
            try:
                et = EdgeType(type_v)
            except ValueError:
                continue
            eff_w = EDGE_WEIGHTS_V1.get(et)
            if eff_w is None:
                continue
            eff = eff_w * float(w or 0.0)
            family = _FAMILY.get(et, "update")
            for a, b in ((src, dst), (dst, src)):
                adj.setdefault(a, {})[b] = (
                    adj.setdefault(a, {}).get(b, 0.0) + eff
                )
                fam.setdefault(a, {}).setdefault(family, {})
                fam[a][family][b] = fam[a][family].get(b, 0.0) + eff
            if src in seed_set and dst not in seed_set:
                u, seed = dst, src
            elif dst in seed_set and src not in seed_set:
                u, seed = src, dst
            else:
                continue
            shared[u] = shared.get(u, 0.0) + eff
            fam_mass.setdefault(u, {})
            fam_mass[u][family] = fam_mass[u].get(family, 0.0) + eff
            if eff > maxlink.get(u, -1.0):
                maxlink[u] = eff
                link_edge[u] = (seed, family, eff)
        stats["edges"] = len(trav_rows)
        r = {
            u: math.tanh(0.5 * shared[u]) + maxlink.get(u, 0.0)
            for u in visited
            if u not in seed_set and u in shared
        }
        stats["iterations"] = 0
        stats["seed_scores"] = {}
        path = {
            u: (link_edge[u][0], u) if u in link_edge else (u,)
            for u in r
        }
        fam_lookup = fam_mass
    else:
        # ---- W over the visited subgraph (bfs3+power) --------------------
        ids = sorted(visited)
        ph = ",".join("?" for _ in ids)
        wrows: list[tuple] = []
        if not cut() and ids:
            wrows = _latest_edges(
                conn.execute(
                    f"SELECT src_unit, dst_unit, type, weight, generation"
                    f" FROM graph_edges WHERE scope_id=? AND generation<=?"
                    f" AND type IN ({types_sql})"
                    f" AND src_unit IN ({ph}) AND dst_unit IN ({ph})"
                    f" ORDER BY src_unit, type, dst_unit, generation",
                    (
                        ctx.scope_id,
                        ctx.generation,
                        *edge_types,
                        *ids,
                        *ids,
                    ),
                ).fetchall()
            )
        stats["edges"] = len(wrows)
        # adjacency[u][v] = effective weight; fam[u][family][v] same per family.
        adj = {}
        fam = {}
        for src, dst, type_v, w in wrows:
            try:
                et = EdgeType(type_v)
            except ValueError:
                continue
            eff = EDGE_WEIGHTS_V1.get(et)
            if eff is None:
                continue
            eff = eff * float(w or 0.0)
            family = _FAMILY.get(et, "update")
            for a, b in ((src, dst), (dst, src)):  # bidirectional relevance
                adj.setdefault(a, {})[b] = (
                    adj.setdefault(a, {}).get(b, 0.0) + eff
                )
                fam.setdefault(a, {}).setdefault(family, {})
                fam[a][family][b] = fam[a][family].get(b, 0.0) + eff

        # ---- power iterations (sorted iteration → bit-deterministic) -----
        r = {u: p.get(u, 0.0) for u in ids}
        fam_mass = {}
        iters_done = 0
        for it in range(POWER_ITERS_V1):
            if cut():
                break
            r2 = {u: DAMPING_V1 * p.get(u, 0.0) for u in ids}
            fm: dict[str, dict[str, float]] = {u: {} for u in ids}
            for u in ids:
                ru = r.get(u, 0.0)
                if ru == 0.0:
                    continue
                out_edges = adj.get(u)
                if not out_edges:
                    continue
                tot = 0.0
                for v in sorted(out_edges):
                    tot += out_edges[v]
                if tot <= 0.0:
                    continue
                scale = (1.0 - DAMPING_V1) * ru / tot
                for v in sorted(out_edges):
                    r2[v] += scale * out_edges[v]
                for family, fmap in fam.get(u, {}).items():
                    for v in sorted(fmap):
                        fm[v][family] = fm[v].get(family, 0.0) + scale * fmap[v]
            r = r2
            fam_mass = fm
            iters_done += 1
        stats["iterations"] = iters_done
        stats["seed_scores"] = {s: r.get(s, 0.0) for s in seeds_elig}

        # ---- widest-path summaries for explain ---------------------------
        # Maximize the product of effective edge weights seed→unit.
        # Effective weights can exceed 1.0 (same-pair contributions sum),
        # so paths are kept simple and capped at PATH_HOP_CAP_V1 hops —
        # both bounds keep the search finite and deterministic.
        path = {}
        if not cut():
            best: dict[str, tuple[float, tuple[str, ...]]] = {
                s: (1.0, (s,)) for s in seeds_elig
            }
            heap = [(-1.0, 0, s, (s,)) for s in seeds_elig]
            heapq.heapify(heap)
            while heap:
                neg, hops, u, pth = heapq.heappop(heap)
                if best.get(u, (0.0, ()))[1] != pth:
                    continue
                if hops >= PATH_HOP_CAP_V1:
                    continue
                for v in sorted(adj.get(u, {})):
                    if v in pth:
                        continue  # simple paths only
                    cand_w = (-neg) * adj[u][v]
                    cur = best.get(v)
                    npth = pth + (v,)
                    if cur is None or cand_w > cur[0] or (
                        cand_w == cur[0] and npth < cur[1]
                    ):
                        best[v] = (cand_w, npth)
                        heapq.heappush(heap, (-cand_w, hops + 1, v, npth))
            path = {u: b[1] for u, b in best.items()}
        fam_lookup = fam_mass

    # ---- candidates -----------------------------------------------------
    order = sorted(
        (u for u in visited if u not in seed_set),
        key=lambda u: (-r.get(u, 0.0), u),
    )
    cap = max(0, int(slice.cap)) if slice.cap is not None else len(order)
    for rank, uid in enumerate(order[:cap], start=1):
        row = unit_cache.get(uid) or {}
        arriving = sum(fam_lookup.get(uid, {}).values())
        comps = {
            k: (fam_lookup.get(uid, {}).get(k, 0.0) / arriving if arriving else 0.0)
            for k in _COMPONENT_KEYS
        }
        hop_path = path.get(uid, (uid,))
        hop_ann = []
        for a, b in zip(hop_path, hop_path[1:]):
            # annotate the hop with its strongest edge family
            eff_w = adj.get(a, {}).get(b, 0.0)
            fams = [
                (fmap.get(b, 0.0), f) for f, fmap in fam.get(a, {}).items()
            ]
            fams.sort(key=lambda t: (-t[0], t[1]))
            hop_ann.append(
                {"unit": b, "via": fams[0][1] if fams else None, "w": eff_w}
            )
        signals = {
            "components": comps,
            "component_mass": dict(fam_lookup.get(uid, {})),
            "seed_mass": DAMPING_V1 * p.get(uid, 0.0),
            "path": [{"unit": hop_path[0], "via": None, "w": 1.0}] + hop_ann,
            "hops": visited.get(uid, 0),
            "round": visited.get(uid, 0),
            "hop_policy": hop_policy,
            # V8-05.06/V8-11.04: the containment arm in force — fusion
            # bars graph-only candidates ranked beyond N_g.
            "contain_N_g": contain_ng,
        }
        if hop_policy == "onehop_additive":
            signals["shared"] = shared.get(uid, 0.0)
            signals["max_link"] = maxlink.get(uid, 0.0)
        out.candidates.append(
            CandidateV7(
                unit_id=uid,
                source_id=str(row.get("source_id") or ""),
                revision=int(row.get("revision") or 0),
                lane=LaneName.GRAPH.value,
                rank=rank,
                raw_score=r.get(uid, 0.0),
                signals=signals,
            )
        )
    # Q6 metric (SPEC_V7_5 §05): emitted candidates unreachable at
    # max_hops=1 — first discovered at depth >= 2.
    stats["hop2_only"] = sum(
        1 for c in out.candidates if visited.get(c.unit_id, 0) >= 2
    )

    _graph_block()
    if stats.get("deadline") and out.status is LaneStatus.OK:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline"
    if truncated and out.status is LaneStatus.OK:
        out.reason = "frontier_cap"
    return out


__all__ = [
    "ARM_CONTAIN_N_G",
    "ARM_EDGE_TYPES",
    "ARM_GATE",
    "ARM_HOP_POLICY",
    "ARM_K_FAN",
    "ARM_M_SEED",
    "CONTAIN_N_G_V8",
    "DECLARED_EDGE_TYPES",
    "DAMPING_V1",
    "EDGE_WEIGHTS_V1",
    "EXPANSION_ROUNDS_V1",
    "FORMULA_STATUS",
    "FRONTIER_CAP_V1",
    "GATE_DECISION_KEY",
    "GRAPH_GATE_V8",
    "GRAPH_MAX_HOPS_KEY",
    "GRAPH_ONEHOP_VERSION",
    "GRAPH_PPR_VERSION",
    "HOP_POLICIES",
    "HOP_POLICY_V8",
    "K_FAN_V8",
    "LANE_STATUS_ERROR",
    "M_SEED_V8",
    "PATH_HOP_CAP_V1",
    "POWER_ITERS_V1",
    "SEED_CAP_V1",
    "SEED_TOP_V1",
    "lane_graph",
]
