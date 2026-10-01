"""V8 dense block compaction + dense-lane arms (SPEC_V8 §08, §21.5).

V8-08.01: the ``source_embed`` write path appends one block per
(source, revision) commit, so a busy scope fragments ``unit_vectors_block``
into thousands of single-row blocks and ``matrix.scan`` pays per-block
Python overhead per row (D8-09 — measured 2,877 blocks / 4,143 rows).
``dense_compact`` is the maintenance pass that rewrites one vector space
``(encoder_id, scope_id, generation)`` into contiguous blocks of up to
:data:`MAX_BLOCK_ROWS` rows.

§21.5 algorithm::

    plan:   list blocks of space; if n_blocks <= B_max and
            mean_rows >= 256: stop
    commit: BEGIN IMMEDIATE
            keys = live block row keys of the space, re-read now
            rows = decode(keys) ordered by unit_id; quant per V7-07.06
            write blocks of <= MAX_BLOCK_ROWS at block_no = max_existing+1 …
            delete the old block_no range
            meta dense_compaction_epoch/<space> += 1
            COMMIT
    invariant: the multiset of (key, vector) in the space is unchanged
               except for purged keys

Closure (V8-19.03): the purge path already rewrites/deletes block rows
inside its own write transaction, so re-reading rowmaps inside
``BEGIN IMMEDIATE`` is the "never resurrect a purged key" guarantee —
keys swept between plan and commit are simply absent. ``f32`` oracle
rows are untouched. The epoch is recorded at
``meta.dense_compaction_epoch/<encoder>/<scope>/<generation>`` (``/``
cannot appear inside a ``require_id`` token, so the key parses
unambiguously) and reported to the dense lane as
``coverage.dense.compaction_epoch``.

Encoder-space separation (V8-08.03): spaces are keyed by the full
``(encoder_id, scope_id, generation)`` triple and compacted strictly
per-space — the ``:hdr:`` tag inside ``encoder_id`` (V7-07.09) means a
header-format change mints a *new* space, never a merge.

Arms (§23): ``dense.B_max`` = 64 (trigger), ``dense.embed_batch`` =
{512 rows / 250 ms} (writer coalescing bound — V8-08.02),
``dense.N_d`` = 10 (guaranteed dense slots — V8-08.04).

Ownership boundary: the ``dense_compact`` *job kind* (``JobKind``
member, queue CHECK, dispatch registration, enqueue site) and the
``source_embed`` write-side coalescing live in ``core/types.py`` /
``ingest.py`` / ``jobs/source_jobs.py`` — outside this module. This
module ships the compaction machinery itself plus
:func:`handle_dense_compact`, a drop-in handler the dispatcher can
register unchanged.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any, Iterator, NamedTuple, Optional

from ...core.types import ErrorCode, VerbatimError, require_id
from ...core.types_v7 import QuantMode
from ...embeddings import matrix as _matrix

# --- §23 arm defaults -------------------------------------------------------

#: ``dense.B_max`` — compaction triggers when a space holds MORE than this
#: many blocks (or its mean rows/block falls below ``MIN_MEAN_ROWS``).
B_MAX_DEFAULT = 64
#: §21.5 second trigger: mean stored rows per block below this means
#: fragmentation even when the block count is under ``B_max``.
MIN_MEAN_ROWS = 256
#: ``dense.embed_batch`` (V8-08.02): the write-side coalescing bound —
#: pending units of one space merge across revisions up to this many rows
#: or this much queue age before ``write_block``. Constants live here; the
#: ``source_embed`` writer consumes them (cross-module handoff).
EMBED_BATCH_ROWS = 512
EMBED_BATCH_MAX_AGE_MS = 250.0
#: ``dense.N_d`` (V8-08.04): top-N dense candidates clearing the noise
#: floor are emitted under slot reservation (``signals["dense_slot"]``).
#: SPEC_V8_5 §06 don't-ship — 0/137 unique rescues measured; default 0.
N_D_DEFAULT = 0

#: The job kind string the dispatcher registers for this handler. Minted
#: here so the (cross-module) ``JobKind``/CHECK registration has one
#: canonical spelling — ``JobKind.DENSE_COMPACT`` lands with the queue DDL.
DENSE_COMPACT_KIND = "dense_compact"

_EPOCH_PREFIX = "dense_compaction_epoch/"


def _present(conn: sqlite3.Connection, name: str) -> bool:
    """Uncached schema-object probe.

    ``has_table``'s positive memo reports a table dropped mid-session as
    still present — a fine trade on hot read paths, wrong on maintenance
    paths where a real drop must degrade honestly (``None``/no-op)
    instead of crashing on a stale positive.
    """
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master"
            " WHERE type IN ('table','view') AND name = ?",
            (name,),
        ).fetchone()
        is not None
    )


class SpaceRef(NamedTuple):
    """One compactable vector space — the ``unit_vectors_block`` PK prefix."""

    encoder_id: str
    scope_id: str
    generation: int


class CompactionPlan(NamedTuple):
    """§21.5 plan phase: block inventory + trigger decision for one space."""

    space: SpaceRef
    n_blocks: int
    n_rows: int
    mean_rows: float
    needed: bool
    reason: str  # "ok" | "below_threshold" | "empty_space"


def epoch_meta_key(encoder_id: str, scope_id: str, generation: int) -> str:
    """``meta`` key recording the compaction epoch of one space."""
    return f"{_EPOCH_PREFIX}{encoder_id}/{scope_id}/{generation}"


def compaction_epoch(
    conn: sqlite3.Connection, encoder_id: str, scope_id: str, generation: int
) -> int:
    """The recorded compaction epoch of ``(encoder_id, scope_id, generation)``.

    ``0`` when the space was never compacted (or ``meta`` is absent) —
    the lane reports this as ``coverage.dense.compaction_epoch``.
    """
    if not _present(conn, "meta"):
        return 0
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key = ?",
        (epoch_meta_key(encoder_id, scope_id, generation),),
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(json.loads(row[0]))
    except (TypeError, ValueError):
        return 0


def _bump_epoch(
    conn: sqlite3.Connection, encoder_id: str, scope_id: str, generation: int
) -> Optional[int]:
    """``dense_compaction_epoch/<space> += 1`` inside the caller's tx."""
    if not _present(conn, "meta"):
        return None
    key = epoch_meta_key(encoder_id, scope_id, generation)
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key = ?", (key,)
    ).fetchone()
    try:
        nxt = int(json.loads(row[0])) + 1 if row is not None else 1
    except (TypeError, ValueError):
        nxt = 1
    conn.execute(
        "INSERT INTO meta(key, value_json) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        (key, json.dumps(nxt)),
    )
    return nxt


def plan_space(
    conn: sqlite3.Connection,
    encoder_id: str,
    scope_id: str,
    generation: int,
    *,
    b_max: int = B_MAX_DEFAULT,
    min_mean_rows: int = MIN_MEAN_ROWS,
) -> CompactionPlan:
    """§21.5 plan: compact iff the space is fragmented AND reducible.

    Spec triggers: ``n_blocks > b_max`` OR ``mean_rows < min_mean_rows``
    (256). Two fixpoint guards keep the pass idempotent and terminating:

    - a single block cannot become more contiguous, so the mean trigger
      needs ``n_blocks > 1`` — a lone under-full block is already packed;
    - the rewrite only runs when it can actually shrink the block count
      (``n_blocks > ceil(n_rows / MAX_BLOCK_ROWS)``): a space above
      ``b_max`` whose blocks are already full rewrites to the same
      artifact, so it plans ``already_packed`` instead of burning an
      epoch and a rewrite on every maintenance pass.

    Metadata only — rowmaps/data are not decoded. An empty space plans
    ``needed=False`` (nothing to rewrite, no epoch burned).
    """
    row = conn.execute(
        f"SELECT COUNT(*), COALESCE(SUM(n_rows), 0) FROM {_matrix.BLOCK_TABLE}"
        " WHERE encoder_id = ? AND scope_id = ? AND generation = ?",
        (encoder_id, scope_id, generation),
    ).fetchone()
    n_blocks, n_rows = int(row[0]), int(row[1])
    mean = (n_rows / n_blocks) if n_blocks else 0.0
    if n_blocks == 0:
        return CompactionPlan(
            SpaceRef(encoder_id, scope_id, generation),
            n_blocks, n_rows, mean, False, "empty_space",
        )
    fragmented = n_blocks > b_max or (
        n_blocks > 1 and mean < float(min_mean_rows)
    )
    if not fragmented:
        return CompactionPlan(
            SpaceRef(encoder_id, scope_id, generation),
            n_blocks, n_rows, mean, False, "below_threshold",
        )
    ideal = -(-n_rows // _matrix.MAX_BLOCK_ROWS)
    if n_blocks <= ideal:
        return CompactionPlan(
            SpaceRef(encoder_id, scope_id, generation),
            n_blocks, n_rows, mean, False, "already_packed",
        )
    return CompactionPlan(
        SpaceRef(encoder_id, scope_id, generation),
        n_blocks, n_rows, mean, True, "ok",
    )


def list_spaces(
    conn: sqlite3.Connection,
    *,
    scope_id: Optional[str] = None,
    encoder_id: Optional[str] = None,
    generation: Optional[int] = None,
) -> list[SpaceRef]:
    """Distinct ``(encoder_id, scope_id, generation)`` spaces present —
    deterministic order; filters narrow the maintenance job's window."""
    sql = (
        f"SELECT DISTINCT encoder_id, scope_id, generation"
        f" FROM {_matrix.BLOCK_TABLE}"
    )
    conds: list[str] = []
    params: list[Any] = []
    if scope_id is not None:
        conds.append("scope_id = ?")
        params.append(scope_id)
    if encoder_id is not None:
        conds.append("encoder_id = ?")
        params.append(encoder_id)
    if generation is not None:
        conds.append("generation = ?")
        params.append(generation)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY encoder_id, scope_id, generation"
    return [SpaceRef(str(e), str(s), int(g)) for e, s, g in conn.execute(sql, params)]


def compact_space_in_tx(
    conn: sqlite3.Connection,
    encoder_id: str,
    scope_id: str,
    generation: int,
    *,
    b_max: int = B_MAX_DEFAULT,
    min_mean_rows: int = MIN_MEAN_ROWS,
) -> dict[str, Any]:
    """§21.5 commit phase — CALLER holds the write transaction.

    Re-plans and re-reads the live block rows NOW (under the caller's
    write lock): rows purged since the plan were already swept from the
    rowmaps by the purge path's own transaction, so they are absent here
    — the closure invariant (V8-19.03). Rows decode per source block and
    rewrite ordered by ``unit_id`` at ``block_no = max_existing + 1 …``;
    the replaced ``block_no`` range is deleted after the new blocks land.
    """
    if not _present(conn, _matrix.BLOCK_TABLE):
        return {
            "encoder_id": encoder_id,
            "scope_id": scope_id,
            "generation": generation,
            "compacted": False,
            "reason": "no_block_table",
            "blocks_before": 0,
            "blocks_after": 0,
            "rows": 0,
            "epoch": 0,
        }
    plan = plan_space(
        conn, encoder_id, scope_id, generation,
        b_max=b_max, min_mean_rows=min_mean_rows,
    )
    report: dict[str, Any] = {
        "encoder_id": encoder_id,
        "scope_id": scope_id,
        "generation": generation,
        "blocks_before": plan.n_blocks,
        "rows_before": plan.n_rows,
        "compacted": plan.needed,
        "reason": plan.reason,
    }
    if not plan.needed:
        report.update(
            blocks_after=plan.n_blocks,
            rows=plan.n_rows,
            epoch=compaction_epoch(conn, encoder_id, scope_id, generation),
        )
        return report

    # Decode the live space once, keeping each block's packed ``array``
    # resident — per-row ``list[float]`` materialization is avoided so
    # resident bytes stay at the storage representation (V7-07.14).
    blocks = list(_matrix.iter_blocks(conn, encoder_id, scope_id, generation))
    live: dict[str, tuple[_matrix.MatrixBlock, int]] = {}
    for blk in blocks:
        for i, key in enumerate(blk.keys):
            # First-seen in block_no order — the same precedence the scan
            # applies to a duplicated rowmap key.
            if key not in live:
                live[key] = (blk, i)
    # Rows rewrite ordered by unit_id (§21.5); sorted keys are also the
    # deterministic order scan-output ordering depends on.
    keys = sorted(live)
    quant = (
        "int8"
        if len(keys) > _matrix.INT8_ROW_THRESHOLD
        else "f32"
    )  # V7-07.06 threshold on the post-compaction count
    start = max(b.block_no for b in blocks) + 1
    written = 0
    bno = start
    for off in range(0, len(keys), _matrix.MAX_BLOCK_ROWS):
        part = keys[off : off + _matrix.MAX_BLOCK_ROWS]
        rows = []
        for key in part:
            blk, i = live[key]
            if blk.quant == QuantMode.INT8.value:
                # int8 dequantizes through MatrixBlock.row (scale·q →
                # f32; re-quantized byte-identically when the space
                # stays int8 — see module docstring).
                rows.append((key, blk.row(i)))
            else:
                # f32: the packed row slice feeds write_block's
                # f32-round-trip verbatim — no per-row list[float].
                base = i * blk.dims
                rows.append((key, blk.data[base : base + blk.dims]))
        written += _matrix.write_block(
            conn, encoder_id, scope_id, generation, bno, rows, quant=quant
        )
        bno += 1
    old_bnos = [b.block_no for b in blocks]
    for old in old_bnos:
        conn.execute(
            f"DELETE FROM {_matrix.BLOCK_TABLE}"
            " WHERE encoder_id = ? AND scope_id = ?"
            "   AND generation = ? AND block_no = ?",
            (encoder_id, scope_id, generation, old),
        )
    epoch = _bump_epoch(conn, encoder_id, scope_id, generation)
    report.update(
        blocks_after=bno - start,
        rows=written,
        dropped_rows=plan.n_rows - written,
        quant=quant,
        epoch=epoch,
        reason="ok",
    )
    return report


def compact_space(
    store: Any,
    encoder_id: str,
    scope_id: str,
    generation: int,
    *,
    b_max: int = B_MAX_DEFAULT,
    min_mean_rows: int = MIN_MEAN_ROWS,
) -> dict[str, Any]:
    """Plan on a read snapshot, then rewrite the space inside ``store.tx()``.

    Idempotent and re-runnable: a space already at/below both thresholds
    reports ``compacted=False`` without touching a byte. ``store`` needs
    the ``read()``/``tx()`` context-manager contract (``Store`` or the
    test shim).
    """
    require_id(encoder_id, "encoder_id")
    require_id(scope_id, "scope_id")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid generation {generation!r}"
        )
    if not isinstance(b_max, int) or isinstance(b_max, bool) or b_max < 0:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"b_max must be a non-negative int, got {b_max!r}"
        )
    if (
        not isinstance(min_mean_rows, int)
        or isinstance(min_mean_rows, bool)
        or min_mean_rows < 1
    ):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"min_mean_rows must be a positive int, got {min_mean_rows!r}",
        )
    with store.read() as conn:
        if not _present(conn, _matrix.BLOCK_TABLE):
            return {
                "encoder_id": encoder_id,
                "scope_id": scope_id,
                "generation": generation,
                "compacted": False,
                "reason": "no_block_table",
                "blocks_before": 0,
                "blocks_after": 0,
                "rows": 0,
                "epoch": 0,
            }
        plan = plan_space(
            conn, encoder_id, scope_id, generation,
            b_max=b_max, min_mean_rows=min_mean_rows,
        )
    if not plan.needed:
        return {
            "encoder_id": encoder_id,
            "scope_id": scope_id,
            "generation": generation,
            "compacted": False,
            "reason": plan.reason,
            "blocks_before": plan.n_blocks,
            "blocks_after": plan.n_blocks,
            "rows": plan.n_rows,
            "epoch": None,
        }
    with store.tx() as conn:
        # §21.5: keys are re-read under the write lock — the commit's own
        # plan_space re-checks the trigger, so a stale plan (a racing
        # compactor or a purge that emptied the space) degrades to an
        # honest no-op rather than rewriting garbage.
        return compact_space_in_tx(
            conn, encoder_id, scope_id, generation,
            b_max=b_max, min_mean_rows=min_mean_rows,
        )


def dense_compaction_derivation(
    conn: sqlite3.Connection,
    scope_id: str,
    generation: Optional[int] = None,
    *,
    b_max: int = B_MAX_DEFAULT,
    min_mean_rows: int = MIN_MEAN_ROWS,
) -> Optional[str]:
    """V8-19.05 ``coverage.derivations.dense_compaction`` —
    ``"ok" | "partial(n)"``.

    ``n`` counts the scope's latest-generation spaces (per
    ``(encoder_id, scope_id)``, at/below the pinned generation — the
    snapshots a query would actually scan) that currently plan
    compaction. ``None`` when the block table is absent and the status
    cannot be evaluated honestly (V8-23.02). Read-only: this reports the
    debt frontier, it never rewrites.
    """
    if not _present(conn, _matrix.BLOCK_TABLE):
        return None
    if generation is None:
        rows = conn.execute(
            f"SELECT encoder_id, scope_id, MAX(generation)"
            f" FROM {_matrix.BLOCK_TABLE} WHERE scope_id = ?"
            " GROUP BY encoder_id, scope_id",
            (scope_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT encoder_id, scope_id, MAX(generation)"
            f" FROM {_matrix.BLOCK_TABLE}"
            " WHERE scope_id = ? AND generation <= ?"
            " GROUP BY encoder_id, scope_id",
            (scope_id, int(generation)),
        ).fetchall()
    owed = 0
    for enc, scope, gen in rows:
        if plan_space(
            conn, enc, scope, int(gen),
            b_max=b_max, min_mean_rows=min_mean_rows,
        ).needed:
            owed += 1
    return f"partial({owed})" if owed else "ok"


def compact_spaces(
    store: Any,
    spaces: Iterator[SpaceRef],
    *,
    b_max: int = B_MAX_DEFAULT,
    min_mean_rows: int = MIN_MEAN_ROWS,
) -> list[dict[str, Any]]:
    """Compact each space in turn — per-space reports, deterministic order.

    One space's failure never silently merges or skips another: the
    exception propagates after earlier spaces' commits stand (each space
    is its own transaction — never a half-rewritten space).
    """
    reports = []
    for sp in sorted(set(spaces)):
        reports.append(
            compact_space(
                store, sp.encoder_id, sp.scope_id, sp.generation,
                b_max=b_max, min_mean_rows=min_mean_rows,
            )
        )
    return reports


# ---------------------------------------------------------------------------
# dense_compact maintenance job (dispatch registration is a cross-module
# handoff — see module docstring)
# ---------------------------------------------------------------------------


def _opt_str(refs: dict[str, Any], key: str) -> Optional[str]:
    val = refs.get(key)
    if val is None:
        return None
    if not isinstance(val, str) or not val:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"input_refs.{key} must be a non-empty string"
        )
    return val


def _opt_int(refs: dict[str, Any], key: str) -> Optional[int]:
    val = refs.get(key)
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, int):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"input_refs.{key} must be an int"
        )
    return val


# ---------------------------------------------------------------------------
# §23 arm resolution on the write (job) path
# ---------------------------------------------------------------------------
#
# The retrieval ``params`` register does not ride ``ingester.policy`` (the
# write path's ``PolicyContext`` has no params map), so a §23 arm reaches a
# job through the documented write-path carriers, most-local first:
#
#   1. the job's ``input_refs`` under the dotted or underscored key
#      (``dense.B_max``, ``dense_b_max``) — the per-job declaration;
#   2. a ``params``/``knobs``/``arms`` mapping inside ``input_refs`` keyed
#      by the same names — the policy-document register channel embedded
#      in a job declaration;
#   3. ``params``/``knobs``/``arms`` maps on ``ingester.retrieval_policy``
#      then ``ingester.policy`` — a retrieval policy hung on the ingester
#      resolves exactly like the query-time register;
#   4. a ``cfg.jobs.<cfg_attr>`` deployment field when the arm names one
#      (``config.JobsConfig`` is the operator-level carrier);
#   5. the caller's §23 prior ``default``.
#
# ``None`` values are absent everywhere in this chain — a declared
# ``None`` never masks an outer carrier (the policy ``params`` convention
# treats ``None`` the same way).  Callers own domain validation; a
# malformed value must raise, never silently revert to the prior.


def job_arm(
    refs: Optional[dict],
    ingester: Any,
    *keys: str,
    cfg_attr: Optional[str] = None,
    default: Any = None,
) -> Any:
    """Resolve one §23 arm value through the write-path carriers.

    ``keys`` are the accepted spellings (canonical dotted name first).
    See the module block above for the carrier order.
    """
    refs = refs or {}
    for key in keys:
        val = refs.get(key)
        if val is not None:
            return val
    for carrier in ("params", "knobs", "arms"):
        mapping = refs.get(carrier)
        if isinstance(mapping, dict):
            for key in keys:
                val = mapping.get(key)
                if val is not None:
                    return val
    for holder in (
        getattr(ingester, "retrieval_policy", None),
        getattr(ingester, "policy", None),
    ):
        if holder is None:
            continue
        for carrier in ("params", "knobs", "arms"):
            mapping = getattr(holder, carrier, None)
            if isinstance(mapping, dict):
                for key in keys:
                    val = mapping.get(key)
                    if val is not None:
                        return val
    if cfg_attr:
        val = getattr(
            getattr(getattr(ingester, "cfg", None), "jobs", None),
            cfg_attr,
            None,
        )
        if val is not None:
            return val
    return default


def _arm_int(value: Any, name: str, minimum: int) -> int:
    """Strict non-negative-int arm coercion — floats must be integral."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"{name} must be an integer >= {minimum}, got {value!r}",
        )
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"{name} must be an integral value, got {value!r}",
            )
        value = int(value)
    if value < minimum:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"{name} must be >= {minimum}, got {value!r}",
        )
    return int(value)


def resolve_b_max(refs: Optional[dict], ingester: Any) -> int:
    """``dense.B_max`` (V8-08.01) — the compaction trigger arm.

    Read order: ``input_refs`` (``dense.B_max``/``dense_b_max``/
    legacy ``b_max``, or an embedded ``params`` map) → policy-carrier
    maps on the ingester → ``cfg.jobs.dense_compact_b_max`` → the prior
    :data:`B_MAX_DEFAULT`. ``0`` is a real arm value (compact whenever
    the space holds any block), never an "absent" alias.
    """
    raw = job_arm(
        refs,
        ingester,
        "dense.B_max",
        "dense_b_max",
        "b_max",
        cfg_attr="dense_compact_b_max",
    )
    if raw is None:
        return B_MAX_DEFAULT
    return _arm_int(raw, "dense.B_max", 0)


class EmbedBatchBound(NamedTuple):
    """Resolved ``dense.embed_batch`` (V8-08.02).

    ``rows`` bounds both the pending unit rows one commit may coalesce
    and the texts per ``encoder.encode`` call; ``max_age_ms`` is the
    queue-age bound — an admitted sibling already waiting this long
    closes the batch after itself (a deadline flush, never a skip).
    ``None`` in either slot means the bound is off.
    """

    rows: Optional[int]
    max_age_ms: Optional[float]


def _arm_num(value: Any, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{name} must be numeric, got {value!r}"
        )
    out = float(value)
    if not math.isfinite(out) or out < 0.0 or (out == 0.0 and not allow_zero):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"{name} must be a finite number >= 0, got {value!r}",
        )
    return out


def embed_batch_bound(
    refs: Optional[dict], ingester: Any
) -> EmbedBatchBound:
    """Resolve ``dense.embed_batch`` (V8-08.02) — prior ``{512 rows /
    250 ms queue age}``.

    Accepted forms: a scalar row bound; a flat ``[rows, max_age_ms]``
    pair (the params channel's list form); a mapping
    ``{"rows": …, "max_age_ms": …}``; ``True`` for the priors;
    ``False``/``"off"``/``"none"`` to disarm (unbounded merge). The
    split spellings ``dense.embed_batch.rows`` /
    ``dense.embed_batch.max_age_ms`` (the flattened nested-params form)
    override the individual components, then
    ``cfg.jobs.dense_embed_batch_rows`` /
    ``dense_embed_batch_max_age_ms`` fill undeclared components.
    """
    composite = job_arm(
        refs,
        ingester,
        "dense.embed_batch",
        "dense_embed_batch",
        "embed_batch",
    )
    rows: Optional[int]
    age: Optional[float]
    if composite is None or composite is True:
        rows, age = None, None  # undeclared → component defaults below
    elif composite is False or (
        isinstance(composite, str)
        and composite.strip().lower() in ("off", "none", "disabled")
    ):
        rows, age = 0, 0.0  # disarmed — the flag itself still resolves
    elif isinstance(composite, (list, tuple)):
        if len(composite) != 2:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "dense.embed_batch list form must be [rows, max_age_ms], "
                f"got {composite!r}",
            )
        rows = _arm_int(composite[0], "dense.embed_batch.rows", 0)
        age = _arm_num(composite[1], "dense.embed_batch.max_age_ms")
    elif isinstance(composite, dict):
        rows = (
            _arm_int(composite["rows"], "dense.embed_batch.rows", 0)
            if composite.get("rows") is not None
            else None
        )
        age = (
            _arm_num(
                composite["max_age_ms"], "dense.embed_batch.max_age_ms"
            )
            if composite.get("max_age_ms") is not None
            else None
        )
    elif isinstance(composite, str):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"dense.embed_batch must be a row bound, [rows, ms], "
            f"a mapping, or off — got {composite!r}",
        )
    else:
        rows = _arm_int(composite, "dense.embed_batch", 0)
        age = None
    # Split spellings on the per-job/policy carriers refine the
    # composite's components. ``cfg_attr`` must NOT ride these lookups:
    # the deployment default would outrank the job's own composite
    # declaration — carrier order inverted. The cfg fields fill only
    # components no finer carrier declared (below).
    raw_rows = job_arm(
        refs,
        ingester,
        "dense.embed_batch.rows",
        "dense_embed_batch_rows",
        "embed_batch_rows",
    )
    raw_age = job_arm(
        refs,
        ingester,
        "dense.embed_batch.max_age_ms",
        "dense_embed_batch_max_age_ms",
        "embed_batch_max_age_ms",
    )
    if raw_rows is not None:
        rows = _arm_int(raw_rows, "dense.embed_batch.rows", 0)
    if raw_age is not None:
        age = _arm_num(raw_age, "dense.embed_batch.max_age_ms")
    disarmed = composite is False or (
        isinstance(composite, str)
        and composite.strip().lower() in ("off", "none", "disabled")
    )
    if disarmed:
        # ``off`` means unbounded; split-key declarations still re-arm
        # individual components (a document may say
        # ``{"dense": {"embed_batch": "off",
        #              "embed_batch.rows": 256}}``). The cfg carrier is a
        # default — it never resurrects a bound a job explicitly killed.
        return EmbedBatchBound(
            0 if rows is None else rows, 0.0 if age is None else age)
    # Deployment carrier — fills only components no finer carrier named.
    jobs_cfg = getattr(getattr(ingester, "cfg", None), "jobs", None)
    if rows is None:
        cfg_rows = getattr(jobs_cfg, "dense_embed_batch_rows", None)
        if cfg_rows is not None:
            rows = _arm_int(
                cfg_rows, "cfg.jobs.dense_embed_batch_rows", 0)
    if age is None:
        cfg_age = getattr(jobs_cfg, "dense_embed_batch_max_age_ms", None)
        if cfg_age is not None:
            age = _arm_num(
                cfg_age, "cfg.jobs.dense_embed_batch_max_age_ms")
    return EmbedBatchBound(
        EMBED_BATCH_ROWS if rows is None else rows,
        EMBED_BATCH_MAX_AGE_MS if age is None else age,
    )


#: ``dense.tier`` (V8-08.05) — the profile→encoder-tier flag. Profile
#: spellings map onto encoder families: ``local_memory`` is the hashing
#: tier; ``local_memory_quality``'s S1 candidate is ``potion-base-8M``
#: (an ``artifact``-family encoder — fail-closed until a reviewed runtime
#: ships). ``auto``/``default`` declare no constraint.
DENSE_TIER_ALIASES = {
    "local_memory": "hashing",
    "local-memory": "hashing",
    "local_memory_quality": "potion-base-8m",
    "local-memory-quality": "potion-base-8m",
}
DENSE_TIER_UNCONSTRAINED = {"auto", "default", "inherit"}


def resolve_dense_tier(refs: Optional[dict], ingester: Any) -> Optional[str]:
    """``dense.tier`` (V8-08.05) — normalized declared tier or ``None``.

    ``None`` (absent, ``auto``, ``default``) means unconstrained. A
    declared tier must be a non-empty string; anything else is a
    VALIDATION failure — a mistyped tier must never read as "no pin".
    """
    raw = job_arm(refs, ingester, "dense.tier", "dense_tier")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"dense.tier must be a non-empty tier name, got {raw!r}",
        )
    norm = raw.strip().lower()
    if norm in DENSE_TIER_UNCONSTRAINED:
        return None
    return DENSE_TIER_ALIASES.get(norm, norm)


def dense_tier_satisfied(
    tier: str, encoder_id: Optional[str], backend: Optional[str]
) -> bool:
    """Whether the provisioned encoder satisfies a declared ``dense.tier``.

    A tier is provisioned when it names the configured backend
    (``hashing``, ``ollama``, …) or a segment of the encoder's pinned id
    (``hashing:subword-ngram:v1`` satisfies ``hashing``; an artifact id
    like ``potion-base-8m:r3`` satisfies ``potion-base-8m``). Anything
    else is unprovisioned — the honest outcome is a recorded deferral,
    not a silent downgrade to whatever encoder is configured.
    """
    norm = tier.strip().lower()
    backend_norm = (backend or "").strip().lower()
    if norm and norm == backend_norm:
        return True
    enc = (encoder_id or "").strip().lower()
    if not enc:
        return False
    return norm == enc or norm in enc.split(":")


def handle_dense_compact(job: dict[str, Any], owner: str, ingester: Any) -> None:
    """``dense_compact`` maintenance job — compact the requested spaces.

    ``input_refs``: ``scope_id``/``namespace`` (default the job's own
    scope), ``encoder_id``, ``generation`` (optional space pins), and the
    arm overrides ``b_max``/``min_mean_rows``. The §23 ``dense.B_max``
    arm resolves through :func:`resolve_b_max` — ``input_refs`` (direct
    or an embedded ``params`` map), a retrieval-policy params map hung
    on the ingester, ``cfg.jobs.dense_compact_b_max`` — defaulting to
    the prior :data:`B_MAX_DEFAULT`; a declared ``0`` is honored, not
    dropped. A job naming no filters sweeps every space in its scope. A
    receipt event (``dense_compacted``) lands in the same store for
    observability; a store without the event clock simply skips it (the
    meta epoch is the durable record).
    """
    refs = job.get("input_refs") or {}
    store = ingester.store
    scope_id = (
        _opt_str(refs, "scope_id")
        or _opt_str(refs, "namespace")
        or job.get("scope_id")
    )
    encoder_id = _opt_str(refs, "encoder_id")
    generation = _opt_int(refs, "generation")
    b_max = resolve_b_max(refs, ingester)
    mmr = _opt_int(refs, "min_mean_rows")
    min_mean_rows = MIN_MEAN_ROWS if mmr is None else mmr

    with store.read() as conn:
        if not _present(conn, _matrix.BLOCK_TABLE):
            return  # nothing provisioned — honest no-op
        spaces = list_spaces(
            conn,
            scope_id=scope_id,
            encoder_id=encoder_id,
            generation=generation,
        )
    if not spaces:
        return
    reports = compact_spaces(
        store, iter(spaces), b_max=b_max, min_mean_rows=min_mean_rows
    )
    n_done = sum(1 for r in reports if r.get("compacted"))
    if not n_done or scope_id is None:
        return
    try:
        from ...storage.repos import EventsRepo

        repo = EventsRepo(store)
        with store.tx() as conn:
            repo.append(
                conn,
                scope_id,
                "dense_compacted",
                "engine",
                {
                    "job_id": job.get("job_id"),
                    "spaces": n_done,
                    # The resolved §23 arm — the trigger that fired is
                    # attributable, not implicit.
                    "dense.B_max": b_max,
                    "blocks_before": sum(
                        r.get("blocks_before", 0) for r in reports
                    ),
                    "blocks_after": sum(
                        r.get("blocks_after", 0) for r in reports
                    ),
                    "rows": sum(r.get("rows", 0) for r in reports),
                },
                "v8",
            )
    except (AttributeError, VerbatimError):
        # Test shims without next_event_us / events — the meta epoch is
        # the durable record either way; never fail the job on a receipt.
        pass


__all__ = [
    "B_MAX_DEFAULT",
    "MIN_MEAN_ROWS",
    "EMBED_BATCH_ROWS",
    "EMBED_BATCH_MAX_AGE_MS",
    "N_D_DEFAULT",
    "DENSE_COMPACT_KIND",
    "DENSE_TIER_ALIASES",
    "DENSE_TIER_UNCONSTRAINED",
    "EmbedBatchBound",
    "SpaceRef",
    "CompactionPlan",
    "dense_tier_satisfied",
    "embed_batch_bound",
    "job_arm",
    "resolve_b_max",
    "resolve_dense_tier",
    "epoch_meta_key",
    "compaction_epoch",
    "plan_space",
    "list_spaces",
    "dense_compaction_derivation",
    "compact_space",
    "compact_space_in_tx",
    "compact_spaces",
    "handle_dense_compact",
]
