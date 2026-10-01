"""V7 dense lane — cosine search over the contiguous block matrix.

SPEC_V7 §07 (V7-07.04/05/06/12/14, D7-17). The lane encodes the query with
the caller-injected ``query_encoder`` and scans ``unit_vectors_block`` via
:func:`verbatim.embeddings.matrix.scan` — a packed-matrix pass that never
decodes per-row vector BLOBs on the hot path.

Honesty contract (V7-04.03, LaneV7 protocol):

- No ``query_encoder`` → ``unavailable`` with ``reason="no_encoder"``.
  The lane NEVER fabricates scores: every emitted score is a real cosine
  over stored block rows.
- ``eligible`` (``ctx.eligible``) is applied to rowmap keys BEFORE scoring
  — set-membership directly on keys, callables through prefetched unit
  rows — and the emitted candidate keys are resolved against ``units``
  again at emission (``source_id``/``revision``). A lane never widens the
  caller's eligible set.
- Deadline: ``slice.deadline_ms`` budgets encode + scan. Expiry before
  work starts → ``deadline``; a cut scan → ``partial`` with
  ``reason="deadline"`` and honest ``examined``/``eligible`` counts.
- Unsupported/missing structures are named in ``stats`` and status —
  absent block table (``no_vector_index``), absent units index
  (``no_units_index``), ambiguous encoder space (``encoder_id_ambiguous``),
  an eligibility predicate that cannot be evaluated
  (``eligibility_unevaluable`` — fail closed, never silently wide-open).

Boundary (V7-07.09): the contextual header (``<date> | <speaker> |
<session>:`` prefix on DOCUMENT text) is applied at embedding-write time
by the encoder pipeline, not by this lane — the lane encodes the raw query
string, which is what the spec's header format targets (headers describe
the embedded unit). Query-side contextual rewriting lands with the S1
artifact work.

``ctx`` wiring (frozen ``LaneContextV7`` has no encoder slot — wave-A
contract): ``ctx.manifest["query_encoder"]`` carries the encoder callable
(or object with ``.encode([text]) -> [blob]``); ``ctx.manifest
["encoder_id"]`` optionally pins the vector space — otherwise the
encoder's own ``encoder_id`` attribute is used, else the single distinct
``encoder_id`` present in this scope+generation's blocks.

V8 additions (SPEC_V8 §08):

- V8-08.01: ``stats["compaction_epoch"]`` reports the scanned snapshot's
  ``meta.dense_compaction_epoch/<space>``; the rewriter lives in
  ``retrieval/v7/dense_compact.py``.
- V8-08.03: compacted blocks make ``matrix.scan``'s one-matmul-per-block
  path dominant again — ``stats["vectorized_blocks"]`` counts them
  (honest-zero on the pure-Python fallback) and ``stats["scan_ms"]``
  records the scan envelope (timing, determinism-exempt).
- V8-08.04 (emit side): the lane's top ``dense.N_d`` candidates — every
  emitted candidate already cleared the null-model floor — are marked
  ``signals["dense_slot"] = True`` for fusion's guaranteed-pool-slot
  enforcement (fusion owns the slot mechanics; the lane only marks).
  ``dense.N_d`` resolves off ``ctx.policy``/manifest, default 10.
- V8-20.03: ``stats["coverage_dense"]`` = ``{slots, floor,
  blocks_scanned, compaction_epoch}`` — the lane-computed block the
  pipeline grafts into ``coverage.dense`` (grafting is pipeline-side).
"""

from __future__ import annotations

import contextlib
import math
import sqlite3
import time
from typing import Any, Optional

from ...core.types import ErrorCode, JobKind, VerbatimError
from ...core.types_v7 import (
    CandidateV7,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    QueryViewV7,
)
from ...embeddings import matrix as _matrix
from ...storage.repos import has_table as _has_table
from . import dense_compact as _compact

LANE = LaneName.DENSE.value
_UNITS_TABLE = "units"  # §30 mirror; integration swap lands with schema_v7

#: Monotonic clock seam — module-level so deterministic tests (and a
#: replay-time pipeline, if it ever pins one) can substitute a scripted
#: clock. Defaults to ``time.monotonic``.
_monotonic = time.monotonic

#: Columns handed to a callable ``ctx.eligible`` predicate (the full unit
#: row — the callable receives a plain dict of all stored columns).
_LOOKUP_CHUNK = 400


def _read_conn(ctx: LaneContextV7) -> contextlib.AbstractContextManager:
    """The caller's pinned read snapshot.

    ``Store.read()`` is reentrant — a lane running inside the pipeline's
    already-open snapshot reuses it; a bare connection is wrapped without
    opening a new transaction (LaneV7 protocol: never open a new tx).
    """
    store = ctx.store
    read = getattr(store, "read", None)
    if callable(read):
        return read()
    if isinstance(store, sqlite3.Connection):
        return contextlib.nullcontext(store)
    conn = getattr(store, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return contextlib.nullcontext(conn)
    raise VerbatimError(
        ErrorCode.VALIDATION, "ctx.store exposes no readable connection"
    )


class _KeyEligibility:
    """Adapt ``ctx.eligible`` to a batch rowmap-key filter for ``scan``.

    Modes: ``None`` → all keys pass; container of unit keys → membership
    (zero SQL); callable → ``eligible(unit_row_dict)`` per key, resolved by
    chunked ``units`` lookups with per-key verdict caching. A key with no
    unit row fails closed. An object that is neither container nor
    callable is ``unevaluable`` — the lane reports ``unavailable`` rather
    than scanning without eligibility.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        eligible: Any,
        scope_id: str,
        generation: int,
    ) -> None:
        self._conn = conn
        self._eligible = eligible
        self._scope_id = scope_id
        self._generation = generation
        self._verdicts: dict[str, bool] = {}
        self.evaluated = 0
        self.unevaluable = False
        if eligible is None:
            self._mode = "all"
        elif isinstance(
            getattr(eligible, "unit_ids", None), (set, frozenset, dict)
        ):
            # A materialized eligible-set (``_Eligible.unit_ids``) resolves
            # every key in memory — the callable path would re-issue one
            # ``units`` lookup per vector block.
            self._mode = "set"
            self._set = eligible.unit_ids
        elif callable(eligible):
            self._mode = "callable"
        elif isinstance(eligible, (str, bytes)):
            self._mode = "invalid"
            self.unevaluable = True
        elif isinstance(eligible, (set, frozenset, dict)):
            self._mode = "set"
            self._set = eligible
        elif isinstance(eligible, (list, tuple)):
            self._mode = "set"
            self._set = frozenset(eligible)
        elif hasattr(eligible, "__contains__"):
            self._mode = "set"
            self._set = eligible
        else:
            self._mode = "invalid"
            self.unevaluable = True

    def filter_keys(self, keys: list[str]) -> set[str]:
        if self._mode == "all":
            return set(keys)
        if self._mode == "set":
            return {k for k in keys if k in self._set}
        if self._mode != "callable":
            return set()
        missing = [k for k in keys if k not in self._verdicts]
        rows = _fetch_unit_rows(
            self._conn, missing, self._scope_id, self._generation
        )
        for k in missing:
            row = rows.get(k)
            self._verdicts[k] = bool(row is not None and self._eligible(row))
        self.evaluated += len(missing)
        return {k for k in keys if self._verdicts.get(k)}


def _fetch_unit_rows(
    conn: sqlite3.Connection,
    keys: list[str],
    scope_id: str,
    generation: int,
) -> dict[str, dict[str, Any]]:
    """Chunked ``units`` lookup → ``unit_id -> full row dict``.

    Scoped to the caller's ``(scope_id, generation)`` — ``units`` is keyed
    ``(unit_id, generation)`` (rebuild generations coexist, V7-30.02), so
    the fence is ``generation <= pinned`` and the newest row per unit_id
    wins (ORDER BY generation DESC, first-seen kept).
    """
    out: dict[str, dict[str, Any]] = {}
    if not keys:
        return out
    for i in range(0, len(keys), _LOOKUP_CHUNK):
        part = keys[i : i + _LOOKUP_CHUNK]
        marks = ",".join("?" for _ in part)
        cur = conn.execute(
            f"SELECT * FROM {_UNITS_TABLE}"
            f" WHERE scope_id = ? AND generation <= ?"
            f" AND unit_id IN ({marks})"
            " ORDER BY unit_id, generation DESC",
            [scope_id, generation, *part],
        )
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            out.setdefault(d["unit_id"], d)
    return out


def _resolve_encoder(ctx: LaneContextV7) -> tuple[Any, Optional[str]]:
    """``(encoder_callable_or_obj, encoder_id|None)`` from ctx wiring.

    Resolution order for the vector-space pin: ``ctx.encoder_id`` →
    ``ctx.manifest["encoder_id"]`` → the encoder's own ``encoder_id``
    attribute → (in the lane) the single distinct ``encoder_id`` stored in
    this scope+generation's blocks.
    """
    manifest = getattr(ctx, "manifest", None) or {}
    enc = getattr(ctx, "query_encoder", None) or manifest.get("query_encoder")
    if enc is None:
        return None, None
    encoder_id = (
        getattr(ctx, "encoder_id", None)
        or manifest.get("encoder_id")
        or getattr(enc, "encoder_id", None)
    )
    if not isinstance(encoder_id, str) or not encoder_id:
        encoder_id = None
    return enc, encoder_id


def _space_has_blocks(
    conn: sqlite3.Connection, encoder_id: str, scope_id: str, generation: int
) -> bool:
    """One-row probe: does this (encoder, scope) space hold any block at
    or below the pinned generation?"""
    return (
        conn.execute(
            f"SELECT 1 FROM {_matrix.BLOCK_TABLE}"
            " WHERE encoder_id = ? AND scope_id = ? AND generation <= ?"
            " LIMIT 1",
            (encoder_id, scope_id, generation),
        ).fetchone()
        is not None
    )


def _coverage_lag(
    conn: sqlite3.Connection,
    ctx: LaneContextV7,
    encoder_id: Optional[str],
    snap_generation: Optional[int],
) -> tuple[int, int]:
    """``(n_pending, n_uncovered)`` — the dense lane's owed-work signals.

    ``n_pending`` counts non-terminal ``source_embed`` jobs in this
    scope (the job row's own ``scope_id``, or the resolved ``namespace``
    carried in ``input_refs_json`` — the two diverge when the control
    artifact binds a different partition than ``sources.scope_id``).
    ``n_uncovered`` counts *live* pinned units — the newest unit slice
    per ``(source_id, revision)`` at/below the pin, the same set the
    producer embeds — minus the rows the scan would actually see: the
    resolved space's blocks at ``snap_generation`` (none when no
    snapshot resolves). A committed-but-uncovered hole — an embed that
    ran before its units were projected, or a slice stranded below the
    newest block generation — surfaces here instead of reading as
    success. Both are honest lower bounds; zero means no owed work is
    visible.
    """
    pending = 0
    if _has_table(conn, "jobs"):
        pending = int(
            conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind = ?"
                " AND state IN ('queued','leased','retry_wait')"
                " AND (scope_id = ?"
                "      OR json_extract(input_refs_json, '$.namespace') = ?"
                "      OR json_extract(input_refs_json, '$.scope_id') = ?)",
                (
                    JobKind.SOURCE_EMBED.value,
                    ctx.scope_id,
                    ctx.scope_id,
                    ctx.scope_id,
                ),
            ).fetchone()[0]
        )
    # A unit is embeddable iff both byte pins resolve inside the payload
    # (``_unit_embed_text``'s rule) — ``byte_start <= byte_end`` and both
    # non-NULL is the SQL-side approximation; units without resolvable
    # pins keep metadata-only coverage and are never owed a vector.
    _PINNED = (
        "u.byte_start IS NOT NULL AND u.byte_end IS NOT NULL"
        " AND u.byte_start <= u.byte_end"
    )
    # Live unit set = newest slice per (source, revision) at/below the
    # pin — the same complete-snapshot rule the producer's loader
    # applies. Coverage is exact: decode the scanned snapshot's rowmaps
    # (no data blobs — cheaper than the scan that just ran) and count
    # live keys actually present, so stale-generation keys sitting in
    # newer blocks can't cancel a real hole.
    if encoder_id is not None and snap_generation is not None:
        live = {
            r[0]
            for r in conn.execute(
                f"SELECT u.unit_id FROM {_UNITS_TABLE} u"
                f" WHERE u.scope_id = ? AND {_PINNED}"
                "  AND u.generation = ("
                "      SELECT MAX(u2.generation) FROM units u2"
                "      WHERE u2.source_id = u.source_id"
                "        AND u2.revision = u.revision"
                "        AND u2.generation <= ?)",
                (ctx.scope_id, ctx.generation),
            ).fetchall()
        }
        covered = live & _matrix.block_row_keys(
            conn, encoder_id, ctx.scope_id, snap_generation
        )
        return pending, len(live) - len(covered)
    # No scannable snapshot — every live pinned unit is owed work.
    units = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {_UNITS_TABLE} u"
            f" WHERE u.scope_id = ? AND {_PINNED}"
            "  AND u.generation = ("
            "      SELECT MAX(u2.generation) FROM units u2"
            "      WHERE u2.source_id = u.source_id"
            "        AND u2.revision = u.revision"
            "        AND u2.generation <= ?)",
            (ctx.scope_id, ctx.generation),
        ).fetchone()[0]
    )
    return pending, units


def _apply_lag(
    out: LaneOutput,
    conn: sqlite3.Connection,
    ctx: LaneContextV7,
    encoder_id: Optional[str],
    snap_generation: Optional[int] = None,
) -> None:
    """Downgrade an otherwise-ok lane to PARTIAL while vector work is
    owed (V7-07.12: embeddings land asynchronously — a search before
    they land reports ``coverage.dense = partial(n_pending)`` without
    blocking, and never ``ok`` with a silent backlog). Stats carry the
    counts whether or not the status moved."""
    pending, uncovered = _coverage_lag(conn, ctx, encoder_id, snap_generation)
    if pending:
        out.stats["n_pending"] = pending
    if uncovered:
        out.stats["units_uncovered"] = uncovered
    if (pending or uncovered) and out.status == LaneStatus.OK:
        out.status = LaneStatus.PARTIAL
        out.reason = "embed_pending" if pending else "units_uncovered"


def _dense_N_d(ctx: LaneContextV7) -> int:
    """V8-08.04 ``dense.N_d`` arm (§23, default 10).

    Primary carrier: the §23 ``policy.params`` map (``dense.N_d`` key)
    via ``policy_param``; the lane manifest (``dense.N_d`` /
    ``dense_N_d``) overrides for harnesses that bypass ``load_policy``.
    ``0`` is a real arm value (slots disabled).
    """
    pol = getattr(ctx, "policy", None)
    try:
        from .policy import policy_param

        val = policy_param(pol, "dense.N_d", None)
        if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
            return val
    except Exception:  # noqa: BLE001 — resolver/module absent → priors
        pass
    for carrier in ("params", "knobs", "arms"):
        mapping = getattr(pol, carrier, None)
        if isinstance(mapping, dict):
            for key in ("dense.N_d", "dense_N_d", "N_d"):
                val = mapping.get(key)
                if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
                    return val
    manifest = getattr(ctx, "manifest", None) or {}
    for key in ("dense.N_d", "dense_N_d"):
        val = manifest.get(key)
        if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
            return val
    return _compact.N_D_DEFAULT


def _emit_coverage_dense(
    out: LaneOutput,
    *,
    slots: Optional[int],
    floor: Optional[float],
    blocks_scanned: Optional[int],
    epoch: Optional[int],
) -> None:
    """V8-20.03 ``coverage.dense`` — exactly what this lane computed.

    ``slots`` counts candidates emitted under slot reservation
    (V8-08.04 — the guaranteed-pool-slot *enforcement* is fusion's);
    ``floor`` is the null-model bound applied; ``blocks_scanned`` the
    compacted-matrix blocks actually scored; ``compaction_epoch`` the
    ``meta.dense_compaction_epoch/<space>`` of the scanned snapshot
    (``None`` until a space resolves — V8-08.01).
    """
    out.stats["coverage_dense"] = {
        "slots": slots,
        # Rounded like stats["noise_floor"] — one value, one precision,
        # no cross-field disagreement in coverage consumers.
        "floor": round(floor, 6) if floor is not None else None,
        "blocks_scanned": blocks_scanned,
        "compaction_epoch": epoch,
    }


def _encode_query(enc: Any, text: str) -> Any:
    """One encoded query vector — accepts Encoder objects or callables."""
    if hasattr(enc, "encode") and callable(enc.encode):
        out = enc.encode([text])
        if not isinstance(out, (list, tuple)) or len(out) != 1:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID,
                "query encoder must return exactly one vector",
            )
        return out[0]
    out = enc(text)
    if (
        isinstance(out, (list, tuple))
        and len(out) == 1
        and isinstance(out[0], (bytes, bytearray, memoryview))
    ):
        return out[0]
    return out


def lane_dense(
    ctx: LaneContextV7, qv: QueryViewV7, slice: LaneSlice
) -> LaneOutput:
    """Dense lane: contiguous-matrix cosine over eligible unit vectors."""
    out = LaneOutput(lane=LANE, status=LaneStatus.OK)
    t0 = _monotonic()

    deadline_ms = getattr(slice, "deadline_ms", None)
    if deadline_ms is not None and (
        not isinstance(deadline_ms, (int, float)) or math.isinf(deadline_ms)
    ):
        deadline_ms = None

    def remaining_ms() -> Optional[float]:
        if deadline_ms is None:
            return None
        return deadline_ms - (_monotonic() - t0) * 1000.0

    cap = int(getattr(slice, "cap", 0) or 0)
    if cap < 1:
        out.stats["note"] = "zero_cap"
        return out

    enc, encoder_id = _resolve_encoder(ctx)
    if enc is None:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "no_encoder"
        return out

    rem = remaining_ms()
    if rem is not None and rem <= 0:
        out.status = LaneStatus.DEADLINE
        out.reason = "deadline"
        out.stats["encoded"] = False
        return out

    try:
        raw = _encode_query(enc, qv.query)
        query_vec = _matrix._as_float_vec(raw)
    except Exception as exc:
        out.status = LaneStatus.UNAVAILABLE
        out.reason = "encoder_failed"
        out.stats["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out.stats["encoded"] = True
    # Null-model floor derives from the space dimension only — computable
    # before the store probe so every exit path can report it.
    dim = len(query_vec) or 1
    floor = 3.0 / math.sqrt(dim)

    with _read_conn(ctx) as conn:
        if not _has_table(conn, _matrix.BLOCK_TABLE):
            out.status = LaneStatus.UNAVAILABLE
            out.reason = "no_vector_index"
            _emit_coverage_dense(
                out, slots=0, floor=floor, blocks_scanned=0, epoch=None
            )
            return out
        if not _has_table(conn, _UNITS_TABLE):
            out.status = LaneStatus.UNAVAILABLE
            out.reason = "no_units_index"
            _emit_coverage_dense(
                out, slots=0, floor=floor, blocks_scanned=0, epoch=None
            )
            return out

        if encoder_id is None:
            # V7-30.02: blocks persist per generation — an encoder whose
            # newest snapshot sits below the pin is still live, so the
            # space inventory fences ``generation <= pinned`` (encoders
            # only indexed ABOVE the pin are correctly invisible).
            ids = [
                r[0]
                for r in conn.execute(
                    f"SELECT DISTINCT encoder_id FROM {_matrix.BLOCK_TABLE}"
                    " WHERE scope_id = ? AND generation <= ?",
                    (ctx.scope_id, ctx.generation),
                ).fetchall()
            ]
            if len(ids) == 0:
                # Nothing indexed for this scope at/below the pin — an
                # honestly empty lane, not a failure (V7-07.12:
                # embeddings land asynchronously behind lexical
                # visibility). Owed work downgrades the empty lane to
                # partial — never ``ok`` with a silent backlog.
                out.stats["note"] = "no_blocks"
                out.stats["encoder_id"] = None
                _emit_coverage_dense(
                    out, slots=0, floor=floor, blocks_scanned=0, epoch=None
                )
                _apply_lag(out, conn, ctx, None)
                return out
            if len(ids) > 1:
                out.status = LaneStatus.UNAVAILABLE
                out.reason = "encoder_id_ambiguous"
                out.stats["encoder_ids"] = ids
                _emit_coverage_dense(
                    out, slots=0, floor=floor, blocks_scanned=0, epoch=None
                )
                return out
            encoder_id = ids[0]
        else:
            # The pin names the model; the producer's block space binds
            # the contextual-header format version inside the id
            # (V7-07.09 ``:hdr:v1`` — write and read sides share
            # ``matrix.unit_vector_encoder_id``). A bare-id space written
            # before the header convention still resolves: prefer the
            # headered space, fall back to the pinned id itself.
            hdr_id = _matrix.unit_vector_encoder_id(encoder_id)
            if hdr_id != encoder_id and _space_has_blocks(
                conn, hdr_id, ctx.scope_id, ctx.generation
            ):
                encoder_id = hdr_id
        out.stats["encoder_id"] = encoder_id

        # V7-30.02: the block space accumulates per-source append batches
        # stamped at each commit's resolved generation — a rebuild that
        # re-embeds every source makes the newest generation a complete
        # snapshot again. Scan the newest generation present at/below the
        # pin and never mix blocks across generations (an older snapshot
        # stays valid after a bump that rewrote nothing here); live units
        # stranded below it surface as ``units_uncovered`` via
        # ``_apply_lag`` — never silently missing.
        snap = conn.execute(
            f"SELECT MAX(generation) FROM {_matrix.BLOCK_TABLE}"
            " WHERE encoder_id = ? AND scope_id = ? AND generation <= ?",
            (encoder_id, ctx.scope_id, ctx.generation),
        ).fetchone()
        if snap is None or snap[0] is None:
            # The pinned encoder has no index at/below this generation —
            # honestly empty, same contract as the probe path above.
            out.stats["note"] = "no_blocks"
            out.stats["eligibility_evaluated"] = 0
            _emit_coverage_dense(
                out, slots=0, floor=floor, blocks_scanned=0, epoch=None
            )
            _apply_lag(out, conn, ctx, encoder_id)
            return out
        snap_generation = int(snap[0])
        out.stats["vector_generation"] = snap_generation
        # V8-08.01 — the scanned snapshot's recorded compaction epoch
        # (meta.dense_compaction_epoch/<space>; 0 = never compacted).
        epoch = _compact.compaction_epoch(
            conn, encoder_id, ctx.scope_id, snap_generation
        )
        out.stats["compaction_epoch"] = epoch

        eligibility = _KeyEligibility(
            conn, ctx.eligible, ctx.scope_id, ctx.generation
        )
        if eligibility.unevaluable:
            out.status = LaneStatus.UNAVAILABLE
            out.reason = "eligibility_unevaluable"
            _emit_coverage_dense(
                out, slots=0, floor=floor, blocks_scanned=0, epoch=epoch
            )
            return out

        rem = remaining_ms()
        t_scan = _monotonic()
        res = _matrix.scan(
            conn,
            encoder_id,
            ctx.scope_id,
            snap_generation,
            query_vec,
            eligible_keys=eligibility,
            k=cap,
            deadline_ms=rem if rem is not None else None,
            clock=_monotonic,
        )
        out.stats["scan_ms"] = round((_monotonic() - t_scan) * 1000.0, 3)
        out.examined = res.examined
        out.eligible = res.eligible
        out.stats.update(res.stats)
        out.stats["eligibility_evaluated"] = eligibility.evaluated
        # V8-08.03/K45 — every scanned block ran one vectorized matmul
        # when numpy engaged; the count is honest-zero on the fallback.
        out.stats["vectorized_blocks"] = (
            res.stats.get("blocks_scanned", 0) if res.stats.get("numpy") else 0
        )
        blocks_scanned = res.stats.get("blocks_scanned", 0)

        if not res:
            if res.partial:
                out.status = LaneStatus.PARTIAL
                out.reason = "deadline"
            _emit_coverage_dense(
                out, slots=0, floor=floor,
                blocks_scanned=blocks_scanned, epoch=epoch,
            )
            _apply_lag(out, conn, ctx, encoder_id, snap_generation)
            return out

        # Null-model noise floor: the cosine of two unrelated unit
        # vectors concentrates at 0 with std 1/sqrt(dim); scores within
        # 3σ of the null carry no semantic evidence, and emitting them
        # would let an unanswerable query "retrieve" its nearest noise
        # (the verdict's calibrated trigger (c) stays disabled until a
        # support_calibration/v2 entry is fitted — V7-11.06 — so the
        # lane must not fabricate candidates it cannot defend).  The
        # bound derives from the space dimension, never from a fitted
        # corpus constant; a measured floor may supersede it once a
        # calibration artifact lands under O9.  (``floor`` itself is
        # computed once, right after the query vector is built.)
        pairs = [(key, score) for key, score in res if score > floor]
        out.stats["noise_floor"] = round(floor, 6)
        out.stats["below_floor"] = len(res) - len(pairs)
        if not pairs:
            out.stats["note"] = "all_below_noise_floor"
            if res.partial:
                out.status = LaneStatus.PARTIAL
                out.reason = "deadline"
            _emit_coverage_dense(
                out, slots=0, floor=floor,
                blocks_scanned=blocks_scanned, epoch=epoch,
            )
            _apply_lag(out, conn, ctx, encoder_id, snap_generation)
            return out

        rows = _fetch_unit_rows(
            conn, [key for key, _s in pairs], ctx.scope_id, ctx.generation
        )
        orphaned = 0
        rank = 0
        for key, score in pairs:
            row = rows.get(key)
            if row is None:
                orphaned += 1
                continue
            rank += 1
            out.candidates.append(
                CandidateV7(
                    unit_id=key,
                    source_id=str(row.get("source_id") or ""),
                    revision=int(row.get("revision") or 0),
                    lane=LANE,
                    rank=rank,
                    raw_score=score,
                    signals={
                        "cosine": score,
                        "score_kind": res.kinds.get(key, "unknown"),
                    },
                )
            )
        if orphaned:
            out.stats["orphaned_rowmap_keys"] = orphaned
            if orphaned == len(res):
                # rowmap disagrees with units entirely — index corruption,
                # surfaced rather than silently empty.
                out.stats["note"] = "all_rowmap_keys_orphaned"

        # V8-08.04 emit side: the lane's top ``dense.N_d`` candidates
        # (all already above the null-model floor — ``pairs`` is
        # floor-filtered) are marked ``signals["dense_slot"]`` so fusion
        # can hold their guaranteed pool slots. The lane emits normally;
        # slot *enforcement* is fusion's — the mark only declares which
        # candidates were produced under reservation.
        n_d = _dense_N_d(ctx)
        marked = 0
        for cand in out.candidates[:n_d]:
            cand.signals["dense_slot"] = True
            marked += 1
        out.stats["dense_slots"] = marked
        out.stats["dense_N_d"] = n_d

        if res.partial:
            out.status = LaneStatus.PARTIAL
            out.reason = "deadline"
        _emit_coverage_dense(
            out, slots=marked, floor=floor,
            blocks_scanned=blocks_scanned, epoch=epoch,
        )
        _apply_lag(out, conn, ctx, encoder_id, snap_generation)
    return out


class DenseLane:
    """``LaneV7``-shaped adapter over :func:`lane_dense` for registries."""

    name = LANE

    def run(
        self, ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice
    ) -> LaneOutput:
        return lane_dense(ctx, query, slice)


# Lane modules self-register at import (lanes_base contract); the pipeline
# never imports this module itself — the registrar/importer seam is owned
# by the main-session integration.
from .lanes_base import register_lane  # noqa: E402

register_lane(LaneName.DENSE, lane_dense)


__all__ = ["LANE", "DenseLane", "lane_dense"]
