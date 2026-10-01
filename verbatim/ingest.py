"""Ingest pipeline: envelope → privacy gate → durable source tx → jobs.

Flow (SPEC §6): trusted envelope → privacy gate → durable source transaction
→ harvest job → candidate claims → admission/decision jobs. Durable guarantee
begins only when the source transaction commits — a queued job is not a
checkpoint (SPEC §33).

v2 drain (SPEC_V2 §39): ``run_pending`` drives every registered ``JobKind`` —
harvest/admit/compare/embed enrichment in the ordinary lane and the reserved
control lane (purge/reindex/review_apply) strictly first, so privacy and
correctness work can never be starved by a saturated enrichment queue. Every
handler commits its domain effects, its operation receipt, and the job
completion inside ONE generation-fenced transaction: a worker holding a
superseded lease generation cannot commit (``JobQueue.assert_lease``), and a
redelivered job replays its recorded receipt instead of reapplying the
effect (the ``operations`` ledger, V2-39.10).

Handlers for kinds whose domain services land elsewhere (episode_index,
procedure_validate, replay) still do deterministic work — they validate the
input and durably record the request as an event — so the drain never wedges
on them and nothing is silently dropped.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from collections.abc import Set as AbstractSet
from typing import Any, Callable, Iterable, Optional

from .config import VerbatimConfig
from .core.claims import propose
from .core.harvest import HARVESTER_VERSION, harvest_source, persist_harvest
from .core.identity import scope_key
from .core.lifecycle import (
    PURGE_ACTOR,
    LifecycleMachine,
    read_claim_head,
)
from .core.policy import (
    PolicyContext,
    _admit_apply,
    _admit_evaluate,
    relate,
)
from .core.time import now_us
from .core.types import (
    ErrorCode,
    IngestReceipt,
    JobKind,
    JobState,
    Lifecycle,
    ReviewState,
    Scope,
    SourceEnvelope,
    SourceKind,
    SpanRef,
    TimeInterval,
    TransitionCommand,
    VerbatimError,
    json_dumps,
    new_id,
    safe_json_loads,
)
from .core.types_v4 import CapabilityName, ReadinessState
from .embeddings.encoder import encode_spans
from .jobs.queue import DEFAULT_LEASE_S, LANES, JobQueue
from .purge import execute_purge
from .readiness import (
    ALL_CAPS,
    ReadinessEngine,
    ingest_receipt_id,
    job_stage_capability,
)
from .security.admission import default_review_state
from .security.labels import attach_label
from .security.quarantine import is_quarantined, open_quarantine
from .security.screening import SecurityVerdict, screen_content
from .storage.repos import (
    ClaimsRepo,
    EventsRepo,
    FtsRepo,
    ReviewsRepo,
    SourcesRepo,
    SpansRepo,
)
from .storage.repos_v2 import (
    EmbeddingInputsRepo,
    OperationsRepo,
    ProjectionRepo,
)
from .storage import commit_notify
from .storage.store import Store, writers_waiting

# Source kinds admitted for automatic harvesting by default (SPEC §10).
_DEFAULT_ALLOWED = {SourceKind.USER_MESSAGE, SourceKind.OPERATOR_RECORD, SourceKind.IMPORT}

# Deterministic record-only handlers: the domain services for these kinds
# land in other components; the durable job records the request so the drain
# completes and the intent is auditable (V2-39).
_RECORD_EVENTS = {
    JobKind.EPISODE_INDEX: "episode_index_requested",
    JobKind.PROCEDURE_VALIDATE: "procedure_validate_requested",
    JobKind.REPLAY: "replay_requested",
}

# One EMBED job covers at most this many spans per drain pass — matches the
# encoder module's batch bound so a backlog drains in bounded slices.
_EMBED_BATCH = 256

# Dequeue order of the non-throughput lanes — the privacy/correctness scan
# an unblock-first pass must still lose to (V6-02.08, v6_contracts §3).
# Mirrors the rank order of ``queue._LANE_ORDER_SQL``.
_NON_ORDINARY_LANES = ("control", "privacy_control", "maintenance", "background")

# The only kinds an unblock-first pass may pull ahead of ordinary work:
# the jobs that settle ``source_lexical_ready``/``source_vector_ready`` —
# the capabilities session barriers wait on (V6-02.08).
_PRIORITY_KINDS = frozenset(
    {JobKind.SOURCE_PROJECT.value, JobKind.SOURCE_EMBED.value}
)

#: Interleave window opened between drained jobs while a foreground
#: writer is stalled on ``BEGIN`` (marked through
#: ``storage.store.writers_waiting``). ~1 ms BEGIN retries then land on
#: an unlocked WAL instead of starving behind this drain's back-to-back
#: write transactions. Bounded per job; ordering, fencing, and the
#: 250 ms busy cap are untouched.
_WRITER_YIELD_S = 0.005

# V3 pipeline dispatch (SPEC_V3 §40): each kind maps to a ``handle_*``
# function in its owning module, lazily imported so feature modules never
# pay import cost at engine load and never edit this file. A kind whose
# module is absent fails CAPABILITY_UNAVAILABLE — loud, not a silent no-op.
_V3_KIND_HANDLERS = {
    JobKind.SCREEN: ("verbatim.security.handlers", "handle_screen"),
    JobKind.SPARSE_INDEX: ("verbatim.retrieval.v3.handlers", "handle_sparse_index"),
    JobKind.LATE_INDEX: ("verbatim.retrieval.v3.handlers", "handle_late_index"),
    JobKind.SIGNATURE_INDEX: ("verbatim.procedures.handlers", "handle_signature_index"),
    JobKind.EPISODE_BUILD: ("verbatim.experience.handlers_v3", "handle_episode_build"),
    JobKind.TRANSITION_BUILD: ("verbatim.experience.handlers_v3", "handle_transition_build"),
    JobKind.PROCEDURE_COMPILE: ("verbatim.procedures.handlers", "handle_procedure_compile"),
    JobKind.PROCEDURE_REFINE: ("verbatim.procedures.handlers", "handle_procedure_refine"),
    JobKind.CONSOLIDATE: ("verbatim.observations.handlers", "handle_consolidate"),
    JobKind.PURGE_DERIVED: ("verbatim.privacy.handlers", "handle_purge_derived"),
    JobKind.PURGE_VAULT: ("verbatim.privacy.handlers", "handle_purge_vault"),
    JobKind.QUARANTINE_REVIEW: ("verbatim.security.handlers", "handle_quarantine_review"),
    JobKind.REVOCATION_NOTIFY: ("verbatim.governance.handlers", "handle_revocation_notify"),
    JobKind.VAULT_ROTATE: ("verbatim.privacy.handlers", "handle_vault_rotate"),
    JobKind.PROJECTION_SYNC: ("verbatim.projection.handlers", "handle_projection_sync"),
    JobKind.CONNECTOR_PULL: ("verbatim.connectors.handlers", "handle_connector_pull"),
}


def _v5_kind_handlers() -> dict:
    """V5 source-pipeline dispatch map (lazy; absent module fails loud)."""
    try:
        from .jobs.v5_handlers import V5_KIND_HANDLERS
    except ImportError:
        return {}
    return V5_KIND_HANDLERS


def envelope_for(store: Store, source_id: str, revision: int) -> Optional[SourceEnvelope]:
    """Rebuild a SourceEnvelope from persisted rows (scope tuple included)."""
    from .core.types import Provenance, SourceKind, Visibility
    from .core.types import Scope as _Scope

    sources = SourcesRepo(store)
    src = sources.get(source_id)
    rev = sources.get_revision(source_id, revision)
    payload = sources.payload(source_id, revision)
    if src is None or rev is None or payload is None:
        return None
    with store.read() as conn:
        srow = conn.execute(
            "SELECT profile_id, principal_id, workspace_id, conversation_id, visibility"
            " FROM scopes WHERE scope_id = ?",
            (src["scope_id"],),
        ).fetchone()
    if srow is None:
        return None
    scope = _Scope(
        profile_id=srow[0],
        principal_id=srow[1],
        workspace_id=srow[2],
        conversation_id=srow[3],
        visibility=Visibility(srow[4]),
    )
    return SourceEnvelope(
        origin=src["origin"],
        source_kind=SourceKind(src["source_kind"]),
        scope=scope,
        speaker_id=src["speaker_id"],
        payload=payload,
        event_us=rev["event_us"],
        captured_us=rev["captured_us"],
        timezone=rev["timezone"],
        provenance=Provenance(rev["provenance"]),
        external_id=src["external_id"],
        source_id=source_id,
        revision=revision,
    )


def _source_scope_id(sources: Any, source_id: str, envelope: Any) -> str:
    """Authoritative partition for derived work on a persisted source.

    The stored ``sources.scope_id`` wins: v3 scope ids are opaque
    partition tokens that need not equal ``scope_key(scope)``, and
    derived objects must stay inside the source's real partition (and
    its purge closure). The digest fallback is identical for ordinary
    v2 writes where the stored row is missing.
    """
    row = sources.get(source_id)
    if row is not None and row.get("scope_id"):
        return row["scope_id"]
    return scope_key(envelope.scope)


def _v3_envelope_kind(store: Store, source_id: str, revision: int) -> Optional[str]:
    """The persisted v3 ``envelope_kind`` for a source revision, or None for
    v2-origin sources (no ``source_envelopes`` row — or no table at all on
    a v2-schema store)."""
    with store.read() as conn:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table'"
            " AND name='source_envelopes'",
        ).fetchone() is None:
            return None
        row = conn.execute(
            "SELECT envelope_kind FROM source_envelopes"
            " WHERE source_id = ? AND revision = ?",
            (source_id, revision),
        ).fetchone()
    return row[0] if row is not None else None


def _harvest_v3_result(
    store: Store, source_id: str, revision: int, envelope: SourceEnvelope
) -> Any:
    """Route the harvest: v3 structured kinds go through harvest_v3's
    structure-aware segmenter (V3-15.13); everything else keeps the v2
    prose path so its admission allowlist and candidate semantics stay
    byte-identical.

    harvest_v3 candidates carry ``kind_hint`` (``command``/``path``/
    ``error``/``test``/``hunk``) into ``Candidate.kind`` — the same free
    string slot v2 uses for ``sentence``/``paragraph`` — and absolute byte
    bounds into ``start_byte``/``end_byte``.
    """
    kind = _v3_envelope_kind(store, source_id, revision)
    if kind is None:
        return harvest_source(envelope)
    from .evidence.envelopes import _STRUCTURAL_HARVEST
    from .core.types_v3 import EnvelopeKind
    from .harvest_v3 import harvest_v3

    try:
        ekind = EnvelopeKind(kind)
    except ValueError:
        ekind = None
    if ekind is None or ekind not in _STRUCTURAL_HARVEST:
        return harvest_source(envelope)
    # V4-13.11: the candidate offsets harvest_v3 emits index into the exact
    # accepted bytes — decoding with ``errors="replace"`` would mint ranges
    # that do not exist in the stored payload. Malformed bytes raise
    # ``UnicodeDecodeError``; the caller maps it to a typed VALIDATION
    # failure (a decode failure never implies clean content, V4-13.12).
    text = (
        bytes(envelope.payload).decode("utf-8")
        if isinstance(envelope.payload, (bytes, bytearray))
        else str(envelope.payload)
    )
    cands = harvest_v3(text, envelope_kind=ekind)
    from .core.harvest import Candidate, HarvestResult

    return HarvestResult(
        tuple(
            Candidate(
                start_byte=int(c["start_byte"]),
                end_byte=int(c["end_byte"]),
                kind=str(c.get("kind_hint") or "statement"),
                context_needed=False,
                sensitive_hint=False,
                negated=False,
                has_condition=False,
                modality_hint=None,
                reason=str(c.get("kind_hint") or "structural"),
            )
            for c in cands
        ),
        0,
    )


def _require_ref(refs: dict[str, Any], key: str) -> str:
    """Typed input validation: a missing/empty ref is a job bug, not a crash."""
    value = refs.get(key)
    if not isinstance(value, str) or not value:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"job input_refs.{key} must be a non-empty string",
        )
    return value


def _require_int_ref(refs: dict[str, Any], key: str) -> int:
    value = refs.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerbatimError(
            ErrorCode.VALIDATION, f"job input_refs.{key} must be an integer"
        )
    return value


def _interval_from(obj: Any) -> Optional[TimeInterval]:
    """Rebuild a TimeInterval from a review/proposal payload dict."""
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise VerbatimError(ErrorCode.VALIDATION, "interval payload must be an object")
    try:
        return TimeInterval(
            from_us=obj.get("from_us"),
            until_us=obj.get("until_us"),
            precision=obj.get("precision", "unknown"),
            timezone=obj.get("timezone"),
            basis=obj.get("basis", "unknown"),
            start_kind=obj.get("start_kind", "exact"),
            end_kind=obj.get("end_kind", "exact"),
            from_us_hi=obj.get("from_us_hi"),
            until_us_hi=obj.get("until_us_hi"),
        )
    except (VerbatimError, ValueError) as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid interval payload: {exc}"
        ) from exc


class _Preencoded:
    """Encoder facade whose ``encode`` replays blobs already computed
    outside the write transaction.

    ``encode_spans`` calls ``encoder.encode`` first inside its own body —
    passing the real encoder would run model inference inside the commit
    transaction. This facade returns the vectors the worker already
    produced before opening the tx, so inference never holds a write tx
    (V2-39.07) while ``encode_spans`` still performs validation, manifest
    registration, and the keyed upsert itself.
    """

    def __init__(self, encoder: Any, blobs: list[bytes]) -> None:
        self._encoder = encoder
        self._blobs = list(blobs)

    @property
    def encoder_id(self) -> str:
        return self._encoder.encoder_id

    @property
    def dimensions(self) -> int:
        return self._encoder.dimensions

    @property
    def normalization(self) -> Optional[str]:
        return self._encoder.normalization

    def encode(self, texts: list[str]) -> list[bytes]:
        return list(self._blobs)

    def available(self) -> bool:
        return True

    def manifest(self) -> dict[str, Any]:
        return self._encoder.manifest()


class _LeaseFencedStore:
    """Store facade binding self-transacting helpers to a worker's lease.

    ``read()`` delegates to a real snapshot read; ``tx()`` opens a real
    write transaction and re-asserts ``(job_id, owner, generation)`` INSIDE
    it before yielding the connection — so helpers that manage their own
    ``store.tx()`` (``relate``, the v3 proposal passes) still commit their
    effects under the worker's live fence (V4-42.01/42.03). A cancelled or
    superseded lease makes ``assert_lease`` raise inside the opened tx,
    rolling the staged writes back with it.

    Inference still runs outside any transaction: ``read()`` and ``tx()``
    are separate contexts, so work a helper performs between them never
    holds a write tx (V4-09.09).
    """

    __slots__ = ("_store", "_jobs", "_job_id", "_owner", "_generation")

    def __init__(
        self,
        store: Store,
        jobs: JobQueue,
        job_id: str,
        owner: str,
        generation: int,
    ) -> None:
        self._store = store
        self._jobs = jobs
        self._job_id = job_id
        self._owner = owner
        self._generation = generation

    def read(self) -> Any:
        return self._store.read()

    def tx(self, **kwargs: Any) -> Any:
        @contextlib.contextmanager
        def _fenced() -> Any:
            with self._store.tx(**kwargs) as conn:
                self._jobs.assert_lease(
                    conn, self._job_id, self._owner, self._generation
                )
                yield conn

        return _fenced()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


class Ingester:
    """Coordinates source persistence and the durable job pipeline."""

    def __init__(
        self,
        store: Store,
        cfg: VerbatimConfig,
        judge: Any = None,
        encoder: Any = None,
        transport_broker: Any = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.judge = judge
        self.encoder = encoder
        self.transport_broker = transport_broker
        self.sources = SourcesRepo(store)
        self.spans = SpansRepo(store)
        self.jobs = JobQueue(store, max_pending=cfg.jobs.max_pending)
        self.policy = PolicyContext(cfg, judge)
        # Schema-v2 auxiliaries are optional — a pre-migration store (and the
        # v1 test shim) lacks these tables; handlers degrade explicitly per
        # feature rather than touching missing tables.
        with self.store.read() as conn:
            self._tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                )
            }
        self.ops = (
            OperationsRepo(store) if "operations" in self._tables else None
        )
        self.embedding_inputs = (
            EmbeddingInputsRepo(store)
            if "embedding_inputs" in self._tables
            else None
        )
        self.projections = (
            ProjectionRepo(store) if "projection_builds" in self._tables else None
        )
        # Durable readiness DAG (SPEC_V4 §14): absent on pre-v4 schemas —
        # obligation recording and fulfillment are skipped wholesale, never
        # half-written.
        self._readiness = (
            ReadinessEngine(store)
            if "readiness_obligations" in self._tables
            else None
        )
        # The most recent ``drain_report`` — the provider's session-end
        # seam drains through ``run_pending`` (the patchable seam) and
        # reads this side-channel for the honest breakdown (V4-14.05).
        self._last_drain_report: Optional[dict[str, Any]] = None

    def _has(self, table: str) -> bool:
        return table in self._tables

    def readiness_engine(self) -> ReadinessEngine:
        """The durable readiness service; ``CAPABILITY_UNAVAILABLE`` on a
        pre-v4 store rather than a silent no-op."""
        if self._readiness is None:
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "readiness_obligations table absent — store schema predates v4",
            )
        return self._readiness

    def _record_capture_obligations(
        self, conn: Any, source_id: str, revision: int, scope_id: str
    ) -> None:
        """Persist the receipt's obligation DAG inside the capture tx.

        ``accepted`` fulfills immediately; ``screened``/``lexical_ready``
        chain forward, and ``semantic_ready``/``derived_ready`` hang off
        ``lexical_ready`` as siblings — so ``wait_ready`` reads durable
        rows, never event counters (V4-14.01/02/03)."""
        if self._readiness is None:
            return
        self._readiness.record_obligations(
            conn,
            ingest_receipt_id(source_id, int(revision)),
            scope_id,
            ALL_CAPS,
            pipeline=True,
        )

    # ------------------------------------------------------------------
    # readiness settle/fail (V4-14.02/03) — always inside the caller's tx
    # ------------------------------------------------------------------

    def _job_source_pairs(
        self, conn: Any, job: dict[str, Any]
    ) -> list[tuple[str, int]]:
        """The ``(source_id, revision)`` pairs a job's refs bind to.

        Pipeline jobs key on ``source_id``/``revision`` directly; ``embed``
        jobs carry ``span_ids`` resolved through the spans table; v3
        ``screen`` jobs may carry an ``object_kind="source"`` reference.
        """
        refs = job.get("input_refs") or {}
        if not isinstance(refs, dict):
            return []
        pairs: list[tuple[str, int]] = []
        sid = refs.get("source_id")
        rev = refs.get("revision")
        if isinstance(sid, str) and sid:
            pairs.append((sid, int(rev) if isinstance(rev, int) else 1))
        if (
            refs.get("object_kind") == "source"
            and isinstance(refs.get("object_id"), str)
        ):
            pairs.append(
                (
                    refs["object_id"],
                    int(rev) if isinstance(rev, int) else 1,
                )
            )
        span_ids = refs.get("span_ids")
        if isinstance(span_ids, (list, tuple)) and span_ids:
            pairs.extend(self._sources_for_spans(conn, span_ids))
        # Dedup while preserving order.
        seen: set[tuple[str, int]] = set()
        out = []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _sources_for_spans(
        self, conn: Any, span_ids: Iterable[str]
    ) -> list[tuple[str, int]]:
        ids = [s for s in dict.fromkeys(span_ids) if isinstance(s, str) and s]
        if not ids:
            return []
        ph = ",".join("?" for _ in ids)
        rows = conn.execute(
            "SELECT DISTINCT source_id, revision FROM spans"
            f" WHERE span_id IN ({ph})",
            ids,
        ).fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    def _sources_for_claim(
        self, conn: Any, claim_id: str
    ) -> list[tuple[str, int]]:
        """Source revisions this claim cites as evidence — used to re-open
        deferred obligations when review activates a claim."""
        rows = conn.execute(
            "SELECT DISTINCT sp.source_id, sp.revision"
            " FROM claim_evidence ce"
            " JOIN spans sp ON sp.span_id = ce.span_id"
            " WHERE ce.claim_id = ?",
            (claim_id,),
        ).fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    def _outstanding_pipeline_jobs(
        self,
        conn: Any,
        source_id: str,
        revision: int,
        *,
        exclude_job_id: Optional[str] = None,
    ) -> int:
        """Live harvest/admit jobs owed for this source revision — the
        receipt's own outstanding work, never a global counter."""
        sql = (
            "SELECT COUNT(*) FROM jobs"
            " WHERE kind IN ('harvest','admit')"
            "   AND state IN ('queued','retry_wait','leased')"
            "   AND json_extract(input_refs_json, '$.source_id') = ?"
            "   AND json_extract(input_refs_json, '$.revision') = ?"
        )
        params: list[Any] = [source_id, int(revision)]
        if exclude_job_id is not None:
            sql += " AND job_id != ?"
            params.append(exclude_job_id)
        row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    def _outstanding_embed_jobs(
        self,
        conn: Any,
        source_id: str,
        revision: int,
        *,
        exclude_job_id: Optional[str] = None,
    ) -> int:
        """Live embed jobs covering any of this revision's spans."""
        sql = (
            "SELECT COUNT(*) FROM jobs j"
            " WHERE j.kind = 'embed'"
            "   AND j.state IN ('queued','retry_wait','leased')"
            "   AND EXISTS (SELECT 1"
            "       FROM json_each(j.input_refs_json, '$.span_ids') je"
            "       JOIN spans sp ON sp.span_id = je.value"
            "       WHERE sp.source_id = ? AND sp.revision = ?)"
        )
        params: list[Any] = [source_id, int(revision)]
        if exclude_job_id is not None:
            sql += " AND j.job_id != ?"
            params.append(exclude_job_id)
        row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    def _source_active_claim(
        self, conn: Any, source_id: str, revision: int
    ) -> bool:
        """An ACTIVE claim cites this source revision — the same join the
        embed gather uses, so readiness tracks exactly what indexed."""
        row = conn.execute(
            "SELECT 1 FROM claims c"
            " JOIN claim_revisions cr ON cr.claim_id = c.claim_id"
            "   AND cr.recorded_until IS NULL AND cr.state = 'active'"
            " JOIN claim_evidence ce ON ce.claim_id = c.claim_id"
            "   AND ce.revision = cr.revision"
            " JOIN spans sp ON sp.span_id = ce.span_id"
            " WHERE sp.source_id = ? AND sp.revision = ? LIMIT 1",
            (source_id, int(revision)),
        ).fetchone()
        return row is not None

    def _source_has_embeddings(
        self, conn: Any, source_id: str, revision: int
    ) -> bool:
        if "embeddings" not in self._tables:
            return False
        row = conn.execute(
            "SELECT 1 FROM embeddings e"
            " JOIN spans sp ON sp.span_id = e.span_id"
            " WHERE sp.source_id = ? AND sp.revision = ? LIMIT 1",
            (source_id, int(revision)),
        ).fetchone()
        return row is not None

    def _settle_receipt_stages(
        self,
        conn: Any,
        source_id: str,
        revision: int,
        *,
        exclude_job_id: Optional[str] = None,
    ) -> None:
        """Roll up one source revision's receipt DAGs after a commit.

        Called inside the job's fenced commit transaction so obligation
        transitions land atomically with the effects they describe
        (V4-14.02). ``screened`` fulfills whenever this settle runs —
        reaching it means the drain-time gate committed. Downstream
        stages settle only once no harvest/admit work for this revision
        remains outstanding; the LAST finishing job performs the roll-up,
        so a sibling still in flight cannot produce premature readiness.

        - ``lexical_ready``: succeeded when an active claim indexed for
          this revision, else ``deferred`` (nothing admissible).
        - ``derived_ready``: succeeded with the claim set (relation passes
          commit inside the admit job), else ``deferred``.
        - ``semantic_ready``: ``deferred`` when no encoder is provisioned;
          pending while an owed embed job is live; succeeded when the
          revision's spans carry embeddings; else ``deferred``.
        """
        if self._readiness is None:
            return
        rids = self._readiness.ensure_for_source(conn, source_id, revision)
        if not rids:
            return
        for rid in rids:
            self._readiness.try_fulfill(conn, rid, CapabilityName.SCREENED)
        if self._outstanding_pipeline_jobs(
            conn, source_id, revision, exclude_job_id=exclude_job_id
        ):
            return
        has_claim = self._source_active_claim(conn, source_id, revision)
        embeds_owed = self._outstanding_embed_jobs(
            conn, source_id, revision, exclude_job_id=exclude_job_id
        )
        has_vectors = (
            self._source_has_embeddings(conn, source_id, revision)
            if has_claim
            else False
        )
        for rid in rids:
            if has_claim:
                self._readiness.try_fulfill(
                    conn, rid, CapabilityName.LEXICAL_READY
                )
                self._readiness.try_fulfill(
                    conn, rid, CapabilityName.DERIVED_READY
                )
            else:
                self._readiness.defer(
                    conn, rid, CapabilityName.LEXICAL_READY,
                    "nothing_admissible", _missing_ok=True,
                )
                self._readiness.defer(
                    conn, rid, CapabilityName.DERIVED_READY,
                    "nothing_admissible", _missing_ok=True,
                )
            if self.encoder is None:
                self._readiness.defer(
                    conn, rid, CapabilityName.SEMANTIC_READY,
                    "encoder_unavailable", _missing_ok=True,
                )
            elif embeds_owed:
                continue  # an owed embed job is still live — keep waiting
            elif has_vectors:
                self._readiness.try_fulfill(
                    conn, rid, CapabilityName.SEMANTIC_READY
                )
            else:
                self._readiness.defer(
                    conn, rid, CapabilityName.SEMANTIC_READY,
                    "no_embeddings" if has_claim else "nothing_to_embed",
                    _missing_ok=True,
                )

    def _reopen_and_settle(
        self, conn: Any, source_id: str, revision: int
    ) -> None:
        """Re-open ``deferred`` downstream stages and re-settle — used when
        review activation lands a claim AFTER the initial drain declared
        the source's pipeline empty (a deferred stage is owed backlog, not
        a permanent verdict)."""
        if self._readiness is None:
            return
        for rid in self._readiness.ensure_for_source(conn, source_id, revision):
            for cap in (
                CapabilityName.LEXICAL_READY,
                CapabilityName.DERIVED_READY,
                CapabilityName.SEMANTIC_READY,
            ):
                if self._readiness.state_of(conn, rid, cap) == (
                    ReadinessState.DEFERRED.value
                ):
                    self._readiness.pend(conn, rid, cap, _missing_ok=True)
        self._settle_receipt_stages(conn, source_id, revision)

    def _fail_job_obligations(
        self, conn: Any, job: dict[str, Any], code: str
    ) -> None:
        """Land a terminal job failure on the receipt's obligation rows —
        same transaction as the job's own ``failed`` transition, so a dead
        pipeline stage is durable and queryable (V4-14.07).

        The stage row fails only when this was the LAST live job able to
        deliver it for the source revision — a sibling admit/embed still
        in flight may yet deliver the capability, so then the receipt's
        ``failed`` indicator fires (a real failure occurred) while the
        stage stays pending. Dependents of a failed stage cancel through
        the readiness cascade; committed upstream work is never rewritten.
        """
        if self._readiness is None:
            return
        cap = job_stage_capability(job.get("kind"))
        if cap is None:
            return
        try:
            pairs = self._job_source_pairs(conn, job)
        except Exception:
            return  # malformed refs — the job failure itself stands
        for sid, rev in pairs:
            try:
                rids = self._readiness.ensure_for_source(conn, sid, rev)
            except VerbatimError:
                continue
            if cap is CapabilityName.SEMANTIC_READY:
                outstanding = self._outstanding_embed_jobs(
                    conn, sid, rev, exclude_job_id=job["job_id"]
                )
            else:
                outstanding = self._outstanding_pipeline_jobs(
                    conn, sid, rev, exclude_job_id=job["job_id"]
                )
            for rid in rids:
                try:
                    if outstanding:
                        self._readiness.flag_failure(
                            conn, rid, code, _missing_ok=True
                        )
                    else:
                        self._readiness.fail(
                            conn, rid, cap, code, _missing_ok=True
                        )
                except VerbatimError as exc:
                    # A terminal row (e.g. already succeeded) is never
                    # rewritten by a late duplicate failure.
                    if exc.code is not ErrorCode.INVALID_TRANSITION:
                        raise

    # Explicit operator actions bypass capture.enabled — that flag governs
    # automatic conversation capture (the provider's per-turn writes), not
    # deliberate imports (SPEC §10: capture defaults are "opt-in per channel").
    _EXPLICIT_KINDS = {SourceKind.IMPORT, SourceKind.OPERATOR_RECORD}

    def _gate(self, envelope: SourceEnvelope) -> Optional[SecurityVerdict]:
        """Privacy/scope checks that run BEFORE any payload is persisted.

        Returns a non-clean screening verdict when the payload matches a
        rules_v1 finding — the caller persists the source under a
        quarantine hold instead of refusing it (§34: held content stays
        inspectable; harvest/admit re-check the hold at drain time)."""
        cfg = self.cfg
        kind = envelope.source_kind
        if kind in self._EXPLICIT_KINDS:
            pass
        elif not cfg.capture.enabled:
            raise VerbatimError(ErrorCode.CAPTURE_DISABLED, "capture is disabled")
        if len(envelope.payload) > cfg.capture.max_source_bytes:
            raise VerbatimError(ErrorCode.VALIDATION, "source exceeds capture.max_source_bytes")
        if kind == SourceKind.USER_MESSAGE and not cfg.capture.user_messages:
            raise VerbatimError(ErrorCode.CAPTURE_DISABLED, "user-message capture disabled")
        if kind == SourceKind.ASSISTANT_MESSAGE and not cfg.capture.assistant_context:
            raise VerbatimError(ErrorCode.CAPTURE_DISABLED, "assistant-context capture disabled")
        if kind == SourceKind.TOOL_OUTPUT and not cfg.capture.tool_outputs:
            raise VerbatimError(ErrorCode.CAPTURE_DISABLED, "tool-output capture disabled")
        # V4-13.11/13.12: the text channel accepts strictly valid UTF-8 —
        # malformed bytes are rejected at the gate, never replacement-
        # decoded into screening, persisted, and later harvested into
        # offsets that do not match the stored bytes. The check is
        # schema-independent: it runs before the optional screening
        # capability probe so a v1-schema store enforces it too.
        try:
            text = envelope.payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "source payload is not valid UTF-8 — binary payloads belong "
                "in artifact references, not the text channel",
            ) from exc
        # §34 stage-1 control on the primary write channel — every input
        # channel screens, not just the v3 envelope path (V3-14.09:
        # claimed provenance cannot clear a finding). Screening consumes
        # exactly the bytes that were accepted.
        if not (self._has("quarantine") and self._has("security_labels")):
            return None
        verdict = screen_content(text, source_trust="unknown")
        if verdict.attack_risk.value in ("suspicious", "blocked"):
            return verdict
        return None

    def _source_held(
        self,
        conn: Any,
        source_id: str,
        revision: int,
        *,
        span_id: Optional[str] = None,
    ) -> bool:
        """Drain-time hold check (§34): a quarantine row opened between
        enqueue and drain still wins — harvest/admit re-check all three
        ref kinds the screening paths write."""
        if not self._has("quarantine"):
            return False
        if is_quarantined(conn, "source", source_id, revision):
            return True
        if span_id is not None and is_quarantined(
            conn, "span", span_id, revision
        ):
            return True
        if self._has("source_envelopes"):
            rows = conn.execute(
                "SELECT envelope_id FROM source_envelopes"
                " WHERE source_id = ? AND revision = ?",
                (source_id, revision),
            ).fetchall()
            for (env_id,) in rows:
                if is_quarantined(
                    conn, "source_envelope", env_id, revision
                ):
                    return True
        return False

    def _source_suppressed(
        self,
        conn: Any,
        source_id: str,
        revision: int,
        *,
        span_id: Optional[str] = None,
    ) -> bool:
        """Erasure-tombstone check inside the caller's transaction.

        Mirrors ``PurgesRepo.suppressed_ids`` (same states, same object-id
        conventions) but reads on ``conn`` so the check rides inside the
        fenced commit: an erasure that lands between the pre-check and the
        commit still fences the admission (V4-42.03 — early read-time
        checks are insufficient).
        """
        if not (self._has("purges") and self._has("purge_targets")):
            return False
        targets = [
            ("source", source_id),
            ("source_revision", f"{source_id}:{revision}"),
        ]
        if span_id is not None:
            targets.append(("span", span_id))
        states = ("suppressed", "purging", "completed")
        clauses = " OR ".join(
            "(pt.object_kind = ? AND pt.object_id = ?)" for _ in targets
        )
        state_ph = ",".join("?" for _ in states)
        params = [v for pair in targets for v in pair] + list(states)
        row = conn.execute(
            "SELECT 1 FROM purge_targets pt"
            " JOIN purges p ON p.purge_id = pt.purge_id"
            f" WHERE ({clauses}) AND p.state IN ({state_ph}) LIMIT 1",
            params,
        ).fetchone()
        return row is not None

    def ingest(self, envelope: SourceEnvelope) -> IngestReceipt:
        """Persist the source revision and enqueue its harvest job atomically."""
        verdict = self._gate(envelope)
        source_id = envelope.source_id or new_id()
        with self.store.tx() as conn:
            sid, created = self.sources.insert(envelope, conn=conn)
            if not created:
                return IngestReceipt(
                    accepted=(), rejected=(), job_ids=(),
                    projection_generation=self._gen_tx(conn),
                    duplicate=True,
                )
            if verdict is not None:
                sid_scope = _source_scope_id(self.sources, sid, envelope)
                attach_label(
                    conn, sid_scope,
                    source_trust="unknown",
                    content_form=verdict.content_form.value,
                    attack_risk=verdict.attack_risk.value,
                    review_state=default_review_state(
                        "unknown",
                        attack_risk=verdict.attack_risk.value,
                        content_form=verdict.content_form.value,
                        findings=list(verdict.findings),
                    ),
                    findings=list(verdict.findings),
                    method=verdict.method,
                    rules_revision=verdict.rules_revision,
                )
                reasons = [
                    f"attack_risk:{verdict.attack_risk.value}"
                ] + sorted(
                    {
                        str(f.get("rule_id"))
                        for f in verdict.findings
                        if f.get("rule_id")
                    }
                )
                open_quarantine(
                    conn,
                    ("source", sid, envelope.revision),
                    reasons,
                    list(verdict.findings),
                    scope_id=sid_scope,
                )
            EventsRepo(self.store).append(
                conn, scope_key(envelope.scope), "source_accepted", "engine",
                {"source_id": sid, "kind": envelope.source_kind.value},
                self.policy.policy_version,
            )
            dedup = self.store.hmac(f"harvest:{sid}:{envelope.revision}".encode())
            # The harvest job carries a stable operation key so a redelivered
            # execution replays its receipt instead of harvesting the same
            # source twice (V2-39.02/39.10).
            job_id = self.jobs.enqueue(
                conn, scope_key(envelope.scope), JobKind.HARVEST,
                {"source_id": sid, "revision": envelope.revision},
                dedup_key=dedup,
                operation_key=(
                    f"harvest:{sid}:{envelope.revision}"
                    if self.jobs.supports_durability
                    else None
                ),
            )
            # V4-14.01: the processing obligations are durable rows in the
            # SAME transaction as the source + harvest job — capture
            # acceptance and its readiness DAG commit or roll back together.
            self._record_capture_obligations(
                conn, sid, envelope.revision, scope_key(envelope.scope)
            )
            return IngestReceipt(
                accepted=(sid,), rejected=(), job_ids=(job_id,),
                projection_generation=self._gen_tx(conn),
            )

    # ------------------------------------------------------------------
    # drain
    # ------------------------------------------------------------------

    def run_pending(
        self,
        scope: Optional[Scope] = None,
        limit: int = 64,
        owner: str = "inline",
        kinds: Optional[Iterable[Any]] = None,
        lane: Optional[str] = None,
        lease_s: float = DEFAULT_LEASE_S,
    ) -> int:
        """Drain due jobs synchronously. Used by CLI/tests; Hermes uses its worker.

        Control lane (purge/reindex/review_apply) drains before ordinary
        work — suppression and correctness obligations can never wait behind
        enrichment (V2-39.11). ``kinds`` narrows the kind set (default: every
        registered JobKind); ``lane`` restricts to one lane for dedicated
        workers. A job past ``deadline_us`` is failed DEADLINE_EXCEEDED
        without executing — expiry is a visible transition, not a silent
        drop (SPEC §43).

        Returns the number of jobs processed; ``drain_report`` exposes the
        honest processed/succeeded/failed/deferred/still-pending breakdown
        (V4-14.05).
        """
        return self.drain_report(
            scope=scope, limit=limit, owner=owner, kinds=kinds,
            lane=lane, lease_s=lease_s,
        )["processed"]

    # Settlement retries: a job left ``leased`` waits out its whole lease
    # window before ``reclaim_expired`` frees it, parking its receipts'
    # readiness obligations (and every barrier waiting on them) for tens
    # of seconds. Transient STORE_BUSY/BACKPRESSURE clears in
    # milliseconds, so a short bounded retry keeps settlement prompt
    # without weakening fencing — the queue still asserts owner +
    # generation inside the retried transaction.
    _SETTLE_ATTEMPTS = 8
    _SETTLE_SLEEP_S = 0.05

    def _settle(self, fn: Callable[[], Any]) -> Any:
        """Retry a settlement transition through transient contention."""
        last: Optional[BaseException] = None
        for _ in range(self._SETTLE_ATTEMPTS):
            try:
                return fn()
            except VerbatimError as exc:
                if not exc.retryable:
                    raise
                last = exc
            except sqlite3.OperationalError as exc:
                low = str(exc).lower()
                if "locked" not in low and "busy" not in low:
                    raise
                last = exc
            time.sleep(self._SETTLE_SLEEP_S)
        assert last is not None
        raise last

    def _settle_tx(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """``_settle`` around ``store.tx()`` — the settlement write itself."""

        def _run() -> Any:
            with self.store.tx() as conn:
                return fn(conn)

        return self._settle(_run)

    def _fail_leased(
        self,
        conn: sqlite3.Connection,
        job: dict[str, Any],
        owner: str,
        code: str,
        retryable: bool,
    ) -> Optional["JobState"]:
        """fail() + same-tx obligation settlement (must stay atomic)."""
        st = self.jobs.fail(
            conn, job["job_id"], owner, job["generation"], code, retryable
        )
        if st is JobState.FAILED:
            self._fail_job_obligations(conn, job, code)
        return st

    def drain_report(
        self,
        scope: Optional[Scope] = None,
        limit: int = 64,
        owner: str = "inline",
        kinds: Optional[Iterable[Any]] = None,
        lane: Optional[str] = None,
        lease_s: float = DEFAULT_LEASE_S,
        priority_sources: Optional[AbstractSet[str]] = None,
    ) -> dict[str, Any]:
        """Drain due jobs and report honest per-outcome counts (V4-14.05).

        ``processed`` counts leased-and-dispatched jobs; ``succeeded`` jobs
        whose effects committed; ``failed`` jobs terminally failed — each
        terminal failure also marks the matching readiness obligation in
        the same transaction, so a dead pipeline stage is durable, not
        just a job row; ``deferred`` jobs returned to ``retry_wait`` for a
        later attempt; ``still_pending`` is the durable queue remainder —
        a caller-supplied ``limit`` reaching zero is NOT completion.

        ``priority_sources`` (V6-02.08, v6_contracts §3): the source ids a
        live session barrier is blocked on. While the set is non-empty the
        per-job pick runs three phases — the normal lane-priority scan
        over the non-ordinary lanes, then up to ``limit`` marked
        ``source_project``/``source_embed`` jobs through
        ``JobQueue.lease_priority`` under identical fencing, then ordinary
        dequeue. The mark is advisory only: privacy/correctness lanes
        still drain first, and ``priority_processed`` counts how many
        jobs were taken via the unblock-first pass.
        """
        if limit < 1:
            raise VerbatimError(ErrorCode.VALIDATION, "limit must be >= 1")
        if lane is not None and lane not in LANES:
            raise VerbatimError(ErrorCode.VALIDATION, f"unknown job lane {lane!r}")
        kind_list = list(kinds) if kinds is not None else list(JobKind)
        prio = frozenset(
            s
            for s in (priority_sources or ())
            if isinstance(s, str) and s
        )
        # Reclaim expired leases first: a worker that crashed mid-job leaves
        # the row ``leased`` forever — the bumped generation fences its
        # stale commits while the obligation returns to the queue (V2-39).
        self._settle(self.jobs.reclaim_expired)
        # Drain in dequeue-priority order (V3-40): privacy/correctness lanes
        # first, then maintenance, then background learning, then ordinary.
        # V4-14.06: the priority scan runs before EVERY job, not once per
        # lane — a lower-priority job that enqueues follow-up work into an
        # earlier lane mid-drain must see that work reconsidered before the
        # rest of its lane drains, never deferred to a later pass. The
        # lease's ORDER BY already carries the global lane ordering, so one
        # unscoped scan per iteration yields the highest-priority due job.
        report: dict[str, Any] = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "deferred": 0,
            "expired": 0,
            "priority_processed": 0,
            "errors": [],
            "limit": limit,
        }
        while report["processed"] < limit:
            # Foreground-writer interleave: a writer stalled in BEGIN marks
            # the per-path writer-wait registry in ``storage.store``;
            # yield a real unlocked window BEFORE leasing the next job so
            # its next ~1 ms retry lands on a free WAL write lock rather
            # than starving behind this drain's back-to-back transactions.
            # Advisory + bounded — dequeue order, fencing, and the busy
            # cap are unchanged.
            try:
                if writers_waiting(self.store.path):
                    time.sleep(_WRITER_YIELD_S)
            except Exception:
                pass
            job, is_priority = self._drain_lease(
                scope, kind_list, owner=owner, lane=lane,
                lease_s=lease_s, priority_sources=prio,
            )
            if job is None:
                break
            if is_priority:
                report["priority_processed"] += 1
                # V8-13.05 honoring V6-02.08: a job leased through the
                # unblock-first pass coalesces only siblings that serve
                # the same barrier — other sources' queued work stays
                # queued for the ordinary pass, so a bounded priority
                # drain never spends its budget off-target.  Ephemeral
                # key: never persisted, never part of the op digest.
                job["_coalesce_sources"] = prio
            deadline = job.get("deadline_us")
            if deadline is not None and now_us() > deadline:
                # Expiry is a visible terminal transition — and it must
                # fail the receipt's obligation in the SAME transaction so
                # the pipeline stage cannot sit pending forever (V4-14.03).
                st = self._settle_tx(
                    lambda conn, _job=job: self._fail_leased(
                        conn, _job, owner,
                        ErrorCode.DEADLINE_EXCEEDED.value, False,
                    )
                )
                report["failed"] += 1
                report["expired"] += 1
                report["errors"].append(
                    {"job_id": job["job_id"], "kind": job["kind"],
                     "code": ErrorCode.DEADLINE_EXCEEDED.value}
                )
                report["processed"] += 1
                continue
            try:
                self._execute(job, owner)
            except VerbatimError as exc:
                st = self._settle_tx(
                    lambda conn, _job=job, _exc=exc: self._fail_leased(
                        conn, _job, owner, _exc.code.value, _exc.retryable,
                    )
                )
                if st is JobState.RETRY_WAIT:
                    report["deferred"] += 1
                elif st is JobState.FAILED:
                    report["failed"] += 1
                    report["errors"].append(
                        {"job_id": job["job_id"], "kind": job["kind"],
                         "code": exc.code.value}
                    )
            except Exception:
                self._settle_tx(
                    lambda conn, _job=job: self._fail_leased(
                        conn, _job, owner, "INTERNAL", True,
                    )
                )
                raise
            else:
                self._settle(
                    lambda _job=job: self.jobs.complete(
                        _job["job_id"], owner, _job["generation"]
                    )
                )
                report["succeeded"] += 1
            report["processed"] += 1
            # Yield the GIL between jobs: during a long drain the worker's
            # Python-side prescan/derivation competes with live readers —
            # a bare sleep(0) lets a waiting reader thread cut in at job
            # boundaries instead of mid-operation, and costs nothing.
            time.sleep(0)
            if report["processed"] % 8 == 0:
                # Keep the WAL small *within* a drain pass: a PASSIVE
                # checkpoint every few jobs (never blocking, on its own
                # connection) prevents SQLite's autocheckpoint from
                # firing inside whichever write tx next crosses the page
                # threshold — mid-commit checkpoints cost 100-250 ms and
                # stall every writer queued behind them. Best-effort: a
                # failed checkpoint just retries at the next interval.
                try:
                    self.store.checkpoint_passive()
                except Exception:
                    pass
        report["errors"] = report["errors"][:32]
        stats = self.jobs.stats(scope)
        report["still_pending"] = int(stats["pending"]) + int(stats["leased"])
        report["queue"] = stats
        report["complete"] = report["still_pending"] == 0
        if self._readiness is not None:
            # ``scope`` arrives as ``Union[Scope, str, None]`` — a raw
            # scope_id string is already canonical (and was validated by
            # ``jobs.stats`` above); only a Scope needs deriving.
            scope_id = (
                scope_key(scope)
                if isinstance(scope, Scope)
                else scope
            )
            # Depth only — a COUNT(*) with pending()'s identical filter;
            # materializing every obligation row per pass scales the
            # drain loop with table size, not work done.
            report["pending_obligations"] = self._readiness.pending_count(
                scope_id
            )
        if report["processed"]:
            # Keep the WAL small between drain passes: a PASSIVE
            # checkpoint here (never blocking, on its own connection)
            # prevents SQLite's autocheckpoint from firing inside an
            # unrelated write tx — mid-commit checkpoints cost
            # 100-250 ms under drain load. Best-effort: a failed
            # checkpoint just retries on the next pass.
            try:
                self.store.checkpoint_passive()
            except Exception:
                pass
        self._last_drain_report = report
        return report

    def _drain_lease(
        self,
        scope: Optional[Scope],
        kind_list: list,
        *,
        owner: str,
        lane: Optional[str],
        lease_s: float,
        priority_sources: frozenset,
    ) -> tuple[Optional[dict[str, Any]], bool]:
        """Pick the next job for the drain loop; the flag marks picks made
        through the unblock-first pass (V6-02.08).

        Per pick, while ``priority_sources`` is non-empty:

        1. the normal lane-priority scan over the non-ordinary lanes —
           privacy/correctness work is re-examined before EVERY job and
           can never be demoted behind a marked source (V4-14.06);
        2. the bounded unblock-first pass — ``source_project``/
           ``source_embed`` jobs whose payload ``source_id`` is in the
           marked set, leased through ``JobQueue.lease_priority`` under
           identical fencing, restricted to kinds the caller permits;
        3. ordinary dequeue.

        With no marked sources — or a pinned ``lane`` — the pick is the
        existing single lane-ordered lease, byte-for-byte the old path.
        """
        prio_kinds: list[str] = []
        if priority_sources:
            for k in kind_list:
                try:
                    v = k.value if isinstance(k, JobKind) else JobKind(k).value
                except ValueError:
                    continue
                if v in _PRIORITY_KINDS:
                    prio_kinds.append(v)
        if lane is not None:
            # A pinned-lane drain keeps its lane contract: marked source
            # jobs inside that lane still go first, then the lane's
            # ordinary ordering — nothing outside the lane is touched.
            if prio_kinds:
                got = self._settle(
                    lambda: self.jobs.lease_priority(
                        scope, prio_kinds, source_ids=priority_sources,
                        owner=owner, limit=1, lease_s=lease_s, lane=lane,
                    )
                )
                if got:
                    return got[0], True
            got = self._settle(
                lambda: self.jobs.lease(
                    scope, kind_list, owner=owner, limit=1,
                    lease_s=lease_s, lane=lane,
                )
            )
            return (got[0], False) if got else (None, False)
        if not prio_kinds:
            got = self._settle(
                lambda: self.jobs.lease(
                    scope, kind_list, owner=owner, limit=1,
                    lease_s=lease_s,
                )
            )
            return (got[0], False) if got else (None, False)
        # Phase 1 — privacy/correctness lanes before anything else.
        got = self._settle(
            lambda: self.jobs.lease_priority(
                scope, kind_list, owner=owner, limit=1,
                lease_s=lease_s, lane=_NON_ORDINARY_LANES,
            )
        )
        if got:
            return got[0], False
        # Phase 2 — the unblock-first pass over marked sources. These jobs
        # live in the ordinary lane; anything non-ordinary was already
        # leased above, so the pass can never demote a privacy lane.
        got = self._settle(
            lambda: self.jobs.lease_priority(
                scope, prio_kinds, source_ids=priority_sources,
                owner=owner, limit=1, lease_s=lease_s, lane="ordinary",
            )
        )
        if got:
            return got[0], True
        # Phase 3 — ordinary throughput.
        got = self._settle(
            lambda: self.jobs.lease(
                scope, kind_list, owner=owner, limit=1,
                lease_s=lease_s, lane="ordinary",
            )
        )
        return (got[0], False) if got else (None, False)

    def _execute(self, job: dict[str, Any], owner: str) -> None:
        """Dispatch one leased job to its handler.

        Every registered JobKind has a handler: a job that resolves to no
        handler is a VALIDATION failure, never a silent no-op — and so is a
        malformed ``input_refs`` payload.
        """
        try:
            kind = JobKind(job["kind"])
        except ValueError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"no handler for job kind {job['kind']!r}"
            ) from exc
        if not isinstance(job["input_refs"], dict):
            raise VerbatimError(
                ErrorCode.VALIDATION, "job input_refs must be a mapping"
            )
        if kind == JobKind.HARVEST:
            self._do_harvest(job, owner)
        elif kind == JobKind.ADMIT:
            self._do_admit(job, owner)
        elif kind == JobKind.COMPARE:
            # Declared kind without a handler — never misroute to
            # admission (a claim-refs payload would fail VALIDATION or
            # trigger a spurious second admit).
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "job kind 'compare' has no handler in this build",
            )
        elif kind == JobKind.EMBED:
            self._do_embed(job, owner)
        elif kind == JobKind.PURGE:
            self._do_purge(job, owner)
        elif kind == JobKind.REVIEW_APPLY:
            self._do_review_apply(job, owner)
        elif kind == JobKind.REINDEX:
            self._do_reindex(job, owner)
        elif kind in _RECORD_EVENTS:
            self._do_record_only(job, owner)
        elif kind in _V3_KIND_HANDLERS:
            self._do_v3(job, owner, kind)
        elif kind in _v5_kind_handlers():
            self._do_v5(job, owner, kind)
        else:
            # V3 pipeline kinds without a registered handler fail loudly
            # instead of completing as silent no-ops (V3-40, §03 F-flags).
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"job kind {kind.value!r} has no handler in this build",
            )

    def _do_v3(self, job: dict[str, Any], owner: str, kind: JobKind) -> None:
        """Dispatch a v3 pipeline job to its owning module's handler.

        Handlers are lazily imported (``_V3_KIND_HANDLERS``): an absent or
        unprovisioned module surfaces CAPABILITY_UNAVAILABLE, never a silent
        no-op and never a retryable crash loop.
        """
        module_name, func_name = _V3_KIND_HANDLERS[kind]
        try:
            import importlib

            module = importlib.import_module(module_name)
            handler = getattr(module, func_name)
        except (ImportError, AttributeError) as exc:
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"job kind {kind.value!r} handler unavailable: {exc}",
            ) from exc
        if kind is JobKind.SCREEN:
            self._preflight_screen_payload(job)
        handler(job, owner, self)

    def _do_v5(self, job: dict[str, Any], owner: str, kind: JobKind) -> None:
        """Dispatch a v5 source-pipeline job (SPEC_V5 §08.16).

        Same lazy-import convention as ``_do_v3`` — an absent module is a
        loud CAPABILITY_UNAVAILABLE, never a silent no-op.
        """
        module_name, func_name = _v5_kind_handlers()[kind]
        try:
            import importlib

            module = importlib.import_module(module_name)
            handler = getattr(module, func_name)
        except (ImportError, AttributeError) as exc:
            raise VerbatimError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"job kind {kind.value!r} handler unavailable: {exc}",
            ) from exc
        handler(job, owner, self)

    def _preflight_screen_payload(self, job: dict[str, Any]) -> None:
        """Strict UTF-8 re-check before a ``screen`` job resolves a
        ``source`` object (V4-13.11/13.12).

        Screening must consume exactly the bytes the harvester would see —
        a persisted payload that cannot decode (corruption, or a row that
        predates strict acceptance) fails the job with a typed VALIDATION
        error instead of reaching a replacement-decode path where
        malformed bytes could screen as clean content. Other ref shapes
        (inline ``text``, ``text_ref`` spans) are already strictly decoded
        or validated by the handler itself and pass through untouched.
        """
        refs = job["input_refs"]
        if refs.get("object_kind") != "source":
            return
        object_id = refs.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            return  # the handler's own _require_ref reports this
        revision = refs.get("revision")
        if isinstance(revision, bool):
            return
        try:
            rev = int(revision) if revision is not None else 1
        except (TypeError, ValueError):
            return
        # Verified read: SourcesRepo.payload re-checks payload_hmac, so a
        # tampered revision fails STORE_CORRUPT here instead of being
        # screened as content by the handler (V4-07.02).
        payload = self.sources.payload(object_id, rev)
        if payload is None:
            return  # the handler raises EVIDENCE_UNAVAILABLE
        try:
            bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"screen source {object_id!r} rev {rev} payload is not"
                " valid UTF-8 — a decode failure is not clean content",
            ) from exc

    # ------------------------------------------------------------------
    # fencing + operation receipts (V2-39.01/39.06/39.10)
    # ------------------------------------------------------------------

    def _op_digest(self, job: dict[str, Any]) -> bytes:
        """Stable input digest for the job's operation receipt.

        Covers the kind plus the caller-visible refs — the private enqueue
        timestamp is already stripped by the queue's row decode.
        """
        canonical = json_dumps({"kind": job["kind"], "input_refs": job["input_refs"]})
        return self.store.hmac(canonical.encode("utf-8"))

    def _commit_effects(
        self,
        conn: Any,
        job: dict[str, Any],
        owner: str,
        effect_kind: str,
        apply_fn: Any,
        *,
        complete_job: bool = True,
    ) -> dict[str, Any]:
        """Fenced, receipt-checked commit for one job's domain effects.

        Order inside the caller's transaction: ``assert_lease`` → operation
        receipt lookup (a committed receipt replays and skips the effect
        entirely) → ``apply_fn(conn)`` → record the receipt → ``complete``.
        The lease fence rides in the same transaction as the effects, so a
        worker holding a superseded generation can never commit (V2-39.05),
        and effect + receipt + completion commit or roll back together
        (V2-39.01/39.10).

        ``complete_job=False`` defers the completion transition to the
        caller (used by ``_do_admit``): the receipt still commits with the
        effects, but the job stays leased so additional fenced follow-up
        transactions — each re-asserting the same generation — can run
        before the terminal completion (V4-42.03).
        """
        job_id = job["job_id"]
        self.jobs.assert_lease(conn, job_id, owner, job["generation"])
        op_key = job.get("operation_key") if self.ops is not None else None
        if op_key:
            prior = self.ops.check(
                conn, job["scope_id"], op_key, self._op_digest(job)
            )
            if prior is not None:
                if complete_job:
                    self.jobs.complete(conn, job_id, owner, job["generation"])
                return {
                    "replayed": True,
                    "result": safe_json_loads(prior["receipt_json"]),
                }
        result = apply_fn(conn)
        if op_key:
            self.ops.record(
                conn,
                job["scope_id"],
                op_key,
                input_digest=self._op_digest(job),
                effect_kind=effect_kind,
                receipt=result if isinstance(result, dict) else {"result": result},
                committed_event=(
                    result.get("event_seq")
                    if isinstance(result, dict)
                    else None
                ),
            )
        if complete_job:
            self.jobs.complete(conn, job_id, owner, job["generation"])
        return {"replayed": False, "result": result}

    def _pre_fence(self, job: dict[str, Any], owner: str) -> None:
        """Cheap early fence before unfenceable work.

        ``admit``/``relate`` self-transact and encoder inference must run
        outside any write tx, so neither can be fenced directly — this
        read-snapshot check skips them when the lease is already dead. It is
        not a commit guarantee: ``assert_lease`` still runs inside the
        commit transaction.
        """
        with self.store.read() as conn:
            if not self.jobs.commit_if_current(
                conn, job["job_id"], owner, job["generation"]
            ):
                raise VerbatimError(
                    ErrorCode.LEASE_LOST,
                    f"job {job['job_id']} lease lost before domain work",
                    retryable=True,
                )

    def _replay_done(self, job: dict[str, Any], owner: str) -> bool:
        """Short-circuit a redelivered job whose receipt already committed.

        Used by handlers whose domain call self-transacts (``admit`` /
        ``relate`` must not be wrapped in an outer write tx): the receipt
        check and job completion still run in one generation-fenced tx, so
        the redelivery is fenced exactly like the commit path.
        """
        op_key = job.get("operation_key")
        if not op_key or self.ops is None:
            return False
        with self.store.tx() as conn:
            self.jobs.assert_lease(conn, job["job_id"], owner, job["generation"])
            prior = self.ops.check(
                conn, job["scope_id"], op_key, self._op_digest(job)
            )
            if prior is None:
                return False
            self.jobs.complete(conn, job["job_id"], owner, job["generation"])
            return True

    # ------------------------------------------------------------------
    # enrichment handlers (ordinary lane)
    # ------------------------------------------------------------------

    def _do_harvest(self, job: dict[str, Any], owner: str) -> None:
        refs = job["input_refs"]
        source_id = _require_ref(refs, "source_id")
        revision = _require_int_ref(refs, "revision")
        row = self.sources.get_revision(source_id, revision)
        if row is None:
            raise VerbatimError(ErrorCode.EVIDENCE_UNAVAILABLE, "source revision missing")
        envelope = envelope_for(self.store, source_id, revision)
        if envelope is None:
            raise VerbatimError(ErrorCode.EVIDENCE_UNAVAILABLE, "source revision missing")
        # §34 drain fence: a hold opened between enqueue and drain still
        # wins — a held source never derives claims (V3-14.10).
        with self.store.read() as hconn:
            if self._source_held(hconn, source_id, revision):
                raise VerbatimError(
                    ErrorCode.QUARANTINED,
                    f"source {source_id}@{revision} is held in quarantine",
                )
        try:
            result = _harvest_v3_result(self.store, source_id, revision, envelope)
        except UnicodeDecodeError as exc:
            # A persisted payload that cannot decode as UTF-8 (corruption,
            # or a row written before strict acceptance) is a permanent
            # typed failure — never retried, never harvested into guessed
            # offsets (V4-13.11/13.12).
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"source {source_id}@{revision} payload is not valid UTF-8",
            ) from exc
        scope_id = _source_scope_id(self.sources, source_id, envelope)
        op_key = f"harvest:{source_id}:{revision}:{HARVESTER_VERSION}"

        def _apply(conn: Any) -> dict[str, Any]:
            # persist_harvest supplies the span/context-group side of the
            # durable obligation (V2-11.13): deterministic span ids, one
            # context group with member roles + completeness, inside this
            # commit transaction. Replays produce identical identities.
            ph = persist_harvest(self.store, conn, envelope, result, op_key)
            for span in ph.spans:
                dedup = self.store.hmac(f"admit:{span.span_id}".encode())
                self.jobs.enqueue(
                    conn, scope_id, JobKind.ADMIT,
                    {"span_id": span.span_id, "source_id": source_id,
                     "revision": revision},
                    dedup_key=dedup,
                    operation_key=(
                        f"admit:{span.span_id}"
                        if self.jobs.supports_durability else None
                    ),
                )
            seq = EventsRepo(self.store).append(
                conn, scope_id, "harvested", "engine",
                {"source_id": source_id, "candidates": len(result.candidates),
                 "overflow": result.overflow_count,
                 "context_group_id": ph.group_id,
                 "completeness": ph.completeness},
                self.policy.policy_version,
            )
            # V4-14.02: the drain-time screen+segment gate committed —
            # ``screened`` fulfills on every receipt for this revision, and
            # a zero-span harvest settles the downstream stages as
            # deferred (nothing admissible) inside the same transaction.
            self._settle_receipt_stages(
                conn, source_id, revision, exclude_job_id=job["job_id"]
            )
            return {
                "source_id": source_id,
                "candidates": len(result.candidates),
                "overflow": result.overflow_count,
                "context_group_id": ph.group_id,
                "completeness": ph.completeness,
                "event_seq": seq,
            }

        with self.store.tx() as conn:
            self._commit_effects(conn, job, owner, "harvest", _apply)

    def _do_admit(self, job: dict[str, Any], owner: str) -> None:
        """Admit one harvested span under a generation-fenced lease.

        Atomicity contract (F4-09 / V4-09.02/09.03/42.03): the lease
        re-verification, the dependency re-checks (quarantine hold and
        erasure tombstone), the claim write set, the index publication,
        the embed obligation, the audit event, and the operation receipt
        all commit inside ONE transaction — a worker whose lease was
        cancelled or superseded between dequeue and commit rolls back
        with the tx and leaves no claim, edge, index row, receipt, or
        follow-up job behind.

        Follow-on relation work (``relate`` + the v3 proposal passes)
        cannot join that commit — their judge/encoder calls must not run
        inside a write tx (V4-09.09). They run afterwards through
        ``_LeaseFencedStore``, whose ``tx()`` re-asserts the lease inside
        each of their commit transactions, and the terminal ``complete``
        lands under the same generation. A redelivery after a crash mid-
        sequence replays the committed receipt and still heals the
        follow-on phase (edges/reviews dedup-converge).
        """
        refs = job["input_refs"]
        job_id = job["job_id"]
        generation = job["generation"]
        span_id = _require_ref(refs, "span_id")
        source_id = _require_ref(refs, "source_id")
        revision = _require_int_ref(refs, "revision")
        self._pre_fence(job, owner)

        # Receipt replay (V2-39.10): a committed receipt IS the durable
        # record — skip re-deriving the proposal (the span may have been
        # purged since) but still run the fenced follow-on phase and the
        # completion below so a crash between the admission commit and
        # the relation pass converges instead of stranding the job.
        receipt: Optional[dict[str, Any]] = None
        op_key = job.get("operation_key") if self.ops is not None else None
        if op_key:
            with self.store.read() as rconn:
                prior = self.ops.check(
                    rconn, job["scope_id"], op_key, self._op_digest(job)
                )
            if prior is not None:
                decoded = safe_json_loads(prior["receipt_json"])
                receipt = decoded if isinstance(decoded, dict) else {}

        ekind = _v3_envelope_kind(self.store, source_id, revision)
        scope_id = job["scope_id"]

        if receipt is None:
            text = self.spans.text(span_id)
            if text is None:
                raise VerbatimError(ErrorCode.EVIDENCE_UNAVAILABLE, "span purged")
            span_row = self.spans.get(span_id)
            if span_row is None:
                raise VerbatimError(ErrorCode.EVIDENCE_UNAVAILABLE, "span missing")
            envelope = envelope_for(self.store, source_id, revision)
            if envelope is None:
                raise VerbatimError(ErrorCode.EVIDENCE_UNAVAILABLE, "source revision missing")
            # §34 drain fence pre-check: a hold opened between enqueue and
            # drain still wins — a held span/source/envelope never admits
            # a claim. Both checks are RE-VERIFIED inside the commit tx
            # (``_apply``); the early pass only skips wasted work.
            with self.store.read() as aconn:
                if self._source_held(
                    aconn, source_id, revision, span_id=span_id
                ):
                    raise VerbatimError(
                        ErrorCode.QUARANTINED,
                        f"span {span_id} source {source_id}@{revision} is held"
                        " in quarantine",
                    )
                if self._source_suppressed(
                    aconn, source_id, revision, span_id=span_id
                ):
                    raise VerbatimError(
                        ErrorCode.EVIDENCE_UNAVAILABLE,
                        f"span {span_id} source {source_id}@{revision} is"
                        " suppressed by erasure",
                    )
            # Persisted absolute offsets — never re-derive from the excerpt
            # length, or spans mid-source would point at the wrong bytes
            # (v1 defect: second-paragraph claims cited byte 0).
            span = SpanRef(
                span_id=span_id, source_id=source_id, revision=revision,
                start_byte=span_row["start_byte"], end_byte=span_row["end_byte"],
            )
            proposal = propose(text, span, envelope, envelope.event_us)
            # Judge/registry/condition evaluation happens outside the write
            # tx; the plan carries its results into the fenced commit.
            plan = _admit_evaluate(
                self.store, proposal, envelope, self.policy
            )
            scope_id = _source_scope_id(self.sources, source_id, envelope)

            def _apply(conn: Any) -> dict[str, Any]:
                # In-transaction dependency re-verification: a hold or an
                # erasure tombstone landing between the pre-check above
                # and this commit still wins (V4-09.02/42.03).
                if self._source_held(
                    conn, source_id, revision, span_id=span_id
                ):
                    raise VerbatimError(
                        ErrorCode.QUARANTINED,
                        f"span {span_id} source {source_id}@{revision} is"
                        " held in quarantine",
                    )
                if self._source_suppressed(
                    conn, source_id, revision, span_id=span_id
                ):
                    raise VerbatimError(
                        ErrorCode.EVIDENCE_UNAVAILABLE,
                        f"span {span_id} source {source_id}@{revision} is"
                        " suppressed by erasure",
                    )
                outcome = _admit_apply(
                    self.store, conn, proposal, envelope, self.policy, plan
                )
                if outcome.claim_id and outcome.state == Lifecycle.ACTIVE:
                    # Projection rows + the embed obligation join the
                    # effect's commit transaction — a fenced-out worker
                    # leaves no half-visible claim (V2-39.01/39.06).
                    self._index_claim_text_tx(conn, outcome.claim_id, scope_id, text)
                    self._enqueue_embed(conn, scope_id, [span_id])
                if outcome.claim_id and ekind is not None:
                    # V3-15.13 + §15 table: claims harvested from
                    # host-observed structured output are state facts, not
                    # durable personal facts — declared `volatile` so
                    # retrieval delivers them with a `verify`
                    # recommendation until revalidated (V3-24.04/24.05).
                    from .evidence.envelopes import _STRUCTURAL_HARVEST

                    if ekind in {k.value for k in _STRUCTURAL_HARVEST}:
                        rev = conn.execute(
                            "SELECT MAX(revision) FROM claim_revisions"
                            " WHERE claim_id = ?",
                            (outcome.claim_id,),
                        ).fetchone()
                        if rev is not None and rev[0] is not None:
                            from .observations.freshness import set_freshness

                            set_freshness(
                                conn,
                                (scope_id, "claim", outcome.claim_id, int(rev[0])),
                                "volatile",
                            )
                seq = EventsRepo(self.store).append(
                    conn, scope_id, "admitted", "engine",
                    {"span_id": span_id, "claim_id": outcome.claim_id,
                     "state": outcome.state.value, "reason": outcome.reason},
                    self.policy.policy_version,
                )
                if outcome.claim_id and self._has("derivations"):
                    # V3-17.02: the claim is derived FROM its evidence span —
                    # deletion closure, invalidation, and influence tracing
                    # traverse this edge. Redelivery replays the identical
                    # edge (PK dedup); it rides the same commit so a fenced
                    # worker leaves no claim missing its derivation.
                    head_rev = conn.execute(
                        "SELECT MAX(revision) FROM claim_revisions"
                        " WHERE claim_id = ?",
                        (outcome.claim_id,),
                    ).fetchone()
                    if head_rev is not None and head_rev[0] is not None:
                        from .derivations import record_edge

                        record_edge(
                            conn,
                            ("claim", outcome.claim_id, int(head_rev[0])),
                            ("span", span_id, revision),
                            "producer",
                            "v3.admit",
                            scope_id,
                            seq=seq,
                        )
                return {
                    "span_id": span_id,
                    "claim_id": outcome.claim_id,
                    "state": outcome.state.value,
                    "event_seq": seq,
                }

            with self.store.tx() as conn:
                res = self._commit_effects(
                    conn, job, owner, "admit", _apply, complete_job=False
                )
            result = res["result"]
            claim_id = (
                result.get("claim_id") if isinstance(result, dict) else None
            )
        else:
            claim_id = receipt.get("claim_id")

        # --- fenced follow-on relation passes ---------------------------------
        # Each helper commits its own artifacts inside a tx that re-asserts
        # the lease (``_LeaseFencedStore``) — cancellation between the
        # admission commit and a follow-up leaves the claim whole but the
        # staged relation writes roll back (V4-42.03).
        if claim_id:
            fenced = _LeaseFencedStore(
                self.store, self.jobs, job_id, owner, generation
            )
            try:
                relate(fenced, claim_id, ctx=self.policy)
            except VerbatimError as exc:
                if exc.code is not ErrorCode.NOT_FOUND_OR_FORBIDDEN:
                    raise
                # The claim was erased between admission and the relation
                # pass — nothing left to relate; the receipt still stands.
            # V3 relation layer: verbatim retirement markers ("`x` was
            # retired") propose supersession to the review queue — the
            # detector binds identifier + shared topic, never applies
            # (G3 bounds its false-supersession rate). v2-origin sources
            # skip this step so the v2 contract stays byte-identical.
            if ekind is not None:
                from .evidence.supersession import (
                    propose_retirement_supersessions,
                )
                from .evidence.relations import (
                    propose_unstructured_relations,
                )

                propose_retirement_supersessions(
                    fenced, claim_id, scope_id
                )
                # V3-18.03/F26: unstructured claims never enter relate()'s
                # structured-predicate comparison — deterministic signals +
                # optional encoder corroboration flag contradiction pairs
                # for review instead. Never applies a transition.
                propose_unstructured_relations(
                    fenced, claim_id, scope_id, encoder=self.encoder
                )

        # Terminal fence: completion lands inside its own transaction under
        # the same generation — a cancel that raced any earlier phase stops
        # the job here instead of letting a stale worker report success.
        with self.store.tx() as conn:
            self.jobs.assert_lease(conn, job_id, owner, generation)
            self.jobs.complete(conn, job_id, owner, generation)
            # V4-14.02: the last outstanding admit job for this revision
            # rolls the receipt's lexical/derived/semantic obligations up
            # in the same transaction as its own completion.
            self._settle_receipt_stages(conn, source_id, revision)

    # ------------------------------------------------------------------
    # embed (ordinary lane)
    # ------------------------------------------------------------------

    def _enqueue_embed(self, conn: Any, scope_id: str, span_ids: Iterable[str]) -> None:
        """Queue one deduped EMBED job per span for the configured encoder.

        Runs inside the caller's commit transaction so the processing
        obligation is atomic with the state change that motivates it. A
        missing encoder is a capability note, not a failure — the embed
        handler itself records ``encoder_unavailable`` when it drains a job
        with no configured encoder.
        """
        if self.encoder is None:
            return
        enc_id = self.encoder.encoder_id
        for sid in dict.fromkeys(span_ids):
            dedup = self.store.hmac(f"embed:{sid}:{enc_id}".encode())
            self.jobs.enqueue(
                conn, scope_id, JobKind.EMBED,
                {"span_ids": [sid], "encoder_id": enc_id},
                dedup_key=dedup,
                operation_key=(
                    f"embed:{sid}:{enc_id}" if self.jobs.supports_durability else None
                ),
            )

    def _embed_rows(
        self, scope_id: str, encoder_id: str, span_ids: Optional[list[str]]
    ) -> list[dict[str, str]]:
        """Authorized span texts needing an embedding for ``encoder_id``.

        ``span_ids=None`` gathers primary evidence spans of the scope's
        active claims that lack a row for this encoder; an explicit list
        re-verifies scope membership on every span. Text is reconstructed
        from persisted source bytes at the span's stored offsets — the
        embedding always covers exactly the authorized bytes.
        """
        with self.store.read() as conn:
            if span_ids is not None:
                if not span_ids:
                    return []
                ph = ",".join("?" for _ in span_ids)
                rows = conn.execute(
                    "SELECT sp.span_id"
                    " FROM spans sp"
                    " JOIN sources s ON s.source_id = sp.source_id"
                    " JOIN source_revisions sr"
                    "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
                    f" WHERE sp.span_id IN ({ph}) AND s.scope_id = ?"
                    "   AND sr.payload IS NOT NULL"
                    " ORDER BY sp.span_id LIMIT ?",
                    (*span_ids, scope_id, _EMBED_BATCH + 1),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT DISTINCT sp.span_id"
                    " FROM claims c"
                    " JOIN claim_revisions cr ON cr.claim_id = c.claim_id"
                    "   AND cr.recorded_until IS NULL AND cr.state = 'active'"
                    " JOIN claim_evidence ce ON ce.claim_id = c.claim_id"
                    "   AND ce.revision = cr.revision AND ce.evidence_role = 'primary'"
                    " JOIN spans sp ON sp.span_id = ce.span_id"
                    " JOIN source_revisions sr"
                    "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
                    " WHERE c.scope_id = ? AND sr.payload IS NOT NULL"
                    "   AND NOT EXISTS (SELECT 1 FROM embeddings e"
                    "                   WHERE e.span_id = sp.span_id"
                    "                     AND e.encoder_id = ?)"
                    " ORDER BY sp.span_id LIMIT ?",
                    (scope_id, encoder_id, _EMBED_BATCH + 1),
                ).fetchall()
        out: list[dict[str, str]] = []
        for (sid,) in rows[: _EMBED_BATCH]:
            # Verified slice: SpansRepo.text re-checks the excerpt_hmac
            # over the stored bytes — a tampered payload fails the embed
            # job STORE_CORRUPT rather than embedding forged text
            # (V4-07.02); absent/purged excerpts skip honestly.
            text = self.spans.text(sid)
            if text is None:
                continue
            out.append({"span_id": sid, "text": text})
        return out

    def _do_embed(self, job: dict[str, Any], owner: str) -> None:
        refs = job["input_refs"]
        scope_id = job["scope_id"]

        def _note(reason: str) -> dict[str, Any]:
            def _apply(conn: Any) -> dict[str, Any]:
                seq = EventsRepo(self.store).append(
                    conn, scope_id, "embed_skipped", "engine",
                    {"job_id": job["job_id"], "reason": reason},
                    self.policy.policy_version,
                )
                # V4-14.02: a skipped embed still settles the receipt —
                # semantic_ready defers honestly rather than pending
                # forever on a job that produced no vectors.
                for sid, rev in self._job_source_pairs(conn, job):
                    self._settle_receipt_stages(
                        conn, sid, rev, exclude_job_id=job["job_id"]
                    )
                return {"encoded": 0, "note": reason, "event_seq": seq}

            with self.store.tx() as conn:
                self._commit_effects(conn, job, owner, "embed", _apply)

        if self.encoder is None:
            # Capability disabled is a recorded outcome, not a failure:
            # the job completes and the degradation is auditable
            # (V2-27.02 — never a silent no-op).
            _note("encoder_unavailable")
            return
        enc_id = self.encoder.encoder_id
        want = refs.get("encoder_id")
        if want is not None and want != enc_id:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"embed job targets encoder {want!r}; configured is {enc_id!r}",
            )
        span_ids = refs.get("span_ids")
        if span_ids is not None:
            if (
                not isinstance(span_ids, (list, tuple))
                or len(span_ids) > _EMBED_BATCH
                or not all(isinstance(s, str) and s for s in span_ids)
            ):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "embed span_ids must be a list of span id strings",
                )
            span_ids = list(dict.fromkeys(span_ids))
        if not self.encoder.available():
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE,
                "encoder backend is unavailable",
                retryable=True,
            )
        if self._replay_done(job, owner):
            return
        rows = self._embed_rows(scope_id, enc_id, span_ids)
        if not rows:
            _note("no_pending_inputs")
            return
        # Inference strictly before the write transaction opens (V2-39.07).
        texts = [r["text"] for r in rows]
        from .embeddings.encoder import encoder_requires_permit

        if encoder_requires_permit(self.encoder):
            # F4-04: remote encoding dispatches under a scoped dispatch
            # permit — consent, reservation, and payload binding all
            # verified before any transport I/O. No broker → EGRESS_DENIED.
            if self.transport_broker is None:
                raise VerbatimError(
                    ErrorCode.EGRESS_DENIED,
                    "remote encoder has no transport broker — dispatch "
                    "disabled (open through api.open_store to wire one)",
                    retryable=False,
                )
            blobs = self.transport_broker.encode_permitted(
                self.encoder,
                texts,
                scope_ids=(scope_id,),
                purpose="embed_document",
                caller=owner,
                job_id=job["job_id"],
                input_refs=refs,
            )
        else:
            blobs = self.encoder.encode(texts)
        if len(blobs) != len(rows):
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID,
                f"encoder returned {len(blobs)} vectors for {len(rows)} texts",
            )
        pre = _Preencoded(self.encoder, blobs)
        preprocessing = pre.manifest().get("preprocessing_version") or "v1"
        dep_digests = {
            r["span_id"]: self.store.hmac(r["text"].encode("utf-8")) for r in rows
        }

        def _apply(conn: Any) -> dict[str, Any]:
            n = encode_spans(self.store, conn, pre, rows)
            seq_marker = conn.execute(
                "SELECT COALESCE(MAX(event_seq), 0) FROM events"
            ).fetchone()[0]
            if self.embedding_inputs is not None:
                for r in rows:
                    self.embedding_inputs.record(
                        conn, r["span_id"], enc_id, preprocessing,
                        dep_digests[r["span_id"]], seq_marker,
                    )
            seq = EventsRepo(self.store).append(
                conn, scope_id, "embedded", "engine",
                {"job_id": job["job_id"], "encoder_id": enc_id, "encoded": n},
                self.policy.policy_version,
            )
            # V4-14.02: vectors committed — when this was the last owed
            # embed job for a source revision, its receipt's
            # semantic_ready fulfills in the same transaction.
            for sid, rev in self._sources_for_spans(
                conn, [r["span_id"] for r in rows]
            ):
                self._settle_receipt_stages(
                    conn, sid, rev, exclude_job_id=job["job_id"]
                )
            return {"encoded": n, "encoder_id": enc_id, "event_seq": seq}

        with self.store.tx() as conn:
            self._commit_effects(conn, job, owner, "embed", _apply)

    # ------------------------------------------------------------------
    # control lane: purge / review_apply / reindex
    # ------------------------------------------------------------------

    def _do_purge(self, job: dict[str, Any], owner: str) -> None:
        refs = job["input_refs"]
        purge_id = _require_ref(refs, "purge_id")

        def _apply(conn: Any) -> dict[str, Any]:
            prow = conn.execute(
                "SELECT state, scope_id FROM purges WHERE purge_id = ?",
                (purge_id,),
            ).fetchone()
            if prow is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "purge not found"
                )
            if prow[0] == "completed":
                # Idempotent re-run: the physical erasure already committed.
                return {
                    "purge_id": purge_id,
                    "scope_id": prow[1],
                    "already_completed": True,
                }
            result = execute_purge(self.store, purge_id, conn=conn)
            seq = EventsRepo(self.store).append(
                conn, result["scope_id"], "purge_executed", PURGE_ACTOR,
                {"purge_id": purge_id, "job_id": job["job_id"],
                 "targets": len(result["erased"]),
                 "erasure_epoch": result["erasure_epoch"]},
                self.policy.policy_version,
            )
            result["event_seq"] = seq
            return result

        with self.store.tx() as conn:
            self._commit_effects(conn, job, owner, "purge", _apply)

    def _do_review_apply(self, job: dict[str, Any], owner: str) -> None:
        refs = job["input_refs"]
        review_id = _require_ref(refs, "review_id")
        machine = LifecycleMachine(
            self.store, policy_version=self.policy.policy_version
        )

        def _apply(conn: Any) -> dict[str, Any]:
            row = conn.execute(
                "SELECT scope_id, state, proposed_effect_json,"
                " expected_versions_json FROM reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "review not found"
                )
            scope_id, state, effect_json, expected_json = row
            if state != "open":
                # Already resolved — nothing to apply; completing is the
                # honest outcome for a redelivered/duplicated apply job.
                return {"review_id": review_id, "skipped": state}
            effect = safe_json_loads(effect_json)
            expected = safe_json_loads(expected_json)
            if not isinstance(effect, dict) or not isinstance(expected, dict):
                raise VerbatimError(
                    ErrorCode.VALIDATION, "review payload malformed"
                )
            eff = effect.get("effect")
            target = effect.get("claim_id") or effect.get("predecessor_id")
            if not isinstance(eff, str) or not eff or not isinstance(target, str):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "review effect missing effect/target claim",
                )
            # Expected-version fence across every participating object
            # (V2-19.08): a drifted revision leaves the review open and
            # records why — it is NOT an execution failure.
            for cid, rev in expected.items():
                head = read_claim_head(conn, cid)
                if head is None or head.revision != int(rev):
                    EventsRepo(self.store).append(
                        conn, scope_id, "review_apply_stale", "engine",
                        {"review_id": review_id, "claim_id": cid,
                         "expected": rev,
                         "current": None if head is None else head.revision},
                        self.policy.policy_version,
                    )
                    return {"review_id": review_id, "stale": True, "claim_id": cid}
            fenced_rev = expected.get(target)
            head = read_claim_head(conn, target)
            if fenced_rev is None or head is None:
                # A review that does not fence its own target is stale by
                # definition (mirrors policy.apply_review) — leave it open.
                EventsRepo(self.store).append(
                    conn, scope_id, "review_apply_stale", "engine",
                    {"review_id": review_id, "claim_id": target,
                     "expected": fenced_rev,
                     "current": None if head is None else head.revision},
                    self.policy.policy_version,
                )
                return {"review_id": review_id, "stale": True, "claim_id": target}
            cmd = TransitionCommand(
                claim_id=target,
                expected_revision=int(fenced_rev),
                effect=eff,
                actor_id=effect.get("actor_id") or "review",
                reason=effect.get("reason") or "review_apply",
                successor_claim_id=(
                    effect.get("successor_claim_id") or effect.get("successor_id")
                ),
                interval=_interval_from(effect.get("interval")),
            )
            try:
                seq = machine.apply(cmd, conn)
            except VerbatimError as exc:
                if exc.code == ErrorCode.STALE_PROPOSAL:
                    EventsRepo(self.store).append(
                        conn, scope_id, "review_apply_stale", "engine",
                        {"review_id": review_id, "claim_id": target,
                         "error": exc.message},
                        self.policy.policy_version,
                    )
                    return {"review_id": review_id, "stale": True, "claim_id": target}
                raise
            ReviewsRepo(self.store).resolve(
                conn, review_id, ReviewState.APPROVED, seq
            )
            self._index_active_claim_tx(conn, target)
            EventsRepo(self.store).append(
                conn, scope_id, "review_applied", "engine",
                {"review_id": review_id, "claim_id": target,
                 "effect": eff, "event_seq": seq},
                self.policy.policy_version,
            )
            return {
                "review_id": review_id,
                "applied": eff,
                "claim_id": target,
                "event_seq": seq,
            }

        with self.store.tx() as conn:
            self._commit_effects(conn, job, owner, "review_apply", _apply)

    def _do_reindex(self, job: dict[str, Any], owner: str) -> None:
        refs = job["input_refs"]
        partition = refs.get("scope_partition")
        if partition is None:
            partition = refs.get("scope_id")
        if partition is not None and not isinstance(partition, str):
            raise VerbatimError(
                ErrorCode.VALIDATION, "reindex scope_partition must be a string"
            )
        if not self.store.fts_enabled:
            raise VerbatimError(
                ErrorCode.STORE_WRITE_FAILED,
                "FTS5 unavailable — lexical rebuild cannot run",
            )

        def _apply(conn: Any) -> dict[str, Any]:
            if partition:
                # Partition-scoped rebuild: retag at the CURRENT
                # generation, read through this tx. A global bump here
                # would strand every other partition's rows at the old
                # generation — per-partition generations do not exist.
                current = self.store._meta_get(conn, "projection_generation")
                gen = int(current or 0)
                scope_ids = [partition]
            else:
                # The generation bump rides in the same transaction as the
                # rebuilt rows — readers see the old projection or the new
                # one, never a half-applied rebuild (V2-28).
                gen = self.store.bump_generation(conn)
                scope_ids = sorted(
                    {
                        r[0]
                        for r in conn.execute(
                            "SELECT scope_id FROM fts_rows"
                            " UNION SELECT scope_id FROM claims"
                        )
                    }
                )
            indexed = self._rebuild_fts(conn, scope_ids, gen)
            seq = EventsRepo(self.store).append(
                conn, job["scope_id"], "reindexed", "engine",
                {"generation": gen, "partitions": len(scope_ids),
                 "claims": indexed, "scope_partition": partition},
                self.policy.policy_version,
            )
            return {
                "generation": gen,
                "claims_indexed": indexed,
                "partitions": len(scope_ids),
                "event_seq": seq,
            }

        with self.store.tx() as conn:
            self._commit_effects(conn, job, owner, "reindex", _apply)

    def _rebuild_fts(self, conn: Any, scope_ids: list[str], gen: int) -> int:
        """Replace a scope partition's FTS rows under generation ``gen``.

        Only live ACTIVE claims are indexed — pending/rejected/erased heads
        never enter the projection, and rows for claims that left ACTIVE
        die with the old generation.
        """
        indexed = 0
        for sid in scope_ids:
            conn.execute(
                "DELETE FROM facts_fts WHERE fts_row_id IN"
                " (SELECT row_id FROM fts_rows WHERE scope_id = ?)",
                (sid,),
            )
            conn.execute("DELETE FROM fts_rows WHERE scope_id = ?", (sid,))
            rows = conn.execute(
                "SELECT c.claim_id, cr.revision, sp.span_id"
                " FROM claims c"
                " JOIN claim_revisions cr ON cr.claim_id = c.claim_id"
                "   AND cr.recorded_until IS NULL AND cr.state = 'active'"
                " JOIN claim_evidence ce ON ce.claim_id = c.claim_id"
                "   AND ce.revision = cr.revision AND ce.evidence_role = 'primary'"
                " JOIN spans sp ON sp.span_id = ce.span_id"
                " JOIN source_revisions sr"
                "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
                " WHERE c.scope_id = ? AND sr.payload IS NOT NULL"
                " ORDER BY c.claim_id, ce.span_id",
                (sid,),
            ).fetchall()
            seen: set[str] = set()
            for claim_id, revision, span_id in rows:
                if claim_id in seen:
                    continue
                seen.add(claim_id)
                # Verified slice: SpansRepo.text re-checks the excerpt
                # digest — forged or shifted bytes abort the rebuild
                # STORE_CORRUPT rather than indexing them (V4-07.02).
                text = self.spans.text(span_id)
                if text is None:
                    continue
                FtsRepo(self.store).index(
                    conn, claim_id, revision, sid, gen, text
                )
                indexed += 1
            if self.projections is not None:
                # Deterministic build audit: a published build row plus a
                # CAS publish into active_projections (V2-28).
                latest = conn.execute(
                    "SELECT COALESCE(MAX(event_seq), 0) FROM events"
                ).fetchone()[0]
                bid = self.projections.build_create(
                    conn, "fts", sid,
                    manifest={"generation": gen, "claims": len(seen)},
                    snapshot_seq=latest,
                )
                self.projections.build_set_state(
                    conn, bid, "published", caught_up_seq=latest
                )
                cur = self.projections.active_get(sid, "fts")
                self.projections.active_publish(
                    conn, sid, "fts", bid, latest,
                    expected_cas=cur["cas_revision"] if cur else -1,
                )
        return indexed

    # ------------------------------------------------------------------
    # record-only kinds (domain services land elsewhere)
    # ------------------------------------------------------------------

    def _do_record_only(self, job: dict[str, Any], owner: str) -> None:
        """Deterministic core for kinds without an in-process domain service.

        Validates the input shape and durably records the request as an
        event inside the fenced commit — the job completes, the intent is
        auditable, and nothing is silently dropped (V2-39).
        """
        kind = JobKind(job["kind"])
        event_kind = _RECORD_EVENTS[kind]

        def _apply(conn: Any) -> dict[str, Any]:
            seq = EventsRepo(self.store).append(
                conn, job["scope_id"], event_kind, "engine",
                {"job_id": job["job_id"], "input_refs": job["input_refs"]},
                self.policy.policy_version,
            )
            return {"recorded": kind.value, "event_seq": seq}

        with self.store.tx() as conn:
            self._commit_effects(conn, job, owner, kind.value, _apply)

    # ------------------------------------------------------------------
    # lexical projection helpers (inside the caller's tx)
    # ------------------------------------------------------------------

    def _gen_tx(self, conn: Any) -> int:
        """Projection generation read through the caller's write tx —
        a bump earlier in the same tx is visible here, while the
        reader-conn probe returns the last committed value and would
        mis-tag the row against a generation the tx never saw."""
        return int(
            self.store._meta_get(conn, "projection_generation") or 0
        )

    def _index_claim_text_tx(
        self, conn: Any, claim_id: str, scope_id: str, text: str
    ) -> None:
        """Index a claim's head revision under the current generation —
        in-tx variant used by commit paths."""
        if not self.store.fts_enabled:
            return
        head = read_claim_head(conn, claim_id)
        if head is None:
            return
        FtsRepo(self.store).index(
            conn, claim_id, head.revision, scope_id,
            self._gen_tx(conn), text,
        )

    def _index_active_claim_tx(self, conn: Any, claim_id: str) -> None:
        """Index a claim that just landed ACTIVE — caller's transaction.

        Reconstructs the primary evidence span's exact text from persisted
        bytes (the same join retrieval uses) and queues the embed
        obligation for the span set.
        """
        head = read_claim_head(conn, claim_id)
        if head is None or head.state != Lifecycle.ACTIVE:
            return
        row = conn.execute(
            "SELECT ce.span_id"
            " FROM claim_evidence ce"
            " JOIN spans sp ON sp.span_id = ce.span_id"
            " JOIN source_revisions sr"
            "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
            " WHERE ce.claim_id = ? AND ce.revision = ?"
            "   AND ce.evidence_role = 'primary'"
            "   AND sr.payload IS NOT NULL"
            " ORDER BY ce.span_id LIMIT 1",
            (claim_id, head.revision),
        ).fetchone()
        if row is None:
            return
        span_id = row[0]
        # Verified slice: SpansRepo.text re-checks the excerpt_hmac over
        # the stored bytes — a tampered revision fails the transition
        # STORE_CORRUPT rather than indexing forged text into the
        # projection (V4-07.02); absent/purged excerpts index nothing.
        text = self.spans.text(span_id)
        if text is None:
            return
        if self.store.fts_enabled:
            FtsRepo(self.store).index(
                conn, claim_id, head.revision, head.scope_id,
                self._gen_tx(conn), text,
            )
        self._enqueue_embed(conn, head.scope_id, [span_id])
        # V4-14.02: a claim landing ACTIVE after the initial drain re-opens
        # its receipt's deferred stages — the lexical index + embed
        # obligation are owed again, then settled in this same tx.
        if self._readiness is not None:
            for sid, rev in self._sources_for_claim(conn, claim_id):
                self._reopen_and_settle(conn, sid, rev)

    def index_claim(self, claim_id: str, scope: Scope, text: str) -> None:
        """Index a claim's head-revision quotation under the current
        projection generation (SPEC §20). Pending/rejected claims never
        reach this — recall reads only the FTS projection."""
        scope_id = scope_key(scope)
        with self.store.tx() as conn:
            head = ClaimsRepo(self.store).current(claim_id)
            if head is not None:
                FtsRepo(self.store).index(
                    conn, claim_id, head["revision"], scope_id,
                    self._gen_tx(conn), text,
                )
            # V4-14.02: an out-of-band index (e.g. ``remember``) re-opens
            # deferred receipt stages and settles them in this tx.
            if self._readiness is not None:
                for sid, rev in self._sources_for_claim(conn, claim_id):
                    self._reopen_and_settle(conn, sid, rev)

    def index_active_claim(self, claim_id: str) -> None:
        """(Re)index a claim when a transition lands it in ACTIVE.

        Reconstructs the primary evidence span's exact text from persisted
        bytes — the same join retrieval uses — so a pending claim admitted
        through review becomes searchable without re-running harvest."""
        with self.store.tx() as conn:
            self._index_active_claim_tx(conn, claim_id)
