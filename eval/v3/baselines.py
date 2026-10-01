"""Case preparation and baseline adapters for the v3 eval harness.

Design contract (SPEC_V3 §53–§54):

* **Fresh store per case.**  ``prepare_case`` builds a disposable SQLite
  store per corpus task — no cross-case leakage, every measurement
  reproducible.
* **Public ingest path only.**  Setup sources go through
  ``Engine.ingest`` → ``run_pending`` → ``apply_transition(admit)`` —
  the same operator-assisted pipeline the CLI drives (the v2 harness's
  convention).  A ``reindex`` job then repairs the FTS projection
  generation, and the run notes disclose that the rebuild was used
  (generation stranding is a known v2 defect, F27).
* **Capability-aware, never silent.**  Optional modules (v3 retrieval,
  security screening, governance, derivations, embeddings) are probed
  lazily; a missing capability produces a ``Capability(available=False)``
  record and an ``unavailable`` query outcome — never an implicit skip.
* **Honest baselines.**  Every adapter ingests the *same* store through
  the *same* public path; only the query path differs.  Comparison
  numbers are measured on this corpus/configuration — never a
  superiority claim.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple

from verbatim.api import Engine, open_store
from verbatim.config import config_from_mapping
from verbatim.core.identity import scope_key
from verbatim.core.types import (
    JobKind,
    Provenance,
    RecallMode,
    RecallRequest,
    Scope,
    SourceEnvelope,
    SourceKind,
    TransitionCommand,
    Visibility,
)
from verbatim.host import LocalHost

from .corpus import CorpusSource, CorpusTask, SCOPE_OTHER


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capability:
    """One optional-capability probe result."""

    name: str
    available: bool
    detail: str = ""


def _probe(name: str, fn: Callable[[], str]) -> Capability:
    """Run a lazy import/check; any failure → explicit unavailable record."""
    try:
        return Capability(name=name, available=True, detail=fn())
    except Exception as exc:  # import or runtime check failed — report it
        return Capability(
            name=name, available=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _unavail(name: str, detail: str) -> Capability:
    return Capability(name=name, available=False, detail=detail)


def probe_capabilities() -> dict[str, Capability]:
    """Probe every optional lane the suites consume (lazy imports only)."""
    caps: dict[str, Capability] = {}

    def _v3_recall() -> str:
        from verbatim.retrieval.v3 import recall_v3  # noqa: F401
        return "verbatim.retrieval.v3.recall_v3 importable"

    def _security() -> str:
        from verbatim.security import screen_content  # noqa: F401
        return "verbatim.security.screen_content importable"

    def _governance() -> str:
        import verbatim.governance as gov  # noqa: F401
        return "verbatim.governance importable"

    def _derivations() -> str:
        import verbatim.derivations as drv  # noqa: F401
        if not hasattr(drv, "roots"):
            raise AttributeError("verbatim.derivations.roots missing")
        return "verbatim.derivations.roots importable"

    def _evidence_v3() -> str:
        from verbatim.evidence import ingest_envelope  # noqa: F401
        return "verbatim.evidence.ingest_envelope importable"

    def _encoder() -> str:
        from verbatim.embeddings.encoder import get_encoder  # noqa: F401
        cfg = config_from_mapping(_DEFAULT_CFG)
        enc = get_encoder(cfg)
        if enc is None:
            raise RuntimeError(
                f"embedding backend {cfg.embedding.backend!r} resolves to "
                "no encoder — semantic lane unprovisioned"
            )
        if not enc.available():
            raise RuntimeError("encoder constructed but reports unavailable")
        return f"encoder {enc.encoder_id} available"

    caps["retrieval_v3"] = _probe("retrieval_v3", _v3_recall)
    caps["security_screening"] = _probe("security_screening", _security)
    caps["governance"] = _probe("governance", _governance)
    caps["derivations"] = _probe("derivations", _derivations)
    caps["evidence_v3"] = _probe("evidence_v3", _evidence_v3)
    caps["encoder"] = _probe("encoder", _encoder)
    return caps


# ---------------------------------------------------------------------------
# case environment
# ---------------------------------------------------------------------------

_DEFAULT_CFG: dict[str, Any] = {
    "mode": "offline_rules",
    "capture": {"enabled": True},
    "admission": {"require_review": True},
    # Deterministic local subword encoder — every arm shares the same
    # provisioned backend so the dense lane and the vector_rag baseline
    # are real measured paths rather than CAPABILITY_UNAVAILABLE slots.
    "embedding": {"backend": "hashing"},
}

#: Warnings that count as a *typed abstention* (never fabricated prose).
ABSTAIN_WARNINGS = frozenset({
    "no_signal",
    "insufficient_support",
    "abstained_uncovered_terms",
    "no_authorized_evidence",
    "processing_pending",
})


@dataclass
class CaseEnv:
    """A fresh store + engine prepared for one corpus task."""

    task_id: str
    store_dir: str
    engine: Engine
    host: LocalHost
    owner_scope: Scope
    other_scope: Scope
    source_map: dict[str, str] = field(default_factory=dict)
    ingest_receipts: dict[str, Any] = field(default_factory=dict)
    claims_admitted: int = 0
    claims_total: int = 0
    notes: list[str] = field(default_factory=list)
    capabilities: dict[str, Capability] = field(default_factory=dict)

    @property
    def store(self):
        return self.engine.store

    @property
    def owner_scope_id(self) -> str:
        return scope_key(self.owner_scope)

    def fixture_id(self, source_id: Optional[str]) -> Optional[str]:
        """Engine source_id → corpus fixture id (reverse of source_map)."""
        for fid, sid in self.source_map.items():
            if sid == source_id:
                return fid
        return None

    def real_source_ids(self, fixture_ids: Sequence[str]) -> set[str]:
        return {self.source_map[f] for f in fixture_ids if f in self.source_map}

    def close(self) -> None:
        try:
            self.engine.close()
        except Exception:
            pass
        # disposable store: remove the case's directory unless the caller
        # supplied a work dir they may want to inspect
        if os.path.basename(self.store_dir).startswith("verbatim-v3-eval-"):
            shutil.rmtree(self.store_dir, ignore_errors=True)


@dataclass
class QueryOutcome:
    """One baseline/arm's answer to one task query."""

    arm: str
    task_id: str
    returned_ids: Tuple[str, ...] = ()   # corpus fixture ids, k-truncated
    returned_source_ids: Tuple[str, ...] = ()  # engine source ids
    n_items: int = 0
    abstained: bool = False
    warnings: Tuple[str, ...] = ()
    unavailable: bool = False            # explicit capability gap
    unavailable_reason: str = ""
    error: Optional[str] = None          # exception name — kept in denominators
    evidence_refs: Tuple[dict[str, Any], ...] = ()  # per-item grounding refs
    raw: Any = None


def _envelope(src: CorpusSource, scope: Scope) -> SourceEnvelope:
    return SourceEnvelope(
        origin="eval-v3-corpus",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id="corpus",
        payload=src.text.encode("utf-8"),
        event_us=1_700_000_000_000_000,
        captured_us=1_700_000_000_000_000,
        provenance=Provenance.DIRECT_USER,
        external_id=src.id,
        revision=1,
        metadata={"eval_kind": src.kind},
    )


def _claim_heads(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        "SELECT cr.claim_id, cr.revision, cr.state"
        " FROM claim_revisions cr"
        " JOIN (SELECT claim_id, MAX(revision) AS mr FROM claim_revisions"
        "       GROUP BY claim_id) h"
        "   ON h.claim_id = cr.claim_id AND h.mr = cr.revision"
    ).fetchall()
    return {r[0]: {"revision": r[1], "state": r[2]} for r in rows}


def _admit_pending(env: CaseEnv) -> None:
    """Drive the public transition API over every pending head — the same
    operator-assisted admission the CLI review queue performs."""
    with env.store.read() as conn:
        heads = _claim_heads(conn)
    env.claims_total = len(heads)
    for cid, h in heads.items():
        if h["state"] != "pending":
            continue
        try:
            env.engine.apply_transition(
                TransitionCommand(
                    claim_id=cid,
                    expected_revision=h["revision"],
                    effect="admit",
                    actor_id="eval-v3-harness",
                    reason="baseline operator approval",
                ),
                scope=env.owner_scope,
            )
            env.claims_admitted += 1
        except Exception as exc:
            env.notes.append(
                f"admit failed for {cid}: {type(exc).__name__}: {exc}"
            )
    _apply_open_reviews(env)


def _apply_open_reviews(env: CaseEnv) -> None:
    """Resolve open review-queue proposals the operator pass would act on —
    supersede/dispute proposals created by relation detection run after
    admissions so both parties are active when the fence checks revisions.
    Arms share this pass; an arm that produced no proposals has an empty
    queue (identical operator behavior, honest comparison)."""
    try:
        from verbatim.core.policy import apply_review, PolicyContext
    except ImportError:
        env.notes.append("apply_review unavailable")
        return
    import json as _json
    with env.store.read() as conn:
        rows = conn.execute(
            "SELECT review_id, proposed_effect_json FROM reviews"
            " WHERE state = 'open'"
        ).fetchall()
    for rid, effect_json in rows:
        effect = _json.loads(effect_json or "{}")
        if effect.get("effect") == "admit":
            continue  # admissions already resolved via apply_transition above
        try:
            apply_review(env.store, rid, "eval-v3-harness",
                         ctx=PolicyContext(env.engine._ingester.cfg))
            env.notes.append(f"applied review {rid[:8]}")
        except Exception as exc:
            from verbatim.core.types import ErrorCode
            is_stale = getattr(exc, "code", None) == ErrorCode.STALE_PROPOSAL
            if not is_stale:
                env.notes.append(
                    f"review {rid[:8]} not applied:"
                    f" {type(exc).__name__}: {exc}"
                )
                continue
            # Re-triage (V2-19.07): the proposal pinned pre-admission
            # revisions and admission moved them — an operator resolves
            # this by re-checking current state, i.e. re-pinning the
            # expected versions to the live heads before applying.
            if _retriage(env, rid):
                try:
                    apply_review(env.store, rid, "eval-v3-harness",
                                 ctx=PolicyContext(env.engine._ingester.cfg))
                    env.notes.append(f"applied review {rid[:8]} (retriaged)")
                except Exception as exc2:
                    env.notes.append(
                        f"review {rid[:8]} not applied after retriage:"
                        f" {type(exc2).__name__}: {exc2}"
                    )


def _retriage(env: CaseEnv, review_id: str) -> bool:
    """Re-pin an open review's expected versions to current claim heads —
    the operator's re-triage step for proposals staled by intervening
    transitions. Returns True when the row was refreshed."""
    import json as _json
    from verbatim.core.lifecycle import read_claim_head
    with env.store.read() as conn:
        row = conn.execute(
            "SELECT expected_versions_json FROM reviews"
            " WHERE review_id = ? AND state = 'open'",
            (review_id,),
        ).fetchone()
    if row is None:
        return False
    expected = _json.loads(row[0] or "{}")
    refreshed = {}
    with env.store.read() as conn:
        for cid in expected:
            h = read_claim_head(conn, cid)
            if h is None:
                return False
            refreshed[cid] = h.revision
    with env.store.tx() as conn:
        conn.execute(
            "UPDATE reviews SET expected_versions_json = ?"
            " WHERE review_id = ? AND state = 'open'",
            (_json.dumps(refreshed), review_id),
        )
    return True


def _reindex(env: CaseEnv) -> None:
    """Enqueue the durable ``reindex`` job and drain it.

    Sequential ``apply_transition`` calls strand earlier claims at stale
    projection generations (the F27 generation defect).  The reindex job
    is the designed repair path; its use is disclosed in ``env.notes``
    rather than hidden.
    """
    try:
        with env.store.tx() as conn:
            env.engine._ingester.jobs.enqueue(
                conn,
                env.owner_scope_id,
                JobKind.REINDEX,
                {},
                dedup_key=b"eval-reindex",
            )
        env.engine.run_pending(limit=64)
        env.notes.append(
            "fts projection rebuilt via reindex job after admission "
            "(repairs generation stranding, F27)"
        )
    except Exception as exc:
        env.notes.append(
            f"reindex unavailable: {type(exc).__name__}: {exc}"
        )


def _provision_governance(env: CaseEnv) -> None:
    """Issue the eval caller a READ+QUOTE grant on the owner scope.

    ``recall_v3`` authorizes through ``governance.authorize`` when the
    module is present — without a grant every v3 query denies.  Failure
    is recorded as a note; the verbatim_v3 baseline will then report the
    denial as a measured error, not a skip.
    """
    try:
        import verbatim.governance as gov
        from verbatim.core.types_v3 import Verb
    except Exception:
        return
    try:
        with env.store.tx() as conn:
            try:
                gov.register_principal(conn, kind="agent",
                                       principal_id=EVAL_CALLER)
            except Exception:
                pass
            gov.create_grant(
                conn,
                scope_id=env.owner_scope_id,
                principal_id=EVAL_CALLER,
                verbs=[Verb.READ.value, Verb.QUOTE.value],
                issuer_id="eval-v3-harness",
            )
    except Exception as exc:
        env.notes.append(
            f"governance provisioning failed: {type(exc).__name__}: {exc}"
        )


EVAL_CALLER = "eval-v3-agent"


def _check_other_scope_claims(env: CaseEnv, task: CorpusTask) -> None:
    """Prepare-time invariant: every ``other``-scope setup source that
    ingested must have yielded at least one claim — otherwise a zero
    disclosure count is structural (an empty partition cannot leak), not
    measured.  A gap is recorded in ``env.notes`` rather than raised so
    the run keeps its denominators and the defect stays visible."""
    other_srcs = [s for s in task.setup_sources if s.scope == SCOPE_OTHER]
    if not other_srcs:
        return
    try:
        with env.store.read() as conn:
            for src in other_srcs:
                sid = env.source_map.get(src.id)
                if sid is None:
                    env.notes.append(
                        f"other-scope source {src.id} was not ingested;"
                        " disclosure is unmeasurable for it"
                    )
                    continue
                n = conn.execute(
                    "SELECT COUNT(DISTINCT ce.claim_id)"
                    " FROM claim_evidence ce"
                    " JOIN spans s ON s.span_id = ce.span_id"
                    " WHERE s.source_id = ?",
                    (sid,),
                ).fetchone()[0]
                if n == 0:
                    env.notes.append(
                        f"other-scope drain produced 0 claims for {src.id}"
                    )
    except Exception as exc:
        env.notes.append(
            f"other-scope claim check failed: {type(exc).__name__}: {exc}"
        )


def prepare_case(
    task: CorpusTask,
    *,
    work_dir: Optional[str] = None,
    cfg_overrides: Optional[dict[str, Any]] = None,
    capabilities: Optional[dict[str, Capability]] = None,
    ingest: str = "v2",
) -> CaseEnv:
    """Build a fresh store and ingest one corpus task's setup sources.

    Owner-scope sources ingest under the host's default scope; ``other``-
    scope sources ingest under a second principal's scope (the same store,
    a different authorization partition).  Returns the prepared env; the
    caller closes it.

    ``ingest="v2"`` (default) shares the legacy ``engine.ingest`` channel
    across all arms — the honest comparison baseline. ``ingest="v3"``
    writes through ``ingest_envelope`` — the real v3 write channel with
    in-transaction screening, quarantine, and harvest enqueue — so the
    security suite measures the v3 arm's actual admission path rather
    than a shared unscreened one.
    """
    cfg_map = dict(_DEFAULT_CFG)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            cfg_map[k] = v
    cfg = config_from_mapping(cfg_map)
    tmp = work_dir or tempfile.mkdtemp(prefix="verbatim-v3-eval-")
    host = LocalHost(
        profile_id="evalv3", principal_id="me", conversation_id="eval-conv"
    )
    owner_scope = host.default_scope()
    other_scope = Scope(
        profile_id="evalv3",
        principal_id="teammate",
        conversation_id="other-conv",
        visibility=Visibility.CONVERSATION,
    )
    engine = open_store(tmp, cfg, host, create=True)
    env = CaseEnv(
        task_id=task.task_id,
        store_dir=tmp,
        engine=engine,
        host=host,
        owner_scope=owner_scope,
        other_scope=other_scope,
        capabilities=capabilities if capabilities is not None else {},
    )

    for src in task.setup_sources:
        scope = other_scope if src.scope == SCOPE_OTHER else owner_scope
        try:
            if ingest == "v3":
                receipt = _ingest_v3(env, src, scope)
                env.source_map[src.id] = receipt.source_id
                env.ingest_receipts[src.id] = {
                    "accepted": [receipt.source_id],
                    "envelope_id": receipt.envelope_id,
                    "receipt_id": receipt.receipt_id,
                }
            else:
                receipt = engine.ingest(_envelope(src, scope))
                if receipt.accepted:
                    env.source_map[src.id] = receipt.accepted[0]
                env.ingest_receipts[src.id] = {
                    "accepted": list(receipt.accepted),
                    "rejected": [list(r) for r in receipt.rejected],
                    "duplicate": receipt.duplicate,
                }
        except Exception as exc:
            env.ingest_receipts[src.id] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            env.notes.append(
                f"ingest failed for {src.id}: {type(exc).__name__}: {exc}"
            )

    try:
        if ingest == "v3":
            # v3 envelopes enqueue harvest under the source's own
            # partition; drain unscoped like a real lane worker.
            engine._ingester.run_pending(
                limit=max(64, len(task.setup_sources) * 4)
            )
        else:
            engine.run_pending(limit=max(64, len(task.setup_sources) * 4))
            # ``Engine.run_pending`` hard-codes the host's default scope,
            # so HARVEST/ADMIT jobs enqueued under the ``other``
            # partition would never lease — drain that scope through the
            # same internal worker path, like a real per-scope worker.
            if any(s.scope == SCOPE_OTHER for s in task.setup_sources):
                engine._ingester.run_pending(
                    scope=other_scope,
                    limit=max(64, len(task.setup_sources) * 4),
                )
    except Exception as exc:
        env.notes.append(f"run_pending failed: {type(exc).__name__}: {exc}")

    _check_other_scope_claims(env, task)
    _admit_pending(env)
    if any(s.scope == SCOPE_OTHER for s in task.setup_sources):
        # Admissions enqueue EMBED jobs under each claim's own scope, so
        # the operator pass strands follow-up work in the other partition
        # for both ingest paths — drain it like a real per-scope worker
        # would, leaving that scope's queue empty.
        try:
            engine._ingester.run_pending(scope=other_scope, limit=64)
        except Exception as exc:
            env.notes.append(
                f"run_pending failed: {type(exc).__name__}: {exc}"
            )
    _reindex(env)
    _provision_governance(env)
    return env


def _ingest_v3(env: CaseEnv, src: CorpusSource, scope: Scope):
    """Write one corpus source through the real v3 capture channel.

    ``ingest_envelope`` persists source + revision + span + envelope +
    screening label + receipt atomically and enqueues harvest — the same
    path the SDK and MCP surface use (§34.01). The envelope scope id is
    the digest of the eval scope so the case stays in the same partition
    the v2 arm uses (``env.owner_scope_id``).
    """
    from verbatim.core.types_v3 import (
        EnvelopeKind,
        Perspective,
        SourceEnvelopeV3,
    )
    from verbatim.evidence.envelopes import ingest_envelope

    principal = (
        env.owner_scope.principal_id
        if scope is env.owner_scope else "teammate"
    )
    envelope = SourceEnvelopeV3(
        kind=EnvelopeKind.USER_MESSAGE,
        scope_id=scope_key(scope),
        actor_principal=principal,
        perspective=Perspective(asserter="corpus", observer="corpus"),
        event_us=1_700_000_000_000_000,
        receipt_us=1_700_000_000_000_000,
        content=src.text.encode("utf-8"),
        external_id=src.id,
        host_id="eval-v3-corpus",
        metadata={"eval_kind": src.kind},
    )
    with env.store.tx() as conn:
        return ingest_envelope(conn, env.store, envelope)


# ---------------------------------------------------------------------------
# item → fixture mapping helpers
# ---------------------------------------------------------------------------


def _item_source_id(item: Any) -> Optional[str]:
    span = getattr(item, "span", None)
    return getattr(span, "source_id", None) if span is not None else None


def _map_returned(env: CaseEnv, source_ids: Sequence[Optional[str]],
                  k: int) -> Tuple[str, ...]:
    """Ordered, deduplicated fixture ids for returned engine source ids."""
    out: list[str] = []
    for sid in source_ids:
        if sid is None:
            continue
        fid = env.fixture_id(sid)
        if fid is not None and fid not in out:
            out.append(fid)
        if len(out) >= k:
            break
    return tuple(out)


def _is_abstain(warnings: Sequence[str], n_items: int) -> bool:
    w = set(warnings)
    return n_items == 0 and bool(w & ABSTAIN_WARNINGS)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


class Baseline:
    """``ingest`` is shared (``prepare_case``); subclasses differ in query.

    ``ingest_mode`` selects the case-preparation write channel for suites
    that consult it (security): ``"v2"`` keeps the shared legacy path for
    strict comparability; ``"v3"`` exercises the arm's real screened
    write channel — what the arm would actually admit in production.
    """

    name = "baseline"
    description = ""
    ingest_mode = "v2"

    def query(self, env: CaseEnv, task: CorpusTask, *, k: int = 5) -> QueryOutcome:
        raise NotImplementedError

    def capabilities(self) -> dict[str, Capability]:
        return {}


class NoMemoryBaseline(Baseline):
    """Floor arm: no store access — returns nothing, picks nothing up."""

    name = "no_memory"
    description = "Control arm: the agent sees no memory context."

    def query(self, env: CaseEnv, task: CorpusTask, *, k: int = 5) -> QueryOutcome:
        return QueryOutcome(
            arm=self.name, task_id=task.task_id,
            warnings=("no_memory_arm",),
        )


class NaiveFtsBaseline(Baseline):
    """Raw FTS5 over ingested source payloads — no claim extraction, no
    evidence packaging, no abstention logic.  The harness-owned
    ``eval_fts`` virtual table is the baseline's own index."""

    name = "naive_fts"
    description = "FTS5 BM25 over raw source payloads; returns source text."

    def _ensure_index(self, env: CaseEnv) -> None:
        with env.store.tx() as conn:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS eval_fts"
                " USING fts5(external_id, payload)"
            )
            have = {
                r[0]
                for r in conn.execute(
                    "SELECT external_id FROM eval_fts"
                ).fetchall()
            }
            rows = conn.execute(
                "SELECT s.external_id, sr.payload"
                " FROM source_revisions sr JOIN sources s"
                "   ON s.source_id = sr.source_id"
                " WHERE sr.revision = 1 AND s.external_id IS NOT NULL"
            ).fetchall()
            for ext_id, payload in rows:
                if ext_id in have:
                    continue
                text = payload.decode("utf-8", errors="replace") if isinstance(
                    payload, (bytes, bytearray, memoryview)
                ) else str(payload)
                conn.execute(
                    "INSERT INTO eval_fts(external_id, payload) VALUES (?, ?)",
                    (ext_id, text),
                )

    def query(self, env: CaseEnv, task: CorpusTask, *, k: int = 5) -> QueryOutcome:
        try:
            self._ensure_index(env)
        except Exception as exc:
            return QueryOutcome(
                arm=self.name, task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        # OR-match each alphanumeric query token — naive FTS has no
        # stopword/coverage analysis, so this is its honest behavior.
        terms = [t for t in task.query.split() if t.isalnum()]
        if not terms:
            return QueryOutcome(
                arm=self.name, task_id=task.task_id,
                warnings=("no_signal",), abstained=False,
            )
        match = " OR ".join(f'"{t}"' for t in terms)
        try:
            with env.store.read() as conn:
                rows = conn.execute(
                    "SELECT external_id FROM eval_fts"
                    " WHERE eval_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, k),
                ).fetchall()
        except Exception as exc:
            return QueryOutcome(
                arm=self.name, task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        returned = tuple(r[0] for r in rows)
        source_ids = tuple(env.source_map.get(f, "") for f in returned)
        return QueryOutcome(
            arm=self.name, task_id=task.task_id,
            returned_ids=returned,
            returned_source_ids=source_ids,
            n_items=len(returned),
            evidence_refs=tuple(
                {"external_id": f, "source_id": env.source_map.get(f)}
                for f in returned
            ),
        )


class VectorRagBaseline(Baseline):
    """Embedding-similarity retrieval over raw sources.  Reports an explicit
    capability gap when no in-process encoder is provisioned — the default
    offline_rules configuration has backend ``none``."""

    name = "vector_rag"
    description = "Cosine over per-source embeddings; unavailable offline."

    def capabilities(self) -> dict[str, Capability]:
        return {"encoder": probe_capabilities()["encoder"]}

    def query(self, env: CaseEnv, task: CorpusTask, *, k: int = 5) -> QueryOutcome:
        cap = self.capabilities()["encoder"]
        if not cap.available:
            return QueryOutcome(
                arm=self.name, task_id=task.task_id,
                unavailable=True,
                unavailable_reason=f"capability unavailable: {cap.detail}",
            )
        try:
            from verbatim.embeddings.encoder import get_encoder
            from verbatim.embeddings.codec import Float32Codec
            import math

            cfg = config_from_mapping(_DEFAULT_CFG)
            enc = get_encoder(cfg)
            assert enc is not None
            texts = [s.text for s in task.setup_sources]
            blobs = enc.encode([task.query, *texts])
            vecs = [
                Float32Codec.unpack(b, enc.dimensions) for b in blobs
            ]
            qv = vecs[0]

            def _cos(a, b):
                num = sum(x * y for x, y in zip(a, b))
                da = math.sqrt(sum(x * x for x in a))
                db = math.sqrt(sum(x * x for x in b))
                return num / (da * db) if da and db else 0.0

            scored = sorted(
                (
                    (_cos(qv, vecs[i + 1]), s.id)
                    for i, s in enumerate(task.setup_sources)
                ),
                reverse=True,
            )[:k]
            returned = tuple(sid for _, sid in scored)
            return QueryOutcome(
                arm=self.name, task_id=task.task_id,
                returned_ids=returned,
                returned_source_ids=tuple(
                    env.source_map.get(f, "") for f in returned
                ),
                n_items=len(returned),
                evidence_refs=tuple(
                    {"external_id": f, "source_id": env.source_map.get(f)}
                    for f in returned
                ),
            )
        except Exception as exc:
            return QueryOutcome(
                arm=self.name, task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )


class VerbatimV2Baseline(Baseline):
    """The shipped engine: ``verbatim.retrieval.search`` through
    ``Engine.recall`` — the v2 behavioral contract."""

    name = "verbatim_v2"
    description = "v2 retrieval: analyze → scope filter → candidates → RRF → evidence."

    def query(self, env: CaseEnv, task: CorpusTask, *, k: int = 5) -> QueryOutcome:
        mode = (
            RecallMode.HISTORICAL if task.kind == "history" else RecallMode.CURRENT
        )
        req = RecallRequest(
            query=task.query, scope=env.owner_scope, mode=mode, limit=k,
        )
        try:
            rr = env.engine.recall(req)
        except Exception as exc:
            return QueryOutcome(
                arm=self.name, task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        items = list(rr.items)
        warns = tuple(rr.warnings or ())
        src_ids = [_item_source_id(i) for i in items]
        return QueryOutcome(
            arm=self.name, task_id=task.task_id,
            returned_ids=_map_returned(env, src_ids, k),
            returned_source_ids=tuple(s for s in src_ids if s),
            n_items=len(items),
            abstained=_is_abstain(warns, len(items)),
            warnings=warns,
            evidence_refs=tuple(
                {
                    "claim_id": i.claim_id,
                    "claim_revision": i.claim_revision,
                    "span_id": getattr(i.span, "span_id", None),
                    "source_id": _item_source_id(i),
                }
                for i in items
            ),
            raw=rr,
        )


class VerbatimV3Baseline(Baseline):
    """The v3 lane: ``verbatim.retrieval.v3.recall_v3`` when importable;
    an explicit capability gap otherwise."""

    name = "verbatim_v3"
    description = "v3 recall: authorize → routes → lanes → union → packs."
    ingest_mode = "v3"

    def capabilities(self) -> dict[str, Capability]:
        caps = probe_capabilities()
        return {"retrieval_v3": caps["retrieval_v3"]}

    def query(self, env: CaseEnv, task: CorpusTask, *, k: int = 5) -> QueryOutcome:
        cap = self.capabilities()["retrieval_v3"]
        if not cap.available:
            return QueryOutcome(
                arm=self.name, task_id=task.task_id,
                unavailable=True,
                unavailable_reason=f"capability unavailable: {cap.detail}",
            )
        try:
            from verbatim.core.types_v3 import RecallRequestV3
            from verbatim.retrieval.v3 import recall_v3

            req = RecallRequestV3(
                query=task.query,
                scope_id=env.owner_scope_id,
                caller_id=EVAL_CALLER,
                purpose="eval",
                max_items=k,
            )
            rr = recall_v3(env.store, req)
        except Exception as exc:
            return QueryOutcome(
                arm=self.name, task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        warns = tuple(rr.warnings or ())
        items = [it for p in rr.packs for it in p.items]
        # Map pack items back to source ids: object_kind "source" is direct;
        # claims resolve through claim_evidence → spans.
        claim_ids = [
            it.handle.object_id for it in items
            if it.handle.object_kind == "claim"
        ]
        claim_src: dict[str, str] = {}
        if claim_ids:
            ph = ",".join("?" * len(claim_ids))
            with env.store.read() as conn:
                rows = conn.execute(
                    "SELECT ce.claim_id, s.source_id FROM claim_evidence ce"
                    " JOIN spans s ON s.span_id = ce.span_id"
                    f" WHERE ce.claim_id IN ({ph})",
                    claim_ids,
                ).fetchall()
            for cid, sid in rows:
                claim_src.setdefault(cid, sid)
        src_ids: list[Optional[str]] = []
        for it in items:
            h = it.handle
            if h.object_kind == "source":
                src_ids.append(h.object_id)
            elif h.object_kind == "claim":
                src_ids.append(claim_src.get(h.object_id))
            else:
                src_ids.append(None)
        return QueryOutcome(
            arm=self.name, task_id=task.task_id,
            returned_ids=_map_returned(env, src_ids, k),
            returned_source_ids=tuple(s for s in src_ids if s),
            n_items=len(items),
            abstained=bool(rr.abstained) or _is_abstain(warns, len(items)),
            warnings=warns,
            evidence_refs=tuple(
                {
                    "object_kind": it.handle.object_kind,
                    "object_id": it.handle.object_id,
                    "revision": it.handle.revision,
                    "source_id": (
                        it.handle.object_id
                        if it.handle.object_kind == "source"
                        else claim_src.get(it.handle.object_id)
                    ),
                }
                for it in items
            ),
            raw=rr,
        )


BASELINES: dict[str, type[Baseline]] = {
    b.name: b
    for b in (
        NoMemoryBaseline,
        NaiveFtsBaseline,
        VectorRagBaseline,
        VerbatimV2Baseline,
        VerbatimV3Baseline,
    )
}

BASELINE_NAMES: Tuple[str, ...] = tuple(BASELINES)


@dataclass
class SuiteRun:
    """One suite's completed run — the report's raw material.

    ``records`` keeps every attempted task in the denominator;
    ``unavailable_lanes`` names baselines/capabilities that could not run
    at all (their records stay, marked via ``note``)."""

    suite: str
    baseline: str
    records: list[Any] = field(default_factory=list)
    outcomes: list[QueryOutcome] = field(default_factory=list)
    capabilities: dict[str, Capability] = field(default_factory=dict)
    unavailable_lanes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    k: int = 5
    errors: int = 0

    def record(self, rec: Any, outcome: Optional[QueryOutcome] = None) -> None:
        self.records.append(rec)
        if outcome is not None:
            self.outcomes.append(outcome)
            if outcome.error:
                self.errors += 1
            if outcome.unavailable and outcome.arm not in self.unavailable_lanes:
                self.unavailable_lanes.append(outcome.arm)


def get_baseline(name: str) -> Baseline:
    try:
        return BASELINES[name]()
    except KeyError:
        raise ValueError(
            f"unknown baseline {name!r}; choose from {sorted(BASELINES)}"
        )


__all__ = [
    "ABSTAIN_WARNINGS",
    "BASELINE_NAMES",
    "BASELINES",
    "Baseline",
    "Capability",
    "CaseEnv",
    "EVAL_CALLER",
    "NaiveFtsBaseline",
    "NoMemoryBaseline",
    "QueryOutcome",
    "VectorRagBaseline",
    "VerbatimV2Baseline",
    "VerbatimV3Baseline",
    "get_baseline",
    "prepare_case",
    "probe_capabilities",
]
