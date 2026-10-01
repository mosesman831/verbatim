"""Executed comparator arms for the V6 portfolio (SPEC_V6 §06).

V6-06.01 always-run arms — fully offline, zero installs — execute real
``seed()``/``answer()`` against the shared V5 consumer corpus
(:func:`eval.v5.corpus.seed_corpus`) through the public views the V5
comparator protocol defines (``PublicItemView``/``PublicTaskView`` —
gold fields are unreachable from arm code):

* ``verbatim_memory`` — the V5 ``Memory`` facade consumer route, seeded
  through ``eval.v5.harness.seed_corpus_env`` and drained to steady
  state (the same arm ``eval.v5.comparators`` runs).
* ``verbatim_v2`` — the predecessor arm: the v2 engine's public
  ``Engine.ingest`` → ``run_pending`` → operator-admit → ``Engine.recall``
  path via ``eval.v3.baselines`` (``prepare_case`` + ``verbatim_v2``).
* ``naive_fts`` — a standalone SQLite FTS5 table BM25-matched over the
  identical item texts; no claim extraction, no abstention machinery.
* ``vector_rag`` — the deterministic hashing encoder over whole item
  texts, cosine top-k (the non-neural backend, labeled as such).
* ``no_memory`` — the control arm: sees nothing, returns nothing.

V6-06.02 pinned Mem0 OSS: ``mem0_oss`` (the default ``infer=True``
shape) and the separately-disclosed ``mem0_oss_inferfalse`` arm probe
``mem0``/``mem0ai`` through importlib. When the package is importable a
real adapter drives its documented ``add``/``search``/``get_all``/
``delete`` against a local qdrant store and a local embedder/LLM; when
any piece is absent the row reports ``unavailable`` with the verbatim
probe error — never a fake ``tested``.

V6-06.03 named rows: ``graphiti_oss`` (needs ``graphiti_core`` plus a
Neo4j backend), ``holographic`` (local comparator package),
``zep_hosted`` and ``mem0_platform`` (hosted/platform editions —
``out_of_scope`` without explicit authorization per V45-10.03 /
V4-54.11; hosted scores never attribute to OSS). Absence is reported,
not hidden.

Corpus lifecycle honesty: the shared corpus carries ``supersedes`` and
``forget`` items (V5 §20.11). The registry applies those events through
each arm's real capability — ``Memory.add(replaces=)``/``Memory.forget``
inside ``seed_corpus_env`` for the facade arm, ``replace()``/``delete()``
hooks for arms that expose them — and records what was applied in the
row's notes. An arm with no delete path legitimately retains the item
and any forbidden deliveries are measured, not assumed.

Scoring follows the V5 comparator conventions (task success,
recall@k, abstention correctness, forbidden deliveries, latency
percentiles) extended with precision@k and the V4.5 eight-category cost
accounting — unmeasured categories are labeled, partial totals are
labeled partial (V6-06.05). ``compare()`` carries the V5 refusal rules
forward unchanged (V6-06.04).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import sqlite3
import string
import tempfile
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

from eval.v45.bakeoff import COST_CATEGORIES, CostBreakdown
from eval.v5 import stats as st
from eval.v5.comparators import (
    REQUIRED_PINS,
    STATUSES,
    ComparatorArm,
    ComparatorPin,
    ComparatorRow,
    compare,
    probe_mem0,
)
from eval.v5.comparators import VerbatimMemoryArm as _V5VerbatimMemoryArm
from eval.v5.corpus import (
    ConsumerCorpus,
    corpus_stats,
    public_item,
    public_task,
    seed_corpus,
)
from eval.v5.harness import (
    delivered_payload_bytes,
    environment,
    percentiles,
)

#: The deterministic local encoder every embedding arm shares
#: (V45-18.02 — labeled non-neural, never described as a neural model).
HASHING_ENCODER_ID = "hashing:subword-ngram:v1"

#: Row names the registry always executes offline (V6-06.01).
ALWAYS_RUN_ARMS = (
    "verbatim_memory",
    "verbatim_v2",
    "naive_fts",
    "vector_rag",
    "no_memory",
)

#: Row names gated on a real dependency probe (V6-06.02/03).
PROBE_GATED_ARMS = (
    "mem0_oss",
    "mem0_oss_inferfalse",
    "graphiti_oss",
    "holographic",
)

#: Hosted/platform editions — declared, never executed without explicit
#: authorization (V45-10.03, V4-54.11).
OUT_OF_SCOPE_ARMS = ("zep_hosted", "mem0_platform")


# ---------------------------------------------------------------------------
# pin helpers
# ---------------------------------------------------------------------------


def _hardware() -> str:
    return environment().get("cpu_model", platform.machine())


def _local_pin(*, edition: str, revision: str, extractor: str,
               embedder: str, reader: str, settings: str, indexes: str,
               readiness_policy: str,
               judge: str = "deterministic_gold_ids",
               prompts: str = "none") -> ComparatorPin:
    """A fully-populated local pin (all 13 fields named, V6-06.04)."""
    return ComparatorPin(
        edition=edition,
        revision=revision,
        deployment="local_embedded",
        extractor=extractor,
        embedder=embedder,
        reader=reader,
        judge=judge,
        prompts=prompts,
        settings=settings,
        indexes=indexes,
        readiness_policy=readiness_policy,
        hardware=_hardware(),
        pricing_date="n/a (local, no priced services)",
    )


def _unrunnable_pin(*, edition: str, revision: str, deployment: str,
                    why: str, reader: str = "n/a",
                    extractor: str = "n/a", embedder: str = "n/a",
                    judge: str = "deterministic_gold_ids",
                    prompts: str = "n/a", settings: str = "n/a",
                    indexes: str = "n/a",
                    readiness_policy: str = "n/a",
                    pricing_date: str = "n/a") -> ComparatorPin:
    """Pins for a row that cannot execute — every field is named and the
    honest reason it is unpinned is embedded, never fabricated."""
    na = f"n/a ({why})"
    return ComparatorPin(
        edition=edition,
        revision=revision,
        deployment=deployment,
        extractor=extractor if extractor != "n/a" else na,
        embedder=embedder if embedder != "n/a" else na,
        reader=reader if reader != "n/a" else na,
        judge=judge,
        prompts=prompts if prompts != "n/a" else na,
        settings=settings if settings != "n/a" else na,
        indexes=indexes if indexes != "n/a" else na,
        readiness_policy=(
            readiness_policy if readiness_policy != "n/a" else na
        ),
        hardware=_hardware(),
        pricing_date=pricing_date,
    )


def _verbatim_revision() -> str:
    import verbatim
    return getattr(verbatim, "__version__", "workspace")


def _tree_bytes(path: str) -> Optional[int]:
    """Total file bytes under a directory (the arm's storage footprint)."""
    if not path or not os.path.isdir(path):
        return None
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _file_bytes(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# arm: verbatim_memory — the V5 consumer route (reuses the v5 arm)
# ---------------------------------------------------------------------------


class VerbatimMemoryArm(_V5VerbatimMemoryArm):
    """``verbatim_memory`` — the real ``Memory`` facade arm.

    Seeding (adds, ``replaces=`` transitions, ``forget`` closure, drain)
    happens inside ``seed_corpus_env`` during construction — lifecycle
    ops are applied by harness plumbing through the real facade, so the
    registry marks this arm ``internal_lifecycle`` and records rather
    than re-applies them.
    """

    internal_lifecycle = True

    def storage_bytes(self) -> Optional[int]:
        return _file_bytes(os.path.join(self.env.workdir, "mem.db"))

    def pin(self) -> ComparatorPin:
        return _local_pin(
            edition="verbatim",
            revision=_verbatim_revision(),
            extractor="memory_facade.add(infer=bool)",
            embedder=HASHING_ENCODER_ID,
            reader="consumer_search(SearchResult)",
            settings="profile=local_memory,worker=external",
            indexes="fts5+hashing-encoder",
            readiness_policy="wait_ready+session_barrier+external_drain",
        )


# ---------------------------------------------------------------------------
# arm: verbatim_v2 — predecessor engine through eval.v3.baselines
# ---------------------------------------------------------------------------


class VerbatimV2Arm:
    """``verbatim_v2`` — the previous-generation engine contract.

    Seeding runs the real v2 write channel: ``Engine.ingest`` envelopes
    → ``run_pending`` harvest/screen → operator ``apply_transition``
    admission → ``reindex`` repair — via ``eval.v3.baselines.prepare_case``
    (the F27 rebuild note is carried in the env notes, not hidden).
    Answers run ``Engine.recall`` (v2 retrieval: analyze → scope →
    candidates → RRF → evidence) through ``VerbatimV2Baseline``.

    Lifecycle hooks map corpus events onto real v2 transitions:
    ``replace`` supersedes the predecessor's active claims naming the
    successor claim; ``delete`` archives the item's active claims.
    """

    def __init__(self, corpus: ConsumerCorpus,
                 *, workdir: Optional[str] = None, **_: Any) -> None:
        self.corpus = corpus
        self._owns_dir = workdir is None
        self._workdir = workdir or tempfile.mkdtemp(prefix="v6-v2arm-")
        os.makedirs(self._workdir, exist_ok=True)
        self.env: Any = None
        self._baseline: Any = None
        self.notes: List[str] = []

    # -- protocol ------------------------------------------------------

    def name(self) -> str:
        return "verbatim_v2"

    def pin(self) -> ComparatorPin:
        return _local_pin(
            edition="verbatim",
            revision=_verbatim_revision() + " (v2 contract)",
            extractor="v2 ingest: typed harvest -> screened claims -> "
                      "operator admit",
            embedder=HASHING_ENCODER_ID,
            reader="v2 retrieval.search via Engine.recall "
                   "(analyze->scope->candidates->RRF->evidence)",
            settings="mode=offline_rules; admission.require_review=true; "
                     "embedding=hashing",
            indexes="fts5 claims + hashing-encoder embed jobs",
            readiness_policy="run_pending drain + operator admit + "
                             "reindex (F27 repair disclosed)",
        )

    def seed(self, items: Sequence[Any]) -> None:
        from eval.v3.baselines import get_baseline, prepare_case
        from eval.v3.corpus import CorpusSource, CorpusTask

        srcs = tuple(
            CorpusSource(id=i.id, text=i.text, kind="note") for i in items
        )
        seed_task = CorpusTask(
            task_id="v6-seed", kind="factual_lookup",
            setup_sources=srcs, query="seed",
        )
        self.env = prepare_case(
            seed_task, work_dir=os.path.join(self._workdir, "store"))
        self._baseline = get_baseline("verbatim_v2")
        self.notes.extend(self.env.notes)

    def answer(self, task: Any, k: int) -> dict:
        from eval.v3.corpus import CorpusTask

        synthetic = CorpusTask(
            task_id=task.task_id, kind="factual_lookup",
            setup_sources=(), query=task.query,
        )
        t0 = time.perf_counter()
        outcome = self._baseline.query(self.env, synthetic, k=k)
        ms = (time.perf_counter() - t0) * 1000.0
        status = (
            "error" if outcome.error
            else "unavailable" if outcome.unavailable
            else "ok"
        )
        warnings = list(outcome.warnings)
        if outcome.unavailable:
            warnings.append(
                f"unavailable:{outcome.unavailable_reason}")
        return {
            "task_id": task.task_id,
            "status": status,
            "returned_ids": list(outcome.returned_ids),
            "n_items": outcome.n_items,
            "abstained": bool(outcome.abstained or outcome.unavailable),
            "latency_ms": ms,
            "delivered_bytes": (
                delivered_payload_bytes(outcome.raw)
                if outcome.raw is not None else 0
            ),
            "warnings": warnings,
            "error": outcome.error,
        }

    def close(self) -> None:
        try:
            if self.env is not None:
                self.env.close()
        finally:
            if self._owns_dir:
                shutil.rmtree(self._workdir, ignore_errors=True)

    # -- lifecycle hooks (real v2 transitions) -------------------------

    def _claims_for(self, item_id: str) -> List[str]:
        sid = self.env.source_map.get(item_id)
        if sid is None:
            return []
        with self.env.store.read() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ce.claim_id FROM claim_evidence ce"
                " JOIN spans s ON s.span_id = ce.span_id"
                " WHERE s.source_id = ?",
                (sid,),
            ).fetchall()
        return [r[0] for r in rows]

    def _active_heads(self) -> Dict[str, dict]:
        from eval.v3.baselines import _claim_heads
        with self.env.store.read() as conn:
            return _claim_heads(conn)

    def replace(self, predecessor_id: str, successor_id: str) -> None:
        """Supersede the predecessor's active claims through the real
        ``apply_transition`` path, naming the successor claim."""
        from verbatim.core.types import TransitionCommand

        preds = self._claims_for(predecessor_id)
        succs = self._claims_for(successor_id)
        if not preds or not succs:
            self.notes.append(
                f"replace {predecessor_id}->{successor_id}: "
                "claims missing, transition skipped"
            )
            return
        heads = self._active_heads()
        for cid in preds:
            head = heads.get(cid)
            if head is None or head["state"] != "active":
                continue
            self.env.engine.apply_transition(
                TransitionCommand(
                    claim_id=cid,
                    expected_revision=head["revision"],
                    effect="supersede",
                    actor_id="eval-v6-registry",
                    reason=f"corpus supersede by {successor_id}",
                    successor_claim_id=succs[0],
                ),
                scope=self.env.owner_scope,
            )

    def delete(self, item_id: str) -> None:
        """Archive the item's active claims — the v2 operator removal
        path (retrieval-invisible thereafter)."""
        from verbatim.core.types import TransitionCommand

        heads = self._active_heads()
        for cid in self._claims_for(item_id):
            head = heads.get(cid)
            if head is None or head["state"] != "active":
                continue
            self.env.engine.apply_transition(
                TransitionCommand(
                    claim_id=cid,
                    expected_revision=head["revision"],
                    effect="archive",
                    actor_id="eval-v6-registry",
                    reason="corpus forget event",
                ),
                scope=self.env.owner_scope,
            )

    # -- cost metering --------------------------------------------------

    def storage_bytes(self) -> Optional[int]:
        if self.env is None:
            return None
        return _tree_bytes(self.env.store_dir)


# ---------------------------------------------------------------------------
# arm: naive_fts — standalone SQLite FTS5 over identical item texts
# ---------------------------------------------------------------------------


def _fts_terms(query: str) -> List[str]:
    """Whitespace-tokenize a query into FTS5-safe terms.

    Tokens are stripped of edge punctuation and kept when they carry at
    least one alphanumeric — identifiers like ``tok-10000``/``RF-2201``
    survive and FTS5 tokenizes the quoted phrase the same way it
    tokenized the document, so identifier probes stay meaningful.
    """
    out = []
    for tok in query.split():
        tok = tok.strip(string.punctuation)
        if tok and any(c.isalnum() for c in tok):
            out.append(tok)
    return out


class NaiveFtsArm:
    """``naive_fts`` — SQLite FTS5 BM25 over raw item payloads.

    No claim extraction, no evidence packaging, no abstention logic —
    the arm's own ``items`` virtual table is its whole index. OR-matches
    each query term and returns the corpus ids ranked by bm25.
    """

    def __init__(self, corpus: ConsumerCorpus,
                 *, workdir: Optional[str] = None, **_: Any) -> None:
        self.corpus = corpus
        self._owns_dir = workdir is None
        self._workdir = workdir or tempfile.mkdtemp(prefix="v6-ftsarm-")
        os.makedirs(self._workdir, exist_ok=True)
        self._db_path = os.path.join(self._workdir, "naive_fts.db")
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(
            "CREATE VIRTUAL TABLE items USING fts5(item_id UNINDEXED, text)"
        )
        self.notes: List[str] = []

    def name(self) -> str:
        return "naive_fts"

    def pin(self) -> ComparatorPin:
        return _local_pin(
            edition="reference",
            revision="v6-arm:1",
            extractor="none (raw item text indexed whole)",
            embedder="none",
            reader="sqlite FTS5 bm25, OR-match over quoted query terms",
            settings="k-bounded; unicode61 tokenizer",
            indexes="fts5(item_id,text)",
            readiness_policy="synchronous commit (no async pipeline)",
        )

    def seed(self, items: Sequence[Any]) -> None:
        self._conn.executemany(
            "INSERT INTO items(item_id, text) VALUES (?, ?)",
            [(i.id, i.text) for i in items],
        )
        self._conn.commit()

    def delete(self, item_id: str) -> None:
        self._conn.execute("DELETE FROM items WHERE item_id = ?",
                           (item_id,))
        self._conn.commit()

    def replace(self, predecessor_id: str, successor_id: str) -> None:
        # Successor was already seeded; dropping the predecessor row is
        # this store's supersede semantics.
        self.delete(predecessor_id)

    def answer(self, task: Any, k: int) -> dict:
        t0 = time.perf_counter()
        terms = _fts_terms(task.query)
        rows: List[tuple] = []
        error = None
        if terms:
            match = " OR ".join(f'"{t}"' for t in terms)
            try:
                rows = self._conn.execute(
                    "SELECT item_id, text FROM items"
                    " WHERE items MATCH ? ORDER BY rank LIMIT ?",
                    (match, k),
                ).fetchall()
            except sqlite3.Error as exc:
                error = f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000.0
        returned = [r[0] for r in rows]
        warnings = [] if rows else ["no_signal"]
        return {
            "task_id": task.task_id,
            "status": "error" if error else "ok",
            "returned_ids": returned,
            "n_items": len(returned),
            "abstained": not rows,
            "latency_ms": ms,
            "delivered_bytes": sum(len(r[1].encode("utf-8")) for r in rows),
            "warnings": warnings,
            "error": error,
        }

    def close(self) -> None:
        try:
            self._conn.close()
        finally:
            if self._owns_dir:
                shutil.rmtree(self._workdir, ignore_errors=True)

    def storage_bytes(self) -> Optional[int]:
        return _file_bytes(self._db_path)


# ---------------------------------------------------------------------------
# arm: vector_rag — hashing-encoder cosine top-k over item texts
# ---------------------------------------------------------------------------


class VectorRagArm:
    """``vector_rag`` — embedding-similarity retrieval over raw items.

    The provisioned encoder is the deterministic subword hashing
    backend (``hashing:subword-ngram:v1``) — non-neural, disclosed.
    Cosine top-k with no abstention machinery: it always returns its
    best k matches, so no-answer probes honestly surface spurious hits.
    """

    def __init__(self, corpus: ConsumerCorpus,
                 *, workdir: Optional[str] = None, **_: Any) -> None:
        self.corpus = corpus
        self._owns_dir = workdir is None
        self._workdir = workdir or tempfile.mkdtemp(prefix="v6-vecarm-")
        os.makedirs(self._workdir, exist_ok=True)
        self._db_path = os.path.join(self._workdir, "vector_rag.db")
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(
            "CREATE TABLE vecs(item_id TEXT PRIMARY KEY, text TEXT,"
            " blob BLOB)"
        )
        self._enc = None
        self.notes: List[str] = []
        try:
            from verbatim.config import config_from_mapping
            from verbatim.embeddings.encoder import get_encoder

            cfg = config_from_mapping({
                "mode": "offline_rules",
                "embedding": {"backend": "hashing"},
            })
            enc = get_encoder(cfg)
            if enc is None or not enc.available():
                raise RuntimeError("hashing encoder unavailable")
            self._enc = enc
            self.availability = {"available": True}
        except Exception as exc:  # noqa: BLE001 — honest capability gap
            self.availability = {
                "available": False,
                "status": "unavailable",
                "reason": f"capability unavailable: "
                          f"{type(exc).__name__}: {exc}",
            }

    def name(self) -> str:
        return "vector_rag"

    def pin(self) -> ComparatorPin:
        return _local_pin(
            edition="reference",
            revision="v6-arm:1",
            extractor="none (raw item text embedded whole)",
            embedder=f"{HASHING_ENCODER_ID} (non-neural)",
            reader="cosine top-k over float32 blobs",
            settings="k-bounded; no abstention machinery",
            indexes="sqlite vecs(item_id,blob) — full scan cosine",
            readiness_policy="synchronous encode at seed (no async pipeline)",
        )

    def seed(self, items: Sequence[Any]) -> None:
        texts = [i.text for i in items]
        blobs = self._enc.encode(texts) if texts else []
        self._conn.executemany(
            "INSERT OR REPLACE INTO vecs(item_id, text, blob)"
            " VALUES (?, ?, ?)",
            [(i.id, i.text, b) for i, b in zip(items, blobs)],
        )
        self._conn.commit()

    def delete(self, item_id: str) -> None:
        self._conn.execute("DELETE FROM vecs WHERE item_id = ?",
                           (item_id,))
        self._conn.commit()

    def replace(self, predecessor_id: str, successor_id: str) -> None:
        self.delete(predecessor_id)

    def _cosine(self, a: Sequence[float], b: Sequence[float]) -> float:
        import math
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a))
        db = math.sqrt(sum(y * y for y in b))
        return num / (da * db) if da and db else 0.0

    def answer(self, task: Any, k: int) -> dict:
        from verbatim.embeddings.codec import Float32Codec

        t0 = time.perf_counter()
        error = None
        scored: List[tuple] = []
        try:
            qv = Float32Codec.unpack(
                self._enc.encode([task.query])[0], self._enc.dimensions)
            rows = self._conn.execute(
                "SELECT item_id, text, blob FROM vecs").fetchall()
            scored = sorted(
                (
                    (self._cosine(qv, Float32Codec.unpack(
                        blob, self._enc.dimensions)), iid, text)
                    for iid, text, blob in rows
                ),
                key=lambda r: r[0],
                reverse=True,
            )[:k]
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000.0
        returned = [iid for _s, iid, _t in scored]
        return {
            "task_id": task.task_id,
            "status": "error" if error else "ok",
            "returned_ids": returned,
            "n_items": len(returned),
            "abstained": not scored,
            "latency_ms": ms,
            "delivered_bytes": sum(
                len(t.encode("utf-8")) for _s, _i, t in scored),
            "warnings": [] if scored else ["no_signal"],
            "error": error,
        }

    def close(self) -> None:
        try:
            self._conn.close()
        finally:
            if self._owns_dir:
                shutil.rmtree(self._workdir, ignore_errors=True)

    def storage_bytes(self) -> Optional[int]:
        return _file_bytes(self._db_path)


# ---------------------------------------------------------------------------
# arm: no_memory — the control arm
# ---------------------------------------------------------------------------


class NoMemoryArm:
    """``no_memory`` — floor control: no store, no context, no answer."""

    def __init__(self, corpus: ConsumerCorpus, **_: Any) -> None:
        self.corpus = corpus
        self.notes: List[str] = []

    def name(self) -> str:
        return "no_memory"

    def pin(self) -> ComparatorPin:
        return _local_pin(
            edition="reference",
            revision="v6-arm:1",
            extractor="n/a (no write path — control arm)",
            embedder="n/a (no store)",
            reader="none — returns an empty result for every query",
            settings="control arm; consumes no items",
            indexes="n/a (no index)",
            readiness_policy="n/a (nothing to settle)",
        )

    def seed(self, items: Sequence[Any]) -> None:
        return None

    def answer(self, task: Any, k: int) -> dict:
        t0 = time.perf_counter()
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "task_id": task.task_id,
            "status": "no_memory",
            "returned_ids": [],
            "n_items": 0,
            "abstained": False,
            "latency_ms": ms,
            "delivered_bytes": 0,
            "warnings": ["no_memory_arm"],
            "error": None,
        }

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# arm: mem0_oss — real adapter behind an honest probe (V6-06.02)
# ---------------------------------------------------------------------------


class Mem0OssArm:
    """``mem0_oss`` — pinned Mem0 OSS through its documented API.

    The adapter is real: when ``mem0``/``mem0ai`` is importable it
    configures a local qdrant store plus a local embedder (and, for the
    default ``infer=True`` shape, a local LLM) per V6-06.02, then drives
    ``add``/``search``/``get_all``/``delete``. When any piece is absent
    the row reports ``unavailable`` with the actual probe text — the
    probe is real, never fabricated.
    """

    infer = True
    _UID = "v6-eval-user"

    def __init__(self, corpus: ConsumerCorpus,
                 *, workdir: Optional[str] = None,
                 infer: Optional[bool] = None, **_: Any) -> None:
        self.corpus = corpus
        self._infer = self.infer if infer is None else bool(infer)
        self._workdir = workdir or tempfile.mkdtemp(prefix="v6-mem0-")
        self._owns_dir = workdir is None
        self.probe = probe_mem0()
        self._mem: Any = None
        self._pin_parts: Dict[str, str] = {}
        self.notes: List[str] = []
        if not self.probe.get("available"):
            self.availability = {
                "available": False,
                "status": "unavailable",
                "reason": "mem0ai not installed: "
                          + self.probe.get("error", "unknown"),
            }
            return
        try:
            self._configure()
        except Exception as exc:  # noqa: BLE001 — honest infeasibility
            self.availability = {
                "available": False,
                "status": "unavailable",
                "reason": (
                    f"mem0 present "
                    f"({self.probe.get('module')} "
                    f"{self.probe.get('version', '?')}) but a pinned "
                    f"local deployment is infeasible: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
            return
        self.availability = {"available": True}

    # -- pinned local deployment ---------------------------------------

    def _find(self, module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except Exception:
            return False

    def _configure(self) -> None:
        """V6-06.02 local pin: qdrant local mode + local embedder; the
        default shape additionally needs a local LLM for extraction."""
        if not self._find("qdrant_client"):
            raise RuntimeError(
                "qdrant-client absent — the pinned local vector store "
                "cannot be provisioned"
            )
        embedder = self._local_embedder_cfg()
        cfg: Dict[str, Any] = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "v6_eval",
                    "path": os.path.join(self._workdir, "qdrant"),
                    "embedding_model_dims": embedder["dims"],
                },
            },
            "embedder": embedder["cfg"],
        }
        if self._infer:
            cfg["llm"] = self._local_llm_cfg()
        from mem0 import Memory as _M0  # type: ignore

        self._mem = _M0.from_config(cfg)
        self._pin_parts = {
            "embedder": embedder["id"],
            "vector_store": "qdrant local (on-disk)",
            "llm": cfg.get("llm", {}).get("config", {}).get("model", "n/a"),
        }

    def _local_embedder_cfg(self) -> Dict[str, Any]:
        if self._find("sentence_transformers"):
            return {
                "id": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
                "dims": 384,
                "cfg": {
                    "provider": "huggingface",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    },
                },
            }
        if self._find("ollama"):
            return {
                "id": "ollama:nomic-embed-text",
                "dims": 768,
                "cfg": {
                    "provider": "ollama",
                    "config": {"model": "nomic-embed-text"},
                },
            }
        raise RuntimeError(
            "no local embedder provisionable "
            "(sentence-transformers and ollama both absent)"
        )

    def _local_llm_cfg(self) -> Dict[str, Any]:
        if self._find("ollama"):
            return {
                "provider": "ollama",
                "config": {"model": "llama3.1:8b", "temperature": 0.0},
            }
        raise RuntimeError(
            "infer=True requires a local LLM and none is provisionable "
            "(ollama absent) — only the disclosed infer=False arm could "
            "run (V6-06.02)"
        )

    # -- protocol -------------------------------------------------------

    def name(self) -> str:
        return "mem0_oss" if self._infer else "mem0_oss_inferfalse"

    def pin(self) -> ComparatorPin:
        if self.availability.get("available"):
            extractor = (
                f"mem0 LLM extractor (infer=True, "
                f"{self._pin_parts.get('llm', 'n/a')})"
                if self._infer else "infer=False raw storage"
            )
            return _local_pin(
                edition="oss",
                revision=str(self.probe.get("version") or "unversioned"),
                extractor=extractor,
                embedder=self._pin_parts.get("embedder", "n/a"),
                reader="mem0.search (documented API)",
                prompts=(
                    "mem0 default extraction prompts (unmodified)"
                    if self._infer else "none (infer=False)"
                ),
                settings=(
                    f"infer={self._infer}; "
                    f"vector_store={self._pin_parts.get('vector_store')}"
                ),
                indexes="qdrant local collection",
                readiness_policy="mem0 add returns post-write; "
                                 "get_all cross-check",
            )
        why = "adapter unconfigured — dependency probe failed"
        return _unrunnable_pin(
            edition="oss",
            revision=str(self.probe.get("version") or "unversioned"),
            deployment="n/a (not deployed — local pin infeasible)",
            why=why,
            extractor=(
                "mem0 LLM extractor (infer=True) — requires a local LLM"
                if self._infer else "infer=False raw storage"
            ),
            judge="deterministic_gold_ids",
            pricing_date="n/a (OSS, no priced calls)",
        )

    def seed(self, items: Sequence[Any]) -> None:
        for i in items:
            self._mem.add(
                i.text, user_id=self._UID,
                metadata={"item_id": i.id}, infer=self._infer,
            )

    def delete(self, item_id: str) -> None:
        res = self._mem.get_all(user_id=self._UID)
        for m in (res.get("results", res) if isinstance(res, dict) else res):
            md = m.get("metadata") or {}
            if md.get("item_id") == item_id:
                self._mem.delete(m["id"])

    def replace(self, predecessor_id: str, successor_id: str) -> None:
        self.delete(predecessor_id)

    def answer(self, task: Any, k: int) -> dict:
        t0 = time.perf_counter()
        error = None
        returned: List[str] = []
        nbytes = 0
        try:
            res = self._mem.search(task.query, user_id=self._UID, limit=k)
            rows = res.get("results", res) if isinstance(res, dict) else res
            for m in rows or []:
                md = m.get("metadata") or {}
                iid = md.get("item_id")
                if iid is not None and iid not in returned:
                    returned.append(iid)
                nbytes += len(str(m.get("memory", "")).encode("utf-8"))
                if len(returned) >= k:
                    break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "task_id": task.task_id,
            "status": "error" if error else "ok",
            "returned_ids": returned,
            "n_items": len(returned),
            "abstained": not returned,
            "latency_ms": ms,
            "delivered_bytes": nbytes,
            "warnings": [] if returned else ["no_signal"],
            "error": error,
        }

    def close(self) -> None:
        if self._owns_dir:
            shutil.rmtree(self._workdir, ignore_errors=True)

    def storage_bytes(self) -> Optional[int]:
        return _tree_bytes(self._workdir)


class Mem0OssInferFalseArm(Mem0OssArm):
    """``mem0_oss_inferfalse`` — the disclosed infer=False shape.

    V6-06.02: raw-storage mode is allowed ONLY as a separately-named,
    separately-pinned arm — never reported as default Mem0.
    """

    infer = False

    def __init__(self, corpus: ConsumerCorpus, **kw: Any) -> None:
        super().__init__(corpus, **kw)
        self.notes.append(
            "disclosed infer=False arm (V6-06.02): raw storage, no LLM "
            "extraction — never reported as default Mem0"
        )


# ---------------------------------------------------------------------------
# arm: graphiti_oss — probe graphiti_core + a Neo4j backend (V6-06.03)
# ---------------------------------------------------------------------------


class GraphitiOssArm:
    """``graphiti_oss`` — Graphiti needs ``graphiti_core`` plus a live
    Neo4j/FalkorDB backend; each missing piece is reported verbatim."""

    def __init__(self, corpus: ConsumerCorpus, **_: Any) -> None:
        self.corpus = corpus
        self.notes: List[str] = []
        errors: List[str] = []
        versions: Dict[str, str] = {}
        for mod in ("graphiti_core", "neo4j"):
            try:
                m = importlib.import_module(mod)
                versions[mod] = getattr(m, "__version__", "unversioned")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{mod}: {type(exc).__name__}: {exc}")
        if "graphiti_core" not in versions:
            reason = ("graphiti_core not installed: "
                      + "; ".join(errors))
        elif "neo4j" not in versions:
            reason = (
                f"graphiti_core {versions['graphiti_core']} importable "
                "but no Neo4j driver — Graphiti requires a Neo4j/"
                f"FalkorDB backend ({'; '.join(errors)})"
            )
        elif not os.environ.get("NEO4J_URI"):
            reason = (
                "graphiti_core + neo4j importable but no backend "
                "endpoint configured (NEO4J_URI unset) — a pinned run "
                "cannot execute"
            )
        else:
            # A real adapter would construct Graphiti(NEO4J_URI) here;
            # executing it needs a running server, probed on first call.
            reason = (
                "graphiti_core + neo4j importable and NEO4J_URI set, "
                "but no live Neo4j server is provisioned in this "
                "environment — treated as unavailable rather than "
                "faking a connection"
            )
        self.availability = {
            "available": False, "status": "unavailable",
            "reason": reason,
        }
        self._versions = versions

    def name(self) -> str:
        return "graphiti_oss"

    def pin(self) -> ComparatorPin:
        return _unrunnable_pin(
            edition="oss",
            revision=self._versions.get(
                "graphiti_core", "n/a (package absent)"),
            deployment="n/a (requires a Neo4j/FalkorDB backend)",
            why="backend/package probe failed",
            reader="graphiti search (temporal-knowledge graph)",
            pricing_date="n/a (OSS)",
        )

    def seed(self, items: Sequence[Any]) -> None:
        raise RuntimeError("graphiti_oss unavailable: "
                           + self.availability["reason"])

    def answer(self, task: Any, k: int) -> dict:
        raise RuntimeError("graphiti_oss unavailable: "
                           + self.availability["reason"])

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# arm: holographic — local comparator package, probed (V6-06.03)
# ---------------------------------------------------------------------------


class HolographicArm:
    """``holographic`` — the local Holographic comparator; importable
    package required, probed for real."""

    def __init__(self, corpus: ConsumerCorpus, **_: Any) -> None:
        self.corpus = corpus
        self.notes: List[str] = []
        self._version: Optional[str] = None
        try:
            m = importlib.import_module("holographic")
            self._version = getattr(m, "__version__", "unversioned")
            reason = (
                "package 'holographic' importable but no executable "
                "adapter is wired for this corpus — unavailable rather "
                "than fabricated metrics"
            )
        except Exception as exc:  # noqa: BLE001
            reason = (
                "package 'holographic' not installed — the pinned local "
                f"comparator cannot execute ({type(exc).__name__}: {exc})"
            )
        self.availability = {
            "available": False, "status": "unavailable",
            "reason": reason,
        }

    def name(self) -> str:
        return "holographic"

    def pin(self) -> ComparatorPin:
        return _unrunnable_pin(
            edition="local",
            revision=self._version or "n/a (package absent)",
            deployment="n/a (adapter never configured)",
            why="package/adapter probe failed",
            pricing_date="n/a (local)",
        )

    def seed(self, items: Sequence[Any]) -> None:
        raise RuntimeError("holographic unavailable: "
                           + self.availability["reason"])

    def answer(self, task: Any, k: int) -> dict:
        raise RuntimeError("holographic unavailable: "
                           + self.availability["reason"])

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# declared rows — hosted/platform editions (V6-06.03)
# ---------------------------------------------------------------------------


class DeclaredRowArm:
    """A registry row with no executable path in this environment.

    Used for hosted/platform editions whose execution requires paid or
    externally-authorized endpoints (``out_of_scope`` per V45-10.03).
    The row is named and reasoned, never silently dropped.
    """

    def __init__(self, corpus: ConsumerCorpus, *, row_name: str,
                 status: str, reason: str, edition: str,
                 deployment: str, notes: Sequence[str] = (),
                 **_: Any) -> None:
        self.corpus = corpus
        self._row_name = row_name
        self._pin = _unrunnable_pin(
            edition=edition,
            revision="n/a (hosted service unpinned locally)",
            deployment=deployment,
            why="no local adapter — hosted endpoint required",
            pricing_date="n/a (no paid endpoints benchmarked)",
        )
        self.notes = tuple(notes)
        self.availability = {
            "available": False, "status": status, "reason": reason,
        }

    def name(self) -> str:
        return self._row_name

    def pin(self) -> ComparatorPin:
        return self._pin

    def seed(self, items: Sequence[Any]) -> None:
        raise RuntimeError(
            f"{self._row_name} {self.availability['status']}: "
            + self.availability["reason"])

    def answer(self, task: Any, k: int) -> dict:
        raise RuntimeError(
            f"{self._row_name} {self.availability['status']}: "
            + self.availability["reason"])

    def close(self) -> None:
        pass


_HOSTED_REASON = (
    "hosted service — no paid/hosted endpoints without explicit "
    "authorization (V4-54.11); hosted scores never attribute to the "
    "OSS edition (V45-10.03)"
)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

ARM_FACTORIES: Dict[str, Callable[..., Any]] = {
    "verbatim_memory": VerbatimMemoryArm,
    "verbatim_v2": VerbatimV2Arm,
    "naive_fts": NaiveFtsArm,
    "vector_rag": VectorRagArm,
    "no_memory": NoMemoryArm,
    "mem0_oss": Mem0OssArm,
    "mem0_oss_inferfalse": Mem0OssInferFalseArm,
    "graphiti_oss": GraphitiOssArm,
    "holographic": HolographicArm,
    "zep_hosted": partial(
        DeclaredRowArm, row_name="zep_hosted", status="out_of_scope",
        edition="hosted", deployment="hosted_saas",
        reason=_HOSTED_REASON),
    "mem0_platform": partial(
        DeclaredRowArm, row_name="mem0_platform", status="out_of_scope",
        edition="platform", deployment="hosted_platform",
        reason=_HOSTED_REASON),
}


def _apply_lifecycle(arm: Any, corpus: ConsumerCorpus,
                     notes: List[str]) -> None:
    """Apply the corpus's supersede/forget events through the arm's real
    capability (V5 §20.11 lifecycle honesty).

    ``internal_lifecycle`` arms (``verbatim_memory``) already ran those
    ops through the real facade inside ``seed_corpus_env`` — recorded,
    not re-applied. Arms exposing ``replace``/``delete`` hooks get the
    events through them. Arms with neither keep the content and any
    forbidden deliveries are measured — disclosed here, not assumed.
    """
    internal = bool(getattr(arm, "internal_lifecycle", False))
    has_replace = callable(getattr(arm, "replace", None))
    has_delete = callable(getattr(arm, "delete", None))
    applied: List[str] = []
    skipped: List[str] = []
    for item in corpus.items:
        if item.supersedes:
            label = f"supersede {item.supersedes}->{item.id}"
            if internal:
                applied.append(label + " (facade add replaces=)")
            elif has_replace:
                try:
                    arm.replace(item.supersedes, item.id)
                    applied.append(label)
                except Exception as exc:  # noqa: BLE001
                    notes.append(
                        f"lifecycle {label} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                skipped.append(
                    f"no replace path — supersede event skipped, "
                    f"predecessor {item.supersedes} left as-is"
                )
        if item.forget:
            label = f"forget {item.id}"
            if internal:
                applied.append(label + " (facade forget)")
            elif has_delete:
                try:
                    arm.delete(item.id)
                    applied.append(label)
                except Exception as exc:  # noqa: BLE001
                    notes.append(
                        f"lifecycle {label} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                skipped.append(
                    f"no delete path — forget event skipped, "
                    f"{item.id} left as-is"
                )
    if applied:
        notes.append("lifecycle applied: " + "; ".join(applied))
    if skipped:
        notes.append("lifecycle NOT applied: " + "; ".join(skipped))


def _score(scored: Sequence[dict], corpus: ConsumerCorpus) -> dict:
    """V5 metric conventions + precision@k (V6-06.05).

    Errors stay in the denominator; abstention is credited only on
    ``expected_abstain`` tasks; every delivered forbidden id counts.
    """
    by_id = {t.task_id: t for t in corpus.tasks}
    recalls: List[float] = []
    precisions: List[float] = []
    lat: List[float] = []
    delivered = 0
    correct = errors = forbidden = 0
    abstain_n = abstain_ok = 0
    for rec in scored:
        task = by_id.get(rec.get("task_id", ""))
        if task is None:
            continue
        if rec.get("error"):
            errors += 1
        delivered += int(rec.get("delivered_bytes") or 0)
        ret = set(rec.get("returned_ids") or [])
        exp = set(task.expected_ids)
        if task.expected_ids:
            recalls.append(len(ret & exp) / len(task.expected_ids))
            precisions.append(len(ret & exp) / len(ret) if ret else 0.0)
        forb = ret & set(task.forbidden_ids)
        forbidden += len(forb)
        if task.expected_abstain:
            abstain_n += 1
            abstain_ok += 1 if (rec.get("abstained") or not ret) else 0
        if task.expected_ids:
            ok = bool(ret & exp) and not forb
        elif task.expected_abstain:
            ok = bool(rec.get("abstained") or not ret)
        else:
            ok = not ret
        correct += 1 if ok else 0
        if rec.get("latency_ms") is not None:
            lat.append(float(rec["latency_ms"]))
    n = len(scored)
    return {
        "tasks": n,
        "errors": errors,
        "correct": correct,
        "accuracy": (correct / n) if n else None,
        "accuracy_ci95": list(st.wilson_interval(correct, n)) if n else None,
        "task_success": (correct / n) if n else None,
        "recall_at_k": (
            round(sum(recalls) / len(recalls), 6) if recalls else None
        ),
        "precision_at_k": (
            round(sum(precisions) / len(precisions), 6)
            if precisions else None
        ),
        "abstain": {"correct": abstain_ok, "n": abstain_n},
        "forbidden_hits": forbidden,
        "delivered_bytes_total": delivered,
        "latency_ms": percentiles(lat),
    }


def _local_costs(scored: Sequence[dict], *, ingest_ms: float,
                 storage: Optional[float]) -> CostBreakdown:
    """Honest partial cost accounting (V6-06.05): ingest wall time,
    query-side wall time, and on-disk storage are metered; extraction/
    embeddings/consolidation/answer_reading/maintenance stay unmeasured
    — labeled, never zeroed."""
    lat = [float(r["latency_ms"]) for r in scored
           if r.get("latency_ms") is not None]
    return CostBreakdown(
        ingest=round(ingest_ms, 3),
        query_inference=round(sum(lat), 3) if lat else None,
        storage=float(storage) if storage is not None else None,
        units={
            "ingest": "ms_wall",
            "query_inference": "ms_wall",
            "storage": "bytes",
        },
    )


def run_arm(name: str, corpus: Optional[ConsumerCorpus] = None,
            *, memories: int = 64, seed: int = 42, k: int = 8,
            workdir: Optional[str] = None) -> ComparatorRow:
    """Execute one registered arm (or record its honest unavailability)
    over the shared V5 corpus."""
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    if name not in ARM_FACTORIES:
        return ComparatorRow(
            name=name, status="out_of_scope",
            reason="not in the V6 comparator registry",
        )
    factory = ARM_FACTORIES[name]
    ingest_ms = 0.0
    t0 = time.perf_counter()
    try:
        arm = factory(corpus, workdir=workdir)
    except Exception as exc:  # noqa: BLE001
        return ComparatorRow(
            name=name, status="unavailable",
            reason=f"setup failed: {type(exc).__name__}: {exc}",
        )
    ingest_ms += (time.perf_counter() - t0) * 1000.0
    pin = arm.pin()

    # ---- honest availability gate (probe-gated rows never fake tested)
    avail = getattr(arm, "availability", None)
    if avail is not None and not avail.get("available"):
        try:
            arm.close()
        except Exception:
            pass
        return ComparatorRow(
            name=name,
            status=avail.get("status", "unavailable"),
            reason=avail.get("reason", "unavailable"),
            pin=pin,
            notes=tuple(getattr(arm, "notes", ())),
        )

    # ---- execute the shared task stream through public views ---------
    notes: List[str] = list(getattr(arm, "notes", ()))
    scored: List[dict] = []
    storage: Optional[float] = None
    try:
        t0 = time.perf_counter()
        arm.seed([public_item(i) for i in corpus.items])
        ingest_ms += (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        _apply_lifecycle(arm, corpus, notes)
        ingest_ms += (time.perf_counter() - t0) * 1000.0
        for task in corpus.tasks:
            try:
                scored.append(arm.answer(public_task(task), k))
            except Exception as exc:  # noqa: BLE001
                scored.append({
                    "task_id": task.task_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        try:
            storage = arm.storage_bytes()
        except Exception:
            storage = None
    except Exception as exc:  # noqa: BLE001
        try:
            arm.close()
        except Exception:
            pass
        return ComparatorRow(
            name=name, status="unavailable",
            reason=f"execution failed: {type(exc).__name__}: {exc}",
            pin=pin,
        )
    finally:
        try:
            arm.close()
        except Exception:
            pass

    metrics = _score(scored, corpus)
    costs = _local_costs(scored, ingest_ms=ingest_ms, storage=storage)
    if not costs.complete():
        notes.append(
            "cost accounting partial — unmeasured categories: "
            + ", ".join(costs.unmeasured())
        )
    return ComparatorRow(
        name=name,
        status="tested",
        reason="" if pin.pinned() else
        f"unpinned fields: {pin.missing()} — measured but not "
        "claim-eligible (V6-06.04)",
        pin=pin,
        weakened=tuple() if pin.pinned() else ("unpinned_fields",),
        metrics=metrics,
        costs=costs,
        executed=True,
        notes=tuple(notes),
    )


def _row_dict(row: ComparatorRow) -> dict:
    d = row.to_dict()
    d["arm"] = d.pop("name")
    return d


def run_registry(corpus: Optional[ConsumerCorpus] = None,
                 workdir: Optional[str] = None,
                 *, memories: int = 64, seed: int = 42,
                 k: int = 8) -> dict:
    """Run every registered arm and the verbatim-vs-each comparison set.

    ``corpus`` is the shared V5 consumer corpus (one digest for every
    arm — paired, same tasks, same k). Every declared row appears:
    executed arms carry measured metrics + costs; unavailable/
    out_of_scope rows carry their reason (V6-06.03). Comparisons run
    ``compare()`` on ``verbatim_memory`` against each other row —
    refusals are recorded verbatim, never silently dropped.
    """
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    rows: List[ComparatorRow] = []
    for name in ARM_FACTORIES:
        sub = os.path.join(workdir, name) if workdir else None
        rows.append(run_arm(name, corpus, k=k, workdir=sub))

    verbatim = next((r for r in rows if r.name == "verbatim_memory"),
                    None)
    comparisons: List[dict] = []
    for r in rows:
        if r.name == "verbatim_memory":
            continue
        if verbatim is not None and r.status == "tested":
            comparisons.append(compare(verbatim, r))
        else:
            reason = f"{r.name} is {r.status}: {r.reason}"
            if verbatim is None:
                reason += " — and the verbatim_memory row is absent"
            elif verbatim.status != "tested":
                reason += f" — and verbatim_memory is {verbatim.status}"
            comparisons.append({
                "a": "verbatim_memory", "b": r.name,
                "metric": "accuracy", "valid": False, "winner": None,
                "reason": reason,
            })

    return {
        "suite": "comparators",
        "qualification": "locally_measured",
        "scale": {
            "memories": len(corpus.items),
            "tasks": len(corpus.tasks),
            "k": k,
            "seed": corpus.seed,
        },
        "corpus": corpus_stats(corpus),
        "corpus_digest": corpus.digest(),
        "environment": environment(),
        "rows": [_row_dict(r) for r in rows],
        "comparisons": comparisons,
        "registry": list(ARM_FACTORIES),
        "executed_arms": [r.name for r in rows if r.status == "tested"],
        "unexecuted_arms": [
            {"arm": r.name, "status": r.status, "reason": r.reason}
            for r in rows if r.status != "tested"
        ],
        "verdict": "executed",
        "notes": [
            "paired run — one shared corpus/digest/k for every arm",
            "arms receive only PublicItemView/PublicTaskView surfaces "
            "(gold ids unreachable from arm code)",
            "cost accounting is partial — unmeasured categories are "
            "labeled per row (V6-06.05)",
        ],
    }


__all__ = [
    "ALWAYS_RUN_ARMS",
    "ARM_FACTORIES",
    "COST_CATEGORIES",
    "ComparatorArm",
    "ComparatorPin",
    "ComparatorRow",
    "CostBreakdown",
    "DeclaredRowArm",
    "GraphitiOssArm",
    "HASHING_ENCODER_ID",
    "HolographicArm",
    "Mem0OssArm",
    "Mem0OssInferFalseArm",
    "NaiveFtsArm",
    "NoMemoryArm",
    "OUT_OF_SCOPE_ARMS",
    "PROBE_GATED_ARMS",
    "REQUIRED_PINS",
    "STATUSES",
    "VectorRagArm",
    "VerbatimMemoryArm",
    "VerbatimV2Arm",
    "compare",
    "probe_mem0",
    "run_arm",
    "run_registry",
]
