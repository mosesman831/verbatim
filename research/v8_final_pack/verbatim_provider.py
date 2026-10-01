"""VerbatimProvider — AMB (agent-memory-benchmark) MemoryProvider adapter
for the Verbatim memory engine (github.com/mosesman831/verbatim @ 1c86f4f).

Contract notes (see notes_markdown for full citations):
- MemoryProvider ABC (src/memory_bench/memory/base.py): implement sync
  ``ingest(list[Document])`` and sync ``retrieve(query, k, user_id,
  query_timestamp, filters) -> (list[Document], dict|None)``; the base
  wraps both in ``asyncio.to_thread`` — so concurrency-safe means
  THREAD-safe, hence ``threading.Lock`` + ``concurrency = 1``.
- Runner instantiates providers via a zero-arg constructor through the
  hardcoded REGISTRY in memory/__init__.py — no config plumbing: every
  knob below is an env var.
- rag.py joins returned docs as "## Memory {i}\n{doc.content}"; locomo's
  build_rag_prompt json.dumps()es ``_raw_response`` when non-None —
  retrieve() MUST return ``(docs, None)``.
- retrieval.py sends ``k = meta["retrieval_limit"] or 20`` and requires
  ``Document.source_ids`` = the ingested Document.id list for PMB's
  belief-id resolution.
- Verbatim side: ``open_store`` -> ``engine.ingest(SourceEnvelope)`` ->
  ``engine._ingester.run_pending(scope=None)`` (public run_pending pins
  scope=host.default_scope() — private accessor required for per-user
  partitions) -> pending-claim heads -> ``engine.apply_transition`` ->
  ``engine.recall(RecallRequest)`` with ``valid_at_us`` = query_timestamp.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

from memory_bench.memory.base import MemoryProvider
from memory_bench.models import Document

from verbatim.api import open_store
from verbatim.config import config_from_mapping
from verbatim.host import LocalHost
from verbatim.core.time import now_us, parse_rfc3339
from verbatim.core.types import (
    Provenance,
    RecallMode,
    RecallRequest,
    Scope,
    SourceEnvelope,
    SourceKind,
    TransitionCommand,
    Visibility,
)

# ---------------------------------------------------------------------------
# Env knobs (providers get no config plumbing — REGISTRY ctor is zero-arg)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


PROFILE = os.environ.get("AMB_VERB_PROFILE", "amb")
PRINCIPAL = os.environ.get("AMB_VERB_PRINCIPAL", "bench")
CHUNK_CHARS = _env_int("AMB_VERB_CHUNK_CHARS", 1000)   # < harvest max_len=1200
DEADLINE_MS = _env_int("AMB_VERB_DEADLINE_MS", 4000)   # RecallRequest deadline
MAX_BYTES = _env_int("AMB_VERB_MAX_BYTES", 24000)      # RecallRequest cap (max)
TARGET_TOKENS = _env_int("AMB_VERB_TARGET_TOKENS", 4096)
DRAIN_LIMIT = _env_int("AMB_VERB_DRAIN_LIMIT", 4096)   # jobs per drain loop iter
REQUIRE_REVIEW = os.environ.get("AMB_VERB_REQUIRE_REVIEW", "1") != "0"
EMBEDDING = os.environ.get("AMB_VERB_EMBEDDING", "hashing")
MODE = os.environ.get("AMB_VERB_MODE", "offline_rules")

_DOC_MAP_FILE = "doc_map.json"
_CHUNK_SUFFIX_RE = re.compile(r"#c\d{3,}$")


def _cfg() -> Any:
    return config_from_mapping(
        {
            "mode": MODE,
            "capture": {"enabled": True},
            "admission": {"require_review": REQUIRE_REVIEW},
            "embedding": {"backend": EMBEDDING},
        }
    )


def _to_us(ts: Optional[str]) -> Optional[int]:
    """AMB Document.timestamp/query_timestamp (ISO-8601) -> verbatim µs."""
    if not ts:
        return None
    try:
        return parse_rfc3339(ts)
    except Exception:
        return None


def _user_scope(user_id: Optional[str]) -> Scope:
    """Per-user partition: OWNER-visibility scope keyed on principal_id.

    ``can_read`` for OWNER requires same profile + same principal, so a
    recall issued under this scope sees exactly one user's claims —
    AMB's per-unit isolation, expressed as a verbatim partition.
    """
    return Scope(
        profile_id=PROFILE,
        principal_id=user_id or "default",
        visibility=Visibility.OWNER,
    )


def _chunks(text: str, limit: int) -> list[str]:
    """Split a Document's content into <=``limit``-char pieces.

    LoCoMo/LongMemEval documents are ``json.dumps`` of turn lists; each
    turn renders as ``"Speaker: text"`` so the speaker survives inside
    the verbatim claim text. Prose falls back to paragraph packing.
    Pieces stay under harvest's hardcoded ``max_len=1200``.
    """
    text = (text or "").strip()
    if not text:
        return []
    pieces: list[str] = []
    try:
        turns = json.loads(text)
    except Exception:
        turns = None
    if isinstance(turns, list) and turns and all(
        isinstance(t, dict) and ("text" in t or "content" in t) for t in turns
    ):
        units = []
        for t in turns:
            body = (t.get("text") or t.get("content") or "").strip()
            if not body:
                continue
            speaker = (t.get("speaker") or t.get("role") or "").strip()
            units.append(f"{speaker}: {body}" if speaker else body)
    else:
        units = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    cur = ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > limit:
            pieces.append(cur)
            cur = ""
        while len(u) > limit:  # a single over-long unit: hard-split on ws
            cut = u.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append(u[:cut].strip())
            u = u[cut:].strip()
        cur = f"{cur}\n{u}" if cur else u
    if cur:
        pieces.append(cur)
    return pieces


class VerbatimProvider(MemoryProvider):
    """MemoryProvider backed by a local Verbatim store (SQLite)."""

    name = "verbatim"
    description = "Verbatim evidence-first memory engine (local store)"
    kind = "local"
    provider = "verbatim"
    variant = "local"

    # Runner reads this attr for its asyncio.Semaphore. One concurrent
    # retrieve: the read path is thread-safe (per-thread reader conns) but
    # search is GIL-bound sqlite work — parallelism buys nothing.
    concurrency = 1
    supports_filters = False  # AMB tag-group filters have no verbatim lane

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: Any = None
        self._dir: Optional[Path] = None
        # doc.id -> {"user_id","timestamp","context"}
        self._docs: dict[str, dict[str, Any]] = {}
        # verbatim source_id -> doc.id
        self._src2doc: dict[str, str] = {}
        # verbatim scope_id -> principal_id (for the admit pass)
        self._scope_principal: dict[str, str] = {}
        self._extraction_labels: list[dict] = []

    # -- lifecycle -----------------------------------------------------

    def initialize(self) -> None:
        return None

    def set_extraction_labels(self, labels: Optional[list[dict]]) -> None:
        """Optional runner hook (runner.py:148-151, hasattr-guarded).

        PMB supplies entity labels (state/name); verbatim has no entity-
        label concept, so they are stored for inspection only.
        """
        self._extraction_labels = list(labels or [])

    def cleanup(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None

    def prepare(
        self,
        store_dir: Path,
        unit_ids: Optional[set[str]] = None,
        reset: bool = True,
    ) -> None:
        """Runner hook: ``store_dir`` = outputs/<ds>/<run>/_store/<split>/<cat>."""
        del unit_ids
        self._dir = Path(store_dir) / "verbatim"
        if reset and self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._engine is not None:
            self._engine.close()
        self._engine = open_store(
            str(self._dir),
            _cfg(),
            LocalHost(profile_id=PROFILE, principal_id=PRINCIPAL),
            create=True,
        )
        if not reset:
            self._load_map()

    # -- ingest --------------------------------------------------------

    def ingest(self, docs: list[Document]) -> None:
        if not docs:
            return
        with self._lock:
            for doc in docs:
                self._ingest_doc(doc)
            self._drain()
            self._admit_pending()
            self._drain()  # follow-on jobs minted by the admit pass
            self._save_map()

    def _ingest_doc(self, doc: Document) -> None:
        user = doc.user_id or "default"
        scope = _user_scope(doc.user_id)
        from verbatim.core.identity import scope_key

        self._scope_principal[scope_key(scope)] = user
        event_us = _to_us(doc.timestamp) or now_us()
        pieces = _chunks(doc.content, CHUNK_CHARS)
        self._docs[doc.id] = {
            "user_id": user,
            "timestamp": doc.timestamp,
            "context": doc.context,
        }
        if not pieces:
            return
        for i, piece in enumerate(pieces):
            ext = doc.id if len(pieces) == 1 else f"{doc.id}#c{i:03d}"
            env = SourceEnvelope(
                origin="amb",
                source_kind=SourceKind.USER_MESSAGE,
                scope=scope,
                speaker_id=None,
                payload=piece.encode("utf-8"),
                event_us=event_us,
                captured_us=event_us,
                provenance=Provenance.DIRECT_USER,
                external_id=ext,
                revision=1,
                metadata={"doc_id": doc.id},
            )
            receipt = self._engine.ingest(env)
            src_id = receipt.accepted[0] if receipt.accepted else None
            if src_id is None:  # dedup replay: look the source up
                src_id = self._src_by_external(scope, ext)
            if src_id is not None:
                self._src2doc[src_id] = doc.id

    def _src_by_external(self, scope: Scope, external_id: str) -> Optional[str]:
        from verbatim.core.identity import scope_key

        with self._engine.store.read() as conn:
            row = conn.execute(
                "SELECT source_id FROM sources"
                " WHERE scope_id = ? AND origin = 'amb' AND external_id = ?",
                (scope_key(scope), external_id),
            ).fetchone()
        return row[0] if row else None

    def _drain(self) -> None:
        # Public Engine.run_pending/drain_report pin scope to
        # host.default_scope(); per-user job partitions require the
        # cross-scope worker entry point on the private Ingester.
        while self._engine._ingester.run_pending(scope=None, limit=DRAIN_LIMIT):
            pass

    def _claim_heads(self) -> list[tuple[str, int, str, str]]:
        with self._engine.store.read() as conn:
            return conn.execute(
                "SELECT cr.claim_id, cr.revision, cr.state, c.scope_id"
                " FROM claim_revisions cr"
                " JOIN (SELECT claim_id, MAX(revision) mr"
                "       FROM claim_revisions GROUP BY claim_id) h"
                "   ON h.claim_id = cr.claim_id AND h.mr = cr.revision"
                " JOIN claims c ON c.claim_id = cr.claim_id"
            ).fetchall()

    def _admit_pending(self) -> None:
        """Operator admit pass — mirrors eval/v3/baselines.py::_admit_pending.

        require_review=True lands almost every claim PENDING (the admit
        ladder's "abstained" rung); the benchmark acts as operator and
        approves each pending head through the public transition API.
        """
        for cid, rev, state, scope_id in self._claim_heads():
            if state != "pending":
                continue
            principal = self._scope_principal.get(scope_id)
            scope = (
                _user_scope(principal)
                if principal is not None
                else _user_scope(None)
            )
            try:
                self._engine.apply_transition(
                    TransitionCommand(
                        claim_id=cid,
                        expected_revision=rev,
                        effect="admit",
                        actor_id="amb-provider",
                        reason="benchmark operator approval",
                    ),
                    scope=scope,
                )
            except Exception:
                pass  # contention on a head that already moved — harmless

    # -- retrieve ------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: Optional[str] = None,
        query_timestamp: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> tuple[list[Document], Optional[dict]]:
        del filters  # supports_filters=False: runner never passes one
        self._ensure_loaded()
        scope = _user_scope(user_id)
        req = RecallRequest(
            query=query,
            scope=scope,
            mode=RecallMode.CURRENT,
            limit=max(1, min(k, 32)),  # RecallRequest.limit domain is 1..32
            valid_at_us=_to_us(query_timestamp),
            max_bytes=MAX_BYTES,
            deadline_ms=DEADLINE_MS,
            target_tokens=TARGET_TOKENS,
        )
        with self._lock:
            result = self._engine.recall(req)
        out: list[Document] = []
        for item in result.items:
            doc_id = self._src2doc.get(item.span.source_id)
            meta = self._docs.get(doc_id or "", {})
            ctx = meta.get("context") or (doc_id or "verbatim")
            ts = meta.get("timestamp")
            prefix = f"[{ctx}" + (f" — {ts}]" if ts else "]")
            out.append(
                Document(
                    id=doc_id or item.claim_id,
                    content=f"{prefix}\n{item.text}",
                    user_id=user_id,
                    timestamp=ts,
                    context=meta.get("context"),
                    source_ids=[doc_id] if doc_id else [item.span.source_id],
                )
            )
        return out, None  # raw_response=None is REQUIRED for LoCoMo prompts

    # -- doc map persistence (for --skip-ingestion reruns) --------------

    def _map_path(self) -> Optional[Path]:
        return self._dir / _DOC_MAP_FILE if self._dir else None

    def _save_map(self) -> None:
        p = self._map_path()
        if p is None:
            return
        payload = {
            "docs": self._docs,
            "src2doc": self._src2doc,
            "scope_principal": self._scope_principal,
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(p)

    def _load_map(self) -> None:
        p = self._map_path()
        if p is None or not p.exists():
            return
        data = json.loads(p.read_text())
        self._docs.update(data.get("docs") or {})
        self._src2doc.update(data.get("src2doc") or {})
        self._scope_principal.update(data.get("scope_principal") or {})

    def _ensure_loaded(self) -> None:
        """Rehydrate src->doc from the sources table when the map is cold."""
        if self._src2doc or self._engine is None:
            return
        self._load_map()
        if self._src2doc:
            return
        with self._engine.store.read() as conn:
            rows = conn.execute(
                "SELECT source_id, external_id, scope_id"
                " FROM sources WHERE origin = 'amb'"
            ).fetchall()
        for source_id, external_id, scope_id in rows:
            doc_id = _CHUNK_SUFFIX_RE.sub("", external_id or "")
            if doc_id:
                self._src2doc[source_id] = doc_id


# REGISTRY wiring (memory_bench/memory/__init__.py):
#   from .verbatim_provider import VerbatimProvider
#   REGISTRY["verbatim"] = VerbatimProvider
# Then:  amb run --dataset locomo --memory verbatim ...
