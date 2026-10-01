# V8-16/W6 integration notes — AMB provider contract + VerbatimProvider prototype

Verified against `agent-memory-benchmark` @ `/tmp/amb/agent-memory-benchmark` (commit 03c1d0f, github.com/vectorize-io/agent-memory-benchmark) and `verbatim` @ `/tmp/verbatim` (HEAD 1c86f4f). All citations are file:line in those trees.

## 1. Provider contract — verified spec + corrections to "established facts"

### 1.1 The ABC (all verified)

`src/memory_bench/memory/base.py` — `MemoryProvider(ABC)`:

- **Abstract methods are SYNC** `ingest(documents)` (base.py:37-40) and `retrieve(query, k=10, user_id=None, query_timestamp=None, filters=None) -> tuple[list[Document], dict|None]` (base.py:46-52).
- `async_ingest`/`async_retrieve` are concrete wrappers using `asyncio.to_thread` (base.py:42-44, 54-61). **`async_retrieve` forwards `filters` only when `self.supports_filters` is truthy** — otherwise it calls `self.retrieve(query, k, user_id, query_timestamp)` positionally (base.py:59-61). A provider that declares `supports_filters=False` never sees a `filters` argument.
- `concurrency: int = 4` is a **class attribute** (base.py:16, comment: "override to 1 for non-thread-safe providers"), not a config — the runner reads `getattr(memory, "concurrency", _CONCURRENCY)` into an `asyncio.Semaphore` (runner.py:13, 338-339, 403-404).
- Optional hooks: `initialize()` (base.py:21-23; called runner.py:84), `cleanup()` (base.py:25-26), `prepare(store_dir: Path, unit_ids: set[str]|None, reset=True)` (base.py:28-35; called runner.py:153 with `reset=not skip_ingestion`), `retrieve_by_steps` (base.py:63-65; default delegates to retrieve), `direct_answer` (base.py:71-75; only for "agent" mode providers).
- `set_extraction_labels(labels)` is **not** in the ABC — the runner calls it only `if labels and hasattr(memory, "set_extraction_labels")` (runner.py:148-151). Optional, implemented in the prototype as a stored no-op.
- Metadata class attrs consumed elsewhere: `name`, `kind` ("local"/"cloud"), `provider`, `variant`, `link`, `logo` (base.py:9-15).

**Correction to the parent's premise:** the async surface is derived, not abstract — implement `retrieve`/`ingest` sync and you get the async for free, wrapped in a **thread** (so `threading.Lock`, not `asyncio.Lock`, is the right serialization — an `asyncio.Lock` created in `__init__` binds to the wrong/no event loop when `retrieve` runs in a `to_thread` worker).

### 1.2 Registration — correction

Providers are **not** registered via entry points or config files. `src/memory_bench/memory/__init__.py` imports every provider and builds a hardcoded `REGISTRY: dict[str, type[MemoryProvider]]` (memory/__init__.py:15-40); `get_memory_provider(name)` instantiates the class **zero-arg** (memory/__init__.py:42-56 approx). The CLI selects with `amb run --memory/-m NAME` (cli.py:37). To wire verbatim: add `from .verbatim_provider import VerbatimProvider` + `REGISTRY["verbatim"] = VerbatimProvider` in that file. **All provider config must come from env vars or `prepare(store_dir,...)`** — there is no per-provider config plumbing.

### 1.3 Document / call shapes

- `Document` dataclass (models.py:4-13): `id`, `content` (plain text), `user_id` (per-user namespace), `messages` (structured turns — unused by verbatim), `timestamp` (ISO-8601 str), `context` (provenance hint), `source_ids` (**retrieval-side**: ids of ingested docs this memory came from), `tags` (**ingest-side**: dataset-supplied labels for filtering).
- RAG mode (modes/rag.py:44-54): `retrieval_query = meta.get("retrieval_query") or query`; `async_retrieve(retrieval_query, user_id=user_id, query_timestamp=query_timestamp)` — **k is NOT passed → provider default 10**. Context = `"\n\n".join(f"## Memory {i+1}\n{doc.content}")`. `retrieve_ms` wraps only the retrieve call (rag.py:45-50).
- Retrieval mode (modes/retrieval.py:31-62): `k = int(meta.get("retrieval_limit") or 20)`; `filters = meta.get("retrieval_filter") if supports_filters else None`; blank query → 0 docs without calling the provider; returned docs land in `raw_response["documents"]` (retrieval.py:56-60) → `dataset.score_retrieval(q, docs)` (runner.py:222-227); context renders `{i+1}. [{d.id}]{← source_ids}\n{d.content}` (retrieval.py:51-54).
- Timing: `ingestion_ms` wraps `async_ingest`/`ingest` only (runner.py:349-352, 395-399); **`retrieve_time_ms` covers only `async_retrieve`** — ingest is never counted in recall latency.
- `count_tokens` = tiktoken `cl100k_base`, `encode(text, disallowed_special=())` (utils.py:8-14).

### 1.4 LoCoMo (dataset/locomo.py)

- doc `id = {sample_id}_{session_key}`; `content = json.dumps(turns)`; `user_id = sample_id`; `timestamp` = session ISO date; `context = "Conversation between {a} and {b} ({session_key} of {sample_id})"` (locomo.py:281-331).
- `Query.user_id = sample_id`; `query_timestamp` = last session date ISO-8601 UTC (locomo.py:205-279).
- `task_type="open"`, `isolation_unit="conversation"`, category-5 adversarial queries skipped.
- **`build_rag_prompt` does `ctx = json.dumps(meta["_raw_response"]) if meta.get("_raw_response") else context`** (locomo.py:136-163) — a provider returning a raw payload explodes the prompt (~36K tokens on Hindsight). **Verbatim MUST return `(docs, None)`.**

### 1.5 LongMemEval (dataset/longmemeval.py)

- `task_type="open"`, `isolation_unit="question"` (longmemeval.py:67-68) — **each question is its own memory unit**: `Query.user_id = question_id` (line 296), `query_timestamp` = the question's date ISO (lines 280, 299).
- doc `id = {question_id}_{session_id}`; `content = json.dumps(cleaned_turns)` (line 346); `context = "Session {doc_id} - you are the assistant in this conversation - happened on {date} UTC"`.
- `build_rag_prompt` also json.dumps's `_raw_response` when non-None (lines 130-132) → same raw_response=None requirement.
- Per-category judge prompts via `get_judge_prompt_fn` (abstention-aware categories exist — a provider returning nothing/abstaining is scored differently per category).

## 2. Verbatim engine — exact call signatures used (prototype-verified)

```
open_store(data_dir, cfg, host, *, create=False, judge=None, encoder=None, deps=None, store_path=None, prefer_store=None) -> Engine        # api.py:337-390
engine.ingest(envelope: SourceEnvelope, *, caller=None) -> IngestReceipt     # api_ingest.py:70
engine._ingester.run_pending(scope=None, limit=N) -> int                     # ingest.py:1020-1046 (private attr — see 2.4)
engine.apply_transition(TransitionCommand(claim_id, expected_revision, effect="admit", actor_id, reason), scope=scope)  # api_review.py
engine.recall(request: RecallRequest, *, caller=None) -> RecallResult        # api_recall.py:21-31 → retrieval.search(store, request)
```

Dataclasses used (core/types.py): `SourceEnvelope(origin, source_kind, scope, speaker_id, payload:bytes, event_us, captured_us, timezone, provenance, external_id, source_id, revision, metadata)` (types.py:433); `Scope(profile_id, principal_id, workspace_id=None, conversation_id=None, visibility)` (types.py:413); `RecallRequest(query, scope, mode=CURRENT, limit∈1..32, valid_at_us, max_bytes∈512..24000, include_sources, memory_kinds, deadline_ms∈1..10000, target_tokens)` (types.py:709); `RecallResult.items: tuple[EvidenceItem]`; `EvidenceItem(claim_id, claim_revision, text, span:SpanRef(source_id,revision,start_byte,end_byte), speaker_id, lifecycle, ...)` (types.py:762-788).

Time: `verbatim.core.time.parse_rfc3339(iso) -> µs` (time.py:32).

### 2.1 API mismatches found vs the task's premises

1. **There is no `engine.search`** — the recall API is `engine.recall(RecallRequest)` → `retrieval.search(store, request)` (api_recall.py:21-31). `as_of` maps to `RecallRequest.valid_at_us` (µs).
2. **`engine.run_pending()`/`engine.drain_report` pin `scope=host.default_scope()`** (api_ingest.py:149-153, api.py:311-321): jobs enqueued under per-user scopes are never leased through the public API. The prototype uses `engine._ingester.run_pending(scope=None, ...)` — `scope=None` = cross-scope drain, the documented "trusted standalone workers" path (queue.py:466-480). **Build-agent action:** either accept the private accessor (stable at 1c86f4f) or add a public `run_pending(scope=None)` passthrough in verbatim.
3. **`require_review=False` does NOT activate claims** — the admission ladder's "abstained" rung still lands PENDING for ordinary statements (core/policy.py:586-620, 838-865). An operator admit pass is required: pending heads from `claim_revisions` → `engine.apply_transition(effect="admit")` — exactly the `eval/v3/baselines.py::_admit_pending` pattern (baselines.py:244-269). `admit` needs GrantKind.RESOLVE, satisfied by `caller=None` → operator (api_governance.py:49-56; `_EFFECT_GRANTS` api_review.py).
4. **Harvest bounds are hardcoded**: `_do_harvest` → `harvest_source(envelope)` with defaults `max_candidates=32, min_len=32, max_len=1200` chars (ingest.py:240-265 → core/harvest.py:519-525, 646-676). Over-`max_len` pieces are **skipped, not truncated** (harvest.py:537-539). A multi-KB LoCoMo session JSON would yield **zero claims** — the provider MUST chunk documents at ingest (prototype: 1000-char pieces, one `SourceEnvelope` each, `external_id={doc.id}#cNNN`).
5. **Thread-safety is better than reported**: `Store` keeps one writer conn serialized by `threading.Lock` (store.py:308, 929-991) and **per-thread reader conns** via `threading.local` (store.py:330, 838-857). Concurrent `recall` from `to_thread` workers is connection-safe; the provider still sets `concurrency=1` (search is GIL-bound sqlite work — parallelism buys nothing and churns reader conns).
6. **`caller=None` → operator** (`_resolve_caller` → `grants=frozenset(GrantKind), is_operator=True`, api_governance.py:49-56) — no grant provisioning needed for ingest/recall/admit.
7. `valid_at_us` filters claim-revision eligibility by valid-time (retrieval/candidates.py:545-546, 1481-1482); `known_at_seq` is the event-seq as-of variant (candidates.py:553-562) — use `valid_at_us` for AMB `query_timestamp`.

### 2.2 User isolation mapping

`Scope(profile_id="amb", principal_id=<user_id>, visibility=OWNER)` per AMB user_id. `can_read` for OWNER requires same profile + same principal (core/identity.py:44-75) — a recall issued under the user's scope sees exactly that user's claims. Verified live: user2 recall sees 10 items of its own corpus and 0 of user1's.

### 2.3 Dedup / idempotence

`sources` dedup key is partition-local `(origin, external_id)` (schema.py:44-58; repos.py:240-300): re-ingesting the same external_id returns `receipt.duplicate=True` and reuses the source — safe for `--skip-ingested`/resume paths.

### 2.4 Recall behavior notes

- Abstention warnings land in `result.warnings` (`"no_signal"`, `"abstained_uncovered_terms"`, `"no_authorized_evidence"`, `"time_unknown"`; retrieval/__init__.py:133-225). Empty results are honest benchmark behavior — do not treat as errors.
- `deadline_ms` is cooperative (DEADLINE_EXCEEDED when exceeded) — default 4000 in the prototype for cold caches.
- `engine.recall` requires GrantKind.READ_EVIDENCE — covered by operator caller.

## 3. Working provider — /tmp/verbatim_provider.py

Wiring: `from .verbatim_provider import VerbatimProvider` + `REGISTRY["verbatim"] = VerbatimProvider` in `src/memory_bench/memory/__init__.py`, then `amb run --dataset locomo --memory verbatim`.

```python
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
DRAIN_LIMIT = _env_int("AMB_VERB_DRAIN_LIMIT", 4096)
REQUIRE_REVIEW = os.environ.get("AMB_VERB_REQUIRE_REVIEW", "1") != "0"
EMBEDDING = os.environ.get("AMB_VERB_EMBEDDING", "hashing")
MODE = os.environ.get("AMB_VERB_MODE", "offline_rules")

_DOC_MAP_FILE = "doc_map.json"
_CHUNK_SUFFIX_RE = re.compile(r"#c\d{3,}$")


def _cfg() -> Any:
    return config_from_mapping({
        "mode": MODE,
        "capture": {"enabled": True},
        "admission": {"require_review": REQUIRE_REVIEW},
        "embedding": {"backend": EMBEDDING},
    })


def _to_us(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        return parse_rfc3339(ts)
    except Exception:
        return None


def _user_scope(user_id: Optional[str]) -> Scope:
    """OWNER-visibility scope keyed on principal_id: one principal per AMB
    user_id, so a recall sees exactly one user's claims."""
    return Scope(profile_id=PROFILE, principal_id=user_id or "default",
                 visibility=Visibility.OWNER)


def _chunks(text: str, limit: int) -> list[str]:
    """Split a Document's content into <=``limit``-char pieces — under
    harvest's hardcoded max_len=1200. LoCoMo/LME JSON turn lists render as
    ``"Speaker: text"`` lines so speaker survives inside the claim text."""
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
        while len(u) > limit:
            cut = u.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append(u[:cut].strip())
            u = u[cut:].strip()
        cur = f"{cur}\n{u}" if cur else u
    if cur:
        pieces.append(cur)
    return pieces


class VerbatimProvider(MemoryProvider):
    name = "verbatim"
    description = "Verbatim evidence-first memory engine (local store)"
    kind = "local"
    provider = "verbatim"
    variant = "local"
    concurrency = 1          # thread-safe enough, but GIL-bound search gains nothing
    supports_filters = False # AMB tag-group filters have no verbatim lane

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: Any = None
        self._dir: Optional[Path] = None
        self._docs: dict[str, dict[str, Any]] = {}      # doc.id -> meta
        self._src2doc: dict[str, str] = {}              # source_id -> doc.id
        self._scope_principal: dict[str, str] = {}      # scope_id -> principal
        self._extraction_labels: list[dict] = []

    def initialize(self) -> None:
        return None

    def set_extraction_labels(self, labels: Optional[list[dict]]) -> None:
        self._extraction_labels = list(labels or [])  # verbatim: no entity labels

    def cleanup(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None

    def prepare(self, store_dir: Path, unit_ids: Optional[set[str]] = None,
                reset: bool = True) -> None:
        del unit_ids
        self._dir = Path(store_dir) / "verbatim"
        if reset and self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._engine is not None:
            self._engine.close()
        self._engine = open_store(
            str(self._dir), _cfg(),
            LocalHost(profile_id=PROFILE, principal_id=PRINCIPAL), create=True)
        if not reset:
            self._load_map()

    def ingest(self, docs: list[Document]) -> None:
        if not docs:
            return
        with self._lock:
            for doc in docs:
                self._ingest_doc(doc)
            self._drain()
            self._admit_pending()
            self._drain()          # follow-on jobs minted by the admit pass
            self._save_map()

    def _ingest_doc(self, doc: Document) -> None:
        user = doc.user_id or "default"
        scope = _user_scope(doc.user_id)
        from verbatim.core.identity import scope_key
        self._scope_principal[scope_key(scope)] = user
        event_us = _to_us(doc.timestamp) or now_us()
        pieces = _chunks(doc.content, CHUNK_CHARS)
        self._docs[doc.id] = {"user_id": user, "timestamp": doc.timestamp,
                              "context": doc.context}
        if not pieces:
            return
        for i, piece in enumerate(pieces):
            ext = doc.id if len(pieces) == 1 else f"{doc.id}#c{i:03d}"
            env = SourceEnvelope(
                origin="amb", source_kind=SourceKind.USER_MESSAGE, scope=scope,
                speaker_id=None, payload=piece.encode("utf-8"),
                event_us=event_us, captured_us=event_us,
                provenance=Provenance.DIRECT_USER, external_id=ext,
                revision=1, metadata={"doc_id": doc.id})
            receipt = self._engine.ingest(env)
            src_id = receipt.accepted[0] if receipt.accepted else None
            if src_id is None:          # dedup replay: look the source up
                src_id = self._src_by_external(scope, ext)
            if src_id is not None:
                self._src2doc[src_id] = doc.id

    def _src_by_external(self, scope: Scope, external_id: str) -> Optional[str]:
        from verbatim.core.identity import scope_key
        with self._engine.store.read() as conn:
            row = conn.execute(
                "SELECT source_id FROM sources"
                " WHERE scope_id = ? AND origin = 'amb' AND external_id = ?",
                (scope_key(scope), external_id)).fetchone()
        return row[0] if row else None

    def _drain(self) -> None:
        # Public run_pending/drain_report pin scope=host.default_scope();
        # cross-scope drain lives on the private Ingester.
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
                " JOIN claims c ON c.claim_id = cr.claim_id").fetchall()

    def _admit_pending(self) -> None:
        """eval/v3/baselines.py::_admit_pending pattern — operator approves
        every pending claim head through the public transition API."""
        for cid, rev, state, scope_id in self._claim_heads():
            if state != "pending":
                continue
            principal = self._scope_principal.get(scope_id)
            scope = _user_scope(principal) if principal is not None \
                else _user_scope(None)
            try:
                self._engine.apply_transition(
                    TransitionCommand(
                        claim_id=cid, expected_revision=rev, effect="admit",
                        actor_id="amb-provider",
                        reason="benchmark operator approval"),
                    scope=scope)
            except Exception:
                pass   # head already moved — harmless

    def retrieve(self, query: str, k: int = 10,
                 user_id: Optional[str] = None,
                 query_timestamp: Optional[str] = None,
                 filters: Optional[dict] = None,
                 ) -> tuple[list[Document], Optional[dict]]:
        del filters
        self._ensure_loaded()
        req = RecallRequest(
            query=query, scope=_user_scope(user_id), mode=RecallMode.CURRENT,
            limit=max(1, min(k, 32)),       # limit domain is 1..32
            valid_at_us=_to_us(query_timestamp),
            max_bytes=MAX_BYTES, deadline_ms=DEADLINE_MS,
            target_tokens=TARGET_TOKENS)
        with self._lock:
            result = self._engine.recall(req)
        out: list[Document] = []
        for item in result.items:
            doc_id = self._src2doc.get(item.span.source_id)
            meta = self._docs.get(doc_id or "", {})
            ctx = meta.get("context") or (doc_id or "verbatim")
            ts = meta.get("timestamp")
            prefix = f"[{ctx}" + (f" — {ts}]" if ts else "]")
            out.append(Document(
                id=doc_id or item.claim_id,
                content=f"{prefix}\n{item.text}",
                user_id=user_id, timestamp=ts,
                context=meta.get("context"),
                source_ids=[doc_id] if doc_id else [item.span.source_id]))
        return out, None   # raw_response=None is REQUIRED for LoCoMo prompts

    def _map_path(self) -> Optional[Path]:
        return self._dir / _DOC_MAP_FILE if self._dir else None

    def _save_map(self) -> None:
        p = self._map_path()
        if p is None:
            return
        payload = {"docs": self._docs, "src2doc": self._src2doc,
                   "scope_principal": self._scope_principal}
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
                " FROM sources WHERE origin = 'amb'").fetchall()
        for source_id, external_id, scope_id in rows:
            doc_id = _CHUNK_SUFFIX_RE.sub("", external_id or "")
            if doc_id:
                self._src2doc[source_id] = doc_id
```

## 4. Smoke test (ran end-to-end, /tmp/smoke.py, Python 3.12 venv)

Corpus: 50 LoCoMo-shaped docs (JSON turn lists, `Document.context`/`timestamp` set) across 2 users.

```
built 50 docs
ingest 50 docs: 1.4s
[user1] "What breed is Caroline's dog?"        items=10 ctx_tokens=590
[user1] "When does Melanie's pottery class?"   items=10 ctx_tokens=530
[user1] "What marathon is Caroline training?"  items=10 ctx_tokens=510
[user1] "Where does Melanie work?"             items=10 ctx_tokens=519
[user1] "What is Melanie allergic to?"         items=10 ctx_tokens=540
[user2] "What motorcycle is Jon working on?"   items=10 ctx_tokens=525
[user2] "When are Gina's yoga classes?"        items=10 ctx_tokens=530
[user2] "Names of Gina's kittens?"             items=10 ctx_tokens=550
[user1] "What trip is Caroline planning?"      items=10 ctx_tokens=540
[user2] "Why is Jon seeing a physiotherapist?" items=10 ctx_tokens=525
deterministic ordering: True      # same query → identical doc-id sequence twice
cross-user items under user2 scope: 10, all user2_* docs (0 user1 leakage)
SMOKE OK                          # raw_response is None on every call
```

Sample returned doc: `id=user1_session_10`, `source_ids=['user1_session_10']`, `content="[Conversation between Caroline and Melanie (session_10 of user1) — 2023-05-11T15:00:00+00:00]\nCaroline: I adopted a rescue dog named Biscuit, a beagle mix, last weekend."`

Context sizes ~500-600 tokens for k=10 — two orders of magnitude below the Hindsight raw-response blowup; each `## Memory N` line carries speaker + context + date.

## 5. PrecisionMemBench adapter needs

From `dataset/precisionmembench.py` (`task_type="retrieval"`, `isolation_unit=None` → **batch mode**: single `ingest(all_docs)`, then all queries under semaphore; runner.py:395-418):

- **Documents**: `id = belief._id` (the `b-...` belief id), `content = beliefToText(...)` (prose), `user_id = b["user_id"]`, `timestamp = created_at`, `tags = [f"scope:{_tag(sc)}" for sc in b.scope]`, `context = f"beliefId={bid} scope={first_scope}"` (lines 271-301).
- **Queries**: `user_id = c.get("userId") or "test-user"`; `meta` = `{category, description, scope, budget, expect, retrieval_limit=maxBeliefs (default 20)}` (303-336). `retrieval_filter` produces `{any: [[scope_tags]], narrow_any: [{tags, resolve:"exact"|"fuzzy"}], none: ["state:historical","state:open_question"]}` (354-381).
- **Resolution** (`_resolve_belief_ids`, 462-492) in order: (a) any `d.source_ids` element ∈ known belief ids → "source_id"; (b) `d.id` ∈ known → "doc_id"; (c) `\bb-[a-z0-9][a-z0-9-]*\b` regex over `context + content` → "marker"; (d) lexical match. **For verbatim: return `source_ids=[<ingested Document.id>]` per returned doc** — the prototype does this via `span.source_id → external_id → doc.id`, so belief attribution resolves on the strongest "source_id" path. `Document.id` also echoes the belief id (belt). Order preserved — `orderedBefore` scoring depends on it.
- **Scoring** (`score_retrieval`, 515-659): `mustInclude`/`mustExclude`/`shouldInclude`/`shouldOnlyInclude`/`maxCount`/`minCount`/`orderedBefore` over the resolved relevantBeliefs (capped at `budget.maxBeliefs`, env `AMB_PMB_RETURN_CAP`), plus local tiers `pinnedFacts`/`openQuestions`/`personaPrelude` computed dataset-side.
- **`extraction_labels()`** (346-352) → runner calls `set_extraction_labels` if defined — verbatim ignores them (no entity-label lane); harmless.
- **Filter gap (honest)**: `supports_filters=False` → PMB's `retrieval_filter` is **dropped entirely** (retrieval.py:35). Verbatim has no tag vocabulary (`narrow_any` exact/fuzzy tag resolution is a Hindsight bank feature); PMB's own `expect` tiers still grade the unfiltered output. The only cheap partial honor would be post-filtering `none:` tags against `doc.tags` from the provider's doc map — `narrow_any`/`any` can't be honored without entity extraction, and half-honoring would zero out queries that legitimately match by content. Keep `False`; expect a measurable score hit on cases that assume `state:historical` exclusion.

## 6. LongMemEval adapter needs

- `isolation_unit="question"` → **unit-sequential runner**: per-question `async_ingest(unit_docs)` then that unit's queries (runner.py:336-385). `user_id = question_id` everywhere — the prototype's per-principal scope gives each question its own verbatim partition automatically; `unit_ids` passed to `prepare` is informational.
- Docs are `json.dumps(turns)` — the prototype's turn-aware chunker handles them; `context` carries "Session ... happened on {date} UTC" which lands in the `[ctx — ts]` attribution prefix.
- `query_timestamp` = question date → `RecallRequest.valid_at_us` gives as-of retrieval (post-question-date claims excluded).
- Timing: `retrieve_ms` per query wraps only `async_retrieve`; `ingestion_ms` aggregates per-unit ingest — verbatim's ingest includes drain+admit inside `ingest()`, so it is fully counted there (honest, but includes job-pipeline work other providers defer).
- Judge is per-category LLM (`get_judge_prompt_fn`); abstain-sensitive categories exist — verbatim's abstention warnings produce empty doc lists, which is faithful.

## 7. Parallelism / timeout knobs + gotchas

**AMB side:**
- `provider.concurrency` class attr → `asyncio.Semaphore` around each `_process_one` (runner.py:338-339, 403-404). **Set `concurrency = 1`.** (Read path is thread-safe via per-thread conns — store.py:838-857 — but there's no upside.)
- Retries: `_process_one` retries 4× on `"502"/"503"/"529"/"429"/"overloaded"/"quota"` substrings, backoff 15·2^n s (runner.py:170-181) — verbatim errors won't match those substrings, so a verbatim failure fails the query once. Verbatim typed errors (`VerbatimError.retryable`) are not retried by AMB; handle internally if needed.
- Resume flags: `--skip-ingestion` (`prepare(reset=False)` — reopen store + map), `--skip-ingested`, `--only-failed`, `--unit`, `--query-id`, `--query-limit`, `--doc-limit` (cli.py:33-94).
- Env knobs seen: `AMB_PMB_RETURN_CAP`, `AMB_RESUME`, `AMB_MAX_IN_FLIGHT_OPS`, `AMB_RECALL_MAX_TOKENS`, `AMB_RECALL_MAX_CHUNK_TOKENS`, `AMB_RETAIN_MISSION`, `AMB_RETAIN_EXTRACTION_MODE`.
- No per-query wall-clock timeout on the provider side — `retrieve_ms` just measures; keep `deadline_ms` on `RecallRequest` low enough not to stall the unit loop.

**Verbatim side (prototype defaults):**
- `AMB_VERB_CHUNK_CHARS=1000` (must stay <1200), `AMB_VERB_DEADLINE_MS=4000`, `AMB_VERB_MAX_BYTES=24000`, `AMB_VERB_TARGET_TOKENS=4096`, `AMB_VERB_DRAIN_LIMIT=4096`, `AMB_VERB_REQUIRE_REVIEW=1`, `AMB_VERB_EMBEDDING=hashing`, `AMB_VERB_MODE=offline_rules`, `AMB_VERB_PROFILE=amb`, `AMB_VERB_PRINCIPAL=bench`.
- Ingest is **fully synchronous inside `ingest()`** (envelope → drain → admit → drain) — `ingestion_ms` covers it all; no background jobs leak into the query phase.
- `concurrency=1` + `threading.Lock` (sync `retrieve`/`ingest` run in `to_thread` workers — `asyncio.Lock` would bind the wrong loop).
- `cleanup()` closes the engine; runner calls it at end (runner.py:~460). `Store.close()` closes all per-thread reader conns — don't let stray retrieves outlive cleanup.
- One engine + one store per run under `store_dir/verbatim/` (`{profile}.db` + `amb.db.key` + `doc_map.json`); `prepare(reset=True)` wipes it. Batch-mode datasets (PMB) share one engine for all users — scopes partition.
- WAL-mode sqlite under `/tmp` is fine; store files are per-run, nothing global leaks between runs.

## 8. Residual risks / open items for the build agent

1. `engine._ingester` is a private accessor — either accept it (stable @1c86f4f) or add a public `run_pending(scope=None)` to verbatim before W6 lands.
2. `harvest` bounds hardcoded → provider-level chunking is load-bearing; `_chunks` must stay <1200 chars (set to 1000).
3. `RecallRequest.limit` caps at 32 — PMB `maxBeliefs` >32 would silently truncate (default 20; note `AMB_PMB_RETURN_CAP` also truncates).
4. Chunked docs mint 1..N sources per Document.id — `source_ids` reporting is per-chunk; if a returned doc resolves to the same belief twice it's deduped in `_resolve_belief_ids` (`seen` set).
5. `run_pending(scope=None)` drains ALL scopes — fine within one engine; keep engines per-run so no cross-run leakage.
6. Abstention warnings are surfaced in `result.warnings` but not exported to AMB — could be stashed in `raw_response`, but raw_response MUST stay None for LoCoMo/LME; leave it out.
