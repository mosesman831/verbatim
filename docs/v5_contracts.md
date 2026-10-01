# V5 frozen worker contracts (SPEC_V5 build wave 1)

Binding interfaces for parallel implementation. Cross-module seams are
defined here; do not invent alternatives. Everything runs on the existing
authority: `kernel/`, `governance/`, `storage/store.py`, `jobs/coordinator.py`,
`readiness.py`, `retrieval/v3/`. No second engine, store, or policy path.

## 1. Public types (already landed)

`verbatim/memory/types.py` is frozen: `MemoryRef`, `AddResult`, `SearchResult`,
`Hit`, `Inspection`, `ForgetResult`, `Readiness`, `MemoryStatus`,
`CloseReport`, `UpdateCandidate`, all enums, `QUERY_ANALYSIS_VERSION`,
`RANKING_VERSION`, `ENRICHMENT_VERSION`, `SOURCE_STATE_KIND`,
`CAP_SOURCE_LEXICAL` (`source_lexical_ready`), `CAP_SOURCE_VECTOR`
(`source_vector_ready`).

## 2. Profile and store resolution

- Behavioral profile string: `local_memory` (registered by facade worker in
  `verbatim/config.py` validation + `v3` profile set; `embedded`/`local_rules`
  stay Engine-level and unchanged).
- Store path resolution MUST go through `verbatim/storage/resolver.py` —
  `(cfg.data_dir, host.profile_id())`; `local_memory` never enters the tuple.
- `Memory()` binds: store → owner bootstrap (trusted local owner, one personal
  namespace) → bound caller with scoped root verbs
  `{read, quote, ingest, derive, review, admin}` (§05.11).

## 3. Schema v5 (storage worker owns)

`verbatim/storage/schema_v5.py`, `SCHEMA_VERSION = 5`, migration 4→5 in
`migrations.py` (append-only, never edit historical DDL). New tables:

```sql
source_state(                 -- source_state/v1 control artifact (§14.3)
  source_id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL,
  control_version INTEGER NOT NULL,        -- monotonic, CAS target
  mutation_head TEXT NOT NULL,             -- approved head revision id
  disposition TEXT NOT NULL,               -- active|superseded|corrected|retracted
  superseded_by TEXT, effective_at TEXT,   -- RFC3339; future = read-time only
  known_at TEXT NOT NULL, valid_from TEXT, valid_to TEXT,
  updated_at TEXT NOT NULL, producer TEXT NOT NULL)

source_lexical_projection(    -- searchable token projection of source bytes
  source_id TEXT NOT NULL, revision INTEGER NOT NULL,
  scope_id TEXT NOT NULL, generation INTEGER NOT NULL,
  tokens TEXT NOT NULL,                    -- space-joined normalized tokens
  doc_len INTEGER NOT NULL, digest TEXT NOT NULL,
  PRIMARY KEY (source_id, revision))       -- + FTS5 shadow or postings table

source_vectors(               -- hashing encoder vectors, contiguous per ns
  source_id TEXT NOT NULL, revision INTEGER NOT NULL, namespace TEXT NOT NULL,
  encoder TEXT NOT NULL, generation INTEGER NOT NULL,
  vector BLOB NOT NULL, digest TEXT NOT NULL,
  PRIMARY KEY (source_id, revision, encoder))

entity_postings(              -- deterministic identifier/entity mentions
  namespace TEXT NOT NULL, entity TEXT NOT NULL, entity_kind TEXT NOT NULL,
  source_id TEXT NOT NULL, revision INTEGER NOT NULL, offsets TEXT NOT NULL,
  generation INTEGER NOT NULL,
  PRIMARY KEY (namespace, entity, source_id, revision))

duplicate_links(              -- dedup by linking; never deletes (§30.2)
  source_id TEXT NOT NULL, revision INTEGER NOT NULL, group_id TEXT NOT NULL,
  method TEXT NOT NULL,                    -- exact_digest | normalized | minhash
  score REAL NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (source_id, revision, method))

enrichment(                   -- T1 deterministic enrichment output (§30)
  source_id TEXT NOT NULL, revision INTEGER NOT NULL,
  producer TEXT NOT NULL,                  -- e.g. enrich/v1
  type TEXT, polarity TEXT,
  time_precision TEXT, time_status TEXT, event_at TEXT, anchor_at TEXT,
  fields_json TEXT NOT NULL,               -- identifiers, entities, time exprs
  PRIMARY KEY (source_id, revision, producer))

update_candidates(            -- advisory possible_updates (§30.5)
  candidate_id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
  new_source_id TEXT NOT NULL, new_revision INTEGER NOT NULL,
  prior_source_id TEXT NOT NULL, prior_revision INTEGER NOT NULL,
  relation TEXT NOT NULL,                  -- contradicts|newer_value|negates|refines
  score REAL NOT NULL, state TEXT NOT NULL, -- open|adopted|dismissed
  created_at TEXT NOT NULL)

backfill_cursor(              -- durable source-backfill cursor (§07)
  job_key TEXT PRIMARY KEY, last_source_id TEXT NOT NULL,
  generation INTEGER NOT NULL, done INTEGER NOT NULL, updated_at TEXT NOT NULL)
```

All rows are derived projections of retained source bytes: deletion closure
must remove them, rebuilds may regenerate them. Registered repos in
`verbatim/storage/repos_v5.py` (allowlist CRUD only, same style as
`repos_v4.py`).

## 4. Readiness (readiness worker owns `readiness.py`)

- Add `CapabilityName` values `source_lexical_ready`, `source_vector_ready`.
- New DAG branch for source-backed captures: `accepted → screened →
  source_lexical_ready` and `source_vector_ready` (sibling on screened).
  Legacy claim capabilities (`lexical_ready`, `semantic_ready`,
  `derived_ready`) unchanged; a capture may carry both branches.
- `plan_obligations(...)` gains optional `include_source=True` used by the
  facade path; callers decide which branch a capture declares.

## 5. Jobs (source-jobs worker owns new files only)

- `verbatim/jobs/source_jobs.py`: handlers `source_project` (tokenize +
  write `source_lexical_projection`, `entity_postings`, `enrichment`,
  `duplicate_links` discovery, `update_candidates` detection),
  `source_embed` (hashing vector → `source_vectors`), `source_backfill`
  (durable cursor scan of pre-v5 sources).
- `verbatim/jobs/v5_handlers.py`: `V5_KIND_HANDLERS` dispatch map in the same
  lazy-import style as `_V3_KIND_HANDLERS`. Main session merges it into
  `ingest.py` dispatch.
- Job payloads: `{source_id, revision, namespace, scope_id, generation,
  producer}`; source_project also `{utf8_ok: true}`.
- Publication predicate (V5-07.13): valid retained source revision + digest,
  completed screening, allowed indexing/retention class, processing consent,
  no covering quarantine/suppression/erasure, current source-state generation.

## 6. Retrieval (retrieval worker owns new files only)

- `verbatim/retrieval/v3/source_lane.py`: `source_candidates(store, conn, *,
  query_terms, identifiers, entities, eligible_ids|None, namespace,
  snapshot, limit)` → lexical BM25 over `source_lexical_projection` using
  **full eligible-corpus statistics** (reuse `retrieval/candidates.py` F4-11
  machinery — N/df/avgdl over E, never candidate-set stats) + hashing-similarity
  candidates over `source_vectors` + exact identifier/entity posting hits.
  Deterministic, bounded, eligibility-aware iteration.
- `verbatim/retrieval/v3/fusion_v1.py`: `fuse(candidates, signals, *,
  weights=RANKING_V1_WEIGHTS)` — versioned deterministic fusion contract
  `ranking/v1`: normalized signal sum with declared weights/tie-breaks.
  Signals rank admitted candidates only — never generate or admit.
- Entity timeline read (§30.16): `entity_timeline(conn, namespace, entity)`
  over `entity_postings` + `source_state` lifecycle labels.

## 7. Enrichment (enrichment worker owns `verbatim/enrichment/`)

Pure deterministic functions, no store access:

```python
def normalize_text(text: str, *, version: str = "norm/v1") -> str
def extract_identifiers(text: str) -> list[Identifier]  # paths, urls, emails,
    # handles, versions, hashes, ticket keys, quoted strings, code tokens;
    # each: (kind, value, start, end) byte offsets, case preserved
def extract_entities(text: str) -> list[Entity]          # capitalized spans
def parse_temporal(text: str, anchor: str) -> TemporalResult  # precision,
    # status, event_at, anchor_at; ambiguity -> unknown
def polarity(text: str) -> Polarity
def classify_type(text: str) -> MemoryType
def normalized_digest(text: str) -> str                  # blake2b over normalize_text
def shingle_signature(tokens: list[str]) -> frozenset    # for MinHash dedup
```

`ENRICHMENT_VERSION = "enrich/v1"` is the producer label stamped on rows.

## 8. Query analysis + update candidates (querying worker owns)

`verbatim/querying/`:
- `analyze(query: str) -> QueryAnalysis` (query_analysis/v1): classes,
  extracted identifiers/entities/temporal intent. Deterministic, no model.
- `detect_update_candidates(conn, namespace, new_record) -> list[candidate]`
  — prior live records sharing subject/entity/type differing in value,
  polarity, version, or time; writes `update_candidates` rows state=open.

## 9. Dedup (dedup worker owns `verbatim/dedup/`)

- `link_exact(conn, source_id, revision, namespace, digest)`,
  `link_near(conn, source_id, revision, namespace, signature)` — write
  `duplicate_links` + assign `group_id` (earliest live member's source_id).
- Never link across differing polarity/identifier-set/version-number/time/
  type (§30.07). `dedupe` namespace policy: `link` (default) | `none`.

## 10. Facade (facade worker owns `verbatim/memory/facade.py`,
`bootstrap.py`, `errors.py`, `worker.py`, `controls.py`, `aliases.py`,
`verbatim/__init__.py` export, `config.py` `local_memory` registration)

```python
class Memory:
    def __init__(self, path=None, *, user_id=None, profile="local_memory",
                 worker="managed", encoder="hashing", ready_timeout_ms=200,
                 create=True, config=None, host=None)
    def add(self, content, *, infer=True, metadata=None,
            idempotency_key=None, replaces=None, change="supersede",
            effective_at=None) -> AddResult
    def search(self, query, *, limit=8, filters=None, after=None,
               consistency="session", ready_timeout_ms=None,
               timeout_ms=500, strict=False) -> SearchResult
    def inspect(self, ref, *, detail="evidence") -> Inspection
    def forget(self, ref=None, *, query=None, confirmation=None,
               idempotency_key=None) -> ForgetResult
    def wait_ready(self, receipt, *, capabilities=None,
                   timeout_ms=2000) -> Readiness
    def status(self) -> MemoryStatus
    def close(self, *, timeout_ms=5000, strict=False) -> CloseReport
    __enter__/__exit__ -> close()
```

- `verbatim/__init__.py` gains lazy `Memory` via `__getattr__` (import stays
  side-effect free).
- `worker.py`: managed = bounded ref-counted drain thread per store on the
  existing queue/coordinator; `external` = none; fork-safe (reopen after
  fork); `close()` bounded, honest `CloseReport`.
- `aliases.py`: user_id/agent_id/run_id → namespace mapping; run-scoped TTL
  working memory; no authority minting.
- `controls.py`: inspect (evidence/metadata details) + forget (ref → closure;
  query → preview + confirmation token → operation).

## 11. Compat (compat worker owns `verbatim/compat/`)

`verbatim/compat/mem0.py`: `Memory`/`MemoryClient` shim translating
`add/search/get/get_all/update/delete` over the facade — same store, same
authority, honest pending/unsupported states, CAS update via `replaces`.

## 12. Scenarios + ledger (two workers)

- `tests/v5/` — E01–E96 acceptance tests written against these contracts and
  SPEC_V5.md §24/§36 (they fail until features land; mark expected-fail
  clearly with `xfail` on unimplemented gates, assert whatever works).
- `eval/v5/` + `tools/gen_v5_ledger.py` — ledger registration for all
  `V5-NN.MM` requirements parsed from `SPEC_V5.md`, scenario/gate map,
  `--check` mode mirroring `eval/v4/ledger.py` conventions. All rows start
  `planned`/`not_run` — never mark verified without executed evidence.

## 13. Hard rules for every worker

1. One authority: reuse `kernel/`, `governance/`, `coordinator`, `closure`,
   `resolver`, `readiness`. No facade-local policy, no parallel verbs.
2. Derived data (projections, enrichment, links, vectors) is regenerable;
   canonical bytes are sacred — never rewrite sources.
3. Strict UTF-8 acceptance; no silent normalization of canonical bytes.
4. Denial indistinguishability preserved; typed errors only.
5. Honest states: pending/blocked/unavailable are distinct; never fake ready.
6. Tests must exercise real on-disk `Store` instances, not mocks of the
   kernel. Follow `tests/` conventions (conftest helpers, tmp_path stores).
7. Python stdlib only for the default path — no new dependencies.
8. Report files changed, tests added, and any spec ambiguity found — do not
   silently reinterpret the spec.
