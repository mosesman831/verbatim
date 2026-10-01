# V6 frozen worker contracts

Status: frozen for the V6 build wave. Workers code against these
signatures exactly; changes go through the coordinator, not local edits.
Namespace: requirements `V6-NN.MM`, scenarios `F01`–`F32`, gates
`G6-00`–`G6-08` (`SPEC_V6.md`).

## 1. Cross-store commit notification (w6-wake)

New module `verbatim/storage/commit_notify.py`:

```python
def register(path: str) -> "PathKey":            # canonical abspath key
def commit_fired(path: str) -> None              # called by Store after COMMIT
def wait(path: str, timeout_s: float) -> bool    # True if a commit fired
def subscriber_count(path: str) -> int           # diagnostics only
```

- `Store` calls `commit_notify.commit_fired(self._path)` inside the same
  post-commit block that notifies `self._commit_cond` (store.py ~924).
- `readiness._wait_commit` waits on `commit_notify.wait(store._path, d)`
  in addition to `store._commit_cond` — whichever fires first; polls
  still re-verify (advisory wake, never correctness).
- `wait` uses a per-path `threading.Condition` in a module-level
  `{abspath: _PathSignal}` registry; fork-safe (registry rebuilds in
  child, same `_INIT_LOCKS` pattern as facade).
- No new files/sockets; process-local only.

## 2. Barrier-blocked source marking (w6-wake provides, w6-drain consumes)

Same module:

```python
def note_barrier_sources(path: str, source_ids: Iterable[str]) -> None
def blocked_sources(path: str) -> frozenset      # sources blocking live barriers
def clear_barrier_sources(path: str) -> None
```

- `facade.search` calls `note_barrier_sources` with the source ids of
  pending barrier receipts (it already computes `unresolved`); the
  facade edit is owned by w6-typed (§7), NOT w6-wake.
- `blocked_sources` returns the current set; `clear` is called by the
  drainer after those sources' `source_lexical_ready` settles.

## 3. Unblock-first drain (w6-drain)

- `Ingester.drain_report` gains optional kwarg `priority_sources:
  Optional[AbstractSet[str]] = None`. When non-empty, the dequeue pass
  first leases `source_project`/`source_embed` jobs whose payload
  `source_id` is in the set — bounded by `limit`, same fencing, same
  lane rules otherwise. Privacy/correctness lanes still dequeue first.
- `worker.py` obtains the set via `commit_notify.blocked_sources(path)`
  each drain pass and calls `clear_barrier_sources` when the pass leaves
  none of them pending.
- `JobQueue` may add a private `lease_priority(scope, kinds,
  source_ids, owner, limit)` used only when `priority_sources` is
  non-empty — never a public second scheduler.

## 4. Service consumer surface (w6-service)

Routes on the existing `verbatim/service` `HttpServer` + bearer auth:

```
POST /v2/memory/add        {text, infer?, metadata?, replaces?} -> AddResult JSON
POST /v2/memory/search     {query, limit?, consistency?, timeout_ms?,
                            ready_timeout_ms?, after?} -> SearchResult JSON
POST /v2/memory/inspect    {ref, detail?} -> InspectResult JSON
POST /v2/memory/forget     {ref, confirm_token?, preview?} -> ForgetResult JSON
GET  /v2/memory/status     -> Memory.status() JSON
GET  /v2/memory/readiness/{receipt_id} -> wait_ready snapshot JSON
GET  /v2/memory/capabilities -> honest capability dict
```

- One `Memory` instance per service app, constructed at server start
  with the credential's bound principal as `user_id`; a credential
  whose scope != app scope gets `CONFIG_INVALID` at bind time — request
  payloads can never mint identity or namespace.
- All bodies parse through `verbatim.core.serialize.safe_json_loads`;
  responses serialize via `json_dumps`; `Acceptance`/`SearchStatus`/
  `Readiness` enums serialize as their `.value` strings.
- Errors map `VerbatimError.code.value` -> `{error, code, retryable}`
  + HTTP 400/403/409/503 by class; never a bare 500 with a traceback.
- The app factory lives in `verbatim/service/memory_api.py`:
  `create_memory_app(path, *, user_id, worker="managed", tokens=None)
  -> HttpServer`. `service/api.py` may register it under `/v2/memory`
  when constructed with `enable_v5_memory=True`.

## 5. TypeScript client (w6-dist)

`clients/ts/memory-client/` — one package, `src/index.ts` thin client:

```ts
class MemoryClient {
  constructor(opts: {baseUrl: string; token: string; timeoutMs?: number})
  add(text: string, opts?): Promise<AddResult>
  search(query: string, opts?): Promise<SearchResult>
  inspect(ref: string, opts?): Promise<InspectResult>
  forget(ref: string, opts?): Promise<ForgetResult>
  status(): Promise<StatusResult>
  waitReady(receiptId: string, timeoutMs?): Promise<ReadinessResult>
}
```

- `fetch`-only (Node ≥18 / browser), zero deps; types mirror the JSON
  shapes in §4. Build: `tsc` only if available; otherwise ship `.ts` +
  generated `.d.ts`-style JSDoc — verified by a Python-side recorded
  stub test plus one live loopback smoke in `tests/v6/`.

## 6. Neural artifact (w6-neural)

`verbatim/embeddings/artifact_build.py`:

```python
def build_artifact(out_dir: str, *, model: str, revision: str,
                   table: dict[str, list[float]], seed: int = 42) -> dict
   """Writes <out_dir>/<model>/<revision>/ vectors.bin (float32le,
   row-major, dim-prefixed header) + artifact_manifest.json with
   sha256 per file + loader source-hash. Returns the manifest dict."""
```

- `ArtifactEncoder` learns to load `vectors.bin` via a reviewed
  parse-only loader (header: magic `VBV1`, dim u32, count u64, then
  count×dim float32le; sha256 re-verified at load). `available()` True
  only when manifest+files verify. `encoder_id` =
  `artifact:<model>:<revision>`.
- `eval/v6/neural.py`: trains the table offline (e.g., hashing features
  through a pinned dev corpus; deterministic seed), writes the artifact,
  runs paired hashing-vs-artifact quality comparisons.
- `querying/calibration.py` gains the new `encoder_id` entry with a
  fitted floor + `validate()` pass; no copied thresholds.

## 7. Facade search rework (w6-typed owns `memory/facade.py` this wave)

Inside `search`, in this order:

1. existing barrier (unchanged semantics) + NEW:
   `commit_notify.note_barrier_sources(store._path, src_ids)` for the
   pending receipts' sources (resolve receipt → source via
   `source_state`/jobs payload);
2. typed-memory candidates first (`verbatim/retrieval/v3/typed_lane.py`
   — new lane over grounded typed records + identifier/entity postings
   in the bound namespace);
3. fallback to the existing source lane when typed coverage is thin
   (honest `coverage.lanes` reporting either way);
4. NEW exposure emission: after delivery, append one row per delivered
   item to the exposure sink (§8) — same tx discipline as
   `log_decision` (separate short write, never inside the read tx);
5. pack cache consulted after the barrier, fingerprint includes the
   barrier generation.

`Memory.search` signature unchanged; `coverage.lanes` reports which
lanes actually ran.

Consistency note (V6-02.10): `consistency="session"` (default) runs the
per-receipt causal barrier; `consistency="eventual"` skips the session
barrier entirely and the result says so plainly —
`readiness.causal_satisfied` is `false` and `warnings` carries
`consistency_eventual`. An explicit `after` handle still applies in
eventual mode (it is the caller's own causal token), and hold
enforcement is identical on both paths — eventual never leaks held
evidence.

## 8. Exposure/influence on the consumer path (w6-feedback)

- New `verbatim/influence/exposure.py` (or extend
  `retrieval/v3/influence.py`): `record_source_deliveries(conn,
  receipt_id, deliveries)` — rows in a new `source_exposure` table
  (schema_v5 extension, additive migration; PK `(receipt_id, ord)`;
  columns: receipt_id, ord, source_id, revision, score_family,
  delivered_at_us). Writes happen in the facade's post-delivery write,
  not the read tx.
- `eval/v5/feedback.py` probes extend to count `source_exposure` rows.

## 9. auto_safe policy (w6-autosafe)

- `verbatim/querying/auto_update.py`:
  `auto_safe_replace(conn, namespace, new_record, candidate) ->
  Optional[Decision]` returning `{applied: bool, reason}`; applies only
  same-type, same-identifier/entity, unhedged value/version/polarity
  updates; everything else stays `possible_updates` advisory.
- Namespace opt-in via `meta` key `update_policy:<ns>="auto_safe"`.
- `eval/v6/twins.py` — paired auto-update/false-update twin corpus,
  upper-95% false-auto bound published.

## 10. Comparator ladder (w6-comparators)

`eval/v6/comparators.py` — new file; reuses the v5 `ComparatorArm`
protocol (`eval/v5/comparators.py`). Arms:

- `verbatim_memory` (current path), `verbatim_v2`, `naive_fts`,
  `vector_rag`, `no_memory` — all execute through
  `eval.v3.baselines`/v5 harness helpers, fully offline.
- `mem0_oss` — probe `mem0ai`; when importable, a REAL adapter through
  its documented `add`/`search`/`get_all`/`delete` lifecycle with
  pinned 13-field `ComparatorPin`; `infer=False` runs only as
  `mem0_oss_inferfalse` with the disclosure recorded. Otherwise
  `status="unavailable"`, reason recorded.
- `graphiti_oss`, `zep_hosted`, `mem0_platform`, `holographic` —
  probe/named-unavailable rows, never fabricated.

## 11. Ledger (w6-ledger)

- `eval/v6/ledger.py` — clone of v5's generator shape: parses
  `SPEC_V6.md` for `V6-NN.MM`, `F\d+`, `G6-NN`; merges
  `eval/v6/dispositions_v6.json`; writes `ledger_v6.json` +
  `summary.md`; `tools/gen_v6_ledger.py` CLI with `--check`/`--write`.
- Also carries-forward the V5 rows the program re-dispositions
  (V6-04.02/03) in a `carried` block — reads
  `eval/v5/dispositions_v5.json`, never edits it (w6-closure owns V5
  dispositions in phase 2).

## 12. Consolidation (w6-consolidation)

- `eval/v5/consolidation.py`: add corpus fixture items (two same-slot
  whitelist-predicate claims, distinct families — e.g. two
  "My favourite café is X/Y" items on the same subject+predicate), so
  `observations` can pass positively; tighten `corroboration` to
  require folded `hit.corroboration >= 2`.
- `verbatim/observations/`: pin-liveness check —
  `resolve_pins(conn, obs_id)` returning held/erased pins so a summary
  withholds/retires; `never_sole_trace` invariant = every pin resolves
  to retrievable source bytes (test in `tests/observations/`).

## 13. Envelopes + scale (w6-a3)

- `eval/v6/envelopes.py`: A0/A0-cache/A0-neural/A1/A3 harness (port v5
  envelopes, add A3 scale config ~20k records; seed via
  `harness.bulk_seed` — a new harness-side function doing batched
  `Memory.add` + interleaved `drain_memory`, NOT a new public API).
- Add-ack stage profile: wrap facade add path stages
  (`_check_retention_policy`, ingest, obligations, prescan, commit)
  with the `StageProfiler` idiom; report per-n scaling.

## 14. Framework adapter (w6-framework)

`verbatim/adapters/langchain.py`:

```python
class VerbatimRetriever:           # duck-typed, no langchain import
    def __init__(self, memory, *, k: int = 8, search_kwargs=None): ...
    def get_relevant_documents(self, query: str) -> list["Document"]
    async def aget_relevant_documents(self, query: str) -> list["Document"]
```

- `_Doc` minimal stand-in (`page_content`, `metadata={"source_id":…}`)
  when `langchain_core.documents.Document` is absent; uses the real
  class when importable. Verified by `tests/adapters/` against a
  recorded stub + real Memory instance.

## Ownership map (strict — do not edit outside it)

| Worker | Owns |
|---|---|
| w6-ledger | eval/v6/ledger.py, eval/v6/dispositions_v6.json, tools/gen_v6_ledger.py, tests/eval/test_v6_ledger.py |
| w6-wake | verbatim/storage/commit_notify.py, verbatim/storage/store.py (commit_fired call only), verbatim/readiness.py (_wait_commit only), tests/storage/, tests/readiness* |
| w6-drain | verbatim/memory/worker.py, verbatim/ingest.py (priority_sources kwarg), verbatim/jobs/queue.py, tests/jobs/, tests/memory/test_worker* |
| w6-service | verbatim/service/memory_api.py, verbatim/service/api.py (route registration), verbatim/service/auth.py (scope bind), tests/service/ |
| w6-dist | clients/ts/, pyproject.toml, LICENSE, verbatim/py.typed, verbatim/service/__main__.py, verbatim/api_v3/mcp.py (entry point only), README.md |
| w6-consolidation | eval/v5/consolidation.py, verbatim/observations/, tests/observations/, eval/v5/corpus.py (fixture items) |
| w6-feedback | verbatim/influence/ (new) or retrieval/v3/influence.py, verbatim/policy/artifact registration, verbatim/storage/repos_v2.py (set_state attestation), verbatim/config.py (learned_active binding), eval/v5/feedback.py, verbatim/storage/schema_v5.py (source_exposure table), tests/ |
| w6-typed | verbatim/memory/facade.py, verbatim/retrieval/v3/typed_lane.py, verbatim/memory/cards.py (if needed), tests/memory/, tests/v5 facade-touching tests |
| w6-neural | verbatim/embeddings/artifact_build.py, verbatim/embeddings/artifact.py, verbatim/querying/calibration.py, eval/v6/neural.py, tests/embeddings/ |
| w6-comparators | eval/v6/comparators.py, eval/v6/portfolio.py, eval/v6/run.py, tests/eval/test_v6_comparators.py |
| w6-a3 | eval/v6/envelopes.py, eval/v5/harness.py (bulk_seed helper only), tests/eval/test_v6_envelopes.py |
| w6-autosafe | verbatim/querying/auto_update.py, eval/v6/twins.py, tests/querying/test_auto_update.py |
| w6-framework | verbatim/adapters/langchain.py, tests/adapters/test_langchain.py |

Shared-readonly: SPEC_V6.md, docs/v6_contracts.md, eval/v5/* (except
consolidation.py/corpus.py/feedback.py/harness.py as assigned),
verbatim/core/*, verbatim/memory/types.py.

Cross-seam calls only through the frozen signatures above. If a worker
needs a seam change, it notes it in its report — it does not edit the
other module.
