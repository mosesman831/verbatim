# V3 Worker Contracts (frozen interfaces)

Shared contracts every V3 module builds against. Foundation is committed at
`414b1f0` — do not modify shared files; build in your own directories.

## Conventions (from repos.py / repos_v3.py)

- Rows never escape as objects: reads return plain `dict` snapshots.
- Mutating helpers take the caller's transaction `conn`; one logical write
  commits atomically (SPEC §21).
- `store.tx()` → write transaction; `store.read()` → snapshot read.
- `repos_v3.insert/get/query/update/delete(conn, table, ...)` — allowlist-
  validated CRUD for all §39 v3 tables. JSON columns take native values.
- `repos_v3.json_field(row, key, default)` decodes `*_json` columns.
- Errors are `VerbatimError(ErrorCode.X, msg)`; v3 codes live in
  `verbatim/core/types.py` (NOT_FOUND_OR_UNAUTHORIZED, CAPABILITY_UNAVAILABLE,
  CONSENT_REQUIRED, QUARANTINED, STALE_EPOCH, BUDGET_EXCEEDED,
  COMPILATION_UNSUPPORTED, INVESTIGATION_UNSUPPORTED, INTEGRITY, LOCKED,
  RETRYABLE_OPERATION).
- All V3 types live in `verbatim/core/types_v3.py` — import, don't redefine.

## Job handler contract (ingest.py dispatch)

`_V3_KIND_HANDLERS` maps JobKind → `(module, func)`. Implement:

```python
def handle_<kind>(job: dict, owner: str, ingester) -> None:
    # job = {"job_id", "scope_id", "kind", "input_refs", "generation", ...}
    # use ingester.store for tx()/read(); input_refs is your payload dict
```

Registered targets (implement exactly these paths):

| kind | module | function |
|---|---|---|
| screen | `verbatim.security.handlers` | `handle_screen` |
| sparse_index | `verbatim.retrieval.v3.handlers` | `handle_sparse_index` |
| late_index | `verbatim.retrieval.v3.handlers` | `handle_late_index` |
| signature_index | `verbatim.procedures.handlers` | `handle_signature_index` |
| episode_build | `verbatim.experience.handlers_v3` | `handle_episode_build` |
| transition_build | `verbatim.experience.handlers_v3` | `handle_transition_build` |
| procedure_compile | `verbatim.procedures.handlers` | `handle_procedure_compile` |
| procedure_refine | `verbatim.procedures.handlers` | `handle_procedure_refine` |
| consolidate | `verbatim.observations.handlers` | `handle_consolidate` |
| purge_derived | `verbatim.privacy.handlers` | `handle_purge_derived` |
| purge_vault | `verbatim.privacy.handlers` | `handle_purge_vault` |
| quarantine_review | `verbatim.security.handlers` | `handle_quarantine_review` |
| revocation_notify | `verbatim.governance.handlers` | `handle_revocation_notify` |
| vault_rotate | `verbatim.privacy.handlers` | `handle_vault_rotate` |
| projection_sync | `verbatim.projection.handlers` | `handle_projection_sync` |
| connector_pull | `verbatim.connectors.handlers` | `handle_connector_pull` |

## Governance interface (frozen — implement to this signature)

```python
# verbatim/governance/__init__.py exports:
@dataclass(frozen=True)
class CallerV3:
    principal_id: str          # authenticated outside request params (§08.02)
    session_id: str = ""
    host_id: str = ""
    epoch: Optional[int] = None  # pinned authz epoch; None = current

def authorize(conn, caller: CallerV3, scope_id: str, verb: str,
              purpose: Optional[str] = None,
              object_ref: Optional[tuple[str, str, int]] = None) -> None:
    """Raise NOT_FOUND_OR_UNAUTHORIZED when the caller lacks an effective
    grant for (scope, verb[, purpose]) at the caller's epoch. Never reveals
    existence vs authorization (§10.05). Returns None on success."""

def effective_verbs(conn, caller: CallerV3, scope_id: str) -> frozenset: ...
def record_propagation(conn, propagation: Propagation) -> str: ...
```

Until governance lands, callers code against the signature; tests may stub
`authorize = lambda *a, **k: None` only inside their own test files.

## Security-label interface (frozen)

```python
# verbatim/security/__init__.py exports:
def attach_label(conn, scope_id: str, *, source_trust: str,
                 content_form: str = "unknown", attack_risk: str = "unassessed",
                 review_state: str = "not_required",
                 findings: list | None = None, method: str = "rules",
                 rules_revision: str = "") -> str:  # label_id
def is_quarantined(conn, object_kind: str, object_id: str, revision: int) -> bool: ...
def label_for(conn, label_id: str) -> Optional[dict]: ...
```

## Envelope-ingest interface (frozen)

```python
# verbatim/evidence/__init__.py exports:
def ingest_envelope(conn, store, envelope: EnvelopeV3, *,
                    authorization: Optional[CaptureAuthorization] = None
                    ) -> CaptureReceipt:
    """Persist one V3 envelope: sources + source_revisions + spans +
    source_envelopes + screening + receipt, atomically. trust_class/agent
    attribution never crosses (agent text is never human testimony)."""
```

## Drain obligation (deployment)

Every enqueue surface writes durable jobs atomically, but nothing on the
SDK / adapter / `api_v3` path drains implicitly. The only drain is
`Ingester.run_pending`, reached today from `api_ingest.run_pending`, the
CLI, the v2 `provider.on_session_end`, the replay lab, and the
capture-SDK's own drain:

- `CaptureClient.drain_pending(scope=None, *, limit=64, kinds=None,
  lane=None, owner=None)` — synchronous drain over the client store,
  returning the drained count. The lazily-constructed `Ingester` carries
  no judge/encoder provisions, so embedding jobs degrade to recorded
  `encoder_unavailable` notes rather than silently succeeding.
- `NativeAdapter.drain(...)` plus the canonical `"drain"` event op expose
  the same capability to adapter hosts; the concrete adapters also drive
  it on their declared session-end seams (`job_drain` in the capability
  matrix: `on_session_end` for Hermes, session close for ADK).

A deployment that embeds the SDK, binds an adapter, or serves `api_v3`
without also running a worker MUST schedule one of these drains (idle /
session boundary, or a dedicated worker loop). Otherwise harvest→admit,
screens, `episode_build`, and the privacy-control lane's purges wait
forever — accepted sources never become claims and suppression/purge jobs
never execute. The drain is an explicit synchronous call; no library code
spawns threads or loops.

## What "done" means per worker

- New files only in your owned directories (below); do NOT edit
  `types.py`, `types_v3.py`, `schema*.py`, `repos_v3.py`, `config.py`,
  `ingest.py`, `store.py`, `migrations.py`, `pyproject.toml`,
  `tests/conftest.py`, or another worker's directory.
- Tests in your own `tests/<area>/` directory; real SQLite via the store
  fixtures pattern from `tests/storage/test_v3_foundation.py`
  (`Store.create` in tmp_path, or the `TestStore` shim for pure-row tests —
  but note the shim is DDL_V1-only: v3 tests must use real `Store.create`).
- Every mutating path is transaction-scoped and idempotent where the spec
  requires (operation_key/dedup where applicable).
- `python -m pytest tests/<your_area> -x -q` green; then run the FULL suite
  (`python -m pytest -x -q`) before reporting done — the 648-test baseline
  must stay green.
- Cite SPEC_V3 requirement IDs in module docstrings.
