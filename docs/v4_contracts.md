# V4 Worker Contracts (frozen interfaces)

Shared contracts every V4 module builds against. Predecessor: `docs/v3_contracts.md`
(those conventions still apply). `SPEC_V4.md` is the authority; `SPEC_V4_5.md` is the
M5 addendum. This file freezes signatures so parallel workers never collide.

## Conventions (unchanged from v3)

- Rows never escape as objects: reads return plain `dict` snapshots.
- Mutating helpers take the caller's transaction `conn`; one logical write commits
  atomically. `store.tx()` → write transaction; `store.read()` → snapshot read.
- `repos_v3.insert/get/query/update/delete(conn, table, ...)` — allowlist-validated
  CRUD. New v4 tables are registered in `storage/schema_v4.py` + `repos_v4.py`
  allowlists.
- Errors are `VerbatimError(ErrorCode.X, msg)`. New v4 codes live in
  `core/types.py::ErrorCode` — extend the enum, never invent parallel error classes.
- All V4 types live in `verbatim/core/types_v4.py` — import, don't redefine.
- Honest status: anything not actually executed is `unimplemented`/`deferred`/
  `unavailable` with a reason — never silently simulated.

## Frozen types (verbatim/core/types_v4.py)

```python
# Purpose constraint — F4-03 / V4-11.01. Three tagged forms; the tag is part of
# the persisted representation. Empty SET normalizes to NONE, never ANY.
@dataclass(frozen=True)
class PurposeConstraint:
    tag: PurposeTag              # ANY | SET | NONE
    values: frozenset[str] = frozenset()
    @classmethod
    def any(cls) -> "PurposeConstraint": ...
    @classmethod
    def none(cls) -> "PurposeConstraint": ...
    @classmethod
    def set(cls, values: Iterable[str]) -> "PurposeConstraint": ...
    def permits(self, purpose: Optional[str]) -> bool: ...
    def is_subset_of(self, parent: "PurposeConstraint") -> bool: ...

# EffectPlan — V4-09.01. Computed OUTSIDE transactions; applied by the
# coordinator inside one fenced transaction.
@dataclass(frozen=True)
class EffectPlan:
    operation_id: str            # idempotency key
    scope_id: str
    producer_id: str             # producer manifest id
    input_digests: tuple[str, ...]
    expected_revisions: dict[str, int]    # object_id -> revision
    epoch_vector: dict[str, int]          # scope_id -> policy epoch
    effects: tuple[Effect, ...]           # ordered domain effects
    follow_ups: tuple[JobRequest, ...]    # jobs to enqueue atomically
    created_us: int
    deadline_us: int

# Permits — V4-08/09/12. In-process leases bind caller+op+epochs+lifetime;
# serialized signed forms only when crossing an untrusted boundary.
@dataclass(frozen=True)
class EligibilityLease: ...      # resolve_access output
@dataclass(frozen=True)
class VerifiedSlice: ...         # read_verified output: bytes+digest+provenance
@dataclass(frozen=True)
class DeliveryPermit: ...        # seal_delivery output; max age 1s default
@dataclass(frozen=True)
class DispatchPermit: ...        # open_dispatch output; one-use, digest-bound
@dataclass(frozen=True)
class ClosureRun: ...            # resumable deletion closure (§38)
@dataclass(frozen=True)
class ReadinessObligation: ...   # per-receipt capability DAG row (§14)
```

## Module map and ownership (one owner per file — do not cross boundaries)

| Directory / file | Owns | Key spec sections |
| --- | --- | --- |
| `verbatim/kernel/` | resolve_access, read_verified, derive_inputs, seal_delivery, open_dispatch, invalidate | §08, V4-08.* |
| `verbatim/jobs/coordinator.py` | EffectPlan application, fencing tokens, receipts | §09, §42 |
| `verbatim/privacy/broker.py` | transport egress broker, dispatch permits, budget reservations | §12 |
| `verbatim/privacy/closure.py` | closure_runs/closure_frontier resumable deletion | §38 |
| `verbatim/storage/resolver.py` | THE profile store resolver (one policy) | §06, F4-18, V4-07.10 |
| `verbatim/storage/schema_v4.py` | SCHEMA_VERSION=4 DDL + 3→4 migration | §41 |
| `verbatim/storage/repos_v4.py` | v4 table allowlist CRUD | §41 |
| `verbatim/readiness/` | readiness_obligations DAG, wait_ready | §14 |
| `verbatim/retrieval/v4/` | EligibilitySet, scoped BM25, streaming dense, fusion, verdicts, packs, cache | §26–§33 |
| `verbatim/synthesis/` | opt-in grounded views, view_support | §20 |
| `verbatim/profiles/` | profile topics/entries, perspective packs | §21 |
| `verbatim/projections/` | Markdown projection + portable bundles | §47 |
| `verbatim/connectors/` | importer framework, dry-run, cursor ledger | §48 |
| `verbatim/workbench/` | local authenticated loopback workbench | §49 |
| `verbatim/service/` | optional HTTP transport (M4) | §44–§45 |
| `eval/v4/` | conformance suite C01–C93, ledger, manifests | §53–§61 |
| `eval/v45/` | M5 experiment harness, D01–D24 | SPEC_V4_5 |

## Non-negotiable rules for every worker

1. No second Engine, store resolver, grant evaluator, trust score, or packer.
   `VerbatimV3`/`Engine` compatibility surfaces delegate; they never re-implement.
2. Fail closed: a lookup/integrity/quarantine error withholds content; it never
   releases on exception. `except: pass` on an enforcement path is a defect.
3. Every safety fix needs a durable public-surface regression test that fails
   without the fix (run it before/after when feasible).
4. Keep all existing tests green unless a test encodes the unsafe contract —
   then fix the test and record the compat note in the file docstring.
5. No network on import, no downloads, no credential material, no new deps
   without an owner-visible reason pinned in pyproject optional extras.
6. New durable kinds/tables register access, provenance, serializer,
   invalidator, eraser, integrity checks BEFORE first write (V4-19.09).
7. Unreachable later-milestone capability denies explicitly
   (`CAPABILITY_UNAVAILABLE`); it is never falsely verified.
