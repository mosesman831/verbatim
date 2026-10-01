"""V5 durable source-projection jobs (docs/v5_contracts.md §5, SPEC_V5 §08).

Handlers registered through ``v5_handlers.V5_KIND_HANDLERS`` (merged into
the ingest dispatch by the integration session):

* ``source_project`` → :func:`handle_source_project` — strict-decode the
  canonical revision payload, normalize under ``norm/v1``, and publish
  ``source_lexical_projection`` + ``source_fts`` carrier pair +
  ``entity_postings`` + ``enrichment`` inside ONE generation-fenced
  transaction; dedup linking (``link_exact``/``link_near``) and advisory
  ``update_candidates`` detection ride the same commit (V5-30).
* ``source_embed`` → :func:`handle_source_embed` — encode the verified
  payload with the configured encoder and publish ``source_vectors``,
  gated by ``Float32Codec.validate_blob``.
* ``source_backfill`` → :func:`handle_source_backfill` — bounded,
  resumable scan over ``backfill_cursor``: adopts missing ``source_state``
  (governed ``recorded``/unresolved adoption, never inferred approval),
  applies the same publication predicate per source, and advances the
  durable cursor in the same transaction as the batch it describes
  (V5-08.17/08.18). A continuation job is enqueued in-commit while work
  remains; crash-before-commit leaves the cursor untouched so the next
  attempt re-scans the same window — never a skipped batch.

Publication predicate (V5-07.13) — every handler re-verifies, inside the
fenced commit transaction:

* the source revision exists and its bytes verify (``SourcesRepo.payload``
  re-checks ``payload_hmac``; ``None`` → EVIDENCE_UNAVAILABLE),
* ``source_state`` is present/live and matches a pinned
  ``control_version`` when the job carried one
  (``transitions.assert_publishable``; missing state is adopted as
  ``recorded``/unresolved — governed adoption, never inferred approval),
* no covering quarantine (``_source_held``) and no suppression/erasure
  (``_source_suppressed``).

Readiness (V5-08.16): each single-source handler fulfills ONLY its own
capability — ``source_lexical_ready`` for ``source_project``,
``source_vector_ready`` for ``source_embed`` — after ``screened`` is
fulfilled by the same drain-time gate evidence. ``try_fulfill`` is used
throughout: bookkeeping disagreements never roll back the domain commit,
and obligations are only materialized where durable evidence exists
(``ensure_for_source`` never fabricates a source branch with no job-row
evidence). ``source_backfill`` settles per-source obligations it actually
delivered and records its own batch receipt (job event + cursor row).

Payload conventions (contract §5): ``{source_id, revision, namespace,
scope_id, generation, producer}`` plus ``utf8_ok: true`` on
``source_project``; ``control_version`` pins the CAS fence when present.
``source_backfill`` refs: ``{job_key, namespace, batch_size, embed}``.

Namespace authority: ``source_state.namespace`` is authoritative when a
control row exists (contracts §3 — the CAS artifact binds the partition);
otherwise the persisted ``sources.scope_id`` is the partition (the same
fallback ``dedup.links._namespace_members`` applies). A payload
``namespace`` that disagrees with the resolved authority fails VALIDATION
rather than writing into the wrong partition.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from ..core.time import now_us, rfc3339
from ..core.types import (
    ErrorCode,
    JobKind,
    JobState,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)
from ..core.types_v4 import CapabilityName
from ..embeddings import matrix as _matrix
from ..embeddings.codec import Float32Codec
from ..readiness import receipt_ids_for_source
from ..enrichment import (
    ENRICHMENT_VERSION,
    classify_type,
    extract_entities,
    extract_identifiers,
    normalize_text,
    normalized_digest,
    parse_temporal,
    polarity,
    shingle_signature,
)
from ..sourcestate import state as _state
from ..sourcestate import transitions
from ..storage import repos_v5
from ..storage.repos import EventsRepo, has_table

try:  # V7 unit projection (§06/§30) — additive; absent → skipped honestly
    from .units_jobs import project_source_v7 as _project_source_v7
except Exception:  # pragma: no cover - absent on partial checkouts
    _project_source_v7 = None

PRODUCER_PROJECT = "source_project/v1"
PRODUCER_EMBED = "source_embed/v1"
PRODUCER_BACKFILL = "source_backfill/v1"

#: Default and ceiling for one backfill batch (V5-08.17 — bounded work per
#: leased job; the cursor + continuation job carry progress).
DEFAULT_BACKFILL_BATCH = 16
MAX_BACKFILL_BATCH = 64

#: Safety bound on revisions projected per source in one backfill batch —
#: a pathological history stays honest (disposition ``revision_cap``)
#: instead of blowing the bounded-work contract.
MAX_REVISIONS_PER_SOURCE = 64

#: Vector-row digest: blake2b-128 over ``encoder_id \x00 blob`` — a
#: recomputable content digest of what was projected, domain-separated
#: from the normalization digest (``verbatim-n1``) and the feature hash
#: (``verbatim-h1``). The payload_hmac binding lives on the revision row;
#: this digest binds blob + encoder identity so corruption/mixing is
#: detectable on read.
_VECTOR_DIGEST_PERSON = b"verbatim-sv"


# ----------------------------------------------------------------------
# ref validation (same discipline as security/handlers.py)
# ----------------------------------------------------------------------


def _require_ref(refs: dict[str, Any], key: str) -> str:
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


def _opt_int(refs: dict[str, Any], key: str) -> Optional[int]:
    value = refs.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"job input_refs.{key} must be an integer when present",
        )
    return value


def _opt_str(refs: dict[str, Any], key: str) -> Optional[str]:
    value = refs.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"job input_refs.{key} must be a non-empty string when present",
        )
    return value


#: Every v5 table the handlers touch — a partially provisioned store
#: fails SCHEMA_UNSUPPORTED up front rather than crashing mid-commit.
_V5_TABLES = (
    "source_state",
    "source_lexical_projection",
    "source_fts_rows",
    "source_fts",
    "source_vectors",
    "entity_postings",
    "duplicate_links",
    "enrichment",
    "update_candidates",
    "backfill_cursor",
)


def _require_v5(ingester: Any) -> None:
    """The v5 projection plane must be provisioned — never a silent no-op."""
    missing = [t for t in _V5_TABLES if not ingester._has(t)]
    if missing:
        raise VerbatimError(
            ErrorCode.SCHEMA_UNSUPPORTED,
            f"v5 source-projection tables absent: {missing}",
        )


# ----------------------------------------------------------------------
# canonical payload + publication predicate
# ----------------------------------------------------------------------


def _verified_text(ingester: Any, source_id: str, revision: int) -> str:
    """Canonical bytes → strict UTF-8 text (V4-13.11/13.12 for the v5 lane).

    ``SourcesRepo.payload`` re-verifies ``payload_hmac`` on every read —
    corrupted bytes raise STORE_CORRUPT, a missing/purged revision returns
    ``None`` (EVIDENCE_UNAVAILABLE), and malformed UTF-8 is a typed
    VALIDATION failure: replacement decoding would index text that was
    never accepted.
    """
    payload = ingester.sources.payload(source_id, revision)
    if payload is None:
        raise VerbatimError(
            ErrorCode.EVIDENCE_UNAVAILABLE,
            f"source {source_id!r} rev {revision} unavailable",
        )
    try:
        return bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"source {source_id!r} rev {revision} payload is not valid "
            "UTF-8 — projection refuses replacement decoding",
        ) from exc


def _resolve_namespace(
    conn: Any, source_id: str, payload_ns: Optional[str], state: Any
) -> str:
    """Namespace authority: ``source_state.namespace`` when the control
    artifact binds this source, else the persisted ``sources.scope_id``
    partition. A pinned payload namespace that disagrees fails closed."""
    state_ns = getattr(state, "namespace", None) if state is not None else None
    if state_ns:
        resolved = state_ns
    else:
        row = conn.execute(
            "SELECT scope_id FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        resolved = row[0] if row is not None else None
    if resolved is None:
        raise VerbatimError(
            ErrorCode.EVIDENCE_UNAVAILABLE,
            f"source {source_id!r} has no namespace authority",
        )
    if payload_ns is not None and payload_ns != resolved:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"job namespace {payload_ns!r} disagrees with source "
            f"{source_id!r} authority {resolved!r}",
        )
    return resolved


def _publication_gate(
    conn: Any,
    ingester: Any,
    source_id: str,
    revision: int,
    *,
    expected_control_version: Optional[int],
    namespace_hint: Optional[str],
    store: Any = None,
) -> tuple:
    """The V5-07.13 predicate inside the caller's fenced commit.

    Returns ``(state, namespace)``. Missing ``source_state`` is adopted
    through the governed ``adopt`` path (disposition ``recorded``, head
    ``unresolved`` — V5-08.18 keeps ambiguous legacy lifecycle explicitly
    unresolved rather than fabricating approval); ``assert_publishable``
    then fences erasure and control-version drift. Quarantine and
    suppression re-checks ride the same transaction so a hold/erasure
    landing between enqueue and commit still wins (V4-42.03).
    """
    row = conn.execute(
        "SELECT disposition, control_version FROM source_state"
        " WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        # Governed adoption for pre-v5 captures: namespace binds to the
        # persisted partition (or the caller's hint when no row exists
        # yet — fail-closed resolution happens next).
        ns = namespace_hint
        if not ns:
            srow = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            ns = srow[0] if srow is not None else None
        if not ns:
            raise VerbatimError(
                ErrorCode.EVIDENCE_UNAVAILABLE,
                f"source {source_id!r} has no namespace authority to adopt under",
            )
        _state.adopt(conn, source_id, ns, producer=PRODUCER_PROJECT, store=store)
        # The pinned CAS target still applies: a job minted against a
        # specific control version cannot publish under an adopted
        # version-0 row (the pin claims a fence that never existed).
        state = transitions.assert_publishable(
            conn,
            source_id,
            expected_control_version=expected_control_version,
        )
    else:
        state = transitions.assert_publishable(
            conn,
            source_id,
            expected_control_version=expected_control_version,
        )
    if ingester._source_held(conn, source_id, revision):
        raise VerbatimError(
            ErrorCode.QUARANTINED,
            f"source {source_id!r} rev {revision} is quarantine-held",
        )
    if ingester._source_suppressed(conn, source_id, revision):
        raise VerbatimError(
            ErrorCode.EVIDENCE_UNAVAILABLE,
            f"source {source_id!r} rev {revision} is suppressed/erased",
        )
    namespace = _resolve_namespace(conn, source_id, namespace_hint, state)
    return state, namespace


def _projection_generation(conn: Any, ingester: Any, pinned: Optional[int]) -> int:
    """The generation the row is written under.

    Rows always carry the *current* committed projection generation —
    the value readers filter on. A payload-pinned generation newer than
    the store's means the job was minted under a generation this store
    has not reached (restore/fork skew): STALE_DEPENDENCY, retryable.
    """
    current = int(
        ingester.store._meta_get(conn, "projection_generation") or 0
    )
    if pinned is not None and pinned > current:
        raise VerbatimError(
            ErrorCode.STALE_DEPENDENCY,
            f"job pins projection generation {pinned}; store is at {current}",
            retryable=True,
        )
    return current


# ----------------------------------------------------------------------
# derived-row writers (all inside the caller's fenced tx)
# ----------------------------------------------------------------------


def _write_fts_pair(
    conn: Any,
    source_id: str,
    revision: int,
    namespace: str,
    generation: int,
    tokens: str,
) -> None:
    """Replace the external-content FTS pair delete-then-insert.

    ``source_fts_idx`` is maintained by triggers over ``source_fts``;
    reprojection MUST delete the old carrier+content before inserting so
    no stale terms survive (schema_v5 docstring: REPLACE on the carrier
    mints a new rowid without firing the delete trigger). The plain
    carrier/content tables are written unconditionally — on a store
    without FTS5 the pair stays consistent and the index lane reports
    itself unavailable at read, never half-written.
    """
    conn.execute(
        "DELETE FROM source_fts WHERE fts_row_id IN"
        " (SELECT row_id FROM source_fts_rows"
        "  WHERE source_id = ? AND revision = ?)",
        (source_id, revision),
    )
    conn.execute(
        "DELETE FROM source_fts_rows WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    )
    cur = conn.execute(
        "INSERT INTO source_fts_rows (source_id, revision, scope_id,"
        " generation) VALUES (?, ?, ?, ?)",
        (source_id, revision, namespace, generation),
    )
    conn.execute(
        "INSERT INTO source_fts (fts_row_id, text) VALUES (?, ?)",
        (cur.lastrowid, tokens),
    )


def _write_lexical(
    conn: Any,
    *,
    source_id: str,
    revision: int,
    namespace: str,
    generation: int,
    tokens_text: str,
    doc_len: int,
    digest: str,
) -> None:
    repos_v5.upsert(
        conn,
        "source_lexical_projection",
        {
            "source_id": source_id,
            "revision": revision,
            "scope_id": namespace,
            "generation": generation,
            "tokens": tokens_text,
            "doc_len": doc_len,
            "digest": digest,
        },
    )
    _write_fts_pair(conn, source_id, revision, namespace, generation, tokens_text)


def _write_postings(
    conn: Any,
    *,
    namespace: str,
    source_id: str,
    revision: int,
    generation: int,
    identifiers: list,
    entities: list,
) -> int:
    """Identifier + entity mentions with UTF-8 byte offsets (V5-30.17 —
    case preserved, exact match only). ``offsets`` is the producer-
    serialized ``[[start, end], ...]`` list; PK groups by entity value so
    repeated mentions merge under one row."""
    conn.execute(
        "DELETE FROM entity_postings WHERE source_id = ? AND revision = ?",
        (source_id, revision),
    )
    grouped: dict = {}
    for ident in identifiers:
        grouped.setdefault(ident.value, []).append(
            ("identifier", ident.kind, ident.start, ident.end)
        )
    for ent in entities:
        grouped.setdefault(ent.value, []).append(
            ("entity", ent.kind, ent.start, ent.end)
        )
    for value in sorted(grouped):
        mentions = grouped[value]
        spans = sorted({(m[2], m[3]) for m in mentions})
        # entity_kind carries the producer's class + sub-kind labels; a
        # value seen as both identifier and entity joins them with '+'.
        kind = "+".join(sorted({f"{m[0]}:{m[1]}" for m in mentions}))
        repos_v5.upsert(
            conn,
            "entity_postings",
            {
                "namespace": namespace,
                "entity": value,
                "entity_kind": kind,
                "source_id": source_id,
                "revision": revision,
                "offsets": json_dumps([list(s) for s in spans]),
                "generation": generation,
            },
        )
    return len(grouped)


def _write_enrichment(
    conn: Any,
    *,
    source_id: str,
    revision: int,
    mem_type: str,
    pol: str,
    temporal: Any,
    identifiers: list,
    entities: list,
) -> None:
    repos_v5.upsert(
        conn,
        "enrichment",
        {
            "source_id": source_id,
            "revision": revision,
            "producer": ENRICHMENT_VERSION,
            "type": mem_type,
            "polarity": pol,
            "time_precision": temporal.precision,
            "time_status": temporal.status,
            "event_at": temporal.event_at or None,
            "anchor_at": temporal.anchor_at or None,
            "fields_json": json_dumps(
                {
                    "identifiers": [i.to_dict() for i in identifiers],
                    "entities": [e.to_dict() for e in entities],
                    "temporal": temporal.to_dict(),
                    "producer": ENRICHMENT_VERSION,
                }
            ),
        },
    )


def _run_dedup(
    conn: Any,
    *,
    source_id: str,
    revision: int,
    namespace: str,
    payload_hmac_hex: Optional[str],
    digest: str,
    signature: frozenset,
    text_fields: Any = None,
    store: Any = None,
    near_plan: Any = None,
) -> dict[str, Any]:
    """Duplicate-group linking (V5-30.05/06/07) — additive derived data.

    ``exact_digest`` binds byte-identical payloads via ``payload_hmac``;
    ``normalized`` binds the projection digest through the guard rules;
    ``minhash`` scores the shingle signature against the namespace's
    recent projections. Per-namespace ``dedupe`` policy lives in
    ``dedup.links`` — a ``none`` policy records its reason, never links.

    ``near_plan`` — an advisory :func:`dedup.links.plan_near` result
    computed on a read snapshot whose inputs the caller has already
    verified unchanged (``_deps_fingerprint``). ``commit_near`` still
    re-validates policy/membership/liveness on this snapshot and falls
    back to the fused ``link_near`` when the plan is stale.
    """
    from ..dedup import links

    outcomes: dict[str, Any] = {}
    if payload_hmac_hex:
        out = links.link_exact(
            conn,
            source_id=source_id,
            revision=revision,
            namespace=namespace,
            digest=payload_hmac_hex,
            method=links.METHOD_EXACT,
        )
        outcomes["exact_digest"] = out.to_dict()
    out = links.link_exact(
        conn,
        source_id=source_id,
        revision=revision,
        namespace=namespace,
        digest=digest,
        method=links.METHOD_NORMALIZED,
    )
    outcomes["normalized"] = out.to_dict()
    if near_plan is not None:
        out = links.commit_near(
            conn, near_plan, source_id=source_id, revision=revision
        )
        if out is not None:
            outcomes["minhash"] = out.to_dict()
            return outcomes
    out = links.link_near(
        conn,
        source_id=source_id,
        revision=revision,
        namespace=namespace,
        signature=signature,
        text_fields=text_fields,
        store=store,
    )
    outcomes["minhash"] = out.to_dict()
    return outcomes


# ----------------------------------------------------------------------
# advisory prescan — dedup/update scans on a WAL snapshot
# ----------------------------------------------------------------------
#
# ``link_near`` and ``detect_update_candidates`` scan the whole live
# namespace (~20 ms at a few hundred sources) while the fenced commit
# holds the write lock, inflating writer hold time and every other
# writer's wait — including the recall path's journal writes. The scans
# are pure functions of committed state plus this job's own derived
# fields, so the bulk of the work runs ahead of the transaction on a
# WAL snapshot. Inside the commit, ``_deps_fingerprint`` — an exact
# content fold of every table those scans read — proves the inputs are
# unchanged; ``commit_near``/``persist_update_candidates`` then write
# the pre-computed results, and any drift falls back to the fused
# in-transaction calls, which see the serial-order state.

#: Dependency tables watched by the prescan fingerprint. ``meta`` is
#: covered through its own policy-key trigger (below).
_CV_TABLES: tuple = (
    "source_state",
    "source_revisions",
    "enrichment",
    "sources",
    "source_lexical_projection",
)

_CV_FLAG_ATTR = "_v5_cv_triggers"


def _cv_bump_sql(table: str) -> str:
    return (
        "INSERT INTO meta(key, value_json) VALUES('cv:" + table + "', '1')"
        " ON CONFLICT(key) DO UPDATE SET value_json ="
        " CAST(COALESCE(CAST(meta.value_json AS INTEGER), 0) + 1 AS TEXT)"
    )


def _cv_trigger_sql() -> list:
    """DDL for the per-table change counters. One ``meta`` upsert per
    write statement — ~tens of microseconds — buys an O(1) staleness
    check that is airtight: every INSERT/UPDATE/DELETE on a watched
    table bumps its counter, so counter equality at prescan and commit
    proves the scans' inputs are byte-for-byte unchanged."""
    stmts: list = []
    for t in _CV_TABLES:
        for ev in ("INSERT", "UPDATE", "DELETE"):
            stmts.append(
                f"CREATE TRIGGER IF NOT EXISTS v5_cv_{t}_{ev[0].lower()}"
                f" AFTER {ev} ON {t} BEGIN {_cv_bump_sql(t)}; END"
            )
    # The dedupe policy lives in ``meta`` — counter fires only on
    # policy keys so unrelated meta writes never churn the fingerprint.
    stmts.append(
        "CREATE TRIGGER IF NOT EXISTS v5_cv_meta_i AFTER INSERT ON meta"
        " WHEN NEW.key LIKE 'dedupe_policy:%'"
        f" BEGIN {_cv_bump_sql('meta')}; END"
    )
    stmts.append(
        "CREATE TRIGGER IF NOT EXISTS v5_cv_meta_u AFTER UPDATE ON meta"
        " WHEN NEW.key LIKE 'dedupe_policy:%' OR OLD.key LIKE 'dedupe_policy:%'"
        f" BEGIN {_cv_bump_sql('meta')}; END"
    )
    stmts.append(
        "CREATE TRIGGER IF NOT EXISTS v5_cv_meta_d AFTER DELETE ON meta"
        " WHEN OLD.key LIKE 'dedupe_policy:%'"
        f" BEGIN {_cv_bump_sql('meta')}; END"
    )
    return stmts


def _ensure_cv_triggers(store: Any) -> bool:
    """Install the change-counter triggers once per store object.
    Idempotent DDL inside one write tx; ``False`` on any failure
    (read-only store, missing tables) — the caller then uses the
    fold fingerprint instead."""
    if getattr(store, _CV_FLAG_ATTR, False):
        return True
    try:
        with store.tx() as conn:
            for sql in _cv_trigger_sql():
                conn.execute(sql)
        try:
            setattr(store, _CV_FLAG_ATTR, True)
        except Exception:
            pass
        return True
    except Exception:
        return False


_DEPS_FP_SQL: tuple = (
    # source_state — namespace membership, liveness, mutation head,
    # control version (detect scan + member resolution). Short deciding
    # columns are folded verbatim so a same-length in-place UPDATE
    # (``disposition`` 'active'→'erased') still flips the fingerprint.
    "SELECT group_concat(q, '') FROM ("
    " SELECT quote(source_id)||'|'||quote(namespace)||'|'||quote(disposition)"
    " ||'|'||quote(mutation_head)||'|'||quote(control_version)"
    " ||'|'||quote(superseded_by) AS q"
    " FROM source_state ORDER BY source_id)",
    # source_revisions — payload presence/hmac (liveness, resolved-view
    # cache keys) + provenance (member facts). ``payload_hmac`` binds the
    # bytes; ``length(payload)`` catches the purge write (payload X'').
    "SELECT group_concat(q, '') FROM ("
    " SELECT quote(source_id)||'|'||quote(revision)||'|'||quote(length(payload))"
    " ||'|'||quote(hex(payload_hmac))||'|'||quote(provenance) AS q"
    " FROM source_revisions ORDER BY source_id, revision)",
    # enrichment — the guard-field / resolved-prior inputs. Every writer
    # goes through ``repos_v5.upsert`` (INSERT OR REPLACE), so ``rowid``
    # binds rewrites and ``length(fields_json)`` completes it — no
    # in-place fields_json update exists or may be added without moving
    # it into this fold.
    "SELECT group_concat(q, '') FROM ("
    " SELECT quote(rowid)||'|'||quote(source_id)||'|'||quote(revision)"
    " ||'|'||quote(producer)||'|'||quote(type)||'|'||quote(polarity)"
    " ||'|'||quote(event_at)||'|'||quote(anchor_at)"
    " ||'|'||quote(length(fields_json)) AS q"
    " FROM enrichment ORDER BY source_id, revision, producer)",
    # sources — scope membership + created_us ordering; speaker_id is
    # folded verbatim (purge NULLs it in place).
    "SELECT group_concat(q, '') FROM ("
    " SELECT quote(source_id)||'|'||quote(scope_id)||'|'||quote(created_us)"
    " ||'|'||quote(origin)||'|'||quote(speaker_id) AS q"
    " FROM sources ORDER BY source_id)",
    # source_lexical_projection — the near-scan rows. ``digest`` is the
    # persisted normalized-content digest (binds ``tokens`` — the same
    # identity the signature memo keys on); ``rowid`` catches REPLACE
    # rewrites that re-derive identical content.
    "SELECT group_concat(q, '') FROM ("
    " SELECT quote(rowid)||'|'||quote(source_id)||'|'||quote(revision)"
    " ||'|'||quote(digest)||'|'||quote(scope_id)||'|'||quote(generation)"
    " ||'|'||quote(doc_len) AS q"
    " FROM source_lexical_projection ORDER BY source_id, revision)",
    # meta — the per-namespace dedupe policy rows only (generation
    # counters and unrelated keys would churn the fingerprint uselessly)
    "SELECT group_concat(q, '') FROM ("
    " SELECT quote(key)||'|'||quote(value_json) AS q"
    " FROM meta WHERE key LIKE 'dedupe_policy:%' ORDER BY key)",
)


def _deps_fingerprint(conn: Any, mode: str = "fold") -> Optional[tuple]:
    """Staleness fingerprint over every input the dedup/update scans
    read; ``None`` when unprobeable — callers treat that as "no
    prescan", never as a match.

    ``counter`` mode — the trigger-maintained change counters: equality
    proves no INSERT/UPDATE/DELETE touched a dependency table between
    the two reads. O(1) and airtight.

    ``fold`` mode — ``group_concat`` over PK-ordered ``quote()``d
    deciding columns: an equal fingerprint means the deciding columns
    are byte-for-byte identical (big payload/JSON columns fold through
    ``rowid``/``length``/content digests — every writer of those tables
    is INSERT OR REPLACE, and the fold comments pin that invariant).
    """
    if mode == "counter":
        try:
            row = conn.execute(
                "SELECT group_concat(q, '') FROM ("
                " SELECT quote(key)||'|'||quote(value_json) AS q"
                " FROM meta WHERE key LIKE 'cv:%' ORDER BY key)"
            ).fetchone()
            return ("counter", row[0] if row else None)
        except Exception:
            return None
    parts: list = []
    try:
        for sql in _DEPS_FP_SQL:
            row = conn.execute(sql).fetchone()
            parts.append(row[0] if row else None)
    except Exception:
        return None
    return ("fold", tuple(parts))


def _namespace_peek(
    conn: Any, source_id: str, payload_ns: Optional[str]
) -> Optional[str]:
    """Read-only twin of the ``_publication_gate``/``_resolve_namespace``
    outcome — state namespace, else the adoption target (hint, then
    ``sources.scope_id``), else the persisted scope. ``None`` when the
    gate would raise rather than resolve, so the prescan simply skips.
    """
    st = conn.execute(
        "SELECT namespace FROM source_state WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    resolved: Optional[str] = None
    if st is not None and st[0]:
        resolved = st[0]
    else:
        if st is None:
            # Adoption binds the caller's hint first (gate behavior).
            resolved = payload_ns
        if not resolved:
            srow = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            resolved = srow[0] if srow is not None else None
    if resolved is None:
        return None
    if payload_ns is not None and payload_ns != resolved:
        return None  # the gate raises VALIDATION — no plan to reuse
    return resolved


def _prescan_dedup_fields(derived: dict[str, Any]) -> Any:
    """The ``DedupFields`` ``links._fields_for`` returns once this job's
    own enrichment + projection rows exist — reconstructed from
    ``derived`` so the prescan's guard evaluation sees the same fields
    the in-transaction read would (``_merge_fields`` order preserved:
    enrichment row first, token recompute as fill).
    """
    from ..dedup.links import (
        _merge_fields,
        fields_from_enrichment,
        fields_from_text,
    )

    row = {
        "type": derived["mem_type"],
        "polarity": derived["polarity"],
        "event_at": derived["temporal"].event_at or None,
        "fields_json": json_dumps(
            {
                "identifiers": [
                    i.to_dict() for i in derived["identifiers"]
                ],
                "entities": [e.to_dict() for e in derived["entities"]],
                "temporal": derived["temporal"].to_dict(),
                "producer": ENRICHMENT_VERSION,
            }
        ),
    }
    return _merge_fields(
        fields_from_enrichment(row),
        fields_from_text(derived["tokens_text"]),
    )


class _PreScan:
    """The advisory read-phase bundle: input fingerprint (+ its mode),
    resolved namespace, the ``link_near`` plan, and the update
    candidates."""

    __slots__ = ("fp", "mode", "namespace", "near", "cands")

    def __init__(self, fp, mode, namespace, near, cands) -> None:
        self.fp = fp
        self.mode = mode
        self.namespace = namespace
        self.near = near
        self.cands = cands


class _PlanStale(Exception):
    """Internal control flow — never escapes ``_project_body``.

    Raised inside the fenced commit when the prescan's dependency
    fingerprint drifted between the snapshot read and this transaction:
    the commit aborts (a rollback — never partial state), the prescan is
    recomputed on a fresh snapshot, and the whole fenced commit retries.
    The point is the *location* of the namespace scans: the fused
    ``link_near``/``detect_update_candidates`` fallback reads the whole
    live namespace while holding the WAL write lock (measured
    200-400 ms+ under churn — past every other writer's 250 ms busy
    cap), whereas the plan path commits precomputed rows in ~ms.
    Retrying keeps the scans on snapshot reads exactly when they cost
    most; once the bound is spent the commit falls back to the fused
    path — the same honest bound the code always had, now reached only
    after several verified-stale attempts instead of on the first.
    """


#: Planned-commit retries per execution before the fused tail: each
#: stale abort costs one rolled-back commit plus one unlocked snapshot
#: prescan (~hundreds of ms on a busy namespace — cheap next to the
#: starvation a fused scan causes). Bounded so pathological churn still
#: reaches the tail; ordinary contention resolves in one.
_PLAN_STALE_RETRIES = 4

#: The retry budget exists to keep an *expensive* fused namespace scan
#: off the WAL write lock. When the prescan measured ~ms the fused
#: fallback costs the same ~ms under the lock — far below the 250 ms
#: busy cap — and replanning would only delay completion; below this
#: floor a stale plan fuses immediately, exactly as before.
_REPLAN_MIN_SCAN_MS = 80.0


def _prescan(
    ingester: Any,
    *,
    source_id: str,
    revision: int,
    payload_ns: Optional[str],
    derived: dict[str, Any],
    text: str,
) -> Optional[_PreScan]:
    """Run the dedup/update-detection scans on a WAL snapshot.

    Advisory only: any failure returns ``None`` and the commit runs the
    fused in-transaction calls, which raise whatever the prescan would
    have — identical externally visible behavior either way.
    """
    store = ingester.store
    if store is None:
        return None
    try:
        from ..dedup import links
        from ..querying.updates import (
            NewRecord,
            plan_update_candidates,
        )

        mode = "counter" if _ensure_cv_triggers(store) else "fold"
        with store.read() as rconn:
            fp = _deps_fingerprint(rconn, mode)
            if fp is None:
                return None
            namespace = _namespace_peek(rconn, source_id, payload_ns)
            if namespace is None:
                return None
            near = links.plan_near(
                rconn,
                source_id=source_id,
                revision=revision,
                namespace=namespace,
                signature=derived["signature"],
                text_fields=_prescan_dedup_fields(derived),
                store=store,
            )
            rec = NewRecord(
                source_id=source_id,
                revision=revision,
                text=text,
                type=derived["mem_type"],
                polarity=derived["polarity"],
                identifiers=tuple(
                    i.to_dict() for i in derived["identifiers"]
                ),
                entities=tuple(
                    e.to_dict() for e in derived["entities"]
                ),
                event_at=derived["temporal"].event_at or None,
                anchor_at=derived["temporal"].anchor_at or None,
            )
            cands = plan_update_candidates(
                rconn, namespace, rec, store=store
            )
            return _PreScan(fp, mode, namespace, near, cands)
    except Exception:
        return None


class _BackfillPre:
    """Batch prescan bundle — the same advisory contract as
    :class:`_PreScan` but for one ``source_backfill`` batch: the input
    fingerprint, each source's predicted ``resolved_ns`` (state
    namespace or the adopt target), and per-revision
    ``plan_near``/update-candidate plans keyed ``(source_id, revision)``.
    """

    __slots__ = ("fp", "mode", "ns", "plans")

    def __init__(self, fp, mode, ns, plans) -> None:
        self.fp = fp
        self.mode = mode
        self.ns = ns
        self.plans = plans


def _backfill_prescan(
    ingester: Any,
    staged: list,
    job_namespace: Optional[str],
) -> Optional[_BackfillPre]:
    """Dedup/update plans for a staged backfill batch on a WAL snapshot.

    The commit resolves ``resolved_ns`` per source from ``source_state``
    or the adopt target (``sources.scope_id``, else the job's namespace
    filter) — this predicts that same value so the commit can compare;
    a source whose in-transaction resolution disagrees simply does not
    use its plan (the fingerprint gates *staleness*, the namespace
    equality gates *applicability* — never correctness). Advisory only:
    any failure returns ``None`` and the commit runs the fused
    per-revision scans exactly as before.
    """
    store = ingester.store
    if store is None or not staged:
        return None
    try:
        from ..dedup import links
        from ..querying.updates import (
            NewRecord,
            plan_update_candidates,
        )

        mode = "counter" if _ensure_cv_triggers(store) else "fold"
        with store.read() as rconn:
            fp = _deps_fingerprint(rconn, mode)
            if fp is None:
                return None
            ns_map: dict[str, str] = {}
            plans: dict = {}
            # Earlier same-namespace revisions projected by THIS batch
            # become extra ``link_near`` candidates the snapshot scan
            # cannot see (their projection rows are this transaction's
            # own writes). ``update_candidates`` needs no such check:
            # its prior scan requires disposition ``active`` while the
            # batch only ever adopts ``recorded`` and never mutates
            # existing state — the committed prior set is the whole set.
            earlier_sigs: dict[str, list] = {}
            for item in staged:
                sid = item["source_id"]
                row = rconn.execute(
                    "SELECT namespace FROM source_state"
                    " WHERE source_id = ?",
                    (sid,),
                ).fetchone()
                if row is not None:
                    ns = row[0]
                    # A stated source is a namespace member under the
                    # "state" authority — its near plan is consumable.
                    near_plannable = True
                else:
                    srow = rconn.execute(
                        "SELECT scope_id FROM sources WHERE source_id = ?",
                        (sid,),
                    ).fetchone()
                    ns = srow[0] if srow else (job_namespace or "")
                    # The commit adopts this source mid-transaction.
                    # ``_namespace_members`` resolves the "state"
                    # authority whenever the namespace already has
                    # control rows — and an unadopted source is then
                    # ``not_in_namespace``, a plan ``commit_near`` must
                    # discard. Only under the "scope" authority (no
                    # state rows for the namespace yet) is the
                    # to-be-adopted source already a member via
                    # ``sources.scope_id`` and its plan consumable.
                    near_plannable = not bool(
                        ns
                        and rconn.execute(
                            "SELECT 1 FROM source_state"
                            " WHERE namespace = ? LIMIT 1",
                            (ns,),
                        ).fetchone()
                    )
                if job_namespace and ns != job_namespace:
                    continue  # in-tx disposition: namespace_mismatch
                ns_map[sid] = ns
                for r in item["revisions"]:
                    if r.get("text") is None:
                        continue
                    rv = r["revision"]
                    if ingester._source_held(
                        rconn, sid, rv
                    ) or ingester._source_suppressed(rconn, sid, rv):
                        continue  # in-tx skip — planning would be wasted
                    d = r["derived"]
                    new_fields = _prescan_dedup_fields(d)
                    new_sig = frozenset(d["signature"] or ())
                    near = None
                    if near_plannable:
                        near = links.plan_near(
                            rconn,
                            source_id=sid,
                            revision=rv,
                            namespace=ns,
                            signature=d["signature"],
                            text_fields=new_fields,
                            store=store,
                        )
                        # A live intra-batch candidate the plan could
                        # not see: when it could beat the plan's outcome
                        # the fused in-transaction scan is the serial
                        # truth — drop the plan for this revision.
                        if near.out.reason in (
                            "ok",
                            "no_match",
                            "guard_veto",
                        ):
                            for m_sig, m_fields in earlier_sigs.get(
                                ns, ()
                            ):
                                union = new_sig | m_sig
                                score = (
                                    len(new_sig & m_sig) / len(union)
                                    if union
                                    else 0.0
                                )
                                if score < links.DEFAULT_THRESHOLD:
                                    continue
                                if (
                                    links.guard_veto(
                                        new_fields, m_fields
                                    )
                                    is not None
                                ):
                                    continue
                                if near.best is None or (
                                    score >= near.best["score"]
                                ):
                                    near = None
                                    break
                    earlier_sigs.setdefault(ns, []).append(
                        (new_sig, new_fields)
                    )
                    rec = NewRecord(
                        source_id=sid,
                        revision=rv,
                        text=r["text"],
                        type=d["mem_type"],
                        polarity=d["polarity"],
                        identifiers=tuple(
                            i.to_dict() for i in d["identifiers"]
                        ),
                        entities=tuple(
                            e.to_dict() for e in d["entities"]
                        ),
                        event_at=d["temporal"].event_at or None,
                        anchor_at=d["temporal"].anchor_at or None,
                    )
                    cands = plan_update_candidates(
                        rconn, ns, rec, store=store
                    )
                    plans[(sid, rv)] = (near, cands)
            if not plans:
                # Nothing plannable (every revision is purged, held, or
                # namespace-mismatched) — the commit does no scans, so
                # there is no plan to protect and no reason to abort on
                # unrelated drift.
                return None
            return _BackfillPre(fp, mode, ns_map, plans)
    except Exception:
        return None


def _detect_updates(
    conn: Any,
    *,
    namespace: str,
    source_id: str,
    revision: int,
    text: str,
    mem_type: str,
    pol: str,
    temporal: Any,
    identifiers: list,
    entities: list,
    store: Any = None,
) -> list:
    """Advisory ``update_candidates`` (V5-30.19) — never lifecycle."""
    from ..querying.updates import NewRecord, detect_update_candidates

    rec = NewRecord(
        source_id=source_id,
        revision=revision,
        text=text,
        type=mem_type,
        polarity=pol,
        identifiers=tuple(i.to_dict() for i in identifiers),
        entities=tuple(e.to_dict() for e in entities),
        event_at=temporal.event_at or None,
        anchor_at=temporal.anchor_at or None,
    )
    return detect_update_candidates(conn, namespace, rec, store=store)


# ----------------------------------------------------------------------
# readiness settlement (inside the fenced commit)
# ----------------------------------------------------------------------


def _declare_source_cap(
    conn: Any, engine: Any, source_id: str, receipt_id: str, cap: CapabilityName
) -> None:
    """Add the source capability row to a receipt that predates it.

    The durable ``source_*`` job row that dispatched this handler IS the
    evidence the capture owed the capability (the same convention
    ``ReadinessEngine._source_caps_for_source`` applies when
    materializing missing receipts); ``record_obligations`` inserts only
    missing rows, so receipts that already declared the branch are
    untouched and legacy captures converge honestly (V5-08.15). Nothing
    is declared for receipts with no job evidence — the caller only
    reaches here from a leased source job.
    """
    if engine._row(conn, receipt_id, cap) is not None:
        return
    row = conn.execute(
        "SELECT scope_id FROM readiness_obligations WHERE receipt_id = ?"
        " LIMIT 1",
        (receipt_id,),
    ).fetchone()
    scope_id = row[0] if row is not None else None
    if scope_id is None:
        srow = conn.execute(
            "SELECT scope_id FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        scope_id = srow[0] if srow is not None else None
    if scope_id is None:
        return
    engine.record_obligations(
        conn, receipt_id, scope_id, [CapabilityName.SCREENED, cap]
    )


def _settle_source(
    conn: Any,
    ingester: Any,
    source_id: str,
    revision: int,
    capability: CapabilityName,
) -> int:
    """Fulfill ``capability`` on every resolvable receipt for the revision.

    ``ensure_for_source`` materializes obligations only where durable
    evidence exists (job rows / envelopes) — nothing fabricates a source
    branch. ``screened`` fulfills first: reaching this point means the
    drain-time publication gate committed, which is the same evidence the
    claim lane settles on (V4-14.02 convention). ``try_fulfill`` stays
    best-effort — a bookkeeping disagreement never rolls back the domain
    commit. Returns the number of receipt ids touched.
    """
    engine = getattr(ingester, "_readiness", None)
    if engine is None:
        return 0
    rids = engine.ensure_for_source(conn, source_id, revision)
    for rid in rids:
        _declare_source_cap(conn, engine, source_id, rid, capability)
        engine.try_fulfill(conn, rid, CapabilityName.SCREENED)
        engine.try_fulfill(conn, rid, capability)
    return len(rids)


def _defer_source(
    conn: Any,
    ingester: Any,
    source_id: str,
    revision: int,
    capability: CapabilityName,
    reason: str,
) -> int:
    """Terminal ``deferred`` for an undeliverable source capability
    (encoder unprovisioned, FTS unavailable, …) — honest, durable, and
    re-openable by ``pend`` when the capability later exists."""
    engine = getattr(ingester, "_readiness", None)
    if engine is None:
        return 0
    rids = engine.ensure_for_source(conn, source_id, revision)
    for rid in rids:
        _declare_source_cap(conn, engine, source_id, rid, capability)
        engine.try_fulfill(conn, rid, CapabilityName.SCREENED)
        engine.defer(conn, rid, capability, reason, _missing_ok=True)
    return len(rids)


def _fail_source_obligation(
    ingester: Any,
    job: dict[str, Any],
    owner: str,
    source_id: str,
    revision: int,
    capability: CapabilityName,
    code: str,
) -> None:
    """Land a permanent handler error on the source capability itself.

    ``_fail_job_obligations`` (the drain path) treats harvest/admit jobs
    as siblings that may still deliver the stage — true for claim caps,
    but no pipeline job can deliver ``source_lexical_ready`` /
    ``source_vector_ready``. The handler therefore fails the obligation
    directly, in its own fenced tx, before the job's ``failed`` transition
    lands. Best-effort: a lost lease or absent rows never mask the real
    error — the job row still carries the code.
    """
    engine = getattr(ingester, "_readiness", None)
    if engine is None:
        return
    try:
        with ingester.store.tx() as conn:
            ingester.jobs.assert_lease(
                conn, job["job_id"], owner, job["generation"]
            )
            for rid in engine._existing_receipts(
                conn, receipt_ids_for_source(conn, source_id, revision)
            ):
                engine.fail(conn, rid, capability, code, _missing_ok=True)
    except VerbatimError:
        pass  # the job-row failure records the outcome regardless


def _declare_obligations(
    ingester: Any,
    job: dict[str, Any],
    owner: str,
    source_id: str,
    revision: int,
    capability: CapabilityName,
) -> None:
    """Fenced pre-commit that parks the capability row before fallible
    work runs.

    A permanent handler failure must land on the capability AND the
    receipt's ``failed`` indicator (V5-08.16) — that requires the
    obligation row to exist when ``_fail_job_obligations`` runs, not only
    on the success path. Declaration is idempotent (insert-if-missing)
    and fenced by ``assert_lease``, so a superseded worker cannot mint
    obligations for a job it no longer owns.
    """
    engine = getattr(ingester, "_readiness", None)
    if engine is None:
        return
    # Read-precheck on a snapshot: the declare's only effects are
    # receipt materialization (``ensure_for_source``) and missing
    # capability rows (``_declare_source_cap``). When every resolvable
    # receipt already exists AND already carries the capability row the
    # fenced tx is provably a no-op — skipping it costs one read instead
    # of a commit. A receipt/capability can appear between this check
    # and the fenced pass only through another writer, which the lease
    # fence + idempotent inserts absorb exactly as before.
    try:
        with ingester.store.read() as rconn:
            cands = receipt_ids_for_source(rconn, source_id, revision)
            existing = set(engine._existing_receipts(rconn, cands))
            if len(existing) == len(cands) and all(
                engine._row(rconn, rid, capability) is not None
                for rid in cands
            ):
                return
    except Exception:
        pass  # a failed precheck simply runs the fenced declare below
    with ingester.store.tx() as conn:
        ingester.jobs.assert_lease(
            conn, job["job_id"], owner, job["generation"]
        )
        for rid in engine.ensure_for_source(conn, source_id, revision):
            _declare_source_cap(conn, engine, source_id, rid, capability)


def _append_event(
    conn: Any, ingester: Any, job: dict[str, Any], kind: str, fields: dict
) -> int:
    return EventsRepo(ingester.store).append(
        conn,
        job["scope_id"],
        kind,
        "engine",
        fields,
        ingester.policy.policy_version,
    )


# ----------------------------------------------------------------------
# enrichment pipeline (computed outside the write tx — V4-09.09 pattern)
# ----------------------------------------------------------------------


def _anchor_for(rev: dict[str, Any]) -> str:
    """Temporal anchor: explicit ``metadata.event_time`` RFC3339 wins,
    else the revision's event/capture timestamp. Naive anchors are never
    accepted (temporal._parse_anchor demands an explicit offset)."""
    meta = safe_json_loads(rev.get("metadata_json") or "{}")
    if isinstance(meta, dict):
        ev = meta.get("event_time")
        if isinstance(ev, str) and ev.strip():
            try:
                # Validate through the real parser — no home-grown rule.
                from ..enrichment.temporal import _parse_anchor

                _parse_anchor(ev)
                return ev
            except Exception:
                pass
    base = rev.get("event_us") or rev.get("captured_us") or now_us()
    return rfc3339(int(base))


def _derive(text: str, anchor: str) -> dict[str, Any]:
    """All deterministic enrichments for one decoded payload."""
    tokens_text = normalize_text(text)
    token_list = tokens_text.split()
    return {
        "tokens_text": tokens_text,
        "doc_len": len(token_list),
        "digest": normalized_digest(text),
        "signature": shingle_signature(token_list),
        "identifiers": extract_identifiers(text),
        "entities": extract_entities(text),
        "temporal": parse_temporal(text, anchor),
        "polarity": polarity(text).value,
        "mem_type": classify_type(text).value,
    }


def _vector_digest(encoder_id: str, blob: bytes) -> str:
    return hashlib.blake2b(
        encoder_id.encode("utf-8") + b"\x00" + bytes(blob),
        digest_size=16,
        person=_VECTOR_DIGEST_PERSON,
    ).hexdigest()


# ----------------------------------------------------------------------
# dequeue-level coalescing (V8-13.05)
# ----------------------------------------------------------------------
#
# A leased ``source_project``/``source_embed`` job drains its due
# same-kind+same-scope siblings in ONE fenced commit instead of one
# commit per job. Staging (verified payload, derived artifacts, prescan
# plans, encoder inference) happens outside the write tx exactly like
# the leased job's own body; inside the commit each claimed sibling runs
# the queue's own fenced ``_commit_effects`` (lease assert → operation
# receipt replay → apply → receipt → complete) under a per-sibling
# savepoint, so a sibling failure lands exactly like a solo-drain
# failure while the shared commit keeps crash-mid-batch atomic. Job
# *effects* are unchanged — the same writers write the same rows; only
# the transaction count changes (the D8-31 cost).

#: Siblings one leased job may coalesce per commit — bounded so the
#: shared write tx stays well under the WAL writer busy cap.
SOURCE_COALESCE_BATCH = 16
SOURCE_COALESCE_MAX = 64

#: Wall-clock bound (ms) on the shared sibling commit loop
#: (V8-13.05): once the batch has held the write lock this long, the
#: claimed-but-unprocessed tail is released back to ``queued`` so
#: foreground writers are never starved past the busy cap (~250 ms).
#: The leased job's own commit already ran inside the same tx, so the
#: bound stays comfortably below the cap; ``cfg.jobs
#: .source_coalesce_tx_ms`` overrides, the ceiling is a hard safety cap.
SOURCE_COALESCE_TX_MS = 100.0
SOURCE_COALESCE_TX_MS_MAX = 200.0


def _coalesce_limit(ingester: Any) -> int:
    """Sibling batch bound (V8-13.05).

    ``cfg.jobs.source_coalesce`` overrides when the field is provisioned;
    0 disables coalescing entirely; the ceiling caps the shared commit.
    """
    jobs_cfg = getattr(getattr(ingester, "cfg", None), "jobs", None)
    n = getattr(jobs_cfg, "source_coalesce", None)
    if n is None:
        n = SOURCE_COALESCE_BATCH
    try:
        n = int(n)
    except (TypeError, ValueError):
        return 0
    return max(0, min(n, SOURCE_COALESCE_MAX))


def _coalesce_tx_ms(ingester: Any) -> float:
    """Shared-commit wall-clock bound in ms (V8-13.05).

    ``cfg.jobs.source_coalesce_tx_ms`` overrides when provisioned;
    non-positive values mean "commit no more than the first sibling"
    (the real off-switch stays ``source_coalesce = 0``, which skips
    staging entirely).  Ceiling keeps a fat configured value from
    pushing the shared tx over the writer busy cap.
    """
    jobs_cfg = getattr(getattr(ingester, "cfg", None), "jobs", None)
    v = getattr(jobs_cfg, "source_coalesce_tx_ms", None)
    if v is None:
        v = SOURCE_COALESCE_TX_MS
    try:
        v = float(v)
    except (TypeError, ValueError):
        return SOURCE_COALESCE_TX_MS
    if not (v == v) or v in (float("inf"), float("-inf")):
        return SOURCE_COALESCE_TX_MS
    return max(0.0, min(v, SOURCE_COALESCE_TX_MS_MAX))


def _embed_batch(ingester: Any, refs: dict) -> Any:
    """Resolved ``dense.embed_batch`` bound (V8-08.02) for this job.

    The §23 resolver lives in ``retrieval/v7/dense_compact`` (the dense
    arm block's home); job refs are the leased job's ``input_refs`` —
    the per-job declaration channel that outranks the ``cfg.jobs``
    deployment defaults.
    """
    from ..retrieval.v7 import dense_compact as _dc

    return _dc.embed_batch_bound(refs, ingester)


def _jobs_now(ingester: Any) -> int:
    """The queue's clock — an injected ``_now`` on the jobs shim wins so
    tests drive queue age deterministically; else wall clock."""
    clock = getattr(getattr(ingester, "jobs", None), "_now", None)
    if callable(clock):
        try:
            return int(clock())
        except Exception:
            pass
    return now_us()


def _sibling_queue_ages_ms(
    ingester: Any, sibs: list
) -> dict:
    """``job_id → queue age in ms`` for candidate siblings (V8-08.02).

    The enqueue timestamp rides inside ``input_refs_json`` as the
    queue's private ``_enqueued_us`` key (``jobs/queue.py``) — the
    public refs map strips it, so the raw column is read here under the
    same convention as ``JobQueue.stats``. Hand-inserted rows without
    the key fall back to ``not_before_us`` (a retry-aware lower bound);
    a sibling with neither reports no age and never trips the bound.
    """
    ids = [s.get("job_id") for s in sibs if s.get("job_id")]
    if not ids:
        return {}
    enqueued: dict = {}
    try:
        with ingester.store.read() as conn:
            ph = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT job_id, input_refs_json FROM jobs"
                f" WHERE job_id IN ({ph})",
                ids,
            ).fetchall()
        for row in rows:
            try:
                refs = safe_json_loads(row[1] or "{}")
            except Exception:
                refs = {}
            v = refs.get("_enqueued_us") if isinstance(refs, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                enqueued[str(row[0])] = int(v)
    except Exception:
        enqueued = {}  # shim store without a jobs table — age unknown
    now = _jobs_now(ingester)
    out: dict = {}
    for s in sibs:
        jid = s.get("job_id")
        if jid is None:
            continue
        e = enqueued.get(str(jid))
        if e is None:
            nb = s.get("not_before_us")
            e = int(nb) if isinstance(nb, (int, float)) else None
        if e is not None:
            out[str(jid)] = max(0.0, (now - e) / 1000.0)
    return out


def _unit_rows(plan: Optional[dict]) -> int:
    """Pending unit rows a staged sibling would merge (V8-08.02)."""
    if not plan:
        return 0
    return sum(len(v) for v in (plan.get("buckets") or {}).values())


def _resolve_dense_tier(ingester: Any, refs: dict) -> Optional[str]:
    """Declared ``dense.tier`` pin for this job (V8-08.05), normalized."""
    from ..retrieval.v7 import dense_compact as _dc

    return _dc.resolve_dense_tier(refs, ingester)


def _dense_tier_ok(ingester: Any, encoder: Any, tier: str) -> bool:
    """Whether the configured encoder satisfies the declared tier."""
    from ..retrieval.v7 import dense_compact as _dc

    backend = getattr(getattr(ingester, "cfg", None), "embedding", None)
    return _dc.dense_tier_satisfied(
        tier,
        getattr(encoder, "encoder_id", None),
        getattr(backend, "backend", None),
    )


class _Stage:
    """Outside-the-write-tx staged work for one same-scope job.

    Mirrors the leased job's own pre-commit phase: parsed ref pins,
    verified payload text, derived artifacts, dedup signature/fields,
    the prescan plan, and (for embed) the encoded blob + unit plan. The
    fenced commit applies a stage without redoing unfenceable work.
    """

    __slots__ = (
        "job",
        "source_id",
        "revision",
        "expected_cv",
        "pinned_gen",
        "payload_ns",
        "text",
        "payload",
        "rev",
        "hmac_hex",
        "derived",
        "sig",
        "fields",
        "pre",
        "blob",
        "digest",
        "unit_space",
        "unit_plan",
    )

    def __init__(self, job: dict[str, Any], **kw: Any) -> None:
        self.job = job
        for name in self.__slots__[1:]:
            setattr(self, name, kw.get(name))


def _parse_source_refs(refs: Any) -> tuple[str, int]:
    """``(source_id, revision)`` from a source job's refs — same contract
    the handlers enforce on their own jobs."""
    if not isinstance(refs, dict):
        raise VerbatimError(
            ErrorCode.VALIDATION, "job input_refs must be a mapping"
        )
    return _require_ref(refs, "source_id"), _require_int_ref(refs, "revision")


def _stage_project_sibling(
    ingester: Any, sib_job: dict[str, Any]
) -> Optional[_Stage]:
    """Prepare one ``source_project`` sibling's pre-commit work.

    ``None`` leaves the job queued for the normal solo drain — a
    malformed, unreadable, or unverifiable sibling fails under the same
    per-job path it always had (identical codes, identical obligations),
    just one drain pass later.
    """
    try:
        refs = sib_job.get("input_refs") or {}
        source_id, revision = _parse_source_refs(refs)
        expected_cv = _opt_int(refs, "control_version")
        pinned_gen = _opt_int(refs, "generation")
        payload_ns = _opt_str(refs, "namespace")
        utf8_ok = refs.get("utf8_ok")
        if utf8_ok is not None and utf8_ok is not True:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "source_project refs.utf8_ok must be true when declared",
            )
        text = _verified_text(ingester, source_id, revision)
        rev = ingester.sources.get_revision(source_id, revision)
        if rev is None:
            raise VerbatimError(
                ErrorCode.EVIDENCE_UNAVAILABLE,
                f"source {source_id!r} rev {revision} metadata unavailable",
            )
        derived = _derive(text, _anchor_for(rev))
        pre = _prescan(
            ingester,
            source_id=source_id,
            revision=revision,
            payload_ns=payload_ns,
            derived=derived,
            text=text,
        )
        return _Stage(
            sib_job,
            source_id=source_id,
            revision=revision,
            expected_cv=expected_cv,
            pinned_gen=pinned_gen,
            payload_ns=payload_ns,
            text=text,
            rev=rev,
            hmac_hex=rev.get("payload_hmac_hex"),
            derived=derived,
            sig=frozenset(derived["signature"] or ()),
            fields=_prescan_dedup_fields(derived),
            pre=pre,
        )
    except VerbatimError:
        return None
    except Exception:
        # Staging is advisory — a crashed preparation leaves the job for
        # its solo drain, which owns the real failure record.
        return None


def _sibling_allowed(job: dict[str, Any], sib: dict[str, Any]) -> bool:
    """Whether ``sib`` may ride ``job``'s shared commit (V8-13.05).

    A job leased through the priority unblock-first pass carries
    ``_coalesce_sources`` — the marked set the caller is blocked on —
    and coalesces only siblings serving that same barrier (V6-02.08:
    a bounded priority drain takes ONLY marked work).  Unmarked
    siblings stay queued for the ordinary pass; unrestricted drains
    (the key absent) keep full same-scope coalescing.
    """
    restrict = job.get("_coalesce_sources")
    if restrict is None:
        return True
    refs = sib.get("input_refs") or {}
    return refs.get("source_id") in restrict


def _stage_project_siblings(
    ingester: Any, job: dict[str, Any], *, limit: int
) -> list[_Stage]:
    """Stage due ``source_project`` siblings in serial (peek) order."""
    if limit < 1:
        return []
    out: list[_Stage] = []
    for sib in ingester.jobs.pending_siblings(job, limit=limit):
        if not _sibling_allowed(job, sib):
            continue
        st = _stage_project_sibling(ingester, sib)
        if st is not None:
            out.append(st)
    return out


def _near_plan_covers(
    near: Any, new_sig: frozenset, new_fields: Any, members: Any
) -> bool:
    """Whether a prescan ``link_near`` plan still holds when earlier
    same-namespace sources committed inside this batch.

    Serial drain equivalence: a sibling running solo would see every
    earlier source's committed projection as a ``link_near`` candidate;
    the snapshot prescan cannot. When a batch-earlier member could beat
    or veto the plan's outcome the plan is dropped and the fused
    in-transaction scan sees the true serial state — the same guard
    ``_backfill_prescan`` applies to intra-batch candidates.
    """
    if not members:
        return True
    if near.out.reason not in ("ok", "no_match", "guard_veto"):
        return True
    from ..dedup import links

    for m_sig, m_fields in members:
        union = new_sig | m_sig
        score = len(new_sig & m_sig) / len(union) if union else 0.0
        if score < links.DEFAULT_THRESHOLD:
            continue
        if links.guard_veto(new_fields, m_fields) is not None:
            continue
        if near.best is None or score >= near.best["score"]:
            return False
    return True


def _fail_source_obligation_in_tx(
    conn: Any,
    ingester: Any,
    source_id: str,
    revision: int,
    capability: CapabilityName,
    code: str,
) -> None:
    """``_fail_source_obligation``'s row effects inside the batch commit.

    Best-effort exactly like the solo path — a bookkeeping disagreement
    never masks the job row's own failure record.
    """
    engine = getattr(ingester, "_readiness", None)
    if engine is None:
        return
    try:
        for rid in engine._existing_receipts(
            conn, receipt_ids_for_source(conn, source_id, revision)
        ):
            _declare_source_cap(conn, engine, source_id, rid, capability)
            engine.fail(conn, rid, capability, code, _missing_ok=True)
    except VerbatimError:
        pass


def _fail_sibling(
    conn: Any,
    ingester: Any,
    sib_job: dict[str, Any],
    owner: str,
    *,
    source_id: Optional[str],
    revision: Optional[int],
    capability: CapabilityName,
    code: str,
    retryable: bool,
) -> None:
    """The solo-drain failure sequence inside the batch commit.

    Permanent handler errors land on the source capability first (the
    ``_fail_source_obligation`` contract), then the queue's own
    retry/fail transition applies — retryable codes return the job to
    ``retry_wait`` under the same backoff math, terminal ones mark it
    ``failed`` and settle the receipt obligations through the drain's
    own ``_fail_job_obligations``. All inside the shared commit: a crash
    undoes the failure with the batch rather than stranding it.
    """
    if not retryable and source_id is not None:
        _fail_source_obligation_in_tx(
            conn,
            ingester,
            source_id,
            int(revision or 0),
            capability,
            code,
        )
    st = ingester.jobs.fail(
        conn,
        sib_job["job_id"],
        owner,
        sib_job["generation"],
        code,
        retryable,
    )
    if st is JobState.FAILED:
        try:
            ingester._fail_job_obligations(conn, sib_job, code)
        except VerbatimError:
            pass  # the job row's failed transition is the durable record


def _drain_siblings(
    conn: Any,
    ingester: Any,
    job: dict[str, Any],
    owner: str,
    *,
    kind_value: str,
    stages: list,
    apply_stage: Any,
    capability: CapabilityName,
    earlier: Optional[dict] = None,
) -> dict[str, int]:
    """Claim + commit staged siblings inside the caller's fenced commit.

    Per claimed sibling: one savepoint wrapping the queue's own fenced
    ``_commit_effects`` — so receipt replay, lease fencing, receipts,
    and completion are identical to a solo drain, and a crashed batch
    leaves every sibling either fully committed or never claimed
    (at-least-once + idempotent effects — never lost, never double).
    ``earlier`` (namespace → [(sig, fields)]) accumulates committed
    batch members so later siblings' near-plans see serial state.
    """
    stats = {
        "claimed": 0,
        "committed": 0,
        "replayed": 0,
        "failed": 0,
        "deferred": 0,
        "released": 0,
    }
    if not stages:
        return stats
    by_id = {s.job["job_id"]: s for s in stages}
    claimed = ingester.jobs.claim_siblings(
        conn,
        job,
        owner=owner,
        limit=len(by_id),
        job_ids=list(by_id),
    )
    stats["claimed"] = len(claimed)
    t_batch = time.monotonic()
    budget_s = _coalesce_tx_ms(ingester) / 1000.0
    for i, sib in enumerate(claimed):
        # Write-availability bound (V8-13.05): the shared commit already
        # holds the writer for the leased job's own work — once the
        # sibling loop has held it for the budget, the unprocessed tail
        # is released back to ``queued`` so a foreground ``add`` never
        # rides out the busy cap.  At least one sibling always commits.
        if i and (time.monotonic() - t_batch) >= budget_s:
            stats["released"] = ingester.jobs.release_siblings(
                conn, (c["job_id"] for c in claimed[i:]), owner=owner
            )
            break
        stage = by_id.get(sib["job_id"])
        if stage is None:
            continue  # unreachable while claims restrict to staged ids
        conn.execute("SAVEPOINT v8_sibling")
        try:
            res = ingester._commit_effects(
                conn,
                sib,
                owner,
                kind_value,
                lambda c, _s=stage: apply_stage(c, _s),
            )
        except _PlanStale:
            conn.execute("ROLLBACK TO v8_sibling")
            conn.execute("RELEASE v8_sibling")
            _fail_sibling(
                conn,
                ingester,
                sib,
                owner,
                source_id=stage.source_id,
                revision=stage.revision,
                capability=capability,
                code=ErrorCode.STALE_DEPENDENCY.value,
                retryable=True,
            )
            stats["deferred"] += 1
            continue
        except VerbatimError as exc:
            conn.execute("ROLLBACK TO v8_sibling")
            conn.execute("RELEASE v8_sibling")
            _fail_sibling(
                conn,
                ingester,
                sib,
                owner,
                source_id=stage.source_id,
                revision=stage.revision,
                capability=capability,
                code=exc.code.value,
                retryable=exc.retryable,
            )
            stats["deferred" if exc.retryable else "failed"] += 1
            continue
        except Exception:
            # Unexpected sibling fault — contain it: the batch's other
            # work is already staged in this tx and must not be lost to
            # one job's bug. The sibling retries through the queue's own
            # backoff exactly as a solo crash would.
            conn.execute("ROLLBACK TO v8_sibling")
            conn.execute("RELEASE v8_sibling")
            _fail_sibling(
                conn,
                ingester,
                sib,
                owner,
                source_id=stage.source_id,
                revision=stage.revision,
                capability=capability,
                code=ErrorCode.RETRYABLE_OPERATION.value,
                retryable=True,
            )
            stats["deferred"] += 1
            continue
        conn.execute("RELEASE v8_sibling")
        if res.get("replayed"):
            stats["replayed"] += 1
        else:
            stats["committed"] += 1
        if earlier is not None:
            # A committed (or receipt-replayed) sibling's projection rows
            # exist for later siblings' serial-equivalent near scans.
            result = res.get("result") or {}
            ns = result.get("namespace") if isinstance(result, dict) else None
            if ns and stage.sig is not None:
                earlier.setdefault(ns, []).append(
                    (stage.sig, stage.fields)
                )
    return stats


def handle_source_project(job: dict[str, Any], owner: str, ingester: Any) -> None:
    """Tokenize one source revision into the lexical projection plane.

    Publishes ``source_lexical_projection`` + ``source_fts`` pair +
    ``entity_postings`` + ``enrichment`` and runs dedup/update-candidate
    discovery — all inside ``_commit_effects``'s generation-fenced tx, so
    lease loss, predicate failure, derived writes, readiness settlement,
    and job completion are one atomic unit. Idempotent: rows are keyed
    on (source_id, revision); a redelivery replays the operation receipt.
    """
    refs = job["input_refs"]
    try:
        source_id = _require_ref(refs, "source_id")
        revision = _require_int_ref(refs, "revision")
    except VerbatimError:
        # Malformed refs carry no source binding — the job row's failure
        # is the whole record; there is nothing to settle against.
        raise
    try:
        _project_body(ingester, job, owner, source_id, revision)
    except VerbatimError as exc:
        if not exc.retryable:
            _fail_source_obligation(
                ingester, job, owner, source_id, revision,
                CapabilityName.SOURCE_LEXICAL_READY, exc.code.value,
            )
        raise


def _apply_projection(
    conn: Any,
    ingester: Any,
    *,
    stage: _Stage,
    fp0: Any,
    strict: bool,
    earlier: Optional[dict] = None,
    coalesced: bool = False,
) -> dict[str, Any]:
    """One source revision's projection commit body (V8-13.05 shared).

    Runs inside the caller's fenced transaction for the leased job AND
    for each claimed sibling — identical publication predicate, derived
    writes, dedup/update detection, readiness settlement, and event, so
    a coalesced job produces exactly the rows a solo drain would.
    ``earlier`` (namespace → [(sig, fields)]) covers the batch's own
    earlier members for the intra-batch near-plan guard.
    """
    source_id, revision = stage.source_id, stage.revision
    derived = stage.derived
    pre = stage.pre
    state, namespace = _publication_gate(
        conn,
        ingester,
        source_id,
        revision,
        expected_control_version=stage.expected_cv,
        namespace_hint=stage.payload_ns,
        store=ingester.store,
    )
    if (
        strict
        and pre is not None
        and fp0 is not None
        and fp0 != pre.fp
        and pre.namespace == namespace
    ):
        raise _PlanStale()
    generation = _projection_generation(conn, ingester, stage.pinned_gen)
    _write_lexical(
        conn,
        source_id=source_id,
        revision=revision,
        namespace=namespace,
        generation=generation,
        tokens_text=derived["tokens_text"],
        doc_len=derived["doc_len"],
        digest=derived["digest"],
    )
    postings = _write_postings(
        conn,
        namespace=namespace,
        source_id=source_id,
        revision=revision,
        generation=generation,
        identifiers=derived["identifiers"],
        entities=derived["entities"],
    )
    _write_enrichment(
        conn,
        source_id=source_id,
        revision=revision,
        mem_type=derived["mem_type"],
        pol=derived["polarity"],
        temporal=derived["temporal"],
        identifiers=derived["identifiers"],
        entities=derived["entities"],
    )
    from ..querying.updates import write_term_postings

    write_term_postings(
        conn,
        namespace=namespace,
        source_id=source_id,
        revision=revision,
        text=stage.text,
    )
    # V7 unit projection (V7-06.05, §30): derive units + populate the
    # V7 artifact plane inside the same generation-fenced commit. The
    # wrapper resolves the persisted source row + persisted add-args
    # itself; an additive-plane fault rolls back to a savepoint and
    # lands on the stats channel — never hostage to the V5 commit.
    v7_stats: dict[str, Any] = {}
    if _project_source_v7 is not None:
        v7_stats = _project_source_v7(
            conn,
            source_id=source_id,
            revision=revision,
            text=stage.text,
            revision_meta=stage.rev,
            namespace=namespace,
            generation=generation,
        )
    use_plan = (
        pre is not None
        and fp0 is not None
        and fp0 == pre.fp
        and pre.namespace == namespace
    )
    near_plan = pre.near if use_plan else None
    if near_plan is not None and earlier:
        # A solo drain would see the batch's earlier committed members —
        # a plan that cannot account for them falls back to the fused
        # in-transaction scan (serial equivalence, V8-13.05).
        if not _near_plan_covers(
            near_plan, stage.sig, stage.fields, earlier.get(namespace)
        ):
            near_plan = None
    links_out = _run_dedup(
        conn,
        source_id=source_id,
        revision=revision,
        namespace=namespace,
        payload_hmac_hex=stage.hmac_hex,
        digest=derived["digest"],
        signature=derived["signature"],
        text_fields=None,
        store=ingester.store,
        near_plan=near_plan,
    )
    if use_plan:
        from ..querying.updates import persist_update_candidates

        candidates = persist_update_candidates(conn, pre.cands)
    else:
        candidates = _detect_updates(
            conn,
            namespace=namespace,
            source_id=source_id,
            revision=revision,
            text=stage.text,
            mem_type=derived["mem_type"],
            pol=derived["polarity"],
            temporal=derived["temporal"],
            identifiers=derived["identifiers"],
            entities=derived["entities"],
            store=ingester.store,
        )
    settled = _settle_source(
        conn,
        ingester,
        source_id,
        revision,
        CapabilityName.SOURCE_LEXICAL_READY,
    )
    seq = _append_event(
        conn,
        ingester,
        stage.job,
        "source_projected",
        {
            "source_id": source_id,
            "revision": revision,
            "namespace": namespace,
            "generation": generation,
            "doc_len": derived["doc_len"],
            "digest": derived["digest"],
            "producer": PRODUCER_PROJECT,
            # V8-13.05 latency stage: this commit is when the source's
            # lexical plane becomes visible to readers.
            "stage": "lexical_visible",
            "coalesced": True if coalesced else None,
        },
    )
    return {
        "source_id": source_id,
        "revision": revision,
        "namespace": namespace,
        "control_version": state.control_version,
        "generation": generation,
        "doc_len": derived["doc_len"],
        "digest": derived["digest"],
        "postings": postings,
        "links": links_out,
        "update_candidates": len(candidates),
        "receipts_settled": settled,
        "event_seq": seq,
        "v7_units": v7_stats.get("units", 0),
        "v7_stats": v7_stats,
    }


def _project_body(
    ingester: Any, job: dict[str, Any], owner: str,
    source_id: str, revision: int,
) -> None:
    refs = job["input_refs"]
    _require_v5(ingester)
    expected_cv = _opt_int(refs, "control_version")
    pinned_gen = _opt_int(refs, "generation")
    payload_ns = _opt_str(refs, "namespace")
    utf8_ok = refs.get("utf8_ok")
    if utf8_ok is not None and utf8_ok is not True:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "source_project refs.utf8_ok must be true when declared — a "
            "declared-false byte check cannot project",
        )

    # Park the obligation first: a permanent failure below (decode,
    # predicate, fence) must land on the capability row, which only exists
    # once declared (V5-08.16).
    _declare_obligations(
        ingester, job, owner, source_id, revision,
        CapabilityName.SOURCE_LEXICAL_READY,
    )

    # Verified canonical bytes + derived artifacts computed outside the
    # write tx (V4-09.01/09.09 — no inference inside a write tx).
    text = _verified_text(ingester, source_id, revision)
    rev = ingester.sources.get_revision(source_id, revision)
    if rev is None:
        raise VerbatimError(
            ErrorCode.EVIDENCE_UNAVAILABLE,
            f"source {source_id!r} rev {revision} metadata unavailable",
        )
    derived = _derive(text, _anchor_for(rev))
    payload_hmac_hex = rev.get("payload_hmac_hex")

    # Advisory read phase: the dedup/update scans run on a WAL snapshot
    # ahead of the fenced commit so the ~ms-scale namespace scans do not
    # ride inside the writer's hold window. The commit re-fingerprints
    # the same dependency tables; identical fingerprint => identical
    # plan inputs => the plan commits as-is, any drift falls back to the
    # fused in-transaction calls.
    _t0 = time.monotonic()
    pre = _prescan(
        ingester,
        source_id=source_id,
        revision=revision,
        payload_ns=payload_ns,
        derived=derived,
        text=text,
    )
    prescan_ms = (time.monotonic() - _t0) * 1e3

    # Strict planned commits first: while a plan exists, a fingerprint
    # drift at commit time aborts the tx (rollback — never partial
    # state) and replans instead of running the fused namespace scans
    # under the write lock; ``retries_left`` bounds the loop and the
    # exhausted attempt clears ``strict`` so the fused path remains the
    # honest bound it always was. The budget only exists when the fused
    # scans are expensive enough to threaten a peer's busy cap — below
    # ``_REPLAN_MIN_SCAN_MS`` a fused fallback is ~ms and replanning
    # would just delay completion.
    strict = pre is not None and _PLAN_STALE_RETRIES > 0
    retries_left = (
        _PLAN_STALE_RETRIES
        if prescan_ms >= _REPLAN_MIN_SCAN_MS
        else 0
    )

    primary = _Stage(
        job,
        source_id=source_id,
        revision=revision,
        expected_cv=expected_cv,
        pinned_gen=pinned_gen,
        payload_ns=payload_ns,
        text=text,
        rev=rev,
        hmac_hex=payload_hmac_hex,
        derived=derived,
        sig=frozenset(derived["signature"] or ()),
        fields=_prescan_dedup_fields(derived),
        pre=pre,
    )
    # V8-13.05: stage due same-scope siblings for the shared commit.
    siblings = _stage_project_siblings(
        ingester, job, limit=_coalesce_limit(ingester)
    )
    # namespace → [(sig, fields)] of batch members committed earlier in
    # this tx — the intra-batch serial-equivalence guard inputs.
    earlier: dict[str, list] = {}

    def _apply(conn: Any) -> dict[str, Any]:
        # Committed-state fingerprint at the top of the tx — before the
        # gate's own adoption write touches a dependency table. Any
        # staged plan's mode works: prescans all resolve counter|fold
        # the same way on this store.
        fp_mode = (
            pre.mode
            if pre is not None
            else next(
                (s.pre.mode for s in siblings if s.pre is not None), None
            )
        )
        fp0 = (
            _deps_fingerprint(conn, fp_mode) if fp_mode is not None else None
        )
        earlier.clear()  # a replanned attempt re-accumulates from zero
        result = _apply_projection(
            conn,
            ingester,
            stage=primary,
            fp0=fp0,
            strict=strict,
            earlier=earlier,
        )
        if siblings:
            earlier.setdefault(result["namespace"], []).append(
                (primary.sig, primary.fields)
            )
            result["coalesced"] = _drain_siblings(
                conn,
                ingester,
                job,
                owner,
                kind_value="source_project",
                stages=siblings,
                apply_stage=lambda c, st: _apply_projection(
                    c,
                    ingester,
                    stage=st,
                    fp0=fp0,
                    strict=False,
                    earlier=earlier,
                    coalesced=True,
                ),
                capability=CapabilityName.SOURCE_LEXICAL_READY,
                earlier=earlier,
            )
        return result

    while True:
        try:
            with ingester.store.tx() as conn:
                ingester._commit_effects(
                    conn, job, owner, "source_project", _apply
                )
            return
        except _PlanStale:
            if retries_left > 0:
                retries_left -= 1
                # Fresh snapshot, fresh plan, another strict commit —
                # the scans stay off the write lock. Re-measure the
                # scan cost: a namespace that got cheap no longer needs
                # the retry budget (the fused hold is now harmless).
                _t0 = time.monotonic()
                new_pre = _prescan(
                    ingester,
                    source_id=source_id,
                    revision=revision,
                    payload_ns=payload_ns,
                    derived=derived,
                    text=text,
                )
                prescan_ms = (time.monotonic() - _t0) * 1e3
                if new_pre is not None:
                    pre = new_pre
                    primary.pre = new_pre
                    strict = True
                    if prescan_ms < _REPLAN_MIN_SCAN_MS:
                        retries_left = 0
                    continue
                # A prescan that comes back empty after a real stale
                # abort leaves nothing to verify — the fused path below
                # is the only remaining way to see the live state.
            # Planned commits exhausted: the fused path is the honest
            # bound — a job must still complete under genuinely
            # sustained churn rather than starve on the write side,
            # which is the same guarantee this code always had. The
            # retries above keep it rare; when the namespace quieted in
            # the meantime the commit's own fingerprint check can still
            # reuse the last plan instead of scanning.
            strict = False


# ----------------------------------------------------------------------
# source_embed — V7 unit vectors (V75-03.02, V7-07.04/07.06/07.09)
# ----------------------------------------------------------------------
#
# The same fenced commit that upserts ``source_vectors`` also produces
# the contiguous per-(encoder, scope, generation) unit-vector matrix the
# dense lane scans: every unit of the revision's newest unit-slice at or
# below the commit's resolved projection generation is encoded as
# ``header + payload[byte_start:byte_end]`` (``hdr:v1`` — the header
# version rides inside the block space's encoder_id), chunked into
# ``write_block`` calls of ≤ ``matrix.MAX_BLOCK_ROWS`` rows at
# ``block_no = MAX+1``, and mirrored into the per-row f32 oracle
# (``unit_vectors``) the int8 rescore phase reads.


def _hdr_field(value: Any) -> str:
    """One ``hdr:v1`` header field — whitespace collapsed, ``|`` flattened
    to ``/`` so a field can never forge the `` | `` separator. Anything
    non-string/empty maps to the empty field."""
    if not isinstance(value, str) or not value:
        return ""
    return " ".join(value.split()).replace("|", "/")


def _unit_header_date(unit: dict) -> str:
    """``hdr:v1`` date field: UTC ``YYYY-MM-DD`` of ``occurred_start_us``,
    falling back to ``recorded_at_us``; ``""`` when neither pins an int.
    (``rfc3339`` renders UTC; its first ten chars are the calendar day —
    the same convention ``units_jobs._when_tokens`` uses.)"""
    for key in ("occurred_start_us", "recorded_at_us"):
        us = unit.get(key)
        if isinstance(us, int) and not isinstance(us, bool):
            return rfc3339(us)[:10]
    return ""


def _unit_embed_text(unit: dict, payload: bytes) -> Optional[str]:
    """The ``hdr:v1`` contextual text embedded for one unit row.

    ``<date> | <speaker> | <session label>: `` + the unit's byte-pinned
    payload slice (V7-07.09):

    * ``date`` — UTC ``YYYY-MM-DD`` of ``occurred_start_us`` (fallback
      ``recorded_at_us``); empty when neither is an int.
    * ``speaker`` — ``speaker_canon``; empty when absent.
    * ``session label`` — the unit's ``session_id``. In this schema the
      session's canonical label IS its ``session_id``: ``derive_units``
      mints a ``kind='session'`` unit whose ``session_id`` carries the
      session's label, and member units share it, so the unit row's own
      ``session_id`` is exactly the session row's label when that row
      exists (and the caller's session id when none was projected);
      empty for sessionless units.

    ``None`` when the unit carries no resolvable byte pins — unpinned
    units keep metadata-only coverage (``units_jobs._unit_text``'s rule:
    never index unverifiable text, never fabricate a payload).
    """
    bs, be = unit.get("byte_start"), unit.get("byte_end")
    if not (
        isinstance(bs, int)
        and isinstance(be, int)
        and not isinstance(bs, bool)
        and not isinstance(be, bool)
        and 0 <= bs <= be <= len(payload)
    ):
        return None
    try:
        text = payload[bs:be].decode("utf-8")
    except UnicodeDecodeError:
        return None
    header = (
        f"{_unit_header_date(unit)} | "
        f"{_hdr_field(unit.get('speaker_canon'))} | "
        f"{_hdr_field(unit.get('session_id'))}: "
    )
    return header + text


_UNITS_FOR_EMBED_SQL = """
SELECT unit_id, scope_id, generation, kind, session_id,
       speaker_canon, recorded_at_us, occurred_start_us,
       byte_start, byte_end
FROM units
WHERE source_id = ? AND revision = ?
  AND generation = (
      SELECT MAX(generation) FROM units
      WHERE source_id = ? AND revision = ? AND generation <= ?)
ORDER BY unit_id
"""


def _load_embed_units(
    conn: Any, source_id: str, revision: int, bound: int
) -> list[dict]:
    """The revision's unit slice at the newest generation ≤ ``bound``.

    A ``(source, revision)`` unit set is a complete per-generation
    snapshot (``project_units_v7`` writes the whole slice in one
    generation-fenced tx, and ``unit_id`` is content-addressed *with*
    its generation), so ``generation <= bound`` alone would union every
    surviving generation's slice and re-projected units would embed
    twice. The newest slice at/below the fence is the revision's live
    unit set — the same complete-snapshot rule the dense lane applies
    when it scans ``MAX(block generation) <= pin``.
    """
    return [
        {
            "unit_id": r[0],
            "scope_id": r[1],
            "generation": int(r[2]),
            "kind": r[3],
            "session_id": r[4],
            "speaker_canon": r[5],
            "recorded_at_us": r[6],
            "occurred_start_us": r[7],
            "byte_start": r[8],
            "byte_end": r[9],
        }
        for r in conn.execute(
            _UNITS_FOR_EMBED_SQL,
            (source_id, revision, source_id, revision, bound),
        ).fetchall()
    ]


def _units_signature(rows: list[dict]) -> tuple:
    """Everything the embed plan depends on, in the loader's
    deterministic ``ORDER BY unit_id`` — a drift between the encode
    snapshot and the fenced commit aborts and re-plans (the
    ``_project_body`` prescan discipline)."""
    return tuple(
        (
            r["unit_id"],
            r["scope_id"],
            r["generation"],
            r["kind"],
            r["session_id"],
            r["speaker_canon"],
            r["recorded_at_us"],
            r["occurred_start_us"],
            r["byte_start"],
            r["byte_end"],
        )
        for r in rows
    )


def _prepare_unit_vectors(
    ingester: Any,
    encoder: Any,
    source_id: str,
    revision: int,
    payload: bytes,
    *,
    batch_rows: Optional[int] = None,
) -> dict:
    """Snapshot the revision's unit slice and encode each pinned unit —
    model work strictly between snapshot and fenced commit (V2-39.07).

    Returns a plan: ``signature`` (drift fence the commit re-verifies),
    ``buckets`` (``scope_id → [(unit_id, f32 blob)]`` in the loader's
    deterministic ``unit_id`` order), ``units``/``unpinned`` counts, and
    ``units_generation`` (the generation that minted the slice — the
    generation the block snapshot belongs to). A store without the V7
    unit plane gets an empty plan; the commit then records the absence
    honestly instead of fabricating rows.

    ``batch_rows`` is the resolved ``dense.embed_batch`` row bound
    (V8-08.02): ``encode`` runs in chunks of at most that many texts so
    one revision's inference is never an unbounded call. ``None`` uses
    the caller-free resolution (config/policy carriers only).
    """
    plan: dict[str, Any] = {
        "signature": (),
        "buckets": {},
        "units": 0,
        "unpinned": 0,
        "units_generation": None,
    }
    with ingester.store.read() as conn:
        if not has_table(conn, "units"):
            return plan
        bound = int(
            ingester.store._meta_get(conn, "projection_generation") or 0
        )
        units = _load_embed_units(conn, source_id, revision, bound)
    plan["units"] = len(units)
    if not units:
        return plan
    plan["units_generation"] = units[0]["generation"]

    pinned: list[dict] = []
    texts: list[str] = []
    for u in units:
        t = _unit_embed_text(u, payload)
        if t is None:
            plan["unpinned"] += 1
            continue
        pinned.append(u)
        texts.append(t)
    if texts:
        # V8-08.02 ``dense.embed_batch``: the row bound also caps one
        # ``encode`` call's text count — a fat revision encodes in
        # bounded chunks, never one unbounded batch. ``0``/``None``
        # leave the call unchunked (the bound governs merge size; a
        # zero cap cannot split an atomic encode).
        step = batch_rows
        if step is None:
            step = _embed_batch(ingester, {}).rows
        if step is None or step <= 0 or len(texts) <= step:
            blobs = encoder.encode(texts)
        else:
            blobs = []
            for i in range(0, len(texts), step):
                blobs.extend(encoder.encode(texts[i : i + step]))
        if len(blobs) != len(texts):
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID,
                f"encoder returned {len(blobs)} vectors for "
                f"{len(texts)} unit texts",
            )
        buckets: dict[str, list] = {}
        for u, blob in zip(pinned, blobs):
            blob = bytes(blob)
            # Same codec gate as the source-vector write: a malformed
            # blob is VECTOR_INVALID and never lands.
            Float32Codec.validate_blob(blob, encoder.dimensions)
            buckets.setdefault(u["scope_id"], []).append(
                (u["unit_id"], blob)
            )
        plan["buckets"] = buckets
    plan["signature"] = _units_signature(units)
    return plan


def _commit_unit_vectors(
    conn: Any,
    *,
    space_id: str,
    plan: dict,
    source_id: str,
    revision: int,
    generation: int,
) -> dict:
    """Append the revision's unit vectors inside the fenced commit.

    The commit re-loads the unit slice at the resolved generation and
    refuses to write when it drifted from the encoded snapshot
    (``_PlanStale`` → the caller re-encodes on a fresh snapshot). Blocks
    are stamped at the commit's resolved projection ``generation`` — the
    same fence the ``source_vectors`` upsert carries (V75-03.02: "at the
    pinned generation") — and appended at ``block_no = MAX+1`` per
    ``(encoder_id, scope_id, generation)``. Stamping at commit time lets
    a re-embed after a generation bump land the same units' vectors in
    the newest snapshot the dense lane scans. Every pinned unit also
    lands an f32 oracle row (``unit_vectors``) for the int8 rescore
    phase.
    """
    stats: dict[str, Any] = {
        "encoder_id": space_id,
        "units": plan["units"],
        "embedded": 0,
        "unpinned": plan["unpinned"],
        "skipped_present": 0,
        "blocks": 0,
        "oracle": 0,
        "quant": [],
    }
    if not has_table(conn, "units"):
        if plan["signature"]:
            raise _PlanStale(
                "units table vanished between snapshot and commit"
            )
        stats["note"] = "no_units_table"
        return stats
    rows_now = _load_embed_units(conn, source_id, revision, generation)
    if _units_signature(rows_now) != plan["signature"]:
        raise _PlanStale(
            f"units for {source_id}@{revision} drifted between the "
            "encode snapshot and the fenced commit"
        )
    if not plan["buckets"]:
        return stats
    if not has_table(conn, _matrix.BLOCK_TABLE):
        # A partially provisioned store (units without the §30 block
        # table) can't take the write — record the absence, don't fail
        # the commit: dense coverage reports the hole honestly.
        stats["note"] = "no_block_table"
        return stats
    _matrix.ensure_oracle_table(conn)
    quants: set = set()
    for scope_id in sorted(plan["buckets"]):
        # Idempotent append: unit ids already in this generation's space
        # (a duplicate embed job for the same revision, e.g. a stale
        # sibling leased before a re-plan) are skipped, not re-appended —
        # re-writing the same deterministic vectors buys nothing and the
        # scan must never see one unit twice.
        present = _matrix.block_row_keys(conn, space_id, scope_id, generation)
        items = [
            (key, blob)
            for key, blob in plan["buckets"][scope_id]
            if key not in present
        ]
        stats["skipped_present"] += len(plan["buckets"][scope_id]) - len(
            items
        )
        if not items:
            continue
        row = conn.execute(
            f"SELECT COALESCE(SUM(n_rows), 0), COALESCE(MAX(block_no), -1)"
            f" FROM {_matrix.BLOCK_TABLE}"
            " WHERE encoder_id = ? AND scope_id = ? AND generation = ?",
            (space_id, scope_id, generation),
        ).fetchone()
        existing, block_no = int(row[0]), int(row[1]) + 1
        # V7-07.06: the space stores int8 once its row count crosses the
        # threshold; this batch's quant is decided on the post-write
        # total so a space straddling the bound switches on the write
        # that crosses it.
        quant = (
            "int8"
            if existing + len(items) > _matrix.INT8_ROW_THRESHOLD
            else "f32"
        )
        quants.add(quant)
        for i in range(0, len(items), _matrix.MAX_BLOCK_ROWS):
            chunk = items[i : i + _matrix.MAX_BLOCK_ROWS]
            _matrix.write_block(
                conn,
                space_id,
                scope_id,
                generation,
                block_no,
                chunk,
                quant=quant,
            )
            block_no += 1
            stats["blocks"] += 1
            for key, blob in chunk:
                _matrix.write_oracle_row(conn, space_id, key, blob)
                stats["oracle"] += 1
        stats["embedded"] += len(items)
    stats["quant"] = sorted(quants)
    stats["vector_generation"] = generation
    return stats


# ----------------------------------------------------------------------
# source_embed
# ----------------------------------------------------------------------


def _stage_embed_sibling(
    ingester: Any,
    encoder: Any,
    sib_job: dict[str, Any],
    *,
    batch_rows: Optional[int] = None,
) -> Optional[_Stage]:
    """Prepare one ``source_embed`` sibling's pre-commit work.

    Verified payload + the unit-vector plan come from read snapshots
    (same discipline as the leased body); the source-vector blob is
    batch-encoded by the caller afterwards. ``None`` leaves the job for
    the solo drain — identical failure semantics one pass later.
    ``batch_rows`` is the batch's resolved ``dense.embed_batch`` row
    bound, forwarded to the unit-vector encode (V8-08.02).
    """
    try:
        refs = sib_job.get("input_refs") or {}
        source_id, revision = _parse_source_refs(refs)
        expected_cv = _opt_int(refs, "control_version")
        pinned_gen = _opt_int(refs, "generation")
        payload_ns = _opt_str(refs, "namespace")
        want = refs.get("encoder") or refs.get("encoder_id")
        if want is not None and want != encoder.encoder_id:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"source_embed job targets encoder {want!r}; "
                f"configured is {encoder.encoder_id!r}",
            )
        # A sibling carrying its own ``dense.tier`` pin stays queued
        # when the batch's encoder doesn't satisfy it — its solo drain
        # records the deferral under its own refs.
        tier = _resolve_dense_tier(ingester, refs)
        if tier is not None and not _dense_tier_ok(ingester, encoder, tier):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"source_embed job pins dense.tier {tier!r}; "
                f"configured is {encoder.encoder_id!r}",
            )
        text = _verified_text(ingester, source_id, revision)
        payload = bytes(ingester.sources.payload(source_id, revision) or b"")
        unit_plan = _prepare_unit_vectors(
            ingester,
            encoder,
            source_id,
            revision,
            payload,
            batch_rows=batch_rows,
        )
        return _Stage(
            sib_job,
            source_id=source_id,
            revision=revision,
            expected_cv=expected_cv,
            pinned_gen=pinned_gen,
            payload_ns=payload_ns,
            text=text,
            payload=payload,
            unit_space=_matrix.unit_vector_encoder_id(encoder.encoder_id),
            unit_plan=unit_plan,
        )
    except VerbatimError:
        return None
    except Exception:
        return None


def _stage_embed_siblings(
    ingester: Any, encoder: Any, job: dict[str, Any], *, limit: int
) -> tuple:
    """Stage + batch-encode due ``source_embed`` siblings.

    Returns ``(ready_stages, batch_stats)``. The §23
    ``dense.embed_batch`` arm (V8-08.02, resolved off the leased job's
    refs via :func:`_embed_batch`) bounds the merge: a sibling's pending
    unit rows join only while the cumulative count stays within
    ``rows`` (first overflow closes the batch — serial order is never
    reordered), and an admitted sibling whose queue age already reached
    ``max_age_ms`` closes the batch after itself — a deadline flush,
    never a skip. The source-vector encode honors the same row bound
    per ``encode`` call; siblings whose blob fails the codec gate stay
    queued for the solo drain's identical VECTOR_INVALID failure.
    """
    bound = _embed_batch(ingester, job.get("input_refs") or {})
    stats = {
        "rows_bound": bound.rows,
        "max_age_ms": bound.max_age_ms,
        "merged_rows": 0,
        "closed_by": None,
    }
    if limit < 1:
        return [], stats
    candidates = ingester.jobs.pending_siblings(job, limit=limit)
    ages = (
        _sibling_queue_ages_ms(ingester, candidates)
        if bound.max_age_ms is not None
        else {}
    )
    staged: list[_Stage] = []
    merged_rows = 0
    for sib in candidates:
        if not _sibling_allowed(job, sib):
            continue
        st = _stage_embed_sibling(
            ingester, encoder, sib, batch_rows=bound.rows
        )
        if st is None:
            continue
        sib_rows = max(_unit_rows(st.unit_plan), 1)
        if (
            bound.rows is not None
            and staged
            and merged_rows + sib_rows > bound.rows
        ):
            # Row bound — the batch is full; this sibling heads the next
            # drain's batch (FIFO preserved, never leapfrogged).
            stats["closed_by"] = "rows"
            break
        staged.append(st)
        merged_rows += sib_rows
        if (
            bound.max_age_ms is not None
            and ages.get(str(sib.get("job_id"))) is not None
            and ages[str(sib["job_id"])] >= bound.max_age_ms
        ):
            # Queue-age bound — the pending unit hit its deadline; the
            # flush lands now and later arrivals form the next batch.
            stats["closed_by"] = "age"
            break
    stats["merged_rows"] = merged_rows
    if not staged:
        return [], stats
    texts = [s.text for s in staged]
    blobs: list[Any] = [None] * len(staged)
    try:
        step = bound.rows
        if step is None or step <= 0 or len(texts) <= step:
            out = encoder.encode(texts)
        else:
            # ``dense.embed_batch`` rows cap one encode call too.
            out = []
            for i in range(0, len(texts), step):
                out.extend(encoder.encode(texts[i : i + step]))
        if len(out) == len(texts):
            blobs = [bytes(b) for b in out]
    except Exception:
        # Batch inference fault — fall back to per-sibling encode so one
        # bad payload cannot lose the whole batch; siblings that still
        # fail stay queued for the solo drain's identical failure.
        for i, s in enumerate(staged):
            try:
                b = encoder.encode([s.text])
                if len(b) == 1:
                    blobs[i] = bytes(b[0])
            except Exception:
                pass
    if all(b is None for b in blobs):
        return [], stats
    ready: list[_Stage] = []
    for st, blob in zip(staged, blobs):
        if blob is None:
            continue
        try:
            Float32Codec.validate_blob(blob, encoder.dimensions)
        except VerbatimError:
            continue  # solo drain owns the VECTOR_INVALID record
        st.blob = blob
        st.digest = _vector_digest(encoder.encoder_id, blob)
        ready.append(st)
    stats["encoded_rows"] = len(ready)
    return ready, stats


def _apply_embed(
    conn: Any,
    ingester: Any,
    *,
    stage: _Stage,
    enc_id: str,
    coalesced: bool = False,
) -> dict[str, Any]:
    """One source revision's vector commit body (V8-13.05 shared).

    Identical gate, upsert, unit-matrix write, readiness settle, and
    event as the leased job — the shared commit changes transaction
    count, never job effects.
    """
    source_id, revision = stage.source_id, stage.revision
    state, namespace = _publication_gate(
        conn,
        ingester,
        source_id,
        revision,
        expected_control_version=stage.expected_cv,
        namespace_hint=stage.payload_ns,
        store=ingester.store,
    )
    generation = _projection_generation(conn, ingester, stage.pinned_gen)
    repos_v5.upsert(
        conn,
        "source_vectors",
        {
            "source_id": source_id,
            "revision": revision,
            "namespace": namespace,
            "encoder": enc_id,
            "generation": generation,
            "vector": stage.blob,
            "digest": stage.digest,
        },
    )
    # V75-03.02: the contiguous unit-vector matrix lands in the same
    # fenced commit as the source_vectors upsert.
    unit_stats = _commit_unit_vectors(
        conn,
        space_id=stage.unit_space,
        plan=stage.unit_plan,
        source_id=source_id,
        revision=revision,
        generation=generation,
    )
    settled = _settle_source(
        conn,
        ingester,
        source_id,
        revision,
        CapabilityName.SOURCE_VECTOR_READY,
    )
    seq = _append_event(
        conn,
        ingester,
        stage.job,
        "source_embedded",
        {
            "source_id": source_id,
            "revision": revision,
            "namespace": namespace,
            "encoder": enc_id,
            "unit_encoder": stage.unit_space,
            "generation": generation,
            "digest": stage.digest,
            "unit_vectors": unit_stats,
            "producer": PRODUCER_EMBED,
            # V8-13.05 latency stage: the T0 enrichment plane is
            # populated by this commit.
            "stage": "t0_enriched",
            "coalesced": True if coalesced else None,
        },
    )
    return {
        "source_id": source_id,
        "revision": revision,
        "namespace": namespace,
        "control_version": state.control_version,
        "encoder": enc_id,
        "unit_encoder": stage.unit_space,
        "generation": generation,
        "digest": stage.digest,
        "unit_vectors": unit_stats,
        "receipts_settled": settled,
        "event_seq": seq,
    }


def _apply_embed_note(
    conn: Any,
    ingester: Any,
    *,
    stage: _Stage,
    reason: str,
) -> dict[str, Any]:
    """The ``_note`` deferral body for a coalesced sibling — same event
    + obligation deferral the leased job records."""
    seq = _append_event(
        conn,
        ingester,
        stage.job,
        "source_embed_skipped",
        {
            "job_id": stage.job["job_id"],
            "source_id": stage.source_id,
            "revision": stage.revision,
            "reason": reason,
            "producer": PRODUCER_EMBED,
            "stage": "t0_enriched",
            "coalesced": True,
        },
    )
    settled = _defer_source(
        conn,
        ingester,
        stage.source_id,
        stage.revision,
        CapabilityName.SOURCE_VECTOR_READY,
        reason,
    )
    return {
        "encoded": 0,
        "note": reason,
        "receipts_settled": settled,
        "event_seq": seq,
    }


def _stage_embed_notes(
    ingester: Any, job: dict[str, Any], *, limit: int
) -> list[_Stage]:
    """Thin stages for the no-encoder deferral batch — only refs are
    parsed; malformed jobs stay queued for the solo drain."""
    if limit < 1:
        return []
    out: list[_Stage] = []
    for sib in ingester.jobs.pending_siblings(job, limit=limit):
        if not _sibling_allowed(job, sib):
            continue
        try:
            sid, rev = _parse_source_refs(sib.get("input_refs") or {})
        except VerbatimError:
            continue
        out.append(_Stage(sib, source_id=sid, revision=rev))
    return out


def handle_source_embed(job: dict[str, Any], owner: str, ingester: Any) -> None:
    """Encode one source revision into ``source_vectors``.

    The encoder is a configured capability: absent → recorded deferral
    (``source_vector_ready`` defers ``encoder_unavailable``, mirroring
    ``_do_embed``'s ``_note`` path — a capability gap is a recorded
    outcome, not a failure and not a pending lie). A configured-but-down
    backend raises ENCODER_UNAVAILABLE (retryable); a malformed blob is
    VECTOR_INVALID (permanent) and lands on the receipt through
    ``_fail_job_obligations``.
    """
    refs = job["input_refs"]
    try:
        source_id = _require_ref(refs, "source_id")
        revision = _require_int_ref(refs, "revision")
    except VerbatimError:
        raise
    try:
        _embed_body(ingester, job, owner, source_id, revision)
    except VerbatimError as exc:
        if not exc.retryable:
            _fail_source_obligation(
                ingester, job, owner, source_id, revision,
                CapabilityName.SOURCE_VECTOR_READY, exc.code.value,
            )
        raise


def _embed_body(
    ingester: Any, job: dict[str, Any], owner: str,
    source_id: str, revision: int,
) -> None:
    refs = job["input_refs"]
    _require_v5(ingester)
    expected_cv = _opt_int(refs, "control_version")
    pinned_gen = _opt_int(refs, "generation")
    payload_ns = _opt_str(refs, "namespace")

    encoder = getattr(ingester, "encoder", None)
    if encoder is None:
        encoder = getattr(ingester.store, "encoder", None)

    def _note(reason: str) -> None:
        def _apply(conn: Any) -> dict[str, Any]:
            seq = _append_event(
                conn,
                ingester,
                job,
                "source_embed_skipped",
                {
                    "job_id": job["job_id"],
                    "source_id": source_id,
                    "revision": revision,
                    "reason": reason,
                    "producer": PRODUCER_EMBED,
                    "stage": "t0_enriched",
                },
            )
            settled = _defer_source(
                conn,
                ingester,
                source_id,
                revision,
                CapabilityName.SOURCE_VECTOR_READY,
                reason,
            )
            return {"encoded": 0, "note": reason,
                    "receipts_settled": settled, "event_seq": seq}

        # V8-13.05: same-scope siblings record the identical deferral in
        # the same commit — one transaction for the whole batch.
        notes = _stage_embed_notes(
            ingester, job, limit=_coalesce_limit(ingester)
        )
        with ingester.store.tx() as conn:
            ingester._commit_effects(conn, job, owner, "source_embed", _apply)
            _drain_siblings(
                conn,
                ingester,
                job,
                owner,
                kind_value="source_embed",
                stages=notes,
                apply_stage=lambda c, st: _apply_embed_note(
                    c, ingester, stage=st, reason=reason
                ),
                capability=CapabilityName.SOURCE_VECTOR_READY,
            )

    if encoder is None:
        _note("encoder_unavailable")
        return

    # Same pre-declaration as source_project: permanent failures below
    # land on the obligation row only when it exists.
    _declare_obligations(
        ingester, job, owner, source_id, revision,
        CapabilityName.SOURCE_VECTOR_READY,
    )

    enc_id = encoder.encoder_id
    want = refs.get("encoder") or refs.get("encoder_id")
    if want is not None and want != enc_id:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"source_embed job targets encoder {want!r}; configured is {enc_id!r}",
        )
    # V8-08.05 ``dense.tier``: a declared tier pin constrains which
    # encoder may write this job's vectors. An unprovisioned tier is a
    # recorded deferral (``dense_tier_unprovisioned``) — writing under a
    # different encoder would mint vectors attributed to a tier the job
    # never declared. The read-side tier selection lives in
    # ``retrieval/v7/dense.py`` (cross-module handoff; this gate covers
    # the write side).
    tier = _resolve_dense_tier(ingester, refs)
    if tier is not None and not _dense_tier_ok(ingester, encoder, tier):
        _note("dense_tier_unprovisioned")
        return
    if not encoder.available():
        raise VerbatimError(
            ErrorCode.ENCODER_UNAVAILABLE,
            "encoder backend is unavailable",
            retryable=True,
        )
    # V8-08.02 ``dense.embed_batch`` — the merged-commit/encode bound,
    # resolved once off this job's declared channels.
    bound = _embed_batch(ingester, refs)

    # Verified bytes + inference strictly before the write tx (V2-39.07).
    text = _verified_text(ingester, source_id, revision)
    payload = bytes(ingester.sources.payload(source_id, revision) or b"")
    blobs = encoder.encode([text])
    if len(blobs) != 1:
        raise VerbatimError(
            ErrorCode.VECTOR_INVALID,
            f"encoder returned {len(blobs)} vectors for 1 text",
        )
    blob = bytes(blobs[0])
    # Codec gate: dimension/shape/finite checks ride the same validator
    # the claim-embedding path enforces — a bad blob is a permanent
    # VECTOR_INVALID, never written.
    Float32Codec.validate_blob(blob, encoder.dimensions)
    digest = _vector_digest(enc_id, blob)
    # The unit-vector space is the model id + the contextual-header
    # format version (V7-07.09) — the dense lane derives the same id from
    # the caller's pinned encoder id.
    unit_space = _matrix.unit_vector_encoder_id(enc_id)

    primary = _Stage(
        job,
        source_id=source_id,
        revision=revision,
        expected_cv=expected_cv,
        pinned_gen=pinned_gen,
        payload_ns=payload_ns,
        text=text,
        payload=payload,
        blob=blob,
        digest=digest,
        unit_space=unit_space,
    )
    # V8-13.05: stage + batch-encode due same-scope siblings for the
    # shared commit — the §23 ``dense.embed_batch`` bound (V8-08.02)
    # governs how much pending unit work rides this commit.
    siblings, batch_stats = _stage_embed_siblings(
        ingester, encoder, job, limit=_coalesce_limit(ingester)
    )

    retries_left = _PLAN_STALE_RETRIES
    while True:
        # Unit-slice snapshot + encode stay outside the write tx; the
        # commit re-verifies the slice before writing a byte of it.
        primary.unit_plan = _prepare_unit_vectors(
            ingester,
            encoder,
            source_id,
            revision,
            payload,
            batch_rows=bound.rows,
        )

        def _apply(conn: Any) -> dict[str, Any]:
            result = _apply_embed(conn, ingester, stage=primary, enc_id=enc_id)
            if siblings:
                result["coalesced"] = _drain_siblings(
                    conn,
                    ingester,
                    job,
                    owner,
                    kind_value="source_embed",
                    stages=siblings,
                    apply_stage=lambda c, st: _apply_embed(
                        c, ingester, stage=st, enc_id=enc_id, coalesced=True
                    ),
                    capability=CapabilityName.SOURCE_VECTOR_READY,
                )
            # The resolved §23 arm + what it admitted — honest
            # provenance for the coalesced write (V8-08.02).
            result["embed_batch"] = {
                "rows_bound": batch_stats["rows_bound"],
                "max_age_ms": batch_stats["max_age_ms"],
                "merged_rows": batch_stats["merged_rows"],
                "staged_siblings": len(siblings),
                "closed_by": batch_stats["closed_by"],
            }
            return result

        try:
            with ingester.store.tx() as conn:
                ingester._commit_effects(
                    conn, job, owner, "source_embed", _apply
                )
            return
        except _PlanStale:
            if retries_left <= 0:
                # Sustained unit-plane churn — the queue's retry/backoff
                # is the honest bound (the job is retried, never lost).
                raise VerbatimError(
                    ErrorCode.STALE_DEPENDENCY,
                    f"units for {source_id}@{revision} kept drifting "
                    "across embed retries",
                    retryable=True,
                )
            retries_left -= 1


# ----------------------------------------------------------------------
# source_backfill
# ----------------------------------------------------------------------


def _backfill_members(
    conn: Any, last_source_id: str, namespace: Optional[str], limit: int
) -> list[str]:
    """Next bounded window of source ids after the cursor.

    Deterministic ``source_id`` order; namespace filtering honors the
    state-first authority rule (control artifact when present, else the
    persisted scope partition) so a scan never crosses partitions.
    """
    params: list[Any] = [last_source_id]
    sql = (
        "SELECT s.source_id FROM sources s WHERE s.source_id > ?"
    )
    if namespace:
        sql += (
            " AND (s.scope_id = ? OR EXISTS (SELECT 1 FROM source_state ss"
            " WHERE ss.source_id = s.source_id AND ss.namespace = ?))"
        )
        params += [namespace, namespace]
    sql += " ORDER BY s.source_id LIMIT ?"
    params.append(limit)
    return [str(r[0]) for r in conn.execute(sql, params).fetchall()]


def handle_source_backfill(job: dict[str, Any], owner: str, ingester: Any) -> None:
    """Bounded resumable projection backfill over ``backfill_cursor``.

    One leased job = one bounded batch. Per source: adopt missing
    ``source_state`` (governed ``recorded`` + unresolved head), re-check
    the publication predicate, project every retained revision (capped),
    embed when requested and an encoder is configured — each source under
    its own savepoint so a single bad source records its disposition and
    the batch still commits. The cursor advance, per-source dispositions,
    the batch receipt event, and the continuation job all commit in the
    same fenced transaction (V5-08.17): crash-before-commit retries the
    identical window; crash-after-commit resumes from the durable cursor.
    ``done`` flips only when the scan window returned no further sources.

    The per-revision ``link_near``/``detect_update_candidates`` namespace
    scans run ahead of the commit on a WAL snapshot (phase 2.5 —
    ``_backfill_prescan``, the batch twin of ``_project_body``'s read
    phase) so they stay off the write lock; a verified dependency
    fingerprint at commit time proves the plans' inputs unchanged, any
    drift aborts the whole batch (rollback — cursor, dispositions, and
    continuation are never left partial) and replans on a fresh snapshot,
    and the bounded budget ends in the fused in-transaction scans — the
    same honest bound this handler always had.
    """
    refs = job["input_refs"]
    _require_v5(ingester)
    namespace = _opt_str(refs, "namespace")
    job_key = _opt_str(refs, "job_key") or (
        f"source_backfill:{namespace or 'all'}"
    )
    batch_size = refs.get("batch_size")
    if batch_size is None:
        batch_size = DEFAULT_BACKFILL_BATCH
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or batch_size > MAX_BACKFILL_BATCH
    ):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"source_backfill batch_size must be 1..{MAX_BACKFILL_BATCH}",
        )
    want_embed = bool(refs.get("embed", True))
    pinned_gen = _opt_int(refs, "generation")

    encoder = getattr(ingester, "encoder", None) or getattr(
        ingester.store, "encoder", None
    )
    embed_live = bool(
        want_embed
        and encoder is not None
        and encoder.available()
    )

    # Phase 1 — cursor + window selection on a read snapshot.
    with ingester.store.read() as conn:
        cur = repos_v5.get(conn, "backfill_cursor", {"job_key": job_key})
        last = str(cur["last_source_id"]) if cur else ""
        already_done = bool(cur["done"]) if cur else False
        batch_ids = (
            []
            if already_done
            else _backfill_members(conn, last, namespace, batch_size)
        )

    # Phase 2 — verified payload reads + derivation outside the write tx.
    staged: list[dict[str, Any]] = []
    for sid in batch_ids:
        with ingester.store.read() as conn:
            rev_rows = conn.execute(
                "SELECT revision, event_us, captured_us, metadata_json,"
                " hex(payload_hmac) FROM source_revisions"
                " WHERE source_id = ? ORDER BY revision LIMIT ?",
                (sid, MAX_REVISIONS_PER_SOURCE + 1),
            ).fetchall()
        capped = len(rev_rows) > MAX_REVISIONS_PER_SOURCE
        revs = []
        for (rv, ev_us, cap_us, meta_json, hmac_hex) in rev_rows[
            :MAX_REVISIONS_PER_SOURCE
        ]:
            payload = ingester.sources.payload(sid, int(rv))
            if payload is None:
                revs.append({"revision": int(rv), "text": None,
                             "hmac_hex": hmac_hex})
                continue
            try:
                text = bytes(payload).decode("utf-8")
            except UnicodeDecodeError:
                revs.append({"revision": int(rv), "text": None,
                             "decode_failed": True, "hmac_hex": hmac_hex})
                continue
            anchor = _anchor_for(
                {
                    "event_us": ev_us,
                    "captured_us": cap_us,
                    "metadata_json": meta_json,
                }
            )
            d = _derive(text, anchor)
            blob = None
            vector_error = None
            if embed_live:
                try:
                    blob = bytes(encoder.encode([text])[0])
                    Float32Codec.validate_blob(blob, encoder.dimensions)
                except VerbatimError as exc:
                    vector_error = exc.code.value
                except Exception:
                    vector_error = "encode_failed"
            revs.append(
                {
                    "revision": int(rv),
                    "text": text,
                    "derived": d,
                    "blob": blob,
                    "vector_error": vector_error,
                    "hmac_hex": hmac_hex,
                }
            )
        staged.append({"source_id": sid, "capped": capped, "revisions": revs})

    # Phase 2.5 — batch prescan on a WAL snapshot (same contract as
    # ``_project_body``): the per-revision ``link_near`` /
    # ``detect_update_candidates`` namespace scans otherwise run inside
    # the fenced commit below, multiplying the writer's hold window by
    # batch_size × revisions — past every peer's 250 ms busy cap on a
    # populated namespace. The commit re-fingerprints the dependency
    # tables first; drift aborts (rollback, never partial state) and
    # replans on a fresh snapshot, and the bounded budget ending in the
    # fused path is the honest bound this handler always had.
    _t0 = time.monotonic()
    pre = _backfill_prescan(ingester, staged, namespace)
    prescan_ms = (time.monotonic() - _t0) * 1e3
    strict = pre is not None and _PLAN_STALE_RETRIES > 0
    retries_left = (
        _PLAN_STALE_RETRIES
        if prescan_ms >= _REPLAN_MIN_SCAN_MS
        else 0
    )

    def _apply(conn: Any) -> dict[str, Any]:
        # Committed-state fingerprint before the batch's own writes —
        # ``_state.adopt``/projection upserts touch watched tables.
        fp0 = _deps_fingerprint(conn, pre.mode) if pre is not None else None
        if (
            strict
            and pre is not None
            and fp0 is not None
            and fp0 != pre.fp
        ):
            raise _PlanStale()
        generation = _projection_generation(conn, ingester, pinned_gen)
        dispositions: dict[str, str] = {}
        covered = 0
        for item in staged:
            sid = item["source_id"]
            conn.execute("SAVEPOINT backfill_src")
            try:
                row = conn.execute(
                    "SELECT 1 FROM source_state WHERE source_id = ?", (sid,)
                ).fetchone()
                if row is None:
                    srow = conn.execute(
                        "SELECT scope_id FROM sources WHERE source_id = ?",
                        (sid,),
                    ).fetchone()
                    _state.adopt(
                        conn,
                        sid,
                        (srow[0] if srow else (namespace or "")),
                        producer=PRODUCER_BACKFILL,
                        store=ingester.store,
                    )
                state = transitions.assert_publishable(conn, sid)
                resolved_ns = state.namespace
                if namespace and resolved_ns != namespace:
                    # Membership filter passed on scope_id but the control
                    # artifact binds a different partition — skip, never
                    # write across partitions.
                    dispositions[sid] = "namespace_mismatch"
                    conn.execute("RELEASE backfill_src")
                    continue
                revs_written = 0
                skipped_held = False
                vector_errors = False
                for r in item["revisions"]:
                    rv = r["revision"]
                    if r.get("decode_failed"):
                        dispositions[sid] = "decode_failed"
                        continue
                    if r.get("text") is None:
                        dispositions[sid] = "purged"
                        continue
                    if ingester._source_held(conn, sid, rv) or (
                        ingester._source_suppressed(conn, sid, rv)
                    ):
                        skipped_held = True
                        continue
                    d = r["derived"]
                    _write_lexical(
                        conn,
                        source_id=sid,
                        revision=rv,
                        namespace=resolved_ns,
                        generation=generation,
                        tokens_text=d["tokens_text"],
                        doc_len=d["doc_len"],
                        digest=d["digest"],
                    )
                    _write_postings(
                        conn,
                        namespace=resolved_ns,
                        source_id=sid,
                        revision=rv,
                        generation=generation,
                        identifiers=d["identifiers"],
                        entities=d["entities"],
                    )
                    _write_enrichment(
                        conn,
                        source_id=sid,
                        revision=rv,
                        mem_type=d["mem_type"],
                        pol=d["polarity"],
                        temporal=d["temporal"],
                        identifiers=d["identifiers"],
                        entities=d["entities"],
                    )
                    from ..querying.updates import write_term_postings

                    write_term_postings(
                        conn,
                        namespace=resolved_ns,
                        source_id=sid,
                        revision=rv,
                        text=r["text"],
                    )
                    # Planned commit: the prescan's fingerprint held
                    # (checked at the top of this tx) and this source's
                    # namespace resolved as predicted — reuse the
                    # snapshot-computed near/update plans; anything else
                    # runs the fused in-transaction scans.
                    plan_pair = None
                    if (
                        pre is not None
                        and fp0 is not None
                        and fp0 == pre.fp
                        and pre.ns.get(sid) == resolved_ns
                    ):
                        plan_pair = pre.plans.get((sid, rv))
                    if plan_pair is not None:
                        _run_dedup(
                            conn,
                            source_id=sid,
                            revision=rv,
                            namespace=resolved_ns,
                            payload_hmac_hex=r["hmac_hex"],
                            digest=d["digest"],
                            signature=d["signature"],
                            text_fields=None,
                            store=ingester.store,
                            near_plan=plan_pair[0],
                        )
                        from ..querying.updates import (
                            persist_update_candidates,
                        )

                        persist_update_candidates(conn, plan_pair[1])
                    else:
                        _run_dedup(
                            conn,
                            source_id=sid,
                            revision=rv,
                            namespace=resolved_ns,
                            payload_hmac_hex=r["hmac_hex"],
                            digest=d["digest"],
                            signature=d["signature"],
                            text_fields=None,
                            store=ingester.store,
                        )
                        _detect_updates(
                            conn,
                            namespace=resolved_ns,
                            source_id=sid,
                            revision=rv,
                            text=r["text"],
                            mem_type=d["mem_type"],
                            pol=d["polarity"],
                            temporal=d["temporal"],
                            identifiers=d["identifiers"],
                            entities=d["entities"],
                            store=ingester.store,
                        )
                    if embed_live and r["blob"] is not None:
                        repos_v5.upsert(
                            conn,
                            "source_vectors",
                            {
                                "source_id": sid,
                                "revision": rv,
                                "namespace": resolved_ns,
                                "encoder": encoder.encoder_id,
                                "generation": generation,
                                "vector": r["blob"],
                                "digest": _vector_digest(
                                    encoder.encoder_id, r["blob"]
                                ),
                            },
                        )
                    revs_written += 1
                    if r.get("vector_error") is not None:
                        vector_errors = True
                    # Only obligations that already exist may settle —
                    # backfill never materializes a source branch that
                    # capture did not declare (V5-08.17).
                    engine = getattr(ingester, "_readiness", None)
                    if engine is not None:
                        for rid in engine._existing_receipts(
                            conn, receipt_ids_for_source(conn, sid, rv)
                        ):
                            engine.try_fulfill(
                                conn, rid, CapabilityName.SCREENED
                            )
                            engine.try_fulfill(
                                conn, rid, CapabilityName.SOURCE_LEXICAL_READY
                            )
                            if embed_live and r["blob"] is not None:
                                engine.try_fulfill(
                                    conn, rid,
                                    CapabilityName.SOURCE_VECTOR_READY,
                                )
                if sid not in dispositions:
                    if skipped_held:
                        dispositions[sid] = "held_partial"
                    elif item["capped"]:
                        dispositions[sid] = "revision_cap"
                    elif vector_errors:
                        dispositions[sid] = "vector_failed"
                    else:
                        dispositions[sid] = "projected"
                if revs_written:
                    covered += 1
            except VerbatimError as exc:
                conn.execute("ROLLBACK TO backfill_src")
                conn.execute("RELEASE backfill_src")
                dispositions[sid] = f"error:{exc.code.value}"
                continue
            conn.execute("RELEASE backfill_src")

        done = not batch_ids or len(batch_ids) < batch_size
        new_last = batch_ids[-1] if batch_ids else last
        repos_v5.upsert(
            conn,
            "backfill_cursor",
            {
                "job_key": job_key,
                "last_source_id": new_last,
                "generation": generation,
                "done": 1 if done else 0,
                "updated_at": rfc3339(now_us()),
            },
        )
        continuation = None
        if not done:
            # The next bounded batch rides the same queue — enqueued
            # in-commit so a crash resumes from the durable cursor rather
            # than stranding progress (no dedup/op key: the cursor row IS
            # the idempotence receipt for backfill work).
            continuation = ingester.jobs.enqueue(
                conn,
                job["scope_id"],
                JobKind.SOURCE_BACKFILL,
                {
                    "job_key": job_key,
                    "namespace": namespace,
                    "batch_size": batch_size,
                    "embed": want_embed,
                    "generation": generation,
                    "producer": PRODUCER_BACKFILL,
                },
                lane="maintenance" if ingester.jobs._has_lane else None,
            )
        seq = _append_event(
            conn,
            ingester,
            job,
            "source_backfill_batch",
            {
                "job_key": job_key,
                "namespace": namespace,
                "batch": len(batch_ids),
                "covered": covered,
                "done": done,
                "generation": generation,
                "producer": PRODUCER_BACKFILL,
                # Bounded by batch_size — the honest per-source record of
                # what this pass covered, skipped, or failed.
                "dispositions": dispositions,
            },
        )
        return {
            "job_key": job_key,
            "namespace": namespace,
            "batch": len(batch_ids),
            "covered": covered,
            "done": done,
            "last_source_id": new_last,
            "generation": generation,
            "dispositions": dispositions,
            "embed": embed_live,
            "continuation_job_id": continuation,
            "event_seq": seq,
        }

    while True:
        try:
            with ingester.store.tx() as conn:
                ingester._commit_effects(
                    conn, job, owner, "source_backfill", _apply
                )
            return
        except _PlanStale:
            if retries_left > 0:
                retries_left -= 1
                # Fresh snapshot, fresh plans, another strict commit —
                # the batch's namespace scans stay off the write lock.
                _t0 = time.monotonic()
                new_pre = _backfill_prescan(ingester, staged, namespace)
                prescan_ms = (time.monotonic() - _t0) * 1e3
                if new_pre is not None:
                    pre = new_pre
                    strict = True
                    if prescan_ms < _REPLAN_MIN_SCAN_MS:
                        retries_left = 0
                    continue
                # An empty prescan after a real stale abort leaves
                # nothing to verify — fused is the only remaining way
                # to see the live state.
            # Planned commits exhausted: the fused path is the honest
            # bound — a backfill batch must still complete under
            # genuinely sustained churn. When the namespace quieted in
            # the meantime the commit's own fingerprint check can still
            # reuse the last plan instead of scanning.
            strict = False


# ----------------------------------------------------------------------
# enqueue seams
# ----------------------------------------------------------------------


def _receipt_target(
    conn: Any, receipt_id: str
) -> Optional[tuple[str, int]]:
    """Resolve a readiness receipt id to its ``(source_id, revision)``.

    Reads on the caller's conn — the capture path invokes this inside the
    accept transaction, where the just-written rows are only visible.
    ``rc_ingest:{sid}:{rev}`` parses directly; ``cr_*`` ids reverse through
    ``source_envelopes`` (the deterministic digest is one-way, so the scan
    is the honest lookup — mirrors ``ReadinessEngine._ensure_receipt``).
    """
    if receipt_id.startswith("rc_ingest:"):
        sid, _, rev = receipt_id[len("rc_ingest:"):].rpartition(":")
        try:
            return sid, int(rev)
        except ValueError:
            return None
    if receipt_id.startswith("cr_"):
        from ..evidence.receipts import _receipt_id

        if not has_table(conn, "source_envelopes"):
            return None
        for sid, rev, kind in conn.execute(
            "SELECT DISTINCT source_id, revision, envelope_kind"
            " FROM source_envelopes"
        ).fetchall():
            try:
                if _receipt_id(str(sid), int(rev), str(kind)) == receipt_id:
                    return str(sid), int(rev)
            except Exception:
                continue
    return None


def enqueue_source_jobs(
    conn: Any,
    store: Any,
    *,
    receipt_id: str,
    queue: Any = None,
) -> dict[str, Any]:
    """Declare the source branch's work for one accepted capture.

    Called by the facade inside the capture transaction: resolves the
    receipt to its source revision, then enqueues ``source_project`` +
    ``source_embed`` with generation-scoped dedup/operation keys — so a
    retried capture converges on the same durable jobs, a redelivered
    job replays its operation receipt, and a later projection-generation
    bump mints fresh keys (reprojection is never swallowed as replay).
    All reads run on ``conn`` — the capture's own rows are still
    uncommitted. Returns ``{"job_ids": [...], "source_id", "revision"}``;
    raises a typed error (the facade records ``deferred``) when the
    target cannot be resolved or the v5 plane is unprovisioned.
    """
    target = _receipt_target(conn, receipt_id)
    if target is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            f"cannot resolve source for receipt {receipt_id!r}",
        )
    source_id, revision = target
    src = conn.execute(
        "SELECT scope_id FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    if src is None:
        raise VerbatimError(
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            f"source {source_id!r} not found for receipt {receipt_id!r}",
        )
    scope_id = src[0]
    st = conn.execute(
        "SELECT namespace, control_version FROM source_state"
        " WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    namespace = st[0] if st else scope_id
    control_version = int(st[1]) if st else None
    generation = int(store._meta_get(conn, "projection_generation") or 0)

    # Declare the source branch on this receipt in the same tx (V5-08.15)
    # — the facade records ``include_source=True`` before calling, so for
    # it this is a no-op; SDK/legacy capture paths that route through the
    # seam get the same durable declaration. Insert-if-missing; never
    # rewrites an existing row.
    try:
        from ..readiness import SOURCE_CAPS, ReadinessEngine
    except ImportError:
        ReadinessEngine = None  # readiness plane unprovisioned
    if ReadinessEngine is not None:
        engine = ReadinessEngine(store)
        if engine.available:
            engine.record_obligations(
                conn,
                receipt_id,
                scope_id,
                [CapabilityName.SCREENED, *SOURCE_CAPS],
            )

    jobs = queue
    if jobs is None:
        from .queue import JobQueue

        jobs = JobQueue(store)
    refs_common = {
        "source_id": source_id,
        "revision": revision,
        "namespace": namespace,
        "scope_id": scope_id,
        "generation": generation,
        "producer": PRODUCER_PROJECT,
    }
    if control_version is not None:
        refs_common["control_version"] = control_version
    job_ids = []
    durable = jobs.supports_durability
    for kind, extra in (
        (JobKind.SOURCE_PROJECT, {"utf8_ok": True}),
        (JobKind.SOURCE_EMBED, {}),
    ):
        refs = dict(refs_common) | extra
        key = f"{kind.value}:{source_id}:{revision}:g{generation}"
        job_ids.append(
            jobs.enqueue(
                conn,
                scope_id,
                kind,
                refs,
                dedup_key=store.hmac(key.encode("utf-8")),
                operation_key=key if durable else None,
            )
        )
    return {
        "job_ids": job_ids,
        "source_id": source_id,
        "revision": revision,
        "namespace": namespace,
        "generation": generation,
    }


def enqueue_source_backfill(
    conn: Any,
    ingester: Any,
    *,
    scope_id: str,
    namespace: Optional[str] = None,
    job_key: Optional[str] = None,
    batch_size: int = DEFAULT_BACKFILL_BATCH,
    embed: bool = True,
) -> str:
    """Enqueue a bounded ``source_backfill`` plan (V5-08.17).

    The plan is durable and scope-authorized: it rides the caller's
    transaction, so plan construction and the first batch job commit
    together. The cursor row appears at first drain; coverage stays
    partial until ``done`` commits — callers read ``backfill_cursor``
    for the honest progress answer.
    """
    key = job_key or f"source_backfill:{namespace or 'all'}"
    refs: dict[str, Any] = {
        "job_key": key,
        "namespace": namespace,
        "batch_size": batch_size,
        "embed": bool(embed),
        "producer": PRODUCER_BACKFILL,
    }
    return ingester.jobs.enqueue(
        conn,
        scope_id,
        JobKind.SOURCE_BACKFILL,
        refs,
        dedup_key=ingester.store.hmac(key.encode("utf-8")),
        lane="maintenance" if ingester.jobs._has_lane else None,
    )


__all__ = [
    "DEFAULT_BACKFILL_BATCH",
    "MAX_BACKFILL_BATCH",
    "PRODUCER_BACKFILL",
    "PRODUCER_EMBED",
    "PRODUCER_PROJECT",
    "enqueue_source_backfill",
    "enqueue_source_jobs",
    "handle_source_backfill",
    "handle_source_embed",
    "handle_source_project",
]
