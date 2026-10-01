"""Evaluation harness — drives the public Engine pipeline end to end.

The harness is deliberately conservative about what it claims to measure:

* **Extraction coverage** counts claim heads and claim→span evidence links
  recorded in the store — the persisted interpretive layer — *not* recall
  hits.  These are reported separately so a broken projection cannot
  masquerade as an extraction failure or vice versa.
* **Recall metrics** are scored against gold ``source_id`` links: a query
  hits iff a returned item's evidence span names the source that produced
  the expected statement.  Fuzzy text similarity is never used for scoring.
* **Latency** uses ``time.perf_counter_ns``; it is a measurement, not an
  input to any expected result, so the run stays logically deterministic.
* **Operator-assisted admission.**  The default configuration requires
  review before claims become active.  The harness drives the public
  ``apply_transition`` API exactly the way the CLI review flow does, and
  the report discloses this instead of pretending unassisted operation.
* **FTS projection rebuild.**  ``rebuild_fts=True`` re-indexes every claim at
  the current generation through the normal indexing path. The harness records
  both pre- and post-rebuild recall so generation-stranding regressions become
  visible instead of being hidden by maintenance.

Everything runs in a fresh temporary store; no network, no wall-clock-
derived expectations.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from verbatim.api import open_store
from verbatim.config import config_from_mapping
from verbatim.core.types import (
    Provenance,
    RecallMode,
    RecallRequest,
    SourceEnvelope,
    SourceKind,
    TimeInterval,
    TransitionCommand,
)
from verbatim.host import LocalHost

from .corpus import Corpus, Query, Statement

_DEFAULT_CFG: dict[str, Any] = {
    "mode": "offline_rules",
    "capture": {"enabled": True},
    "admission": {"require_review": True},
}


# ---------------------------------------------------------------------------
# result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    kind: str
    query_text: str
    expected_sources: tuple[str, ...]
    returned_sources: tuple[str, ...]
    hit: bool
    latency_ns: int
    n_items: int
    warnings: tuple[str, ...] = ()


@dataclass
class HarnessResult:
    """Everything the report needs; JSON-serializable via ``to_dict``."""

    corpus: dict[str, Any]
    config: dict[str, Any]
    ingest: dict[str, Any] = field(default_factory=dict)
    processing: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    recall: dict[str, Any] = field(default_factory=dict)
    fts: dict[str, Any] = field(default_factory=dict)
    store: dict[str, Any] = field(default_factory=dict)
    defects: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    per_query: list[QueryResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "config": self.config,
            "ingest": self.ingest,
            "processing": self.processing,
            "coverage": self.coverage,
            "recall": self.recall,
            "fts": self.fts,
            "store": self.store,
            "defects": self.defects,
            "notes": self.notes,
            "elapsed_s": self.elapsed_s,
            "per_query": [
                {
                    "query_id": q.query_id,
                    "kind": q.kind,
                    "query": q.query_text,
                    "expected": list(q.expected_sources),
                    "returned": list(q.returned_sources),
                    "hit": q.hit,
                    "latency_ns": q.latency_ns,
                    "n_items": q.n_items,
                    "warnings": list(q.warnings),
                }
                for q in self.per_query
            ],
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _envelope(stmt: Statement, scope) -> SourceEnvelope:
    return SourceEnvelope(
        origin="eval-corpus",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id=stmt.speaker_id,
        payload=stmt.text.encode("utf-8"),
        event_us=stmt.event_us,
        captured_us=stmt.event_us,
        provenance=Provenance.DIRECT_USER,
        external_id=stmt.stmt_id,
        revision=1,
    )


def _claim_heads(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """claim_id -> {revision, state, predicate} for the head revision."""
    rows = conn.execute(
        "SELECT cr.claim_id, cr.revision, cr.state, c.predicate"
        " FROM claim_revisions cr"
        " JOIN (SELECT claim_id, MAX(revision) AS mr FROM claim_revisions"
        "       GROUP BY claim_id) h ON h.claim_id = cr.claim_id"
        "   AND h.mr = cr.revision"
        " JOIN claims c ON c.claim_id = cr.claim_id"
    ).fetchall()
    return {
        r[0]: {"revision": r[1], "state": r[2], "predicate": r[3]}
        for r in rows
    }


def _source_claim_map(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """source_id -> [claim_id] via span -> claim_evidence links."""
    rows = conn.execute(
        "SELECT s.source_id, ce.claim_id FROM claim_evidence ce"
        " JOIN spans s ON s.span_id = ce.span_id"
    ).fetchall()
    out: dict[str, list[str]] = {}
    for sid, cid in rows:
        out.setdefault(sid, []).append(cid)
    return out


def _fts_text_for(conn: sqlite3.Connection, claim_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT f.text FROM facts_fts f"
        " JOIN fts_rows r ON r.row_id = f.fts_row_id"
        " WHERE r.claim_id = ? ORDER BY r.projection_generation DESC LIMIT 1",
        (claim_id,),
    ).fetchone()
    return row[0] if row else None


def _item_source_id(item: Any) -> Optional[str]:
    span = getattr(item, "span", None)
    if span is None:
        return None
    return getattr(span, "source_id", None)


def _percentile(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i] / 1e6  # ns -> ms


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------


def run_harness(
    corpus: Corpus,
    *,
    work_dir: Optional[str] = None,
    cfg_overrides: Optional[dict[str, Any]] = None,
    approve: bool = True,
    rebuild_fts: bool = True,
    query_limit: Optional[int] = None,
    top_k: int = 5,
) -> HarnessResult:
    """Run the corpus through a fresh engine and score the results.

    Parameters
    ----------
    approve:
        Drive ``apply_transition(effect='admit')`` for every pending head —
        the same operation an operator performs in the CLI review queue.
    rebuild_fts:
        Re-index every claim at the current projection generation after all
        transitions. Both pre- and post-rebuild recall numbers are reported;
        they should not diverge because ordinary transitions index in-tx.
    """
    t_start = time.perf_counter()
    cfg_map = dict(_DEFAULT_CFG)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            cfg_map[k] = v
    cfg = config_from_mapping(cfg_map)

    tmp = work_dir or tempfile.mkdtemp(prefix="verbatim-eval-")
    host = LocalHost(profile_id="eval", principal_id="me", conversation_id="eval-c1")
    eng = open_store(tmp, cfg, host, create=True)
    scope = host.default_scope()
    store = eng.store

    res = HarnessResult(
        corpus={
            "seed": corpus.seed,
            "sha256": corpus.sha256(),
            "statements": len(corpus.statements),
            "queries": len(corpus.queries),
            "update_pairs": len(corpus.update_pairs),
            "contradiction_pairs": len(corpus.contradiction_pairs),
        },
        config={"mode": cfg.mode.value, "require_review": cfg.admission.require_review,
                "approve": approve, "rebuild_fts": rebuild_fts, "top_k": top_k},
        store={"path": tmp},
    )

    # ------------------------------------------------------------------
    # 1. ingest
    # ------------------------------------------------------------------
    accepted = 0
    duplicates = 0
    stmt_to_source: dict[str, str] = {}
    source_to_stmt: dict[str, str] = {}
    for stmt in corpus.statements:
        receipt = eng.ingest(_envelope(stmt, scope))
        if receipt.duplicate:
            duplicates += 1
        if receipt.accepted:
            accepted += 1
            stmt_to_source[stmt.stmt_id] = receipt.accepted[0]
            source_to_stmt[receipt.accepted[0]] = stmt.stmt_id
    res.ingest = {
        "envelopes": len(corpus.statements),
        "accepted": accepted,
        "duplicates": duplicates,
    }

    # ------------------------------------------------------------------
    # 2. pending processing (harvest/admit/compare)
    # ------------------------------------------------------------------
    n_processed = eng.run_pending(limit=max(64, len(corpus.statements) * 4))
    with store.read() as conn:
        heads = _claim_heads(conn)
        src_claims = _source_claim_map(conn)
        job_states = dict(
            conn.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state").fetchall()
        )
    res.processing = {
        "jobs_processed_reported": n_processed,
        "job_states": job_states,
        "claims_created": len(heads),
        "pending": sum(1 for h in heads.values() if h["state"] == "pending"),
        "active_before_review": sum(1 for h in heads.values() if h["state"] == "active"),
        "structured_claims": sum(1 for h in heads.values() if h["predicate"]),
    }
    stmts_with_claims = sum(
        1 for sid in stmt_to_source if stmt_to_source[sid] in src_claims
    )
    res.coverage = {
        "statements_with_claims": stmts_with_claims,
        "extraction_coverage": round(stmts_with_claims / max(1, accepted), 4),
        "claim_yield": round(len(heads) / max(1, accepted), 3),
    }

    # ------------------------------------------------------------------
    # 3. operator-assisted admission (public transition API)
    # ------------------------------------------------------------------
    admit_failures = 0
    if approve:
        with store.read() as conn:
            heads = _claim_heads(conn)
        for cid, h in heads.items():
            if h["state"] != "pending":
                continue
            try:
                eng.apply_transition(
                    TransitionCommand(
                        claim_id=cid,
                        expected_revision=h["revision"],
                        effect="admit",
                        actor_id="eval-harness",
                        reason="baseline operator approval",
                    ),
                    scope=scope,
                )
            except Exception:
                admit_failures += 1
    res.processing["admit_failures"] = admit_failures

    # ------------------------------------------------------------------
    # 4. update pairs: supersede old -> new through the public API
    # ------------------------------------------------------------------
    superseded = 0
    supersede_failures = 0
    if approve:
        with store.read() as conn:
            heads = _claim_heads(conn)
        for pair in corpus.update_pairs:
            old_src, new_src = stmt_to_source.get(pair.old_stmt_id), stmt_to_source.get(pair.new_stmt_id)
            if not old_src or not new_src:
                continue
            old_claims = src_claims.get(old_src) or []
            new_claims = src_claims.get(new_src) or []
            if not old_claims or not new_claims:
                continue
            old_cid = old_claims[0]
            new_cid = new_claims[0]
            h = heads.get(old_cid)
            if not h or h["state"] != "active":
                continue
            try:
                eng.apply_transition(
                    TransitionCommand(
                        claim_id=old_cid,
                        expected_revision=h["revision"],
                        effect="supersede",
                        actor_id="eval-harness",
                        reason="corpus update pair supersession",
                        successor_claim_id=new_cid,
                        interval=TimeInterval(from_us=pair.change_us),
                    ),
                    scope=scope,
                )
                superseded += 1
                with store.read() as conn:
                    heads = _claim_heads(conn)
            except Exception:
                supersede_failures += 1
    res.processing["superseded"] = superseded
    res.processing["supersede_failures"] = supersede_failures

    with store.read() as conn:
        heads = _claim_heads(conn)
        gen_now = store.projection_generation()
        visible_now = conn.execute(
            "SELECT COUNT(DISTINCT claim_id) FROM fts_rows WHERE projection_generation = ?",
            (gen_now,),
        ).fetchone()[0]
    res.processing["active_final"] = sum(1 for h in heads.values() if h["state"] == "active")
    res.processing["superseded_final"] = sum(
        1 for h in heads.values() if h["state"] == "superseded"
    )
    res.processing["reviews_open"] = 0
    res.fts["projection_generation"] = gen_now
    res.fts["claims_indexed_at_current_generation"] = visible_now
    res.fts["rebuild_used"] = rebuild_fts
    if visible_now < res.processing["active_final"]:
        res.defects.append(
            "FTS projection generation defect: sequential transitions leave "
            f"only {visible_now} of {res.processing['active_final']} active "
            "claims indexed at the current generation; earlier claims are "
            "lexically invisible until the projection is rebuilt."
        )

    # ------------------------------------------------------------------
    # 5. recall queries — measured before and after the optional rebuild.
    # ------------------------------------------------------------------
    def _run_queries(queries: list[Query]) -> list[QueryResult]:
        out: list[QueryResult] = []
        for q in queries:
            mode = RecallMode.CURRENT if q.kind in ("point", "current", "no_answer") else RecallMode.HISTORICAL
            req = RecallRequest(
                query=q.text, scope=scope, mode=mode, limit=top_k,
                valid_at_us=q.valid_at_us,
            )
            t0 = time.perf_counter_ns()
            try:
                rr = eng.recall(req)
                items = list(rr.items)
                warns = tuple(rr.warnings or ())
            except Exception as exc:  # keep failures in the denominator
                items = []
                warns = (f"error:{type(exc).__name__}",)
            lat = time.perf_counter_ns() - t0
            returned = tuple(
                s for s in (_item_source_id(i) for i in items) if s is not None
            )
            returned_stmts = tuple(source_to_stmt.get(s, s) for s in returned)
            hit = any(sid in set(q.expect) for sid in returned_stmts)
            out.append(QueryResult(
                query_id=q.query_id, kind=q.kind, query_text=q.text,
                expected_sources=q.expect, returned_sources=returned_stmts,
                hit=hit, latency_ns=lat, n_items=len(items), warnings=warns,
            ))
        return out

    query_list = list(corpus.queries)
    if query_limit is not None:
        query_list = query_list[:query_limit]

    pre_results = _run_queries(query_list)
    res.recall["pre_rebuild"] = _summarize(pre_results)

    if rebuild_fts:
        # Re-index every claim that was ever indexed — active AND inactive.
        # HISTORICAL recall legitimately surfaces superseded claims, and the
        # projection rebuild is what would place their rows at the current
        # generation; retrieval's own eligibility filter still applies.
        with store.read() as conn:
            heads = _claim_heads(conn)
            texts = {cid: _fts_text_for(conn, cid) for cid in heads}
        indexed = 0
        for cid, text in texts.items():
            if text:
                try:
                    eng._ingester.index_claim(cid, scope, text)
                    indexed += 1
                except Exception:
                    pass
        res.fts["reindexed_claims"] = indexed
        res.notes.append(
            "rebuild_fts re-indexed every previously indexed claim (active "
            "and superseded — HISTORICAL recall needs both) at the current "
            "projection generation via the ingester's index_claim path, "
            "simulating the projection-rebuild job the generation design "
            "anticipates."
        )

    post_results = _run_queries(query_list)
    res.recall.update(_summarize(post_results))
    res.per_query = post_results

    # ------------------------------------------------------------------
    # 6. store size + finish
    # ------------------------------------------------------------------
    db_path = os.path.join(tmp, f"{host.profile_id()}.db")
    try:
        res.store["size_bytes"] = os.path.getsize(db_path)
    except OSError:
        res.store["size_bytes"] = None
    res.elapsed_s = round(time.perf_counter() - t_start, 3)
    eng.close()
    return res


def _summarize(results: list[QueryResult]) -> dict[str, Any]:
    by_kind: dict[str, list[QueryResult]] = {}
    for r in results:
        by_kind.setdefault(r.kind, []).append(r)

    def _agg(rs: list[QueryResult]) -> dict[str, Any]:
        n = len(rs)
        hits = sum(1 for r in rs if r.hit)
        returned = sum(r.n_items for r in rs)
        return {
            "n": n,
            "hits": hits,
            "hit_rate": round(hits / n, 4) if n else None,
            "items_returned": returned,
            "errors": sum(1 for r in rs if any(w.startswith("error:") for w in r.warnings)),
        }

    out: dict[str, Any] = {}
    for kind, rs in sorted(by_kind.items()):
        out[kind] = _agg(rs)
    # no-answer false-positive rate: fraction returning any item
    na = by_kind.get("no_answer", [])
    if na:
        fp = sum(1 for r in na if r.n_items > 0)
        out["no_answer"]["false_positive_rate"] = round(fp / len(na), 4)
    lat = sorted(r.latency_ns for r in results)
    out["latency_ms"] = {
        "p50": round(_percentile(lat, 0.50), 3),
        "p95": round(_percentile(lat, 0.95), 3),
        "p99": round(_percentile(lat, 0.99), 3),
        "max": round(lat[-1] / 1e6, 3) if lat else 0.0,
        "n": len(lat),
    }
    return out
