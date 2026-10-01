"""F27 investigation harness — the saved 0.58/0.89 historical-recall
delta, reproduced and attributed per query (SPEC_V3 §03 item F27,
V3-03.01, V3-53.15, scenario B49, gate G0).

What F27 actually is: the saved v2 baseline report records historical
hit rate 116/200 = 0.58 before the FTS projection rebuild and
178/200 = 0.89 after it. The spec is explicit (§03, §62 risk table): the
root cause was **not** re-established in review — this module exists to
reproduce the gap through public ingestion/review/recall paths and to
make each query's outcome attributable, not to assert a diagnosis.

What this harness measures for every scored query, per phase
(``pre_rebuild`` / ``post_rebuild``):

* ``path`` — which lexical producer fed the candidate set:
  ``fts_repo`` (FtsRepo.search), ``fts_sql`` (the equivalent
  parameterized _fts_match), ``scan`` (the bounded substring fallback
  used when FTS5 is absent or can't segment the terms), or ``none``
  (the lexical stage never ran). Determined by instrumenting the same
  branch logic retrieval uses — the public API is never bypassed.
* ``expected_candidate_rank`` — the expected claim's 1-based rank in
  that producer's returned rows, or ``None`` when it was not a lexical
  candidate.
* ``expected_indexed_at_generation`` — whether the expected claim had an
  fts_rows entry at the *current* projection generation (the
  generation-stranding check: transitions index the changed claim in-tx
  at the new generation; historical probes asked at that generation can
  only see claims whose rows were populated there).
* ``hit`` — the v2 gold-source criterion, unchanged: a returned item's
  evidence span names the source that produced the expected statement.

Per-query ``delta`` and ``attribution`` then classify the outcome:
``stable_hit``, ``stable_miss``, ``generation_stranded`` (miss→hit with
the expected claim unindexed at the queried generation pre-rebuild),
``rank_shift`` (expected was a candidate both phases but its rank moved
across the packed boundary), ``packed_out`` (a lexical candidate that
never survived downstream fusion/packaging), ``gained``/``lost`` with
the same sub-classification, and ``path:<name>`` tags so a scan-vs-fts
explanation is separable from a ranking explanation.

The saved baseline numbers are embedded as the *reproduction target*
(``saved_baseline``) — they are evidence from the archived report, not
measurements this run produced; the run's own numbers live under
``recall``.
"""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, Sequence

from verbatim.api import open_store
from verbatim.config import config_from_mapping
from verbatim.core.types import (
    RecallMode,
    RecallRequest,
    TimeInterval,
    TransitionCommand,
)
from verbatim.host import LocalHost
from verbatim.retrieval import candidates as _cand

from ..corpus import Corpus, Query, generate_corpus
from ..harness import (  # reuse the exact v2 measurement helpers
    _DEFAULT_CFG,
    _claim_heads,
    _envelope,
    _fts_text_for,
    _item_source_id,
    _source_claim_map,
)

#: The archived reproduction target (eval/report.md, corpus seed 42,
#: size 1000): historical recall 116/200 pre-rebuild, 178/200 post.
SAVED_BASELINE = {
    "source": "eval/report.md (v2 baseline, corpus sha recorded there)",
    "corpus_size": 1000,
    "seed": 42,
    "historical_pre_rebuild": {"hits": 116, "n": 200, "hit_rate": 0.58},
    "historical_post_rebuild": {"hits": 178, "n": 200, "hit_rate": 0.89},
    "note": (
        "Reproduction target, not a rerun result (SPEC_V3 §03 F27, "
        "V3-53.15). A benchmark-only repair cannot close F27."
    ),
}


# ---------------------------------------------------------------------------
# path instrumentation
# ---------------------------------------------------------------------------


class PathRecorder:
    """Records which lexical producer ran during one recall call and the
    claim ids it returned, in rank order.

    The effective path for a query is the LAST producer to run inside
    the recall: ``_fts_search`` only invokes a later producer when the
    earlier one's rows will not be used (repo error → SQL fallback;
    empty FTS + unsegmented terms → bounded scan). So the tail of the
    recorded call list is the branch whose rows became candidates.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def record(self, path: str, claim_ids: Sequence[str]) -> None:
        self.calls.append((path, [str(c) for c in claim_ids]))

    @property
    def path(self) -> str:
        return self.calls[-1][0] if self.calls else "none"

    @property
    def candidate_ids(self) -> list[str]:
        return self.calls[-1][1] if self.calls else []

    def clear(self) -> None:
        self.calls.clear()


@contextmanager
def instrumented_lexical_paths(recorder: PathRecorder):
    """Wrap the three lexical producers so each call is recorded.

    Only ``eval.v3`` internals are patched, inside this context — shared
    engine modules are never modified on disk.
    """
    orig_repo_search = None
    try:
        from verbatim.storage.repos import FtsRepo

        orig_repo_search = FtsRepo.search

        def _repo_search(self, scope_ids, match_query, generation, limit=32):
            hits = orig_repo_search(
                self, scope_ids, match_query, generation, limit
            )
            recorder.record("fts_repo", [h[0] for h in hits])
            return hits

        FtsRepo.search = _repo_search
    except Exception:  # noqa: BLE001 — repo may be absent in some builds
        orig_repo_search = None

    orig_fts_match = _cand._fts_match
    orig_fallback = _cand._fts_fallback_scan

    def _fts_match_rec(conn, scope_ids, match_query, generation, limit):
        rows = orig_fts_match(
            conn, scope_ids, match_query, generation, limit
        )
        recorder.record("fts_sql", [r[0] for r in rows])
        return rows

    def _fallback_rec(conn, scope_ids, plan, generation, limit):
        rows = orig_fallback(conn, scope_ids, plan, generation, limit)
        recorder.record("scan", [r[0] for r in rows])
        return rows

    _cand._fts_match = _fts_match_rec
    _cand._fts_fallback_scan = _fallback_rec
    try:
        yield recorder
    finally:
        _cand._fts_match = orig_fts_match
        _cand._fts_fallback_scan = orig_fallback
        if orig_repo_search is not None:
            FtsRepo.search = orig_repo_search


# ---------------------------------------------------------------------------
# corpus → engine setup (same public-entry-point sequence as the v2 harness)
# ---------------------------------------------------------------------------


def _indexed_at_generation(
    conn, claim_ids: Sequence[str], generation: int
) -> dict[str, bool]:
    if not claim_ids:
        return {}
    ph = ",".join("?" for _ in claim_ids)
    rows = conn.execute(
        "SELECT DISTINCT claim_id FROM fts_rows"
        f" WHERE projection_generation = ? AND claim_id IN ({ph})",
        [generation, *claim_ids],
    ).fetchall()
    present = {r[0] for r in rows}
    return {c: c in present for c in claim_ids}


def _fts_rows_at_generation(conn, generation: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM fts_rows WHERE projection_generation = ?",
        (generation,),
    ).fetchone()[0]


def _expected_claim_ids(
    q: Query,
    stmt_to_source: dict[str, str],
    src_claims: dict[str, list[str]],
) -> list[str]:
    out: list[str] = []
    for stmt_id in q.expect:
        src = stmt_to_source.get(stmt_id)
        if not src:
            continue
        out.extend(src_claims.get(src) or [])
    return out


def _run_instrumented_query(
    eng,
    q: Query,
    scope,
    top_k: int,
    recorder: PathRecorder,
    source_to_stmt: dict[str, str],
) -> dict[str, Any]:
    mode = (
        RecallMode.CURRENT
        if q.kind in ("point", "current", "no_answer")
        else RecallMode.HISTORICAL
    )
    req = RecallRequest(
        query=q.text, scope=scope, mode=mode, limit=top_k,
        valid_at_us=q.valid_at_us,
    )
    recorder.clear()
    t0 = time.perf_counter_ns()
    try:
        rr = eng.recall(req)
        items = list(rr.items)
        warns = [str(w) for w in (rr.warnings or ())]
        error = ""
    except Exception as exc:  # keep failures in the denominator (§53.05)
        items = []
        warns = []
        error = f"{type(exc).__name__}: {exc}"
    lat = time.perf_counter_ns() - t0
    returned = tuple(
        s for s in (_item_source_id(i) for i in items) if s is not None
    )
    returned_stmts = tuple(source_to_stmt.get(s, s) for s in returned)
    hit = any(sid in set(q.expect) for sid in returned_stmts)
    return {
        "query_id": q.query_id,
        "kind": q.kind,
        "query": q.text,
        "expected_stmts": list(q.expect),
        "returned_stmts": list(returned_stmts),
        "hit": hit,
        "n_items": len(items),
        "latency_ns": lat,
        "warnings": warns,
        "error": error,
        "path": recorder.path,
        "lexical_candidate_ids": list(recorder.candidate_ids),
    }


def _phase_snapshot(
    eng, conn, queries, scope, top_k, recorder,
    stmt_to_source, source_to_stmt, src_claims,
) -> dict[str, Any]:
    generation = eng.store.projection_generation()
    per_query = []
    for q in queries:
        r = _run_instrumented_query(
            eng, q, scope, top_k, recorder, source_to_stmt
        )
        expected_claims = _expected_claim_ids(q, stmt_to_source, src_claims)
        cand_ids = r.pop("lexical_candidate_ids")
        rank = None
        for cid in expected_claims:
            if cid in cand_ids:
                rank = cand_ids.index(cid) + 1
                break
        indexed = _indexed_at_generation(conn, expected_claims, generation)
        r["expected_claim_ids"] = expected_claims
        r["expected_candidate_rank"] = rank
        r["lexical_candidate_count"] = len(cand_ids)
        r["expected_indexed_at_generation"] = any(indexed.values())
        r["projection_generation"] = generation
        per_query.append(r)
    return {
        "projection_generation": generation,
        "fts_rows_at_generation": _fts_rows_at_generation(
            conn, generation
        ),
        "fts5_available": _cand._fts5_available(conn),
        "per_query": per_query,
    }


def _attribute(pre: dict, post: dict) -> list[str]:
    """Classify the pre→post outcome for one query.

    The categories separate the hypotheses F27 must distinguish:
    generation stranding (expected claim simply had no row at the
    queried generation), ranking shifts among indexed candidates, and
    downstream packaging losses (a lexical candidate that never reached
    the packed result).
    """
    tags = [f"path_pre:{pre['path']}", f"path_post:{post['path']}"]
    if not pre["expected_indexed_at_generation"]:
        tags.append("expected_unindexed_pre")
    pre_rank = pre["expected_candidate_rank"]
    post_rank = post["expected_candidate_rank"]
    # A lexical candidate that never reached the packed result is a
    # downstream packaging loss in whichever phase it occurred — tag it
    # for stable misses as well as delta queries.
    if pre_rank is not None and not pre["hit"]:
        tags.append("packed_out_pre")
    if post_rank is not None and not post["hit"]:
        tags.append("packed_out_post")
    if pre["hit"] and post["hit"]:
        tags.append("stable_hit")
        return tags
    if not pre["hit"] and not post["hit"]:
        tags.append("stable_miss")
        return tags
    delta = "gained" if post["hit"] else "lost"
    tags.append(delta)
    if not pre["expected_indexed_at_generation"] and post.get(
        "expected_indexed_at_generation"
    ):
        tags.append("generation_stranded")
        return tags
    if pre_rank is not None and post_rank is not None and pre_rank != post_rank:
        tags.append("rank_shift")
    if pre_rank is None and post_rank is not None:
        tags.append("entered_lexical_candidates")
    if pre_rank is not None and post_rank is None:
        tags.append("left_lexical_candidates")
    if "rank_shift" not in tags and "packed_out_pre" not in tags:
        tags.append("unattributed")
    return tags


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def investigate_f27(
    *,
    corpus_size: int = 200,
    seed: int = 42,
    work_dir: Optional[str] = None,
    top_k: int = 5,
    query_kinds: Sequence[str] = ("historical",),
    approve: bool = True,
    corpus: Optional[Corpus] = None,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Reproduce the F27 setup and return the attribution report.

    Uses the identical public-path sequence as the v2 harness — ingest
    envelopes, ``run_pending``, operator ``apply_transition`` admission,
    update-pair supersession — then measures recall on the chosen query
    slice before and after re-indexing every claim at the current
    projection generation, with per-query path/rank/generation
    instrumentation.

    ``query_kinds`` defaults to the historical slice where the saved
    delta was recorded; pass additional kinds to widen the slice.
    """
    corpus = corpus or generate_corpus(size=corpus_size, seed=seed)
    cfg_map = dict(_DEFAULT_CFG)
    if cfg_overrides:
        cfg_map.update(cfg_overrides)
    cfg = config_from_mapping(cfg_map)

    tmp = work_dir or tempfile.mkdtemp(prefix="verbatim-f27-")
    host = LocalHost(
        profile_id="f27", principal_id="me", conversation_id="f27-c1"
    )
    eng = open_store(tmp, cfg, host, create=True)
    scope = host.default_scope()
    store = eng.store

    report: dict[str, Any] = {
        "schema": 1,
        "investigation": "F27",
        "requirement_ids": ["V3-03.01", "V3-53.15"],
        "scenario": "B49",
        "corpus": {
            "seed": corpus.seed,
            "sha256": corpus.sha256(),
            "statements": len(corpus.statements),
            "queries_total": len(corpus.queries),
        },
        "config": {
            "mode": cfg.mode.value,
            "require_review": cfg.admission.require_review,
            "approve": approve,
            "top_k": top_k,
            "query_kinds": list(query_kinds),
        },
        "saved_baseline": SAVED_BASELINE,
        "store_path": tmp,
    }

    try:
        # 1. ingest (public envelope path)
        stmt_to_source: dict[str, str] = {}
        source_to_stmt: dict[str, str] = {}
        for stmt in corpus.statements:
            receipt = eng.ingest(_envelope(stmt, scope))
            if receipt.accepted:
                stmt_to_source[stmt.stmt_id] = receipt.accepted[0]
                source_to_stmt[receipt.accepted[0]] = stmt.stmt_id

        # 2. pending processing + operator admission + supersession
        eng.run_pending(limit=max(64, len(corpus.statements) * 4))
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
                            actor_id="f27-harness",
                            reason="f27 reproduction admission",
                        ),
                        scope=scope,
                    )
                except Exception:
                    pass
            with store.read() as conn:
                heads = _claim_heads(conn)
                src_claims_now = _source_claim_map(conn)
            for pair in corpus.update_pairs:
                old_src = stmt_to_source.get(pair.old_stmt_id)
                new_src = stmt_to_source.get(pair.new_stmt_id)
                if not old_src or not new_src:
                    continue
                old_claims = src_claims_now.get(old_src) or []
                new_claims = src_claims_now.get(new_src) or []
                if not old_claims or not new_claims:
                    continue
                h = heads.get(old_claims[0])
                if not h or h["state"] != "active":
                    continue
                try:
                    eng.apply_transition(
                        TransitionCommand(
                            claim_id=old_claims[0],
                            expected_revision=h["revision"],
                            effect="supersede",
                            actor_id="f27-harness",
                            reason="f27 reproduction supersession",
                            successor_claim_id=new_claims[0],
                            interval=TimeInterval(from_us=pair.change_us),
                        ),
                        scope=scope,
                    )
                    with store.read() as conn:
                        heads = _claim_heads(conn)
                except Exception:
                    pass

        with store.read() as conn:
            src_claims = _source_claim_map(conn)

        queries = [q for q in corpus.queries if q.kind in query_kinds]
        report["corpus"]["queries_scored"] = len(queries)

        recorder = PathRecorder()
        with instrumented_lexical_paths(recorder):
            # 3. pre-rebuild measurement
            with store.read() as conn:
                pre = _phase_snapshot(
                    eng, conn, queries, scope, top_k, recorder,
                    stmt_to_source, source_to_stmt, src_claims,
                )

            # 4. rebuild: re-index every previously indexed claim at the
            #    current generation via the ingester's index_claim path —
            #    the same simulated projection rebuild the v2 harness ran.
            with store.read() as conn:
                heads = _claim_heads(conn)
                texts = {cid: _fts_text_for(conn, cid) for cid in heads}
            reindexed = 0
            for cid, text in texts.items():
                if text:
                    try:
                        eng._ingester.index_claim(cid, scope, text)
                        reindexed += 1
                    except Exception:
                        pass

            # 5. post-rebuild measurement
            with store.read() as conn:
                post = _phase_snapshot(
                    eng, conn, queries, scope, top_k, recorder,
                    stmt_to_source, source_to_stmt, src_claims,
                )

        # 6. per-query deltas + attribution
        post_by_id = {r["query_id"]: r for r in post["per_query"]}
        per_query = []
        counts = {"gained": 0, "lost": 0, "stable_hit": 0, "stable_miss": 0}
        for r in pre["per_query"]:
            p = post_by_id[r["query_id"]]
            tags = _attribute(r, p)
            outcome = next(
                (t for t in tags
                 if t in ("gained", "lost", "stable_hit", "stable_miss")),
                "unattributed",
            )
            counts[outcome] = counts.get(outcome, 0) + 1
            per_query.append({
                "query_id": r["query_id"],
                "kind": r["kind"],
                "query": r["query"],
                "expected_stmts": r["expected_stmts"],
                "pre": r,
                "post": p,
                "attribution": tags,
            })

        def _agg(rows):
            n = len(rows)
            hits = sum(1 for r in rows if r["hit"])
            errs = sum(1 for r in rows if r["error"])
            return {
                "n": n, "hits": hits,
                "hit_rate": round(hits / n, 4) if n else None,
                "errors": errs,
            }

        report["reindexed_claims"] = reindexed
        report["phases"] = {
            "pre_rebuild": {
                "projection_generation": pre["projection_generation"],
                "fts_rows_at_generation": pre["fts_rows_at_generation"],
                "fts5_available": pre["fts5_available"],
            },
            "post_rebuild": {
                "projection_generation": post["projection_generation"],
                "fts_rows_at_generation": post["fts_rows_at_generation"],
                "fts5_available": post["fts5_available"],
            },
        }
        report["recall"] = {
            "pre_rebuild": _agg(pre["per_query"]),
            "post_rebuild": _agg(post["per_query"]),
            "delta_hits": (
                _agg(post["per_query"])["hits"]
                - _agg(pre["per_query"])["hits"]
            ),
        }
        report["outcome_counts"] = counts
        report["per_query"] = per_query
        report["interpretation"] = (
            "attribution tags distinguish generation stranding, rank "
            "shifts among indexed candidates, scan-vs-fts path selection, "
            "and downstream packaging losses; 'unattributed' marks "
            "outcomes still needing explanation — F27 stays open until "
            "none remain (B49)."
        )
        return report
    finally:
        eng.close()


def summarize_attribution(report: dict) -> dict[str, Any]:
    """Roll the per-query attribution tags into counts by mechanism —
    the B49 evidence table."""
    mech: dict[str, int] = {}
    for q in report.get("per_query", []):
        for t in q["attribution"]:
            key = t.split(":", 1)[0] if t.startswith("path_") else t
            mech[key] = mech.get(key, 0) + 1
    return {
        "queries": len(report.get("per_query", [])),
        "mechanisms": dict(sorted(mech.items())),
        "unattributed": sum(
            1 for q in report.get("per_query", [])
            if "unattributed" in q["attribution"]
        ),
    }
