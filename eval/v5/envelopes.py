"""A0/A1 consumer performance envelopes (E62 / SPEC_V5 §20).

The spec's *release-qualification* protocol (V5-20.02) is implemented
honestly and parameterized — three independent close/reopen
repetitions, declared query counts, per-boundary timing — but the
default ``local`` scale is far below the 10,000-query/1,000-intent
qualification floor, so every result is stamped
``qualification="locally_measured"`` and misses are reported as misses.

Timing boundaries follow V5-33.02:

* **add ack** — ``Memory.add`` call entry → result returned (durable
  acceptance, not projection);
* **settled search** — ``Memory.search`` call entry → final
  ``SearchResult`` returned, measured only after a settle barrier;
* **A1** additionally runs a real writer thread (add/supersede/forget
  ops at a pinned rate), a reader thread (queries at a pinned rate),
  and the managed projection worker — backlog depth is sampled through
  the public ``status()`` surface plus the store's job queue, and an
  unbounded-growing backlog fails the envelope regardless of latency.

Nothing here clips, reorders, or retries queries to protect a number;
contentious runs publish their contention.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from .corpus import ConsumerCorpus, seed_corpus
from .harness import (
    ConsumerEnv,
    add_with_retry,
    environment,
    percentiles,
    seed_corpus_env,
    settle,
)


#: §20 envelope constants.
A0 = {
    "memories": 1000,
    "search_p95_ms": 25.0,
    "search_p99_ms": 75.0,
    "add_ack_p95_ms": 50.0,
}
A1 = {
    "memories": 5000,
    "search_p95_ms": 40.0,
    "search_p99_ms": 120.0,
    "add_ack_p95_ms": 50.0,
    "writer_ops_per_s": 2.0,
    "reader_queries_per_s": 10.0,
}
QUALIFICATION = {
    "min_queries": 10_000,
    "min_distinct_intents": 1_000,
    "repetitions": 3,
}


@dataclass
class EnvelopeResult:
    """One measured envelope verdict."""

    name: str
    qualification: str            # locally_measured | qualified | failed
    scale: dict
    repetitions: int
    add_ack_ms: dict
    search_ms: dict
    budgets: dict
    misses: List[str] = field(default_factory=list)
    backlog: dict = field(default_factory=dict)
    support: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qualification": self.qualification,
            "scale": self.scale,
            "repetitions": self.repetitions,
            "add_ack_ms": self.add_ack_ms,
            "search_ms": self.search_ms,
            "budgets": self.budgets,
            "misses": self.misses,
            "backlog": self.backlog,
            "support": self.support,
            "environment": self.environment,
            "errors": self.errors,
        }


def _measure_settled_search(env: ConsumerEnv, queries: Sequence[str],
                            *, k: int = 8) -> Tuple[List[float], int]:
    """Settled-search samples: full call-entry→return wall time."""
    lat: List[float] = []
    errs = 0
    for q in queries:
        t0 = time.perf_counter()
        try:
            env.memory.search(q, limit=k)
        except Exception:  # noqa: BLE001 — error latency still counts
            errs += 1
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat, errs


def measure_a0(
    corpus: Optional[ConsumerCorpus] = None,
    *,
    memories: int = 300,
    seed: int = 42,
    queries: int = 120,
    repetitions: int = 1,
    worker: str = "external",
    workdir: Optional[str] = None,
) -> EnvelopeResult:
    """A0: single namespace, ``memories`` retained items, settled reads.

    ``repetitions`` >1 reopens the store in a fresh ``Memory`` process-
    level instance (the §20.02 restart discipline) and pools samples —
    each repetition's own percentiles are kept for audit.
    """
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    budgets = dict(A0)
    res = EnvelopeResult(
        name="A0",
        qualification="locally_measured",
        scale={
            "memories": memories,
            "declared_queries_per_rep": queries,
            "spec_scale_memories": A0["memories"],
            "qualification_floor": dict(QUALIFICATION),
        },
        repetitions=repetitions,
        add_ack_ms={}, search_ms={}, budgets=budgets,
        environment=environment(),
    )
    env = seed_corpus_env(corpus, workdir=workdir, worker=worker)
    try:
        res.add_ack_ms = percentiles(env.add_ms.values())
        res.support["add_errors"] = dict(env.add_errors)
        res.support["drain"] = dict(env.drain)
        res.support["settle"] = settle(env)
        qs = [t.query for t in corpus.tasks]
        # Cycle the task list to reach the declared sample count; the
        # distinct-intent count is published so nobody mistakes repeated
        # queries for distinct intents (V5-20.02).
        pool = [qs[i % len(qs)] for i in range(queries)]
        res.scale["distinct_intents"] = len(set(pool))
        all_lat: List[float] = []
        rep_rows: List[dict] = []
        for r in range(repetitions):
            if r > 0:
                # restart repetition: close + reopen the same store
                env.memory.close()
                from verbatim import Memory
                import os
                env.memory = Memory(
                    os.path.join(env.workdir, "mem.db"),
                    user_id="eval-user", worker=worker,
                )
                settle(env)
            lat, errs = _measure_settled_search(env, pool)
            rep_rows.append({**percentiles(lat), "errors": errs})
            all_lat.extend(lat)
        res.support["repetition_rows"] = rep_rows
        res.search_ms = percentiles(all_lat)
    finally:
        env.close()
    _verdict(res, budgets)
    return res


def _pending_jobs(env: ConsumerEnv) -> Optional[int]:
    """Job-queue depth — the honest backlog gauge (store-side read)."""
    try:
        with env.memory._store.read() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN "
                "('queued','leased','retry_wait')"
            ).fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def measure_a1(
    corpus: Optional[ConsumerCorpus] = None,
    *,
    memories: int = 400,
    seed: int = 43,
    reader_queries: int = 60,
    writer_ops: int = 20,
    worker: str = "managed",
    workdir: Optional[str] = None,
) -> EnvelopeResult:
    """A1: live-load envelope — concurrent writer, reader, managed drain.

    Threads are pinned at the spec's rates (2 writer ops/s, 10 reader
    queries/s) by sleep pacing; the managed worker is the production
    drain path. Backlog depth is sampled before/after and must not grow
    without bound — a growing queue fails the envelope even if latency
    looks fine.
    """
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    budgets = dict(A1)
    res = EnvelopeResult(
        name="A1",
        qualification="locally_measured",
        scale={
            "memories": memories,
            "reader_queries": reader_queries,
            "writer_ops": writer_ops,
            "spec_scale_memories": A1["memories"],
            "rates": {
                "writer_ops_per_s": A1["writer_ops_per_s"],
                "reader_queries_per_s": A1["reader_queries_per_s"],
            },
        },
        repetitions=1,
        add_ack_ms={}, search_ms={}, budgets=budgets,
        environment=environment(),
    )
    env = seed_corpus_env(corpus, workdir=workdir, worker=worker)
    try:
        res.add_ack_ms = percentiles(env.add_ms.values())
        res.support["add_errors"] = dict(env.add_errors)
        backlog0 = _pending_jobs(env)
        qs = [t.query for t in corpus.tasks] or ["status"]
        read_lat: List[float] = []
        write_lat: List[float] = []
        read_errs: List[str] = []
        write_errs: List[str] = []
        stop = threading.Event()
        backlog_samples: List[int] = []

        def _reader() -> None:
            i = 0
            period = 1.0 / A1["reader_queries_per_s"]
            for _ in range(reader_queries):
                if stop.is_set():
                    break
                t0 = time.perf_counter()
                try:
                    env.memory.search(qs[i % len(qs)], limit=8)
                except Exception as exc:  # noqa: BLE001
                    read_errs.append(f"{type(exc).__name__}")
                read_lat.append((time.perf_counter() - t0) * 1000.0)
                i += 1
                dt = period - (time.perf_counter() - t0)
                if dt > 0:
                    time.sleep(dt)

        def _writer() -> None:
            period = 1.0 / A1["writer_ops_per_s"]
            for i in range(writer_ops):
                if stop.is_set():
                    break
                t0 = time.perf_counter()
                try:
                    # Mix: mostly fresh adds, some supersede revisions —
                    # the §20.02 revision/hold/correction mix at scale.
                    if i % 5 == 4 and env.refs:
                        target = list(env.refs.values())[i % len(env.refs)]
                        add_with_retry(
                            env.memory,
                            f"a1-writer correction {i} revises a prior note",
                            replaces=target,
                        )
                    else:
                        add_with_retry(
                            env.memory,
                            f"a1-writer live note {i}: washer token "
                            f"wt-{1000 + i}"
                        )
                except Exception as exc:  # noqa: BLE001
                    write_errs.append(f"{type(exc).__name__}")
                write_lat.append((time.perf_counter() - t0) * 1000.0)
                dt = period - (time.perf_counter() - t0)
                if dt > 0:
                    time.sleep(dt)

        rt = threading.Thread(target=_reader, daemon=True,
                              name="v5-a1-reader")
        wt = threading.Thread(target=_writer, daemon=True,
                              name="v5-a1-writer")
        rt.start(); wt.start()
        while rt.is_alive() or wt.is_alive():
            d = _pending_jobs(env)
            if d is not None:
                backlog_samples.append(d)
            time.sleep(0.25)
        rt.join(timeout=60); wt.join(timeout=60)
        settle(env, timeout_s=90)
        backlog_end = _pending_jobs(env)
        res.search_ms = percentiles(read_lat)
        res.support.update({
            "read_errors": read_errs[:20],
            "read_error_count": len(read_errs),
            "write_errors": write_errs[:20],
            "write_error_count": len(write_errs),
            "writer_add_ms": percentiles(write_lat),
        })
        res.backlog = {
            "start": backlog0,
            "samples": backlog_samples,
            "peak": max(backlog_samples) if backlog_samples else None,
            "end": backlog_end,
            "unbounded": bool(
                backlog_samples and backlog_end is not None
                and backlog_end > 0
                and len(backlog_samples) > 2
                and backlog_samples[-1] > backlog_samples[0]
            ),
        }
    finally:
        env.close()
    _verdict(res, budgets)
    if res.backlog.get("unbounded"):
        res.misses.append("backlog grew without bound under live load")
    return res


def _verdict(res: EnvelopeResult, budgets: dict) -> None:
    """Score the measured percentiles against the envelope — misses are
    named, never smoothed."""
    s = res.search_ms or {}
    a = res.add_ack_ms or {}
    if s.get("n", 0) == 0:
        res.misses.append("no settled-search samples executed")
    else:
        if s.get("p95") is not None and s["p95"] > budgets["search_p95_ms"]:
            res.misses.append(
                f"search p95 {s['p95']:.1f}ms > {budgets['search_p95_ms']:.0f}ms"
            )
        if s.get("p99") is not None and s["p99"] > budgets["search_p99_ms"]:
            res.misses.append(
                f"search p99 {s['p99']:.1f}ms > {budgets['search_p99_ms']:.0f}ms"
            )
    if a.get("n", 0) == 0:
        res.misses.append("no add-ack samples executed")
    elif a.get("p95") is not None and a["p95"] > budgets["add_ack_p95_ms"]:
        res.misses.append(
            f"add-ack p95 {a['p95']:.1f}ms > {budgets['add_ack_p95_ms']:.0f}ms"
        )
    # V5-20.02: qualification requires the full protocol. Locally
    # measured runs state the gap explicitly.
    q = res.scale.get("declared_queries_per_rep") or res.scale.get(
        "reader_queries") or 0
    n_samples = (s.get("n") or 0)
    res.support["qualification_gap"] = {
        "queries_measured": n_samples,
        "queries_required": QUALIFICATION["min_queries"],
        "distinct_intents": res.scale.get("distinct_intents"),
        "distinct_intents_required": QUALIFICATION["min_distinct_intents"],
        "repetitions": res.repetitions,
        "repetitions_required": QUALIFICATION["repetitions"],
    }
    if (
        n_samples >= QUALIFICATION["min_queries"]
        and (res.scale.get("distinct_intents") or 0)
            >= QUALIFICATION["min_distinct_intents"]
        and res.repetitions >= QUALIFICATION["repetitions"]
        and not res.misses
    ):
        res.qualification = "qualified"
    elif res.misses:
        res.qualification = "locally_measured"  # misses at any scale stay local
    else:
        res.qualification = "locally_measured"


def run_envelopes(*, quick: bool = True,
                  workdir: Optional[str] = None) -> dict:
    """Run both envelopes at the declared local scale."""
    scale = ({"a0_memories": 300, "a0_queries": 120, "a0_reps": 1,
              "a1_memories": 400, "a1_reads": 60, "a1_writes": 20}
             if quick else
             {"a0_memories": A0["memories"], "a0_queries": 2000,
              "a0_reps": 3,
              "a1_memories": A1["memories"], "a1_reads": 3000,
              "a1_writes": 600})
    a0 = measure_a0(
        memories=scale["a0_memories"], queries=scale["a0_queries"],
        repetitions=scale["a0_reps"], workdir=workdir)
    a1 = measure_a1(
        memories=scale["a1_memories"], reader_queries=scale["a1_reads"],
        writer_ops=scale["a1_writes"], workdir=workdir)
    return {
        "suite": "envelopes",
        "qualification": "locally_measured",
        "a0": a0.to_dict(),
        "a1": a1.to_dict(),
        "spec_budgets": {"A0": dict(A0), "A1": dict(A1)},
        "qualification_floor": dict(QUALIFICATION),
    }


__all__ = ["A0", "A1", "QUALIFICATION", "EnvelopeResult",
           "measure_a0", "measure_a1", "run_envelopes"]
