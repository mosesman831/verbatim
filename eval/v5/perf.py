"""Long-history A3 probe at disclosed local scale (E93 / SPEC_V5 §20.03).

The spec's A3 is ~1M retained source tokens / ~20k records with
revisions, duplicates, updates, and holds, measured at settled state.
A local run cannot honestly claim that scale, so this suite:

* builds a *real* mixed-history corpus (fillers + explicit revision
  pairs + verbatim duplicates + forget-distractors — the §20.03 mix,
  scaled down, disclosed exactly);
* measures the §20.03 surface — settled search p95/p99, add-ack p95,
  source-lexical readiness lag, peak RSS, drain CPU share — at two
  scales so a **sub-linear check** is computed from real paired points
  (scaling exponent α where latency ∝ n^α), never from one number;
* stamps every run ``qualification="locally_measured"`` with the exact
  scale executed and the spec target alongside.

A3 release numbers still need the full-scale run; this module is the
honest probe that says where the curve is heading.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .corpus import ConsumerCorpus, ConsumerTask, CorpusItem, seed_corpus
from .harness import (
    ConsumerEnv,
    add_with_retry,
    drain_memory,
    environment,
    peak_rss_mib,
    percentiles,
    seed_corpus_env,
    settle,
)

#: §20.03 targets (kept beside every local result).
A3_TARGETS = {
    "retained_source_tokens": 1_000_000,
    "source_records": 20_000,
    "search_p95_ms": 120.0,
    "search_p99_ms": 300.0,
    "add_ack_p95_ms": 50.0,
    "lexical_readiness_p95_ms": 500.0,
    "peak_rss_mib": 1024.0,
    "maintenance_cpu_share": 0.10,
}


def history_corpus(memories: int, seed: int) -> ConsumerCorpus:
    """A mixed-history corpus: fillers + revision pairs + verbatim
    duplicates + forget-distractors (the §20.03 content mix, scaled).

    Roughly every 8th filler becomes a revision pair (stale + current
    via ``replaces``) and every 16th a verbatim duplicate of a prior
    item — both are real ``add`` payloads through the consumer route.
    """
    base = seed_corpus(memories=max(64, memories // 2), seed=seed)
    items = list(base.items)
    tasks = list(base.tasks)
    import random as _r
    rng = _r.Random(seed + 1)
    n_extra = memories - len(items)
    i = 0
    while len(items) < memories:
        j = len(items)
        tok = f"hist-{j:05d}"
        if i % 8 == 4 and j + 1 < memories:
            # revision pair: stale then current
            items.append(CorpusItem(
                f"hist-{j:05d}-a",
                f"History note {tok}: the storage unit PIN was "
                f"{rng.randrange(1000, 9999)} (superseded)."))
            items.append(CorpusItem(
                f"hist-{j:05d}-b",
                f"History note {tok}: the storage unit PIN is now "
                f"{rng.randrange(1000, 9999)}.",
                supersedes=f"hist-{j:05d}-a"))
            i += 2
            continue
        if i % 16 == 8 and items:
            # verbatim duplicate of a prior item — dedup pressure
            src = items[rng.randrange(len(items))]
            items.append(CorpusItem(
                f"hist-{j:05d}-dup", src.text, infer=src.infer,
                tags=("duplicate",)))
            i += 1
            continue
        if i % 24 == 16:
            items.append(CorpusItem(
                f"hist-{j:05d}-gone",
                f"History note {tok}: temporary gate code "
                f"{rng.randrange(10000, 99999)} (expired).",
                forget=True))
            i += 1
            continue
        items.append(CorpusItem(
            f"hist-{j:05d}",
            f"History note {tok}: archived detail about topic "
            f"{rng.randrange(100)} recorded on day {rng.randrange(1, 365)}.",
            infer=False, tags=("history",)))
        i += 1
    # probes over history tokens so the query set scales with n
    for j in range(0, len(items), max(1, len(items) // 32)):
        it = items[j]
        tok = it.id.rsplit("-", 1)[0]
        if it.id.startswith("hist-") and not it.forget and not it.supersedes:
            tasks.append(ConsumerTask(
                f"t-hist-{j:05d}", f"history note {tok}",
                category="identifier", expected_ids=(it.id,)))
    return ConsumerCorpus(
        name=f"v5-history-{memories}s{seed}", seed=seed,
        items=tuple(items), tasks=tuple(tasks),
    )


@dataclass
class ScalePoint:
    """One scale's measured A3 surface."""

    memories: int
    items_added: int
    add_errors: int
    add_ack_ms: dict
    search_ms: dict
    lexical_readiness_ms: dict
    peak_rss_mib: Optional[float]
    drain_wall_s: float
    drain_cpu_s: float
    drain_cpu_share: Optional[float]
    store_bytes: int
    targets: dict = field(default_factory=lambda: dict(A3_TARGETS))

    def to_dict(self) -> dict:
        return {
            "memories": self.memories,
            "items_added": self.items_added,
            "add_errors": self.add_errors,
            "add_ack_ms": self.add_ack_ms,
            "search_ms": self.search_ms,
            "lexical_readiness_ms": self.lexical_readiness_ms,
            "peak_rss_mib": self.peak_rss_mib,
            "drain_wall_s": round(self.drain_wall_s, 2),
            "drain_cpu_s": round(self.drain_cpu_s, 2),
            "maintenance_cpu_share": self.drain_cpu_share,
            "store_bytes": self.store_bytes,
            "spec_targets": self.targets,
        }


def _lexical_readiness_samples(env: ConsumerEnv, n: int = 12) -> List[float]:
    """Acceptance→ready lag, measured on fresh adds.

    With the external worker the drain is synchronous: each sample is
    ``add`` → ``drain_report`` (the same durable-queue pass the managed
    worker runs) → ``wait_ready`` — i.e., the real settle path, not a
    timeout artifact. Managed mode measures the worker's async lag
    through ``wait_ready`` alone.
    """
    out: List[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        try:
            res = env.memory.add(
                f"readiness probe {i}: meter serial MS-{3000 + i}",
                infer=False)
        except Exception:
            continue
        if env.worker == "external":
            drain_memory(env)
        try:
            env.memory.wait_ready(res, timeout_ms=5000)
        except Exception:
            pass
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def measure_scale(memories: int, *, seed: int = 42,
                  queries: int = 60,
                  workdir: Optional[str] = None) -> ScalePoint:
    """One A3 scale point — real adds, real drain, real settled reads."""
    corpus = history_corpus(memories, seed)
    rss0 = peak_rss_mib()
    env = seed_corpus_env(corpus, workdir=workdir, worker="external")
    try:
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        settle(env, timeout_s=max(120.0, memories * 0.5))
        drain_wall = time.perf_counter() - t0
        drain_cpu = time.process_time() - cpu0
        ready_ms = _lexical_readiness_samples(env)
        settle(env, timeout_s=60)
        qs = [t.query for t in corpus.tasks] or ["status"]
        lat: List[float] = []
        for i in range(queries):
            t1 = time.perf_counter()
            try:
                env.memory.search(qs[i % len(qs)], limit=8)
            except Exception:
                pass
            lat.append((time.perf_counter() - t1) * 1000.0)
        rss = peak_rss_mib()
        try:
            store_bytes = os.path.getsize(
                os.path.join(env.workdir, "mem.db"))
        except OSError:
            store_bytes = 0
        return ScalePoint(
            memories=memories,
            items_added=len(env.add_results),
            add_errors=len(env.add_errors),
            add_ack_ms=percentiles(env.add_ms.values()),
            search_ms=percentiles(lat),
            lexical_readiness_ms=percentiles(ready_ms),
            peak_rss_mib=rss,
            drain_wall_s=drain_wall,
            drain_cpu_s=drain_cpu,
            drain_cpu_share=(
                drain_cpu / drain_wall if drain_wall > 0 else None
            ),
            store_bytes=store_bytes,
        )
    finally:
        env.close()


def sublinear_check(small: ScalePoint, large: ScalePoint) -> dict:
    """Scaling exponent from two real points: latency ∝ n^α.

    ``α < 1`` is sub-linear growth; ``α ≥ 1`` is named honestly. Only
    computed when both p95s are positive — otherwise ``unavailable``.
    """
    s95, l95 = (small.search_ms or {}).get("p95"), \
        (large.search_ms or {}).get("p95")
    sn, ln = small.memories, large.memories
    if not s95 or not l95 or s95 <= 0 or ln <= sn:
        return {"verdict": "unavailable",
                "reason": "insufficient paired scale points"}
    import math
    alpha = math.log(l95 / s95) / math.log(ln / sn)
    return {
        "verdict": "sublinear" if alpha < 1.0 else "superlinear",
        "alpha": round(alpha, 3),
        "p95_small_ms": s95, "p95_large_ms": l95,
        "n_small": sn, "n_large": ln,
        "note": "α<1 means latency grew slower than corpus size",
    }


def run_a3_probe(*, scales: Sequence[int] = (256, 1024),
                 seed: int = 42, queries: int = 60,
                 workdir: Optional[str] = None) -> dict:
    """Two-scale A3 probe at disclosed local sizes."""
    points = [measure_scale(m, seed=seed, queries=queries,
                            workdir=workdir) for m in scales]
    check = (
        sublinear_check(points[0], points[-1])
        if len(points) >= 2 else {"verdict": "unavailable"}
    )
    misses: List[str] = []
    big = points[-1]
    t = A3_TARGETS
    s95 = (big.search_ms or {}).get("p95")
    s99 = (big.search_ms or {}).get("p99")
    a95 = (big.add_ack_ms or {}).get("p95")
    if s95 is not None and s95 > t["search_p95_ms"]:
        misses.append(f"search p95 {s95:.1f}ms > {t['search_p95_ms']:.0f}ms")
    if s99 is not None and s99 > t["search_p99_ms"]:
        misses.append(f"search p99 {s99:.1f}ms > {t['search_p99_ms']:.0f}ms")
    r95 = (big.lexical_readiness_ms or {}).get("p95")
    if a95 is not None and a95 > t["add_ack_p95_ms"]:
        misses.append(f"add-ack p95 {a95:.1f}ms > {t['add_ack_p95_ms']:.0f}ms")
    if r95 is not None and r95 > t["lexical_readiness_p95_ms"]:
        misses.append(
            f"lexical-readiness p95 {r95:.1f}ms > "
            f"{t['lexical_readiness_p95_ms']:.0f}ms")
    if big.peak_rss_mib is not None and big.peak_rss_mib > t["peak_rss_mib"]:
        misses.append(
            f"peak RSS {big.peak_rss_mib:.0f}MiB > {t['peak_rss_mib']}MiB")
    return {
        "suite": "a3_probe",
        "qualification": "locally_measured",
        "scale_disclosure": (
            f"local scales {list(scales)} items vs spec ~20,000 "
            "records / ~1M source tokens — this run is a scaling probe, "
            "not A3 qualification"
        ),
        "environment": environment(),
        "points": [p.to_dict() for p in points],
        "sublinear_check": check,
        "misses": misses,
        "spec_targets": dict(t),
    }


__all__ = [
    "A3_TARGETS",
    "ScalePoint",
    "history_corpus",
    "measure_scale",
    "run_a3_probe",
    "sublinear_check",
]
