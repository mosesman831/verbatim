"""V6 consumer performance envelopes (SPEC_V6 §02.2, V6-02.11–02.16).

Five envelopes measured through the public ``verbatim.Memory`` route,
in the same honesty idiom as the V5 envelopes: budgets beside
measurements, misses named, scale disclosed.

* **A0** — 1 namespace, 1,000 short memories, hashing encoder, pack
  cache OFF, warm pages: settled search p95 ≤25 ms, p99 ≤75 ms, 4 KiB
  add-ack p95 ≤50 ms. Per V6-02.13 the cache-off configuration is the
  qualifying one and pending/partial rows cannot qualify settled
  latency — non-settled statuses are counted and named, never
  smoothed into the percentile.
* **A0-cache** — same workload, ``retrieval.cache.enabled`` on,
  repeated intents: settled search p95 ≤8 ms. The V4-33 final-pack
  cache is consulted on the governed recall lane inside
  ``Memory.search`` (``verbatim.retrieval.cache.configured`` resolves
  it from the facade config); when this build cannot provision it the
  envelope reports ``unavailable`` with the real reason (V6-02.06).
* **A0-neural** — same size, pinned local artifact encoder: p95 ≤40
  ms, p99 ≤120 ms. The offline artifact path is a sibling workstream
  (V6-03.06–03.10); until ``Memory`` accepts ``encoder="artifact"``
  AND the encoder reports ``available()``, this envelope reports
  ``unavailable`` — never a hashing rerun relabeled neural.
* **A1** — 5,000 live memories, 2 writes/s + 10 queries/s on one
  ``Memory``, managed drain: settled search p95 ≤40 ms, add-ack p95
  ≤50 ms, no unbounded backlog — plus the V6-02.11 same-session
  visibility probe (add→search observes the write ≥95% within 200 ms
  or reports pending/blocked honestly).
* **A3** — ~20,000 records / ~1M source tokens, one namespace: search
  p95 ≤150 ms, add-ack p95 ≤80 ms, sublinear index growth, peak RSS
  ≤1.5 GiB. Seeding goes through ``eval.v5.harness.bulk_seed`` — the
  harness-side bulk path V6-02.15 allows (periodic drain, never
  serial-ack-bound); every measured read and every profiled add still
  goes through the public API. Add-ack is profiled per stage at the
  declared n-checkpoints (V6-02.16): every reachable stage inside
  ``Memory.add`` is wrapped with a ``perf_counter`` pair (the
  ``eval.v5.timers`` StageProfiler idiom) and any stage whose per-op
  cost grows faster than linear is named in ``superlinear_stages``.

The qualification floor (V5-20.02, carried) — 10,000 queries, 1,000
distinct intents, 3 close/reopen repetitions — is published beside
every result; local runs are ``locally_measured`` unless they truly
meet it.
"""

from __future__ import annotations

import functools
import math
import os
import random
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.v5.corpus import ConsumerCorpus, seed_corpus
from eval.v5.harness import (
    ConsumerEnv,
    add_with_retry,
    bulk_seed,
    drain_memory,
    environment,
    peak_rss_mib,
    percentiles,
    seed_corpus_env,
    settle,
)
from eval.v5.timers import StageSamples


#: SPEC_V6 §02.2 envelope budgets — declared acceptance targets, kept
#: beside every measurement (V6-02.14: a miss stays published; the
#: target may only be replaced by a later reviewed target, never a
#: silent edit).
BUDGETS: Dict[str, Dict[str, Any]] = {
    "A0": {
        "memories": 1_000,
        "search_p95_ms": 25.0,
        "search_p99_ms": 75.0,
        "add_ack_p95_ms": 50.0,
        "workload": "1 namespace, hashing, cache off, warm pages",
    },
    "A0_CACHE": {
        "memories": 1_000,
        "search_p95_ms": 8.0,
        "workload": "same as A0, pack cache on, repeated intents",
    },
    "A0_NEURAL": {
        "memories": 1_000,
        "search_p95_ms": 40.0,
        "search_p99_ms": 120.0,
        "workload": "same as A0, pinned local encoder",
    },
    "A1": {
        "memories": 5_000,
        "search_p95_ms": 40.0,
        "add_ack_p95_ms": 50.0,
        "writer_ops_per_s": 2.0,
        "reader_queries_per_s": 10.0,
        "unbounded_backlog": False,
        "same_session_visibility_ms": 200.0,
        "same_session_visibility_min": 0.95,
        "workload": "5,000 live memories, 2 writes/s + 10 queries/s, "
                    "drain on",
    },
    "A3": {
        "memories": 20_000,
        "source_tokens": 1_000_000,
        "search_p95_ms": 150.0,
        "add_ack_p95_ms": 80.0,
        "index_growth_alpha_max": 1.0,
        "peak_rss_mib": 1_536.0,
        "workload": "~20,000 records / ~1M source tokens, one namespace",
    },
}

#: Carried from the V5 envelope harness (V5-20.02): what a *qualified*
#: number still requires. Local runs publish the gap explicitly.
QUALIFICATION = {
    "min_queries": 10_000,
    "min_distinct_intents": 1_000,
    "repetitions": 3,
}

#: Search statuses that count as *settled* for V6-02.13 — a completed
#: retrieval with an honest insufficient verdict is settled; pending,
#: partial, blocked, unavailable, and error rows are not.
_SETTLED_STATUSES = frozenset({"ready", "insufficient"})


@dataclass
class EnvelopeResult:
    """One measured V6 envelope — the V5 EnvelopeResult shape plus the
    portfolio-facing ``status``/``verdict``/``measurements`` fields."""

    name: str
    qualification: str            # locally_measured | qualified | unavailable | failed
    scale: dict
    repetitions: int
    add_ack_ms: dict
    search_ms: dict
    budgets: dict
    status: str = "measured"      # measured | unavailable | failed
    verdict: str = "pending"      # passed | missed | unavailable | failed
    measurements: dict = field(default_factory=dict)
    misses: List[str] = field(default_factory=list)
    backlog: dict = field(default_factory=dict)
    support: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "verdict": self.verdict,
            "qualification": self.qualification,
            "scale": self.scale,
            "repetitions": self.repetitions,
            "add_ack_ms": self.add_ack_ms,
            "search_ms": self.search_ms,
            "budgets": self.budgets,
            "measurements": self.measurements,
            "misses": self.misses,
            "backlog": self.backlog,
            "support": self.support,
            "environment": self.environment,
            "errors": self.errors,
        }


def _new_result(name: str, budgets: dict, scale: dict,
                *, repetitions: int = 1) -> EnvelopeResult:
    return EnvelopeResult(
        name=name,
        qualification="locally_measured",
        scale=scale,
        repetitions=repetitions,
        add_ack_ms={},
        search_ms={},
        budgets=dict(budgets),
        environment=environment(),
    )


def _unavailable(res: EnvelopeResult, reason: str,
                 *, probe: Optional[dict] = None) -> dict:
    """Capability-absent result: the honest ``unavailable`` report —
    never a simulated measurement (V5-§24 rule carried into V6)."""
    res.status = "unavailable"
    res.verdict = "unavailable"
    res.qualification = "unavailable"
    res.errors.append(reason)
    res.support["unavailable_reason"] = reason
    if probe is not None:
        res.support["capability_probe"] = probe
    return res.to_dict()


def _measure_settled_search(
    env: ConsumerEnv, queries: Sequence[str], *, k: int = 8
) -> Tuple[List[float], int, Dict[str, int]]:
    """Settled-search samples: full call-entry→return wall time, plus
    the per-row status histogram V6-02.13 requires (pending/partial
    rows cannot qualify settled latency — they are counted, not
    dropped)."""
    lat: List[float] = []
    errs = 0
    statuses: Dict[str, int] = {}
    for q in queries:
        t0 = time.perf_counter()
        try:
            r = env.memory.search(q, limit=k)
            st = str(getattr(r, "status", "unknown"))
        except Exception:  # noqa: BLE001 — error latency still counts
            errs += 1
            st = "error"
        statuses[st] = statuses.get(st, 0) + 1
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat, errs, statuses


def _immediate_add_search(env: ConsumerEnv, *, probes: int = 4) -> dict:
    """``immediate_add_search`` boundary (V6-02.12): add→search back to
    back, so the causal-barrier cost is visible beside settled search —
    never averaged away. Runs AFTER the settled loop because on an
    external-worker env the barrier correctly waits out its budget on
    undrained receipts."""
    lat: List[float] = []
    statuses: Dict[str, int] = {}
    for i in range(probes):
        tok = f"imm-{20000 + i}"
        t0 = time.perf_counter()
        try:
            env.memory.add(
                f"immediate probe {i}: cabinet key {tok}", infer=False)
            r = env.memory.search(tok, limit=2)
            st = str(getattr(r, "status", "unknown"))
        except Exception:  # noqa: BLE001
            st = "error"
        statuses[st] = statuses.get(st, 0) + 1
        lat.append((time.perf_counter() - t0) * 1000.0)
    return {**percentiles(lat), "statuses": statuses, "probes": probes}


def measure_a0(
    corpus: Optional[ConsumerCorpus] = None,
    *,
    memories: int = 300,
    seed: int = 42,
    queries: int = 120,
    repetitions: int = 1,
    worker: str = "external",
    workdir: Optional[str] = None,
    memory_kwargs: Optional[dict] = None,
    immediate_probes: int = 4,
) -> dict:
    """A0: single namespace, ``memories`` retained items, settled reads.

    ``repetitions`` >1 reopens the store in a fresh ``Memory``
    process-level instance (the §20.02 restart discipline carried
    forward) and pools samples — each repetition's own percentiles are
    kept for audit. Cache OFF is the qualifying configuration
    (V6-02.13); ``memory_kwargs`` stays caller-controlled so the
    cache-on variant is a different function, not a flag flip.
    """
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    budgets = dict(BUDGETS["A0"])
    res = _new_result(
        "A0", budgets,
        {
            "memories": memories,
            "declared_queries_per_rep": queries,
            "spec_scale_memories": budgets["memories"],
            "cache": "off",
            "qualification_floor": dict(QUALIFICATION),
        },
        repetitions=repetitions,
    )
    env = seed_corpus_env(corpus, workdir=workdir, worker=worker,
                          memory_kwargs=memory_kwargs)
    try:
        res.add_ack_ms = percentiles(env.add_ms.values())
        res.support["add_errors"] = dict(env.add_errors)
        res.support["drain"] = dict(env.drain)
        res.support["settle"] = settle(env)
        qs = [t.query for t in corpus.tasks]
        # Cycle the task list to reach the declared sample count; the
        # distinct-intent count is published so nobody mistakes repeated
        # queries for distinct intents (V5-20.02 convention).
        pool = [qs[i % len(qs)] for i in range(queries)]
        res.scale["distinct_intents"] = len(set(pool))
        all_lat: List[float] = []
        rep_rows: List[dict] = []
        status_totals: Dict[str, int] = {}
        for r in range(repetitions):
            if r > 0:
                # restart repetition: close + reopen the same store
                env.memory.close()
                from verbatim import Memory
                env.memory = Memory(
                    os.path.join(env.workdir, "mem.db"),
                    user_id="eval-user", worker=worker,
                    **(memory_kwargs or {}),
                )
                settle(env)
            lat, errs, sts = _measure_settled_search(env, pool)
            rep_rows.append({**percentiles(lat), "errors": errs})
            for k_, v_ in sts.items():
                status_totals[k_] = status_totals.get(k_, 0) + v_
            all_lat.extend(lat)
        res.support["repetition_rows"] = rep_rows
        res.search_ms = percentiles(all_lat)
        res.measurements["search_status"] = status_totals
        if immediate_probes:
            res.measurements["immediate_add_search_ms"] = (
                _immediate_add_search(env, probes=immediate_probes))
    finally:
        env.close()
    _verdict(res, budgets)
    _unsettled_note(res)
    return res.to_dict()


def measure_a0_cache(
    corpus: Optional[ConsumerCorpus] = None,
    *,
    memories: int = 300,
    seed: int = 42,
    queries: int = 60,
    passes: int = 3,
    worker: str = "external",
    workdir: Optional[str] = None,
) -> dict:
    """A0-cache: A0 workload with the V4-33 final-pack cache enabled.

    The cache is provisioned through the real config surface —
    ``config={"retrieval": {"cache": {"enabled": True}}}`` on
    ``Memory`` — never by poking the store. Repeated intents run in
    passes: pass 0 is the cold population, later passes are the
    cache-served rows the ≤8 ms budget applies to. Real counters from
    ``retrieval.cache.stats_for`` are published — if the cache is
    never consulted or never hits, that is the finding.
    """
    budgets = dict(BUDGETS["A0_CACHE"])
    res = _new_result(
        "A0_CACHE", budgets,
        {
            "memories": memories,
            "queries_per_pass": queries,
            "passes": passes,
            "spec_scale_memories": budgets["memories"],
            "cache": "on",
        },
    )
    try:
        from verbatim.retrieval import cache as _rcache
    except Exception as exc:  # noqa: BLE001
        return _unavailable(
            res,
            f"verbatim.retrieval.cache not importable: "
            f"{type(exc).__name__}: {exc}")
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    mk = {"config": {"retrieval": {"cache": {"enabled": True}}}}
    try:
        env = seed_corpus_env(corpus, workdir=workdir, worker=worker,
                              memory_kwargs=mk)
    except Exception as exc:  # noqa: BLE001 — provisioning must be real
        return _unavailable(
            res,
            f"pack cache could not be provisioned through Memory "
            f"config: {type(exc).__name__}: {exc}",
            probe={"config": mk["config"]})
    try:
        res.add_ack_ms = percentiles(env.add_ms.values())
        res.support["add_errors"] = dict(env.add_errors)
        res.support["settle"] = settle(env)
        qs = [t.query for t in corpus.tasks]
        pool = [qs[i % len(qs)] for i in range(queries)]
        res.scale["distinct_intents"] = len(set(pool))
        pass_rows: List[dict] = []
        warm_lat: List[float] = []
        status_totals: Dict[str, int] = {}
        for p in range(max(1, passes)):
            lat, errs, sts = _measure_settled_search(env, pool)
            pass_rows.append({
                "pass": p,
                "role": "cold" if p == 0 else "warm",
                **percentiles(lat),
                "errors": errs,
            })
            for k_, v_ in sts.items():
                status_totals[k_] = status_totals.get(k_, 0) + v_
            if p > 0:
                warm_lat.extend(lat)
        res.support["pass_rows"] = pass_rows
        # The budget applies to cache-served (warm-pass) rows; the cold
        # pass is published separately so population cost is visible.
        res.search_ms = percentiles(warm_lat)
        res.measurements["search_status"] = status_totals
        res.measurements["cold_pass_ms"] = pass_rows[0] if pass_rows else {}
        try:
            res.measurements["cache"] = _rcache.stats_for(
                env.memory._store, env.memory._cfg)
        except Exception as exc:  # noqa: BLE001
            res.measurements["cache"] = {
                "stats_error": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()
    _verdict(res, budgets)
    _unsettled_note(res)
    cache = res.measurements.get("cache") or {}
    if cache.get("lookups", 0) == 0 and "stats_error" not in cache:
        res.misses.append(
            "pack cache never consulted on the consumer search route "
            "(retrieval.cache.enabled provisioned but zero lookups)")
    elif cache.get("hits", 0) == 0 and cache.get("lookups", 0) > 0:
        res.misses.append(
            "repeated intents never hit the pack cache "
            f"(lookups={cache.get('lookups')}, hits=0)")
    _finish_verdict(res)
    return res.to_dict()


def _artifact_capability_probe() -> dict:
    """Probe the offline-artifact chain end to end — builder → manifest
    → hash-verify → ``available()`` — on a throwaway 16-dim table, so
    an ``unavailable`` verdict names the exact missing link (facade
    allowlist vs. builder vs. verifier) rather than guessing."""
    out: Dict[str, Any] = {}
    try:
        from verbatim.embeddings.artifact import ArtifactEncoder
        out["artifact_encoder_importable"] = True
    except Exception as exc:  # noqa: BLE001
        out["artifact_encoder_importable"] = False
        out["import_error"] = f"{type(exc).__name__}: {exc}"
        return out
    tmp = tempfile.mkdtemp(prefix="verbatim-v6-artifact-probe-")
    try:
        try:
            from verbatim.embeddings.artifact_build import build_artifact
        except ImportError:
            out["artifact_builder"] = "absent"
            return out
        out["artifact_builder"] = "present"
        table = {
            f"probe-{i}": [
                1.0 if j == i % 16 else 0.0 for j in range(16)
            ]
            for i in range(16)
        }
        build_artifact(
            os.path.join(tmp, "models"),
            model="v6-probe", revision="v1", table=table, dim=16)
        from pathlib import Path
        from verbatim.config import EmbeddingConfig
        enc = ArtifactEncoder(
            EmbeddingConfig(
                backend="artifact", model="v6-probe",
                artifact_revision="v1"),
            data_dir=Path(tmp))
        out["build_verify_available"] = bool(enc.available())
        out["probe_encoder_id"] = getattr(enc, "encoder_id", None)
    except Exception as exc:  # noqa: BLE001 — probe reports, never raises
        out["build_verify_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def measure_a0_neural(
    corpus: Optional[ConsumerCorpus] = None,
    *,
    memories: int = 300,
    seed: int = 42,
    queries: int = 60,
    worker: str = "external",
    workdir: Optional[str] = None,
    memory_kwargs: Optional[dict] = None,
) -> dict:
    """A0-neural: A0 workload on the pinned local artifact encoder.

    Probes the REAL provisioning path — ``Memory(encoder="artifact")``
    plus an ``available()``-true encoder — and reports ``unavailable``
    with the actual reason when this build cannot run it. The artifact
    build path is a sibling workstream (V6-03.06–03.10): when the
    facade accepts the encoder and the verified artifact loads, this
    same function measures the envelope unchanged. The capability
    probe additionally exercises build→manifest→verify→``available()``
    so the named gap is precise.
    """
    budgets = dict(BUDGETS["A0_NEURAL"])
    res = _new_result(
        "A0_NEURAL", budgets,
        {
            "memories": memories,
            "declared_queries_per_rep": queries,
            "spec_scale_memories": budgets["memories"],
            "encoder": "artifact",
        },
    )
    probe = _artifact_capability_probe()
    mk = {"encoder": "artifact"}
    mk.update(memory_kwargs or {})
    probe["memory_kwargs"] = dict(mk)
    # Capability probe on a throwaway store — cheap, honest, and never
    # a mock: the real constructor decides.
    probe_dir = tempfile.mkdtemp(prefix="verbatim-v6-a0n-probe-")
    try:
        try:
            from verbatim import Memory
            probe_mem = Memory(
                os.path.join(probe_dir, "mem.db"),
                user_id="eval-user", worker=worker, **mk)
        except Exception as exc:  # noqa: BLE001
            return _unavailable(
                res,
                f"Memory rejected encoder='artifact': "
                f"{type(exc).__name__}: {exc} — the pinned local "
                "artifact path (V6-03.06–03.10) is not provisioned "
                "in this build",
                probe=probe)
        try:
            enc = getattr(probe_mem, "_encoder", None)
            probe["encoder_id"] = getattr(
                probe_mem, "_encoder_id", None)
            probe["encoder_available"] = bool(
                getattr(enc, "available", lambda: False)())
            probe["warnings"] = list(getattr(probe_mem, "_warnings", []))
        finally:
            try:
                probe_mem.close()
            except Exception:
                pass
        if not probe["encoder_available"]:
            return _unavailable(
                res,
                "artifact encoder constructed but available() is False "
                "— no verified pinned artifact loaded (V6-03.08)",
                probe=probe)
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

    # Provisioned for real: run the A0 protocol on this encoder.
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    res.support["capability_probe"] = probe
    env = seed_corpus_env(corpus, workdir=workdir, worker=worker,
                          memory_kwargs=mk)
    try:
        res.add_ack_ms = percentiles(env.add_ms.values())
        res.support["add_errors"] = dict(env.add_errors)
        res.support["settle"] = settle(env)
        qs = [t.query for t in corpus.tasks]
        pool = [qs[i % len(qs)] for i in range(queries)]
        res.scale["distinct_intents"] = len(set(pool))
        lat, errs, sts = _measure_settled_search(env, pool)
        res.search_ms = percentiles(lat)
        res.measurements["search_status"] = sts
        res.support["search_errors"] = errs
    finally:
        env.close()
    _verdict(res, budgets)
    _unsettled_note(res)
    return res.to_dict()


def _pending_jobs(env: ConsumerEnv) -> Optional[int]:
    """Job-queue depth — the honest backlog gauge (store-side read,
    same as the V5 harness)."""
    try:
        with env.memory._store.read() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN "
                "('queued','leased','retry_wait')"
            ).fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def _visibility_probe(env: ConsumerEnv, *, probes: int,
                      budget_ms: float) -> dict:
    """V6-02.11 same-session visibility: add → poll search for the
    write up to ``budget_ms``.

    Per probe the outcome is one of:

    * ``observed`` — the write was delivered within the budget;
    * ``honest_pending`` — budget exhausted with the last result
      reporting pending/blocked/unavailable or
      ``causal_satisfied=false`` (the contract's honest answer);
    * ``unobserved_ready`` — budget exhausted on a ready-looking
      result that never surfaced the write: a barrier-or-ranking
      anomaly, named as such.
    """
    out: Dict[str, Any] = {
        "probes": 0,
        "observed": 0,
        "honest_pending": 0,
        "unobserved_ready": 0,
        "budget_ms": budget_ms,
        "samples_ms": [],
        "last_statuses": {},
    }
    for i in range(probes):
        tok = f"vis-{30000 + i}"
        try:
            r = env.memory.add(
                f"a1 visibility probe {i}: marker token {tok}",
                infer=False)
        except Exception as exc:  # noqa: BLE001
            out["last_statuses"]["add_error"] = (
                out["last_statuses"].get("add_error", 0) + 1)
            out.setdefault("add_errors", []).append(
                f"{type(exc).__name__}: {exc}")
            continue
        t0 = time.perf_counter()
        deadline = t0 + budget_ms / 1000.0
        seen = False
        last_status = "unknown"
        causal = False
        while True:
            try:
                sr = env.memory.search(tok, limit=2)
                last_status = str(getattr(sr, "status", "unknown"))
                causal = bool(
                    (getattr(sr, "readiness", {}) or {})
                    .get("causal_satisfied"))
                items = list(getattr(sr, "items", []) or [])
                if any(
                    getattr(h, "ref", "") == r.ref
                    or getattr(h, "memory_id", "") == r.memory_id
                    for h in items
                ):
                    seen = True
                    break
            except Exception:  # noqa: BLE001
                last_status = "error"
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.01)
        out["samples_ms"].append(
            round((time.perf_counter() - t0) * 1000.0, 2))
        out["probes"] += 1
        out["last_statuses"][last_status] = (
            out["last_statuses"].get(last_status, 0) + 1)
        if seen:
            out["observed"] += 1
        elif last_status in ("pending", "blocked", "unavailable") \
                or not causal:
            out["honest_pending"] += 1
        else:
            out["unobserved_ready"] += 1
    n = out["probes"]
    satisfied = out["observed"] + out["honest_pending"]
    out["satisfied"] = satisfied
    out["satisfied_pct"] = round(satisfied / n, 4) if n else None
    out["latency_ms"] = percentiles(out["samples_ms"])
    return out


def measure_a1(
    corpus: Optional[ConsumerCorpus] = None,
    *,
    memories: int = 400,
    seed: int = 43,
    reader_queries: int = 60,
    writer_ops: int = 20,
    worker: str = "managed",
    workdir: Optional[str] = None,
    visibility_probes: int = 6,
) -> dict:
    """A1: live-load envelope — concurrent writer, reader, managed
    drain, at the V6 rates (2 writer ops/s, 10 reader queries/s, the
    same pins V5 used so the V6 number is comparable to the published
    miss).

    Adds the V6-02.11 same-session visibility probe after the threaded
    phase: each probe add is searched until observed or the 200 ms
    budget lapses; satisfied = observed or honestly pending/blocked.
    """
    budgets = dict(BUDGETS["A1"])
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    res = _new_result(
        "A1", budgets,
        {
            "memories": memories,
            "reader_queries": reader_queries,
            "writer_ops": writer_ops,
            "spec_scale_memories": budgets["memories"],
            "visibility_probes": visibility_probes,
            "rates": {
                "writer_ops_per_s": budgets["writer_ops_per_s"],
                "reader_queries_per_s": budgets["reader_queries_per_s"],
            },
        },
    )
    env = seed_corpus_env(corpus, workdir=workdir, worker=worker)
    try:
        res.support["add_errors"] = dict(env.add_errors)
        backlog0 = _pending_jobs(env)
        qs = [t.query for t in corpus.tasks] or ["status"]
        read_lat: List[float] = []
        write_lat: List[float] = []
        read_errs: List[str] = []
        write_errs: List[str] = []
        read_status: Dict[str, int] = {}
        stop = threading.Event()
        backlog_samples: List[int] = []

        def _reader() -> None:
            i = 0
            period = 1.0 / budgets["reader_queries_per_s"]
            for _ in range(reader_queries):
                if stop.is_set():
                    break
                t0 = time.perf_counter()
                try:
                    r = env.memory.search(qs[i % len(qs)], limit=8)
                    st = str(getattr(r, "status", "unknown"))
                except Exception as exc:  # noqa: BLE001
                    read_errs.append(f"{type(exc).__name__}")
                    st = "error"
                read_status[st] = read_status.get(st, 0) + 1
                read_lat.append((time.perf_counter() - t0) * 1000.0)
                i += 1
                dt = period - (time.perf_counter() - t0)
                if dt > 0:
                    time.sleep(dt)

        def _writer() -> None:
            period = 1.0 / budgets["writer_ops_per_s"]
            for i in range(writer_ops):
                if stop.is_set():
                    break
                t0 = time.perf_counter()
                try:
                    # Same mix as V5: mostly fresh adds, some supersede
                    # revisions — the live-write blend.
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
                              name="v6-a1-reader")
        wt = threading.Thread(target=_writer, daemon=True,
                              name="v6-a1-writer")
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
        res.measurements["search_status"] = read_status
        # Non-settled rows under LIVE load are the contract working —
        # a search that exhausts its barrier budget reports pending
        # honestly (V6-02.09). They stay disclosed here and in the
        # status histogram; whether the write itself was observed is
        # the visibility probe's verdict below, not this count's.
        res.support["non_settled_rows"] = {
            k: v for k, v in read_status.items()
            if k not in _SETTLED_STATUSES
        }
        # The A1 add-ack budget is the writer's own acks under live
        # load — the seed trace stays in support for audit.
        res.add_ack_ms = percentiles(write_lat)
        res.support.update({
            "seed_add_ms": percentiles(env.add_ms.values()),
            "read_errors": read_errs[:20],
            "read_error_count": len(read_errs),
            "write_errors": write_errs[:20],
            "write_error_count": len(write_errs),
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
        if visibility_probes:
            res.measurements["same_session_visibility"] = (
                _visibility_probe(
                    env, probes=visibility_probes,
                    budget_ms=budgets["same_session_visibility_ms"]))
    finally:
        env.close()
    _verdict(res, budgets)
    if res.backlog.get("unbounded"):
        res.misses.append("backlog grew without bound under live load")
    vis = res.measurements.get("same_session_visibility") or {}
    if vis.get("probes"):
        need = budgets["same_session_visibility_min"]
        if (vis.get("satisfied_pct") or 0.0) < need:
            res.misses.append(
                f"same-session add→search satisfied "
                f"{vis['satisfied_pct']:.2f} < {need:.2f} "
                f"(observed={vis['observed']}, "
                f"honest_pending={vis['honest_pending']}, "
                f"unobserved_ready={vis['unobserved_ready']})")
    _finish_verdict(res)
    return res.to_dict()


# ---------------------------------------------------------------------------
# A3 — scale envelope with add-ack stage profiling
# ---------------------------------------------------------------------------

#: The A3 seeding text shape: ~50 tokens per record so ~20,000 records
#: carry ~1M source tokens, matching the spec workload's order.
_A3_TEMPLATES: Tuple[str, ...] = (
    "archive record {tok}: filed note about the {thing} account kept "
    "for the {month} review cycle with reference sheet {ref} and a "
    "duplicate copy stored in the second cabinet",
    "archive record {tok}: household binder page for the {thing} file "
    "updated during the {month} audit with contact extension {ref} "
    "written beside the column of older entries",
    "archive record {tok}: workshop log entry covering the {thing} "
    "maintenance task from {month} listing part number {ref} and the "
    "torque specification copied from the manual",
    "archive record {tok}: ledger line item for the {thing} purchase "
    "recorded in {month} with receipt serial {ref} stapled behind the "
    "divider labeled for this year",
)
_A3_THINGS: Tuple[str, ...] = (
    "garden", "tax", "insurance", "warranty", "prescription",
    "subscription", "lease", "membership", "invoice", "utility",
)


def a3_texts(n: int, seed: int = 42) -> List[str]:
    """Deterministic A3 seed payloads — ``n`` records of ~50 tokens
    with a unique ``a3-NNNNN`` token each (exact-match probes)."""
    rng = random.Random(seed)
    return [
        _A3_TEMPLATES[i % len(_A3_TEMPLATES)].format(
            tok=f"a3-{i:05d}",
            thing=_A3_THINGS[rng.randrange(len(_A3_THINGS))],
            month=rng.choice(("January", "March", "April", "June",
                              "September", "November")),
            ref=rng.randrange(10_000, 99_999),
        )
        for i in range(n)
    ]


def _a3_queries(seeded_texts: Sequence[str], n: int, seed: int) -> List[str]:
    """Identifier queries over actually-seeded tokens."""
    rng = random.Random(seed + 7)
    if not seeded_texts:
        return ["archive record a3-00000"]
    idx = rng.sample(
        range(len(seeded_texts)), min(n, len(seeded_texts)))
    return [f"archive record a3-{i:05d}" for i in idx]


def _store_bytes(workdir: str) -> int:
    """On-disk size of the store family (db + wal + shm) — the honest
    index+content footprint."""
    total = 0
    for name in ("mem.db", "mem.db-wal", "mem.db-shm"):
        try:
            total += os.path.getsize(os.path.join(workdir, name))
        except OSError:
            pass
    return total


def _growth_alpha(m1: Optional[float], m2: Optional[float],
                  n1: int, n2: int) -> Optional[float]:
    """``alpha`` in ``cost ∝ n^alpha`` between two checkpoints."""
    if n2 <= n1 or m1 is None or m2 is None or m2 <= 0:
        return None
    if m1 <= 0:
        return float("inf")
    return math.log(m2 / m1) / math.log(n2 / n1)


#: Documented heuristic for V6-02.16's "superlinear stage" — published
#: in every artifact so the rule is auditable, not vibes.
SUPERLINEAR_METHOD = (
    "per-stage alpha = ln(mean_last/mean_first)/ln(n_last/n_first) over "
    "the first→last checkpoints of the profiled adds; a stage is a "
    "named defect when alpha > 1.0 AND its mean at the largest "
    "checkpoint is >= 0.5 ms (below the floor perf_counter noise "
    "dominates — alphas are still published for every stage)"
)
_SUPERLINEAR_FLOOR_MS = 0.5


def _stage_scaling(rows: List[dict]) -> dict:
    """Per-stage growth exponents + the named superlinear defects."""
    alphas: Dict[str, Optional[float]] = {}
    defects: List[dict] = []
    if len(rows) < 2 or rows[0]["n"] >= rows[-1]["n"]:
        return {
            "alphas": alphas,
            "superlinear_stages": defects,
            "method": SUPERLINEAR_METHOD,
            "note": "insufficient distinct checkpoints for a ratio",
        }
    first, last = rows[0], rows[-1]
    n1, n2 = first["n"], last["n"]
    stages: set = set()
    for r in rows:
        stages.update((r.get("stages_ms") or {}).keys())
    for st in sorted(stages):
        m1 = (first["stages_ms"].get(st) or {}).get("mean")
        m2 = (last["stages_ms"].get(st) or {}).get("mean")
        alpha = _growth_alpha(m1, m2, n1, n2)
        alphas[st] = (
            round(alpha, 3)
            if alpha is not None and math.isfinite(alpha)
            else ("inf" if alpha is not None else None)
        )
        if (
            alpha is not None
            and alpha > 1.0
            and m2 is not None
            and m2 >= _SUPERLINEAR_FLOOR_MS
        ):
            defects.append({
                "stage": st,
                "alpha": round(alpha, 3) if math.isfinite(alpha) else "inf",
                "mean_first_ms": m1,
                "mean_last_ms": m2,
                "n_first": n1,
                "n_last": n2,
            })
    return {
        "alphas": alphas,
        "superlinear_stages": defects,
        "method": SUPERLINEAR_METHOD,
    }


#: (label, module, attr) — the module-level call targets inside
#: ``Memory.add`` that resolve at call time. Patched by attribute so a
#: partial checkout simply reports no samples for that stage.
_ADD_MODULE_STAGES: Tuple[Tuple[str, str, str], ...] = (
    ("authorize", "verbatim.governance", "authorize"),
    ("capture_tx.ingest", "verbatim.memory.facade", "ingest_envelope"),
    ("update_prescan", "verbatim.memory.facade", "detect_update_candidates"),
    ("source_state", "verbatim.sourcestate.state", "ensure_state"),
    ("transition", "verbatim.sourcestate.transitions", "transition"),
)

#: (label, memory-or-engine attribute) — instance call targets on THIS
#: env's objects: obligation fan-out, source-job enqueue, session
#: receipt tracking (the V6-02.16 suspects), and the post-commit
#: readiness snapshot.
_ADD_INSTANCE_STAGES: Tuple[Tuple[str, str], ...] = (
    ("obligations", "_engine.record_obligations"),
    ("obligation_defer", "_engine.defer"),
    ("source_jobs", "_source_jobs"),
    ("session_receipt_tx", "_record_session_receipt"),
    ("session_compact", "_compact_session"),
    ("receipt_caps_post", "_receipt_caps"),
)


def _wrap_attr(module_name: str, attr: str, sink: StageSamples,
               stage: str) -> Optional[Tuple[Any, str, Any]]:
    """Module-level stage wrap (timers.py idiom)."""
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return None
    orig = getattr(mod, attr, None)
    if orig is None or not callable(orig):
        return None

    @functools.wraps(orig)
    def _timed(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            sink.add(stage, (time.perf_counter() - t0) * 1000.0)

    setattr(mod, attr, _timed)
    return (mod, attr, orig)


def _wrap_obj(obj: Any, attr: str, sink: StageSamples,
              stage: str) -> Optional[Tuple[Any, str, Any]]:
    """Instance-level stage wrap — binds on the instance so the call
    site (``self._engine.record_obligations(...)`` etc.) sees it."""
    if obj is None:
        return None
    orig = getattr(obj, attr, None)
    if orig is None or not callable(orig):
        return None

    @functools.wraps(orig)
    def _timed(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            sink.add(stage, (time.perf_counter() - t0) * 1000.0)

    try:
        setattr(obj, attr, _timed)
    except Exception:
        return None
    return (obj, attr, orig)


class AddStageProfiler:
    """Per-stage timing inside ``Memory.add`` (V6-02.16).

    Monkey-patches the real call targets for the measurement window
    and restores them on exit — same measurement-only discipline as
    ``eval.v5.timers.StageProfiler``, pointed at the write path
    instead of the read path:

    * ``commit_tx`` — the whole ``store.tx()`` block (lock wait +
      capture + obligations + state + transition + prescan + commit);
    * ``authorize``, ``capture_tx.ingest``, ``update_prescan``,
      ``source_state``, ``transition`` — module-level stages the
      facade resolves at call time;
    * ``obligations``, ``obligation_defer``, ``source_jobs``,
      ``session_receipt_tx``, ``session_compact``,
      ``receipt_caps_post`` — the obligation fan-out / session
      receipt tracking suspects V6-02.16 names;
    * residual add time lands in ``add_other`` so the column adds up.
      NOTE the inner stages NEST inside ``commit_tx``: per-add time =
      ``commit_tx`` + ``session_compact`` + ``receipt_caps_post`` +
      ``add_other`` — never the sum of every row.
    """

    def __init__(self, env: ConsumerEnv) -> None:
        self.env = env
        self.sink = StageSamples()
        self._undo: List[Tuple[Any, str, Any]] = []

    def __enter__(self) -> "AddStageProfiler":
        for stage, mod, attr in _ADD_MODULE_STAGES:
            rec = _wrap_attr(mod, attr, self.sink, stage)
            if rec is not None:
                self._undo.append(rec)
        mem = self.env.memory
        for stage, dotted in _ADD_INSTANCE_STAGES:
            obj: Any = mem
            attr = dotted
            if "." in dotted:
                owner_name, attr = dotted.split(".", 1)
                obj = getattr(mem, owner_name, None)
            rec = _wrap_obj(obj, attr, self.sink, stage)
            if rec is not None:
                self._undo.append(rec)
        # Whole-transaction timing: wrap store.tx so the returned
        # context manager reports enter→exit wall time (lock admission
        # through COMMIT).
        store = getattr(mem, "_store", None)
        orig_tx = getattr(store, "tx", None)
        if store is not None and orig_tx is not None:
            sink = self.sink

            class _TimedTx:
                def __init__(self, cm: Any) -> None:
                    self._cm = cm
                    self._t0: Optional[float] = None

                def __enter__(self) -> Any:
                    self._t0 = time.perf_counter()
                    return self._cm.__enter__()

                def __exit__(self, *exc: Any) -> Any:
                    try:
                        return self._cm.__exit__(*exc)
                    finally:
                        if self._t0 is not None:
                            sink.add(
                                "commit_tx",
                                (time.perf_counter() - self._t0) * 1000.0,
                            )

            @functools.wraps(orig_tx)
            def _timed_tx(*a: Any, **kw: Any) -> Any:
                return _TimedTx(orig_tx(*a, **kw))

            try:
                setattr(store, "tx", _timed_tx)
                self._undo.append((store, "tx", orig_tx))
            except Exception:
                pass
        return self

    def __exit__(self, *exc: Any) -> None:
        for obj, attr, orig in reversed(self._undo):
            try:
                setattr(obj, attr, orig)
            except Exception:
                pass
        self._undo.clear()


def _profile_adds(env: ConsumerEnv, n_adds: int, *, tag: str) -> dict:
    """``n_adds`` real ``mem.add`` calls under the AddStageProfiler."""
    add_ms: List[float] = []
    with AddStageProfiler(env) as prof:
        for i in range(n_adds):
            t0 = time.perf_counter()
            try:
                env.memory.add(
                    f"add-ack profile {tag} item {i}: profile token "
                    f"prof-{tag}-{i}", infer=False)
            except Exception as exc:  # noqa: BLE001
                env.add_errors[f"prof-{tag}-{i}"] = (
                    f"{type(exc).__name__}: {exc}")
            add_ms.append((time.perf_counter() - t0) * 1000.0)
    stages = dict(prof.sink.samples)
    # Residual facade time so the columns add up honestly. The inner
    # stages (ingest, obligations, prescan, state, session receipt)
    # NEST inside commit_tx — the in-tx aggregate is commit_tx plus
    # the post-commit stages, not the sum of every sample.
    total = sum(add_ms)
    in_tx = sum(stages.get("commit_tx") or [])
    post_tx = sum(
        sum(v) for k, v in stages.items()
        if k in ("session_compact", "receipt_caps_post"))
    residual_per_op = max(0.0, total - in_tx - post_tx) / max(1, len(add_ms))
    stages.setdefault("add_other", []).append(residual_per_op)
    return {"add_ms": add_ms, "stages": stages}


def measure_add_ack_scaling(
    memories_list: Sequence[int] = (1_000, 5_000, 20_000),
    *,
    seed: int = 42,
    profile_adds: int = 24,
    worker: str = "external",
    workdir: Optional[str] = None,
    drain_every: int = 64,
) -> dict:
    """Standalone V6-02.16 probe: add-ack cost + stage breakdown at
    each checkpoint of ``memories_list``.

    Seeds cumulatively through :func:`eval.v5.harness.bulk_seed` (the
    harness-side path — profiled adds still go through ``mem.add``),
    then times ``profile_adds`` adds under :class:`AddStageProfiler`
    at each n. Any stage whose per-op cost grows faster than linear is
    named in ``superlinear_stages`` with the published heuristic —
    never absorbed silently.
    """
    ns = sorted({int(m) for m in memories_list if int(m) > 0})
    env, tmpdir = _open_a3_env(workdir=workdir, worker=worker, seed=seed)
    rows: List[dict] = []
    seeded = 0
    # One generator call — slices share the same unique-token space.
    texts = a3_texts(ns[-1] if ns else 0, seed)
    try:
        for n in ns:
            rep = bulk_seed(
                env, texts[seeded:n], drain_every=drain_every)
            seeded += rep["added"]
            prof = _profile_adds(env, profile_adds, tag=f"n{n}")
            rows.append({
                "n": seeded,
                "add_ack_ms": percentiles(prof["add_ms"]),
                "stages_ms": {
                    k: percentiles(v) for k, v in sorted(prof["stages"].items())
                },
            })
            drain_memory(env)
    finally:
        env.close()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    scaling = _stage_scaling(rows)
    return {
        "suite": "add_ack_scaling",
        "qualification": "locally_measured",
        "scale_disclosure": (
            f"checkpoints {ns} vs spec n=1k/5k/20k — local probe, not "
            "A3 qualification"
        ),
        "environment": environment(),
        "profile_adds": profile_adds,
        "checkpoints": rows,
        "stage_scaling": scaling,
        "superlinear_stages": scaling["superlinear_stages"],
        "spec_reference": "V6-02.16",
    }


def _open_a3_env(*, workdir: Optional[str], worker: str,
                 seed: int) -> Tuple[ConsumerEnv, Optional[str]]:
    """A bare ConsumerEnv for bulk-seed envelopes (no corpus items —
    seeding is bulk, not per-item)."""
    from verbatim import Memory

    tmpdir = None
    if workdir is None:
        tmpdir = tempfile.mkdtemp(prefix="verbatim-v6-eval-")
        workdir = tmpdir
    os.makedirs(workdir, exist_ok=True)
    memory = Memory(os.path.join(workdir, "mem.db"),
                    user_id="eval-user", worker=worker)
    env = ConsumerEnv(
        corpus=seed_corpus(memories=24, seed=seed),
        memory=memory, workdir=workdir, worker=worker, _tmpdir=tmpdir)
    return env, tmpdir


def measure_a3(
    *,
    memories: int = 20_000,
    seed: int = 42,
    queries: int = 120,
    checkpoints: Sequence[int] = (1_000, 5_000, 20_000),
    profile_adds: int = 24,
    checkpoint_queries: int = 32,
    worker: str = "external",
    workdir: Optional[str] = None,
    drain_every: int = 64,
    seed_budget_s: Optional[float] = None,
) -> dict:
    """A3: ~20k records / ~1M source tokens, one namespace.

    Seeding is harness-side via ``bulk_seed`` (V6-02.15): the corpus
    goes through ``Memory.add`` with a periodic drain so an
    external-worker env doesn't serialize obligations — the measured
    reads, profiled adds, and forget/settle still go through the
    public API. ``seed_budget_s`` is the wall-time guard: when the
    seed exceeds it the run truncates at the reached n and says so.

    At every declared checkpoint: add-ack is profiled per stage
    (``AddStageProfiler``), the store footprint is recorded, and a
    settled-search sample runs — giving the scaling exponents for
    add-ack, per-stage cost, index bytes, and search latency from real
    paired points, never a single number extrapolated.
    """
    budgets = dict(BUDGETS["A3"])
    cps = sorted({int(c) for c in checkpoints if 0 < int(c) < memories})
    if not cps or cps[-1] != memories:
        cps.append(memories)
    res = _new_result(
        "A3", budgets,
        {
            "memories": memories,
            "queries": queries,
            "checkpoints": cps,
            "spec_scale": {
                "records": budgets["memories"],
                "source_tokens": budgets["source_tokens"],
            },
            "seeding": "harness bulk_seed (V6-02.15) — periodic drain; "
                       "measured reads/adds through the public API",
            "scale_disclosure": (
                "spec-scale run" if memories >= budgets["memories"]
                else f"local n={memories} vs spec ~{budgets['memories']:,} "
                     "records — a scaling probe, not A3 qualification"),
        },
    )
    env, tmpdir = _open_a3_env(workdir=workdir, worker=worker, seed=seed)
    ckpt_rows: List[dict] = []
    seeded = 0
    truncated = False
    seed_wall = 0.0
    t_seed0 = time.perf_counter()
    # One generator call — slices share the same unique-token space.
    texts = a3_texts(memories, seed)
    try:
        for c in cps:
            if seed_budget_s is not None and (
                time.perf_counter() - t_seed0 > seed_budget_s
            ):
                truncated = True
                break
            rep = bulk_seed(
                env, texts[seeded:c], drain_every=drain_every)
            seeded += rep["added"]
            seed_wall += rep["wall_s"]
            # add-ack stage profile at this n (V6-02.16)
            prof = _profile_adds(env, profile_adds, tag=f"n{c}")
            drain_memory(env)
            # settled-search sample at this n — the scaling point
            qset = _a3_queries(
                texts[:seeded], checkpoint_queries, seed + c)
            lat, errs, sts = _measure_settled_search(env, qset)
            ckpt_rows.append({
                "n": seeded,
                "add_ack_ms": percentiles(prof["add_ms"]),
                "stages_ms": {
                    k: percentiles(v)
                    for k, v in sorted(prof["stages"].items())
                },
                "search_ms": percentiles(lat),
                "search_status": sts,
                "search_errors": errs,
                "store_bytes": _store_bytes(env.workdir),
            })
        res.support["settle"] = settle(
            env, timeout_s=max(120.0, seeded * 0.05))
        # The declared-scale read: full query count at the final n.
        qset = _a3_queries(texts[:seeded], queries, seed)
        lat, errs, sts = _measure_settled_search(env, qset)
        res.search_ms = percentiles(lat)
        res.measurements["search_status"] = sts
        res.support["search_errors"] = errs
        res.add_ack_ms = (
            ckpt_rows[-1]["add_ack_ms"] if ckpt_rows else {})
        rss = peak_rss_mib()
        scaling = _stage_scaling(ckpt_rows)
        search_alpha = _growth_alpha(
            (ckpt_rows[0]["search_ms"] or {}).get("p95"),
            (ckpt_rows[-1]["search_ms"] or {}).get("p95"),
            ckpt_rows[0]["n"], ckpt_rows[-1]["n"],
        ) if len(ckpt_rows) >= 2 else None
        bytes_alpha = _growth_alpha(
            float(ckpt_rows[0]["store_bytes"] or 0),
            float(ckpt_rows[-1]["store_bytes"] or 0),
            ckpt_rows[0]["n"], ckpt_rows[-1]["n"],
        ) if len(ckpt_rows) >= 2 else None
        add_alpha = _growth_alpha(
            (ckpt_rows[0]["add_ack_ms"] or {}).get("mean"),
            (ckpt_rows[-1]["add_ack_ms"] or {}).get("mean"),
            ckpt_rows[0]["n"], ckpt_rows[-1]["n"],
        ) if len(ckpt_rows) >= 2 else None
        res.measurements.update({
            "seed": {
                "added": seeded,
                "wall_s": round(seed_wall, 3),
                "drain_totals": dict(env.drain),
                "truncated": truncated,
                "seed_budget_s": seed_budget_s,
            },
            "add_ack_profile": {
                "checkpoints": ckpt_rows,
                "profile_adds": profile_adds,
            },
            "stage_scaling": scaling,
            "superlinear_stages": scaling["superlinear_stages"],
            "search_scaling": {
                "checkpoints": [
                    {"n": r["n"], "search_ms": r["search_ms"]}
                    for r in ckpt_rows
                ],
                "alpha": (
                    round(search_alpha, 3)
                    if search_alpha is not None
                    and math.isfinite(search_alpha) else search_alpha),
                "note": "search p95 ∝ n^alpha across measured "
                        "checkpoints",
            },
            "index_growth": {
                "bytes_by_n": [
                    {"n": r["n"], "store_bytes": r["store_bytes"]}
                    for r in ckpt_rows
                ],
                "alpha": (
                    round(bytes_alpha, 3)
                    if bytes_alpha is not None
                    and math.isfinite(bytes_alpha) else bytes_alpha),
                "note": "store bytes (db+wal+shm) ∝ n^alpha — the "
                        "spec's 'sublinear index growth' read as "
                        "alpha < 1",
            },
            "add_ack_scaling": {
                "alpha": (
                    round(add_alpha, 3)
                    if add_alpha is not None
                    and math.isfinite(add_alpha) else add_alpha),
            },
            "peak_rss_mib": rss,
            "store_bytes": _store_bytes(env.workdir),
        })
    finally:
        env.close()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    _verdict(res, budgets)
    _unsettled_note(res)
    m = res.measurements
    for d in (m.get("superlinear_stages") or []):
        res.misses.append(
            f"add-ack stage {d['stage']!r} grew superlinearly "
            f"(alpha={d['alpha']}, mean {d['mean_first_ms']:.2f}ms → "
            f"{d['mean_last_ms']:.2f}ms over n={d['n_first']}→"
            f"{d['n_last']}) — V6-02.16 named defect")
    ig = m.get("index_growth") or {}
    if ig.get("alpha") is not None and math.isfinite(ig["alpha"]) and (
        ig["alpha"] > budgets["index_growth_alpha_max"]
    ):
        res.misses.append(
            f"index growth alpha {ig['alpha']} > "
            f"{budgets['index_growth_alpha_max']} (superlinear)")
    rss = m.get("peak_rss_mib")
    if rss is not None and rss > budgets["peak_rss_mib"]:
        res.misses.append(
            f"peak RSS {rss:.0f}MiB > {budgets['peak_rss_mib']:.0f}MiB")
    if truncated:
        res.misses.append(
            f"seed truncated at n={seeded} by seed_budget_s — the "
            "declared scale was never reached")
    _finish_verdict(res)
    return res.to_dict()


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def _verdict(res: EnvelopeResult, budgets: dict) -> None:
    """Score measured percentiles against the envelope — misses named,
    never smoothed (V5 idiom + the V6 qualification floor)."""
    s = res.search_ms or {}
    a = res.add_ack_ms or {}
    if s.get("n", 0) == 0:
        res.misses.append("no settled-search samples executed")
    else:
        p95 = budgets.get("search_p95_ms")
        if p95 is not None and s.get("p95") is not None and s["p95"] > p95:
            res.misses.append(
                f"search p95 {s['p95']:.1f}ms > {p95:.0f}ms")
        p99 = budgets.get("search_p99_ms")
        if p99 is not None and s.get("p99") is not None and s["p99"] > p99:
            res.misses.append(
                f"search p99 {s['p99']:.1f}ms > {p99:.0f}ms")
    if budgets.get("add_ack_p95_ms") is not None:
        if a.get("n", 0) == 0:
            res.misses.append("no add-ack samples executed")
        elif a.get("p95") is not None and a["p95"] > budgets["add_ack_p95_ms"]:
            res.misses.append(
                f"add-ack p95 {a['p95']:.1f}ms > "
                f"{budgets['add_ack_p95_ms']:.0f}ms")
    # V5-20.02 carried: qualification requires the full protocol.
    q = res.scale.get("declared_queries_per_rep") or res.scale.get(
        "reader_queries") or res.scale.get("queries") or 0
    n_samples = s.get("n") or 0
    res.support["qualification_gap"] = {
        "queries_measured": n_samples,
        "queries_required": QUALIFICATION["min_queries"],
        "distinct_intents": res.scale.get("distinct_intents"),
        "distinct_intents_required": QUALIFICATION["min_distinct_intents"],
        "repetitions": res.repetitions,
        "repetitions_required": QUALIFICATION["repetitions"],
        "declared_queries_per_rep": q,
    }
    if (
        n_samples >= QUALIFICATION["min_queries"]
        and (res.scale.get("distinct_intents") or 0)
            >= QUALIFICATION["min_distinct_intents"]
        and res.repetitions >= QUALIFICATION["repetitions"]
        and not res.misses
    ):
        res.qualification = "qualified"
    elif res.qualification != "unavailable":
        res.qualification = "locally_measured"


def _unsettled_note(res: EnvelopeResult) -> None:
    """V6-02.13: pending/partial/blocked/error rows cannot qualify
    settled latency — count them as a named finding."""
    sts = res.measurements.get("search_status") or {}
    bad = {k: v for k, v in sts.items() if k not in _SETTLED_STATUSES}
    if bad:
        res.misses.append(
            f"settled-search rows returned non-settled status "
            f"{dict(sorted(bad.items()))} (V6-02.13 — these rows "
            "cannot qualify)")
    _finish_verdict(res)


def _finish_verdict(res: EnvelopeResult) -> None:
    """Resolve the final verdict after every miss source ran."""
    if res.status == "measured":
        res.verdict = "missed" if res.misses else "passed"
        if res.misses and res.qualification == "qualified":
            res.qualification = "locally_measured"


def run_envelopes(
    scale: str = "quick",
    *,
    seed: int = 42,
    workdir: Optional[str] = None,
    params: Optional[dict] = None,
    suites: Optional[Sequence[str]] = None,
) -> dict:
    """Run the V6 envelope set at the declared scale.

    ``scale="quick"`` is the CI-shaped run (~2k A3 seed, disclosed);
    ``"full"`` walks toward the spec scales (1k/5k A0/A1, 20k A3).
    ``params`` overrides individual knobs (tests, smoke runs);
    ``suites`` selects a subset. Every envelope returns
    ``{status, measurements, misses, verdict}`` in the
    EnvelopeResult-ish shape the portfolio consumes.
    """
    p: Dict[str, Any] = {
        "a0_memories": 256, "a0_queries": 60, "a0_reps": 1,
        "cache_memories": 256, "cache_queries": 40, "cache_passes": 3,
        "neural_memories": 256, "neural_queries": 40,
        "a1_memories": 256, "a1_reads": 40, "a1_writes": 16,
        "a1_vis": 6,
        "a3_memories": 2_048, "a3_queries": 40,
        "a3_checkpoints": (512, 1_024, 2_048), "a3_profile_adds": 16,
    } if scale == "quick" else {
        "a0_memories": 1_000, "a0_queries": 400, "a0_reps": 3,
        "cache_memories": 1_000, "cache_queries": 200, "cache_passes": 3,
        "neural_memories": 1_000, "neural_queries": 200,
        "a1_memories": 5_000, "a1_reads": 300, "a1_writes": 120,
        "a1_vis": 20,
        "a3_memories": 20_000, "a3_queries": 120,
        "a3_checkpoints": (1_000, 5_000, 20_000), "a3_profile_adds": 24,
    }
    if params:
        p.update(params)
    want = set(suites or ("a0", "a0_cache", "a0_neural", "a1", "a3"))
    out: Dict[str, Any] = {
        "suite": "envelopes_v6",
        "scale": scale,
        "qualification": "locally_measured",
        "environment": environment(),
        "spec_budgets": {k: dict(v) for k, v in BUDGETS.items()},
        "qualification_floor": dict(QUALIFICATION),
        "envelopes": {},
    }
    envs = out["envelopes"]

    if "a0" in want:
        envs["a0"] = _guard("a0", measure_a0,
                            memories=p["a0_memories"], seed=seed,
                            queries=p["a0_queries"],
                            repetitions=p["a0_reps"], workdir=workdir)
    if "a0_cache" in want:
        envs["a0_cache"] = _guard(
            "a0_cache", measure_a0_cache,
            memories=p["cache_memories"], seed=seed,
            queries=p["cache_queries"], passes=p["cache_passes"],
            workdir=workdir)
    if "a0_neural" in want:
        envs["a0_neural"] = _guard(
            "a0_neural", measure_a0_neural,
            memories=p["neural_memories"], seed=seed,
            queries=p["neural_queries"], workdir=workdir)
    if "a1" in want:
        envs["a1"] = _guard(
            "a1", measure_a1, memories=p["a1_memories"], seed=seed + 1,
            reader_queries=p["a1_reads"], writer_ops=p["a1_writes"],
            visibility_probes=p["a1_vis"], workdir=workdir)
    if "a3" in want:
        envs["a3"] = _guard(
            "a3", measure_a3, memories=p["a3_memories"], seed=seed,
            queries=p["a3_queries"], checkpoints=p["a3_checkpoints"],
            profile_adds=p["a3_profile_adds"], workdir=workdir)

    verdicts = [
        (e.get("verdict") or e.get("status")) for e in envs.values()
    ]
    out["verdict"] = (
        "failed" if any(v == "failed" for v in verdicts)
        else "missed" if any(v == "missed" for v in verdicts)
        else "inconclusive" if any(v == "unavailable" for v in verdicts)
        else "passed"
    )
    return out


def _guard(name: str, fn: Any, **kw: Any) -> dict:
    """One envelope failing must not zero the whole run — the failure
    is recorded as the envelope's honest result."""
    try:
        out = fn(**kw)
        if hasattr(out, "to_dict"):
            out = out.to_dict()
        elif not isinstance(out, dict):
            out = {"value": out}
        out.setdefault("status", "measured")
        return out
    except Exception as exc:  # noqa: BLE001 — honest failure record
        import traceback
        return {
            "name": name,
            "status": "failed",
            "verdict": "failed",
            "qualification": "failed",
            "measurements": {},
            "misses": [],
            "errors": [f"{type(exc).__name__}: {exc}"],
            "traceback": traceback.format_exc()[-2000:],
        }


__all__ = [
    "AddStageProfiler",
    "BUDGETS",
    "EnvelopeResult",
    "QUALIFICATION",
    "SUPERLINEAR_METHOD",
    "a3_texts",
    "measure_a0",
    "measure_a0_cache",
    "measure_a0_neural",
    "measure_a1",
    "measure_a3",
    "measure_add_ack_scaling",
    "run_envelopes",
]
