"""Shared plumbing for the V5 consumer-route eval harness.

Every suite here measures the *consumer route* — the public
``verbatim.Memory`` facade (add → wait/search → inspect → forget →
close) — through real disposable stores. Nothing mocks the kernel; a
capability that is absent is recorded ``unavailable``, never simulated
(SPEC_V5 §24 hard rules).

Conventions carried over from ``eval/v3/baselines.py``:

* fresh disposable store per run; corpus ids map to engine-assigned
  source ids via ``MemoryRef`` parsing (gold scoring is by id);
* errors stay in the denominator as typed outcomes;
* the managed worker is the default consumer path; harnesses that need
  deterministic settle use ``worker="external"`` plus an explicit
  ``Ingester.drain_report`` pass — the same durable queue the managed
  worker drains, disclosed in ``env.notes``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple

from .corpus import ConsumerCorpus, ConsumerTask, CorpusItem

#: Warnings that mean "nothing was delivered" — a typed non-answer, not
#: a hit (eval.v3 convention, adapted to the facade's warning surface).
NO_ANSWER_WARNINGS = frozenset({
    "no_authorized_evidence",
    "no_evidence_matched",
    "no_signal",
    "insufficient_support",
    "abstained_uncovered_terms",
    "processing_pending",
    "evidence_pending_derivation",
})

#: Default bounds for seeding patience.
_ADD_ATTEMPTS = 30
_ADD_SLEEP_S = 0.05
_DRAIN_ROUNDS = 40
_DRAIN_LIMIT = 4096


@dataclass
class ConsumerEnv:
    """A live ``Memory`` instance seeded with one corpus.

    ``item_source`` maps corpus item id → facade source id (parsed from
    the add result's ``MemoryRef``); ``source_item`` is the reverse.
    ``add_ms``/``forget_ms`` keep per-call wall time so envelope suites
    can reuse the seeding trace instead of re-measuring.
    """

    corpus: ConsumerCorpus
    memory: Any
    workdir: str
    worker: str = "external"
    item_source: dict = field(default_factory=dict)
    source_item: dict = field(default_factory=dict)
    refs: dict = field(default_factory=dict)          # item_id -> ref str
    add_results: dict = field(default_factory=dict)   # item_id -> AddResult
    forget_results: dict = field(default_factory=dict)
    add_ms: dict = field(default_factory=dict)        # item_id -> ms
    forget_ms: dict = field(default_factory=dict)
    add_errors: dict = field(default_factory=dict)
    drain: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    _tmpdir: Optional[str] = None

    # ---- id mapping ---------------------------------------------------
    def corpus_id(self, source_id: Optional[str]) -> Optional[str]:
        return self.source_item.get(source_id) if source_id else None

    def source_id(self, corpus_id: str) -> Optional[str]:
        return self.item_source.get(corpus_id)

    def close(self) -> None:
        try:
            if self.memory is not None:
                self.memory.close()
        except Exception:
            pass
        if self._tmpdir and os.path.basename(self._tmpdir).startswith(
            "verbatim-v5-eval-"
        ):
            shutil.rmtree(self._tmpdir, ignore_errors=True)


def _is_contention(exc: BaseException) -> bool:
    """Store-busy/backpressure is transient writer contention — the
    documented retry surface (tests/v5 ``add_wait`` convention)."""
    name = type(exc).__name__.upper()
    msg = str(exc).lower()
    return (
        "locked" in msg
        or "busy" in msg
        or "BACKPRESSURE" in name
        or "STORE_BUSY" in name
    )


def add_with_retry(memory: Any, text: str, *,
                   attempts: int = _ADD_ATTEMPTS,
                   sleep_s: float = _ADD_SLEEP_S,
                   **kw: Any) -> Any:
    """``memory.add`` tolerant of managed-worker write contention.

    The projection worker legitimately holds the write lock for bounded
    windows; retrying preserves the assertion (the add must succeed)
    without weakening it. A non-contention error raises immediately.
    """
    last: Optional[BaseException] = None
    for _ in range(attempts):
        try:
            return memory.add(text, **kw)
        except Exception as exc:  # noqa: BLE001 — contention is typed or raw
            last = exc
            if not _is_contention(exc):
                raise
            time.sleep(sleep_s)
    assert last is not None
    raise last


def drain_memory(env: ConsumerEnv, *, limit: int = _DRAIN_LIMIT,
                 rounds: int = _DRAIN_ROUNDS) -> dict:
    """Drive the durable job queue to empty on an external-worker env.

    Uses the same ``Ingester.drain_report`` the managed worker calls —
    lane priority, generation fencing, and retry semantics included.
    The report is recorded on ``env.drain`` (processed/succeeded/failed/
    still_pending) so a stalled queue is visible, never silent.
    """
    if env.worker != "external":
        env.notes.append("managed worker drains asynchronously")
        return env.drain
    try:
        from verbatim.ingest import Ingester
    except ImportError as exc:
        env.notes.append(f"drain unavailable: {exc}")
        env.drain["error"] = f"import: {exc}"
        return env.drain
    try:
        ing = Ingester(
            env.memory._store, env.memory._cfg,
            encoder=getattr(env.memory, "_encoder", None),
        )
        totals = dict(env.drain)  # accumulate across settle() calls
        totals.setdefault("processed", 0)
        totals.setdefault("succeeded", 0)
        totals.setdefault("failed", 0)
        totals.setdefault("deferred", 0)
        totals.setdefault("rounds", 0)
        for _ in range(rounds):
            rep = ing.drain_report(limit=limit)
            for k in ("processed", "succeeded", "failed", "deferred"):
                totals[k] += int(rep.get(k, 0))
            totals["rounds"] += 1
            totals["still_pending"] = int(rep.get("still_pending", 0))
            if rep.get("processed", 0) == 0:
                break
        env.drain.update(totals)
    except Exception as exc:  # noqa: BLE001
        env.notes.append(f"drain failed: {type(exc).__name__}: {exc}")
        env.drain["error"] = f"{type(exc).__name__}: {exc}"
    return env.drain


def settle(env: ConsumerEnv, *, timeout_s: float = 120.0) -> dict:
    """Wait for the seeded corpus to reach steady state.

    External-worker envs drain synchronously then verify the last
    receipt's readiness obligations through the public
    ``wait_ready`` path. Managed envs wait on the last receipt (the
    worker's FIFO ordering makes it the frontier). Reports the honest
    terminal state — ``ready``/``partial``/``blocked`` — and never
    relabels a stalled pipeline.
    """
    t0 = time.monotonic()
    out: dict[str, Any] = {"state": "unknown", "waited_s": 0.0}
    if env.worker == "external":
        rep = drain_memory(env)
        out["drain"] = dict(rep)
    # Confirm through the public readiness surface on the newest receipt.
    results = [r for r in env.add_results.values()]
    if results:
        last = results[-1]
        deadline = t0 + timeout_s
        state = "pending"
        while True:
            try:
                rd = env.memory.wait_ready(last, timeout_ms=2000)
                state = rd.state
            except Exception as exc:  # noqa: BLE001
                state = f"error:{type(exc).__name__}"
            if state in ("ready", "blocked", "unavailable") or (
                time.monotonic() > deadline
            ):
                break
            time.sleep(0.05)
        out["state"] = state
    out["waited_s"] = round(time.monotonic() - t0, 3)
    return out


def seed_corpus_env(
    corpus: ConsumerCorpus,
    *,
    workdir: Optional[str] = None,
    worker: str = "external",
    user_id: str = "eval-user",
    drain: bool = True,
    forget: bool = True,
    on_add: Optional[Callable[[CorpusItem, Any, float], None]] = None,
    memory_kwargs: Optional[dict] = None,
) -> ConsumerEnv:
    """Open a ``Memory`` and run the corpus's full lifecycle setup.

    Item order is corpus order; ``supersedes`` items add *after* their
    predecessor with the real ``replaces=`` transition. ``forget`` items
    are added, settled, then deleted through ``memory.forget`` — the
    distractor is produced by the real closure path, not by omission.
    Setup failures are recorded per item and stay visible.
    """
    from verbatim import Memory  # lazy: eval harness, not a package import
    from verbatim.memory.types import MemoryRef

    tmpdir = None
    if workdir is None:
        tmpdir = tempfile.mkdtemp(prefix="verbatim-v5-eval-")
        workdir = tmpdir
    os.makedirs(workdir, exist_ok=True)
    kw = {"worker": worker}
    kw.update(memory_kwargs or {})
    memory = Memory(os.path.join(workdir, "mem.db"), user_id=user_id, **kw)
    env = ConsumerEnv(corpus=corpus, memory=memory, workdir=workdir,
                      worker=worker, _tmpdir=tmpdir)

    for item in corpus.items:
        t0 = time.perf_counter()
        kw_add: dict[str, Any] = {"infer": item.infer}
        if item.supersedes:
            pred_ref = env.refs.get(item.supersedes)
            if pred_ref is None:
                env.add_errors[item.id] = (
                    f"supersede target {item.supersedes!r} not added"
                )
                continue
            kw_add["replaces"] = pred_ref
        try:
            res = add_with_retry(memory, item.text, **kw_add)
        except Exception as exc:  # noqa: BLE001 — recorded, kept visible
            env.add_errors[item.id] = f"{type(exc).__name__}: {exc}"
            env.add_ms[item.id] = (time.perf_counter() - t0) * 1000.0
            continue
        ms = (time.perf_counter() - t0) * 1000.0
        env.add_ms[item.id] = ms
        env.add_results[item.id] = res
        env.refs[item.id] = res.ref
        try:
            sid = MemoryRef.parse(res.ref).source_id
            env.item_source[item.id] = sid
            env.source_item[sid] = item.id
        except Exception:
            env.notes.append(f"{item.id}: unparseable ref {res.ref!r}")
        if on_add is not None:
            on_add(item, res, ms)

    # Deletions run through the same route AFTER a settle — a distractor
    # must have been live to prove closure, not merely never indexed.
    if forget:
        forgotten = [i for i in corpus.items if i.forget]
        if forgotten:
            settle(env)
            for item in forgotten:
                ref = env.refs.get(item.id)
                if ref is None:
                    env.add_errors.setdefault(
                        item.id, "forget item never added"
                    )
                    continue
                t0 = time.perf_counter()
                try:
                    res = env.memory.forget(ref)
                    env.forget_results[item.id] = res
                except Exception as exc:  # noqa: BLE001
                    env.add_errors[item.id] = (
                        f"forget: {type(exc).__name__}: {exc}"
                    )
                env.forget_ms[item.id] = (
                    (time.perf_counter() - t0) * 1000.0
                )
    if drain:
        settle(env)
    return env


def bulk_seed(
    env: ConsumerEnv,
    texts: Sequence[str],
    *,
    drain_every: int = 64,
    infer: bool = False,
    on_add: Optional[Callable[[int, Any, float], None]] = None,
) -> dict:
    """Bulk-seed raw ``texts`` through the public ``Memory.add`` path.

    The V6 scale envelopes (SPEC_V6 V6-02.15) may seed through a
    dedicated bulk path so seeding cost stays honest without becoming
    serial-ack-bound. Every item still goes through the real facade
    with the contention-retry discipline — nothing touches the store
    directly — but on an external-worker env this drains the durable
    queue every ``drain_every`` adds instead of once at the end.
    Without the periodic drain, tens of thousands of receipts pile
    their obligations into one backlog and the measured seed wall is
    the queue's serialization artifact, not the store's ingest cost.
    On a managed-worker env :func:`drain_memory` is already a
    documented no-op (the shared worker drains concurrently), so the
    same call sites stay honest in both modes. ``drain_every=0``
    disables the periodic drain entirely (one final drain still runs).

    Per-item receipts are NOT retained — a 20k-item seed would pin
    20k result objects for no scoring purpose. The last successful
    receipt is kept on ``env.add_results`` under the synthetic
    ``"bulk:last"`` key so :func:`settle` can still verify the
    readiness frontier; failures land in ``env.add_errors`` exactly
    like :func:`seed_corpus_env` records them. ``env.add_ms`` is not
    filled per item (seed timing is reported as the returned
    aggregate, not a per-item trace); callers that need per-add
    latency or ids pass ``on_add(index, result_or_None, ms)``.

    Returns ``{added, errors, drain_totals, wall_s}`` — real counts
    from the accumulated drain report (``env.drain``), never
    projected ones.
    """
    t0 = time.perf_counter()
    added = 0
    errors = 0
    last_res = None
    n = len(texts)
    for i, text in enumerate(texts):
        t_add = time.perf_counter()
        try:
            res = add_with_retry(env.memory, text, infer=infer)
        except Exception as exc:  # noqa: BLE001 — recorded, kept visible
            env.add_errors[f"bulk-{i}"] = f"{type(exc).__name__}: {exc}"
            errors += 1
            res = None
        ms = (time.perf_counter() - t_add) * 1000.0
        if res is not None:
            added += 1
            last_res = res
        if on_add is not None:
            on_add(i, res, ms)
        if drain_every > 0 and (i + 1) % drain_every == 0 and i + 1 < n:
            drain_memory(env)
    # Final drain covers the remainder slice (and the whole seed when
    # drain_every >= n). On managed envs this is the documented no-op —
    # the worker has been draining concurrently throughout.
    drain_memory(env)
    if last_res is not None:
        env.add_results["bulk:last"] = last_res
    return {
        "added": added,
        "errors": errors,
        "drain_totals": dict(env.drain),
        "wall_s": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@dataclass
class ScoredQuery:
    """One task's measured outcome through the consumer route."""

    task_id: str
    category: str
    status: str
    returned_ids: Tuple[str, ...]
    n_items: int
    abstained: bool
    expected_ids: Tuple[str, ...]
    expected_abstain: bool
    forbidden_hits: Tuple[str, ...]
    latency_ms: float
    delivered_bytes: int
    warnings: Tuple[str, ...] = ()
    error: Optional[str] = None

    @property
    def recall(self) -> Optional[float]:
        if not self.expected_ids:
            return None
        got = set(self.returned_ids) & set(self.expected_ids)
        return len(got) / len(self.expected_ids)

    @property
    def precision(self) -> Optional[float]:
        if not self.returned_ids:
            return None
        got = set(self.returned_ids) & set(self.expected_ids)
        return len(got) / len(self.returned_ids)

    @property
    def abstain_correct(self) -> Optional[bool]:
        if not self.expected_abstain:
            return None
        return self.abstained

    @property
    def clean(self) -> bool:
        """No forbidden id was delivered (closure/staleness honesty)."""
        return not self.forbidden_hits


def hit_source_id(env: ConsumerEnv, hit: Any) -> Optional[str]:
    """Resolve one ``Hit`` to its source id (ref → MemoryRef → source).

    Claim/view hits carry ``object_ref`` instead of a source ref; they
    resolve through ``claim_evidence`` → ``spans`` on a scoring-side read
    — the same mapping convention as the v3 harness's ``claim_src``.
    """
    ref = getattr(hit, "ref", "") or ""
    if ref:
        try:
            from verbatim.memory.types import MemoryRef
            return MemoryRef.parse(ref).source_id
        except Exception:
            pass
    obj = getattr(hit, "object_ref", "") or ""
    if obj:
        return _object_source_id(env, obj)
    return None


def _object_source_id(env: ConsumerEnv, object_ref: str) -> Optional[str]:
    """``vobj1.<kind>.<id>.<rev>`` → source via claim_evidence/spans."""
    parts = object_ref.split(".")
    if len(parts) < 4 or parts[1] != "claim":
        return None
    claim_id = parts[2]
    try:
        with env.memory._store.read() as conn:
            row = conn.execute(
                "SELECT s.source_id FROM claim_evidence ce"
                " JOIN spans s ON s.span_id = ce.span_id"
                " WHERE ce.claim_id = ? LIMIT 1",
                (claim_id,),
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def delivered_payload_bytes(result: Any) -> int:
    """The full delivered payload, serialized (V5-32.02).

    ``SearchResult.to_dict()`` covers quotes, refs, labels, warnings,
    and wrappers — the same bytes the caller would have to read. Falls
    back to ``str()`` for arms without ``to_dict``.
    """
    import json
    try:
        payload = json.dumps(
            result.to_dict(), sort_keys=True, default=str
        )
    except Exception:
        payload = str(result)
    return len(payload.encode("utf-8"))


def score_search(env: ConsumerEnv, task: ConsumerTask, result: Any,
                 latency_ms: float, *, k: int = 8) -> ScoredQuery:
    """Map one SearchResult to corpus ids and score it (by id, never by
    fuzzy text)."""
    items = list(getattr(result, "items", []) or [])
    returned: list[str] = []
    for h in items:
        cid = env.corpus_id(hit_source_id(env, h))
        if cid is not None and cid not in returned:
            returned.append(cid)
        if len(returned) >= k:
            break
    warns = tuple(str(w) for w in (getattr(result, "warnings", []) or []))
    status = str(getattr(result, "status", "unknown"))
    # A zero-item response carrying only non-answer warnings is a typed
    # abstention for scoring; a nonempty response is a delivered answer.
    abstained = not items and (
        status in ("pending", "blocked", "unavailable")
        or bool(set(warns) & NO_ANSWER_WARNINGS)
        or not warns  # empty with no explanation still abstains
    )
    forbidden = [c for c in returned if c in set(task.forbidden_ids)]
    return ScoredQuery(
        task_id=task.task_id,
        category=task.category,
        status=status,
        returned_ids=tuple(returned[:k]),
        n_items=len(items),
        abstained=abstained,
        expected_ids=tuple(task.expected_ids),
        expected_abstain=bool(task.expected_abstain),
        forbidden_hits=tuple(forbidden),
        latency_ms=latency_ms,
        delivered_bytes=delivered_payload_bytes(result),
        warnings=warns,
    )


def run_task(env: ConsumerEnv, task: ConsumerTask, *, k: int = 8,
             **search_kw: Any) -> ScoredQuery:
    """Execute one task query through ``memory.search`` and score it.
    Errors stay in the denominator as ``error`` outcomes (§53.05 idiom).
    """
    t0 = time.perf_counter()
    try:
        res = env.memory.search(task.query, limit=k, **search_kw)
    except Exception as exc:  # noqa: BLE001
        return ScoredQuery(
            task_id=task.task_id, category=task.category,
            status="error", returned_ids=(), n_items=0,
            abstained=False, expected_ids=tuple(task.expected_ids),
            expected_abstain=bool(task.expected_abstain),
            forbidden_hits=(),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            delivered_bytes=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    return score_search(
        env, task, res, (time.perf_counter() - t0) * 1000.0, k=k
    )


def percentiles(samples: Sequence[float]) -> dict:
    """Nearest-rank p50/p95/p99/max over milliseconds samples.

    ``n=0`` yields explicit ``None``s — an empty latency denominator is
    ``null`` in the report, never a fabricated 0.
    """
    vals = sorted(float(s) for s in samples)
    if not vals:
        return {"n": 0, "p50": None, "p95": None, "p99": None,
                "max": None, "mean": None}
    def _p(q: float) -> float:
        i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        return vals[i]
    return {
        "n": len(vals),
        "p50": _p(0.50),
        "p95": _p(0.95),
        "p99": _p(0.99),
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
    }


def peak_rss_mib() -> Optional[float]:
    """Peak resident set of this process (MiB) — ru_maxrss on Linux."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def current_rss_mib() -> Optional[float]:
    """Current RSS via /proc when available (Linux), else None."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def environment() -> dict:
    """The §20 reference-machine disclosure block — pinned honestly."""
    import platform
    import sqlite3
    import sys
    env = {
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        env["loadavg_1m"] = os.getloadavg()[0]
    except (OSError, AttributeError):
        pass
    return env


__all__ = [
    "ConsumerEnv",
    "NO_ANSWER_WARNINGS",
    "ScoredQuery",
    "add_with_retry",
    "bulk_seed",
    "current_rss_mib",
    "delivered_payload_bytes",
    "drain_memory",
    "environment",
    "hit_source_id",
    "peak_rss_mib",
    "percentiles",
    "run_task",
    "score_search",
    "seed_corpus_env",
    "settle",
]
