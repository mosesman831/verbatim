"""Per-stage timing + SQL statement counts for the V5 consumer route
(E95 / SPEC_V5 §33.02–33.06).

Two honest instruments, both disclosed in the report:

1. **Boundary timings** (V5-33.02) — measured at the public call edges:
   add ack, source-visibility lag (acceptance commit → source
   projection visible), settled search, immediate add→search, readiness
   wait, forget ack, close.

2. **Stage attribution** (V5-33.03) — a measurement-only profiler wraps
   the *real* stage functions inside the governed recall lane
   (``analyze``, ``_authorized_scopes``, ``classify``, controller
   planning, ``run_lanes``, ``build_union``, ``fuse``,
   ``assemble_groups``, abstention verdict, ``assemble_packs``,
   ``log_decision``) and the facade's own stages (causal barrier via
   ``engine.wait_ready``, ``source_candidates``, source-lane fusion).
   Wrapping is monkey-patching for the duration of the measurement —
   no production code changes; the wrapper adds a ``perf_counter`` pair
   per call and nothing else. Residual facade time is reported as
   ``facade_other`` so the column adds up honestly.

3. **SQL counts** (V5-33.05) — ``sqlite3.set_trace_callback`` on the
   store's shared reader/writer connections counts *actual statements
   executed* during a measured call. Not estimates, not ORM counters.

Everything is opt-in at run time and reported with denominators.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .harness import ConsumerEnv, percentiles, settle


# ---------------------------------------------------------------------------
# stage attribution
# ---------------------------------------------------------------------------

#: (stage label, module, attribute) — the real functions recall_v3 calls
#: at its stage boundaries. Patched by attribute on the defining/binding
#: module so call-time lookup sees the wrapper.
_GOVERNED_STAGES: Tuple[Tuple[str, str, str], ...] = (
    ("analyze", "verbatim.retrieval.v3.recall", "analyze"),
    ("authorize", "verbatim.retrieval.v3.recall", "_authorized_scopes"),
    ("classify", "verbatim.retrieval.v3.recall", "classify"),
    ("controller.discretize",
     "verbatim.retrieval.v3.controller", "discretize"),
    ("controller.plan_routes",
     "verbatim.retrieval.v3.controller", "plan_routes"),
    ("lanes", "verbatim.retrieval.v3.lanes", "run_lanes"),
    ("union", "verbatim.retrieval.v3.union", "build_union"),
    ("fusion", "verbatim.retrieval.v3.fusion_v3", "fuse"),
    ("groups", "verbatim.retrieval.v3.pack", "assemble_groups"),
    ("abstain", "verbatim.retrieval.v3.recall", "_abstain_fn"),
    ("packs", "verbatim.retrieval.v3.pack", "assemble_packs"),
    ("journal", "verbatim.retrieval.v3.controller", "log_decision"),
)

#: facade-level stages (module attribute names inside
#: verbatim.memory.facade) — the consumer route's own work.
_FACADE_STAGES: Tuple[Tuple[str, str, str], ...] = (
    ("source_lane", "verbatim.memory.facade", "_source_candidates"),
    ("source_fuse", "verbatim.memory.facade", "_fuse"),
)


@dataclass
class StageSamples:
    """Accumulated per-stage wall-clock samples (ms)."""

    samples: Dict[str, List[float]] = field(default_factory=dict)

    def add(self, stage: str, ms: float) -> None:
        self.samples.setdefault(stage, []).append(ms)

    def rows(self) -> Dict[str, dict]:
        return {k: percentiles(v) for k, v in sorted(self.samples.items())}


def _wrap_stage(module_name: str, attr: str, sink: StageSamples,
                stage: str) -> Optional[Tuple[Any, Any, Any]]:
    """Install a timing wrapper; returns (module, attr, original) or
    None when the attribute is absent (an unprovisioned stage reports
    nothing rather than failing)."""
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return None
    orig = getattr(mod, attr, None)
    if orig is None:
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


class StageProfiler:
    """Context manager: wraps the stage functions, restores on exit.

    Also wraps ``engine.wait_ready`` on this env's engine instance —
    the causal-barrier stage of ``Memory.search``.
    """

    def __init__(self, env: ConsumerEnv) -> None:
        self.env = env
        self.sink = StageSamples()
        self._undo: List[Tuple[Any, str, Any]] = []

    def __enter__(self) -> "StageProfiler":
        for stage, mod, attr in (
            _GOVERNED_STAGES + _FACADE_STAGES
        ):
            rec = _wrap_stage(mod, attr, self.sink, stage)
            if rec is not None:
                self._undo.append(rec)
        engine = getattr(self.env.memory, "_engine", None)
        wr = getattr(engine, "wait_ready", None)
        if wr is not None:
            sink = self.sink
            @functools.wraps(wr)
            def _timed_ready(*a: Any, **kw: Any) -> Any:
                t0 = time.perf_counter()
                try:
                    return wr(*a, **kw)
                finally:
                    sink.add("causal_barrier",
                             (time.perf_counter() - t0) * 1000.0)
            self._undo.append((engine, "wait_ready", wr))
            setattr(engine, "wait_ready", _timed_ready)
        return self

    def __exit__(self, *exc: Any) -> None:
        for mod, attr, orig in reversed(self._undo):
            try:
                setattr(mod, attr, orig)
            except Exception:
                pass
        self._undo.clear()


# ---------------------------------------------------------------------------
# SQL statement counting
# ---------------------------------------------------------------------------


class SqlCounter:
    """Counts real statements on the store's reader and writer
    connections via ``set_trace_callback`` for the window it's open."""

    def __init__(self, env: ConsumerEnv) -> None:
        self.env = env
        self.count = 0
        self._saved: List[Tuple[Any, Any]] = []

    def __enter__(self) -> "SqlCounter":
        store = self.env.memory._store
        conns = []
        for getter in ("_reader",):
            try:
                conn = getattr(store, getter)()
                conns.append(conn)
            except Exception:
                pass
        writer = getattr(store, "_writer", None)
        if writer is not None:
            conns.append(writer)
        for conn in conns:
            try:
                prev = conn.set_trace_callback(self._tick)
                self._saved.append((conn, prev))
            except Exception:
                pass
        return self

    def _tick(self, _sql: str) -> None:
        self.count += 1

    def __exit__(self, *exc: Any) -> None:
        for conn, prev in self._saved:
            try:
                conn.set_trace_callback(prev)
            except Exception:
                pass
        self._saved.clear()


# ---------------------------------------------------------------------------
# boundary timings (V5-33.02)
# ---------------------------------------------------------------------------


def measure_boundaries(env: ConsumerEnv, *, repeats: int = 8) -> dict:
    """The §33.02 timing boundaries, each measured at its real edge."""
    mem = env.memory
    out: dict[str, Any] = {}

    # add ack — public add entry → durable acceptance returned
    add_ms: List[float] = []
    receipts: List[Any] = []
    for i in range(repeats):
        t0 = time.perf_counter()
        try:
            receipts.append(mem.add(
                f"boundary probe {i}: locker code LC-{700 + i}"))
        except Exception:
            pass
        add_ms.append((time.perf_counter() - t0) * 1000.0)
    out["add_ack_ms"] = percentiles(add_ms)

    # readiness wait — wait_ready entry → obligations resolved
    wait_ms: List[float] = []
    states: List[str] = []
    for r in receipts:
        t0 = time.perf_counter()
        try:
            rd = mem.wait_ready(r, timeout_ms=3000)
            states.append(rd.state)
        except Exception:
            states.append("error")
        wait_ms.append((time.perf_counter() - t0) * 1000.0)
    out["readiness_wait_ms"] = percentiles(wait_ms)
    out["readiness_states"] = states

    # source-visibility lag — acceptance commit → projection publication.
    # Approximated honestly: settle drains the queue, then we poll for
    # the newest source's visibility via search with explicit barriers.
    settle(env)
    lag_ms: List[float] = []
    for i, r in enumerate(receipts[:4]):
        t0 = time.perf_counter()
        seen = False
        deadline = t0 + 3.0
        while time.perf_counter() < deadline:
            try:
                sr = mem.search(f"LC-{700 + i}", limit=2,
                                consistency="session")
                if any(h.ref == r.ref for h in sr.items):
                    seen = True
                    break
            except Exception:
                pass
            time.sleep(0.005)
        lag_ms.append((time.perf_counter() - t0) * 1000.0)
        if not seen:
            lag_ms[-1] = float("inf")
    out["source_visibility_lag_ms"] = percentiles(
        [x for x in lag_ms if x != float("inf")])
    out["source_visibility_unseen"] = sum(
        1 for x in lag_ms if x == float("inf"))

    # settled search — steady-state reads over existing corpus tasks
    settled: List[float] = []
    qs = [t.query for t in env.corpus.tasks] or ["status"]
    for i in range(max(repeats * 4, 24)):
        t0 = time.perf_counter()
        try:
            mem.search(qs[i % len(qs)], limit=8)
        except Exception:
            pass
        settled.append((time.perf_counter() - t0) * 1000.0)
    out["settled_search_ms"] = percentiles(settled)

    # immediate add→search — add entry → subsequent search completion
    imm: List[float] = []
    for i in range(min(4, repeats)):
        t0 = time.perf_counter()
        try:
            r = mem.add(f"boundary imm {i}: gate fob GF-{900 + i}")
            mem.search(f"GF-{900 + i}", limit=2)
        except Exception:
            pass
        imm.append((time.perf_counter() - t0) * 1000.0)
    out["immediate_add_search_ms"] = percentiles(imm)

    # forget ack — forget entry → durable suppression receipt
    forget_ms: List[float] = []
    for r in receipts[:4]:
        t0 = time.perf_counter()
        try:
            mem.forget(r.ref)
        except Exception:
            pass
        forget_ms.append((time.perf_counter() - t0) * 1000.0)
    out["forget_ack_ms"] = percentiles(forget_ms)
    return out


def run_timers(*, memories: int = 128, seed: int = 42,
               queries: int = 40,
               workdir: Optional[str] = None) -> dict:
    """Profile the consumer route: boundaries + stages + SQL counts."""
    from .corpus import seed_corpus
    from .harness import seed_corpus_env

    corpus = seed_corpus(memories=memories, seed=seed)
    env = seed_corpus_env(corpus, workdir=workdir, worker="external")
    try:
        boundaries = measure_boundaries(env)

        # stage attribution across settled searches
        qs = [t.query for t in corpus.tasks] or ["status"]
        total_ms: List[float] = []
        with StageProfiler(env) as prof:
            for i in range(queries):
                t0 = time.perf_counter()
                try:
                    env.memory.search(qs[i % len(qs)], limit=8)
                except Exception:
                    pass
                total_ms.append((time.perf_counter() - t0) * 1000.0)
            stage_rows = prof.sink.rows()

        attributed = sum(
            (v.get("mean") or 0.0) * (v.get("n") or 0)
            for v in stage_rows.values()
        ) / max(1, queries)
        mean_total = sum(total_ms) / max(1, len(total_ms))
        stage_rows["facade_other"] = {
            "n": len(total_ms),
            "mean": max(0.0, mean_total - attributed),
            "note": "residual: facade glue outside instrumented stages",
        }
        stage_rows["total"] = percentiles(total_ms)

        # SQL counts per settled search
        sql_counts: List[int] = []
        with SqlCounter(env) as counter:
            for i in range(queries):
                before = counter.count
                try:
                    env.memory.search(qs[i % len(qs)], limit=8)
                except Exception:
                    pass
                sql_counts.append(counter.count - before)

        # writer-side metrics the store already measures honestly
        diag = {}
        try:
            diag = env.memory._store.diagnostics()
        except Exception:
            diag = {"unavailable": True}
    finally:
        env.close()

    return {
        "suite": "timers",
        "qualification": "locally_measured",
        "scale": {"memories": memories, "queries": queries},
        "boundaries_ms": boundaries,
        "stages_ms": stage_rows,
        "sql_per_search": percentiles(sql_counts),
        "store_diagnostics": diag,
        "instrumentation": {
            "method": "monkeypatch-wrapped real stage functions + "
                      "sqlite trace callback; no production changes",
            "governed_stages": [s[0] for s in _GOVERNED_STAGES],
            "facade_stages": [s[0] for s in _FACADE_STAGES]
            + ["causal_barrier"],
        },
    }


__all__ = [
    "SqlCounter",
    "StageProfiler",
    "StageSamples",
    "measure_boundaries",
    "run_timers",
]
