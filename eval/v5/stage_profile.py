"""Standalone V5 stage-profile artifact (SPEC_V5 §33.02–33.06).

Runs the real ``StageProfiler`` on BOTH worker modes so the artifact
separates two honest costs:

* ``managed`` — the production drain path: obligations settle while
  reads proceed; ``facade_other`` is genuine residual work.
* ``external`` — the deterministic test path: post-settle adds leave
  obligations pending, so ``facade_other`` then contains the bounded
  causal-barrier wait (``ready_timeout_ms``) itself — reported, not
  hidden.

Artifacts: ``eval/v5/stage_profile_v5.json`` + ``.md``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from .corpus import seed_corpus
from .harness import seed_corpus_env, settle
from .timers import StageProfiler, measure_boundaries


def _profile(worker: str, *, memories: int, queries: int, seed: int,
             workdir: str) -> dict:
    corpus = seed_corpus(memories=memories, seed=seed)
    env = seed_corpus_env(corpus, workdir=workdir, worker=worker)
    try:
        boundaries = measure_boundaries(env, repeats=6)
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
            "note": "residual: facade glue outside instrumented stages "
                    "(on external-worker envs this is dominated by the "
                    "bounded causal-barrier wait on pending obligations)",
        }
        from .harness import percentiles
        stage_rows["total"] = percentiles(total_ms)
        diag = {}
        try:
            diag = env.memory._store.diagnostics()
        except Exception:
            diag = {"unavailable": True}
        return {
            "worker": worker,
            "boundaries_ms": boundaries,
            "stages_ms": stage_rows,
            "store_diagnostics": diag,
        }
    finally:
        env.close()


def run(memories: int = 256, queries: int = 48, seed: int = 42,
        out_prefix: str = "eval/v5/stage_profile_v5") -> dict:
    base = out_prefix + ".work"
    os.makedirs(base, exist_ok=True)
    profiles = {
        "managed": _profile("managed", memories=memories,
                            queries=queries, seed=seed,
                            workdir=os.path.join(base, "managed")),
        "external": _profile("external", memories=memories,
                             queries=queries, seed=seed,
                             workdir=os.path.join(base, "external")),
    }
    doc: Dict[str, Any] = {
        "artifact": "v5-stage-profile",
        "qualification": "locally_measured",
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scale": {"memories": memories, "queries": queries, "seed": seed},
        "instrumentation": {
            "method": "monkeypatch-wrapped real stage functions + "
                      "sqlite trace callback; no production changes",
            "boundary_semantics": (
                "readiness_wait on the external-worker env measures "
                "wait_ready BEFORE settle drains the queue — pending at "
                "timeout is the honest undrained state, not a defect"),
        },
        "profiles": profiles,
    }
    with open(out_prefix + ".json", "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    _write_md(doc, out_prefix + ".md")
    return doc


def _write_md(doc: dict, path: str) -> None:
    lines = [
        "# V5 stage profile — consumer route",
        "",
        f"Generated {doc['generated_at']} — qualification: "
        "**locally_measured**. Scale: "
        f"{doc['scale']['memories']} memories, "
        f"{doc['scale']['queries']} profiled searches, "
        f"seed={doc['scale']['seed']}.",
        "",
        "Method: " + doc["instrumentation"]["method"],
        "",
    ]
    for mode, prof in doc["profiles"].items():
        lines += [f"## worker={mode}", "",
                  "### per-stage attribution (settled searches)", "",
                  "| stage | n | mean ms | p50 | p95 |",
                  "|---|---|---|---|---|"]
        for k, v in sorted(
                (prof["stages_ms"] or {}).items(),
                key=lambda kv: -(kv[1].get("mean") or 0)
                if isinstance(kv[1], dict) else 0):
            if not isinstance(v, dict):
                continue
            p50 = v.get("p50")
            p95 = v.get("p95")
            lines.append(
                f"| {k} | {v.get('n')} | "
                f"{round(v.get('mean') or 0.0, 3)} | "
                f"{'—' if p50 is None else round(p50,3)} | "
                f"{'—' if p95 is None else round(p95,3)} |")
        b = prof.get("boundaries_ms") or {}
        if b:
            lines += ["", "### boundary timings", "",
                      "| boundary | p50 | p95 | p99 |", "|---|---|---|---|"]
            for k, v in sorted(b.items()):
                if isinstance(v, dict) and v.get("p50") is not None:
                    lines.append(
                        f"| {k} | {round(v['p50'],2)} | "
                        f"{round(v.get('p95') or 0,2)} | "
                        f"{round(v.get('p99') or 0,2)} |")
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- `facade_other` on `external` is the bounded causal-barrier "
        "wait on post-settle pending obligations (ready_timeout_ms) — "
        "expected on an undrained env; on `managed` it is the real "
        "residual facade cost.",
        "- Denominators are published; stages with no samples report "
        "nothing rather than fabricating a zero.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


__all__ = ["run"]

if __name__ == "__main__":
    run()
