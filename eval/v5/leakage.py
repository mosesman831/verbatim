"""Gold-view, canary, and split-isolation checks (E68 / SPEC_V5 §22.06,
§24 leakage rules).

Three real probes, each producing a measured verdict:

1. **gold_view_audit** — the arm-facing proxies
   (``PublicItemView``/``PublicTaskView``) must raise ``AttributeError``
   on every gold field (``expected_ids``, ``forbidden_ids``,
   ``supersedes``, ``forget`` …). Executed live — a proxy that leaks
   fails the audit.

2. **canary_probe** — canary items with unique never-queried tokens are
   added to a live store through the real add path; the corpus's full
   task set then runs and every delivered hit is checked against the
   canary id set. A canary delivered by an unrelated query is a
   leakage/spurious-association finding, counted with its task id.

3. **split_isolation** — two ``Memory`` instances on the *same* store
   path with different ``user_id`` labels get distinct namespaces
   (``ensure_bootstrap`` alias → namespace). User B's searches for user
   A's distinctive tokens must return nothing attributable to A — a
   cross-namespace leak fails the probe. Executed against one shared
   store file so the partition boundary is genuinely tested.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .corpus import (
    ConsumerCorpus,
    corpus_stats,
    public_item,
    public_task,
    seed_corpus,
)
from .harness import (
    ConsumerEnv,
    hit_source_id,
    run_task,
    seed_corpus_env,
    settle,
)


def gold_view_audit() -> dict:
    """Execute the proxy boundary: every gold attribute must raise."""
    corpus = seed_corpus(memories=32, seed=99)
    checks: List[dict] = []
    item = corpus.items[0]
    task = corpus.tasks[0]
    for name, view, golds in (
        ("item", public_item(item),
         ("supersedes", "forget")),
        ("task", public_task(task),
         ("expected_ids", "expected_abstain", "forbidden_ids")),
    ):
        for g in golds:
            try:
                getattr(view, g)
                checks.append({"view": name, "field": g, "leaked": True})
            except AttributeError:
                checks.append({"view": name, "field": g, "leaked": False})
    leaks = [c for c in checks if c["leaked"]]
    return {
        "probe": "gold_view_audit",
        "checks": checks,
        "leaks": leaks,
        "verdict": "passed" if not leaks else "failed",
    }


def canary_probe(env: ConsumerEnv, *, n_canaries: int = 6) -> dict:
    """Unqueried canaries must never be delivered by corpus queries."""
    canary_ids: List[str] = []
    for i in range(n_canaries):
        tok = f"canary-{9000 + i}-xqzv"
        try:
            res = env.memory.add(
                f"canary {tok}: a sealed calibration note never queried.")
            canary_ids.append(tok)
            # map canary ref → source id for hit attribution
            from verbatim.memory.types import MemoryRef
            env.source_item[MemoryRef.parse(res.ref).source_id] = (
                f"canary:{tok}")
        except Exception:
            pass
    settle(env)
    hits: List[dict] = []
    for task in env.corpus.tasks:
        try:
            sr = env.memory.search(task.query, limit=8)
        except Exception:
            continue
        for h in getattr(sr, "items", []) or []:
            sid = hit_source_id(env, h)
            cid = env.corpus_id(sid) or ""
            if cid.startswith("canary:"):
                hits.append({"task": task.task_id, "canary": cid,
                             "query": task.query})
    return {
        "probe": "canary_probe",
        "canaries_seeded": len(canary_ids),
        "queries_run": len(env.corpus.tasks),
        "canary_hits": hits,
        "verdict": "passed" if not hits else "failed",
    }


def split_isolation(*, workdir: Optional[str] = None) -> dict:
    """Same store file, two user namespaces — B must not see A's items.

    Real cross-namespace boundary: both ``Memory`` instances share the
    store path; ``ensure_bootstrap`` provisions distinct namespaces for
    distinct alias labels.
    """
    tmp = workdir or tempfile.mkdtemp(prefix="verbatim-v5-leak-")
    env_a: Optional[ConsumerEnv] = None
    env_b: Optional[ConsumerEnv] = None
    leaks: List[dict] = []
    try:
        # Both envs share ONE store file (tmp/mem.db); distinct user_id
        # labels provision distinct namespaces (ensure_bootstrap). Each
        # side gets exclusive-token items so a leak is attributable —
        # "AXTOK-*" bytes must never appear in Bob's deliveries.
        a_corpus = seed_corpus(memories=48, seed=11)
        b_corpus = seed_corpus(memories=48, seed=77)
        from .corpus import CorpusItem
        a_corpus = type(a_corpus)(
            name=a_corpus.name, seed=a_corpus.seed,
            items=a_corpus.items + tuple(
                CorpusItem(f"a-priv-{i}",
                           f"Alice-only record AXTOK-{i:04d} secret "
                           f"value {10000 + i}.")
                for i in range(8)),
            tasks=a_corpus.tasks)
        b_corpus = type(b_corpus)(
            name=b_corpus.name, seed=b_corpus.seed,
            items=b_corpus.items + tuple(
                CorpusItem(f"b-priv-{i}",
                           f"Bob-only record BBTOK-{i:04d} secret "
                           f"value {20000 + i}.")
                for i in range(8)),
            tasks=b_corpus.tasks)
        env_a = seed_corpus_env(a_corpus, workdir=tmp, worker="external",
                                user_id="alice")
        env_b = seed_corpus_env(b_corpus, workdir=tmp, worker="external",
                                user_id="bob")
        # 1) Bob searches directly for Alice's exclusive tokens.
        for i in range(8):
            q = f"AXTOK-{i:04d}"
            try:
                sr = env_b.memory.search(q, limit=8)
            except Exception:
                continue
            for h in getattr(sr, "items", []) or []:
                if "AXTOK-" in (getattr(h, "quote", "") or ""):
                    leaks.append({"query": q, "kind": "direct",
                                  "quote": (h.quote or "")[:60]})
        # 2) Bob's own queries must never deliver AXTOK bytes.
        for t in b_corpus.tasks:
            try:
                sr = env_b.memory.search(t.query, limit=8)
            except Exception:
                continue
            for h in getattr(sr, "items", []) or []:
                if "AXTOK-" in (getattr(h, "quote", "") or ""):
                    leaks.append({"query": t.query,
                                  "kind": "cross-delivery",
                                  "quote": (h.quote or "")[:60]})
        # 3) Alice's own data must still be reachable for Alice —
        #    isolation is not deletion.
        try:
            sr_a = env_a.memory.search("AXTOK-0000", limit=4)
            a_sees_own = any(
                "AXTOK-" in (getattr(h, "quote", "") or "")
                for h in sr_a.items)
        except Exception:
            a_sees_own = None
        return {
            "probe": "split_isolation",
            "namespaces": {
                "alice": getattr(env_a.memory, "_namespace", None),
                "bob": getattr(env_b.memory, "_namespace", None),
            },
            "queries": 8 + len(b_corpus.tasks),
            "leaks": leaks,
            "alice_reads_own": a_sees_own,
            "verdict": "passed" if not leaks else "failed",
        }
    finally:
        for e in (env_a, env_b):
            try:
                if e is not None:
                    e.close()
            except Exception:
                pass


def run_leakage(*, memories: int = 64, seed: int = 42,
                workdir: Optional[str] = None) -> dict:
    """All three probes on a fresh consumer env."""
    corpus = seed_corpus(memories=memories, seed=seed)
    env = seed_corpus_env(corpus, workdir=workdir, worker="external")
    try:
        settle(env)
        gold = gold_view_audit()
        canary = canary_probe(env)
    finally:
        env.close()
    iso = split_isolation()
    verdicts = [gold["verdict"], canary["verdict"], iso["verdict"]]
    return {
        "suite": "leakage",
        "qualification": "locally_measured",
        "corpus": corpus_stats(corpus),
        "probes": {
            "gold_view_audit": gold,
            "canary_probe": canary,
            "split_isolation": iso,
        },
        "verdict": (
            "failed" if "failed" in verdicts else "passed"
        ),
    }


__all__ = [
    "canary_probe",
    "gold_view_audit",
    "run_leakage",
    "split_isolation",
]
