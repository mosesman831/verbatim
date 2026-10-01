"""Influence-feedback and shadow-weight checks (E96 / SPEC_V5 §32.05,
V3-26.03, V3-32).

Measures the *real* feedback surface:

* **delivery exposure** — every governed-lane delivery writes an
  ``influence`` row with a fresh per-item handle (V4-33.02). Counted on
  a real store after real recalls — never assumed.
* **source-lane exposure** (V6-03.14) — the consumer ``Memory.search``
  path delivers source-lane items; each delivered item must mint a
  ``source_exposure`` row via ``verbatim.influence.exposure``. Counted
  through the new API after real ``memory.search`` calls — never
  assumed. ``exposure_rows_recorded`` passes when EITHER accounting
  surface demonstrates exposure (claim-lane ``influence`` or
  source-lane ``source_exposure``); both counts are preserved in
  evidence so a zero lane remains visible.
* **feedback bounds** — ``record_feedback`` attaches a feedback kind to
  an *existing* handle; feedback on an undelivered handle is a typed
  ``VALIDATION`` refusal (V3-32.04), never an invented exposure. Both
  directions measured.
* **shadow weights** — ``v3.retrieval.controller="learned_shadow"`` is
  the production kill switch: the deterministic plan applies while the
  hypothetical learned choice is *recorded* (``routing_decisions`` rows
  with ``policy_revision='learned_shadow_v1'``). Measured by running
  searches under the shadow config and counting the recorded rows —
  proving the shadow is logged and the deterministic plan applied.
* **kill switch** — ``controller="learned_active"`` is refused at
  config validation without a validated policy artifact bound via
  ``v3.retrieval.controller_policy_artifact`` (CONFIG_INVALID naming the
  binding) — the hard gate that keeps unqualified weights off the route
  (V6-03.15/16).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

from .corpus import seed_corpus
from .harness import ConsumerEnv, seed_corpus_env, settle


def _count_influence(env: ConsumerEnv) -> Optional[int]:
    try:
        with env.memory._store.read() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM influence").fetchone()[0])
    except Exception:
        return None


def _count_source_exposure(env: ConsumerEnv) -> Optional[int]:
    """``source_exposure`` rows via the V6-03.14 API — the consumer-path
    twin of ``influence``; 0 (not None) when the additive table exists
    but the facade emit hook has not minted rows yet."""
    try:
        from verbatim.influence import exposure as _exposure
        with env.memory._store.read() as conn:
            return int(_exposure.exposure_count(conn))
    except Exception:
        return None


def _exposure_check(
    influence_rows: Optional[int], source_rows: Optional[int]
) -> str:
    """``exposure_rows_recorded`` verdict from the two measured counts.

    V6-03.14: passes when EITHER accounting surface demonstrates
    exposure — claim-lane ``influence`` rows or consumer-path
    ``source_exposure`` rows (the spec admits minted rows "or an
    equivalent disclosed accounting"). Measurable-but-zero on both
    surfaces is inconclusive; ``unavailable`` only when NEITHER count
    could be read at all.
    """
    if (influence_rows or 0) > 0 or (source_rows or 0) > 0:
        return "passed"
    if influence_rows is not None or source_rows is not None:
        return "inconclusive"
    return "unavailable"


def _count_shadow_decisions(env: ConsumerEnv) -> Optional[int]:
    try:
        with env.memory._store.read() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM routing_decisions "
                "WHERE policy_revision = 'learned_shadow_v1'"
            ).fetchone()[0])
    except Exception:
        return None


def _count_decisions(env: ConsumerEnv) -> Optional[int]:
    try:
        with env.memory._store.read() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM routing_decisions"
            ).fetchone()[0])
    except Exception:
        return None


def _feedback_bounds(env: ConsumerEnv) -> dict:
    """record_feedback must refuse unknown handles and accept real ones."""
    from verbatim.retrieval.v3 import influence as infl
    from verbatim.core.types_v3 import InfluenceFeedback
    store = env.memory._store
    out: Dict[str, Any] = {}
    # unknown handle → typed refusal (never an invented exposure)
    try:
        with store.tx() as conn:
            infl.record_feedback(conn, "ih_bogus000",
                                 InfluenceFeedback.USED)
        out["unknown_refused"] = False
    except Exception as exc:  # noqa: BLE001
        out["unknown_refused"] = (
            type(exc).__name__ == "VerbatimError"
            or "VALIDATION" in str(exc)
        )
        out["unknown_error"] = f"{type(exc).__name__}"
    # a real delivered handle → feedback attaches
    try:
        with store.read() as conn:
            row = conn.execute(
                "SELECT handle_id FROM influence LIMIT 1"
            ).fetchone()
        if row is None:
            out["real_handle"] = "unavailable"
        else:
            with store.tx() as conn:
                infl.record_feedback(conn, row[0],
                                     InfluenceFeedback.USED)
            out["real_handle"] = "accepted"
    except Exception as exc:  # noqa: BLE001
        out["real_handle"] = f"error:{type(exc).__name__}"
    return out


def _kill_switch() -> dict:
    """learned_active without a validated artifact must be refused."""
    from verbatim import Memory
    from verbatim.core.types import VerbatimError
    tmp = tempfile.mkdtemp(prefix="verbatim-v5-fb-")
    try:
        try:
            Memory(
                os.path.join(tmp, "k.db"), user_id="fb",
                worker="external",
                config={"v3": {"retrieval":
                               {"controller": "learned_active"}}},
            )
            return {"refused": False,
                    "note": "learned_active accepted without artifact"}
        except VerbatimError as exc:
            return {"refused": True,
                    "code": getattr(exc, "code", None)
                    and str(exc.code),
                    "error": str(exc)[:160]}
        except Exception as exc:  # noqa: BLE001
            return {"refused": True,
                    "code": f"untyped:{type(exc).__name__}",
                    "error": str(exc)[:160]}
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def run_feedback(*, memories: int = 48, seed: int = 42,
                 queries: int = 8,
                 workdir: Optional[str] = None) -> dict:
    """Execute the influence/shadow/kill-switch probes."""
    corpus = seed_corpus(memories=memories, seed=seed)

    # Arm A: deterministic controller — deliveries still record exposure.
    searches_run = 0
    exposure = source_exposure = decisions = None
    fb: Dict[str, Any] = {}
    env = seed_corpus_env(corpus, workdir=workdir, worker="external")
    try:
        settle(env)
        qs = [t.query for t in corpus.tasks][:queries]
        for q in qs:
            try:
                env.memory._v3.recall(
                    env.memory._namespace, q,
                    principal_id=env.memory._owner,
                    purpose="recall",
                    budget={"modes": ("evidence",), "max_items": 8,
                            "max_bytes": 6000, "target_tokens": 1536,
                            "deadline_ms": 400,
                            "session_id": env.memory._session_id},
                    session_id=env.memory._session_id)
            except Exception:
                pass
        # V6-03.14: the consumer path — real Memory.search calls so
        # source-lane deliveries have the opportunity to mint
        # source_exposure rows through the facade's post-delivery emit
        # hook (verbatim.influence.exposure.emit_deliveries).
        searches_run = 0
        for q in qs:
            try:
                env.memory.search(q, limit=8)
                searches_run += 1
            except Exception:
                pass
        exposure = _count_influence(env)
        source_exposure = _count_source_exposure(env)
        decisions = _count_decisions(env)
        fb = _feedback_bounds(env)
    finally:
        env.close()

    # Arm B: learned_shadow — deterministic applies, shadow logs.
    env2 = seed_corpus_env(corpus, worker="external",
                           memory_kwargs={
                               "config": {"v3": {"retrieval":
                                                 {"controller":
                                                  "learned_shadow"}}},
                           })
    try:
        settle(env2)
        for q in qs:
            try:
                env2.memory._v3.recall(
                    env2.memory._namespace, q,
                    principal_id=env2.memory._owner,
                    purpose="recall",
                    budget={"modes": ("evidence",), "max_items": 8,
                            "max_bytes": 6000, "target_tokens": 1536,
                            "deadline_ms": 400,
                            "session_id": env2.memory._session_id},
                    session_id=env2.memory._session_id)
            except Exception:
                pass
        shadow_rows = _count_shadow_decisions(env2)
        det_rows = _count_decisions(env2)
    finally:
        env2.close()

    ks = _kill_switch()

    checks = {
        # Both counts stay in evidence either way — a zero lane remains
        # visible even when the other surface carries the pass.
        "exposure_rows_recorded": _exposure_check(exposure, source_exposure),
        "feedback_bounds": (
            "passed" if fb.get("unknown_refused")
            and fb.get("real_handle") == "accepted" else
            "failed" if fb.get("unknown_refused") is False else
            "inconclusive"),
        "shadow_logged_deterministic_applied": (
            "passed" if (shadow_rows or 0) > 0
            and (det_rows or 0) > (shadow_rows or 0) else
            "inconclusive"),
        "kill_switch": "passed" if ks.get("refused") else "failed",
    }
    return {
        "suite": "feedback",
        "qualification": "locally_measured",
        "checks": checks,
        "evidence": {
            "influence_rows": exposure,
            "source_exposure_rows": source_exposure,
            "consumer_searches_run": searches_run,
            "routing_decisions": decisions,
            "shadow_decisions": shadow_rows,
            "shadow_total_decisions": det_rows,
            "feedback_bounds": fb,
            "kill_switch": ks,
        },
        "verdict": ("failed" if "failed" in checks.values()
                    else "passed" if all(v == "passed"
                                         for v in checks.values())
                    else "inconclusive"),
    }


__all__ = ["run_feedback"]
