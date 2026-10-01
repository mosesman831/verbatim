"""V3 job handlers for the procedure pipeline (SPEC_V3 §22, §40).

Registered in ``ingest._V3_KIND_HANDLERS``:

* ``procedure_compile`` — ``input_refs {"episode_id"}`` →
  :func:`compiler.compile_episode`. The durable job is what makes
  compilation a life-cycle operation with a receipt (V3-22.01); the
  commit rides the ingester's generation-fenced ``_commit_effects`` so a
  redelivery replays the recorded result instead of recompiling.
* ``procedure_refine`` — ``input_refs {"procedure_id",
  "contrasting_episode_id"}`` → :func:`compiler.refine_procedure`
  contrastive refinement (V3-22.03).
* ``signature_index`` — ``input_refs {"scope_id"? , "procedure_id"?}`` →
  rebuild ``procedure_signatures`` rows for one procedure or a whole
  scope (V3-21.04).

A job whose episode cannot compile still *succeeds* — the
``CompilationResult`` (``unsupported``/``incomplete`` + reason) is the
recorded outcome; the episode's evidence is untouched (V3-22.13).
"""

from __future__ import annotations

from typing import Any

from ..core.types import ErrorCode, VerbatimError
from .compiler import compile_episode, refine_procedure
from .signatures import index_procedure, index_scope


def _require_ref(refs: dict[str, Any], key: str) -> str:
    value = refs.get(key)
    if not isinstance(value, str) or not value:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"job input_refs.{key} must be a non-empty string",
        )
    return value


def handle_procedure_compile(job: dict, owner: str, ingester) -> None:
    """``procedure_compile``: compile one completed episode (§22)."""
    refs = job.get("input_refs") or {}
    episode_id = _require_ref(refs, "episode_id")
    store = ingester.store
    with store.tx() as conn:
        def _apply(c: Any) -> dict[str, Any]:
            return compile_episode(
                c, episode_id, hmac_fn=ingester.store.hmac
            ).as_dict()

        ingester._commit_effects(conn, job, owner, "procedure_compile", _apply)


def handle_procedure_refine(job: dict, owner: str, ingester) -> None:
    """``procedure_refine``: contrast one same-family episode (V3-22.03)."""
    refs = job.get("input_refs") or {}
    procedure_id = _require_ref(refs, "procedure_id")
    contrasting_id = _require_ref(refs, "contrasting_episode_id")
    store = ingester.store
    with store.tx() as conn:
        def _apply(c: Any) -> dict[str, Any]:
            return refine_procedure(
                c, procedure_id, contrasting_id,
                hmac_fn=ingester.store.hmac,
            )

        ingester._commit_effects(conn, job, owner, "procedure_refine", _apply)


def handle_signature_index(job: dict, owner: str, ingester) -> None:
    """``signature_index``: (re)build signature rows (V3-21.04)."""
    refs = job.get("input_refs") or {}
    procedure_id = refs.get("procedure_id")
    scope_id = refs.get("scope_id") or job.get("scope_id")
    store = ingester.store
    with store.tx() as conn:
        def _apply(c: Any) -> dict[str, Any]:
            if isinstance(procedure_id, str) and procedure_id:
                return {"indexed": index_procedure(c, procedure_id)}
            if not isinstance(scope_id, str) or not scope_id:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    "signature_index needs input_refs.scope_id or a job scope",
                )
            return {"indexed": index_scope(c, scope_id), "scope_id": scope_id}

        ingester._commit_effects(conn, job, owner, "signature_index", _apply)
