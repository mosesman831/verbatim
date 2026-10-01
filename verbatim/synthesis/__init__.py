"""Opt-in grounded synthesis and derived views (SPEC_V4 §20).

Deterministic grounded composition only — no neural producer exists in
this environment. Public surface:

- ``Synthesizer(store).register_producer()`` — per-producer opt-in
  (``producer_manifests`` row).
- ``Synthesizer.compose(scope_id, query|topic, caller=..., purpose=...,
  grant_check=...)`` — build + persist a ``DerivedView`` whose statements
  each carry ``view_support`` bindings to the exact evidence refs that
  ground them.
- ``Synthesizer.deliver(view_id, caller=...)`` — re-authorized,
  freshness-checked delivery; quote text materializes only under the
  caller's ``quote`` verb.
- ``Synthesizer.note_invalidation(conn, report)`` — eager suppression of
  views a ``kernel.invalidate`` report touches; ``deliver`` enforces the
  same suppression lazily regardless.
- ``Synthesizer.erase(view_id, caller=...)`` / ``.status(...)`` /
  ``.capabilities()``.
"""

from .synthesizer import DEFAULT_PRODUCER_ID, RUBRIC_REVISION, Synthesizer
from .types import (
    Confidence,
    DerivedView,
    StatementForm,
    SupportBinding,
    SupportVerdict,
    SYNTHESIS_MODE,
    SUPPORTED_VIEW_KINDS,
    VIEW_OBJECT_KIND,
    VIEW_SUPPORT_KIND,
    ViewKind,
    ViewStatement,
)

__all__ = [
    "Confidence",
    "DEFAULT_PRODUCER_ID",
    "DerivedView",
    "RUBRIC_REVISION",
    "StatementForm",
    "SupportBinding",
    "SupportVerdict",
    "Synthesizer",
    "SYNTHESIS_MODE",
    "SUPPORTED_VIEW_KINDS",
    "VIEW_OBJECT_KIND",
    "VIEW_SUPPORT_KIND",
    "ViewKind",
    "ViewStatement",
]
