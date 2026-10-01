"""``source_state/v1`` control artifact (SPEC_V5 §14.3).

The registered source-lifecycle record: CAS-anchored mutation head,
typed transitions (``supersede``/``correct``/``retract`` …), bitemporal
validity (``at_time`` valid-time + ``known_at`` record-time evaluation),
erasure tombstones, and inspectable history — all on the existing
object/revision/coordinator authority. See ``state.py`` for the storage
model and the V5-03.06 handler registration map.

Writers take the caller's transaction ``conn`` (open ``store.tx()``);
reads work on any connection. ``install_schema`` is the idempotent
migration handler for the frozen ``source_state`` table.
"""

from .state import (
    CHANGE_ALIASES,
    DDL_SOURCE_STATE,
    DISPOSITIONS,
    EFFECT_KIND,
    HEAD_UNRESOLVED,
    OBJECT_KIND,
    PRODUCER_ID,
    SOURCE_STATE_KIND,
    TABLE,
    TRANSITION_DISPOSITIONS,
    CurrentState,
    SourceState,
    adopt,
    artifact_id,
    current_state,
    doc_digest,
    ensure_state,
    get,
    get_state,
    history,
    install_schema,
    parse_doc,
    parse_head,
    recover,
    register_producer,
    serialize_doc,
    verify,
)
from .transitions import (
    apply_erasure,
    assert_publishable,
    is_revision_current,
    transition,
)

__all__ = [
    "CHANGE_ALIASES",
    "DDL_SOURCE_STATE",
    "DISPOSITIONS",
    "EFFECT_KIND",
    "HEAD_UNRESOLVED",
    "OBJECT_KIND",
    "PRODUCER_ID",
    "SOURCE_STATE_KIND",
    "TABLE",
    "TRANSITION_DISPOSITIONS",
    "CurrentState",
    "SourceState",
    "adopt",
    "apply_erasure",
    "artifact_id",
    "assert_publishable",
    "current_state",
    "doc_digest",
    "ensure_state",
    "get",
    "get_state",
    "history",
    "install_schema",
    "is_revision_current",
    "parse_doc",
    "parse_head",
    "recover",
    "register_producer",
    "serialize_doc",
    "transition",
    "verify",
]
