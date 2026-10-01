"""Deduplication by linking (SPEC_V5 §30.2; contracts §9).

Links, never merges: every duplicate keeps its bytes, attribution, and
receipt. See ``links.py`` for the contract notes on hard guards
(V5-30.07), the per-namespace ``dedupe`` policy (V5-30.08), pack-time
collapse (V5-30.09), and deletion-closure semantics.
"""

from .links import (
    DEFAULT_SCAN_LIMIT,
    DEFAULT_THRESHOLD,
    LINK_METHODS,
    METHOD_EXACT,
    METHOD_MINHASH,
    METHOD_NORMALIZED,
    POLICY_LINK,
    POLICY_NONE,
    DedupFields,
    LinkOutcome,
    collapse_for_hit,
    corroboration_count,
    drop_member,
    fields_from_enrichment,
    fields_from_text,
    get_dedupe_policy,
    group_members,
    guard_veto,
    link_exact,
    link_near,
    member_refs,
    representative,
    set_dedupe_policy,
)

__all__ = [
    "DEFAULT_SCAN_LIMIT",
    "DEFAULT_THRESHOLD",
    "LINK_METHODS",
    "METHOD_EXACT",
    "METHOD_MINHASH",
    "METHOD_NORMALIZED",
    "POLICY_LINK",
    "POLICY_NONE",
    "DedupFields",
    "LinkOutcome",
    "collapse_for_hit",
    "corroboration_count",
    "drop_member",
    "fields_from_enrichment",
    "fields_from_text",
    "get_dedupe_policy",
    "group_members",
    "guard_veto",
    "link_exact",
    "link_near",
    "member_refs",
    "representative",
    "set_dedupe_policy",
]
