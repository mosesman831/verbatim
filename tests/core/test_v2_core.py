"""v2 evidence-semantics core tests (SPEC_V2).

Covers the owned-core v2 surface against the real storage layer:

1. ``persist_harvest`` — spans + context groups + members, atomic on the
   caller's connection, deterministic replay under the same operation key.
2. Unstructured admission — verbatim evidence preserved, interpretation
   status stamped, slot-conflict detection skipped.
3. Predicate registry — idempotent ``vb`` seeding and admission-time
   consultation.
4. Bitemporal interval endpoints — uncertain/unknown kinds persisted and
   carried through lifecycle transitions.
5. Negation-aware conflicts — ``conflicts_with`` edge + conflict-group
   membership, mutable→supersede vs immutable→dispute proposals.
6. ``apply_review`` — atomic transition + fence + review update + in-tx FTS
   indexing; stale expected versions fail closed leaving the review open.
7. Evidence families — duplicated span/payload lineage registered without
   merging the distinct evidence records.
"""

from __future__ import annotations

import dataclasses

import pytest

from verbatim.config import (
    AdmissionConfig,
    CaptureConfig,
    VerbatimConfig,
)
from verbatim.core.claims import (
    BUILTIN_PREDICATES,
    BUILTIN_REGISTRY_NAMESPACE,
    BUILTIN_REGISTRY_VERSION,
    propose,
)
from verbatim.core.harvest import (
    HARVESTER_VERSION,
    harvest_source,
    persist_harvest,
)
from verbatim.core.lifecycle import (
    LifecycleMachine,
    read_claim_head,
    read_intervals,
)
from verbatim.core.policy import (
    PolicyContext,
    admit,
    apply_review,
    relate,
)
from verbatim.core.types import (
    ClaimProposal,
    EndpointKind,
    ErrorCode,
    Lifecycle,
    Polarity,
    Precision,
    Provenance,
    SourceKind,
    TimeInterval,
    TransitionCommand,
    VerbatimError,
    safe_json_loads,
)

from tests.core.conftest import (
    make_envelope,
    insert_source,
    q,
    seed_claim,
    seed_span_for,
)


def _cfg(**kw):
    capture = kw.pop("capture", CaptureConfig(enabled=True, user_messages=True))
    admission = kw.pop("admission", AdmissionConfig(require_review=True))
    return VerbatimConfig(capture=capture, admission=admission, **kw)


def _rules_ctx() -> PolicyContext:
    return PolicyContext(
        cfg=_cfg(admission=AdmissionConfig(require_review=False))
    )


def _admit_text(store, scope, text, ctx=None, **env_kw):
    """Harvest-free admit helper: one span covering the whole payload."""
    env = make_envelope(scope, text, **env_kw)
    span = seed_span_for(store, env)
    prop = propose(text, span, env, env.event_us)
    return admit(store, prop, env, ctx or _rules_ctx()), env, span


# ---------------------------------------------------------------------------
# 1. persist_harvest — context-group persistence
# ---------------------------------------------------------------------------


def test_persist_harvest_writes_spans_group_and_members(store, scope):
    text = "I don't use vim. I switched to Helix last week."
    env = make_envelope(scope, text)
    source_id = insert_source(store, env)
    env = dataclasses.replace(env, source_id=source_id)
    result = harvest_source(env)
    assert result.candidates, "expected harvest candidates"

    key = f"harvest:{source_id}:1:{HARVESTER_VERSION}"
    with store.tx() as conn:
        ph = persist_harvest(store, conn, env, result, key)

    assert ph.group_id
    assert ph.created is True
    assert len(ph.spans) == len(result.candidates)

    # Spans persist exact source-relative byte offsets (SPEC_V2 §12.02).
    span_rows = q(
        store,
        "SELECT start_byte, end_byte, operation_key FROM spans"
        " WHERE source_id = ? ORDER BY start_byte",
        (source_id,),
    )
    assert len(span_rows) == len(result.candidates)
    assert {r[2] for r in span_rows} == {key}
    for cand, (s0, e0, _k) in zip(
        sorted(result.candidates, key=lambda c: c.start_byte), span_rows
    ):
        assert (s0, e0) == (cand.start_byte, cand.end_byte)
        assert env.payload[s0:e0]  # offsets slice real bytes

    group = q(
        store,
        "SELECT scope_id, source_id, revision, parser_version, completeness"
        " FROM context_groups WHERE group_id = ?",
        (ph.group_id,),
    )[0]
    assert group[1] == source_id and group[2] == 1
    assert group[3] == HARVESTER_VERSION

    members = q(
        store,
        "SELECT span_id, role, required, ord FROM context_members"
        " WHERE group_id = ? ORDER BY ord, role",
        (ph.group_id,),
    )
    primaries = [m for m in members if m[1] == "primary"]
    assert len(primaries) == len(result.candidates)
    assert all(m[2] == 1 for m in primaries)
    # The negated candidate carries a 'negation' context role.
    assert any(m[1] == "negation" for m in members)
    # "last week" contributes a temporal facet.
    assert any(m[1] == "temporal" for m in members)


def test_persist_harvest_idempotent_replay(store, scope):
    env = make_envelope(scope, "I use neovim. I prefer dark themes.")
    source_id = insert_source(store, env)
    env = dataclasses.replace(env, source_id=source_id)
    result = harvest_source(env)
    key = f"harvest:{source_id}:1:{HARVESTER_VERSION}"

    with store.tx() as conn:
        first = persist_harvest(store, conn, env, result, key)
    with store.tx() as conn:
        second = persist_harvest(store, conn, env, result, key)

    assert second.group_id == first.group_id
    assert second.created is False
    assert [s.span_id for s in second.spans] == [s.span_id for s in first.spans]
    assert q(store, "SELECT COUNT(*) FROM context_groups")[0][0] == 1
    assert q(
        store, "SELECT COUNT(*) FROM spans WHERE source_id = ?", (source_id,)
    )[0][0] == len(result.candidates)


def test_persist_harvest_overflow_marks_partial(store, scope):
    text = " ".join(f"Sentence number {i} is here." for i in range(6))
    env = make_envelope(scope, text)
    source_id = insert_source(store, env)
    env = dataclasses.replace(env, source_id=source_id)
    result = harvest_source(env, max_candidates=2)
    assert result.overflow_count > 0

    with store.tx() as conn:
        ph = persist_harvest(
            store, conn, env, result, f"harvest:{source_id}:1:{HARVESTER_VERSION}"
        )
    assert ph.completeness == "partial"
    assert ph.overflow_count == result.overflow_count
    row = q(
        store,
        "SELECT completeness FROM context_groups WHERE group_id = ?",
        (ph.group_id,),
    )[0]
    assert row[0] == "partial"


def test_persist_harvest_requires_source_id(store, scope):
    env = make_envelope(scope, "I use neovim.")
    result = harvest_source(env)
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            persist_harvest(store, conn, env, result, "k1")
    assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# 2. Unstructured admission lane
# ---------------------------------------------------------------------------


def test_unstructured_admission_preserves_evidence(store, scope):
    out, env, span = _admit_text(
        store, scope, "The quarterly club meeting moved to Thursdays."
    )
    assert out.claim_id is not None
    # No typed predicate matched → pending + review under the rules profile.
    assert out.state == Lifecycle.PENDING
    assert out.reason == "abstained"
    assert out.review_id is not None

    claim = q(
        store,
        "SELECT predicate, interpretation_status FROM claims WHERE claim_id = ?",
        (out.claim_id,),
    )[0]
    assert claim == (None, "unstructured")

    rev = q(
        store,
        "SELECT predicate, interpretation_status, method FROM claim_revisions"
        " WHERE claim_id = ?",
        (out.claim_id,),
    )[0]
    assert rev[0] is None and rev[1] == "unstructured" and rev[2]

    # Verbatim evidence intact: the span still quotes exact source bytes.
    ev = q(
        store,
        "SELECT span_id, evidence_role FROM claim_evidence WHERE claim_id = ?",
        (out.claim_id,),
    )
    assert ev == [(span.span_id, "primary")]
    srow = q(
        store,
        "SELECT start_byte, end_byte FROM spans WHERE span_id = ?",
        (span.span_id,),
    )[0]
    assert env.payload[srow[0] : srow[1]].decode("utf-8") == env.payload.decode(
        "utf-8"
    )


def test_structured_admission_status(store, scope):
    out, _env, _span = _admit_text(store, scope, "I use Neovim.")
    assert out.state == Lifecycle.ACTIVE
    row = q(
        store,
        "SELECT predicate, interpretation_status FROM claims WHERE claim_id = ?",
        (out.claim_id,),
    )[0]
    assert row == ("editor", "structured")
    rev = q(
        store,
        "SELECT interpretation_status FROM claim_revisions WHERE claim_id = ?",
        (out.claim_id,),
    )[0]
    assert rev[0] == "structured"


def test_explicit_remember_stays_unstructured(store, scope):
    out, _env, _span = _admit_text(store, scope, "Remember that I like tea.")
    # Explicit remember activates immediately but never invents structure.
    assert out.state == Lifecycle.ACTIVE
    assert out.reason == "explicit_remember"
    row = q(
        store,
        "SELECT predicate, interpretation_status FROM claims WHERE claim_id = ?",
        (out.claim_id,),
    )[0]
    assert row == (None, "unstructured")


def test_unstructured_skips_slot_conflict_detection(store, scope):
    _admit_text(store, scope, "I use Neovim.")
    out, _env, _span = _admit_text(
        store, scope, "The quarterly club meeting moved to Thursdays."
    )
    # An unstructured claim must produce no edges/groups/reviews via relate().
    assert relate(store, out.claim_id, _rules_ctx()) == []
    assert q(store, "SELECT COUNT(*) FROM edges")[0][0] == 0
    assert q(store, "SELECT COUNT(*) FROM conflict_groups")[0][0] == 0


# ---------------------------------------------------------------------------
# 3. Predicate registry
# ---------------------------------------------------------------------------


def test_builtin_registry_seeded_on_first_admit(store, scope):
    _admit_text(store, scope, "I use Neovim.")
    rows = q(
        store,
        "SELECT name, cardinality, mutable, sensitive, authority_policy"
        " FROM predicate_definitions WHERE namespace = ? ORDER BY name",
        (BUILTIN_REGISTRY_NAMESPACE,),
    )
    assert {r[0] for r in rows} >= {name for name, *_ in BUILTIN_PREDICATES}
    by_name = {r[0]: r for r in rows}
    assert by_name["editor"][1:4] == ("single", 1, 0)
    assert by_name["residence"][1:4] == ("single", 1, 1)
    assert by_name["preference"][1:4] == ("set", 1, 0)
    assert all(r[4] == "builtin" for r in rows)


def test_registry_seed_is_idempotent(store, scope):
    _admit_text(store, scope, "I use Neovim.")
    _admit_text(store, scope, "I prefer dark themes.")
    n = q(store, "SELECT COUNT(*) FROM predicate_definitions")[0][0]
    assert n == len({name for name, *_ in BUILTIN_PREDICATES})


def test_registry_consulted_for_slot_behavior(store, scope):
    _admit_text(store, scope, "I use Neovim.")  # seeds the registry
    # An operator tightening 'editor' to sensitive must affect the NEXT admit.
    with store.tx() as conn:
        conn.execute(
            "UPDATE predicate_definitions SET sensitive = 1"
            " WHERE namespace = ? AND name = 'editor'",
            (BUILTIN_REGISTRY_NAMESPACE,),
        )
    out, _env, _span = _admit_text(store, scope, "I use Emacs.")
    assert out.reason == "sensitive_hint"
    assert out.state == Lifecycle.PENDING
    # restore
    with store.tx() as conn:
        conn.execute(
            "UPDATE predicate_definitions SET sensitive = 0"
            " WHERE namespace = ? AND name = 'editor'",
            (BUILTIN_REGISTRY_NAMESPACE,),
        )


def test_registry_version_stamped_on_revision(store, scope):
    out, _env, _span = _admit_text(store, scope, "I use Neovim.")
    row = q(
        store,
        "SELECT registry_version FROM claim_revisions WHERE claim_id = ?",
        (out.claim_id,),
    )[0]
    assert row[0] == BUILTIN_REGISTRY_VERSION


def test_unknown_predicate_not_coerced(store, scope):
    # A predicate outside the vb table stays unslotted (no nearby-slot grab).
    env = make_envelope(scope, "I use Neovim.")
    span = seed_span_for(store, env)
    prop = ClaimProposal(
        evidence=(span,),
        predicate="martian_dialect",
        object_json={"kind": "literal", "text": "x"},
        subject_entity_id=None,
        polarity=Polarity.AFFIRMATIVE,
        valid=TimeInterval(),
        method="test",
        confidence_kind="rule",
    )
    out = admit(store, prop, env, _rules_ctx())
    assert out.claim_id is not None
    # Unknown → no slot → cannot take the rule_whitelist path.
    assert out.reason == "abstained"
    assert out.state == Lifecycle.PENDING


# ---------------------------------------------------------------------------
# 4. Bitemporal interval endpoints
# ---------------------------------------------------------------------------


def test_uncertain_interval_endpoints_persisted(store, scope):
    env = make_envelope(scope, "I use Neovim.")
    span = seed_span_for(store, env)
    iv = TimeInterval(
        from_us=1_000,
        until_us=None,
        precision=Precision.DAY,
        basis="test",
        start_kind=EndpointKind.UNCERTAIN_RANGE,
        end_kind=EndpointKind.UNKNOWN,
        from_us_hi=9_000,
    )
    prop = ClaimProposal(
        evidence=(span,),
        predicate="editor",
        object_json={"kind": "literal", "text": "neovim"},
        subject_entity_id=None,
        polarity=Polarity.AFFIRMATIVE,
        valid=iv,
        method="test",
        confidence_kind="rule",
    )
    out = admit(store, prop, env, _rules_ctx())
    row = q(
        store,
        "SELECT from_us, until_us, start_kind, end_kind, from_us_hi, until_us_hi"
        " FROM valid_intervals WHERE claim_id = ?",
        (out.claim_id,),
    )[0]
    assert row == (1_000, None, "uncertain_range", "unknown", 9_000, None)
    with store.tx() as conn:
        head = read_claim_head(conn, out.claim_id)
        ivs = read_intervals(conn, out.claim_id, head.revision)
    assert ivs[0].start_kind == EndpointKind.UNCERTAIN_RANGE
    assert ivs[0].end_kind == EndpointKind.UNKNOWN
    assert ivs[0].from_us_hi == 9_000


def test_transition_preserves_interval_endpoints(store, scope):
    env = make_envelope(scope, "I use Neovim.")
    span = seed_span_for(store, env)
    iv = TimeInterval(
        from_us=1_000,
        precision=Precision.DAY,
        basis="test",
        start_kind=EndpointKind.UNCERTAIN_RANGE,
        end_kind=EndpointKind.UNBOUNDED,
        from_us_hi=5_000,
    )
    prop = ClaimProposal(
        evidence=(span,),
        predicate="editor",
        object_json={"kind": "literal", "text": "neovim"},
        subject_entity_id=None,
        polarity=Polarity.AFFIRMATIVE,
        valid=iv,
        method="test",
        confidence_kind="rule",
    )
    ctx = PolicyContext(cfg=_cfg(admission=AdmissionConfig(require_review=True)))
    out = admit(store, prop, env, ctx)
    assert out.state == Lifecycle.PENDING
    apply_review(store, out.review_id, "operator", ctx)
    with store.tx() as conn:
        head = read_claim_head(conn, out.claim_id)
        assert head.state == Lifecycle.ACTIVE
        assert head.interpretation_status == "structured"
        ivs = read_intervals(conn, out.claim_id, head.revision)
    assert ivs[0].start_kind == EndpointKind.UNCERTAIN_RANGE
    assert ivs[0].end_kind == EndpointKind.UNBOUNDED
    assert ivs[0].from_us_hi == 5_000
    # No fabricated bounds: unknown/unbounded ends stay NULL.
    assert ivs[0].until_us is None


# ---------------------------------------------------------------------------
# 5. Negation-aware conflicts + conflict groups
# ---------------------------------------------------------------------------


def test_negated_preference_polarity_extraction(store, scope):
    # Surface "don't like" and inherently negative verbs share ONE polarity —
    # required so the pair classifier sees them as one proposition.
    for text in ("I don't like cilantro.", "I hate cilantro."):
        env = make_envelope(scope, text)
        span = seed_span_for(store, env)
        prop = propose(text, span, env, env.event_us)
        assert prop.predicate == "preference"
        assert prop.polarity == Polarity.NEGATED
    env = make_envelope(scope, "I like cilantro.")
    span = seed_span_for(store, env)
    prop = propose("I like cilantro.", span, env, env.event_us)
    assert prop.polarity == Polarity.AFFIRMATIVE


def test_natural_editor_retraction_creates_supersession_review(store, scope):
    ctx = _rules_ctx()
    old, _env, _span = _admit_text(
        store, scope, "I switched from VS Code to Neovim", ctx=ctx
    )
    new, _env, _span = _admit_text(
        store, scope,
        "Actually scratch that about Neovim, I went back to VS Code",
        ctx=ctx,
    )
    artifacts = relate(store, new.claim_id, ctx=ctx)
    assert artifacts
    edge = q(
        store,
        "SELECT edge_type FROM edges WHERE source_id IN (?, ?)"
        " AND target_id IN (?, ?)",
        (old.claim_id, new.claim_id, old.claim_id, new.claim_id),
    )
    assert edge == [("conflicts_with",)]
    effects = [
        safe_json_loads(row[0])
        for row in q(
            store,
            "SELECT proposed_effect_json FROM reviews"
            " WHERE proposed_effect_json LIKE '%supersede%'",
        )
    ]
    assert any(
        effect.get("predecessor_id") == old.claim_id
        and effect.get("successor_id") == new.claim_id
        for effect in effects
    )


def test_negation_conflict_creates_edge_group_and_supersede_review(
    store, scope
):
    old, _ = seed_claim(
        store,
        scope,
        predicate="preference",
        object_text="cilantro",
        polarity="affirmative",
        span_text="I like cilantro.",
    )
    new, _ = seed_claim(
        store,
        scope,
        predicate="preference",
        object_text="cilantro",
        polarity="negated",
        span_text="I don't like cilantro.",
    )
    out = relate(store, new, _rules_ctx())
    assert len(out) == 2  # edge + review

    edge = q(
        store,
        "SELECT source_id, target_id FROM edges WHERE edge_type='conflicts_with'",
    )[0]
    assert {edge[0], edge[1]} == {old, new}

    groups = q(store, "SELECT group_id, status FROM conflict_groups")
    assert len(groups) == 1 and groups[0][1] == "open"
    members = q(
        store,
        "SELECT claim_id FROM conflict_members WHERE group_id = ?",
        (groups[0][0],),
    )
    assert {m[0] for m in members} == {old, new}

    review = q(store, "SELECT proposed_effect_json FROM reviews")[0]
    effect = safe_json_loads(review[0])
    # 'preference' is mutable → supersession is *proposed*, never applied.
    assert effect["effect"] == "supersede"
    assert effect["reason"] == "negation_conflict"
    with store.tx() as conn:
        assert read_claim_head(conn, old).state == Lifecycle.ACTIVE
        assert read_claim_head(conn, new).state == Lifecycle.ACTIVE


def test_immutable_conflict_proposes_dispute_not_supersede(store, scope):
    old, _ = seed_claim(
        store,
        scope,
        predicate="language",
        object_text="spanish",
        polarity="affirmative",
        span_text="I speak spanish.",
    )
    new, _ = seed_claim(
        store,
        scope,
        predicate="language",
        object_text="spanish",
        polarity="negated",
        span_text="I don't speak spanish.",
    )
    out = relate(store, new, _rules_ctx())
    assert len(out) == 2
    review = q(store, "SELECT proposed_effect_json FROM reviews")[0]
    effect = safe_json_loads(review[0])
    # 'language' is immutable → disputed, never a supersession proposal.
    assert effect["effect"] == "dispute"
    assert effect["claim_id"] == old
    members = q(store, "SELECT claim_id FROM conflict_members")
    assert {m[0] for m in members} == {old, new}


def test_conflict_groups_merge_on_shared_claim(store, scope):
    a, _ = seed_claim(
        store,
        scope,
        predicate="preference",
        object_text="cilantro",
        polarity="affirmative",
        span_text="I like cilantro.",
    )
    b, _ = seed_claim(
        store,
        scope,
        predicate="preference",
        object_text="cilantro",
        polarity="negated",
        span_text="I don't like cilantro.",
    )
    c, _ = seed_claim(
        store,
        scope,
        predicate="preference",
        object_text="cilantro",
        polarity="negated",
        span_text="I hate cilantro.",
    )
    relate(store, b, _rules_ctx())
    relate(store, c, _rules_ctx())
    # All three claims share ONE open conflict group.
    groups = q(store, "SELECT group_id FROM conflict_groups")
    assert len(groups) == 1
    members = q(
        store,
        "SELECT claim_id FROM conflict_members WHERE group_id = ?",
        (groups[0][0],),
    )
    assert {m[0] for m in members} == {a, b, c}


# ---------------------------------------------------------------------------
# 6. Atomic review application + stale fencing
# ---------------------------------------------------------------------------


def test_apply_review_admits_pending_claim_atomically(store, scope):
    ctx = PolicyContext(cfg=_cfg())  # require_review default
    out, _env, _span = _admit_text(store, scope, "I use Neovim.", ctx=ctx)
    assert out.state == Lifecycle.PENDING and out.review_id
    gen_before = store.projection_generation()

    seq = apply_review(store, out.review_id, "operator", ctx)
    assert seq > 0
    with store.tx() as conn:
        head = read_claim_head(conn, out.claim_id)
        assert head.state == Lifecycle.ACTIVE
    review = q(store, "SELECT state, resolved_event FROM reviews")[0]
    assert review == ("approved", seq)
    # V2-19.10: an ordinary transition must NOT bump the global projection
    # generation (that would strand every other indexed claim); the newly
    # active claim is indexed at the current generation in-tx instead.
    assert store.projection_generation() == gen_before
    indexed = q(
        store,
        "SELECT COUNT(*) FROM fts_rows"
        " WHERE claim_id = ? AND projection_generation = ?",
        (out.claim_id, gen_before),
    )[0]
    assert indexed[0] >= 1, "newly active claim must be searchable in-tx"
    event = q(
        store,
        "SELECT kind FROM events WHERE event_seq = ?",
        (seq,),
    )[0]
    assert event[0] == "claim_transition"


def test_apply_review_stale_version_stays_open(store, scope):
    ctx = PolicyContext(cfg=_cfg())
    out, _env, _span = _admit_text(store, scope, "I use Neovim.", ctx=ctx)
    claim_id, review_id = out.claim_id, out.review_id

    # Move the claim forward so the review's fence no longer matches.
    machine = LifecycleMachine(store)
    with store.tx() as conn:
        machine.apply(
            TransitionCommand(
                claim_id=claim_id,
                expected_revision=1,
                effect="reject",
                actor_id="operator",
                reason="superseded by newer info",
            ),
            conn,
        )

    with pytest.raises(VerbatimError) as ei:
        apply_review(store, review_id, "operator", ctx)
    assert ei.value.code == ErrorCode.STALE_PROPOSAL
    # The review remains open and no admit transition was committed.
    assert q(store, "SELECT state FROM reviews")[0][0] == "open"
    with store.tx() as conn:
        assert read_claim_head(conn, claim_id).state == Lifecycle.REJECTED


def test_apply_review_twice_is_stale(store, scope):
    ctx = PolicyContext(cfg=_cfg())
    out, _env, _span = _admit_text(store, scope, "I use Neovim.", ctx=ctx)
    apply_review(store, out.review_id, "operator", ctx)
    with pytest.raises(VerbatimError) as ei:
        apply_review(store, out.review_id, "operator", ctx)
    assert ei.value.code == ErrorCode.STALE_PROPOSAL


def test_apply_review_missing(store):
    with pytest.raises(VerbatimError) as ei:
        apply_review(store, "ghost", "operator", _rules_ctx())
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


# ---------------------------------------------------------------------------
# 7. Evidence families
# ---------------------------------------------------------------------------


def test_duplicate_claim_registers_evidence_family(store, scope):
    first, _e1, _s1 = _admit_text(store, scope, "I use Neovim.")
    # Identical payload bytes in a second source → same span bytes + same
    # payload representation.
    env2 = make_envelope(
        scope,
        "I use Neovim.",
        kind=SourceKind.IMPORT,
        provenance=Provenance.LEGACY_IMPORT,
    )
    span2 = seed_span_for(store, env2)
    prop2 = propose("I use Neovim.", span2, env2, env2.event_us)
    second = admit(store, prop2, env2, _rules_ctx())

    fams = q(store, "SELECT family_id, origin_kind, origin_id FROM evidence_families")
    assert len(fams) == 1
    assert fams[0][1] == "claim" and fams[0][2] == first.claim_id
    members = q(
        store,
        "SELECT object_kind, object_id, role FROM family_members"
        " WHERE family_id = ?",
        (fams[0][0],),
    )
    claim_members = {m[1] for m in members if m[0] == "claim"}
    assert claim_members == {first.claim_id, second.claim_id}
    # Distinct evidence records are preserved — nothing was merged.
    assert q(store, "SELECT COUNT(*) FROM claims")[0][0] == 2


def test_non_duplicate_no_family(store, scope):
    _admit_text(store, scope, "I use Neovim.")
    _admit_text(store, scope, "I prefer dark themes.")
    assert q(store, "SELECT COUNT(*) FROM evidence_families")[0][0] == 0


def test_family_registration_idempotent_replay(store, scope):
    first, _e1, _s1 = _admit_text(store, scope, "I use Neovim.")
    env2 = make_envelope(scope, "I use Neovim.")
    span2 = seed_span_for(store, env2)
    prop2 = propose("I use Neovim.", span2, env2, env2.event_us)
    admit(store, prop2, env2, _rules_ctx())
    fam_count = q(store, "SELECT COUNT(*) FROM evidence_families")[0][0]
    member_count = q(store, "SELECT COUNT(*) FROM family_members")[0][0]
    assert fam_count == 1
    # Re-admitting a third identical copy joins the SAME family.
    env3 = make_envelope(scope, "I use Neovim.")
    span3 = seed_span_for(store, env3)
    prop3 = propose("I use Neovim.", span3, env3, env3.event_us)
    admit(store, prop3, env3, _rules_ctx())
    assert q(store, "SELECT COUNT(*) FROM evidence_families")[0][0] == 1
    assert (
        q(store, "SELECT COUNT(*) FROM family_members")[0][0] > member_count
    )
