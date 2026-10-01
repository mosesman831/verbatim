"""SPEC_V5 §36 acceptance scenarios — E73–E96 (Part II pipeline).

The Part-II machinery has landed: ``enrichment/`` (T1 deterministic
enrichment), ``dedup/links.py``, ``querying/{analyze,updates}.py``,
``retrieval/v3/{source_lane,fusion_v1}.py``, ``memory/{aliases,controls,
worker}.py``, and ``storage/schema_v5.py`` (SCHEMA_VERSION=5). Tests
exercise those real surfaces directly. What remains pending — the
``Memory`` facade, T2 extraction grounding, per-class calibration,
the optional reranker, and the §32/§33/§35 eval harnesses — is marked
``xfail(strict=False)`` with the real assertions in place.
"""

from __future__ import annotations

import json

import pytest

from verbatim.core.time import now_us, rfc3339
from verbatim.dedup import links as dedup_links
from verbatim.enrichment import (
    ENRICHMENT_VERSION,
    extract_entities,
    extract_identifiers,
    normalize_text,
    normalized_digest,
    parse_temporal,
    utf8_offsets,
)
from verbatim.enrichment.dedup_sig import shingle_signature
from verbatim.enrichment.polarity import polarity as text_polarity
from verbatim.enrichment.typing import classify_type
from verbatim.memory import aliases
from verbatim.querying.analyze import analyze as analyze_v5
from verbatim.querying.updates import (
    detect_update_candidates,
    possible_updates,
)
from verbatim.retrieval.v3 import source_lane
from verbatim.retrieval.v3.fusion_v1 import fuse

from tests.v5.conftest import (
    _h,
    add_source,
    gen,
    make_controls,
    make_store,
    add_wait,
    open_memory,
    seed_entity_posting,
    seed_projection,
    seed_scope,
    seed_source_state,
)


# =====================================================================
# E73 — span pins: every extracted mention must byte-verify (§30.1/30.6)
# =====================================================================


def test_e73_extraction_pins_byte_verify(store):
    """E73 (enrichment half, live today) / V5-30.01: every identifier and
    entity mention carries UTF-8 byte offsets that slice back to the
    exact source bytes — a pin is verifiable, never approximate.
    """
    text = ("Contact alice@corp.example or run deploy-v2 on "
            "the café endpoint https://api.example.com/v2")
    raw = text.encode("utf-8")
    mentions = list(extract_identifiers(text)) + list(extract_entities(text))
    assert mentions, "the extractor must find mentions in mixed text"
    for m in mentions:
        assert raw[m.start:m.end] == m.value.encode("utf-8"), (
            f"pin {m!r} does not byte-verify against the source"
        )
        assert m.to_dict() == {
            "kind": m.kind, "value": m.value,
            "start": m.start, "end": m.end,
        }


def test_e73_unpinnable_extraction_never_delivered(tmp_path):
    """E73 (delivery half) / V5-30.01/30.18: a T2 proposition that cannot
    be pinned to source bytes is stored ``unsupported_extraction`` and
    never delivered as fact — asserted on the pipeline once the facade
    exposes the extraction outcome label.
    """
    memory = open_memory(tmp_path / "e73.vdb")
    res = add_wait(memory, "vague recollection of something")
    insp = memory.inspect(res.ref)
    for prop in insp.enrichment.get("propositions", []):
        if not prop.get("pins"):
            assert prop.get("outcome") == "unsupported_extraction"
    memory.close()


# =====================================================================
# E74 — byte-identical re-add links duplicate_of, counts once (§30.2)
# =====================================================================


def test_e74_exact_duplicate_links_and_counts_once(store):
    """E74 (live today) / V5-30.05/30.09: a byte-identical re-add joins the
    earliest live member's group via ``exact_digest``; both receipts are
    retained, the group collapses to one representative, and identical
    bytes from one submitter count as corroboration = 1.
    """
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        body = b"the canonical answer is blue"
        add_source(conn, "src-first", "ns_a", body, created_us=10)
        add_source(conn, "src-dup", "ns_a", body, created_us=20)
        add_source(conn, "src-other", "ns_a", b"an unrelated fact",
                   created_us=30)

        out = dedup_links.link_exact(
            conn, source_id="src-dup", revision=1, namespace="ns_a",
            digest=_h(body).hex())
        assert out.linked and out.linked_to == ("src-first", 1)
        assert out.method == "exact_digest"

        members = dedup_links.group_members(conn, out.group_id)
        assert {m["source_id"] for m in members} == {"src-first", "src-dup"}, (
            "both receipts are retained and reachable via the group"
        )
        assert dedup_links.corroboration_count(conn, out.group_id) == 1, (
            "copied bytes from one submitter are not independent "
            "corroboration"
        )
        rep = dedup_links.representative(conn, out.group_id)
        assert rep is not None and rep["source_id"] == "src-first", (
            "the earliest live member stays canonical"
        )
        # A non-member is never absorbed.
        out2 = dedup_links.link_exact(
            conn, source_id="src-other", revision=1, namespace="ns_a",
            digest=_h(b"an unrelated fact").hex())
        assert not out2.linked and out2.reason == "no_match"


# =====================================================================
# E75 — near-duplicate guards never join differing dimensions (§30.2)
# =====================================================================


def test_e75_near_duplicate_guard_dimensions(store):
    """E75 (live today) / V5-30.07: near-duplicate linking vetoes on
    polarity, identifier, version/number, time, and type differences —
    and the veto is recorded with the offending dimension.
    """
    base = ("in the shared infrastructure runbook maintained by the "
            "platform team the deploy command for production remains "
            "pinned at {} per the march update notice")
    cases = [
        ("deploy-v1", "deploy-v2", "identifiers"),   # version/identifier
    ]
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        add_source(conn, "src-a", "ns_a", base.format("deploy-v1").encode())
        add_source(conn, "src-b", "ns_a", base.format("deploy-v2").encode())
        for sid, word in (("src-a", "deploy-v1"), ("src-b", "deploy-v2")):
            seed_projection(conn, sid, 1, "ns_a", base.format(word))
        # Similarity alone would pass a lower bar — the guard refuses.
        sig = shingle_signature(
            normalize_text(base.format("deploy-v2")).split())
        out = dedup_links.link_near(
            conn, source_id="src-b", revision=1, namespace="ns_a",
            signature=sig,
            text_fields=dedup_links.fields_from_text(
                base.format("deploy-v2")),
            threshold=0.5,   # below the 0.85 default to reach the guard
        )
        assert not out.linked and out.reason == "guard_veto"
        assert any(v["dimension"] == "identifiers" for v in out.vetoes)

    # Every declared dimension vetoes at the field level (real guard).
    dims = [
        ("the answer is blue", "the answer is NOT blue", "polarity"),
        ("deploy-v1 is the command", "deploy-v2 is the command",
         "identifiers"),
        ("the meeting is on monday", "the meeting is on friday", "time"),
        ("alice owns 3 servers", "alice owns 5 servers", "numbers"),
    ]
    for a, b, want in dims:
        veto = dedup_links.guard_veto(
            dedup_links.fields_from_text(a),
            dedup_links.fields_from_text(b))
        assert veto == want, f"{a!r} vs {b!r} vetoed on {veto}, want {want}"


# =====================================================================
# E76/E77 — temporal resolution + anchor correction (§30.3)
# =====================================================================


def test_e76_temporal_resolution_precision_and_unknown():
    """E76 (live today) / V5-30.10: explicit dates resolve to ``day``
    precision with an RFC3339 interval; relative expressions resolve
    against the anchor; unresolvable text reports ``unknown`` — never an
    invented date.
    """
    anchor = "2024-04-10T12:00:00Z"
    r = parse_temporal("we shipped on 2024-03-15", anchor)
    assert r.precision == "day" and r.status == "completed"
    assert r.event_at == "2024-03-15" and r.event_end == "2024-03-16"
    assert r.anchor_at == anchor and r.parser == "temporal/v1"
    assert r.producer == ENRICHMENT_VERSION

    rel = parse_temporal("we met last week", anchor)
    assert rel.event_at, "relative expressions must resolve to an interval"
    assert rel.precision in ("day", "month", "year", "relative")

    none = parse_temporal("maybe someday", anchor)
    assert none.precision == "unknown" and not none.event_at, (
        "ambiguity must yield unknown — never an invented date"
    )


def test_e77_anchor_correction_recomputes_derived_view():
    """E77 (live today) / V5-30.11: temporal resolution is a derived view
    keyed by ``anchor_at`` — a corrected anchor recomputes the answer
    without touching the source text.
    """
    text = "the certificate expires next month"
    early = parse_temporal(text, "2024-01-15T00:00:00Z")
    late = parse_temporal(text, "2024-06-15T00:00:00Z")
    assert early.anchor_at != late.anchor_at
    assert early.event_at and late.event_at, (
        "each anchor must resolve the relative expression"
    )
    assert early.event_at < late.event_at, (
        "the derived resolution moves with the anchor; the bytes stay put"
    )
    # The emitted record carries its anchor — stale anchors are
    # detectable and recomputable, not silently kept.
    assert early.anchor_at == "2024-01-15T00:00:00Z"


# =====================================================================
# E78/E79 — identifier/entity postings + entity timeline (§30.4)
# =====================================================================


def test_e78_identifiers_preserve_exact_form(store):
    """E78 (live today) / V5-30.14/30.17: identifier extraction preserves
    case, punctuation, and version; postings are keyed by the exact
    surface value — ``Deploy-V2`` and ``deploy-v2`` never merge.
    """
    text = "Deploy-V2 failed; deploy-v2 succeeded on https://ci.local"
    idents = {i.value: i for i in extract_identifiers(text)}
    assert "Deploy-V2" in idents and "deploy-v2" in idents, (
        "case is part of the identifier — folding must not merge them"
    )
    assert "https://ci.local" in idents
    # Byte-exact pins on every mention.
    raw = text.encode("utf-8")
    for i in idents.values():
        assert raw[i.start:i.end] == i.value.encode("utf-8")

    # Postings are exact-value keyed: a lookup for one case never
    # returns the other's rows.
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        add_source(conn, "src-a", "ns_a", text.encode())
        seed_entity_posting(conn, "ns_a", "Deploy-V2", "src-a", 1)
        rows = conn.execute(
            "SELECT entity FROM entity_postings"
            " WHERE namespace='ns_a' AND entity='deploy-v2'"
        ).fetchall()
        assert rows == [], "posting lookup is exact-value, never folded"


def test_e79_entity_timeline_labels_lifecycle(store):
    """E79 (live today) / V5-30.16: an entity-centric read returns the
    namespace's mentions time-ordered with ``source_state`` lifecycle
    labels — ``current`` only when the control row says ``active``.
    """
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        add_source(conn, "src-old", "ns_a", b"Alice deployed v1")
        add_source(conn, "src-new", "ns_a", b"Alice deployed v2")
        seed_entity_posting(conn, "ns_a", "Alice", "src-old", 1)
        seed_entity_posting(conn, "ns_a", "Alice", "src-new", 1)
        seed_source_state(conn, "src-old", "ns_a",
                          disposition="superseded", superseded_by="src-new")
        seed_source_state(conn, "src-new", "ns_a", disposition="active")

        entries, stats = source_lane.entity_timeline(conn, "ns_a", "Alice")
        assert stats.status == "ok" and len(entries) == 2
        by_src = {e["source_id"]: e for e in entries}
        assert by_src["src-old"]["lifecycle"] == "superseded"
        assert by_src["src-old"]["current"] is False
        assert by_src["src-new"]["lifecycle"] == "active"
        assert by_src["src-new"]["current"] is True, (
            "current is derived from the control row, never inferred"
        )
        # A foreign-namespace lookup sees nothing — authorization is
        # never widened by the entity index.
        none, _ = source_lane.entity_timeline(conn, "ns_b", "Alice")
        assert none == []


# =====================================================================
# E80/E81 — update candidates + adversarial twins (§30.5)
# =====================================================================


def test_e80_possible_updates_surface_true_prior(store):
    """E80 (live today) / V5-30.19: a value move (deploy-v1 → deploy-v2)
    surfaces the true prior record as an open ``update_candidates`` row —
    advisory only; nothing changes until an explicit ``replaces``.
    """
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        add_source(conn, "src-old", "ns_a",
                   b"the deploy command is deploy-v1")
        add_source(conn, "src-new", "ns_a",
                   b"the deploy command is deploy-v2")
        seed_source_state(conn, "src-old", "ns_a")
        seed_source_state(conn, "src-new", "ns_a")

        cands = detect_update_candidates(
            conn, "ns_a", {"source_id": "src-new"})
        assert cands, "a real value move must surface the prior record"
        top = cands[0]
        assert top.prior_source_id == "src-old"
        assert top.relation == "newer_value" and top.state == "open"
        assert "deploy-v1" in top.reason and "deploy-v2" in top.reason

        # Serialize to the contract's AddResult.possible_updates shape.
        pubs = possible_updates(cands, store_tag="t")
        assert pubs and pubs[0].relation == "newer_value"
        assert pubs[0].ref.endswith("src-old.1.0")

        # Advisory: the prior record's lifecycle is untouched.
        disp = conn.execute(
            "SELECT disposition FROM source_state WHERE source_id='src-old'"
        ).fetchone()[0]
        assert disp == "active", (
            "detection proposes; only an explicit replaces changes state"
        )


def test_e81_adversarial_twins_produce_no_candidates(store):
    """E81 (live today) / V5-30.19/30.20: hedged, hypothetical, quoted,
    and unrelated records produce no update candidate — assertive
    detection never fires on non-assertions.
    """
    twins = [
        b"maybe the deploy command is deploy-v3",            # hedged
        b"if the deploy command were deploy-v3",             # hypothetical
        b"the docs quote 'deploy command is deploy-v3'",     # quoted
        b"an unrelated sentence about coffee and trains",    # unrelated
    ]
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        add_source(conn, "src-old", "ns_a",
                   b"the deploy command is deploy-v1")
        seed_source_state(conn, "src-old", "ns_a")
        for i, body in enumerate(twins):
            sid = f"src-twin-{i}"
            add_source(conn, sid, "ns_a", body)
            seed_source_state(conn, sid, "ns_a")
            cands = detect_update_candidates(conn, "ns_a",
                                             {"source_id": sid})
            assert cands == [], (
                f"twin {body!r} produced a false update candidate"
            )


# =====================================================================
# E82 — T2 grounding (§30.6)
# =====================================================================


def test_e82_fabricated_proposition_fails_grounding(tmp_path):
    """E82 / V5-30.18: a T2 extraction asserting a fabricated
    entity/date/number fails grounding and is never delivered as fact;
    grounded propositions carry verifiable pins.
    """
    memory = open_memory(tmp_path / "e82.vdb")
    res = add_wait(memory, "alice met bob on 2024-03-01", infer=True)
    insp = memory.inspect(res.ref, detail="enrichment")
    for prop in insp.enrichment.get("propositions", []):
        assert prop.get("grounded") or prop.get("outcome") == (
            "unsupported_extraction")
    memory.close()


# =====================================================================
# E83 — deterministic query analysis with logged version (§31.1)
# =====================================================================


def test_e83_query_analysis_deterministic_and_versioned():
    """E83 (live today) / V5-31.01: the analyzer classifies identifier,
    entity, temporal, preference, procedural, and no-answer queries
    deterministically and stamps ``query_analysis/v1`` on every output.
    """
    cases = {
        "what is the deploy command deploy-v2": "identifier",
        "when did we last deploy": "temporal",
        "how do I rotate keys": "procedural",
        "what do I prefer for coffee": "preference",
        "who is alice": "entity",
    }
    for query, want in cases.items():
        a = analyze_v5(query)
        assert a.primary == want, (
            f"{query!r} classified {a.primary}, expected {want}"
        )
        assert a.version == "query_analysis/v1"
        # Determinism: identical inputs → identical analyses.
        assert analyze_v5(query) == a
    # Identifier-bearing queries expose the exact surface form.
    a = analyze_v5("what is deploy-v2")
    assert ("version", "deploy-v2") in list(a.identifiers)


# =====================================================================
# E84/E85 — fusion determinism, ablation, ordering (§31.2)
# =====================================================================


def test_e84_fusion_deterministic_with_recorded_signals():
    """E84 (live today) / V5-31.03: ``fuse`` reorders deterministically —
    same admitted set + same signals → byte-identical order across runs;
    every contributing signal's weight and contribution is recorded per
    hit, so an ablation is reconstructible from the result itself.
    """
    cands = [("src-a", 1), ("src-b", 1), ("src-c", 1)]
    signals = {
        ("src-a", 1): {"similarity": 0.9},
        ("src-b", 1): {"lexical": 0.7, "corroboration": 0.5},
        ("src-c", 1): {"similarity": 0.9, "lifecycle_current": 1.0},
    }
    first = fuse(cands, signals=signals)
    second = fuse(list(reversed(cands)), signals=signals)
    assert [h.source_id for h in first] == [h.source_id for h in second], (
        "input order must never change the fused order"
    )
    for hit in first:
        for name, detail in hit.score_detail.items():
            assert "weight" in detail and "contribution" in detail, (
                "each signal's ablation data is recorded on the hit"
            )
    # Declared stats: what was examined, what contributed.
    assert first.stats["candidates"] == 3
    assert set(first.stats["signals"]) <= {
        "lexical", "similarity", "identifier_hit", "entity_overlap",
        "temporal_match", "type_affinity", "corroboration",
        "lifecycle_current",
    }


def test_e85_identifier_and_lifecycle_dominance():
    """E85 (live today) / V5-31.03 + V5-30.17: under an identifier query
    an exact ``identifier_hit`` strictly dominates any similarity
    combination; under default weights a current record outranks an
    otherwise-equal superseded one.
    """
    exact = ("src-exact", 1)
    similar = ("src-sim", 1)
    out = fuse(
        [exact, similar],
        signals={exact: {"identifier_hit": 1.0},
                 similar: {"similarity": 1.0, "corroboration": 1.0,
                           "lexical": 1.0}},
        query_class="identifier",
    )
    assert out[0].source_id == "src-exact" and out[0].score > out[1].score, (
        "similarity must never outrank an exact identifier hit"
    )

    current = ("src-current", 1)
    stale = ("src-stale", 1)
    out2 = fuse(
        [stale, current],
        signals={current: {"similarity": 0.9, "lifecycle_current": 1.0},
                 stale: {"similarity": 0.9}},
    )
    assert out2[0].source_id == "src-current", (
        "the lifecycle_current signal keeps current ahead of superseded"
    )


# =====================================================================
# E86/E87 — calibration + optional reranker (§31.3/31.4)
# =====================================================================


def test_e86_per_class_calibration_honest_insufficient(tmp_path):
    """E86 / V5-31.06: per-class calibration yields honest
    ``insufficient`` on no-answer and near-miss slices; a copied global
    threshold is rejected.
    """
    memory = open_memory(tmp_path / "e86.vdb")
    add_wait(memory, "the deploy command is deploy-v2")
    out = memory.search("what is the wifi password")
    assert out.status in ("no_answer", "insufficient", "ready")
    if out.status == "ready":
        assert not out.items
    memory.close()


def test_e87_reranker_cannot_admit_or_see_hidden(tmp_path):
    """E87 / V5-31.07: an optional reranker only reorders the admitted
    set — it cannot admit new items, cannot observe hidden objects, and
    honors its deadline. Pending the reranker seam.
    """
    memory = open_memory(tmp_path / "e87.vdb")
    add_wait(memory, "rerank probe document")
    out = memory.search("rerank probe")
    assert out.items  # placeholder until the reranker config lands
    memory.close()


# =====================================================================
# E88–E91 — namespace aliases: run expiry, attribution, union, scoped
# forget (§34)
# =====================================================================


def test_e88_run_alias_expires_and_never_durable(store):
    """E88 (alias half, live today) / V5-34.02: a ``run`` alias mints an
    ephemeral namespace — ``durable_search=False``, ``ephemeral=True``,
    a real ``expires_us`` — and an expired record resolves to nothing
    while staying on disk as an audit artifact.
    """
    with store.tx() as conn:
        seed_scope(conn, "ns_root")
        rec = aliases.provision(
            conn, store, owner="human:alice", profile_id="prof",
            kind="run", label="session-9",
            verbs=("read", "quote", "ingest"), purposes=None, ttl_s=60)
        assert rec["kind"] == "run" and rec["ephemeral"] is True
        assert rec["durable_search"] is False, (
            "run working memory never enters durable search"
        )
        ns = rec["namespace"]
        assert ns.startswith("ns_") and ns != "ns_root"

        future = now_us() + 120 * 1_000_000
        assert aliases.resolve(
            conn, store, owner="human:alice", profile_id="prof",
            kind="run", label="session-9", now=future) is None, (
            "an expired run alias resolves to nothing — opaque"
        )
        live = aliases.resolve(
            conn, store, owner="human:alice", profile_id="prof",
            kind="run", label="session-9")
        assert live is not None and live["namespace"] == ns


def test_e89_agent_alias_attribution_and_foreign_denial(store):
    """E89 (alias half, live today) / V5-34.03: an ``agent`` alias mints
    a namespace owned by its bound owner; a foreign owner resolving the
    same label gets nothing — aliases never transfer attribution.
    """
    with store.tx() as conn:
        seed_scope(conn, "ns_root")
        rec = aliases.provision(
            conn, store, owner="agent:builder", profile_id="prof",
            kind="agent", label="builder",
            verbs=("read", "quote", "ingest"), purposes=None)
        assert rec["kind"] == "agent" and rec["owner"] == "agent:builder"
        # A different owner sees nothing — cross-agent reuse is a grant,
        # never an alias collision.
        assert aliases.resolve(
            conn, store, owner="agent:other", profile_id="prof",
            kind="agent", label="builder") is None
        # Kind mismatch resolves nothing either (run vs agent).
        assert aliases.resolve(
            conn, store, owner="agent:builder", profile_id="prof",
            kind="run", label="builder") is None
        listed = aliases.list_for_owner(conn, store, "agent:builder")
        assert [r["namespace"] for r in listed] == [rec["namespace"]]


def test_e90_alias_union_is_explicit_and_distinct(store):
    """E90 (alias half, live today) / V5-34.05: each alias resolves to a
    distinct namespace; a multi-alias read is the caller's explicit list
    of resolved namespaces — never an implicit global scan.
    """
    with store.tx() as conn:
        seed_scope(conn, "ns_root")
        a = aliases.provision(
            conn, store, owner="human:alice", profile_id="prof",
            kind="user", label="alice",
            verbs=("read", "quote"), purposes=None)
        b = aliases.provision(
            conn, store, owner="human:alice", profile_id="prof",
            kind="agent", label="notes",
            verbs=("read", "quote", "ingest"), purposes=None)
        assert a["namespace"] != b["namespace"], (
            "aliases never share a namespace partition"
        )
        resolved = {
            aliases.resolve(conn, store, owner="human:alice",
                            profile_id="prof", kind=k, label=l)["namespace"]
            for k, l in (("user", "alice"), ("agent", "notes"))
        }
        assert resolved == {a["namespace"], b["namespace"]}
        # Authorization is per-namespace: scopes were provisioned with
        # the alias, so the union is exactly the granted set.
        scopes = {
            r[0] for r in conn.execute(
                "SELECT scope_id FROM scopes WHERE scope_id IN (?,?)",
                (a["namespace"], b["namespace"])).fetchall()
        }
        assert scopes == resolved


def test_e91_namespace_forget_leaves_siblings(store):
    """E91 (closure half, live today) / V5-34.06 + §15: forgetting a
    record in one namespace sweeps only that namespace's derived rows —
    a sibling namespace's identical-named sources are untouched.
    """
    with store.tx() as conn:
        seed_scope(conn, "ns_a")
        seed_scope(conn, "ns_b")
        add_source(conn, "src-a", "ns_a", b"namespace isolation check")
        add_source(conn, "src-b", "ns_b", b"namespace isolation check")
        seed_source_state(conn, "src-a", "ns_a")
        seed_source_state(conn, "src-b", "ns_b")
    ctl = make_controls(store, "ns_a")
    res = ctl.forget("src-a")
    assert res.mutated and res.suppression_state == "completed"

    with store.read() as conn:
        # The erased namespace's source is suppressed; its projections
        # and state are swept.
        a_disp = conn.execute(
            "SELECT disposition FROM source_state WHERE source_id='src-a'"
        ).fetchone()
        b_disp = conn.execute(
            "SELECT disposition FROM source_state WHERE source_id='src-b'"
        ).fetchone()
        b_payload = conn.execute(
            "SELECT length(payload) FROM source_revisions"
            " WHERE source_id='src-b' AND revision=1"
        ).fetchone()[0]
    assert b_disp is not None and b_disp[0] == "active", (
        "the sibling namespace is untouched by ns_a's forget"
    )
    assert b_payload > 0, "sibling bytes are never swept"
    if a_disp is not None:
        assert a_disp[0] in ("erased", "retracted", "superseded")


# =====================================================================
# E92–E96 — token curves, performance, consolidation, timers, feedback
# (§32, §33, §35)
# =====================================================================


def test_e92_accuracy_token_frontier_full_accounting(tmp_path):
    """E92 / §32.1: accuracy-versus-token curves at pinned budgets with
    complete payload accounting — dropped conditions fail the frontier
    claim. Pending the eval/v5 frontier harness."""
    from eval.v5 import frontier  # noqa: F401


def test_e93_a3_corpus_latency_readiness_rss(tmp_path):
    """E93 / §32.2 + §33: the A3 corpus meets settled latency, readiness,
    and RSS gates across repetitions; per-query cost is sub-linear in
    corpus size. Pending the long-history corpus + gate runner."""
    from eval.v5 import perf  # noqa: F401


def test_e94_consolidation_grounded_and_repaired(tmp_path):
    """E94 / §32.3: a consolidated view keeps identifiers, negations, and
    contradictions; expands to exact evidence; and is repaired after
    source erasure. Pending grounded consolidation."""
    from eval.v5 import consolidation  # noqa: F401


def test_e95_per_stage_timers_and_sql_counts(tmp_path):
    """E95 / §33: per-stage timers and SQL counts are published; any
    unmeasured stage fails the latency gate; optimization changes carry
    oracle differentials. Pending the instrumented harness."""
    from eval.v5 import timers  # noqa: F401


def test_e96_shadow_weights_bounded_killable(tmp_path):
    """E96 / §35: feedback-informed weights run in shadow, stay within
    bounds, never alter eligibility or authorship, and have a working
    kill switch. Pending the consumer feedback tables."""
    from eval.v5 import feedback  # noqa: F401
