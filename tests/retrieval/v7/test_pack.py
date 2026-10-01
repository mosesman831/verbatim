"""S8 pack tests — tok/v1, greedy packing, collapse, neighbors,
grouping, computed answers, and the `pack_render/v1` reader view.

Types are constructed directly (V7 wave-A rule: no sibling imports).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from verbatim.core.types_v7 import (
    IntentClass,
    IntentResult,
    IntervalUs,
    LifecycleLabel,
    MissingDescriptor,
    NormAnalysis,
    OccurredPrecision,
    PackItemV7,
    QueryViewV7,
    ScoredCandidate,
    SupportLabel,
)
from verbatim.retrieval.v7.computed import computed_items
from verbatim.retrieval.v7.pack import (
    assemble_pack,
    estimate_tokens,
)
from verbatim.retrieval.v7.render import render_reader_view


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _us(y, m, d, hh=12):
    return int(datetime(y, m, d, hh, tzinfo=timezone.utc).timestamp() * 1e6)


def _iv(y, m, d, precision=OccurredPrecision.DAY):
    start = _us(y, m, d, 0)
    end = _us(y, m, d, 0) + 86_400_000_000
    return IntervalUs(start_us=start, end_us=end, precision=precision)


def _norm(q: str) -> NormAnalysis:
    return NormAnalysis(analyzer_id="norm/v2", terms=(), identifiers=(), text=q)


def _query(intent=IntentClass.LOOKUP, q="q", qt=None) -> QueryViewV7:
    return QueryViewV7(
        query=q,
        norm=_norm(q),
        intent=IntentResult(primary=intent, classes=(intent,)),
        query_time_us=qt,
    )


def _item(
    ref,
    text,
    *,
    unit_id=None,
    speaker="Jordan",
    recorded="2023-06-04T10:00:00",
    occurred=None,
    session=None,
    lifecycle=LifecycleLabel.CURRENT,
    support=SupportLabel.SUPPORTED,
    derived=False,
    proof_count=0,
    pins=None,
) -> PackItemV7:
    return PackItemV7(
        ref=ref,
        unit_id=unit_id or ref.split(":")[-1],
        quote=text.encode("utf-8"),
        speaker=speaker,
        recorded_at=recorded,
        occurred=occurred,
        session=session,
        lifecycle=lifecycle,
        support=support,
        derived=derived,
        proof_count=proof_count,
        pins=dict(pins or {}),
    )


def _scored(item: PackItemV7, score: float) -> ScoredCandidate:
    return ScoredCandidate(
        unit_id=item.unit_id,
        source_id="src",
        revision=1,
        score=score,
        score_family="ranking/v7",
        detail={"item": item},
    )


def _scored_fields(unit_id, text, score, **detail) -> ScoredCandidate:
    d = {"quote": text.encode("utf-8")}
    d.update(detail)
    return ScoredCandidate(
        unit_id=unit_id,
        source_id="src",
        revision=1,
        score=score,
        score_family="ranking/v7",
        detail=d,
    )


_SESS_A = {"id": "s-14", "date": "2023-06-04", "speakers": ["Jordan", "Assistant"]}
_SESS_B = {"id": "s-15", "date": "2023-06-18", "speakers": ["Jordan", "Assistant"]}


# ---------------------------------------------------------------------------
# tok/v1 (§32.15)
# ---------------------------------------------------------------------------


def test_estimate_tokens_formula_exact():
    # "Hello, world!" -> 13 utf8 bytes, punct adjacent to word chars:
    # ',' and '!' -> 13/4 + 0.25*2 = 3.75 -> ceil 4; floor 0.75*2=1.5->2
    assert estimate_tokens("Hello, world!") == 4
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1  # ceil(0.25) -> 1


def test_estimate_tokens_sanity_vs_len4():
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "I finally joined the pottery class yesterday!",
        "word " * 200,
        "Un texte en français avec des accents éèê.",
        "average length words flowing together naturally.",
    ]
    for s in samples:
        est = estimate_tokens(s)
        naive = len(s.encode("utf-8")) / 4
        assert abs(est - naive) <= max(2.0, 0.5 * naive), (s, est, naive)


def test_estimate_tokens_word_floor():
    # many 1-char words: bytes small, word floor binds (0.75 * wc)
    s = "a a a a a a a a"  # 15 bytes -> ~4; wc 8 -> floor 6
    assert estimate_tokens(s) == 6
    # punctuation-free prose: floor below byte estimate
    assert estimate_tokens("hello world") >= 2


def test_estimate_tokens_deterministic_and_typed():
    assert estimate_tokens("same text") == estimate_tokens("same text")
    assert estimate_tokens(b"bytes ok") == estimate_tokens("bytes ok")


# ---------------------------------------------------------------------------
# greedy packing: skip-long-continue, never-empty, budget
# ---------------------------------------------------------------------------


def test_skip_long_continue():
    small1 = _scored_fields("1", "tiny one.", 3.0)
    big = _scored_fields("2", "x " * 400, 2.0)  # ~200 tokens
    small2 = _scored_fields("3", "tiny two.", 1.0)
    res = assemble_pack(
        [small1, big, small2], _query(), max_tokens=50, limit=10
    )
    refs = [i.ref for i in res.items]
    assert "u:1" in refs and "u:3" in refs
    assert "u:2" not in refs
    assert res.omitted == 1
    assert res.tokens_items <= 50


def test_never_empty_truncated_sentence_boundary():
    text = "First sentence here. Second sentence is quite a bit longer than the first. Third sentence too."
    it = _item("u:1", text)
    res = assemble_pack([it], _query(), max_tokens=8, limit=5)
    assert len(res.items) == 1
    assert res.truncated is True
    assert res.items[0].pins.get("truncated") is True
    out = res.items[0].quote.decode()
    assert out == "First sentence here."  # sentence boundary honored
    assert estimate_tokens(out) <= 8


def test_never_empty_word_boundary_fallback():
    text = "Supercalifragilisticexpialidocious antidisestablishmentarianism floccinaucinihilipilification"
    it = _item("u:1", text)
    res = assemble_pack([it], _query(), max_tokens=6, limit=5)
    assert len(res.items) == 1
    assert res.items[0].pins.get("truncated") is True
    out = res.items[0].quote.decode()
    assert "." not in out  # no sentence boundary available
    assert estimate_tokens(out) <= 6


def test_budget_respected_items_only():
    items = [
        _scored_fields(str(i), f"content number {i} " * 5, 10 - i)
        for i in range(10)
    ]
    res = assemble_pack(items, _query(), max_tokens=60, limit=50)
    assert res.tokens_items <= 60
    total = sum(
        estimate_tokens(i.quote.decode())
        + sum(estimate_tokens(c.quote.decode()) for c in i.context)
        for i in res.items
    )
    assert res.tokens_items == total


def test_metadata_excluded_from_token_count():
    bare = _item("u:1", "same text", speaker=None, recorded=None, session=None)
    rich = _item(
        "u:1",
        "same text",
        speaker="A very long speaker name that would cost tokens",
        recorded="2023-06-04T10:00:00",
        session=_SESS_A,
    )
    r1 = assemble_pack([bare], _query(), max_tokens=100, limit=5)
    r2 = assemble_pack([rich], _query(), max_tokens=100, limit=5)
    assert r1.tokens_items == r2.tokens_items


def test_limit_binds():
    items = [_scored_fields(str(i), f"text {i}", 10 - i) for i in range(6)]
    res = assemble_pack(items, _query(), max_tokens=10_000, limit=2)
    assert len(res.items) == 2
    assert res.omitted == 4


def test_scored_sorted_by_score_stable():
    a = _scored_fields("a", "aaa", 0.5)
    b = _scored_fields("b", "bbb", 0.9)
    c = _scored_fields("c", "ccc", 0.9)
    res = assemble_pack([a, c, b], _query(), max_tokens=100, limit=5)
    assert [i.unit_id for i in res.items] == ["c", "b", "a"]


# ---------------------------------------------------------------------------
# duplicate collapse + conflicts (V7-12.04)
# ---------------------------------------------------------------------------


def test_duplicate_collapse():
    i1 = _item("u:1", "I love hiking on weekends.")
    i2 = _item("u:2", "i love hiking on weekends")  # near-dup
    i3 = _item("u:3", "Completely different content.")
    res = assemble_pack([i1, i2, i3], _query(), max_tokens=500, limit=10)
    refs = [i.ref for i in res.items]
    assert "u:1" in refs and "u:2" not in refs and "u:3" in refs
    assert res.collapsed_duplicates == 1
    rep = res.items[0]
    assert rep.pins.get("collapsed_duplicates") == 1


def test_conflicts_never_collapsed():
    # same normalized text but one superseded -> both must ship
    cur = _item("u:1", "Home city is Denver.", lifecycle=LifecycleLabel.CURRENT)
    sup = _item(
        "u:2", "Home city is Denver.", lifecycle=LifecycleLabel.SUPERSEDED
    )
    res = assemble_pack([cur, sup], _query(), max_tokens=500, limit=10)
    assert {i.ref for i in res.items} == {"u:1", "u:2"}
    assert res.collapsed_duplicates == 0


def test_conflict_group_atomic_or_unresolved():
    a = _item("u:1", "The meeting is on Monday.", pins={"conflict_group": "when"})
    b = _item("u:2", "The meeting is on Tuesday.", pins={"conflict_group": "when"})
    # together they fit: ship as one atomic group
    res = assemble_pack([a, b], _query(), max_tokens=500, limit=10)
    assert {i.ref for i in res.items} == {"u:1", "u:2"}
    assert res.conflicts == [("conflict:when", ("u:1", "u:2"))]
    assert not res.conflict_unresolved
    # over budget: the whole group drops, unresolved flag rises
    res2 = assemble_pack([a, b], _query(), max_tokens=10, limit=10)
    assert res2.conflict_unresolved is True
    # never-empty may deliver a truncated head — the group did NOT ship whole
    assert res2.unresolved_conflicts[0][0] == "conflict:when"


def test_state_key_group_atomic():
    cur = _item(
        "u:1",
        "My home city is Denver.",
        pins={"state_key": "home_city", "state_value": "Denver"},
    )
    sup = _item(
        "u:2",
        "My home city is Austin.",
        lifecycle=LifecycleLabel.SUPERSEDED,
        pins={"state_key": "home_city", "state_value": "Austin"},
    )
    res = assemble_pack([cur, sup], _query(), max_tokens=6, limit=10)
    # cannot fit both -> group withheld, flagged, not silently split
    assert res.conflict_unresolved is True


# ---------------------------------------------------------------------------
# neighbors (V7-12.05)
# ---------------------------------------------------------------------------


def test_neighbor_context_counted_not_ranked():
    main = _item("u:10", "I joined the pottery class yesterday!", session=_SESS_A)
    nb = _item(
        "u:11",
        "That's great — how was the first session?",
        speaker="Assistant",
        session=_SESS_A,
    )
    calls = []

    def neighbors_fn(item, n):
        calls.append((item.ref, n))
        return [nb] if item.ref == "u:10" else []

    res = assemble_pack(
        [main, _item("u:9", "other")],
        _query(),
        max_tokens=500,
        limit=10,
        neighbor_window=1,
        neighbors_fn=neighbors_fn,
    )
    parent = next(i for i in res.items if i.ref == "u:10")
    assert [c.ref for c in parent.context] == ["u:11"]
    # context counts against the item-token budget
    assert res.tokens_items == estimate_tokens(
        "I joined the pottery class yesterday!"
    ) + estimate_tokens("other") + estimate_tokens(
        "That's great — how was the first session?"
    )
    # never ranked independently: u:11 is not a top-level delivered item
    assert "u:11" not in [i.ref for i in res.items]
    assert calls and calls[0][1] == 1


def test_neighbor_budget_skip_and_dedup():
    main = _item("u:10", "short", session=_SESS_A)
    nb = _item("u:11", "a much longer neighbor turn " * 20, session=_SESS_A)

    res = assemble_pack(
        [main],
        _query(),
        max_tokens=30,
        limit=10,
        neighbor_window=1,
        neighbors_fn=lambda i, n: [nb],
    )
    assert res.items[0].context == []  # over budget -> skipped, not fatal

    # a neighbor that is itself delivered never double-ships as context
    res2 = assemble_pack(
        [main, nb],
        _query(),
        max_tokens=10_000,
        limit=10,
        neighbor_window=1,
        neighbors_fn=lambda i, n: [nb],
    )
    assert res2.items[0].context == []


def test_neighbor_window_zero_disables():
    main = _item("u:10", "short", session=_SESS_A)
    res = assemble_pack(
        [main],
        _query(),
        max_tokens=500,
        limit=10,
        neighbor_window=0,
        neighbors_fn=lambda i, n: [_item("u:11", "nbr", session=_SESS_A)],
    )
    assert res.items[0].context == []


# ---------------------------------------------------------------------------
# session grouping + order (V7-12.06)
# ---------------------------------------------------------------------------


def _two_session_pack(order="relevance", intent=IntentClass.LOOKUP):
    # s-15 item ranks first; s-14 session is the earlier one
    a = _item("u:81", "joined pottery class", session=_SESS_A,
              recorded="2023-06-04T09:00:00", occurred=_iv(2023, 6, 3))
    b = _item("u:97", "workshop was exhausting", session=_SESS_B,
              recorded="2023-06-18T09:00:00", occurred=_iv(2023, 6, 17))
    return assemble_pack(
        [_scored(b, 0.9), _scored(a, 0.5)],
        _query(intent=intent),
        max_tokens=500,
        limit=10,
        order=order,
    )


def test_session_grouping_relevance_order():
    res = _two_session_pack(order="relevance")
    assert [g.session_id for g in res.groups] == ["s-15", "s-14"]


def test_session_grouping_chronological_param():
    res = _two_session_pack(order="chronological")
    assert [g.session_id for g in res.groups] == ["s-14", "s-15"]


def test_session_grouping_chronological_by_intent():
    res = _two_session_pack(order="relevance", intent=IntentClass.HISTORY_OF)
    assert [g.session_id for g in res.groups] == ["s-14", "s-15"]
    res2 = _two_session_pack(order="relevance", intent=IntentClass.TEMPORAL_POINT)
    assert [g.session_id for g in res2.groups] == ["s-14", "s-15"]


def test_items_carry_v7_12_07_fields():
    res = _two_session_pack()
    it = res.items[0]
    assert it.ref and isinstance(it.quote, bytes)
    assert it.speaker == "Jordan"
    assert it.recorded_at == "2023-06-18T09:00:00"
    assert it.occurred is not None and it.occurred.start_us is not None
    assert it.session["id"] == "s-15"
    assert it.lifecycle is LifecycleLabel.CURRENT
    assert it.support is SupportLabel.SUPPORTED


# ---------------------------------------------------------------------------
# derived items (V7-12.08)
# ---------------------------------------------------------------------------


def test_derived_item_gets_support_and_labeling():
    sup_unit = _item("u:40", "I go hiking every weekend.")
    obs = _item(
        "o:12",
        "Jordan prefers hiking",
        derived=True,
        proof_count=3,
        pins={"supports": [sup_unit], "support_refs": ["u:40"]},
    )
    res = assemble_pack([obs], _query(), max_tokens=500, limit=10)
    assert res.items[0].derived is True
    assert res.items[0].proof_count == 3
    # supporting unit attached as context (not ranked independently)
    assert [c.ref for c in res.items[0].context] == ["u:40"]
    assert "u:40" not in [i.ref for i in res.items]
    assert not res.warnings


def test_derived_item_unsupported_warns_not_fails():
    obs = _item(
        "o:12",
        "Jordan prefers hiking",
        derived=True,
        proof_count=1,
        pins={"support_refs": ["u:99"]},  # ref exists but undelivered
    )
    res = assemble_pack([obs], _query(), max_tokens=500, limit=10)
    assert res.warnings == ["derived_without_support:o:12"]


# ---------------------------------------------------------------------------
# computed items (V7-12.09) — incl. the mutation target
# ---------------------------------------------------------------------------


def test_computed_timeline_temporal_point():
    a = _item("u:81", "joined pottery class", occurred=_iv(2023, 6, 3))
    b = _item("u:97", "workshop day", occurred=_iv(2023, 6, 17))
    res = assemble_pack(
        [a, b], _query(IntentClass.TEMPORAL_POINT), max_tokens=500, limit=10
    )
    assert len(res.computed) == 1
    ci = res.computed[0]
    assert ci.kind == "timeline"
    assert set(ci.inputs) <= {i.ref for i in res.items}
    assert "2023-06-03" in ci.text and "u:81" in ci.text


def test_computed_order_and_duration():
    a = _item("u:1", "first thing", occurred=_iv(2023, 6, 3))
    b = _item("u:2", "second thing", occurred=_iv(2023, 6, 17))
    res = assemble_pack(
        [a, b], _query(IntentClass.TEMPORAL_ORDER), max_tokens=500, limit=10
    )
    assert res.computed[0].kind == "order"
    assert "before" in res.computed[0].text

    res = assemble_pack(
        [a, b], _query(IntentClass.DURATION), max_tokens=500, limit=10
    )
    assert res.computed[0].kind == "duration"
    assert "14 days" in res.computed[0].text
    assert set(res.computed[0].inputs) == {"u:1", "u:2"}


def test_computed_count_aggregate():
    items = [_item(f"u:{i}", f"trip {i}") for i in range(3)]
    res = assemble_pack(
        items, _query(IntentClass.COUNT_AGGREGATE), max_tokens=500, limit=10
    )
    assert res.computed[0].kind == "count"
    assert "3 distinct" in res.computed[0].text
    assert set(res.computed[0].inputs) == {"u:0", "u:1", "u:2"}


def test_computed_current_value_with_predecessor():
    sup = _item(
        "u:31",
        "I live in Austin.",
        lifecycle=LifecycleLabel.SUPERSEDED,
        pins={"state_key": "home_city", "state_value": "Austin"},
    )
    cur = _item(
        "u:120",
        "I moved to Denver.",
        pins={
            "state_key": "home_city",
            "state_value": "Denver",
            "valid_from": "2023-09",
        },
    )
    res = assemble_pack(
        [_scored(cur, 0.9), _scored(sup, 0.4)],
        _query(IntentClass.CURRENT_VALUE),
        max_tokens=500,
        limit=10,
    )
    assert res.computed[0].kind == "current_value"
    text = res.computed[0].text
    assert "home_city: Denver" in text
    assert "current since 2023-09" in text
    assert "previously Austin [u:31]" in text
    assert set(res.computed[0].inputs) == {"u:120", "u:31"}


def test_computed_item_never_cites_undelivered_input():
    """The V7-12.09 mutation target: a predecessor ref that was NOT
    delivered must not appear in a computed item's inputs — the ref is
    dropped or the item is dropped, never cited."""
    cur = _item(
        "u:120",
        "I moved to Denver.",
        pins={"state_key": "home_city", "state_value": "Denver"},
    )
    sup = _item(
        "u:31",
        "I live in Austin. " + "pad " * 60,
        lifecycle=LifecycleLabel.SUPERSEDED,
        pins={"state_value": "Austin"},  # no state_key -> no atomic group
    )
    res = assemble_pack(
        [_scored(cur, 0.9), _scored(sup, 0.4)],
        _query(IntentClass.CURRENT_VALUE),
        max_tokens=40,  # enough for cur, not for the padded superseded one
        limit=10,
    )
    delivered = {i.ref for i in res.items}
    assert "u:31" not in delivered
    for ci in res.computed:
        assert set(ci.inputs) <= delivered
        assert "[u:31]" not in ci.text


def test_missing_fields_stay_none():
    bare = PackItemV7(ref="u:1", unit_id="1", quote=b"hi")
    res = assemble_pack([bare], _query(), max_tokens=50, limit=5)
    it = res.items[0]
    assert it.speaker is None and it.recorded_at is None
    assert it.occurred is None and it.session is None
    assert it.perspective is None
    view = render_reader_view(res)
    assert "- [u:1] unknown: hi" in view


def test_computed_items_direct_api_asserts_delivered():
    a = _item("u:1", "thing one", occurred=_iv(2023, 6, 3))
    out = computed_items(_query(IntentClass.TEMPORAL_POINT), [a])
    assert out and out[0].inputs == ("u:1",)
    # no dated items -> no timeline emitted (nothing to support it)
    b = _item("u:2", "undated thing")
    assert computed_items(_query(IntentClass.TEMPORAL_POINT), [b]) == []


def test_computed_items_none_for_other_intents():
    a = _item("u:1", "thing one", occurred=_iv(2023, 6, 3))
    assert computed_items(_query(IntentClass.LOOKUP), [a]) == []
    assert computed_items(_query(IntentClass.PREFERENCE), [a]) == []


# ---------------------------------------------------------------------------
# reader view — pack_render/v1 (V7-12.10/12.13/12.14)
# ---------------------------------------------------------------------------


def _full_pack():
    sup_unit = _item(
        "u:40", "I go hiking every weekend.", session=_SESS_A,
        recorded="2023-06-01T10:00:00",
    )
    obs = _item(
        "o:12",
        "Jordan prefers hiking over cycling",
        derived=True,
        proof_count=3,
        pins={"supports": [sup_unit], "support_refs": ["u:40"]},
    )
    a = _item(
        "u:81",
        "I finally joined the pottery class yesterday!",
        session=_SESS_A,
        recorded="2023-06-04T10:00:00",
        occurred=_iv(2023, 6, 3),
    )
    nb = _item(
        "u:82",
        "That's great — how was the first session?",
        speaker="Assistant",
        session=_SESS_A,
        recorded="2023-06-04T10:01:00",
    )
    b = _item(
        "u:97",
        "The workshop yesterday was exhausting but fun.",
        session=_SESS_B,
        recorded="2023-06-18T10:00:00",
        occurred=_iv(2023, 6, 17),
    )
    res = assemble_pack(
        [_scored(obs, 0.95), _scored(a, 0.8), _scored(b, 0.6)],
        _query(IntentClass.TEMPORAL_ORDER, qt=_us(2024, 2, 10)),
        max_tokens=2000,
        limit=10,
        neighbor_window=1,
        neighbors_fn=lambda i, n: [nb] if i.ref == "u:81" else [],
    )
    return res


def test_reader_view_matches_spec_structure():
    view = render_reader_view(_full_pack())
    lines = view.split("\n")

    assert lines[0] == "# MEMORY (as of 2024-02-10)"
    # section order: COMPUTED -> FACTS -> EVIDENCE -> CONFLICTS -> MISSING
    idx = {s: lines.index(s) for s in
           ("## COMPUTED", "## FACTS", "## EVIDENCE", "## CONFLICTS",
            "## MISSING")}
    assert idx["## COMPUTED"] < idx["## FACTS"] < idx["## EVIDENCE"]
    assert idx["## EVIDENCE"] < idx["## CONFLICTS"] < idx["## MISSING"]

    # computed line shape: `- <kind>: <text>`
    cidx = idx["## COMPUTED"] + 1
    assert re.match(r"- (timeline|count|current_value|order|duration): ",
                    lines[cidx])

    # facts: derived item with proof + ← support refs
    fidx = idx["## FACTS"] + 1
    assert lines[fidx].startswith("- [o:12] ")
    assert "(proof 3; current)" in lines[fidx]
    assert "← [u:40]" in lines[fidx]

    # session headers carry id, date, speakers (spec's `·` separators)
    heads = [l for l in lines if l.startswith("### session ")]
    assert heads == [
        "### session s-14 · 2023-06-04 · Jordan, Assistant",
        "### session s-15 · 2023-06-18 · Jordan, Assistant",
    ]

    # evidence lines: - [ref] <date> <speaker>: <quote> [(event ...)]
    ev = [l for l in lines if l.startswith("- [u:81]")]
    assert ev == [
        "- [u:81] 2023-06-04 Jordan: I finally joined the pottery "
        "class yesterday! (event 2023-06-03, day)"
    ]
    # context-only neighbor is marked distinctly
    ctx = [l for l in lines if l.startswith("- [u:82]")]
    assert ctx and ctx[0].endswith("(context)")

    assert lines[-2] == "- none"  # MISSING
    assert "- none" in lines[idx["## CONFLICTS"] + 1]


def test_reader_view_byte_identical_across_runs():
    p = _full_pack()
    assert render_reader_view(p) == render_reader_view(p)
    # a second, identical assembly renders identically (V7-12.13)
    assert render_reader_view(p) == render_reader_view(_full_pack())


def test_reader_view_no_internal_scores_or_lanes():
    view = render_reader_view(_full_pack())
    for leaked in ("rrf", "score", "lane", "lex", "dense", "0.95"):
        assert leaked not in view


def test_reader_view_missing_descriptor():
    p = _full_pack()
    p.missing = MissingDescriptor(
        facets={"entity": ("Zebra",), "time": ("2022-01",)},
        note="no support found",
    )
    view = render_reader_view(p)
    midx = view.split("\n").index("## MISSING")
    body = view.split("\n")[midx + 1 : midx + 4]
    assert body[0] == "- entity: Zebra"
    assert body[1] == "- time: 2022-01"
    assert body[2] == "- no support found"


def test_reader_view_empty_pack():
    res = assemble_pack([], _query(qt=_us(2024, 2, 10)), max_tokens=100, limit=5)
    view = render_reader_view(res)
    assert "## COMPUTED" not in view  # optional section absent when empty
    assert "- none" in view
    assert view.startswith("# MEMORY (as of 2024-02-10)\n")


def test_reader_view_truncated_marked():
    it = _item("u:1", "First sentence. Second long sentence goes on and on.")
    res = assemble_pack([it], _query(qt=_us(2024, 2, 10)), max_tokens=8,
                        limit=5)
    view = render_reader_view(res)
    assert "(truncated)" in view
    assert "First sentence." in view


def test_render_unknown_query_time_honest():
    res = assemble_pack([_item("u:1", "x")], _query(), max_tokens=50, limit=5)
    assert render_reader_view(res).startswith("# MEMORY (as of unknown)")
