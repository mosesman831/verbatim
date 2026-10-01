"""V8 verdict tests — ``support_verdict/v3`` (SPEC_V8 §12, scenarios
K68–K73).

Exercises the real machinery in ``verbatim/querying/verdict_v2.py``:

- K68 — ``verdict.premise_speaker`` default OFF (V8-12.01): a cat-5
  question whose groups are all authored by the other speaker stays
  ``ready`` with ``unverified_premise``/``partial`` answerability, and
  every item is delivered.  Speaker attribution here comes through the
  real ``units.speaker_canon`` probe (``_unit_speakers``) on a real
  ``ensure_v7_additive`` schema — not a stubbed ctx.
- K69 — ``insufficient`` is reachable only through the four declared
  structural triggers (V8-12.02): (b) empty pool, (a) unmatched asked
  identifier, (c) fitted calibration, (d) shipped negative evidence.
  Coverage labels never set status.
- K70 — ``answerability`` is a declared ``SearchResult`` field carried
  by ``to_dict``/``asdict`` (the SDK shape) and the MCP
  ``_to_jsonable`` wire payload.
- K71 — the V8-12.04 presupposition verifier reports
  ``contradicted_premise`` only when all three conditions hold; its
  decision record carries the dev TPR/FPR metrics.
- K72 — graph/dense/propagation-only support is ``associative``
  (V8-12.05); such a group labels ``weak`` unless a real-support member
  corroborates it.
- K73 — deadline-cut expansion lanes cannot alter verdict status or
  answerability (V8-12.06): identical verdicts at the 500 ms (graph
  cut) and 2 s (graph complete) budgets when the core lanes complete in
  both runs.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from verbatim.core.types_v7 import (
    BudgetClass,
    GroupVerdict,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    ResultStatus,
    RetrievalPolicyV7,
    ScoredCandidate,
    SupportLabel,
)
from verbatim.memory.facade import _v7_answerability
from verbatim.memory.types import SearchResult
from verbatim.querying import verdict_v2 as vv
from verbatim.storage.schema_v7 import ensure_v7_additive


# ---------------------------------------------------------------------------
# builders — real contract types, same shape as test_verdict_v2
# ---------------------------------------------------------------------------

_POS = [0]


def _term(text: str, channel: str = "text") -> NormTerm:
    _POS[0] += len(text) + 1
    return NormTerm(text, channel, _POS[0] - len(text), _POS[0])


def qv(
    text: str = "",
    *,
    terms=(),
    ids=(),
    intent: IntentClass = IntentClass.LOOKUP,
    ents=(),
    speaker=None,
    facets=(),
) -> QueryViewV7:
    _POS[0] = 0
    t_terms = tuple(_term(t, "text") for t in terms)
    t_ids = tuple(_term(i, "identifier") for i in ids)
    norm = NormAnalysis("norm/v2", t_terms + t_ids, t_ids, text)
    ir = IntentResult(intent, (intent,))
    return QueryViewV7(
        query=text,
        norm=norm,
        intent=ir,
        entity_canons=tuple(ents),
        speaker_canon=speaker,
        facets=tuple(facets),
    )


def cand(
    unit: str,
    text=None,
    *,
    score: float = 0.5,
    source: str = None,
    detail=None,
    signals=None,
    group=None,
    lane=None,
) -> ScoredCandidate:
    d = dict(detail or {})
    if signals:
        d.setdefault("signals", dict(signals))
    if text is not None:
        d.setdefault("text", text)
    if group is not None:
        d.setdefault("group", group)
    if lane is not None:
        # the real pipeline shape — ``detail["lane_ranks"]`` is what
        # fusion/scoring stamps on every ScoredCandidate (V7-10.05).
        d.setdefault("lane_ranks", {lane: 1})
    return ScoredCandidate(
        unit_id=unit,
        source_id=source or f"src-{unit}",
        revision=1,
        score=score,
        score_family="ranking/v7",
        detail=d,
    )


def mk_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    return conn


SCOPE = "scope-v8v"
GEN = 1
T0 = 1_700_000_000_000_000


def add_unit(conn, uid, *, speaker=None, scope=SCOPE, gen=GEN, seq=None):
    """Real ``units`` row — the ``_unit_speakers`` probe reads
    ``speaker_canon`` off this table."""
    conn.execute(
        "INSERT OR REPLACE INTO units (unit_id, source_id, revision,"
        " scope_id, kind, parent_unit_id, session_id, seq, speaker_canon,"
        " perspective, recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            uid,
            f"src-{uid}",
            1,
            scope,
            "turn",
            None,
            f"sess-{uid[:1]}",
            seq,
            speaker,
            "user_stated",
            T0,
            None,
            None,
            "unknown",
            "unknown",
            None,
            None,
            gen,
        ),
    )


def add_mention(conn, uid, canon, *, scope=SCOPE, gen=GEN):
    conn.execute(
        "INSERT INTO entity_mentions (scope_id, canon, unit_id,"
        " generation, surface, byte_start, byte_end, role)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (scope, canon, uid, gen, canon, 0, 4, "mention"),
    )


def mk_ctx(conn=None, *, manifest=None, scope=SCOPE, gen=GEN, **extra):
    """A real ``LaneContextV7`` — the shape ``classify_groups`` probes for
    flags, cut lanes, and the pinned-read speaker/mention SQL."""
    man = dict(manifest or {})
    return LaneContextV7(
        store=conn,
        scope_id=scope,
        generation=gen,
        eligible=lambda row: True,
        query_time_us=T0,
        profile="test",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7",
            profile="test",
            lanes=(LaneName.LEX,),
            lane_weights={},
        ),
        manifest=man,
    )


def _run(items, query, ctx=None, calibration=None):
    groups = vv.classify_groups(items, query, ctx)
    report = vv.result_verdict(groups, query, calibration, ctx=ctx)
    return groups, report


def _member(groups, unit):
    for g in groups:
        for md in g.detail.get("member_details") or ():
            if md.get("unit_id") == unit:
                return md
    raise AssertionError(f"no member detail for {unit!r}")


# ===========================================================================
# K68 — verdict.premise_speaker default OFF (V8-12.01)
# ===========================================================================


class TestK68PremiseSpeakerDefaultOff:
    """A cat-5 question — 'what did <speaker> say about X' — whose
    evidence groups are all authored by the OTHER speaker.  With the
    flag at its shipped default (off) the mismatch is advisory only:
    status stays ready, answerability reports the premise doubt, and
    every candidate is still delivered."""

    def _cat5(self, conn, *, flag=None):
        """Two groups, both authored by ``melanie`` in the real units
        table; the asked subject is ``caroline``."""
        for uid in ("u1", "u2"):
            add_unit(conn, uid, speaker="melanie")
        manifest = {}
        if flag is not None:
            manifest["premise_speaker"] = flag
        ctx = mk_ctx(conn, manifest=manifest)
        q = qv(
            "what did caroline say about the painting",
            terms=("caroline", "say", "about", "painting"),
            speaker="caroline",
        )
        items = [
            cand("u1", "the painting was finished on friday",
                 signals={"lexical": 3.1}, lane="lex", group="sess-u"),
            cand("u2", "the painting needed two coats",
                 signals={"lexical": 2.4}, lane="lex", group="sess-u"),
        ]
        return _run(items, q, ctx)

    def test_k68_default_off_ready_and_delivered(self):
        conn = mk_conn()
        groups, report = self._cat5(conn)
        # speaker attribution came through the real units probe
        assert _member(groups, "u1")["speaker"] == "melanie"
        # the mismatch is measured and recorded — advisory only
        assert all(
            g.detail["premise_mismatch"] == "speaker" for g in groups
        )
        assert report.status == ResultStatus.READY
        assert report.status_trigger is None
        assert report.triggers == ()
        assert report.answerability in (
            vv.ANSWERABILITY_UNVERIFIED_PREMISE,
            vv.ANSWERABILITY_PARTIAL,
        )
        assert report.detail["premise_speaker"] == "off"
        # every item is delivered — premise doubt never withholds
        delivered = vv.deliverable_groups(groups)
        delivered_ids = {
            u for g in delivered for u in g.detail["members"]
        }
        assert delivered_ids == {"u1", "u2"}

    def test_k68_unverified_premise_when_subject_silent(self):
        """With real support but zero items authored-by or mentioning
        the asked subject, answerability is unverified_premise (§21.9
        row 4) — the exact cat-5 signature."""
        conn = mk_conn()
        groups, report = self._cat5(conn)
        assert report.answerability == (
            vv.ANSWERABILITY_UNVERIFIED_PREMISE
        )
        for g in groups:
            for md in g.detail["member_details"]:
                assert md["subject_role"] is None

    def test_k68_flag_on_fires_trigger_d(self):
        """The flag-on path is preserved verbatim for the ablation
        harness: premise_mismatch + premise_speaker=on contributes
        trigger (d) — measured non-discriminating in D8-02, which is
        why the default ships off."""
        conn = mk_conn()
        groups, report = self._cat5(conn, flag=True)
        assert report.status == ResultStatus.INSUFFICIENT
        assert vv.TRIGGER_NEGATIVE in report.triggers
        assert report.detail["premise_speaker"] == "on"
        # ... and the items still exist, labeled
        assert len(groups) == 1 or all(
            g.detail["members"] for g in groups
        )

    def test_k68_subject_mentioned_lifts_unverified(self):
        """A real member that mentions the asked subject satisfies §21.9
        row 4's 'by or about' clause — answerability drops to
        partial/supported instead of unverified_premise."""
        conn = mk_conn()
        add_unit(conn, "u1", speaker="melanie")
        add_unit(conn, "u2", speaker="caroline")
        add_mention(conn, "u2", "caroline")
        ctx = mk_ctx(conn)
        q = qv(
            "what did caroline say about the painting",
            terms=("caroline", "say", "about", "painting"),
            speaker="caroline",
        )
        items = [
            cand("u1", "the painting was finished on friday",
                 signals={"lexical": 3.1}, lane="lex", group="sess-u"),
            cand("u2", "caroline loved the painting",
                 signals={"lexical": 4.0}, lane="lex", group="sess-u"),
        ]
        groups, report = _run(items, q, ctx)
        assert report.status == ResultStatus.READY
        # u2 is authored by AND mentions the subject
        assert _member(groups, "u2")["subject_role"] is not None
        assert report.answerability != (
            vv.ANSWERABILITY_UNVERIFIED_PREMISE
        )


# ===========================================================================
# K69 — insufficient only via the four structural triggers (V8-12.02)
# ===========================================================================


class TestK69StructuralTriggersOnly:
    def test_k69_b_empty_pool(self):
        q = qv("q", terms=("alpha",), ids=("ID-9",))
        groups, report = _run([], q)
        assert report.status == ResultStatus.INSUFFICIENT
        assert report.status_trigger == vv.TRIGGER_EMPTY
        assert report.triggers == (vv.TRIGGER_EMPTY,)
        assert report.answerability == vv.ANSWERABILITY_NO_EVIDENCE
        assert report.missing is not None

    def test_k69_a_unmatched_identifier(self):
        q = qv("ticket?", ids=("ABC-999",), intent=IntentClass.IDENTIFIER)
        items = [cand("u1", "some other ticket ABC-123 entirely")]
        groups, report = _run(items, q)
        assert report.status == ResultStatus.INSUFFICIENT
        assert report.status_trigger == vv.TRIGGER_IDENTIFIER
        assert report.missing.facets["identifier"] == ("ABC-999",)
        # the group survives labeled — triggers never delete
        assert len(groups) == 1

    def test_k69_c_fitted_calibration(self):
        q = qv("q", terms=("alpha", "beta"))
        items = [cand("u1", "alpha beta", score=0.31)]
        cal = {"threshold": 0.55, "separates": True,
               "precision": 0.9, "recall": 0.8, "n": 200}
        groups, report = _run(items, q, calibration=cal)
        assert report.status == ResultStatus.INSUFFICIENT
        assert report.status_trigger == vv.TRIGGER_CALIBRATED
        assert report.detail["calibration"]["status"] == "fitted"

    def test_k69_d_negative_evidence(self):
        q = qv("when did i visit mars",
               terms=("when", "did", "i", "visit", "mars"),
               intent=IntentClass.ABSTAIN_LIKELY)
        items = [
            cand("u1", "you have never been to mars",
                 detail={"negative_evidence": True},
                 signals={"lexical": 1.0}, lane="lex"),
        ]
        groups, report = _run(items, q)
        assert report.status == ResultStatus.INSUFFICIENT
        assert report.status_trigger == vv.TRIGGER_NEGATIVE

    def test_k69_unfitted_calibration_is_not_a_trigger(self):
        """Absent/unfitted calibration disables trigger (c) entirely —
        a low score is not an abstention reason (V7-11.06)."""
        q = qv("q", terms=("alpha",))
        items = [cand("u1", "alpha", score=0.01)]
        for cal in (None,
                    {"threshold": 0.9, "separates": False},
                    {"threshold": 0.9}):
            groups, report = _run(items, q, calibration=cal)
            assert report.status == ResultStatus.READY
            assert report.triggers == ()

    def test_k69_weak_labels_never_set_status(self):
        """The V8-12.02 core: a pool made entirely of weak/partial
        coverage groups is still ``ready`` — group coverage labels are
        not a structural trigger (D8-26)."""
        q = qv("q", terms=("unfindable", "terms", "here"))
        items = [
            cand("u1", "nothing matching at all", lane="graph",
                 signals={"graph_ppr": 0.7}),
            cand("u2", "unfindable was mentioned once", lane="lex",
                 signals={"lexical": 0.2}),
        ]
        groups, report = _run(items, q)
        assert report.status == ResultStatus.READY
        assert report.status_trigger is None
        assert report.triggers == ()
        # the weak-only pool reports weak_only; a real partial member
        # lifts it to partial — never to insufficient
        assert report.answerability in (
            vv.ANSWERABILITY_WEAK_ONLY,
            vv.ANSWERABILITY_PARTIAL,
        )

    def test_k69_trigger_order_b_a_d_c(self):
        """Evaluation order is declared (b → a → d → c); the first
        firing trigger is ``status_trigger`` and all firings are in
        ``triggers`` (V8-20.03)."""
        q = qv("id?", ids=("MISS-1",), intent=IntentClass.IDENTIFIER)
        # identifier unmatched AND calibration fitted-low → a and c both
        # fire; status_trigger names the first in order.
        items = [cand("u1", "nothing", score=0.01)]
        cal = {"threshold": 0.9, "separates": True}
        groups, report = _run(items, q, calibration=cal)
        assert report.status == ResultStatus.INSUFFICIENT
        assert vv.TRIGGER_IDENTIFIER in report.triggers
        assert vv.TRIGGER_CALIBRATED in report.triggers
        assert report.status_trigger == vv.TRIGGER_IDENTIFIER
        assert "trigger=" in report.missing.note
        assert "calibration=fitted" in report.missing.note


# ===========================================================================
# K70 — answerability on SearchResult, MCP output, SDK (V8-12.03/20.02)
# ===========================================================================


class TestK70AnswerabilitySurface:
    def test_k70_declared_field_and_to_dict(self):
        """``answerability`` is a declared ``SearchResult`` field so the
        SDK-facing serialization (``to_dict`` → ``asdict``) carries it —
        a dynamic attribute would be invisible on the wire."""
        names = {f.name for f in dataclasses.fields(SearchResult)}
        assert "answerability" in names
        r = SearchResult(answerability=vv.ANSWERABILITY_PARTIAL)
        assert r.answerability == "partial"
        d = r.to_dict()
        assert d["answerability"] == "partial"
        assert asdict(r)["answerability"] == "partial"

    def test_k70_default_none_never_fabricated(self):
        r = SearchResult()
        assert r.answerability is None
        assert r.to_dict()["answerability"] is None

    def test_k70_v7_answerability_extraction(self):
        """The facade's extractor reads the pipeline's coverage.verdict
        block (and the direct-attribute path) — the two shapes the
        pipeline actually produces."""
        v7 = SimpleNamespace(
            coverage={"verdict": {
                "answerability": vv.ANSWERABILITY_UNVERIFIED_PREMISE}},
            explain=None,
        )
        assert _v7_answerability(v7) == "unverified_premise"
        v7b = SimpleNamespace(
            answerability=vv.ANSWERABILITY_WEAK_ONLY,
            coverage=None, explain=None,
        )
        assert _v7_answerability(v7b) == "weak_only"
        assert _v7_answerability(None) is None
        assert _v7_answerability(SimpleNamespace()) is None

    def test_k70_mcp_recall_payload(self):
        """``_tool_v5_recall`` serializes via ``_to_jsonable`` (asdict)
        and ``_summary_recall_payload`` — both must keep answerability
        at top level (summary mode only strips ``coverage``)."""
        from verbatim.api_v3.mcp import (
            _summary_recall_payload,
            _to_jsonable,
        )
        r = SearchResult(
            status="ready",
            answerability=vv.ANSWERABILITY_SUPPORTED,
            coverage={"verdict": {"answerability": "supported"}},
        )
        data = _to_jsonable(r)
        assert data["answerability"] == "supported"
        slim = _summary_recall_payload(data)
        assert slim["answerability"] == "supported"
        assert "coverage" not in slim

    def test_k70_verdict_report_block_shape(self):
        """The coverage.verdict block (V8-20.03) the pipeline grafts —
        the values the facade extractor consumes."""
        q = qv("q", terms=("alpha", "beta"))
        items = [cand("u1", "alpha beta", lane="lex",
                      signals={"lexical": 2.0})]
        groups, report = _run(items, q)
        block = report.as_dict()
        assert block["verdict"] == vv.VERDICT_VERSION
        assert block["status"] == report.status.value
        assert block["answerability"] == report.answerability
        assert block["premise_speaker"] == "off"
        assert "deadline_cut_lanes" in block
        # tuple-compatible with the V7 two-tuple contract
        status, missing = report
        assert status == report.status and missing == report.missing


# ===========================================================================
# K71 — presupposition verifier, three conditions (V8-12.04)
# ===========================================================================


class TestK71PresuppositionVerifier:
    """The verifier reports ``contradicted_premise`` only when ALL of:
    (i) exactly one asked subject canon; (ii) the asked predicate/object
    is asserted by a *different* canon among verdict-evidence items;
    (iii) zero verdict-evidence items by/about the subject match that
    predicate/object."""

    def _query(self, **kw):
        # Six asked terms; the asserting item's text covers only
        # "deployed v2" (~0.33) so its group labels PARTIAL — the
        # no-SUPPORTED guard on trigger (d) stays open and a shipped
        # verifier finding can actually land.
        q = qv(
            "who deployed v2 release notes please",
            terms=("who", "deployed", "v2", "release", "notes",
                   "please"),
            speaker=kw.pop("speaker", "caroline"),
            ents=kw.pop("ents", ()),
        )
        # asked relation rides as an advisory attribute — the frozen
        # dataclass accepts it via object.__setattr__
        rel = kw.pop("relation", {"predicate": "deployed",
                                  "object": "v2"})
        if rel is not None:
            object.__setattr__(q, "asked_relation", rel)
        return q

    def _asserting(self, unit, by, *, lane="lex", text="deployed v2 ok"):
        return cand(
            unit, text,
            detail={
                "predicate": "deployed",
                "object_canon": "v2",
                "asserted_by": by,
            },
            signals={"lexical": 2.0},
            lane=lane,
        )

    def _ship_ctx(self, ship, conn=None, **man):
        manifest = {"verifier_ship": ship}
        manifest.update(man)
        return mk_ctx(conn, manifest=manifest)

    def test_k71_all_conditions_contradicted_answerability(self):
        q = self._query()
        items = [self._asserting("u1", "melanie")]
        groups, report = _run(items, q, self._ship_ctx("answerability"))
        ver = report.detail["verifier"]
        assert ver["shipped"] == "answerability"
        assert ver["conditions"] == {
            "one_subject": True,
            "predicate_object_match": True,
            "subject_silent": True,
        }
        assert ver["contradiction"] is True
        assert report.answerability == (
            vv.ANSWERABILITY_CONTRADICTED_PREMISE
        )
        # ship=answerability never withholds
        assert report.status == ResultStatus.READY

    def test_k71_ship_status_sets_trigger_d(self):
        q = self._query()
        items = [self._asserting("u1", "melanie")]
        groups, report = _run(items, q, self._ship_ctx("status"))
        assert report.detail["verifier"]["contradiction"] is True
        assert report.status == ResultStatus.INSUFFICIENT
        assert vv.TRIGGER_NEGATIVE in report.triggers

    def test_k71_ship_off_reports_but_never_lands(self):
        q = self._query()
        items = [self._asserting("u1", "melanie")]
        groups, report = _run(items, q, self._ship_ctx("off"))
        ver = report.detail["verifier"]
        # measured and reported — just not shipped anywhere
        assert ver["shipped"] == "off"
        assert ver["contradiction"] is True
        assert report.status == ResultStatus.READY
        assert report.answerability != (
            vv.ANSWERABILITY_CONTRADICTED_PREMISE
        )

    def test_k71_fails_without_single_subject(self):
        """Condition (i): two asked canons → no 'exactly one subject',
        contradiction can never fire."""
        q = self._query(speaker=None, ents=("caroline", "melanie"))
        items = [self._asserting("u1", "melanie")]
        groups, report = _run(items, q, self._ship_ctx("status"))
        ver = report.detail["verifier"]
        assert ver["conditions"]["one_subject"] is False
        assert ver["contradiction"] is False
        assert report.status == ResultStatus.READY

    def test_k71_fails_without_relation_assertion(self):
        """Condition (ii): no verdict-evidence item asserts the asked
        (predicate, object) under a different canon."""
        q = self._query()
        items = [
            cand("u1", "unrelated words entirely",
                 detail={"predicate": "bought", "object_canon": "v2",
                         "asserted_by": "melanie"},
                 signals={"lexical": 2.0}, lane="lex"),
        ]
        groups, report = _run(items, q, self._ship_ctx("status"))
        ver = report.detail["verifier"]
        assert ver["conditions"]["predicate_object_match"] is False
        assert ver["contradiction"] is False
        assert report.status == ResultStatus.READY

    def test_k71_fails_when_subject_not_silent(self):
        """Condition (iii): the asked subject's own assertion of the
        same predicate/object defeats the contradiction."""
        q = self._query()
        items = [
            self._asserting("u1", "melanie"),
            self._asserting("u2", "caroline"),
        ]
        groups, report = _run(items, q, self._ship_ctx("status"))
        ver = report.detail["verifier"]
        assert ver["conditions"]["subject_silent"] is False
        assert ver["contradiction"] is False
        assert report.status == ResultStatus.READY

    def test_k71_cut_lane_assertions_do_not_feed_verifier(self):
        """V8-12.06 inside the verifier: an assertion produced solely
        by a deadline-cut lane is not verdict evidence — it cannot
        manufacture condition (ii)."""
        q = self._query()
        items = [
            self._asserting("u1", "melanie", lane="graph"),
            cand("u2", "some topical words", lane="lex",
                 signals={"lexical": 1.0}),
        ]
        ctx = self._ship_ctx("status", deadline_cut_lanes=["graph"])
        groups, report = _run(items, q, ctx)
        ver = report.detail["verifier"]
        assert _member(groups, "u1")["verdict_evidence"] is False
        assert ver["conditions"]["predicate_object_match"] is False
        assert ver["contradiction"] is False
        assert report.status == ResultStatus.READY

    def test_k71_decision_record_carries_dev_metrics(self):
        """The §12 exit decision needs dev TPR/FPR on the record —
        ``ctx.verifier_metrics``/``manifest['verdict.verifier']``
        pass straight through into the verifier block."""
        q = self._query()
        items = [self._asserting("u1", "melanie")]
        ctx = self._ship_ctx(
            "answerability",
            verifier_metrics={
                "tpr": 0.93, "fpr": 0.04, "n": 990,
                "decision": "ship=answerability",
            },
        )
        groups, report = _run(items, q, ctx)
        ver = report.detail["verifier"]
        assert ver["tpr"] == 0.93
        assert ver["fpr"] == 0.04
        assert ver["n"] == 990
        assert ver["decision"] == "ship=answerability"


# ===========================================================================
# K72 — associative support labels weak unless corroborated (V8-12.05)
# ===========================================================================


class TestK72AssociativeSupport:
    def test_k72_graph_only_is_associative_and_weak(self):
        """A graph-only candidate — PPR signal, graph lane, no own-text
        match — is ``associative`` support; its group labels weak."""
        q = qv("q", terms=("alpha", "beta", "gamma", "delta"))
        items = [
            cand("u1", "completely unrelated words here",
                 lane="graph", signals={"graph_ppr": 0.91}),
        ]
        groups, report = _run(items, q)
        md = _member(groups, "u1")
        assert md["support"] == "associative"
        assert groups[0].label == SupportLabel.WEAK
        assert groups[0].detail["real_support_members"] == ()
        assert report.status == ResultStatus.READY
        assert report.answerability == vv.ANSWERABILITY_WEAK_ONLY

    def test_k72_dense_only_is_associative(self):
        q = qv("q", terms=("alpha", "beta", "gamma", "delta"))
        items = [
            cand("u1", "completely unrelated words here",
                 lane="dense", signals={"similarity": 0.77}),
        ]
        groups, _ = _run(items, q)
        assert _member(groups, "u1")["support"] == "associative"
        assert groups[0].label == SupportLabel.WEAK
        assert groups[0].detail["soft_signal"] is True

    def test_k72_propagation_only_is_associative(self):
        q = qv("q", terms=("alpha", "beta", "gamma", "delta"))
        items = [
            cand("u1", "completely unrelated words here",
                 lane="prop", signals={"propagated": 1.0}),
        ]
        groups, _ = _run(items, q)
        assert _member(groups, "u1")["support"] == "associative"
        assert groups[0].label == SupportLabel.WEAK

    def test_k72_real_member_corroborates_group(self):
        """One real-support member rescues the group's label — the
        graph-only sibling is still marked associative (per-member
        support is honest), but the group is no longer weak."""
        q = qv("q", terms=("alpha", "beta", "gamma", "delta"))
        items = [
            cand("u1", "completely unrelated words here",
                 lane="graph", signals={"graph_ppr": 0.91},
                 group="sess-9"),
            cand("u2", "alpha beta gamma delta all matched",
                 lane="lex", signals={"lexical": 3.3},
                 group="sess-9"),
        ]
        groups, report = _run(items, q)
        assert len(groups) == 1
        g = groups[0]
        assert _member(groups, "u1")["support"] == "associative"
        assert _member(groups, "u2")["support"] == "real"
        assert g.detail["real_support_members"] == ("u2",)
        assert g.label in (SupportLabel.SUPPORTED, SupportLabel.PARTIAL)
        assert g.label != SupportLabel.WEAK
        assert report.answerability == vv.ANSWERABILITY_SUPPORTED

    def test_k72_corroboration_is_per_group(self):
        """A real member in a DIFFERENT group does not rescue the
        graph-only group — corroboration is scoped to the group."""
        q = qv("q", terms=("alpha", "beta", "gamma", "delta"))
        items = [
            cand("u1", "completely unrelated words here",
                 lane="graph", signals={"graph_ppr": 0.91},
                 group="sess-a"),
            cand("u2", "alpha beta gamma delta all matched",
                 lane="lex", signals={"lexical": 3.3},
                 group="sess-b"),
        ]
        groups, _ = _run(items, q)
        by_key = {g.group_key: g for g in groups}
        assert by_key["grp:sess-a"].label == SupportLabel.WEAK
        assert by_key["grp:sess-b"].label in (
            SupportLabel.SUPPORTED, SupportLabel.PARTIAL
        )


# ===========================================================================
# K73 — deadline-cut lanes cannot alter the verdict (V8-12.06)
# ===========================================================================


class TestK73DeadlineCutLanes:
    def _items(self):
        return [
            cand("u_lex", "alpha beta gamma delta all matched",
                 lane="lex", signals={"lexical": 3.3},
                 group="sess-1"),
            cand("u_graph", "completely unrelated words here",
                 lane="graph", signals={"graph_ppr": 0.91},
                 group="sess-2"),
        ]

    def _query(self):
        return qv("q", terms=("alpha", "beta", "gamma", "delta"))

    def test_k73_identical_verdict_cut_vs_complete(self):
        """The 500 ms run (graph lane deadline-cut) and the 2 s run
        (graph complete) produce identical status and answerability —
        expansion-lane evidence never moves the verdict."""
        q = self._query()
        items = self._items()

        ctx_fast = mk_ctx(manifest={"deadline_cut_lanes": ["graph"]})
        g_fast, r_fast = _run(items, q, ctx_fast)
        g_full, r_full = _run(items, q, mk_ctx())

        assert r_fast.status == r_full.status == ResultStatus.READY
        assert r_fast.answerability == r_full.answerability

        # the cut member is still delivered (group members), just not
        # verdict evidence
        md = _member(g_fast, "u_graph")
        assert md["verdict_evidence"] is False
        fast_group = next(
            g for g in g_fast if "u_graph" in g.detail["members"]
        )
        assert "u_graph" not in fast_group.detail["evidence_members"]
        assert _member(g_full, "u_graph")["verdict_evidence"] is True

    def test_k73_cut_lane_declared_in_report(self):
        q = self._query()
        groups, report = _run(
            self._items(), q,
            mk_ctx(manifest={"deadline_cut_lanes": ["graph"]}),
        )
        assert tuple(report.detail["deadline_cut_lanes"]) == ("graph",)
        vi = groups[0].detail["verdict_inputs"]
        assert tuple(vi["deadline_cut_lanes"]) == ("graph",)

    def test_k73_all_cut_is_empty_pool(self):
        """When EVERY delivered candidate came from cut lanes the
        eligible-evidence pool is empty — trigger (b), per the
        V8-12.06 empty-pool clause."""
        q = self._query()
        items = [
            cand("u1", "alpha beta", lane="graph",
                 signals={"graph_ppr": 0.9}),
        ]
        groups, report = _run(
            items, q, mk_ctx(manifest={"deadline_cut_lanes": ["graph"]}),
        )
        assert _member(groups, "u1")["verdict_evidence"] is False
        assert report.status == ResultStatus.INSUFFICIENT
        assert report.status_trigger == vv.TRIGGER_EMPTY
        # but the item is still delivered
        assert groups[0].detail["members"] == ["u1"]

    def test_k73_cut_does_not_hide_real_support(self):
        """A multi-lane candidate (lex + graph) keeps verdict evidence
        when only the graph lane was cut — 'produced solely by' is the
        exclusion rule."""
        q = self._query()
        items = [
            cand("u1", "alpha beta gamma delta all matched",
                 detail={"lanes": {"lex": 1, "graph": 2}},
                 signals={"lexical": 3.3, "graph_ppr": 0.5}),
        ]
        groups, report = _run(
            items, q, mk_ctx(manifest={"deadline_cut_lanes": ["graph"]}),
        )
        assert _member(groups, "u1")["verdict_evidence"] is True
        assert report.status == ResultStatus.READY
