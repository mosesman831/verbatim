"""Decision backend tests: rules abstention, fake determinism, Jev adapter.

The Jev tests inject a fake transport — no real network is ever contacted.
"""

from __future__ import annotations

import math
import socket
from decimal import Decimal

import pytest

from verbatim.config import JudgeConfig, VerbatimConfig
from verbatim.core.types import (
    DecisionRequest,
    ErrorCode,
    Mode,
    TaskKind,
    VerbatimError,
    json_dumps,
)
from verbatim.decisions.backend import (
    BackendRegistry,
    FakeBackend,
    default_registry,
    request_fingerprint,
)
from verbatim.decisions.jev import (
    JEV_URL,
    MAX_BODY_BYTES,
    JevBackend,
    TransportResponse,
)
from verbatim.decisions.rules import RulesBackend


def _req(task, state, labels=("a", "b"), purpose="candidate_curation"):
    return DecisionRequest(
        task=task,
        state=state,
        allowed_labels=tuple(labels),
        deadline_s=0.0,
        policy_epoch=0,
        purpose=purpose,
    )


# ------------------------------------------------------------- rules backend


def test_rules_capabilities_and_availability():
    rb = RulesBackend()
    assert rb.available() is True
    caps = rb.capabilities()
    assert caps["calibrated"] is False
    assert caps["backend"] == "rules"
    assert caps["model_revision"] is None
    assert set(caps["tasks"]) == {t.value for t in TaskKind}


def test_rules_durability_remember_marker():
    rb = RulesBackend()
    r = rb.evaluate(_req(TaskKind.DURABILITY, {"text": "Remember that I use Neovim."}))
    assert r.abstained is False
    assert r.outcome["outcome"] == "rule_match"
    assert r.outcome["label"] == "durable"
    assert r.outcome["rule"] == "durability_remember_marker"


def test_rules_durability_declarative_whitelist():
    rb = RulesBackend()
    r = rb.evaluate(_req(TaskKind.DURABILITY, {"text": "My birthday is 1990-04-01."}))
    assert r.abstained is False
    assert r.outcome["label"] == "durable"


def test_rules_durability_abstains_without_marker():
    rb = RulesBackend()
    r = rb.evaluate(_req(TaskKind.DURABILITY, {"text": "It rained a lot yesterday."}))
    assert r.abstained is True
    # never an invented probability
    assert "prob" not in json_dumps(r.outcome)


def test_rules_pair_relation_flags():
    rb = RulesBackend()
    base = {"subject_equal": True, "predicate_equal": True}

    def label(flags):
        r = rb.evaluate(_req(TaskKind.PAIR_RELATION, {"pairs": [flags]}))
        return r

    r = label({**base, "object_equal": False, "condition_overlap": True})
    assert r.outcome["labels"] == ["incompatible"]
    r = label({**base, "object_equal": True, "condition_overlap": True})
    assert r.outcome["labels"] == ["equivalent"]
    r = label({**base, "object_equal": False, "condition_overlap": False})
    assert r.outcome["labels"] == ["different_scope"]
    # different subject → cannot decide from flags → abstain
    r = label({"subject_equal": False, "predicate_equal": True})
    assert r.abstained is True
    assert r.outcome["labels"] == [None]


def test_rules_change_signal():
    rb = RulesBackend()
    cases = {
        "I switched to Neovim last week.": "states_change",
        "I no longer use VS Code.": "states_change",
        "I was wrong about the meeting time.": "states_correction",
        "I use VS Code for work.": "neither",
    }
    for text, want in cases.items():
        r = rb.evaluate(_req(TaskKind.CHANGE_SIGNAL, {"text": text}))
        assert r.outcome["label"] == want, text
        assert r.abstained is False


def test_rules_relevance_overlap():
    rb = RulesBackend()
    r = rb.evaluate(
        _req(TaskKind.RELEVANCE, {"query": "do you use vim", "text": "I use vim daily"})
    )
    assert r.abstained is False
    assert r.outcome["rule"] == "rule_overlap"
    assert r.outcome["label"] == "relevant"
    assert r.outcome["overlap"] == 0.5  # {use, vim} of {do, you, use, vim}
    r2 = rb.evaluate(
        _req(TaskKind.RELEVANCE, {"query": "quantum physics lecture", "text": "cats are nice"})
    )
    assert r2.abstained is True


def test_rules_importance():
    rb = RulesBackend()
    r = rb.evaluate(_req(TaskKind.IMPORTANCE, {"text": "This is important to me."}))
    assert r.abstained is False and r.outcome["label"] == "important"
    r2 = rb.evaluate(_req(TaskKind.IMPORTANCE, {"text": "had lunch today"}))
    assert r2.abstained is True


def test_rules_condition_match():
    rb = RulesBackend()
    cond = {"op": "all", "children": [{"op": "eq", "key": "ctx", "value": "work"}]}
    r = rb.evaluate(
        _req(TaskKind.CONDITION_MATCH, {"condition": cond, "attributes": {"ctx": "work"}})
    )
    assert r.outcome["label"] == "match"
    r = rb.evaluate(
        _req(TaskKind.CONDITION_MATCH, {"condition": cond, "attributes": {"ctx": "home"}})
    )
    assert r.outcome["label"] == "no_match"
    r = rb.evaluate(_req(TaskKind.CONDITION_MATCH, {"condition": cond, "attributes": {}}))
    assert r.abstained is True


# --------------------------------------------------------------- fake/registry


def test_fake_backend_fixture_and_abstain():
    req = _req(TaskKind.DURABILITY, {"text": "x"})
    key = request_fingerprint(req)
    fb = FakeBackend({key: {"label": "durable", "custom": 1}})
    r = fb.evaluate(req)
    assert r.outcome == {"label": "durable", "custom": 1}
    assert fb.calls == [req]
    # unknown request → deterministic abstention, not fabrication
    r2 = fb.evaluate(_req(TaskKind.DURABILITY, {"text": "other"}))
    assert r2.abstained is True


def test_registry_unknown_backend():
    reg = BackendRegistry()
    with pytest.raises(VerbatimError) as ei:
        reg.create("nope", VerbatimConfig(), None, {})
    assert ei.value.code == ErrorCode.CONFIG_INVALID


def test_default_registry_rules():
    reg = default_registry()
    rb = reg.create("rules", VerbatimConfig(), None, {})
    assert isinstance(rb, RulesBackend)


# ------------------------------------------------------------------ jev setup


class FakeEgress:
    """Records authorize/settle ordering; mimics EgressGate's API."""

    def __init__(self, allow=True):
        self.allow = allow
        self.calls: list[tuple] = []

    def check_only(self, scope, processor, purpose):
        self.calls.append(("check_only", processor, purpose))
        return self.allow

    def authorize(self, scope, processor, purpose, est_tokens):
        self.calls.append(("authorize", processor, purpose, est_tokens))
        if not self.allow:
            raise VerbatimError(ErrorCode.EGRESS_DISABLED, "no consent")
        return "rid-1"

    def settle(self, reservation_id, actual_tokens):
        self.calls.append(("settle", reservation_id, actual_tokens))


class FakeTransport:
    """Scriptable transport capturing dispatch args; never touches a socket."""

    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls: list[tuple] = []

    def __call__(self, method, url, headers, body, timeout_s):
        self.calls.append((method, url, dict(headers), body, timeout_s))
        if self.exc is not None:
            raise self.exc
        return self.response


def _jev_cfg():
    return VerbatimConfig(
        mode=Mode.JEV_ASSISTED,
        judge=JudgeConfig(backend="jev", daily_budget_usd=Decimal("1")),
    )


def _jev(egress=None, transport=None, secret="test-secret-key"):
    return JevBackend(
        _jev_cfg(),
        "prof:alice:-:conv1",
        egress or FakeEgress(),
        secret_getter=lambda name: secret,
        http=transport,
    )


def _pair_state(n=1):
    return {
        "pairs": [
            {
                "old": "I use VS Code at work.",
                "new": "I use Neovim for personal projects.",
                "speaker_relation": "same authenticated speaker",
                "time_overlap": "unknown",
            }
            for _ in range(n)
        ]
    }


def _ok_response(model="jev-1.13.0"):
    return TransportResponse(
        200,
        json_dumps(
            {
                "model": model,
                "answers": {
                    "pair_0_relation": {
                        "type": "choice",
                        "label": "different_scope",
                        "probabilities": {
                            "equivalent": 0.05,
                            "compatible": 0.10,
                            "incompatible": 0.05,
                            "different_scope": 0.70,
                            "insufficient_context": 0.10,
                        },
                    },
                    "pair_0_change": {
                        "type": "choice",
                        "label": "neither",
                        "probabilities": {
                            "states_change": 0.2,
                            "states_correction": 0.1,
                            "neither": 0.7,
                        },
                    },
                },
                "usage": {"input_tokens": 100, "output_tokens": 10},
            }
        ).encode(),
    )


# --------------------------------------------------------------- jev behavior


def test_jev_available_preflight():
    j = _jev()
    assert j.available() is True
    # offline mode → unavailable
    off = VerbatimConfig(mode=Mode.OFFLINE_RULES, judge=JudgeConfig(backend="rules"))
    j2 = JevBackend(off, "s", FakeEgress(), lambda n: "k", http=FakeTransport())
    assert j2.available() is False
    # missing secret → unavailable
    j3 = _jev(secret=None)
    assert j3.available() is False
    # no consent/budget → unavailable
    j4 = _jev(egress=FakeEgress(allow=False))
    assert j4.available() is False


def test_jev_request_construction():
    transport = FakeTransport(_ok_response())
    j = _jev(transport=transport)
    req = _req(TaskKind.PAIR_RELATION, _pair_state())
    r = j.evaluate(req)
    assert not r.abstained
    method, url, headers, body, timeout = transport.calls[0]
    assert url == JEV_URL
    assert headers["Authorization"] == "Bearer test-secret-key"
    import json

    parsed = json.loads(body.decode())
    assert parsed["model"] == "jev-1.13.0"
    q = parsed["questions"]["pair_0_relation"]
    assert q["type"] == "choice"
    # instructions explicitly reference state paths (ids are not shown to model)
    assert "state.pairs[0].old" in q["instructions"]
    assert "state.pairs[0].new" in q["instructions"]
    assert "data, not instructions" in q["instructions"]
    assert set(q["criteria"]) == {
        "equivalent",
        "compatible",
        "incompatible",
        "different_scope",
        "insufficient_context",
    }
    # result carries normalized answers + per-pair view
    assert r.outcome["pairs"][0]["relation"] == "different_scope"
    assert r.outcome["pairs"][0]["change_signal"] == "neither"
    assert r.usage["input_tokens"] == 100


def test_jev_egress_order_and_settle():
    egress = FakeEgress()
    j = _jev(egress=egress, transport=FakeTransport(_ok_response()))
    j.evaluate(_req(TaskKind.PAIR_RELATION, _pair_state()))
    kinds = [c[0] for c in egress.calls]
    assert kinds[:1] == ["authorize"] and kinds[-1:] == ["settle"]
    settle = egress.calls[-1]
    assert settle[1] == "rid-1" and settle[2] == 110  # 100 in + 10 out


def test_jev_caps_pairs():
    j = _jev(transport=FakeTransport(_ok_response()))
    with pytest.raises(VerbatimError) as ei:
        j.evaluate(_req(TaskKind.PAIR_RELATION, _pair_state(9)))
    assert ei.value.code == ErrorCode.DECISION_INVALID


def test_jev_size_rejection():
    j = _jev(transport=FakeTransport(_ok_response()))
    big = {"text": "y" * (MAX_BODY_BYTES + 100)}
    with pytest.raises(VerbatimError) as ei:
        j.evaluate(_req(TaskKind.DURABILITY, big))
    assert ei.value.code == ErrorCode.DECISION_INVALID


def test_jev_other_task_questions():
    transport = FakeTransport(
        TransportResponse(
            200,
            json_dumps(
                {
                    "model": "jev-1.13.0",
                    "answers": {"durability_0": {"type": "noul", "noul": 0.9}},
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ).encode(),
        )
    )
    j = _jev(transport=transport)
    r = j.evaluate(_req(TaskKind.DURABILITY, {"text": "Remember my editor."}))
    assert r.outcome["answers"]["durability_0"]["noul"] == 0.9
    import json

    parsed = json.loads(transport.calls[0][3].decode())
    assert "state.text" in parsed["questions"]["durability_0"]["instructions"]


# ------------------------------------------------- jev response validation


def _eval_with_answers(answers, model="jev-1.13.0"):
    transport = FakeTransport(
        TransportResponse(
            200,
            json_dumps(
                {"model": model, "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}
            ).encode(),
        )
    )
    return _jev(transport=transport).evaluate(_req(TaskKind.DURABILITY, {"text": "hi"}))


def test_jev_missing_and_extra_answers():
    r = _eval_with_answers({})
    assert r.abstained and r.outcome["error"] == "DECISION_INVALID"
    r = _eval_with_answers(
        {"durability_0": {"type": "noul", "noul": 0.5}, "ghost": {"type": "noul", "noul": 0.1}}
    )
    assert r.abstained and r.outcome["error"] == "DECISION_INVALID"


def _eval_raw_body(body: bytes):
    transport = FakeTransport(TransportResponse(200, body))
    return _jev(transport=transport).evaluate(_req(TaskKind.DURABILITY, {"text": "hi"}))


def test_jev_wrong_type_and_nonfinite():
    r = _eval_with_answers({"durability_0": {"type": "choice", "label": "x", "probabilities": {}}})
    assert r.abstained
    # 1e999 parses to a non-finite float — must be rejected by validation
    r = _eval_raw_body(
        b'{"model":"jev-1.13.0","answers":{"durability_0":{"type":"noul","noul":1e999}},'
        b'"usage":{"input_tokens":1,"output_tokens":1}}'
    )
    assert r.abstained
    r = _eval_with_answers({"durability_0": {"type": "noul", "noul": 1.5}})
    assert r.abstained


def test_jev_choice_validation():
    # use pair_relation request to exercise choice answers
    def eval_pair(ans):
        transport = FakeTransport(
            TransportResponse(
                200,
                json_dumps(
                    {"model": "jev-1.13.0", "answers": ans, "usage": {"input_tokens": 1, "output_tokens": 1}}
                ).encode(),
            )
        )
        return _jev(transport=transport).evaluate(
            _req(TaskKind.PAIR_RELATION, _pair_state())
        )

    good_rel = {
        "type": "choice",
        "label": "compatible",
        "probabilities": {
            "equivalent": 0.1,
            "compatible": 0.6,
            "incompatible": 0.1,
            "different_scope": 0.1,
            "insufficient_context": 0.1,
        },
    }
    good_chg = {
        "type": "choice",
        "label": "neither",
        "probabilities": {"states_change": 0.1, "states_correction": 0.1, "neither": 0.8},
    }
    # label not in allowed set
    r = eval_pair({"pair_0_relation": {**good_rel, "label": "bogus"}, "pair_0_change": good_chg})
    assert r.abstained
    # probabilities not summing to 1
    bad = dict(good_rel)
    bad["probabilities"] = dict(good_rel["probabilities"], compatible=0.9)
    r = eval_pair({"pair_0_relation": bad, "pair_0_change": good_chg})
    assert r.abstained
    # missing label key in probabilities
    bad2 = dict(good_rel)
    bad2["probabilities"] = {"equivalent": 1.0}
    r = eval_pair({"pair_0_relation": bad2, "pair_0_change": good_chg})
    assert r.abstained
    # argmax inconsistency: label is not the maximum
    bad3 = dict(good_rel)
    bad3["label"] = "incompatible"
    r = eval_pair({"pair_0_relation": bad3, "pair_0_change": good_chg})
    assert r.abstained


def test_jev_model_drift():
    r = _eval_with_answers({"durability_0": {"type": "noul", "noul": 0.5}}, model="jev-9.9.9")
    assert r.abstained is True
    assert r.outcome["error"] == "MODEL_DRIFT"
    assert r.reason == "MODEL_DRIFT"


# ------------------------------------------------------- jev http mapping


def _dispatch_status(status):
    transport = FakeTransport(TransportResponse(status, b"{}"))
    j = _jev(transport=transport)
    try:
        j.evaluate(_req(TaskKind.DURABILITY, {"text": "hi"}))
    except VerbatimError as e:
        return e, j
    raise AssertionError("expected VerbatimError")


def test_jev_http_401_disables_and_403():
    e, j = _dispatch_status(401)
    assert e.code == ErrorCode.REMOTE_AUTH and e.retryable is False
    assert j.auth_disabled is True
    assert j.available() is False
    # subsequent dispatch refused without transport call
    with pytest.raises(VerbatimError):
        j.evaluate(_req(TaskKind.DURABILITY, {"text": "hi"}))
    e2, _ = _dispatch_status(403)
    assert e2.code == ErrorCode.REMOTE_AUTH


def test_jev_http_422():
    e, _ = _dispatch_status(422)
    assert e.code == ErrorCode.DECISION_INVALID


@pytest.mark.parametrize("status", [429, 500, 502, 529])
def test_jev_http_busy(status):
    e, _ = _dispatch_status(status)
    assert e.code == ErrorCode.REMOTE_BUSY and e.retryable is True


def test_jev_redirect_refused():
    e, _ = _dispatch_status(301)
    assert e.code == ErrorCode.DECISION_INVALID and e.retryable is False


def test_jev_timeout_leaves_reservation():
    egress = FakeEgress()
    j = _jev(egress=egress, transport=FakeTransport(exc=socket.timeout()))
    with pytest.raises(VerbatimError) as ei:
        j.evaluate(_req(TaskKind.DURABILITY, {"text": "hi"}))
    assert ei.value.code == ErrorCode.REMOTE_BUSY and ei.value.retryable is True
    # authorize happened; NO settle — reservation left standing (SPEC §27)
    kinds = [c[0] for c in egress.calls]
    assert kinds == ["authorize"]


def test_jev_no_token_in_errors():
    secret = "super-secret-token-value"
    transport = FakeTransport(TransportResponse(500, b"err"))
    j = _jev(transport=transport, secret=secret)
    with pytest.raises(VerbatimError) as ei:
        j.evaluate(_req(TaskKind.DURABILITY, {"text": "hi"}))
    assert secret not in str(ei.value)
    assert secret not in repr(ei.value.to_dict())


def test_jev_egress_denied_no_dispatch():
    transport = FakeTransport(_ok_response())
    j = _jev(egress=FakeEgress(allow=False), transport=transport)
    with pytest.raises(VerbatimError) as ei:
        j.evaluate(_req(TaskKind.DURABILITY, {"text": "hi"}))
    assert ei.value.code == ErrorCode.EGRESS_DISABLED
    assert transport.calls == []  # nothing was sent


# ------------------------------------------- core.policy integration shapes


def _jev_noul_response(noul):
    return TransportResponse(
        200,
        json_dumps(
            {
                "model": "jev-1.13.0",
                "answers": {"durability_0": {"type": "noul", "noul": noul}},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        ).encode(),
    )


def test_jev_flat_pair_state_normalized():
    """core.policy sends {old,new,speaker_relation,time_overlap} — a flat
    single pair. It must be normalized to state.pairs[0] on the wire so
    instructions can address it, and must project a flat outcome label."""
    transport = FakeTransport(_ok_response())
    j = _jev(transport=transport)
    req = _req(
        TaskKind.PAIR_RELATION,
        {
            "old": "I use VS Code at work.",
            "new": "I use Neovim for personal projects.",
            "speaker_relation": "same",
            "time_overlap": "unknown",
        },
    )
    r = j.evaluate(req)
    import json

    wire = json.loads(transport.calls[0][3].decode())
    assert wire["state"]["pairs"][0]["old"] == "I use VS Code at work."
    assert wire["state"]["pairs"][0]["time_overlap"] == "unknown"
    assert "state.pairs[0].old" in wire["questions"]["pair_0_relation"]["instructions"]
    # flat projections consumed by policy._pair_label_of / _change_signal_of
    assert r.outcome["label"] == "different_scope"
    assert r.outcome["change_signal"] == "neither"
    assert r.outcome["pairs"][0]["relation"] == "different_scope"


def test_jev_durability_flat_label_threshold():
    """noul > 0.5 → 'durable'; the 0.5 boundary falls to 'not_durable'
    which routes to review, never to silent admission."""
    j = _jev(transport=FakeTransport(_jev_noul_response(0.9)))
    r = j.evaluate(_req(TaskKind.DURABILITY, {"text": "remember my editor"}))
    assert r.outcome["label"] == "durable"
    assert r.outcome["noul"] == 0.9

    j = _jev(transport=FakeTransport(_jev_noul_response(0.5)))
    r = j.evaluate(_req(TaskKind.DURABILITY, {"text": "maybe"}))
    assert r.outcome["label"] == "not_durable"

    j = _jev(transport=FakeTransport(_jev_noul_response(0.1)))
    r = j.evaluate(_req(TaskKind.DURABILITY, {"text": "ok"}))
    assert r.outcome["label"] == "not_durable"


def test_jev_change_signal_flat_label():
    transport = FakeTransport(
        TransportResponse(
            200,
            json_dumps(
                {
                    "model": "jev-1.13.0",
                    "answers": {
                        "change_0": {
                            "type": "choice",
                            "label": "states_change",
                            "probabilities": {
                                "states_change": 0.8,
                                "states_correction": 0.1,
                                "neither": 0.1,
                            },
                        }
                    },
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                }
            ).encode(),
        )
    )
    j = _jev(transport=transport)
    r = j.evaluate(_req(TaskKind.CHANGE_SIGNAL, {"text": "I switched to vim"}))
    assert r.outcome["label"] == "states_change"


def test_jev_pair_missing_old_new_rejected():
    j = _jev(transport=FakeTransport(_ok_response()))
    with pytest.raises(VerbatimError) as ei:
        j.evaluate(
            _req(TaskKind.PAIR_RELATION, {"pairs": [{"old": 1, "new": "x"}]})
        )
    assert ei.value.code == ErrorCode.DECISION_INVALID
    # nothing dispatched, nothing reserved
    j2_transport = FakeTransport(_ok_response())
    j2 = _jev(transport=j2_transport)
    with pytest.raises(VerbatimError):
        j2.evaluate(_req(TaskKind.PAIR_RELATION, {"unrelated": True}))
    assert j2_transport.calls == []
