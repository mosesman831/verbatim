"""Retirement-signal supersession detection (V3-17.x / G3 bound).

Unit coverage for the pattern layer plus store-level proposal tests:
detection creates a ``conflicts_with`` edge + an open ``supersede`` review
with pinned expected versions — it never applies a transition.
"""

from __future__ import annotations

import pytest

from verbatim.evidence.supersession import (
    _content_tokens,
    _mentions_ident,
    _retired_idents,
    propose_retirement_supersessions,
)
from verbatim.storage.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


# ---------------------------------------------------------------------------
# pattern layer
# ---------------------------------------------------------------------------


class TestRetiredIdents:
    def test_subject_form(self):
        assert _retired_idents("swagger-gen was retired in June") == [
            "swagger-gen"
        ]

    def test_object_form(self):
        assert _retired_idents("we deprecated the xmlrpc endpoint") == [
            "xmlrpc"
        ]

    def test_backticked_ident(self):
        assert _retired_idents("`make swagger-gen` is deprecated") == [
            "swagger-gen"
        ]

    def test_no_marker(self):
        assert _retired_idents("use `make swagger-gen` to regenerate") == []

    def test_function_words_never_capture(self):
        assert _retired_idents("retired in June") == []
        assert _retired_idents("we removed it") == []
        assert _retired_idents("removed the flag") == ["flag"]

    @pytest.mark.parametrize(
        "text",
        [
            "check whether `deploy.sh` was retired — it wasn't",
            "confirm that keyd was removed before proceeding",
            "I wonder if pytest-legacy is deprecated",
            "maybe webpack was retired",
            "webpack was reportedly retired",
            "ask the team if swagger-gen was superseded",
            "unsure whether black was replaced",
            "question: was xmlrpc deprecated?",
        ],
    )
    def test_hedged_mentions_never_capture(self, text):
        """Questions, hedges, and hearsay are not retirement assertions —
        the G3 false-supersession bound (§17, G3)."""
        assert _retired_idents(text) == []

    def test_modal_after_boundary_does_not_hedge(self):
        """A modal marker in a prior clause doesn't hedge the assertion
        that follows the sentence boundary."""
        assert _retired_idents(
            "Check the logs. swagger-gen was retired"
        ) == ["swagger-gen"]
        assert _retired_idents(
            "Note: swagger-gen was retired. Check the logs"
        ) == ["swagger-gen"]


class TestNonAssertionForms:
    """Guard classes the G3 deep twin corpus surfaced — each is a
    near-miss that carries no retirement assertion (eval/v3/g3_twins.py
    measures the whole category set at ~460 cases)."""

    @pytest.mark.parametrize(
        "text",
        [
            "If `deploy.sh` is retired, we'll have to deploy differently",
            "Unless `keyd` gets removed, the setup stays",
            "Whenever `xmlrpc` is deprecated we can switch",
            "In case `flake8` was retired, pin ruff",
            "When `swagger-gen` is deprecated we can switch",
        ],
    )
    def test_conditional_never_captures(self, text):
        assert _retired_idents(text) == []

    def test_when_past_tense_still_asserts(self):
        """``when`` + past tense is a temporal assertion, not a
        hypothetical — "when X was retired, we moved" declares the end."""
        assert _retired_idents(
            "When swagger-gen was retired, the team moved to buf"
        ) == ["swagger-gen"]

    @pytest.mark.parametrize(
        "text",
        [
            "`syncv1` was retired last year but reinstated in March",
            "`node-12` was deprecated briefly, then restored",
            "we removed `ftp-upload` but rolled back the change",
            "`mysql-5` was retired — it was later brought back",
        ],
    )
    def test_cancelled_retirement_never_captures(self, text):
        assert _retired_idents(text) == []

    @pytest.mark.parametrize(
        "text",
        [
            "Please retire `deploy.sh` when you get a chance",
            "We should deprecate `flake8` soon",
            "Let's replace `xmlrpc` next sprint",
            "The plan is to deprecate `swagger-gen` once migration finishes",
            "We're going to replace `rest-v2` eventually",
            "At some point we want to remove `ftp-upload`",
            "We will remove `solr-4` next quarter",
        ],
    )
    def test_intent_and_request_never_capture(self, text):
        assert _retired_idents(text) == []

    @pytest.mark.parametrize(
        "text",
        [
            "Rumor says `keyd` was removed",
            "Word is `swagger-gen` was deprecated",
            "They say `mysql-5` was retired",
            "I heard `xmlrpc` was decommissioned",
        ],
    )
    def test_hearsay_attribution_never_captures(self, text):
        assert _retired_idents(text) == []

    @pytest.mark.parametrize(
        "text,ident",
        [
            ("we removed `deploy.sh` last week", "deploy.sh"),
            ("the team retired `keyd`", "keyd"),
            ("we replaced `xmlrpc` with `grpc`", "xmlrpc"),
            ("removed the `old-flag` entirely", "old-flag"),
        ],
    )
    def test_object_form_captures_backticked_idents(self, text, ident):
        """Object-form verbs must capture quoted/backticked identifiers —
        the deep corpus showed this silently dropped ~70% of true pairs."""
        assert ident in _retired_idents(text)

    def test_past_declarative_still_asserts(self):
        assert _retired_idents("we removed `deploy.sh` last week") == [
            "deploy.sh"
        ]
        assert _retired_idents(
            "the team replaced `xmlrpc` with `grpc`"
        ) == ["xmlrpc"]


class TestMentionsIdent:
    @pytest.mark.parametrize(
        "text,ident,want",
        [
            ("`make swagger-gen`", "swagger-gen", True),
            ("swagger-gen.", "swagger-gen", True),
            ("xswagger-gen", "swagger-gen", False),
            ("--force", "force", True),
            ("x-force", "force", False),
            ("swagger-gen2", "swagger-gen", False),
            ("swagger-gen.foo", "swagger-gen", False),
            ("deploy /v1/users now", "/v1/users", True),
        ],
    )
    def test_boundaries(self, text, ident, want):
        assert _mentions_ident(text, ident) is want


# ---------------------------------------------------------------------------
# store-level proposals
# ---------------------------------------------------------------------------


def _seed_claim(store, conn, claim_id, scope_id, source_id, span_id, text,
                state="active"):
    payload = text.encode("utf-8")
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, "u1"),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC','direct_user','{}')",
        (source_id, payload, store.hmac(payload)),
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (span_id, source_id, 1, 0, len(payload), store.hmac(payload)),
    )
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,1,1)",
        (claim_id, scope_id),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,"
        "recorded_from,recorded_until) VALUES(?,1,?,1,NULL)",
        (claim_id, state),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,1,?,'primary',NULL)",
        (claim_id, span_id),
    )


def test_retirement_proposes_supersession(store):
    """A claim declaring an identifier retired supersedes the live claim
    instructing it — proposal only, never application."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes(scope_id,profile_id,principal_id,"
            "workspace_id,conversation_id,visibility,acl_revision)"
            " VALUES('sA','prof','p1','ws','c1','conversation',0)"
        )
        _seed_claim(
            store, conn, "cl-stale", "sA", "src-1", "sp-1",
            "Old runbook: regenerate the API types with `make swagger-gen`.",
        )
        _seed_claim(
            store, conn, "cl-new", "sA", "src-2", "sp-2",
            "Regenerate the API types with `make codegen` — "
            "swagger-gen was retired in June.",
        )
    out = propose_retirement_supersessions(store, "cl-new", "sA")
    assert len(out) == 2  # one edge + one review
    with store.read() as conn:
        rev = conn.execute(
            "SELECT proposed_effect_json, expected_versions_json, state"
            " FROM reviews"
        ).fetchone()
        edge = conn.execute(
            "SELECT edge_type FROM edges"
        ).fetchone()
    assert edge is not None and edge[0] == "conflicts_with"
    assert rev is not None and rev[2] == "open"
    import json

    effect = json.loads(rev[0])
    versions = json.loads(rev[1])
    assert effect["effect"] == "supersede"
    assert effect["predecessor_id"] == "cl-stale"
    assert effect["successor_id"] == "cl-new"
    assert effect["change_signal"] == "retirement"
    assert versions == {"cl-stale": 1, "cl-new": 1}
    # claims untouched — proposal never transitions
    with store.read() as conn:
        states = dict(
            conn.execute(
                "SELECT claim_id, state FROM claim_revisions"
            ).fetchall()
        )
    assert states == {"cl-stale": "active", "cl-new": "active"}


def test_retirement_retry_dedups_proposal(store):
    """A redelivered detection pass resolves to the queued open review —
    the same retirement signal must never stack duplicate proposals."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes(scope_id,profile_id,principal_id,"
            "workspace_id,conversation_id,visibility,acl_revision)"
            " VALUES('sA','prof','p1','ws','c1','conversation',0)"
        )
        _seed_claim(
            store, conn, "cl-stale", "sA", "src-1", "sp-1",
            "Old runbook: regenerate the API types with `make swagger-gen`.",
        )
        _seed_claim(
            store, conn, "cl-new", "sA", "src-2", "sp-2",
            "Regenerate the API types with `make codegen` — "
            "swagger-gen was retired in June.",
        )
    first = propose_retirement_supersessions(store, "cl-new", "sA")
    second = propose_retirement_supersessions(store, "cl-new", "sA")
    assert len(first) == 2
    assert second == first
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1


def test_retirement_resolved_review_allows_reproposal(store):
    """Dedup covers open reviews only — after a decision, the same signal
    may be re-proposed (e.g. a rejection the operator wants revisited)."""
    from verbatim.storage.repos import ReviewsRepo

    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes(scope_id,profile_id,principal_id,"
            "workspace_id,conversation_id,visibility,acl_revision)"
            " VALUES('sA','prof','p1','ws','c1','conversation',0)"
        )
        _seed_claim(
            store, conn, "cl-stale", "sA", "src-1", "sp-1",
            "Old runbook: regenerate the API types with `make swagger-gen`.",
        )
        _seed_claim(
            store, conn, "cl-new", "sA", "src-2", "sp-2",
            "Regenerate the API types with `make codegen` — "
            "swagger-gen was retired in June.",
        )
    first = propose_retirement_supersessions(store, "cl-new", "sA")
    with store.tx() as conn:
        ReviewsRepo(store).resolve(conn, first[-1], "rejected", 9)
    second = propose_retirement_supersessions(store, "cl-new", "sA")
    assert len(second) == 2
    assert second[1] != first[1]
    with store.read() as conn:
        open_count = conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE state = 'open'"
        ).fetchone()[0]
    assert open_count == 1


def test_bare_mention_never_supersedes(store):
    """Sharing an identifier without a retirement marker produces nothing
    (the G3 false-supersession bound is measured against this guard)."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes(scope_id,profile_id,principal_id,"
            "workspace_id,conversation_id,visibility,acl_revision)"
            " VALUES('sA','prof','p1','ws','c1','conversation',0)"
        )
        _seed_claim(
            store, conn, "cl-a", "sA", "src-1", "sp-1",
            "Regenerate the API types with `make swagger-gen`.",
        )
        _seed_claim(
            store, conn, "cl-b", "sA", "src-2", "sp-2",
            "The swagger-gen output feeds the codegen pipeline for "
            "API types regeneration.",
        )
    assert propose_retirement_supersessions(store, "cl-b", "sA") == []
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_corroborating_claims_never_supersede(store):
    """A counterparty that itself declares the same retirement is
    corroborating evidence, not a contradicted claim."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes(scope_id,profile_id,principal_id,"
            "workspace_id,conversation_id,visibility,acl_revision)"
            " VALUES('sA','prof','p1','ws','c1','conversation',0)"
        )
        _seed_claim(
            store, conn, "cl-a", "sA", "src-1", "sp-1",
            "Note: swagger-gen was retired; regenerate API types with "
            "`make codegen`.",
        )
        _seed_claim(
            store, conn, "cl-b", "sA", "src-2", "sp-2",
            "Confirmed — swagger-gen was retired in June for API types.",
        )
    assert propose_retirement_supersessions(store, "cl-b", "sA") == []


def test_no_shared_topic_never_supersedes(store):
    """Identifier overlap alone is insufficient — the pair must share
    topic vocabulary beyond the retired identifier."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes(scope_id,profile_id,principal_id,"
            "workspace_id,conversation_id,visibility,acl_revision)"
            " VALUES('sA','prof','p1','ws','c1','conversation',0)"
        )
        _seed_claim(
            store, conn, "cl-a", "sA", "src-1", "sp-1",
            "swagger-gen also names a fictional band in the credits.",
        )
        _seed_claim(
            store, conn, "cl-b", "sA", "src-2", "sp-2",
            "Regenerate the API types with `make codegen` — "
            "swagger-gen was retired.",
        )
    assert propose_retirement_supersessions(store, "cl-b", "sA") == []
