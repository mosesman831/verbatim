"""SPEC_V4_5 §15 acceptance scenarios D01–D12.

Real ``Store.create`` fixtures (v3/v4 schema) with direct-SQL seeding —
the same harness shape as ``tests/v4/test_scenarios_*.py`` and
``tests/retrieval/test_disclosure_tiers.py``. Every test exercises a
public production path; no sleeps (explicit timestamps/sequence numbers
throughout); nothing is silently skipped.

Scenario status at authoring time:

- D01, D02 — ``manifest="counterevidence_first"`` is implemented in
  ``verbatim.retrieval.manifest`` and wired through ``recall_v3``;
  these tests exercise the real v4.5 surface.
- D03, D04 — ``verbatim.repair`` implements the impact-plan/executor
  surface (``plan_repair``/``apply_repair``/``full_rebuild``/
  ``invalidate_dependents``); the tests measure real recompute counts
  and real CONTEXT_INCOMPLETE refusals.
- D05 — ``pack_mode="sufficiency"`` is implemented; the token reduction
  is measured, never assumed.
- D06 — revocation-bound expansion denial via ``expand_item``.
- D07, D08 — ``verbatim.procedures.transfer`` implements both arms of
  qualified delivery and honest transfer success.
- D09, D10 — ``verbatim.branches.BranchService`` implements
  snapshot-pinned, review-gated branches; the tests exercise the real
  isolation/apply and purge-propagation contracts.
- D11, D12 — ``verbatim.refresh.RefreshScheduler`` implements
  utility-budgeted planning, owner-priority exemptions, and the
  periodic comparator on the real ``JobQueue``.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    VerbatimError,
    json_dumps,
)
from verbatim.core.types_v3 import (
    EnvelopeKind,
    EnvironmentFingerprint,
    Perspective,
    RecallRequestV3,
    SourceEnvelopeV3,
    TrustClass,
)
from verbatim.branches import BranchService
from verbatim.evidence.envelopes import ingest_envelope
from verbatim.governance import (
    CallerV3,
    create_grant,
    register_principal,
    revoke_grant,
    seed_purposes,
)
from verbatim.jobs.queue import JobQueue
from verbatim.kernel import Kernel
from verbatim.procedures.compiler import environment_map
from verbatim.procedures.exposures import exposures_for
from verbatim.procedures.transfer import (
    deliver_procedure,
    transfer_success,
)
from verbatim.refresh import (
    REFRESH_EVENT_KIND,
    RefreshBudget,
    RefreshScheduler,
)
from verbatim.repair import (
    apply_repair,
    full_rebuild,
    invalidate_dependents,
    plan_repair,
)
from verbatim.retrieval.v3 import recall_v3
from verbatim.retrieval.v3.recall import expand_item
from verbatim.storage.store import Store


_TEST_HMAC_KEY = b"test-hmac-key-padded-to-32-bytes"

# Deterministic logical timestamps — never a wall clock or a sleep.
T0 = 1_700_000_000_000_000


def _h(data: bytes) -> bytes:
    return hmac.new(_TEST_HMAC_KEY, data, hashlib.sha256).digest()


@pytest.fixture
def store(tmp_path):
    (tmp_path / "v45.db.key").write_bytes(_TEST_HMAC_KEY)
    s = Store.create(str(tmp_path / "v45.db"))
    yield s
    s.close()


def _gen(store) -> int:
    return store.projection_generation()


# ---------------------------------------------------------------------------
# seeding helpers (mirror tests/retrieval/test_disclosure_tiers.py)
# ---------------------------------------------------------------------------

def seed_scope(conn, scope_id, principal="p1", conv="c1", profile="prof",
               vis="conversation"):
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,workspace_id,"
        "conversation_id,visibility,acl_revision) VALUES(?,?,?,?,?,?,0)",
        (scope_id, profile, principal, "ws", conv, vis),
    )


def seed_auth(conn, scope_id, pid="human:alice", purposes=("recall",),
              verbs=("read", "quote")):
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    return create_grant(
        conn, scope_id=scope_id, principal_id=pid, verbs=set(verbs),
        issuer_id=pid,
        purposes=None if purposes is None else list(purposes),
    )


def add_source(conn, source_id, scope_id, payload: bytes, speaker="u1",
               provenance="direct_user"):
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, speaker),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC',?,'{}')",
        (source_id, payload, _h(payload), provenance),
    )


def add_span(conn, span_id, source_id, start, end, rev=1):
    payload = bytes(
        conn.execute(
            "SELECT payload FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (source_id, rev),
        ).fetchone()[0]
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (span_id, source_id, rev, start, end, _h(payload[start:end])),
    )


def add_fts(conn, claim_id, rev, scope_id, text, gen):
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,?,?,?)",
        (claim_id, rev, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def seed_claim(conn, claim_id, scope_id, source_id, span_id, text, gen,
               state="active", recorded_from=1, recorded_until=None,
               rev=1, condition=None, subject=None, predicate=None):
    """Claim + source + span + FTS row — the retrieval-path fixture."""
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,?,?,?,1)",
        (claim_id, scope_id, subject, predicate, recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,condition_json,"
        "recorded_from,recorded_until,perspective_id,freshness)"
        " VALUES(?,?,?,?,?,?,NULL,NULL)",
        (claim_id, rev, state,
         json_dumps(condition) if condition is not None else None,
         recorded_from, recorded_until),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,?,?,'primary',NULL)",
        (claim_id, rev, span_id),
    )
    add_fts(conn, claim_id, rev, scope_id, text, gen)


def seed_structured_claim(conn, claim_id, scope_id, *, subject, predicate,
                          value, recorded_from, state="active",
                          recorded_until=None):
    """A consolidation-eligible structured claim (``object_json`` value) —
    the unit ``RefreshScheduler`` plans over."""
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event) VALUES(?,?,?,?,0)",
        (claim_id, scope_id, subject, predicate),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,object_json,"
        "polarity,modality,recorded_from,recorded_until)"
        " VALUES(?,1,?,?,'affirmative','asserted',?,?)",
        (claim_id, state,
         json_dumps({"kind": "literal", "text": value}),
         recorded_from, recorded_until),
    )


def request(query="term", scope_id="sA", caller="human:alice", **kw):
    kw.setdefault("purpose", "recall")
    return RecallRequestV3(
        query=query, scope_id=scope_id, caller_id=caller, **kw
    )


def items_of(result):
    return [i for p in result.packs for i in p.items]


def bodies_of(result):
    out = []
    for i in items_of(result):
        text = i.text
        start = text.find("{")
        end = text.rfind("}")
        assert start != -1 and end > start, text
        out.append(json.loads(text[start:end + 1]))
    return out


def _expand_ref(result, claim_id):
    for b in bodies_of(result):
        if b.get("claim_id") == claim_id and b.get("expand"):
            return b["expand"]
    raise AssertionError(f"no expand ref on {claim_id}")


def _tokens(result) -> int:
    return sum(int(p.tokens) for p in result.packs)


# ---------------------------------------------------------------------------
# D01 — counterevidence-first manifest (V45-03.01, V45-03.02)
# ---------------------------------------------------------------------------

def test_d01_manifest_ships_authorized_contrary_evidence(store):
    """D01 / V45-03.01: a current-state answer with authorized contrary
    evidence ships that evidence — named on the manifest with its role —
    or an insufficiency label. Both sides of an open conflict are
    caller-visible here, so the manifest must name them as included
    contrary refs and record the open-conflict obligation."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        seed_claim(conn, "clGreen", "sA", "srcG", "spG",
                   "deployment pipeline status release is green", gen)
        seed_claim(conn, "clRed", "sA", "srcR", "spR",
                   "deployment pipeline status release is red", gen)
        conn.execute(
            "INSERT INTO conflict_groups(group_id,scope_id,status)"
            " VALUES('cg1','sA','open')",
        )
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id)"
            " VALUES('cg1','clGreen'),('cg1','clRed')",
        )
    res = recall_v3(
        store,
        request("deployment pipeline status release",
                manifest="counterevidence_first"),
    )
    assert not res.abstained
    manifest = res.capabilities.get("manifest")
    assert manifest is not None and manifest["manifest"] == (
        "counterevidence_first"
    )
    contrary = {r["object_id"]: r for r in manifest["contrary_refs"]}
    # The spec's either/or: authorized contrary evidence is included, or
    # an insufficiency label says why it is not.
    assert contrary or manifest["insufficiency_labels"]
    assert contrary["clGreen"]["disposition"] == "included"
    assert contrary["clRed"]["disposition"] == "included"
    assert contrary["clGreen"]["reason"] == "delivered"
    assert contrary["clRed"]["reason"] == "delivered"
    # And the open contest is an unresolved obligation on the record.
    obligations = manifest["unresolved_obligations"]
    assert any(
        o.get("kind") == "open_conflict"
        and set(o.get("members") or ()) == {"clGreen", "clRed"}
        for o in obligations
    )
    # Both sides also physically shipped in the conflict pack.
    delivered = {i.handle.object_id for i in items_of(res)}
    assert {"clGreen", "clRed"} <= delivered


def test_d01_inaccessible_contrary_is_presence_label_only(store):
    """D01 / V45-03.02: counterevidence the caller may not access (here a
    conflict member living in an unauthorized scope) surfaces ONLY as a
    presence-level insufficiency label — never as a named ref, never
    with an identifier or a count of the hidden side."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_scope(conn, "sB", principal="p2", conv="c2")
        seed_auth(conn, "sA")  # grant covers sA only — sB is invisible
        gen = _gen(store)
        seed_claim(conn, "clVis", "sA", "srcV", "spV",
                   "disputed rollout state alpha", gen)
        seed_claim(conn, "clHidden", "sB", "srcH", "spH",
                   "disputed rollout state beta", gen)
        conn.execute(
            "INSERT INTO conflict_groups(group_id,scope_id,status)"
            " VALUES('cg2','sA','open')",
        )
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id)"
            " VALUES('cg2','clVis'),('cg2','clHidden')",
        )
    res = recall_v3(
        store,
        request("disputed rollout state",
                manifest="counterevidence_first"),
    )
    manifest = res.capabilities.get("manifest")
    assert manifest is not None
    # The hidden member is never named on the manifest.
    named = {r["object_id"] for r in manifest["supporting_refs"]}
    named |= {r["object_id"] for r in manifest["contrary_refs"]}
    assert "clHidden" not in named
    # Its presence surfaces only as labels.
    assert manifest["inaccessible_contrary"] is True
    labels = set(manifest["insufficiency_labels"])
    assert {
        "contrary_evidence_inaccessible",
        "contrary_evidence_incomplete",
    } & labels
    # The label tokens also join the warning surface for callers that
    # never read the manifest.
    assert any(w.startswith("contrary_evidence") for w in res.warnings)


# ---------------------------------------------------------------------------
# D02 — manifest pack vs matched-token top-k on a stale-state slice
#       (V45-03.04)
# ---------------------------------------------------------------------------

def _naive_matched_token_topk(conn, scope_id, terms, k):
    """The comparator arm: ranked full-text term coverage, no lifecycle
    or verdict filtering — what a matched-token retriever ships."""
    rows = conn.execute(
        "SELECT fr.claim_id, ft.text FROM fts_rows fr"
        " JOIN facts_fts ft ON ft.fts_row_id = fr.row_id"
        " WHERE fr.scope_id = ?",
        (scope_id,),
    ).fetchall()
    scored = sorted(
        (
            (
                cid,
                sum(1 for t in terms
                    if t in (text or "").casefold()),
            )
            for cid, text in rows
        ),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return [cid for cid, cov in scored[:k] if cov]


def _false_current(conn, claim_ids):
    """Delivered ids whose head revision is terminal or closed — the
    'false-current' readings of D02."""
    out = []
    for cid in claim_ids:
        head = conn.execute(
            "SELECT state, recorded_until FROM claim_revisions"
            " WHERE claim_id = ? ORDER BY revision DESC LIMIT 1",
            (cid,),
        ).fetchone()
        if head is None:
            continue
        if head[0] in ("superseded", "rejected", "archived") \
                or head[1] is not None:
            out.append(cid)
    return out


def test_d02_manifest_pack_has_fewer_false_current_than_topk(store):
    """D02 / V45-03.04: on a seeded stale-state slice — five superseded
    claims that match every query term plus current claims — a
    matched-token top-k ships stale readings as current while the
    verdict-filtered manifest pack does not, and the pack keeps the
    exact-identifier hit."""
    terms = ("deployment", "pipeline", "status", "release",
             "src/deploy.sh")
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        # Stale side: every query term, terminal/closed revisions.
        for i in range(5):
            seed_claim(
                conn, f"stale{i}", "sA", f"srcS{i}", f"spS{i}",
                "deployment pipeline status release src/deploy.sh"
                " was green last week",
                gen, state="superseded", recorded_from=1,
                recorded_until=2,
            )
        # Current side: partial-term coverage only.
        for i in range(4):
            seed_claim(conn, f"cur{i}", "sA", f"srcC{i}", f"spC{i}",
                       "deployment status is currently teal", gen)
        # Exact-identifier current hit (V45-03.04: never dropped).
        seed_claim(conn, "clExact", "sA", "srcX", "spX",
                   "src/deploy.sh holds the deployment pipeline status"
                   " release gate", gen)

    manifest_res = recall_v3(
        store,
        request("deployment pipeline status release src/deploy.sh",
                manifest="counterevidence_first"),
    )
    assert manifest_res.capabilities.get("manifest") is not None
    manifest_ids = [i.handle.object_id for i in items_of(manifest_res)]

    with store.read() as conn:
        naive_ids = _naive_matched_token_topk(
            conn, "sA", terms, k=max(1, len(manifest_ids)),
        )
        naive_false = _false_current(conn, naive_ids)
        manifest_false = _false_current(conn, manifest_ids)

    # The measured delta — the whole point of the scenario.
    assert len(naive_false) > len(manifest_false)
    assert manifest_false == []
    # Exact-identifier coverage survives the filtering.
    assert "clExact" in manifest_ids


# ---------------------------------------------------------------------------
# D03 — localized correction invalidation (V45-04.02, V45-04.04)
#       — verbatim.repair: invalidate_dependents + plan/apply vs rebuild
# ---------------------------------------------------------------------------

def _dep_edge(conn, child, parent, seq=0, role="derived"):
    ck, cid, crev = child
    pk, pid, prev = parent
    conn.execute(
        "INSERT INTO dependency_edges(child_kind,child_id,child_revision,"
        "parent_kind,parent_id,parent_revision,role,producer_id,"
        "operation_id,seq) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ck, cid, crev, pk, pid, prev, role, "prod", f"op:{seq}", seq),
    )


def test_d03_localized_invalidation_immediate_and_smaller_than_rebuild(
    store,
):
    """D03 / V45-04.01/04.02/04.04: a localized source correction runs
    the real repair pipeline — ``invalidate_dependents`` walks the
    dependency closure inside the caller's transaction (epoch bump +
    held marks are committed-or-nothing, never asynchronous),
    ``plan_repair`` names the affected objects and their recompute
    targets up front, and ``apply_repair`` recomputes at least 50%
    fewer objects than ``full_rebuild`` on the same store — measured,
    not assumed."""
    kernel = Kernel(store)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('sA','prof','owner')"
        )
        # Eight independent claim slots; only c0's slot is corrected.
        for i in range(8):
            seed_structured_claim(
                conn, f"c{i}", "sA", subject="subjA",
                predicate=f"pred{i}", value=f"value {i}",
                recorded_from=1,
            )
        # The corrected parent source + its dependent chain at rev 1.
        conn.execute(
            "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
            "created_us) VALUES('src:s0','test','user_message','sA',1)"
        )
        conn.execute(
            "INSERT INTO source_revisions(source_id,revision,payload,"
            "payload_hmac,event_us,captured_us,provenance)"
            " VALUES('src:s0',1,X'ABCD',X'00',1,1,'direct_user')"
        )
        _dep_edge(conn, ("claim", "c0", 1), ("source", "src:s0", 1),
                  seq=1, role="evidence")
        conn.execute(
            "INSERT INTO observations(observation_id,scope_id,revision,"
            "text) VALUES('obs0','sA',1,'subjA pred0 value 0')"
        )
        _dep_edge(conn, ("observation", "obs0", 1), ("claim", "c0", 1),
                  seq=2)
        # An unrelated dependent — outside the closure, must not be held.
        conn.execute(
            "INSERT INTO observations(observation_id,scope_id,revision,"
            "text) VALUES('obs7','sA',1,'subjA pred7 value 7')"
        )
        _dep_edge(conn, ("observation", "obs7", 1), ("claim", "c7", 1),
                  seq=3)
        # The localized correction itself: c0 rev1 closes, rev2 opens.
        conn.execute(
            "UPDATE claim_revisions SET recorded_until=10"
            " WHERE claim_id='c0' AND revision=1"
        )
        conn.execute(
            "INSERT INTO claim_revisions(claim_id,revision,state,"
            "object_json,polarity,modality,recorded_from,recorded_until)"
            " VALUES('c0',2,'active',?,'affirmative','asserted',10,NULL)",
            (json_dumps({"kind": "literal",
                         "text": "corrected value 0"}),),
        )

        out = invalidate_dependents(
            conn, kernel, "sA", [("source", "src:s0", 1)],
        )
        # Immediate: the epoch moves inside this very transaction.
        assert out["report"].epochs["sA"] >= 1
        # Localized: exactly the dependency closure, nothing more (the
        # seed itself is the changed object, not an "affected" one).
        affected = {tuple(r) for r in out["affected"]}
        assert affected == {
            ("claim", "c0", 1),
            ("observation", "obs0", 1),
        }
        # Dependents are held stale in the same transaction — stale
        # content cannot stay visible while recompute runs; the
        # unrelated dependent is untouched.
        assert out["marked_held"]["observations"] == 1
        assert conn.execute(
            "SELECT stale_since_seq FROM observations"
            " WHERE observation_id='obs0'"
        ).fetchone()[0] is not None
        assert conn.execute(
            "SELECT stale_since_seq FROM observations"
            " WHERE observation_id='obs7'"
        ).fetchone()[0] is None
        # V45-04.01: the impact plan names the changed revision's
        # affected objects and recompute targets before work starts.
        plan = plan_repair(
            conn, "sA", [("source", "src:s0", 1)],
            claim_ids=["c0"], since_seq=5,
        )
        assert plan["complete"] is True
        assert {tuple(r) for r in plan["affected"]} == affected
        assert [t["ref"] for t in plan["targets"]] == [
            ["observation", "obs0", 1],
        ]

    # The executor runs the bounded plan; the comparator rebuilds the
    # whole scope through the same producers.
    rep = apply_repair(store, plan, min_proof=1)
    assert rep["objects_recomputed"] >= 1
    full = full_rebuild(store, "sA", min_proof=1)
    # V45-04.04's measured bar: at least 50% fewer recomputed objects.
    assert rep["objects_recomputed"] <= 0.5 * full["objects_recomputed"], (
        f"localized repair recomputed {rep['objects_recomputed']} vs "
        f"full rebuild {full['objects_recomputed']}"
    )
    # The scan itself is also bounded to the touched window.
    assert rep["objects_scanned"] < full["objects_scanned"]


# ---------------------------------------------------------------------------
# D04 — omitted hidden dependency fails, never a partial view (V45-04.03)
#       — verbatim.repair: declared forecast vs re-walked closure
# ---------------------------------------------------------------------------

def test_d04_unsettled_or_missing_dependency_blocks_the_job(store):
    """D04 / V45-04.03: an impact plan that omits a hidden dependency
    fails the job rather than publishing a partial view — in both
    directions the real ``verbatim.repair`` executor enforces:

    1. a plan whose declared forecast misses a real dependent is
       ``complete=False`` and ``apply_repair`` refuses it with
       CONTEXT_INCOMPLETE before touching anything;
    2. a plan that was complete at planning time but whose dependency
       closure grew before apply is re-walked at apply time and the
       newly-hidden dependent trips the same CONTEXT_INCOMPLETE fence.

    A refused apply commits nothing — the hidden dependent stays
    un-recomputed and un-held, so no partial view ever ships."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('sA','prof','owner')"
        )
        seed_structured_claim(
            conn, "c1", "sA", subject="subjA", predicate="pred1",
            value="value 1", recorded_from=1,
        )
        seed_structured_claim(
            conn, "c2", "sA", subject="subjA", predicate="pred2",
            value="value 2", recorded_from=1,
        )
        # A real dependent of c1 that the declared forecast will omit.
        conn.execute(
            "INSERT INTO observations(observation_id,scope_id,revision,"
            "text) VALUES('obsHidden','sA',1,'subjA pred1 value 1')"
        )
        _dep_edge(conn, ("observation", "obsHidden", 1),
                  ("claim", "c1", 1), seq=1)

        # Arm 1: the declared forecast names only the seed — the
        # closure's real dependent is missing, so the plan itself is
        # incomplete and cannot be applied.
        bad = plan_repair(
            conn, "sA", [("claim", "c1", 1)],
            declared=[("claim", "c1", 1)],
        )
        assert bad["complete"] is False
        assert ("observation", "obsHidden", 1) in {
            tuple(m) for m in bad["missing"]
        }
    with pytest.raises(VerbatimError) as exc:
        apply_repair(store, bad)
    assert exc.value.code is ErrorCode.CONTEXT_INCOMPLETE
    # Nothing recomputed: the hidden dependent was never rewritten.
    with store.read() as conn:
        assert conn.execute(
            "SELECT revision FROM observations"
            " WHERE observation_id='obsHidden'"
        ).fetchone()[0] == 1

    with store.tx() as conn:
        # Arm 2: a complete plan at planning time…
        good = plan_repair(
            conn, "sA", [("claim", "c2", 1)],
            declared=[("claim", "c2", 1)],
        )
        assert good["complete"] is True
        # …then the closure grows — a hidden dependency appears after
        # the forecast was declared.
        conn.execute(
            "INSERT INTO observations(observation_id,scope_id,revision,"
            "text) VALUES('obsLate','sA',1,'subjA pred2 value 2')"
        )
        _dep_edge(conn, ("observation", "obsLate", 1),
                  ("claim", "c2", 1), seq=2)
    # The apply-time fence re-walks the closure and refuses — the grown
    # edge is a hidden dependency the plan never predicted.
    with pytest.raises(VerbatimError) as exc2:
        apply_repair(store, good)
    assert exc2.value.code is ErrorCode.CONTEXT_INCOMPLETE
    with store.read() as conn:
        # No partial publication: the late dependent is neither
        # recomputed (revision unchanged) nor marked held.
        row = conn.execute(
            "SELECT revision, stale_since_seq FROM observations"
            " WHERE observation_id='obsLate'"
        ).fetchone()
        assert tuple(row) == (1, None)


# ---------------------------------------------------------------------------
# D05 — minimal-sufficient progressive context (V45-05.03, V45-05.04)
# ---------------------------------------------------------------------------

def test_d05_sufficiency_preserves_identifiers_at_measured_reduction(
    store,
):
    """D05 / V45-05.03/05.04: ``pack_mode="sufficiency"`` admits the
    coverage band — required identifiers, the condition-bearing claim —
    at the requested tier and renders the remaining depth groups at
    their navigational projection. Measured against ``standard`` on the
    same seeded slice at the same requested tier, delivered tokens drop
    by at least 20% while identifiers, conditions, contradictions, and
    the expansion path all survive."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA")
        gen = _gen(store)
        # Coverage group: names the query's hard identifier and bears a
        # condition — mandatory under the declared coverage rule.
        seed_claim(
            conn, "clCov", "sA", "srcCov", "spCov",
            "deploy migration of src/migrate.sh needs the --dry-run flag",
            gen, condition={"requires_env": {"platform": "linux"}},
        )
        # Depth groups: same terms, no identifier/condition — these are
        # the groups a flat delivery expands eagerly.
        filler = "deploy migration flag " + "padding " * 60
        for i in range(9):
            seed_claim(conn, f"clD{i}", "sA", f"srcD{i}", f"spD{i}",
                       filler + f"variant{i}", gen)

    query = "deploy migration flag src/migrate.sh"
    suf = recall_v3(
        store, request(query, pack_mode="sufficiency"),
        detail_tier="l2",
    )
    std = recall_v3(
        store, request(query, pack_mode="standard"),
        detail_tier="l2",
    )

    suf_tokens, std_tokens = _tokens(suf), _tokens(std)
    assert std_tokens > 0 and suf_tokens > 0
    reduction = 1.0 - suf_tokens / std_tokens
    # V45-05.04's measured bar — computed, never assumed.
    assert reduction >= 0.20, (
        f"sufficiency saved {reduction:.1%} "
        f"({suf_tokens} vs {std_tokens} tokens)"
    )

    coverage = suf.capabilities.get("sufficiency")
    assert coverage is not None
    # Non-inferiority: every required identifier is covered by an
    # admitted group.
    assert set(coverage["required_identifiers"]) <= set(
        coverage["covered_identifiers"]
    )
    assert "src/migrate.sh" in coverage["required_identifiers"]

    suf_bodies = bodies_of(suf)
    by_id = {b["claim_id"]: b for b in suf_bodies}
    # The condition-bearing claim is never demoted for tokens — it
    # ships at the requested tier, not the navigational projection.
    assert by_id["clCov"]["detail_tier"] == "l2"
    # Identifiers survive on every item: claim id + span locator, plus
    # the bound expansion ref that is the documented path to depth.
    for b in suf_bodies:
        assert b["claim_id"] and b["span"]["span_id"]
        assert b["span"]["source_id"]
        assert b.get("expand"), b
    # The depth groups still enumerate — navigational, not dropped.
    demoted = [b for b in suf_bodies if b["detail_tier"] == "l0"]
    assert demoted, "sufficiency never reduced the delivered set"


# ---------------------------------------------------------------------------
# D06 — expansion handle after revocation (V45-05.02)
# ---------------------------------------------------------------------------

def test_d06_expansion_ref_denies_after_grant_revocation(store):
    """D06 / V45-05.02: an L1 delivery mints a caller/scope/purpose/
    revision/expiry-bound expansion ref; once the underlying grant is
    revoked the same ref retrieves nothing — the stored influence row is
    re-resolved under current authority and the denial is the
    indistinguishable NOT_FOUND_OR_UNAUTHORIZED."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        gid = seed_auth(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1",
                   "revocable expansion body", _gen(store))
    res = recall_v3(
        store, request("revocable expansion"), detail_tier="l1",
    )
    ref = _expand_ref(res, "cl1")
    # Positive control: the ref works before revocation.
    expanded = expand_item(
        store, ref, caller_id="human:alice", detail_tier="l2",
        now=T0,
    )
    assert any(
        b.get("text") == "revocable expansion body"
        for b in bodies_of(expanded)
    )
    with store.tx() as conn:
        revoke_grant(conn, gid)
    with pytest.raises(VerbatimError) as exc:
        expand_item(
            store, ref, caller_id="human:alice", detail_tier="l2",
            now=T0,
        )
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


# ---------------------------------------------------------------------------
# D07 — failure-aware reuse vs positive-only reuse (V45-06.03, V45-06.04)
# ---------------------------------------------------------------------------

def _seed_procedure(conn, procedure_id="procA", scope_id="sA",
                    env_platform="linux", env_repo="repo-a",
                    failure_modes=None):
    env = environment_map(
        EnvironmentFingerprint(platform=env_platform, repo_id=env_repo)
    )
    fm = failure_modes if failure_modes is not None else {
        "modes": [{
            "signature": "sig-darwin-crash",
            "description": "installer aborts on darwin hosts",
            "occurrences": 2,
            "evidence_refs": [
                {"kind": "source", "id": "srcErr", "revision": 1}
            ],
            "environments": ["darwin"],
        }],
        "status": "observed",
    }
    conn.execute(
        "INSERT INTO procedures(procedure_id,scope_id,revision,task_label,"
        "state,environment_json,condition_json,recorded_from,row_version,"
        "bindings_json,failure_modes_json,applicability_json,"
        "preconditions_json,risk_class,freshness)"
        " VALUES(?,?,?,?,?,?,?,1,1,?,?,?,?,?,?)",
        (
            procedure_id, scope_id, 1, "run the migration", "active",
            json_dumps(env), None,
            json_dumps([{"name": "target", "kind": "repo_path",
                         "required": True}]),
            json_dumps(fm),
            "[]", "[]", "medium", "stable",
        ),
    )


def test_d07_failure_aware_delivery_blocks_held_out_environment(store):
    """D07 / V45-06.03/06.04: in a held-out environment the
    ``failure_aware`` arm blocks delivery — no card, no exposure row —
    while the ``positive_only`` comparator ships the same procedure
    unqualified into the mismatching environment (the reuse that causes
    negative transfer). Counterexamples and their evidence refs ride the
    failure-aware decision either way."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        _seed_procedure(conn)

        held_out = EnvironmentFingerprint(
            platform="darwin", repo_id="repo-b",
        )
        fa = deliver_procedure(
            conn, "procA", task_id="t-heldout",
            environment=held_out, bindings={"target": "/x"},
            mode="failure_aware",
        )
        assert fa.deliverable is False
        assert fa.disposition == "blocked"
        assert fa.environment_match == "mismatch"
        assert fa.card is None
        assert fa.exposure_id is None
        assert any(c.signature == "sig-darwin-crash"
                   for c in fa.counterexamples)
        # The blocked delivery records no exposure — the host never saw
        # the procedure and the reuse denominators stay honest.
        assert exposures_for(conn, "procA") == []

        po = deliver_procedure(
            conn, "procA", task_id="t-heldout",
            environment=held_out, bindings={"target": "/x"},
            mode="positive_only",
        )
        assert po.deliverable is True
        assert po.disposition == "delivered"
        assert po.environment_match == "unqualified"
        assert po.card is not None
        assert po.exposure_id is not None
        # Positive-only reuse reaches the held-out host — the measured
        # exposure the qualified arm prevented.
        exposures = exposures_for(conn, "procA")
        assert len(exposures) == 1
        assert exposures[0]["task_id"] == "t-heldout"


def test_d07_unknown_environment_delivers_loudly_qualified(store):
    """D07 (companion) / V3-22.15 + V45-06.01: a missing environment is
    never a wildcard pass — the failure-aware arm still delivers, but
    loudly qualified, with the counterexamples attached so the host sees
    where the procedure has already broken."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        _seed_procedure(conn)
        d = deliver_procedure(
            conn, "procA", task_id="t-noenv",
            bindings={"target": "/x"}, mode="failure_aware",
        )
        assert d.deliverable is True
        assert d.disposition == "qualified"
        assert d.environment_match == "unknown"
        assert any(c.signature == "sig-darwin-crash"
                   for c in d.counterexamples)
        assert any(w.startswith("applicability_unknown")
                   for w in d.warnings)


# ---------------------------------------------------------------------------
# D08 — exposure without checker receipts is not transfer success
#       (V45-06.02)
# ---------------------------------------------------------------------------

def _outcome_envelope(store, conn, *, kind, trust, checker, task_id,
                      outcome="success"):
    env = SourceEnvelopeV3(
        kind=kind, scope_id="sA", actor_principal="host",
        perspective=Perspective(asserter="host"),
        event_us=now_us(), receipt_us=0,
        content=b'{"note": "outcome"}',
        media_type="application/json",
        trust_class=trust, task_id=task_id,
        metadata={"outcome": outcome, "checker": checker},
    )
    return ingest_envelope(conn, store, env)


def test_d08_exposure_without_checker_receipt_is_not_success(store):
    """D08 / V45-06.02: a recorded procedure exposure — the reuse card
    actually reached the host — still counts as *no* transfer success
    until a host-attested checker receipt resolves the task's outcome.
    Self-reports and receipts bound elsewhere never move the verdict."""
    with store.tx() as conn:
        seed_scope(conn, "sA")
        _seed_procedure(conn, failure_modes={"modes": [], "status":
                                             "no_failure_evidence"})
        # The exposure itself: procedure delivered for this task.
        delivered = deliver_procedure(
            conn, "procA", task_id="t-1", mode="positive_only",
        )
        assert delivered.exposure_id is not None

        # Exposure alone — no outcome envelopes at all.
        r0 = transfer_success(conn, "procA", "t-1")
        assert r0["delivered_procedure"] is True
        assert r0["transfer_success"] is False
        assert "no_outcome_envelopes" in r0["reasons"]

        # An agent self-report is evidence, never attestation.
        _outcome_envelope(
            store, conn, kind=EnvelopeKind.TEST_RESULT,
            trust=TrustClass.AGENT_GENERATED, task_id="t-1",
            checker={"checker_id": "agent-self", "agent_report": True,
                     "host_attested": False},
        )
        r1 = transfer_success(conn, "procA", "t-1")
        assert r1["transfer_success"] is False
        assert "no_attested_outcome" in r1["reasons"]
        assert r1["non_attested_envelopes"] >= 1

        # A host receipt bound to a *different* task cannot certify this
        # one (V4-23.01 binding, the same rule as C20).
        _outcome_envelope(
            store, conn, kind=EnvelopeKind.VERIFICATION,
            trust=TrustClass.HOST_OBSERVED, task_id="t-1",
            checker={"checker_id": "pytest-runner",
                     "host_attested": True, "invocation_id": "inv-w",
                     "task_id": "other-task", "scope_id": "sA"},
        )
        r2 = transfer_success(conn, "procA", "t-1")
        assert r2["transfer_success"] is False
        assert r2["attested_outcome"] in (None, "none")

        # Only the correctly-bound host-attested receipt completes the
        # pair — delivery + attestation together.
        _outcome_envelope(
            store, conn, kind=EnvelopeKind.VERIFICATION,
            trust=TrustClass.HOST_OBSERVED, task_id="t-1",
            checker={"checker_id": "pytest-runner",
                     "host_attested": True, "invocation_id": "inv-ok",
                     "task_id": "t-1", "scope_id": "sA"},
        )
        r3 = transfer_success(conn, "procA", "t-1")
        assert r3["transfer_success"] is True
        assert r3["attested_outcome"] == "success"


# ---------------------------------------------------------------------------
# D09 — branch correction isolation (V45-07.01, V45-07.03, V45-07.04)
#       — verbatim.branches.BranchService: snapshot-pinned, review-gated
# ---------------------------------------------------------------------------

def test_d09_branch_correction_invisible_until_reviewed_apply(store):
    """D09 / V45-07.01/07.03/07.04: a branch proposing a correction is
    an isolated overlay — creating it never alters live recall;
    applying it without operator review is refused
    (INVALID_TRANSITION); ``submit`` opens the review and records the
    invalidation forecast (V45-07.04); only then does the fenced apply
    move the claim head — and live recall flips to match."""
    caller = CallerV3(principal_id="human:alice")
    svc = BranchService(store)
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", purposes=None,
                  verbs=("read", "quote", "review", "admin"))
        seed_claim(conn, "clX", "sA", "srcX", "spX",
                   "the api endpoint is deprecated", _gen(store))

    query = request("api endpoint deprecated")
    before = {i.handle.object_id for i in items_of(recall_v3(store, query))}
    assert "clX" in before

    branch = svc.create(
        caller, "sA", name="retire-endpoint",
        ops=[{"effect": "archive", "claim_id": "clX"}],
    )
    assert branch["state"] == "live"

    # Isolation: the proposed correction is invisible to live recall.
    mid = {i.handle.object_id for i in items_of(recall_v3(store, query))}
    assert "clX" in mid
    with store.read() as conn:
        head = conn.execute(
            "SELECT state, revision FROM claim_revisions"
            " WHERE claim_id='clX' ORDER BY revision DESC LIMIT 1"
        ).fetchone()
        assert tuple(head) == ("active", 1)

    # Review gate: no review, no apply — the fence holds.
    with pytest.raises(VerbatimError) as exc:
        svc.apply(caller, branch["branch_id"])
    assert exc.value.code is ErrorCode.INVALID_TRANSITION

    sub = svc.submit(caller, branch["branch_id"])
    assert sub["review_id"]
    # V45-07.04: the invalidation forecast is recorded at submit time
    # so the apply can verify the predicted affected set.
    assert sub["forecast"]

    out = svc.apply(caller, branch["branch_id"])
    assert out["state"] == "applied"
    assert out["replayed"] is False

    # After the reviewed apply, live state moved and recall reflects it.
    after = {i.handle.object_id for i in items_of(recall_v3(store, query))}
    assert "clX" not in after
    with store.read() as conn:
        head = conn.execute(
            "SELECT state, revision FROM claim_revisions"
            " WHERE claim_id='clX' ORDER BY revision DESC LIMIT 1"
        ).fetchone()
        assert head[0] == "archived"
        doc = svc.get(caller, branch["branch_id"])
        assert doc["state"] == "applied"


# ---------------------------------------------------------------------------
# D10 — purge propagates into open branches (V45-07.02)
#       — real purge closure + real branch-apply fence
# ---------------------------------------------------------------------------

def test_d10_parent_purge_closes_dependents_and_no_branch_restores(store):
    """D10 / V45-07.02: purging a live parent tombstones its dependents
    through the real closure path — post-purge recall returns nothing
    and a previously-minted expansion ref denies — and an open branch
    pinned on that parent cannot apply: its ``archive`` op would
    restore nothing, so the apply fence refuses with
    NOT_FOUND_OR_UNAUTHORIZED. A failed apply commits nothing — the
    branch stays live and the erased bytes stay erased."""
    from verbatim.api_v3.facade import VerbatimV3

    facade = VerbatimV3(store)
    caller = CallerV3(principal_id="human:alice")
    svc = BranchService(store)
    with store.tx() as conn:
        seed_scope(conn, "sA")
        seed_auth(conn, "sA", purposes=None,
                  verbs=("read", "quote", "review", "admin"))
        gen = _gen(store)
        # parent source + dependent claim — the live "parent" of D10.
        seed_claim(conn, "clParent", "sA", "srcParent", "spParent",
                   "parent slice bytes to erase", gen)
    res = recall_v3(
        store, request("parent slice bytes"), detail_tier="l1",
    )
    ref = _expand_ref(res, "clParent")

    # An open branch pinned on the live parent — it references the
    # claim head, not a byte copy.
    branch = svc.create(
        caller, "sA", name="restore-parent",
        ops=[{"effect": "archive", "claim_id": "clParent"}],
    )
    sub = svc.submit(caller, branch["branch_id"])
    assert sub["review_id"]

    out = facade.delete_source("srcParent", principal_id="human:alice")
    assert out["purge_id"]
    # Closure is immediate: suppression is in effect now.
    assert out["status"] == "suppressed"
    assert out["closure"]["logical"] == "suppressed_now"

    # Post-purge live recall of the parent's slice ships nothing.
    res2 = recall_v3(
        store, request("parent slice bytes"), detail_tier="l1",
    )
    delivered = {i.handle.object_id for i in items_of(res2)}
    assert "clParent" not in delivered
    # And the pre-purge expansion ref cannot retrieve the prior slice.
    with pytest.raises(VerbatimError) as exc:
        expand_item(
            store, ref, caller_id="human:alice", detail_tier="l2",
            now=T0,
        )
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    # The propagation half: the branch's pinned parent is now under
    # suppression, so apply refuses — the branch cannot resurrect the
    # purged bytes (V45-07.02/D10).
    with pytest.raises(VerbatimError) as exc2:
        svc.apply(caller, branch["branch_id"])
    assert exc2.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    doc = svc.get(caller, branch["branch_id"])
    # The failed apply committed nothing — the branch was not applied.
    assert doc["state"] == "live"
    with store.read() as conn:
        # The claim head is still under purge targets — nothing was
        # restored by the refused apply.
        assert conn.execute(
            "SELECT COUNT(*) FROM purge_targets WHERE object_id='clParent'"
        ).fetchone()[0] == 1


# ---------------------------------------------------------------------------
# D11 — utility-budgeted refresh vs periodic reflection (V45-08.04)
# ---------------------------------------------------------------------------

def test_d11_utility_budgeted_refresh_spends_less_than_periodic(store):
    """D11 / V45-08.04: on a scope where only a few claim regions changed
    since the durable watermark, the utility-budgeted plan schedules
    just those regions while the periodic comparator charges a full
    input scan plus a full reflection scan — measured spend (claims
    processed) drops by at least 20% at matched freshness coverage, and
    the whole plan lands as a durable ``refresh_pass`` audit event."""
    sched = RefreshScheduler(store)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('sA','prof','owner')"
        )
        # Eight unchanged regions (pre-watermark), two changed ones.
        for i in range(8):
            seed_structured_claim(
                conn, f"old{i}", "sA", subject="subjA",
                predicate=f"pred{i}", value=f"old value {i}",
                recorded_from=1,
            )
        for i in range(2):
            seed_structured_claim(
                conn, f"new{i}", "sA", subject="subjB",
                predicate=f"pred{i}", value=f"new value {i}",
                recorded_from=10,
            )

    # The production one-call path: plan → enqueue → refresh_pass event.
    plan = sched.refresh("sA", since_seq=5)
    with store.read() as conn:
        periodic = sched.periodic_plan(conn, "sA")

    spend, pspend = plan.spend(), periodic.spend()
    assert spend["claims_processed"] > 0
    assert pspend["claims_processed"] > 0
    reduction = 1.0 - spend["claims_processed"] / pspend["claims_processed"]
    # V45-08.04's measured bar.
    assert reduction >= 0.20, (
        f"budgeted refresh spent {spend['claims_processed']} claims vs "
        f"periodic {pspend['claims_processed']} "
        f"({reduction:.1%} reduction)"
    )
    # Only the changed regions were scheduled — matched freshness
    # coverage of the post-watermark work.
    assert len(plan.scheduled) == 2
    assert plan.deferred == []
    # Real jobs on the real queue, plus the durable audit event.
    assert plan.job_ids and len(plan.job_ids) == len(plan.scheduled)
    with store.read() as conn:
        ev = conn.execute(
            "SELECT payload_json FROM events WHERE kind = ?",
            (REFRESH_EVENT_KIND,),
        ).fetchone()
        jobs = conn.execute(
            "SELECT kind, lane, state FROM jobs"
        ).fetchall()
    assert ev is not None
    payload = json.loads(ev[0])
    assert payload["policy_id"] == plan.policy_id
    assert payload["spend"]["claims_processed"] == (
        spend["claims_processed"]
    )
    assert {j[0] for j in jobs} == {"consolidate"}
    assert all(j[1] == "background" and j[2] == "queued" for j in jobs)


# ---------------------------------------------------------------------------
# D12 — owner-priority refresh survives the scheduler (V45-08.02)
# ---------------------------------------------------------------------------

def test_d12_owner_priority_refresh_is_never_dropped(store):
    """D12 / V45-08.02: with the budget at zero, non-owner candidates
    defer — reported, never silent — while the owner-requested region
    schedules around the bound, enqueues a real consolidate job on the
    background lane, and is leasable by a worker. Priority is explicit:
    it still counts in the spend report."""
    sched = RefreshScheduler(store)
    queue = JobQueue(store)
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('sA','prof','owner')"
        )
        for i in range(2):
            seed_structured_claim(
                conn, f"chg{i}", "sA", subject="subjB",
                predicate=f"pred{i}", value=f"changed {i}",
                recorded_from=10,
            )

    with store.tx() as conn:
        plan = sched.plan(
            conn, "sA", since_seq=5,
            budget=RefreshBudget(max_jobs=0),
            owner_requests=[{"claim_ids": ["chg0", "chg1"],
                             "weight": 4.0}],
        )
        # Owner work schedules around a zero bound; scanner work defers.
        assert len(plan.scheduled) == 1
        owner = plan.scheduled[0]
        assert owner.owner_requested is True
        assert owner.priority >= 4.0
        assert "owner_request" in owner.reasons
        assert len(plan.deferred) == 2
        assert all(
            plan.deferred_reasons.get(c.region_key) == "max_jobs"
            for c in plan.deferred
        )
        # Priority still reports its cost (D12's honest accounting).
        assert plan.spend()["priority_jobs"] == 1
        sched.schedule(conn, plan)

    # The owner job is a real queued consolidate on the background lane.
    assert len(plan.job_ids) == 1
    with store.read() as conn:
        row = conn.execute(
            "SELECT kind, lane, state FROM jobs WHERE job_id = ?",
            (plan.job_ids[0],),
        ).fetchone()
    assert row is not None
    assert row[0] == JobKind.CONSOLIDATE.value
    assert row[1] == "background"
    assert row[2] == "queued"
    # And a worker can actually lease it — scheduled, not dropped.
    leased = queue.lease(
        "sA", [JobKind.CONSOLIDATE], owner="w-refresh",
        limit=1, now_us=T0,
    )
    assert leased and leased[0]["job_id"] == plan.job_ids[0]
