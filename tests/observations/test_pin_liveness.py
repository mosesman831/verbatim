"""Observation pin-liveness invariants (SPEC_V6 §03.2, V6-03.13).

An observation's ``observation_evidence`` pins must resolve to
independently retrievable source bytes — the rendered summary is derived
metadata, never the sole trace of the fact it consolidates:

* ``resolve_pins`` reports each pin's liveness — resolvable / held /
  erased / superseded — and whether the pin chains to a live
  ``source_revisions`` payload;
* ``never_sole_trace`` is the strict invariant: *every* pin must reach
  live bytes — a dangling, emptied, purged, or held endpoint fails
  closed;
* holding or erasing a pinned source removes the claim's support on the
  next pass, so ``consolidate`` retires the summary (V3-23.07 /
  V6-03.13(b)) — never a live, unsupported belief.

All tests run against a real ``Store.create`` (schema v3) and reuse the
``test_observations`` seeding conventions — real ``sources`` /
``source_revisions`` payloads behind every pinned claim.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import json_dumps
from verbatim.observations import consolidate, record_observation
from verbatim.observations.aggregate import (
    never_sole_trace,
    resolve_pins,
)
from verbatim.security import open_quarantine
from verbatim.storage import repos_v3
from verbatim.storage.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "pins.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:pins"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
    return sid


def _claim(
    conn,
    scope_id,
    claim_id,
    *,
    subject="alice",
    predicate="preference",
    value="café lumière",
    state="active",
    revision=1,
    family_id=None,
):
    """One structured claim + head revision (test_observations._claim
    conventions)."""
    conn.execute(
        "INSERT INTO claims (claim_id, scope_id, subject_id, predicate,"
        " created_event) VALUES (?,?,?,?,0)",
        (claim_id, scope_id, subject, predicate),
    )
    conn.execute(
        "INSERT INTO claim_revisions (claim_id, revision, state,"
        " object_json, polarity, modality, condition_json, recorded_from,"
        " recorded_until, perspective_id)"
        " VALUES (?,?,?,?,?,?,?,1,NULL,NULL)",
        (
            claim_id,
            revision,
            state,
            json_dumps({"kind": "literal", "text": value}),
            "affirmative",
            "asserted",
            None,
        ),
    )
    if family_id is not None:
        conn.execute(
            "INSERT OR IGNORE INTO evidence_families"
            " (family_id, scope_id, origin_kind, origin_id, created_event)"
            " VALUES (?,?,?,?,0)",
            (family_id, scope_id, "test", claim_id),
        )
        conn.execute(
            "INSERT INTO family_members (family_id, object_kind, object_id,"
            " role) VALUES (?, 'claim', ?, 'origin')",
            (family_id, claim_id),
        )


def _evidence_chain(conn, store, scope_id, claim_id, *, span_id, source_id,
                    revision=1, payload=b"\x01"):
    """Source + non-empty revision payload + span + claim_evidence — the
    independently-retrievable-bytes chain every pin must bottom out on."""
    conn.execute(
        "INSERT INTO sources (source_id, origin, source_kind, scope_id,"
        " created_us) VALUES (?, 'test', 'user_message', ?, 1)",
        (source_id, scope_id),
    )
    conn.execute(
        "INSERT INTO source_revisions (source_id, revision, payload,"
        " payload_hmac, event_us, captured_us, provenance)"
        " VALUES (?, ?, ?, ?, 1, 1, 'direct_user')",
        (source_id, revision, payload, store.hmac(payload)),
    )
    conn.execute(
        "INSERT INTO spans (span_id, source_id, revision, start_byte,"
        " end_byte, excerpt_hmac, harvester_version)"
        " VALUES (?, ?, ?, 0, 1, ?, 'test-harvest-1')",
        (span_id, source_id, revision, store.hmac(payload[0:1])),
    )
    conn.execute(
        "INSERT INTO claim_evidence (claim_id, revision, span_id)"
        " VALUES (?, ?, ?)",
        (claim_id, revision, span_id),
    )


def _seed_two_family_slot(conn, store, scope_id):
    """Two same-slot claims from distinct evidence families, each with a
    real source-byte chain — the V6-03.11 positive input."""
    _claim(conn, scope_id, "c1", family_id="fam1")
    _claim(conn, scope_id, "c2", family_id="fam2")
    _evidence_chain(conn, store, scope_id, "c1",
                    span_id="sp1", source_id="src1")
    _evidence_chain(conn, store, scope_id, "c2",
                    span_id="sp2", source_id="src2")


def _observations(conn, scope_id):
    return repos_v3.query(conn, "observations", {"scope_id": scope_id})


def test_pins_resolve_and_never_sole_trace(store, scope_id):
    """A consolidated observation's pins all resolve to live source
    bytes — the summary is grounded, never the sole trace (V6-03.13(a))."""
    with store.tx() as conn:
        _seed_two_family_slot(conn, store, scope_id)
        res = consolidate(conn, scope_id)
        assert res["observations_written"] == 1

        obs = _observations(conn, scope_id)
        assert len(obs) == 1
        assert obs[0]["proof_count"] == 2
        oid = obs[0]["observation_id"]

        report = resolve_pins(conn, oid)
        assert report["found"] is True
        assert report["summary"]["total"] == 2
        assert report["summary"]["resolvable"] == 2
        assert report["summary"]["held"] == 0
        assert report["summary"]["erased"] == 0
        assert report["summary"]["byte_trace"] == 2
        # Every pin chains through claim_evidence → spans to the
        # (source_id, revision) that actually carries the bytes.
        sources = {
            tuple(ref) for p in report["pins"] for ref in p["source_refs"]
        }
        assert sources == {("src1", 1), ("src2", 1)}
        assert never_sole_trace(conn, oid) is True

        # Deterministic re-derivation (V6-03.13(c)): an unchanged input
        # set reproduces the identical observation — no new revision,
        # no rewritten evidence.
        res2 = consolidate(conn, scope_id)
        assert res2["observations_written"] == 0
        obs2 = repos_v3.get(
            conn, "observations", {"observation_id": oid}
        )
        assert obs2["revision"] == obs[0]["revision"]
        assert obs2["text"] == obs[0]["text"]


def test_held_source_pin_marks_held_and_retires(store, scope_id):
    """A quarantine hold on a pinned source marks the pin held, breaks
    the byte trace, and the next pass retires the summary — a held pin
    never leaves a live, unsupported observation (V6-03.13(b))."""
    with store.tx() as conn:
        _seed_two_family_slot(conn, store, scope_id)
        consolidate(conn, scope_id)
        oid = _observations(conn, scope_id)[0]["observation_id"]
        assert never_sole_trace(conn, oid) is True

        open_quarantine(
            conn, ("source", "src2", 1), ["test:hold"], [],
            scope_id=scope_id,
        )

        report = resolve_pins(conn, oid)
        by_claim = {p["object_id"]: p for p in report["pins"]}
        assert by_claim["c2"]["held"] is True
        assert by_claim["c2"]["byte_trace"] is False
        assert by_claim["c1"]["held"] is False
        assert by_claim["c1"]["byte_trace"] is True
        assert never_sole_trace(conn, oid) is False

        # Re-consolidate: c2's held evidence contributes nothing, the
        # slot drops below min_proof, and the summary is retired.
        consolidate(conn, scope_id)
        obs = repos_v3.get(
            conn, "observations", {"observation_id": oid}
        )
        assert obs["recorded_until"] is not None


def test_erased_source_breaks_trace_and_retires(store, scope_id):
    """An erased pinned source — purge tombstone plus emptied payload —
    marks the pin erased, fails the trace, and the summary retires on
    the next pass (V6-03.13(b))."""
    with store.tx() as conn:
        _seed_two_family_slot(conn, store, scope_id)
        consolidate(conn, scope_id)
        oid = _observations(conn, scope_id)[0]["observation_id"]

        conn.execute(
            "INSERT INTO purges (purge_id, selection_digest, scope_id,"
            " state, requested_us) VALUES ('pg1', X'00', ?, 'completed', 1)",
            (scope_id,),
        )
        conn.execute(
            "INSERT INTO purge_targets (purge_id, object_kind, object_id)"
            " VALUES ('pg1', 'source', 'src2')",
        )
        conn.execute(
            "UPDATE source_revisions SET payload = X''"
            " WHERE source_id = 'src2' AND revision = 1"
        )

        report = resolve_pins(conn, oid)
        by_claim = {p["object_id"]: p for p in report["pins"]}
        assert by_claim["c2"]["erased"] is True
        assert by_claim["c2"]["byte_trace"] is False
        assert by_claim["c1"]["erased"] is False
        assert never_sole_trace(conn, oid) is False

        consolidate(conn, scope_id)
        obs = repos_v3.get(
            conn, "observations", {"observation_id": oid}
        )
        assert obs["recorded_until"] is not None


def test_dangling_pin_fails_closed(store, scope_id):
    """A pin that resolves to nothing fails closed: unresolvable pins
    are reported, and the trace invariant refuses to assume grounding
    it cannot prove (V6-03.13(a))."""
    with store.tx() as conn:
        oid = record_observation(
            conn,
            scope_id=scope_id,
            text="manual observation on a ghost pin",
            proof_count=1,
            supports=[("claim", "ghost-claim", 1)],
        )
        report = resolve_pins(conn, oid)
        assert report["summary"]["total"] == 1
        assert report["summary"]["resolvable"] == 0
        pin = report["pins"][0]
        assert pin["resolvable"] is False
        assert pin["byte_trace"] is False
        assert never_sole_trace(conn, oid) is False


def test_unknown_observation_fails_closed(store, scope_id):
    """An absent observation reports ``found=False`` — and the strict
    invariant treats it as unproven, not as a pass."""
    with store.tx() as conn:
        report = resolve_pins(conn, "obs_missing")
        assert report["found"] is False
        assert report["pins"] == []
        assert never_sole_trace(conn, "obs_missing") is False
