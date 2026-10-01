"""source_state/v1 — fenced transitions (V5-14.10 … V5-14.16).

CAS on control_version + optional mutation-head/epoch fences, monotonic
versions, supersede/correct/retract dispositions, scheduled-transition
review fencing, the erasure tombstone, and the producer publish recheck —
all inside real ``store.tx()`` transactions.
"""

from __future__ import annotations

import pytest

from verbatim.core import time as _time
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.purge import _erase_source
from verbatim.sourcestate import (
    apply_erasure,
    artifact_id,
    assert_publishable,
    current_state,
    ensure_state,
    get_state,
    history,
    is_revision_current,
    transition,
)
from verbatim.sourcestate.state import OBJECT_KIND
from verbatim.storage import repos_v4

from .conftest import NS, SID, add_revision, seed_scope, seed_source

PRODUCER = "verbatim.memory.facade.v1"


def _setup(conn, store, sid=SID, ns=NS):
    seed_source(conn, store, sid, ns)
    return ensure_state(conn, sid, ns, store=store)


class TestCAS:
    def test_transition_bumps_monotonic(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            s1 = transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
            )
            s2 = transition(
                conn, SID, expected_control_version=1,
                disposition="retract", producer=PRODUCER,
            )
        assert (s1.control_version, s2.control_version) == (1, 2)
        assert s2.mutation_head == "2"  # head retained across retract
        with installed.read() as conn:
            docs = history(conn, SID)
        assert [d["control_version"] for d in docs] == [0, 1, 2]
        assert [d["change"] for d in docs] == ["create", "supersede", "retract"]

    def test_cas_conflict_typed(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
            )
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=0,  # stale
                    disposition="retract", producer=PRODUCER,
                )
            assert ei.value.code is ErrorCode.STALE_DEPENDENCY
            # the failed attempt changed nothing
            assert get_state(conn, SID).control_version == 1
            assert len(history(conn, SID)) == 2

    def test_expected_revision_fence(self, installed):
        """V5-14.11: the predecessor head is compared too — a stale ref
        conflicts instead of silently rebinding (V5-14.02)."""
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
            )
            add_revision(conn, installed, SID, 3, "deploy-v3")
            # caller pinned predecessor rev 1, live head is 2
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=1,
                    expected_revision=1,
                    disposition="supersede", superseded_by=3,
                    producer=PRODUCER,
                )
            assert ei.value.code is ErrorCode.STALE_DEPENDENCY
            ok = transition(
                conn, SID, expected_control_version=1,
                expected_revision=2,
                disposition="supersede", superseded_by=3,
                producer=PRODUCER,
            )
            assert ok.mutation_head == "3"

    def test_epoch_fence(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=0,
                    disposition="supersede", superseded_by=2,
                    producer=PRODUCER,
                    epoch_vector={NS: 99},
                )
            assert ei.value.code is ErrorCode.STALE_EPOCH
            ok = transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2,
                producer=PRODUCER,
                epoch_vector={NS: 0},
            )
            assert ok.control_version == 1

    def test_missing_source_typed(self, installed):
        with installed.tx() as conn:
            seed_scope(conn)
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=0,
                    disposition="retract", producer=PRODUCER,
                )
            assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_generation_bump_in_same_tx(self, installed):
        before = installed.projection_generation()
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
                store=installed,
            )
        assert installed.projection_generation() == before + 1


    def test_operation_id_replay_is_idempotent(self, installed):
        """Coordinator rule (V4-09.05): replaying a committed operation
        returns its prior result; a reused key with different input is a
        typed conflict — never a silent second application."""
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            first = transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
                operation_id="op:replace-1",
            )
            # Replay after commit: the pinned cv is now stale, but the
            # operation key returns the committed result instead.
            again = transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
                operation_id="op:replace-1",
            )
            assert again == first
            assert get_state(conn, SID).control_version == 1
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=0,
                    disposition="retract", producer=PRODUCER,
                    operation_id="op:replace-1",
                )
            assert ei.value.code is ErrorCode.OPERATION_CONFLICT


class TestValidation:
    def test_effective_at_only_for_supersede(self, installed):
        """V5-14.13: effective_at on correct/retract fails validation."""
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            future = _time.rfc3339(_time.now_us() + 3600_000_000)
            for disp in ("correct", "retract"):
                with pytest.raises(VerbatimError) as ei:
                    transition(
                        conn, SID, expected_control_version=0,
                        disposition=disp, superseded_by=(
                            2 if disp == "correct" else None
                        ),
                        effective_at=future, producer=PRODUCER,
                    )
                assert ei.value.code is ErrorCode.VALIDATION

    def test_supersede_requires_successor(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=0,
                    disposition="supersede", producer=PRODUCER,
                )
            assert ei.value.code is ErrorCode.VALIDATION

    def test_successor_rejected_for_retract(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=0,
                    disposition="retract", superseded_by=2,
                    producer=PRODUCER,
                )
            assert ei.value.code is ErrorCode.VALIDATION

    def test_erased_not_a_transition_target(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=0,
                    disposition="erased", producer=PRODUCER,
                )
            assert ei.value.code is ErrorCode.INVALID_TRANSITION

    def test_invalid_window(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            now = _time.now_us()
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=0,
                    disposition="activate",
                    valid_from=_time.rfc3339(now + 10),
                    valid_to=_time.rfc3339(now),  # inverted
                    producer=PRODUCER,
                )
            assert ei.value.code is ErrorCode.VALIDATION

    def test_known_at_cannot_regress(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            st = get_state(conn, SID)
            earlier = _time.parse_rfc3339(st.known_at) - 1_000_000
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=0,
                    disposition="retract", producer=PRODUCER,
                    known_at=earlier,
                )
            assert ei.value.code is ErrorCode.VALIDATION


class TestDispositions:
    def test_correct_vs_supersede(self, installed):
        """correct marks the predecessor retracted-but-preserved; supersede
        declares a boundary with a named successor. Distinct labels
        (V5-14.12/13)."""
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            c = transition(
                conn, SID, expected_control_version=0,
                disposition="correct", superseded_by=2, producer=PRODUCER,
                valid_from="2026-01-01T00:00:00Z",
                valid_to="2026-06-01T00:00:00Z",
            )
        assert c.disposition == "corrected"
        assert c.superseded_by == "2"
        assert c.effective_at is None  # corrections land at record time
        with installed.read() as conn:
            cur = current_state(conn, SID)
        assert cur.current and cur.effective_revision == 2
        assert cur.predecessor_revision == 1
        assert "correct" in cur.detail

    def test_supersede_records_boundary(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            s = transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
            )
        assert s.disposition == "superseded"
        assert s.superseded_by == "2"
        assert s.effective_at is not None  # defaulted to commit time
        with installed.read() as conn:
            docs = history(conn, SID)
        assert docs[1]["predecessor_head"] == "1"

    def test_retract_leaves_nothing_current(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            transition(
                conn, SID, expected_control_version=0,
                disposition="retract", producer=PRODUCER,
            )
        with installed.read() as conn:
            cur = current_state(conn, SID)
        assert cur.disposition == "retracted"
        assert cur.current is False
        assert not is_revision_current(conn, SID, 1)

    def test_scheduled_supersede_requires_resolution(self, installed):
        """V5-14.12: changes on an unresolved scheduled transition need an
        explicit reviewed resolution, not silent schedule replacement."""
        future = _time.rfc3339(_time.now_us() + 3600_000_000)
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2,
                effective_at=future, producer=PRODUCER,
            )
            add_revision(conn, installed, SID, 3, "deploy-v3")
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=1,
                    disposition="supersede", superseded_by=3,
                    producer=PRODUCER,
                )
            assert ei.value.code is ErrorCode.INVALID_TRANSITION
            ok = transition(
                conn, SID, expected_control_version=1,
                disposition="supersede", superseded_by=3,
                producer=PRODUCER, resolve_pending=True,
            )
            assert ok.control_version == 2
            assert ok.mutation_head == "3"


class TestErasure:
    def test_erasure_tombstones_and_never_reactivates(self, installed):
        with installed.tx() as conn:
            _setup(conn, installed)
            tomb = apply_erasure(conn, SID, producer="privacy.closure")
        assert tomb.disposition == "erased"
        assert tomb.control_version == 1
        assert tomb.mutation_head == "1"  # CAS anchor retained
        with installed.tx() as conn:
            # erasure never reactivates an older revision (V5-14.16)
            with pytest.raises(VerbatimError) as ei:
                transition(
                    conn, SID, expected_control_version=1,
                    disposition="activate", producer=PRODUCER,
                )
            assert ei.value.code is ErrorCode.INVALID_TRANSITION
            # and ensure_state does not resurrect it either
            again = ensure_state(conn, SID, NS)
            assert again.disposition == "erased"
            # idempotent re-erasure
            same = apply_erasure(conn, SID, producer="privacy.closure")
            assert same.control_version == 1
        with installed.read() as conn:
            obj = repos_v4.get(
                conn, "objects",
                {"kind": OBJECT_KIND, "object_id": artifact_id(SID)},
            )
            assert obj["disposition"] == "erased"
            docs = history(conn, SID)
            assert docs[-1]["change"] == "erase"

    def test_erasure_cas_guard(self, installed):
        """V5-15.03: a version-bound suppression conflicts rather than
        silently deleting a newer revision."""
        with installed.tx() as conn:
            _setup(conn, installed)
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
            )
            with pytest.raises(VerbatimError) as ei:
                apply_erasure(
                    conn, SID, producer="privacy.closure",
                    expected_control_version=0,  # stale ref
                )
            assert ei.value.code is ErrorCode.STALE_DEPENDENCY
            tomb = apply_erasure(
                conn, SID, producer="privacy.closure",
                expected_control_version=1, expected_revision=2,
            )
            assert tomb.disposition == "erased"

    def test_erasure_unregistered_source_fences(self, installed):
        with installed.tx() as conn:
            tomb = apply_erasure(
                conn, SID, producer="privacy.closure", namespace=NS
            )
        assert tomb.disposition == "erased"
        assert tomb.control_version == 0

    def test_erasure_unregistered_needs_namespace(self, installed):
        with installed.tx() as conn:
            with pytest.raises(VerbatimError) as ei:
                apply_erasure(conn, SID, producer="privacy.closure")
            assert ei.value.code is ErrorCode.VALIDATION

    def test_erasure_composes_with_real_source_purge(self, installed):
        """Closure membership (V5-14.16): the tombstone lands inside the
        same transaction as the real source-byte erasure."""
        with installed.tx() as conn:
            _setup(conn, installed)
            emptied = _erase_source(conn, installed, SID)
            tomb = apply_erasure(conn, SID, producer="privacy.closure")
            assert (SID, 1) in emptied
            payload = conn.execute(
                "SELECT length(payload) FROM source_revisions"
                " WHERE source_id = ? AND revision = 1",
                (SID,),
            ).fetchone()[0]
            assert payload == 0
            assert tomb.disposition == "erased"
            with pytest.raises(VerbatimError):
                assert_publishable(conn, SID)

    def test_publishable_recheck(self, installed):
        """V5-14.14: pending producers re-fence before publish."""
        with installed.tx() as conn:
            _setup(conn, installed)
            st = assert_publishable(conn, SID, expected_control_version=0)
            assert st.control_version == 0
            add_revision(conn, installed, SID, 2, "deploy-v2")
            transition(
                conn, SID, expected_control_version=0,
                disposition="supersede", superseded_by=2, producer=PRODUCER,
            )
            with pytest.raises(VerbatimError) as ei:
                assert_publishable(conn, SID, expected_control_version=0)
            assert ei.value.code is ErrorCode.STALE_DEPENDENCY
            assert_publishable(conn, SID, expected_control_version=1)
