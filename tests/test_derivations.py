"""Derivation-graph API tests (SPEC_V3 §06.06, §17.02, §39.07).

Covers: immutable edge recording, one-hop and recursive traversal,
evidence-plane roots, the purge impact set, cycle safety, and the absence
of any update path over edges.
"""

from __future__ import annotations

import pytest

from verbatim import derivations as deriv
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.storage import repos_v3
from verbatim.storage.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:deriv"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
    return sid


def _edge(conn, child, parent, scope_id, seq, producer="harvester-1"):
    return deriv.record_edge(
        conn, child, parent, "harvester", producer, scope_id, seq
    )


# ---------------------------------------------------------------- recording


def test_record_edge_roundtrip(store, scope_id):
    with store.tx() as conn:
        assert _edge(
            conn, ("claim", "c1", 1), ("span", "sp1", 1), scope_id, 10
        ) is True
        parents = deriv.parents_of(conn, ("claim", "c1", 1))
        assert parents == [("span", "sp1", 1)]
        children = deriv.children_of(conn, ("span", "sp1", 1))
        assert children == [("claim", "c1", 1)]
        prod = deriv.producers_of(conn, ("claim", "c1", 1))
        assert prod[0]["producer_kind"] == "harvester"
        assert prod[0]["seq"] == 10
        assert prod[0]["scope_id"] == scope_id


def test_record_edge_replay_is_idempotent(store, scope_id):
    with store.tx() as conn:
        assert _edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, 1)
        # identical replay: no-op, not an error (job redelivery, §40.01)
        assert _edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, 1) is False
        rows = repos_v3.query(conn, "derivations", {})
        assert len(rows) == 1


def test_record_edge_conflicting_producer_is_integrity_error(store, scope_id):
    with store.tx() as conn:
        _edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, 1)
        with pytest.raises(VerbatimError) as ei:
            deriv.record_edge(
                conn, ("claim", "c1", 1), ("span", "s1", 1),
                "compiler", "other-producer", scope_id, 2,
            )
        assert ei.value.code == ErrorCode.INTEGRITY
        with pytest.raises(VerbatimError) as ei:
            deriv.record_edge(
                conn, ("claim", "c1", 1), ("span", "s1", 1),
                "harvester", "harvester-1", "scope:other", 2,
            )
        assert ei.value.code == ErrorCode.INTEGRITY


def test_record_edge_validation(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            _edge(conn, ("claim", "c1"), ("span", "s1", 1), scope_id, 1)
        assert ei.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError):
            _edge(conn, ("claim", "c1", 0), ("span", "s1", 1), scope_id, 1)
        with pytest.raises(VerbatimError):
            _edge(conn, ("mystery", "c1", 1), ("span", "s1", 1), scope_id, 1)
        with pytest.raises(VerbatimError):
            _edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, -1)
        with pytest.raises(VerbatimError):
            _edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, True)


def test_fact_alias_normalizes_to_claim(store, scope_id):
    with store.tx() as conn:
        _edge(conn, ("fact", "c1", 1), ("span", "s1", 1), scope_id, 1)
        assert deriv.parents_of(conn, ("claim", "c1", 1)) == [("span", "s1", 1)]


# ---------------------------------------------------------------- traversal


def _chain(conn, scope_id):
    """span → claim → procedure → observation."""
    _edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, 1)
    _edge(conn, ("procedure", "p1", 1), ("claim", "c1", 1), scope_id, 2)
    _edge(conn, ("observation", "o1", 1), ("procedure", "p1", 1), scope_id, 3)


def test_ancestors_and_descendants_chain(store, scope_id):
    with store.tx() as conn:
        _chain(conn, scope_id)
        anc = deriv.ancestors_of(conn, ("observation", "o1", 1))
        assert anc == {
            ("procedure", "p1", 1), ("claim", "c1", 1), ("span", "s1", 1)
        }
        desc = deriv.descendants_of(conn, ("span", "s1", 1))
        assert set(desc) == {
            ("claim", "c1", 1), ("procedure", "p1", 1), ("observation", "o1", 1)
        }
        # depth: claim is 1 hop, procedure 2, observation 3
        assert desc[("claim", "c1", 1)] == 1
        assert desc[("procedure", "p1", 1)] == 2
        assert desc[("observation", "o1", 1)] == 3


def test_revision_wildcard_queries(store, scope_id):
    with store.tx() as conn:
        _edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, 1)
        _edge(conn, ("claim", "c1", 2), ("span", "s2", 1), scope_id, 2)
        # rev=None: parents across all revisions of the claim
        parents = deriv.parents_of(conn, ("claim", "c1", None))
        assert set(parents) == {("span", "s1", 1), ("span", "s2", 1)}
        desc = deriv.descendants_of(conn, ("span", "s2", None))
        assert ("claim", "c1", 2) in desc


def test_roots_returns_evidence_frontier(store, scope_id):
    with store.tx() as conn:
        _chain(conn, scope_id)
        rs = deriv.roots(conn, ("observation", "o1", 1))
        assert rs == [("span", "s1", 1)]
        # the span itself is its own root
        assert deriv.roots(conn, ("span", "s1", 1)) == [("span", "s1", 1)]
        # multi-parent object lists every evidence root
        _edge(conn, ("procedure", "p1", 1), ("span", "s9", 1), scope_id, 4)
        rs = deriv.roots(conn, ("procedure", "p1", 1))
        assert rs == [("span", "s1", 1), ("span", "s9", 1)]


def test_diamond_dedup(store, scope_id):
    """Two children share a parent; the grandchild merges both paths."""
    with store.tx() as conn:
        _edge(conn, ("claim", "c1", 1), ("span", "root", 1), scope_id, 1)
        _edge(conn, ("claim", "c2", 1), ("span", "root", 1), scope_id, 2)
        _edge(conn, ("procedure", "g", 1), ("claim", "c1", 1), scope_id, 3)
        _edge(conn, ("procedure", "g", 1), ("claim", "c2", 1), scope_id, 4)
        affected = deriv.affected_by_purge(conn, [("span", "root", 1)])
        derived = affected["derived"]
        assert set(derived) == {
            ("claim", "c1", 1), ("claim", "c2", 1), ("procedure", "g", 1)
        }
        # the shared grandchild appears once, at its shallowest depth
        assert derived[("procedure", "g", 1)] == 2
        children = deriv.children_of(conn, ("span", "root", 1))
        assert sorted(children) == [("claim", "c1", 1), ("claim", "c2", 1)]


def test_cycle_safety(store, scope_id):
    """A synthetic cycle terminates instead of looping forever."""
    with store.tx() as conn:
        _edge(conn, ("claim", "a", 1), ("claim", "b", 1), scope_id, 1)
        _edge(conn, ("claim", "b", 1), ("claim", "a", 1), scope_id, 2)
        desc = deriv.descendants_of(conn, ("claim", "a", 1))
        assert set(desc) == {("claim", "b", 1), ("claim", "a", 1)}
        anc = deriv.ancestors_of(conn, ("claim", "a", 1))
        assert anc == {("claim", "b", 1), ("claim", "a", 1)}
        # depth bound is honored as the cycle guard
        desc1 = deriv.descendants_of(conn, ("claim", "a", 1), max_depth=1)
        assert set(desc1) == {("claim", "b", 1)}


def test_affected_by_purge_multi_seed(store, scope_id):
    with store.tx() as conn:
        _chain(conn, scope_id)
        _edge(conn, ("claim", "c2", 1), ("span", "s2", 1), scope_id, 5)
        affected = deriv.affected_by_purge(
            conn, [("span", "s1", 1), ("span", "s2", 1)]
        )
        assert set(affected["derived"]) == {
            ("claim", "c1", 1), ("procedure", "p1", 1),
            ("observation", "o1", 1), ("claim", "c2", 1),
        }
        assert ("span", "s1", 1) in affected["roots"]


# ---------------------------------------------------------------- immutability


def test_no_update_path_for_edges(store, scope_id):
    """The module exposes no way to mutate a recorded edge (V3-17.02)."""
    assert not hasattr(deriv, "update")
    assert not hasattr(deriv, "update_edge")
    assert not hasattr(deriv, "set_edge")
    # and the write path never calls the shared update helper
    import verbatim.storage.repos_v3 as rv3

    calls = []
    orig = rv3.update

    def _spy(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)

    rv3.update = _spy
    try:
        with store.tx() as conn:
            _edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, 1)
            deriv.parents_of(conn, ("claim", "c1", 1))
            deriv.children_of(conn, ("span", "s1", 1))
            deriv.ancestors_of(conn, ("claim", "c1", 1))
            deriv.descendants_of(conn, ("span", "s1", 1))
            deriv.roots(conn, ("claim", "c1", 1))
            deriv.affected_by_purge(conn, [("span", "s1", 1)])
    finally:
        rv3.update = orig
    assert calls == []


def test_max_depth_validation(store, scope_id):
    with store.tx() as conn:
        _chain(conn, scope_id)
        with pytest.raises(VerbatimError):
            deriv.descendants_of(conn, ("span", "s1", 1), max_depth=0)
        with pytest.raises(VerbatimError):
            deriv.descendants_of(conn, ("span", "s1", 1), max_depth=10_000)
