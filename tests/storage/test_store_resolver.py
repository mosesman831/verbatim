"""Profile-store resolver conformance (SPEC_V4 F4-18, V4-05.10,
V4-07.10, C82).

Two on-disk conventions can coexist under a profile's data directory —
``{profile_id}.db`` (the canonical profile store) and ``v3.db`` (the v3
adapter's historical default). ``resolve_store_path`` is the single
selection policy every surface shares: deterministic inventory, explicit
operator decision channels, and never a silent merge or an empty
replacement of an existing store.
"""

from __future__ import annotations

import os

import pytest

from verbatim.api import open_store
from verbatim.config import config_from_mapping
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v4 import StoreResolutionKind
from verbatim.host import LocalHost
from verbatim.storage.resolver import (
    require_store_path,
    resolve_store_path,
)
from verbatim.storage.store import Store

PROFILE = "ptest"


def _cfg():
    return config_from_mapping({"mode": "offline_rules"})


def _host(profile=PROFILE):
    return LocalHost(profile_id=profile, principal_id="me", conversation_id="c1")


def _store(path):
    """Materialize a real (non-empty) store so existence checks are honest."""
    s = Store.create(str(path))
    s.close()
    return str(path)


def _touch(path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "wb") as fh:
        fh.write(b"existing-bytes")
    return str(path)


# ---------------------------------------------------------------------
# inventory matrix
# ---------------------------------------------------------------------


def test_legacy_only_resolves_profile_store(tmp_path):
    legacy = _store(tmp_path / f"{PROFILE}.db")
    r = resolve_store_path(str(tmp_path), profile_id=PROFILE)
    assert r.kind == StoreResolutionKind.LEGACY_PROFILE
    assert r.path == legacy
    assert not r.conflicts
    assert not r.needs_operator_decision()
    assert require_store_path(r) == legacy


def test_v3_only_adopts_v3_store(tmp_path):
    """An existing v3.db keeps serving its profile — it is adopted, never
    replaced by an empty profile store."""
    v3 = _store(tmp_path / "v3.db")
    r = resolve_store_path(str(tmp_path), profile_id=PROFILE)
    assert r.kind == StoreResolutionKind.V3_DEFAULT
    assert r.path == v3
    assert require_store_path(r) == v3


def test_v3_only_without_profile_id_still_resolves(tmp_path):
    v3 = _store(tmp_path / "v3.db")
    r = resolve_store_path(str(tmp_path))
    assert r.kind == StoreResolutionKind.V3_DEFAULT
    assert r.path == v3


def test_both_present_is_a_typed_conflict(tmp_path):
    legacy = _store(tmp_path / f"{PROFILE}.db")
    v3 = _store(tmp_path / "v3.db")
    r = resolve_store_path(str(tmp_path), profile_id=PROFILE)
    assert r.kind == StoreResolutionKind.NONE
    assert r.path is None
    assert set(r.conflicts) == {legacy, v3}
    assert r.needs_operator_decision()
    with pytest.raises(VerbatimError) as exc:
        require_store_path(r)
    assert exc.value.code == ErrorCode.STORE_CONFLICT
    # The error names both candidates and the decision channels.
    assert legacy in str(exc.value) and v3 in str(exc.value)
    # Nothing was touched: no merge, no overwrite, no third store minted.
    dbs = {n for n in os.listdir(tmp_path) if n.endswith(".db")}
    assert dbs == {f"{PROFILE}.db", "v3.db"}


def test_both_present_prefer_selects(tmp_path):
    legacy = _store(tmp_path / f"{PROFILE}.db")
    v3 = _store(tmp_path / "v3.db")
    r = resolve_store_path(str(tmp_path), profile_id=PROFILE, prefer="legacy")
    assert r.kind == StoreResolutionKind.LEGACY_PROFILE
    assert r.path == legacy
    assert not r.needs_operator_decision()
    r = resolve_store_path(str(tmp_path), profile_id=PROFILE, prefer="v3")
    assert r.kind == StoreResolutionKind.V3_DEFAULT
    assert r.path == v3


def test_both_present_explicit_path_selects(tmp_path):
    legacy = _store(tmp_path / f"{PROFILE}.db")
    v3 = _store(tmp_path / "v3.db")
    r = resolve_store_path(
        str(tmp_path), profile_id=PROFILE, explicit_path=v3
    )
    assert r.kind == StoreResolutionKind.EXPLICIT
    assert r.path == v3
    # The bypassed convention stays visible on the resolution.
    assert legacy in r.candidates
    r = resolve_store_path(
        str(tmp_path), profile_id=PROFILE, explicit_path=legacy
    )
    assert r.path == legacy


def test_prefer_literal_path_selects_candidate(tmp_path):
    legacy = _store(tmp_path / f"{PROFILE}.db")
    _store(tmp_path / "v3.db")
    r = resolve_store_path(
        str(tmp_path), profile_id=PROFILE, prefer=legacy
    )
    assert r.path == legacy


def test_neither_present_create_false_is_none(tmp_path):
    r = resolve_store_path(str(tmp_path), profile_id=PROFILE)
    assert r.kind == StoreResolutionKind.NONE
    assert r.path is None
    with pytest.raises(VerbatimError) as exc:
        require_store_path(r)
    assert exc.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_neither_present_create_targets_profile_store(tmp_path):
    """The canonical create target is ``{profile_id}.db`` — the v3
    version-named file is never minted implicitly."""
    r = resolve_store_path(str(tmp_path), profile_id=PROFILE, create=True)
    assert r.kind == StoreResolutionKind.LEGACY_PROFILE
    assert r.path == str(tmp_path / f"{PROFILE}.db")
    # Resolution is pure discovery — it creates nothing itself.
    assert os.listdir(tmp_path) == []


def test_neither_present_create_prefer_v3(tmp_path):
    r = resolve_store_path(
        str(tmp_path), profile_id=PROFILE, create=True, prefer="v3"
    )
    assert r.kind == StoreResolutionKind.V3_DEFAULT
    assert r.path == str(tmp_path / "v3.db")


def test_create_without_nameable_store_validates(tmp_path):
    with pytest.raises(VerbatimError) as exc:
        resolve_store_path(str(tmp_path), create=True)
    assert exc.value.code == ErrorCode.VALIDATION


def test_explicit_path_wins_over_inventory(tmp_path):
    """An explicit path is the operator's decision — it wins even with a
    conflicting inventory, and may name a fresh file."""
    _store(tmp_path / f"{PROFILE}.db")
    target = str(tmp_path / "operator-chosen.db")
    r = resolve_store_path(
        str(tmp_path), profile_id=PROFILE, explicit_path=target
    )
    assert r.kind == StoreResolutionKind.EXPLICIT
    assert r.path == target
    assert require_store_path(r) == target


def test_explicit_directory_rejected(tmp_path):
    with pytest.raises(VerbatimError) as exc:
        resolve_store_path(
            str(tmp_path), profile_id=PROFILE, explicit_path=str(tmp_path)
        )
    assert exc.value.code == ErrorCode.VALIDATION


def test_prefer_naming_no_discovered_store_rejected(tmp_path):
    """``prefer`` must name a discovered store; it cannot quietly bypass
    the one store on disk for a nonexistent rival."""
    _store(tmp_path / f"{PROFILE}.db")
    with pytest.raises(VerbatimError) as exc:
        resolve_store_path(str(tmp_path), profile_id=PROFILE, prefer="v3")
    assert exc.value.code == ErrorCode.VALIDATION


def test_resolver_normalizes_deterministically(tmp_path):
    _store(tmp_path / f"{PROFILE}.db")
    r1 = resolve_store_path(str(tmp_path) + "/./", profile_id=PROFILE)
    r2 = resolve_store_path(
        os.path.join(str(tmp_path), "sub", ".."), profile_id=PROFILE
    )
    assert r1.path == r2.path == str(tmp_path / f"{PROFILE}.db")


def test_existing_store_never_overwritten(tmp_path):
    """Resolving a create target where a store exists returns the existing
    store — the resolution layer never fabricates an empty replacement."""
    legacy = _store(tmp_path / f"{PROFILE}.db")
    size = os.path.getsize(legacy)
    r = resolve_store_path(str(tmp_path), profile_id=PROFILE, create=True)
    assert r.path == legacy
    assert os.path.getsize(legacy) == size


# ---------------------------------------------------------------------
# open_store integration — the resolver is the single policy (V4-07.10)
# ---------------------------------------------------------------------


def test_open_store_uses_resolver_default(tmp_path):
    eng = open_store(str(tmp_path), _cfg(), _host(), create=True)
    try:
        assert eng.store.path == str(tmp_path / f"{PROFILE}.db")
    finally:
        eng.close()


def test_open_store_adopts_v3_only(tmp_path):
    _store(tmp_path / "v3.db")
    eng = open_store(str(tmp_path), _cfg(), _host())
    try:
        assert eng.store.path == str(tmp_path / "v3.db")
    finally:
        eng.close()


def test_open_store_conflict_is_typed(tmp_path):
    _store(tmp_path / f"{PROFILE}.db")
    _store(tmp_path / "v3.db")
    with pytest.raises(VerbatimError) as exc:
        open_store(str(tmp_path), _cfg(), _host())
    assert exc.value.code == ErrorCode.STORE_CONFLICT
    # create=True does not rescue the conflict — no empty replacement.
    with pytest.raises(VerbatimError) as exc2:
        open_store(str(tmp_path), _cfg(), _host(), create=True)
    assert exc2.value.code == ErrorCode.STORE_CONFLICT


def test_open_store_operator_decision_channels(tmp_path):
    _store(tmp_path / f"{PROFILE}.db")
    _store(tmp_path / "v3.db")
    eng = open_store(
        str(tmp_path), _cfg(), _host(), store_path=str(tmp_path / "v3.db")
    )
    try:
        assert eng.store.path == str(tmp_path / "v3.db")
    finally:
        eng.close()
    eng = open_store(str(tmp_path), _cfg(), _host(), prefer_store="legacy")
    try:
        assert eng.store.path == str(tmp_path / f"{PROFILE}.db")
    finally:
        eng.close()
