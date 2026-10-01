"""Deletion closure for the derivation graph (SPEC_V3 §17.03, §34, §36.02–36.09,
§39.02, §40).

"Correct or delete it completely" made real: when evidence is purged, every
derived object reachable through ``derivations`` is either deleted (no
surviving provenance), suppressed (partial provenance loss — invalid until
recomputed, V3-17.03), or marked for revalidation (a parent was suppressed).
Objects local deletion cannot reach — foreign-scope dependents, propagated
copies, unresolvable kinds — are disclosed as ``outside_boundary`` rather than
silently dropped (V3-36.05, V3-36.09).

Three phases, all transaction-scoped:

* ``plan_closure`` — typed plan from the immutable graph snapshot (V3-06.07).
* ``execute_closure`` — validates and applies the plan in the caller's one
  transaction: children before parents, erasure-ledger rows per purged
  object, derivation-edge removal, propagations marked revoked, and an
  inline ``verify_closure`` — a failed verification raises and rolls the
  whole deletion back, so there is never a half-erased closure (V3-36.04).
* ``verify_closure`` — the honest check: any surviving derivation edge or
  member row that still references a purged object is an orphan = failure.

PRIVACY BOUNDARY: ``execute_closure`` is designed to run inside
``privacy_control`` lane jobs (§40 — ``purge_derived``/``purge_vault`` lane),
never inline on recall paths. Composition, not duplication: evidence-plane
byte erasure reuses the v2 purge machinery (``verbatim.purge``) — span purge
still empties the whole parent revision (V2-41.05) and claim erasure still
goes through the lifecycle machine.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .. import derivations as deriv
from ..core.lifecycle import (
    PURGE_ACTOR,
    Lifecycle,
    LifecycleMachine,
    read_claim_head,
)
from ..core.time import wall_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    require_id,
)
from ..purge import (
    _cancel_jobs,
    _erase_artifact,
    _erase_claim,
    _erase_source,
    _erase_source_revision,
    _erase_span,
    _has_table,
)
from ..storage.repos import EventsRepo
from ..storage.repos_v2 import ErasureRepo
from . import erasure as _erasure

ACTION_DELETE = "delete"
ACTION_SUPPRESS = "suppress"
ACTION_REVALIDATE = "revalidate"
_ACTIONS = frozenset({ACTION_DELETE, ACTION_SUPPRESS, ACTION_REVALIDATE})

_OUTSIDE_KIND = "unresolvable_kind"
_OUTSIDE_SCOPE = "foreign_scope"
_OUTSIDE_PROPAGATED = "propagated_copy"

def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    rs = _rows(cur)
    return rs[0] if rs else None


# ----------------------------------------------------------------------
# plan structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedObject:
    """One derived object + the action closure assigns it."""

    kind: str
    object_id: str
    revision: Optional[int]
    action: str
    depth: int
    reason: str = ""

    @property
    def ref(self) -> tuple[str, str, Optional[int]]:
        return (self.kind, self.object_id, self.revision)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "object_id": self.object_id,
            "revision": self.revision,
            "action": self.action,
            "depth": self.depth,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DeletionPlan:
    """Typed closure plan over an immutable snapshot (V3-06.07)."""

    scope_id: str
    roots: tuple  # tuple[ObjectRef] — exact objects to purge
    derived: tuple  # tuple[PlannedObject]
    unaffected_derived: tuple  # tuple[dict] — reached, needed no action
    outside_boundary: tuple  # tuple[dict] — cannot be resolved locally
    digest: str

    def actions(self, action: str) -> list[PlannedObject]:
        return [d for d in self.derived if d.action == action]

    def purged_refs(self) -> set[tuple[str, str, Optional[int]]]:
        """Every reference scheduled to cease existing."""
        out = set(self.roots)
        for d in self.actions(ACTION_DELETE):
            out.add(d.ref)
        return out


# ----------------------------------------------------------------------
# kind → table machinery
# ----------------------------------------------------------------------


def _object_exists(
    conn: sqlite3.Connection,
    ref: tuple[str, str, Optional[int]],
    scope_id: Optional[str] = None,
) -> bool:
    kind, oid, rev = ref
    if kind == "claim":
        return conn.execute(
            "SELECT 1 FROM claims WHERE claim_id = ?", (oid,)
        ).fetchone() is not None
    if kind == "source":
        return conn.execute(
            "SELECT 1 FROM sources WHERE source_id = ?", (oid,)
        ).fetchone() is not None
    if kind == "source_revision":
        if rev is None:
            return conn.execute(
                "SELECT 1 FROM source_revisions WHERE source_id = ?", (oid,)
            ).fetchone() is not None
        return conn.execute(
            "SELECT 1 FROM source_revisions WHERE source_id = ? AND revision = ?",
            (oid, rev),
        ).fetchone() is not None
    if kind == "span":
        if rev is None:
            return conn.execute(
                "SELECT 1 FROM spans WHERE span_id = ?", (oid,)
            ).fetchone() is not None
        return conn.execute(
            "SELECT 1 FROM spans WHERE span_id = ? AND revision = ?",
            (oid, rev),
        ).fetchone() is not None
    if kind == "envelope" and _has_table(conn, "source_envelopes"):
        return conn.execute(
            "SELECT 1 FROM source_envelopes WHERE envelope_id = ?", (oid,)
        ).fetchone() is not None
    if kind == "artifact" and _has_table(conn, "artifacts"):
        return conn.execute(
            "SELECT 1 FROM artifacts WHERE artifact_id = ?", (oid,)
        ).fetchone() is not None
    simple = {
        "episode": ("episodes", "episode_id"),
        "transition": ("transitions", "transition_id"),
        "procedure": ("procedures", "procedure_id"),
        "observation": ("observations", "observation_id"),
        "plan": ("prospective_records", "record_id"),
        "social": ("social_memory", "record_id"),
        "trajectory": ("trajectories", "trajectory_id"),
        "trajectory_step": ("trajectory_steps", "step_id"),
        "state_anchor": ("state_anchors", "anchor_id"),
        "vault_entry": ("vault_entries", "entry_id"),
    }
    if kind in simple and _has_table(conn, simple[kind][0]):
        table, col = simple[kind]
        return conn.execute(
            f"SELECT 1 FROM {table} WHERE {col} = ?", (oid,)
        ).fetchone() is not None
    if kind == "environment" and _has_table(conn, "environment_state"):
        if scope_id is None:
            return conn.execute(
                "SELECT 1 FROM environment_state WHERE key = ?", (oid,)
            ).fetchone() is not None
        return conn.execute(
            "SELECT 1 FROM environment_state WHERE scope_id = ? AND key = ?",
            (scope_id, oid),
        ).fetchone() is not None
    if kind == "working":
        if _has_table(conn, "working_sets") and conn.execute(
            "SELECT 1 FROM working_sets WHERE set_id = ?", (oid,)
        ).fetchone() is not None:
            return True
        return _has_table(conn, "working_set_items") and conn.execute(
            "SELECT 1 FROM working_set_items WHERE item_id = ?", (oid,)
        ).fetchone() is not None
    if kind == "profile" and _has_table(conn, "profile_entries"):
        return conn.execute(
            "SELECT 1 FROM profile_entries WHERE entry_id = ?", (oid,)
        ).fetchone() is not None
    if kind in _OBJECTS_DISPOSITION_KINDS and _has_table(conn, "objects"):
        return conn.execute(
            "SELECT 1 FROM objects WHERE kind = ? AND object_id = ?",
            (kind, oid),
        ).fetchone() is not None
    return False


#: Kind → (table, pk-column) for direct scope lookups — shared with the
#: closure engine's batched resolver (``closure._scope_bulk``).
_SCOPE_SIMPLE: dict[str, tuple[str, str]] = {
    "claim": ("claims", "claim_id"),
    "source": ("sources", "source_id"),
    "envelope": ("source_envelopes", "envelope_id"),
    "artifact": ("artifacts", "artifact_id"),
    "episode": ("episodes", "episode_id"),
    "transition": ("transitions", "transition_id"),
    "procedure": ("procedures", "procedure_id"),
    "observation": ("observations", "observation_id"),
    "plan": ("prospective_records", "record_id"),
    "social": ("social_memory", "record_id"),
    "trajectory": ("trajectories", "trajectory_id"),
    "trajectory_step": ("trajectory_steps", "step_id"),
    "state_anchor": ("state_anchors", "anchor_id"),
    "vault_entry": ("vault_entries", "entry_id"),
    "profile": ("profile_entries", "entry_id"),
}


def _object_scope(
    conn: sqlite3.Connection, ref: tuple[str, str, Optional[int]]
) -> Optional[str]:
    """Owning scope of an object, or None when it cannot be determined."""
    kind, oid, _rev = ref
    simple = _SCOPE_SIMPLE
    if kind in simple and _has_table(conn, simple[kind][0]):
        table, col = simple[kind]
        r = conn.execute(
            f"SELECT scope_id FROM {table} WHERE {col} = ?", (oid,)
        ).fetchone()
        if r:
            return r[0]
    if kind == "source_revision":
        r = conn.execute(
            "SELECT scope_id FROM sources WHERE source_id = ?", (oid,)
        ).fetchone()
        if r:
            return r[0]
    if kind == "span":
        r = conn.execute(
            "SELECT so.scope_id FROM spans sp"
            " JOIN sources so ON so.source_id = sp.source_id"
            " WHERE sp.span_id = ?",
            (oid,),
        ).fetchone()
        if r:
            return r[0]
    if kind == "environment" and _has_table(conn, "environment_state"):
        r = conn.execute(
            "SELECT scope_id FROM environment_state WHERE key = ? LIMIT 1", (oid,)
        ).fetchone()
        if r:
            return r[0]
    if kind == "working":
        for table, col in (
            ("working_sets", "set_id"),
            ("working_set_items", "item_id"),
        ):
            if _has_table(conn, table):
                if table == "working_sets":
                    r = conn.execute(
                        f"SELECT scope_id FROM {table} WHERE {col} = ?", (oid,)
                    ).fetchone()
                else:
                    r = conn.execute(
                        "SELECT ws.scope_id FROM working_set_items wi"
                        " JOIN working_sets ws ON ws.set_id = wi.set_id"
                        " WHERE wi.item_id = ?",
                        (oid,),
                    ).fetchone()
                if r:
                    return r[0]
    if kind in _OBJECTS_DISPOSITION_KINDS and _has_table(conn, "objects"):
        r = conn.execute(
            "SELECT scope_id FROM objects WHERE kind = ? AND object_id = ?",
            (kind, oid),
        ).fetchone()
        if r:
            return r[0]
    # Fallback: the derivation graph itself is scope-tagged.
    r = conn.execute(
        "SELECT scope_id FROM derivations"
        " WHERE (child_kind = ? AND child_id = ?)"
        "    OR (parent_kind = ? AND parent_id = ?) LIMIT 1",
        (kind, oid, kind, oid),
    ).fetchone()
    return r[0] if r else None


# ----------------------------------------------------------------------
# per-kind physical effects
# ----------------------------------------------------------------------


def _tombstone_branch(
    conn: sqlite3.Connection, store: Any, oid: str, scope_id: str
) -> None:
    """Scrub a branch object to a digest-bound tombstone (V45-12.03).

    Mirrors the derived-view erase idiom: the ``objects`` row flips to
    ``erased`` and each ``object_revisions`` doc is replaced by a minimal
    tombstone carrying no proposed text — the erasure ledger row the
    engine writes is the durable record of what was removed.
    """
    rows = conn.execute(
        "SELECT revision FROM object_revisions"
        " WHERE kind = 'branch' AND object_id = ?",
        (oid,),
    ).fetchall()
    for (rev,) in rows:
        tombstone = {
            "doc": "branch/v1",
            "branch_id": oid,
            "revision": int(rev),
            "scope_id": scope_id,
            "state": "tombstoned",
            "erased": True,
        }
        digest = "hmac-sha256:" + store.hmac(
            json_dumps(tombstone).encode("utf-8")
        ).hex()
        conn.execute(
            "UPDATE object_revisions SET metadata_json = ?, digest = ?"
            " WHERE kind = 'branch' AND object_id = ? AND revision = ?",
            (json_dumps(tombstone), digest, oid, int(rev)),
        )
    conn.execute(
        "UPDATE objects SET disposition = 'erased'"
        " WHERE kind = 'branch' AND object_id = ?",
        (oid,),
    )


def _tombstone_view(
    conn: sqlite3.Connection, store: Any, oid: str, scope_id: str
) -> None:
    """Scrub a derived view to a digest-bound tombstone.

    Same erase idiom as branches: the ``objects`` row flips to
    ``erased`` and every ``object_revisions`` doc is replaced by a
    minimal tombstone — a view that carried purged text can never stay
    servable, while the erasure ledger row the engine writes remains
    the durable record.
    """
    rows = conn.execute(
        "SELECT revision FROM object_revisions"
        " WHERE kind = 'derived_view' AND object_id = ?",
        (oid,),
    ).fetchall()
    for (rev,) in rows:
        tombstone = {
            "doc": "derived_view/v1",
            "view_id": oid,
            "revision": int(rev),
            "scope_id": scope_id,
            "state": "tombstoned",
            "erased": True,
        }
        digest = "hmac-sha256:" + store.hmac(
            json_dumps(tombstone).encode("utf-8")
        ).hex()
        conn.execute(
            "UPDATE object_revisions SET metadata_json = ?, digest = ?"
            " WHERE kind = 'derived_view' AND object_id = ?"
            " AND revision = ?",
            (json_dumps(tombstone), digest, oid, int(rev)),
        )
    conn.execute(
        "UPDATE objects SET disposition = 'erased'"
        " WHERE kind = 'derived_view' AND object_id = ?",
        (oid,),
    )


def _delete_object(
    conn: sqlite3.Connection,
    store: Any,
    machine: LifecycleMachine,
    ref: tuple[str, str, Optional[int]],
    purge_id: str,
    scope_id: str,
) -> None:
    """Apply the kind's erasure semantics for one object.

    Evidence kinds reuse the v2 byte-scrubbing machinery (skeletons stay so
    residual lineage cannot crash); derived learning objects are removed
    with their member rows — the erasure ledger tombstone is what persists.
    """
    kind, oid, rev = ref
    if kind == "claim":
        _erase_claim(conn, store, machine, oid, purge_id)
        return
    if kind == "span":
        _erase_span(conn, store, oid)
        return
    if kind == "source":
        _erase_source(conn, store, oid)
        return
    if kind == "source_revision":
        if rev is not None:
            _erase_source_revision(conn, store, oid, rev)
        else:
            _erase_source(conn, store, oid)
        return
    if kind == "artifact":
        _erase_artifact(conn, oid)
        return
    if kind == "envelope":
        conn.execute(
            "DELETE FROM step_observations WHERE envelope_id = ?", (oid,)
        )
        conn.execute(
            "DELETE FROM source_envelopes WHERE envelope_id = ?", (oid,)
        )
        return
    if kind == "episode":
        tids = [
            r[0]
            for r in conn.execute(
                "SELECT transition_id FROM transitions WHERE episode_id = ?",
                (oid,),
            ).fetchall()
        ]
        for tid in tids:
            conn.execute(
                "DELETE FROM transition_anchors WHERE transition_id = ?", (tid,)
            )
        conn.execute("DELETE FROM transitions WHERE episode_id = ?", (oid,))
        conn.execute("DELETE FROM episode_members WHERE episode_id = ?", (oid,))
        conn.execute("DELETE FROM episodes WHERE episode_id = ?", (oid,))
        return
    if kind == "transition":
        conn.execute(
            "DELETE FROM transition_anchors WHERE transition_id = ?", (oid,)
        )
        conn.execute("DELETE FROM transitions WHERE transition_id = ?", (oid,))
        return
    if kind == "procedure":
        conn.execute(
            "DELETE FROM procedure_steps WHERE procedure_id = ?", (oid,)
        )
        conn.execute(
            "DELETE FROM procedure_signatures WHERE procedure_id = ?", (oid,)
        )
        conn.execute(
            "DELETE FROM procedure_exposures WHERE procedure_id = ?", (oid,)
        )
        conn.execute(
            "DELETE FROM outcome_receipts WHERE procedure_id = ?", (oid,)
        )
        conn.execute("DELETE FROM procedures WHERE procedure_id = ?", (oid,))
        return
    if kind == "observation":
        conn.execute(
            "DELETE FROM observation_evidence WHERE observation_id = ?", (oid,)
        )
        conn.execute(
            "DELETE FROM observations WHERE observation_id = ?", (oid,)
        )
        return
    if kind == "plan":
        conn.execute(
            "DELETE FROM prospective_records WHERE record_id = ?", (oid,)
        )
        return
    if kind == "social":
        conn.execute("DELETE FROM social_memory WHERE record_id = ?", (oid,))
        return
    if kind == "environment":
        conn.execute(
            "DELETE FROM environment_state WHERE scope_id = ? AND key = ?",
            (scope_id, oid),
        )
        return
    if kind == "working":
        conn.execute(
            "DELETE FROM working_set_items WHERE set_id = ? OR item_id = ?",
            (oid, oid),
        )
        conn.execute("DELETE FROM working_sets WHERE set_id = ?", (oid,))
        return
    if kind == "trajectory":
        sids = [
            r[0]
            for r in conn.execute(
                "SELECT step_id FROM trajectory_steps WHERE trajectory_id = ?",
                (oid,),
            ).fetchall()
        ]
        for sid_ in sids:
            conn.execute(
                "DELETE FROM step_observations WHERE step_id = ?", (sid_,)
            )
        conn.execute(
            "DELETE FROM trajectory_steps WHERE trajectory_id = ?", (oid,)
        )
        conn.execute("DELETE FROM trajectories WHERE trajectory_id = ?", (oid,))
        return
    if kind == "trajectory_step":
        conn.execute("DELETE FROM step_observations WHERE step_id = ?", (oid,))
        conn.execute("DELETE FROM trajectory_steps WHERE step_id = ?", (oid,))
        return
    if kind == "state_anchor":
        conn.execute(
            "DELETE FROM transition_anchors WHERE anchor_id = ?", (oid,)
        )
        conn.execute("DELETE FROM state_anchors WHERE anchor_id = ?", (oid,))
        return
    if kind == "vault_entry":
        # Vault erasure is a tombstone (erased_event) + vault_refs scrub —
        # outstanding handles fail closed; the closure engine adds the
        # restore-fence ledger digest itself (V4-38.05). No vault schema →
        # nothing to erase (the gone-check above already returns True).
        if _has_table(conn, "vault_entries"):
            _erasure.erase_entry(conn, oid)
        return
    if kind == "profile":
        # Derived state erases wholesale — every revision, its support
        # pointers, and its audit rows go; the erasure ledger tombstone
        # the engine writes is the durable record (§36). Conflict groups
        # lose the erased member; a group reduced below two alternatives
        # cannot stand as an open contradiction.
        if _has_table(conn, "profile_entries"):
            conn.execute(
                "DELETE FROM profile_entry_support WHERE entry_id = ?",
                (oid,),
            )
            conn.execute(
                "DELETE FROM profile_entries WHERE entry_id = ?", (oid,)
            )
        if _has_table(conn, "profile_conflicts"):
            for crow in conn.execute(
                "SELECT scope_id, conflict_group, members_json"
                " FROM profile_conflicts"
            ).fetchall():
                try:
                    members = json.loads(crow[2] or "[]")
                except (TypeError, ValueError):
                    members = []
                if oid not in members:
                    continue
                members = [m for m in members if m != oid]
                if len(members) < 2:
                    conn.execute(
                        "DELETE FROM profile_conflicts"
                        " WHERE scope_id = ? AND conflict_group = ?",
                        (crow[0], crow[1]),
                    )
                else:
                    conn.execute(
                        "UPDATE profile_conflicts SET members_json = ?"
                        " WHERE scope_id = ? AND conflict_group = ?",
                        (json.dumps(sorted(members)), crow[0], crow[1]),
                    )
        if _has_table(conn, "profile_events"):
            conn.execute(
                "DELETE FROM profile_events WHERE entry_id = ?", (oid,)
            )
        return
    if kind == "branch":
        # V45-12.03: branches join deletion closure as tombstones — the
        # objects row flips to ``erased`` and every revision doc is
        # scrubbed to a digest-bound tombstone so proposed op text (the
        # only content a branch ever held) is gone, while the erasure
        # ledger + the tombstone itself remain the durable record.
        if _has_table(conn, "objects"):
            _tombstone_branch(conn, store, oid, scope_id)
        return
    if kind == "derived_view":
        # Same registry erase idiom: a view bound to purged evidence
        # can never stay servable — its revisions become tombstones.
        if _has_table(conn, "objects"):
            _tombstone_view(conn, store, oid, scope_id)
        return
    raise VerbatimError(
        ErrorCode.VALIDATION, f"cannot delete object kind {kind!r}"
    )


def _apply_suppression(
    conn: sqlite3.Connection,
    ref: tuple[str, str, Optional[int]],
    scope_id: str,
    seq: int,
) -> None:
    """Reversible invalidation marker — same idiom as v2 tombstones.

    The kind's own recorded-time/availability column is authoritative; when
    the kind has none — or the row itself is absent (a phantom edge) — the
    quarantine registry carries the suppression state instead (§34.03:
    release/suppress/purge are separate audited states).
    """
    kind, oid, rev = ref
    marked = False
    if kind == "claim":
        head = read_claim_head(conn, oid)
        if head is not None:
            if head.recorded_until is None:
                conn.execute(
                    "UPDATE claim_revisions SET recorded_until = ?"
                    " WHERE claim_id = ? AND revision = ?",
                    (seq, oid, head.revision),
                )
            marked = True
    else:
        dated = _DATED_TABLES.get(kind)
        if dated is not None and _has_table(conn, dated[0]):
            r = conn.execute(
                f"SELECT recorded_until FROM {dated[0]} WHERE {dated[1]} = ?",
                (oid,),
            ).fetchone()
            if r is not None:
                if r[0] is None:
                    conn.execute(
                        f"UPDATE {dated[0]} SET recorded_until = ?"
                        f" WHERE {dated[1]} = ?",
                        (seq, oid),
                    )
                marked = True
        elif kind == "artifact" and _has_table(conn, "artifacts"):
            if conn.execute(
                "SELECT 1 FROM artifacts WHERE artifact_id = ?", (oid,)
            ).fetchone() is not None:
                conn.execute(
                    "UPDATE artifacts SET availability = 'suppressed'"
                    " WHERE artifact_id = ? AND availability = 'available'",
                    (oid,),
                )
                marked = True
        elif kind in _OBJECTS_DISPOSITION_KINDS and _has_table(
            conn, "objects"
        ):
            # Registry-carried kinds suppress through their disposition —
            # a partially-purged branch holds until an operator rebases
            # it (V45-12.03), a partially-purged view holds exactly like
            # ``Synthesizer.note_invalidation`` marks it. Neither is
            # auto-resurrected.
            r = conn.execute(
                "SELECT disposition FROM objects"
                " WHERE kind = ? AND object_id = ?",
                (kind, oid),
            ).fetchone()
            if r is not None:
                if r[0] == "active":
                    conn.execute(
                        "UPDATE objects SET disposition = 'held'"
                        " WHERE kind = ? AND object_id = ?",
                        (kind, oid),
                    )
                marked = True
    if not marked and _has_table(conn, "quarantine"):
        conn.execute(
            "INSERT OR REPLACE INTO quarantine"
            "(object_kind, object_id, revision, scope_id, reason_codes_json,"
            " findings_json, state, opened_event, decision_json)"
            " VALUES (?, ?, ?, ?, '[]', '[]', 'suppressed', ?, '{}')",
            (kind, oid, rev if rev is not None else 1, scope_id, seq),
        )


def _apply_revalidation(
    conn: sqlite3.Connection,
    ref: tuple[str, str, Optional[int]],
    scope_id: str,
    seq: int,
) -> None:
    """Flag an object as needing recomputation (§24 freshness machinery)."""
    kind, oid, rev = ref
    if _has_table(conn, "freshness"):
        conn.execute(
            "INSERT OR REPLACE INTO freshness"
            "(scope_id, object_kind, object_id, revision, class)"
            " VALUES (?, ?, ?, ?, 'revalidate_after')",
            (scope_id, kind, oid, rev if rev is not None else 1),
        )
    if kind == "observation":
        conn.execute(
            "UPDATE observations SET stale_since_seq = COALESCE(stale_since_seq, ?)"
            " WHERE observation_id = ?",
            (seq, oid),
        )


# ----------------------------------------------------------------------
# plan-time classification
# ----------------------------------------------------------------------


def plan_closure(
    conn: sqlite3.Connection, object_refs: Iterable[Any]
) -> DeletionPlan:
    """Compute the deletion-closure plan for purging ``object_refs``.

    Action semantics (§36): a derived object whose derivation parents are
    *all* purged is ``delete`` (nothing left to stand on); one with mixed
    ancestry is ``suppress`` (invalid until recomputed — V3-17.03); one
    whose parents are suppressed or flagged is ``revalidate``. Objects the
    closure cannot act on locally are reported ``outside_boundary``.
    """
    affected = deriv.affected_by_purge(conn, object_refs)
    roots: list[tuple[str, str, Optional[int]]] = affected["roots"]
    if not roots:
        raise VerbatimError(
            ErrorCode.VALIDATION, "closure needs at least one purge target"
        )

    scopes: set[str] = set()
    for ref in roots:
        if not _object_exists(conn, ref):
            # Missing and foreign objects share one response (§10.05).
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                f"{ref[0]} {ref[1]!r} not found",
            )
        sid = _object_scope(conn, ref)
        if sid is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                f"{ref[0]} {ref[1]!r} not found",
            )
        scopes.add(sid)
    if len(scopes) != 1:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "deletion closure is a single-scope operation",
        )
    scope_id = scopes.pop()

    descendants: dict[tuple[str, str, int], int] = dict(affected["derived"])
    outside: list[dict[str, Any]] = []
    outside_refs: set[tuple[str, str, int]] = set()
    resolvable: dict[tuple[str, str, int], int] = {}
    for ref, depth in sorted(descendants.items()):
        kind, oid, _r = ref
        if kind not in _RESOLVABLE_KINDS:
            outside_refs.add(ref)
            outside.append(
                {"ref": ref, "reason": _OUTSIDE_KIND,
                 "detail": f"no local table for kind {kind!r}"}
            )
            continue
        d_scope = _object_scope(conn, ref)
        if d_scope is not None and d_scope != scope_id:
            outside_refs.add(ref)
            outside.append(
                {"ref": ref, "reason": _OUTSIDE_SCOPE,
                 "detail": f"object is in scope {d_scope}"}
            )
            continue
        resolvable[ref] = depth

    purged: set[tuple[str, str, Optional[int]]] = set(roots)

    def _in_purged(ref: tuple[str, str, int]) -> bool:
        k, i, r = ref
        return (k, i, r) in purged or (k, i, None) in purged

    # Fixpoint: a derived object is 'delete' once every recorded parent is
    # itself deleted. Iterating to a fixed point propagates deletion down
    # chains (claim → procedure → deeper dependents).
    actions: dict[tuple[str, str, int], PlannedObject] = {}
    progressed = True
    while progressed:
        progressed = False
        for ref, depth in resolvable.items():
            if ref in actions:
                continue
            parents = deriv.parents_of(conn, ref)
            if parents and all(_in_purged(p) for p in parents):
                actions[ref] = PlannedObject(
                    ref[0], ref[1], ref[2], ACTION_DELETE, depth,
                    "all_parents_purged",
                )
                purged.add(ref)
                progressed = True

    # 'suppress': partial provenance loss — at least one parent purged, at
    # least one survives.
    for ref, depth in resolvable.items():
        if ref in actions:
            continue
        parents = deriv.parents_of(conn, ref)
        if any(_in_purged(p) for p in parents):
            actions[ref] = PlannedObject(
                ref[0], ref[1], ref[2], ACTION_SUPPRESS, depth,
                "mixed_ancestry",
            )

    # 'revalidate': no parent purged, but at least one parent is suppressed
    # or flagged — its derivation basis is prospectively invalid (V3-17.03).
    suppressed = {r for r, a in actions.items() if a.action == ACTION_SUPPRESS}
    flagged = set(suppressed)
    progressed = True
    while progressed:
        progressed = False
        for ref, depth in resolvable.items():
            if ref in actions:
                continue
            parents = deriv.parents_of(conn, ref)
            if any(p in flagged for p in parents):
                actions[ref] = PlannedObject(
                    ref[0], ref[1], ref[2], ACTION_REVALIDATE, depth,
                    "parent_invalidated",
                )
                flagged.add(ref)
                progressed = True

    unaffected: list[dict[str, Any]] = []
    for ref, depth in resolvable.items():
        if ref in actions:
            continue
        reasons = []
        if not _object_exists(conn, ref, scope_id):
            reasons.append("already_absent")
        if any(p in outside_refs for p in deriv.parents_of(conn, ref)):
            reasons.append("parent_outside_boundary")
        unaffected.append(
            {"ref": ref, "depth": depth,
             "reason": ",".join(reasons) or "parents_survived"}
        )

    # Propagated copies of everything the closure erases cannot be remotely
    # erased here — mark for revocation at execute, disclose now (V3-36.05).
    if _has_table(conn, "propagations"):
        for ref in sorted(purged):
            k, i, r = ref
            for prow in _rows(
                conn.execute(
                    "SELECT propagation_id, recipient_id FROM propagations"
                    " WHERE object_kind = ? AND object_id = ?"
                    " AND revoked_seq IS NULL",
                    (k, i),
                )
            ):
                outside.append(
                    {
                        "ref": ref,
                        "reason": _OUTSIDE_PROPAGATED,
                        "detail": f"recipient {prow['recipient_id']} holds a copy",
                        "propagation_id": prow["propagation_id"],
                        "recipient_id": prow["recipient_id"],
                    }
                )

    derived_objs = tuple(
        actions[r] for r in sorted(actions, key=lambda x: (-resolvable[x], x))
    )
    digest = hashlib.sha256(
        json_dumps(
            {
                "scope_id": scope_id,
                "roots": sorted(roots),
                "derived": [d.as_dict() for d in derived_objs],
            }
        ).encode("utf-8")
    ).hexdigest()
    return DeletionPlan(
        scope_id=scope_id,
        roots=tuple(sorted(roots)),
        derived=derived_objs,
        unaffected_derived=tuple(unaffected),
        outside_boundary=tuple(outside),
        digest=digest,
    )


#: Kinds whose table carries a ``recorded_until`` bitemporal column —
#: suppression closes it (the v2 tombstone idiom).
_DATED_TABLES = {
    "episode": ("episodes", "episode_id"),
    "procedure": ("procedures", "procedure_id"),
    "observation": ("observations", "observation_id"),
    "plan": ("prospective_records", "record_id"),
}

#: Kinds the executor can resolve to physical effects. Unknown kinds are
#: deliberately absent — they report as outside_boundary.
_RESOLVABLE_KINDS = frozenset({
    "claim",
    "source",
    "source_revision",
    "span",
    "envelope",
    "artifact",
    "episode",
    "transition",
    "procedure",
    "observation",
    "plan",
    "social",
    "environment",
    "working",
    "trajectory",
    "trajectory_step",
    "state_anchor",
    "vault_entry",
    "profile",
    "branch",
    "derived_view",
})

#: Kinds whose suppression marker is ``objects.disposition`` — the
#: generic registry carries their lifecycle, so the recorded-time and
#: quarantine fallbacks do not apply. ``closure._suppress_batch`` uses
#: the same set for its batched path.
_OBJECTS_DISPOSITION_KINDS = frozenset({"branch", "derived_view"})


# ----------------------------------------------------------------------
# execution — one transaction, children before parents, verified inline
# ----------------------------------------------------------------------


def execute_closure(
    conn: sqlite3.Connection,
    store: Any,
    plan: DeletionPlan,
    purge_id: str,
) -> dict[str, Any]:
    """Apply a DeletionPlan inside the caller's transaction.

    Order: purge registry → derived deletes (leaf-removal order: a deleted
    object's derivation-children go first) → suppression markers →
    revalidation flags → derivation-edge removal → secondary-reference
    stripping → erasure ledger → propagation revocation marks → job
    cancellation → epochs/generation → receipt event → inline verify.
    Any failure — including the final verify — propagates so the whole
    transaction rolls back; partial closure never commits.
    """
    require_id(purge_id, "purge_id")
    sid = plan.scope_id

    for ref in plan.roots:
        if not _object_exists(conn, ref, sid):
            raise VerbatimError(
                ErrorCode.STALE_PROPOSAL,
                f"plan stale: {ref[0]} {ref[1]!r} no longer present",
            )

    # Purge registry: reuse the v2 tombstone table so one registry answers
    # "what is suppressed/purged" across planes.
    prow = _row(
        conn.execute(
            "SELECT state FROM purges WHERE purge_id = ?", (purge_id,)
        )
    )
    if prow is None:
        digest = store.hmac(plan.digest.encode("utf-8"))
        conn.execute(
            "INSERT INTO purges"
            "(purge_id, selection_digest, scope_id, state, requested_us)"
            " VALUES (?, ?, ?, 'purging', ?)",
            (purge_id, digest, sid, wall_us()),
        )
        for k, i, _r in plan.roots:
            conn.execute(
                "INSERT INTO purge_targets(purge_id, object_kind, object_id)"
                " VALUES (?, ?, ?)",
                (purge_id, k, i),
            )
    elif prow["state"] in ("previewed", "suppressed"):
        conn.execute(
            "UPDATE purges SET state = 'purging' WHERE purge_id = ?",
            (purge_id,),
        )
    else:
        raise VerbatimError(
            ErrorCode.STALE_PROPOSAL,
            f"purge in state {prow['state']!r} cannot execute",
        )

    epoch = _bump_erasure_epoch(store, conn)
    seq = _next_seq(conn)
    machine = LifecycleMachine(store)
    erasure = ErasureRepo(store)
    purged = plan.purged_refs()
    outside_refs = {tuple(o["ref"]) for o in plan.outside_boundary if "ref" in o}

    # 1. Derived deletes — leaf-removal order over the delete set so a
    #    deleted object's derivation-children are always deleted first.
    delete_set = {d.ref for d in plan.actions(ACTION_DELETE)}
    for ref in _leaf_order(conn, delete_set):
        _delete_object(conn, store, machine, ref, purge_id, sid)

    # 2. Evidence roots.
    for ref in plan.roots:
        _delete_object(conn, store, machine, ref, purge_id, sid)

    # 3. Suppression markers (reversible — recorded_until/quarantine).
    for d in plan.actions(ACTION_SUPPRESS):
        _apply_suppression(conn, d.ref, sid, seq)

    # 4. Revalidation flags.
    for d in plan.actions(ACTION_REVALIDATE):
        _apply_revalidation(conn, d.ref, sid, seq)

    # 5. Derivation edges: closure extends to the graph itself (V3-36.02).
    #    Every edge that names a purged object on either side is removed —
    #    including edges owned by outside-boundary children (the pointer
    #    into erased material is what must go).
    for k, i, r in purged:
        if r is None:
            conn.execute(
                "DELETE FROM derivations WHERE (child_kind = ? AND child_id = ?)"
                " OR (parent_kind = ? AND parent_id = ?)",
                (k, i, k, i),
            )
        else:
            conn.execute(
                "DELETE FROM derivations WHERE"
                " (child_kind = ? AND child_id = ? AND child_revision = ?)"
                " OR (parent_kind = ? AND parent_id = ? AND parent_revision = ?)",
                (k, i, r, k, i, r),
            )
        # Legacy v2 dependency rows are scrubbed at object granularity.
        if _has_table(conn, "dependency_refs"):
            conn.execute(
                "DELETE FROM dependency_refs"
                " WHERE (derived_kind = ? AND derived_id = ?)"
                "    OR (input_kind = ? AND input_id = ?)",
                (k, i, k, i),
            )

    # 6. Secondary references held by *surviving* objects: a suppressed or
    #    unrelated row must not keep a live pointer into erased material.
    _strip_references(conn, purged)

    # 7. Erasure ledger tombstones for everything physically erased
    #    (V3-36.03: opaque non-content tombstones prevent resurrection).
    ledger_rows = 0
    for k, i, r in sorted(purged):
        lid = f"{i}:{r}" if (k == "source_revision" and r is not None) else i
        erasure.record(
            conn, sid, k, lid, purge_id=purge_id, erasure_epoch=epoch
        )
        ledger_rows += 1

    # 8. Propagated copies: mark revoked in the ledger now; notification is
    #    the separate revocation_notify lane job (V3-36.05).
    if _has_table(conn, "propagations"):
        for k, i, _r in purged:
            conn.execute(
                "UPDATE propagations SET revoked_seq = ?"
                " WHERE object_kind = ? AND object_id = ?"
                " AND revoked_seq IS NULL",
                (seq, k, i),
            )

    # 9. Pending work referencing purged objects cannot recreate them.
    jobs = _cancel_jobs(conn, sid, [(k, i) for k, i, _r in purged])
    generation = _bump_generation(store, conn)

    conn.execute(
        "UPDATE purges SET state = 'completed', completed_us = ?"
        " WHERE purge_id = ?",
        (wall_us(), purge_id),
    )
    event_seq = EventsRepo(store).append(
        conn,
        sid,
        "deletion_closure_executed",
        PURGE_ACTOR,
        {
            "purge_id": purge_id,
            "plan_digest": plan.digest,
            "roots": [list(r) for r in plan.roots],
            "deleted": len(delete_set),
            "suppressed": len(plan.actions(ACTION_SUPPRESS)),
            "revalidate": len(plan.actions(ACTION_REVALIDATE)),
            "outside_boundary": len(plan.outside_boundary),
            "erasure_epoch": epoch,
        },
        "policy-1",
    )

    verification = verify_closure(conn, plan)
    if not verification["closed"]:
        raise VerbatimError(
            ErrorCode.INTEGRITY,
            f"deletion closure incomplete: {verification['orphans']!r}",
        )
    return {
        "purge_id": purge_id,
        "scope_id": sid,
        "state": "completed",
        "plan_digest": plan.digest,
        "erased": sorted(purged),
        "suppressed": [d.ref for d in plan.actions(ACTION_SUPPRESS)],
        "revalidate": [d.ref for d in plan.actions(ACTION_REVALIDATE)],
        "unaffected_derived": list(plan.unaffected_derived),
        "outside_boundary": list(plan.outside_boundary),
        "ledger_rows": ledger_rows,
        "jobs_cancelled": jobs,
        "erasure_epoch": epoch,
        "projection_generation": generation,
        "event_seq": event_seq,
        "verification": verification,
    }


def _leaf_order(
    conn: sqlite3.Connection,
    delete_set: set[tuple[str, str, Optional[int]]],
) -> list[tuple[str, str, Optional[int]]]:
    """Deletion order: derivation-children before their deleted parents.

    Repeatedly removes objects that have no children left in the delete
    set. A cycle degenerates to deterministic order — intra-cycle FK
    constraints between distinct objects do not exist (member-row FKs are
    handled inside each kind's deleter).
    """
    remaining = set(delete_set)
    order: list[tuple[str, str, Optional[int]]] = []
    while remaining:
        leaves = sorted(
            r
            for r in remaining
            if not any(c in remaining for c in deriv.children_of(conn, r))
        )
        if not leaves:
            leaves = [sorted(remaining)[0]]
        for leaf in leaves:
            order.append(leaf)
            remaining.discard(leaf)
    return order


def _strip_references(
    conn: sqlite3.Connection, purged: set[tuple[str, str, Optional[int]]]
) -> None:
    """Remove surviving rows' secondary references to purged objects."""
    for k, i, r in sorted(purged):
        if r is None:
            conn.execute(
                "DELETE FROM observation_evidence"
                " WHERE object_kind = ? AND object_id = ?",
                (k, i),
            )
        else:
            conn.execute(
                "DELETE FROM observation_evidence"
                " WHERE object_kind = ? AND object_id = ?"
                " AND object_revision = ?",
                (k, i, r),
            )
        for table in ("episode_members", "family_members", "capsule_members",
                      "decision_inputs", "artifact_links"):
            if _has_table(conn, table):
                conn.execute(
                    f"DELETE FROM {table} WHERE object_kind = ? AND object_id = ?",
                    (k, i),
                )
        if k == "span":
            conn.execute(
                "DELETE FROM claim_evidence WHERE span_id = ?", (i,)
            )
            if _has_table(conn, "procedure_steps"):
                conn.execute(
                    "UPDATE procedure_steps SET span_id = NULL WHERE span_id = ?",
                    (i,),
                )
        if _has_table(conn, "profile_entry_support"):
            # Profile support pointers to a purged object go with it —
            # the entry itself is invalidated by the missing derivation
            # parents (reads fail closed; refresh() tombstones it).
            if r is None:
                conn.execute(
                    "DELETE FROM profile_entry_support"
                    " WHERE support_kind = ? AND support_id = ?",
                    (k, i),
                )
            else:
                conn.execute(
                    "DELETE FROM profile_entry_support"
                    " WHERE support_kind = ? AND support_id = ?"
                    " AND support_revision = ?",
                    (k, i, r),
                )
        if k == "state_anchor" and _has_table(conn, "transition_anchors"):
            conn.execute(
                "DELETE FROM transition_anchors WHERE anchor_id = ?", (i,)
            )
        if k == "envelope" and _has_table(conn, "step_observations"):
            conn.execute(
                "DELETE FROM step_observations WHERE envelope_id = ?", (i,)
            )
        if _has_table(conn, "influence"):
            if r is None:
                conn.execute(
                    "UPDATE influence SET redacted = 1"
                    " WHERE object_kind = ? AND object_id = ?",
                    (k, i),
                )
            else:
                conn.execute(
                    "UPDATE influence SET redacted = 1"
                    " WHERE object_kind = ? AND object_id = ? AND revision = ?",
                    (k, i, r),
                )


# ----------------------------------------------------------------------
# verification — the honest closure check (V3-36.04)
# ----------------------------------------------------------------------


def verify_closure(
    conn: sqlite3.Connection, plan: DeletionPlan
) -> dict[str, Any]:
    """Check the plan actually closed: no orphans may reference purged objects.

    ``closed`` is False when (a) any purged object's bytes/row remain
    reachable per its kind's erasure semantics, (b) any derivation edge or
    member row still names a purged object, or (c) a planned
    suppression/revalidation marker is missing. Objects already disclosed
    as ``outside_boundary`` are not double-counted — they are reported as
    unreachable by construction.
    """
    orphans: list[dict[str, Any]] = []
    purged = plan.purged_refs()
    outside = {tuple(o["ref"]) for o in plan.outside_boundary if "ref" in o}
    sid = plan.scope_id

    def in_purged(k: str, i: str, r: Optional[int]) -> bool:
        return (k, i, r) in purged or (k, i, None) in purged

    # 1. Every purged object must be gone per its kind's semantics.
    for ref in sorted(purged):
        if ref in outside:
            continue
        if not _is_gone(conn, ref, sid):
            orphans.append({"ref": ref, "reason": "row_remains"})

    # 2. No derivation edge may still reference a purged object on either
    #    side — bounded per-ref lookups, not a table scan.
    for pk, pi, pr in sorted(purged):
        if pr is None:
            rows = _rows(
                conn.execute(
                    "SELECT child_kind, child_id, child_revision,"
                    " parent_kind, parent_id, parent_revision FROM derivations"
                    " WHERE (parent_kind = ? AND parent_id = ?)"
                    "    OR (child_kind = ? AND child_id = ?)",
                    (pk, pi, pk, pi),
                )
            )
        else:
            rows = _rows(
                conn.execute(
                    "SELECT child_kind, child_id, child_revision,"
                    " parent_kind, parent_id, parent_revision FROM derivations"
                    " WHERE (parent_kind = ? AND parent_id = ?"
                    "        AND parent_revision = ?)"
                    "    OR (child_kind = ? AND child_id = ?"
                    "        AND child_revision = ?)",
                    (pk, pi, pr, pk, pi, pr),
                )
            )
        for row in rows:
            child = (row["child_kind"], row["child_id"], row["child_revision"])
            parent = (row["parent_kind"], row["parent_id"], row["parent_revision"])
            if in_purged(*parent):
                orphans.append(
                    {"ref": child, "reason": "edge_to_purged_parent",
                     "parent": parent}
                )
            if in_purged(*child):
                orphans.append(
                    {"ref": parent, "reason": "edge_from_purged_child",
                     "child": child}
                )

    # 3. Secondary references on surviving rows must be stripped.
    orphans.extend(_reference_orphans(conn, purged))

    # 4. Planned markers must exist.
    for d in plan.actions(ACTION_SUPPRESS):
        if d.ref in outside:
            continue
        if not _is_suppressed(conn, d.ref, sid):
            orphans.append({"ref": d.ref, "reason": "suppression_missing"})
    for d in plan.actions(ACTION_REVALIDATE):
        if d.ref in outside:
            continue
        if not _is_marked_revalidate(conn, d.ref, sid):
            orphans.append({"ref": d.ref, "reason": "revalidation_unmarked"})

    return {"closed": not orphans, "orphans": orphans}


def _is_gone(
    conn: sqlite3.Connection,
    ref: tuple[str, str, Optional[int]],
    scope_id: str,
) -> bool:
    """Kind-aware 'is it erased' check used by verify."""
    kind, oid, rev = ref
    if kind == "claim":
        # Erasure = an erased head revision through the lifecycle machine
        # plus every revision's content scrubbed (bitemporal audit keeps
        # the earlier states — V2-41.10 semantics).
        head = read_claim_head(conn, oid)
        if head is None:
            return True
        if head.state != Lifecycle.ERASED:
            return False
        return conn.execute(
            "SELECT 1 FROM claim_revisions WHERE claim_id = ?"
            " AND (object_json IS NOT NULL OR interpretation_json IS NOT NULL"
            "      OR condition_json IS NOT NULL)",
            (oid,),
        ).fetchone() is None
    if kind == "span":
        r = conn.execute(
            "SELECT sr.payload FROM spans sp"
            " JOIN source_revisions sr"
            "   ON sr.source_id = sp.source_id AND sr.revision = sp.revision"
            " WHERE sp.span_id = ?",
            (oid,),
        ).fetchone()
        return r is None or len(r[0]) == 0
    if kind == "source":
        return conn.execute(
            "SELECT 1 FROM source_revisions WHERE source_id = ?"
            " AND length(payload) > 0",
            (oid,),
        ).fetchone() is None
    if kind == "source_revision":
        if rev is None:
            return conn.execute(
                "SELECT 1 FROM source_revisions WHERE source_id = ?"
                " AND length(payload) > 0",
                (oid,),
            ).fetchone() is None
        r = conn.execute(
            "SELECT length(payload) FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (oid, rev),
        ).fetchone()
        return r is None or r[0] == 0
    if kind == "artifact":
        r = conn.execute(
            "SELECT availability FROM artifacts WHERE artifact_id = ?", (oid,)
        ).fetchone()
        return r is None or r[0] == "purged"
    if kind == "vault_entry":
        if not _has_table(conn, "vault_entries"):
            return True  # no vault schema — nothing to erase
        r = conn.execute(
            "SELECT erased_event FROM vault_entries WHERE entry_id = ?",
            (oid,),
        ).fetchone()
        return r is None or r[0] is not None
    if kind in _OBJECTS_DISPOSITION_KINDS and _has_table(conn, "objects"):
        r = conn.execute(
            "SELECT disposition FROM objects"
            " WHERE kind = ? AND object_id = ?",
            (kind, oid),
        ).fetchone()
        return r is None or r[0] == "erased"
    return not _object_exists(conn, ref, scope_id)


def _is_suppressed(
    conn: sqlite3.Connection,
    ref: tuple[str, str, Optional[int]],
    scope_id: str,
) -> bool:
    kind, oid, rev = ref
    if kind == "claim":
        head = read_claim_head(conn, oid)
        if head is not None:
            return head.recorded_until is not None
        # row absent → the registry marker decides
    else:
        dated = _DATED_TABLES.get(kind)
        if dated is not None and _has_table(conn, dated[0]):
            r = conn.execute(
                f"SELECT recorded_until FROM {dated[0]} WHERE {dated[1]} = ?",
                (oid,),
            ).fetchone()
            if r is not None:
                return r[0] is not None
        elif kind == "artifact" and _has_table(conn, "artifacts"):
            r = conn.execute(
                "SELECT availability FROM artifacts WHERE artifact_id = ?",
                (oid,),
            ).fetchone()
            if r is not None:
                return r[0] == "suppressed"
        elif kind in _OBJECTS_DISPOSITION_KINDS and _has_table(
            conn, "objects"
        ):
            r = conn.execute(
                "SELECT disposition FROM objects"
                " WHERE kind = ? AND object_id = ?",
                (kind, oid),
            ).fetchone()
            if r is not None:
                return r[0] == "held"
    if _has_table(conn, "quarantine"):
        r = conn.execute(
            "SELECT state FROM quarantine"
            " WHERE object_kind = ? AND object_id = ? AND revision = ?",
            (kind, oid, rev if rev is not None else 1),
        ).fetchone()
        return r is not None and r[0] == "suppressed"
    return False


def _is_marked_revalidate(
    conn: sqlite3.Connection,
    ref: tuple[str, str, Optional[int]],
    scope_id: str,
) -> bool:
    kind, oid, rev = ref
    if not _has_table(conn, "freshness"):
        return False
    return conn.execute(
        "SELECT 1 FROM freshness"
        " WHERE scope_id = ? AND object_kind = ? AND object_id = ?"
        " AND revision = ? AND class = 'revalidate_after'",
        (scope_id, kind, oid, rev if rev is not None else 1),
    ).fetchone() is not None


def _reference_orphans(
    conn: sqlite3.Connection,
    purged: set[tuple[str, str, Optional[int]]],
) -> list[dict[str, Any]]:
    """Surviving rows in member/link tables that still name a purged object."""
    orphans: list[dict[str, Any]] = []
    # Member tables name objects at (kind, id) granularity — a surviving row
    # is an orphan regardless of which revision it pinned.
    purged_ids = {(k, i) for k, i, _r in purged}

    for table in ("observation_evidence", "episode_members", "family_members",
                  "capsule_members", "decision_inputs", "artifact_links"):
        if not _has_table(conn, table):
            continue
        for k, i in sorted(purged_ids):
            for r in _rows(
                conn.execute(
                    f"SELECT * FROM {table}"
                    " WHERE object_kind = ? AND object_id = ?",
                    (k, i),
                )
            ):
                rev = r.get("object_revision", r.get("revision"))
                orphans.append(
                    {"ref": (k, i, rev), "reason": "reference_remains",
                     "table": table}
                )
    if _has_table(conn, "claim_evidence"):
        span_ids = {i for k, i, _r in purged if k == "span"}
        for sid_ in span_ids:
            for r in _rows(
                conn.execute(
                    "SELECT claim_id, revision FROM claim_evidence"
                    " WHERE span_id = ?",
                    (sid_,),
                )
            ):
                orphans.append(
                    {"ref": ("span", sid_, None),
                     "reason": "reference_remains", "table": "claim_evidence",
                     "detail": r}
                )
    if _has_table(conn, "procedure_steps"):
        span_ids = {i for k, i, _r in purged if k == "span"}
        for sid_ in span_ids:
            if conn.execute(
                "SELECT 1 FROM procedure_steps WHERE span_id = ? LIMIT 1",
                (sid_,),
            ).fetchone():
                orphans.append(
                    {"ref": ("span", sid_, None),
                     "reason": "reference_remains", "table": "procedure_steps"}
                )
    if _has_table(conn, "transition_anchors"):
        anchor_ids = {i for k, i, _r in purged if k == "state_anchor"}
        for aid in anchor_ids:
            if conn.execute(
                "SELECT 1 FROM transition_anchors WHERE anchor_id = ? LIMIT 1",
                (aid,),
            ).fetchone():
                orphans.append(
                    {"ref": ("state_anchor", aid, None),
                     "reason": "reference_remains", "table": "transition_anchors"}
                )
    if _has_table(conn, "profile_entry_support"):
        # Profile support pointers use (support_kind, support_id) —
        # same (kind, id) granularity as member tables.
        for k, i in sorted(purged_ids):
            for r in _rows(
                conn.execute(
                    "SELECT entry_id, revision FROM profile_entry_support"
                    " WHERE support_kind = ? AND support_id = ?",
                    (k, i),
                )
            ):
                orphans.append(
                    {"ref": ("profile", r["entry_id"], r["revision"]),
                     "reason": "reference_remains",
                     "table": "profile_entry_support"}
                )
    return orphans


# ----------------------------------------------------------------------
# small shared helpers (v2 idioms kept in one place)
# ----------------------------------------------------------------------


def _next_seq(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
        ).fetchone()[0]
    )


def _bump_generation(store: Any, conn: sqlite3.Connection) -> int:
    bump = getattr(store, "bump_generation", None)
    if callable(bump):
        return int(bump(conn))
    return 0


def _bump_erasure_epoch(store: Any, conn: sqlite3.Connection) -> int:
    """Advance the store's erasure epoch; fences later commits (V2-41.08)."""
    get = getattr(store, "_meta_get", None)
    set_ = getattr(store, "_meta_set", None)
    if not (callable(get) and callable(set_)):
        return 0
    current = get(conn, "erasure_epoch") or 0
    nxt = int(current) + 1
    set_(conn, "erasure_epoch", nxt)
    return nxt


__all__ = [
    "ACTION_DELETE",
    "ACTION_REVALIDATE",
    "ACTION_SUPPRESS",
    "DeletionPlan",
    "PlannedObject",
    "execute_closure",
    "plan_closure",
    "verify_closure",
]
