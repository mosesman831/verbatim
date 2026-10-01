"""Profile service — SPEC_V4 §21 profiles, preferences, and
perspective-aware personalization.

Design contract
---------------

* Profiles are *derived* state. Every entry either carries derivation
  parents (immutable ``derivations`` + ``dependency_edges`` rows with
  ``child_kind='profile'``, ``child_id=entry_id``) or is an owner
  declaration (``explicit``/``task_local``/``sensitive`` kinds may stand
  on the owner's word). Derived kinds (``observed``/``inferred``) REQUIRE
  at least one live support ref — a parentless derivation is fabrication.
* Entries never authorize anything. Serving an entry re-verifies the
  caller's ``read`` lease AND every derivation parent through
  ``Kernel.resolve_access``; lifting support bytes is the kernel's
  job (``read_verified``), never this module's.
* Contradictions are retained: distinct live values under a
  ``single``-multiplicity topic all stay inspectable, marked
  ``conflicted`` inside a shared ``conflict_group`` — never silently
  resolved (V4-21.05).
* Audience is closed: an entry with empty ``audience`` is subject-only.
  Non-subject principals need explicit audience membership; ``admin``
  holders may *inspect* but ordinary personalization reads cannot bypass
  audience restrictions (V4-21.03/08).
* Sensitive-trait inference is off by default and needs three opt-ins:
  ``topic.inference='on'``, a purpose inside ``inference_purposes``, and
  a per-subject ``profile_policies`` consent row. Absence of a detector
  match is never consent (V4-21.04).
* Purge/invalidation: reads gate on kernel resolution of every parent
  ref, so purged/quarantined/suppressed evidence withholds the entry.
  ``refresh()`` persists that verdict (``withheld``/``tombstoned``) and
  scrubs support pointers to provably-erased objects so no dangling
  reference into erased material survives.
* Perspective packs narrow/rank a caller's own candidate set — output is
  always a subset of input; a pack whose caller lacks ``read`` evaluates
  to nothing and ``quote``-less packs never mark items liftable.

Known boundaries (honest status): compilation is on-demand — no durable
``profile_compile`` JobKind exists because widening the jobs-table CHECK
constraint belongs to the schema owner; the report says so. Closure
enumeration discovers profile entries through the derivations graph but
classifies the kind ``outside`` the physically-erasable set
(``privacy/deletion.py`` leaves ``profile`` unresolved); deletion
semantics here are *withhold-on-read plus durable tombstone via
refresh/delete*, disclosed in closure ``boundary_json``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterable, Optional

from .. import derivations
from ..core.lifecycle import read_claim_head
from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, new_id, require_id
from ..core.types_v3 import Verb
from ..governance import CallerV3
from ..governance import perspectives as _perspectives
from ..governance import purposes as _purposes
from ..kernel.service import Kernel
from ..security import quarantine as _quarantine
from ..storage import repos_v4
from ..storage.repos import has_table
from . import schema as _schema
from .types import (
    MAX_COMPILE_CLAIMS,
    MAX_ENTRIES_PER_READ,
    MAX_SUPPORT_REFS,
    PROFILE_KIND,
    PRODUCER_COMPILE,
    PRODUCER_REFRESH,
    PRODUCER_UPSERT,
    SERVABLE_STATES,
    ConflictPolicy,
    EntryKind,
    EntryState,
    InferenceMode,
    Multiplicity,
    PerspectivePack,
    Sensitivity,
    SUPPORT_RELATIONS,
)

_PACK_TTL_US = 60_000_000  # perspective packs are one-minute declarations

_ENTRY_VISIBLE_STATES = SERVABLE_STATES
_CLAIM_VISIBLE = ("active", "disputed")

# ---------------------------------------------------------------------------
# X5 — topic-directed extraction safety guard (SPEC_V4_5 §09 X5,
# V45-09.05; parent binding V4-16.06; acceptance D17)
# ---------------------------------------------------------------------------
#
# A topic's ``match`` spec is a *positive* steering filter: it decides
# which claims inform a topic's derived entries. It may NEVER drop a
# safety-bearing event — deletion, consent withdrawal, correction/
# invalidation, or security — because a narrow allowlist would then make
# the event invisible to every retention surface that consults the
# policy. The guard therefore works at both boundaries:
#
# * authoring: ``put_topic`` rejects ``match`` keys outside the known
#   positive filters — an exclusion-shaped spec (``exclude_predicates``,
#   ``not_keywords``, ``drop``…) would otherwise persist as a silent
#   no-op while the author believes a category was suppressed;
# * compilation: :meth:`ProfileService._safety_category` classifies a
#   claim head, safety-bearing heads bypass :meth:`_matches`, and the
#   compile report names them under ``safety_events`` — steered, never
#   suppressed, and always visible.

#: Lowercase substrings marking a safety-bearing predicate. A false
#: positive is *safe* (the claim stays visible); a false negative is the
#: suppression this guard exists to prevent.
_SAFETY_EVENT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("deletion", ("delet", "erase", "erasure", "purge", "forget")),
    (
        "consent_withdrawal",
        (
            "consent_withdrawn",
            "consent_revoked",
            "consent_withdrawal",
            "withdraw_consent",
            "revoke_consent",
            "consent",  # bare consent events — withdrawal is the risky case
        ),
    ),
    (
        "correction",
        (
            "correct",
            "invalidat",
            "supersed",
            "retract",
            "rescind",
            "amend",
        ),
    ),
    (
        "security",
        (
            "security",
            "quarantine",
            "injection",
            "poison",
            "attack",
            "exfiltrat",
            "breach",
        ),
    ),
)


def safety_event_category(head: Any) -> Optional[str]:
    """Classify a claim head's safety category, or ``None``.

    Looks at the revision-level predicate first (authoritative) then the
    claims-table predicate; returns the first matching category name —
    ``deletion`` | ``consent_withdrawal`` | ``correction`` | ``security``.
    """
    for pred in (
        getattr(head, "rev_predicate", None),
        getattr(head, "predicate", None),
    ):
        if not pred:
            continue
        text = str(pred).lower()
        for category, markers in _SAFETY_EVENT_MARKERS:
            if any(m in text for m in markers):
                return category
    return None


#: The only legitimate ``match`` spec keys — positive steering filters.
#: Anything else is an exclusion shape the engine does not honor and
#: therefore refuses to persist (V45-09.05).
_TOPIC_MATCH_KEYS = frozenset({"keywords", "predicates", "subjects"})

#: Entry kinds that stand on owner declaration (support refs optional).
_DECLARED_KINDS = frozenset(
    {EntryKind.EXPLICIT.value, EntryKind.TASK_LOCAL.value,
     EntryKind.SENSITIVE.value}
)
#: Kinds a compile may (re)write — explicit/sensitive owner data is never
#: auto-rewritten by derivation.
_DERIVED_KINDS = frozenset(
    {EntryKind.OBSERVED.value, EntryKind.INFERRED.value}
)


def _deny() -> None:
    raise VerbatimError(
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        "not found or not authorized",
    )


def _validation(msg: str) -> None:
    raise VerbatimError(ErrorCode.VALIDATION, msg)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _entry_digest(row: dict[str, Any], supports: list[dict[str, Any]]) -> str:
    material = {
        "entry_id": row["entry_id"],
        "revision": row["revision"],
        "scope_id": row["scope_id"],
        "profile_id": row["profile_id"],
        "subject_id": row["subject_id"],
        "topic_key": row["topic_key"],
        "entry_kind": row["entry_kind"],
        "state": row["state"],
        "value": _schema.loads(row["value_json"]),
        "confidence": row["confidence"],
        "basis": row["basis"],
        "conflict_group": row["conflict_group"],
        "audience": sorted(_schema.loads(row["audience_json"], [])),
        "effective_us": row["effective_us"],
        "expires_us": row["expires_us"],
        "recorded_us": row["recorded_us"],
        "actor_id": row["actor_id"],
        "purpose": row["purpose"],
        "operation_id": row["operation_id"],
        "producer_id": row["producer_id"],
        "supersedes_entry_id": row["supersedes_entry_id"],
        "prev_revision": row["prev_revision"],
        "supports": [
            [s["kind"], s["id"], int(s["revision"]),
             s.get("relation", "supports")]
            for s in supports
        ],
    }
    return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()


def _slot_group(scope_id: str, subject_id: str, topic_key: str) -> str:
    h = hashlib.sha256(
        f"profile-conflict|{scope_id}|{subject_id}|{topic_key}".encode()
    ).hexdigest()
    return f"pcg:{h[:24]}"


class ProfileService:
    """The §21 profile plane. Every public method takes the caller's
    ``CallerV3`` and re-authorizes through the kernel — profile code
    never becomes a second grant evaluator."""

    def __init__(self, store: Any, *, kernel: Optional[Kernel] = None) -> None:
        self._store = store
        self._kernel = kernel or Kernel(store)

    # ------------------------------------------------------------------
    # authorization plumbing (fail closed — denied leases raise)
    # ------------------------------------------------------------------

    def _lease(
        self,
        conn: sqlite3.Connection,
        caller: CallerV3,
        verb: str,
        purpose: Optional[str],
        scope_ids: Iterable[str],
        object_refs: Iterable[Any] = (),
    ):
        if not isinstance(caller, CallerV3):
            raise VerbatimError(
                ErrorCode.VALIDATION, "caller must be a bound CallerV3"
            )
        lease = self._kernel.resolve_access(
            conn, caller, verb, purpose, tuple(scope_ids), tuple(object_refs)
        )
        if lease.denied:
            _deny()
        return lease

    def _allowed(
        self,
        conn: sqlite3.Connection,
        caller: CallerV3,
        verb: str,
        purpose: Optional[str],
        scope_ids: Iterable[str],
        object_refs: Iterable[Any] = (),
    ) -> bool:
        lease = self._kernel.resolve_access(
            conn, caller, verb, purpose, tuple(scope_ids), tuple(object_refs)
        )
        return not lease.denied

    def _is_admin(
        self, conn: sqlite3.Connection, caller: CallerV3, scope_id: str
    ) -> bool:
        return self._allowed(
            conn, caller, Verb.ADMIN.value, "admin", (scope_id,)
        )

    def _prepare(self, conn: sqlite3.Connection) -> None:
        """Register schema + builtin purposes before any write —
        ``resolve_access`` refuses unregistered purposes, so the
        registry must exist before the first lease is minted here."""
        _schema.ensure_schema(conn)
        if has_table(conn, "purposes"):
            _purposes.seed_purposes(conn)

    def _profile_id(self, conn: sqlite3.Connection, scope_id: str) -> str:
        row = conn.execute(
            "SELECT profile_id FROM scopes WHERE scope_id=?", (scope_id,)
        ).fetchone()
        if row is None:
            _deny()
        return row[0]

    def _log(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        event: str,
        actor_id: str,
        *,
        entry_id: Optional[str] = None,
        purpose: Optional[str] = None,
        operation_id: Optional[str] = None,
        detail: Optional[dict] = None,
        us: Optional[int] = None,
    ) -> None:
        _schema.insert_event(
            conn,
            {
                "seq": _schema.next_event_seq(conn, scope_id),
                "scope_id": scope_id,
                "event": event,
                "entry_id": entry_id,
                "actor_id": actor_id,
                "purpose": purpose,
                "operation_id": operation_id,
                "us": int(us if us is not None else now_us()),
                "detail_json": _schema.dumps(detail or {}),
            },
        )

    # ------------------------------------------------------------------
    # topic configuration (V4-21.01)
    # ------------------------------------------------------------------

    def put_topic(
        self,
        caller: CallerV3,
        scope_id: str,
        topic: dict[str, Any],
        *,
        purpose: str = "admin",
    ) -> dict:
        """Create or replace a topic configuration. ``admin``-gated —
        topic policy is owner configuration, not a read-path decision."""
        require_id(scope_id, "scope_id")
        key = require_id(str(topic.get("topic_key") or ""), "topic_key")
        mult = str(topic.get("multiplicity") or Multiplicity.SINGLE.value)
        sens = str(topic.get("sensitivity") or Sensitivity.NORMAL.value)
        conflict = str(
            topic.get("conflict_policy") or ConflictPolicy.KEEP_BOTH.value
        )
        inference = str(topic.get("inference") or InferenceMode.OFF.value)
        state = str(topic.get("state") or "active")
        for value, allowed, name in (
            (mult, {m.value for m in Multiplicity}, "multiplicity"),
            (sens, {s.value for s in Sensitivity}, "sensitivity"),
            (conflict, {c.value for c in ConflictPolicy}, "conflict_policy"),
            (inference, {i.value for i in InferenceMode}, "inference"),
            (state, {"active", "disabled", "deleted"}, "state"),
        ):
            if value not in allowed:
                _validation(f"{name} must be one of {sorted(allowed)}")
        required = tuple(
            require_id(str(p), "required_purposes")
            for p in (topic.get("required_purposes") or ())
        )
        inference_purposes = tuple(
            require_id(str(p), "inference_purposes")
            for p in (topic.get("inference_purposes") or ())
        )
        preferred = tuple(
            str(p) for p in (topic.get("preferred_evidence") or ())
        )
        match = topic.get("match") or {}
        if not isinstance(match, dict):
            _validation("match must be an object")
        unknown = sorted(set(match) - _TOPIC_MATCH_KEYS)
        if unknown:
            # V45-09.05: match specs are positive steering filters only.
            # An exclusion-shaped key would persist as a silent no-op —
            # the author believes a category is suppressed when nothing
            # is. Refuse it at authoring time.
            _validation(
                f"match keys {unknown} are not supported — only "
                f"{sorted(_TOPIC_MATCH_KEYS)} (positive filters; "
                "exclusion-shaped topic policy is not a valid way to "
                "suppress safety events, V4-16.06)"
            )
        for mk in ("keywords", "predicates", "subjects"):
            if mk in match and not isinstance(match[mk], (list, tuple)):
                _validation(f"match.{mk} must be a list")
        expiry_s = topic.get("expiry_s")
        if expiry_s is not None:
            expiry_s = int(expiry_s)
            if expiry_s <= 0:
                _validation("expiry_s must be positive")
        max_entries = int(topic.get("max_entries") or 8)
        min_support = int(topic.get("min_support") or 1)
        if max_entries < 1 or min_support < 1:
            _validation("max_entries/min_support must be >= 1")
        if inference == InferenceMode.ON.value and not inference_purposes:
            _validation(
                "inference='on' requires declared inference_purposes"
            )
        now = now_us()
        with self._store.tx() as conn:
            self._prepare(conn)
            self._lease(conn, caller, Verb.ADMIN.value, purpose, (scope_id,))
            existing = _schema.get_topic_row(conn, scope_id, key)
            _schema.put_topic_row(
                conn,
                {
                    "scope_id": scope_id,
                    "topic_key": key,
                    "value_kind": str(topic.get("value_kind") or "freeform"),
                    "multiplicity": mult,
                    "sensitivity": sens,
                    "preferred_evidence_json": _schema.dumps(
                        sorted(preferred)
                    ),
                    "required_purposes_json": _schema.dumps(
                        sorted(required)
                    ),
                    "inference": inference,
                    "inference_purposes_json": _schema.dumps(
                        sorted(inference_purposes)
                    ),
                    "conflict_policy": conflict,
                    "expiry_s": expiry_s,
                    "max_entries": max_entries,
                    "min_support": min_support,
                    "match_json": _schema.dumps(match),
                    "state": state,
                    "created_us": existing["created_us"]
                    if existing
                    else now,
                    "updated_us": now,
                    "updated_by": caller.principal_id,
                },
            )
            self._log(
                conn, scope_id, "topic_put", caller.principal_id,
                purpose=purpose,
                detail={"topic_key": key, "created": existing is None},
                us=now,
            )
        return self.get_topic(caller, scope_id, key, purpose=purpose)

    def get_topic(
        self,
        caller: CallerV3,
        scope_id: str,
        topic_key: str,
        *,
        purpose: str = "admin",
    ) -> Optional[dict]:
        require_id(scope_id, "scope_id")
        require_id(topic_key, "topic_key")
        with self._store.read() as conn:
            if not _schema.tables_present(conn):
                return None
            self._lease(conn, caller, Verb.READ.value, purpose, (scope_id,))
            row = _schema.get_topic_row(conn, scope_id, topic_key)
            return self._topic_view(row) if row else None

    def list_topics(
        self, caller: CallerV3, scope_id: str, *, purpose: str = "admin"
    ) -> list[dict]:
        require_id(scope_id, "scope_id")
        with self._store.read() as conn:
            if not _schema.tables_present(conn):
                return []
            self._lease(conn, caller, Verb.READ.value, purpose, (scope_id,))
            return [
                self._topic_view(r) for r in _schema.topic_rows(conn, scope_id)
            ]

    def _topic_view(self, row: dict[str, Any]) -> dict:
        return {
            "scope_id": row["scope_id"],
            "topic_key": row["topic_key"],
            "value_kind": row["value_kind"],
            "multiplicity": row["multiplicity"],
            "sensitivity": row["sensitivity"],
            "preferred_evidence": _schema.loads(
                row["preferred_evidence_json"], []
            ),
            "required_purposes": _schema.loads(
                row["required_purposes_json"], []
            ),
            "inference": row["inference"],
            "inference_purposes": _schema.loads(
                row["inference_purposes_json"], []
            ),
            "conflict_policy": row["conflict_policy"],
            "expiry_s": row["expiry_s"],
            "max_entries": row["max_entries"],
            "min_support": row["min_support"],
            "match": _schema.loads(row["match_json"], {}),
            "state": row["state"],
            "created_us": row["created_us"],
            "updated_us": row["updated_us"],
            "updated_by": row["updated_by"],
        }

    def set_inference(
        self,
        caller: CallerV3,
        scope_id: str,
        subject_id: str,
        enabled: bool,
        *,
        purpose: str = "admin",
    ) -> dict:
        """Owner consent switch for per-subject inference (V4-21.04/07).
        ``off`` rows are still consent decisions — the default absent row
        is *also* off; the row exists for auditability."""
        require_id(scope_id, "scope_id")
        require_id(subject_id, "subject_id")
        now = now_us()
        with self._store.tx() as conn:
            self._prepare(conn)
            self._lease(conn, caller, Verb.ADMIN.value, purpose, (scope_id,))
            _schema.put_policy_row(
                conn,
                {
                    "scope_id": scope_id,
                    "subject_id": subject_id,
                    "inference": InferenceMode.ON.value
                    if enabled
                    else InferenceMode.OFF.value,
                    "updated_us": now,
                    "updated_by": caller.principal_id,
                },
            )
            self._log(
                conn, scope_id, "policy_set", caller.principal_id,
                purpose=purpose,
                detail={"subject_id": subject_id, "inference": enabled},
                us=now,
            )
        return {"subject_id": subject_id, "inference": "on" if enabled else "off"}

    def _inference_ok(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        subject_id: str,
        topic: dict[str, Any],
        purpose: Optional[str],
    ) -> bool:
        """V4-21.04 triple gate: topic inference on + purpose in the
        declared set + per-subject consent row. All three or nothing."""
        if topic["inference"] != InferenceMode.ON.value:
            return False
        allowed = set(_schema.loads(topic["inference_purposes_json"], []))
        if purpose is None or purpose not in allowed:
            return False
        pol = _schema.get_policy_row(conn, scope_id, subject_id)
        return bool(pol and pol["inference"] == InferenceMode.ON.value)

    # ------------------------------------------------------------------
    # entry writes (V4-21.02/05)
    # ------------------------------------------------------------------

    def upsert_entry(
        self,
        caller: CallerV3,
        scope_id: str,
        entry: dict[str, Any],
        *,
        purpose: str = "derive",
    ) -> dict:
        """Declare or revise a profile entry. Returns an update report
        (changed fields, previous revision, effective time, supports)."""
        require_id(scope_id, "scope_id")
        subject_id = require_id(
            str(entry.get("subject_id") or ""), "subject_id"
        )
        topic_key = require_id(
            str(entry.get("topic_key") or ""), "topic_key"
        )
        kind = str(entry.get("entry_kind") or EntryKind.EXPLICIT.value)
        if kind not in {k.value for k in EntryKind}:
            _validation(f"entry_kind must be one of {[k.value for k in EntryKind]}")
        value = entry.get("value")
        if value is None:
            _validation("value is required")
        supports = self._normalize_supports(entry.get("supports") or ())
        now = now_us()
        with self._store.tx() as conn:
            self._prepare(conn)
            lease = self._lease(
                conn, caller, Verb.DERIVE.value, purpose, (scope_id,)
            )
            topic = _schema.get_topic_row(conn, scope_id, topic_key)
            if topic is None or topic["state"] != "active":
                _validation(f"topic {topic_key!r} is not configured/active")
            self._check_write_rules(
                conn, scope_id, topic, kind, purpose, subject_id,
                entry, supports,
            )
            self._check_supports_live(conn, scope_id, supports)
            report = self._write_entry(
                conn,
                caller=caller,
                scope_id=scope_id,
                topic=topic,
                subject_id=subject_id,
                kind=kind,
                value=value,
                supports=supports,
                confidence=entry.get("confidence"),
                basis=entry.get("basis"),
                audience=entry.get("audience"),
                effective_us=entry.get("effective_us"),
                expires_us=entry.get("expires_us"),
                entry_id=entry.get("entry_id"),
                profile_id=entry.get("profile_id"),
                purpose=purpose,
                operation_id=entry.get("operation_id")
                or lease.operation_id,
                producer_id=PRODUCER_UPSERT,
                now=now,
            )
            return report

    def _check_write_rules(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        topic: dict[str, Any],
        kind: str,
        purpose: Optional[str],
        subject_id: str,
        entry: dict[str, Any],
        supports: list[dict[str, Any]],
    ) -> None:
        if topic["conflict_policy"] == ConflictPolicy.EXPLICIT_ONLY.value:
            if kind != EntryKind.EXPLICIT.value:
                _deny()
        if kind in _DERIVED_KINDS and not supports:
            _validation(
                f"{kind} entries require at least one support ref — "
                "a parentless derivation is fabrication"
            )
        if kind == EntryKind.TASK_LOCAL.value:
            if not entry.get("expires_us") and not topic["expiry_s"]:
                _validation(
                    "task_local entries must expire (expires_us or "
                    "topic expiry_s)"
                )
        if kind == EntryKind.SENSITIVE.value:
            if topic["sensitivity"] != Sensitivity.SENSITIVE.value:
                _deny()
            required = set(
                _schema.loads(topic["required_purposes_json"], [])
            )
            if not required or purpose not in required:
                _deny()
        if kind in _DERIVED_KINDS and (
            topic["sensitivity"] == Sensitivity.SENSITIVE.value
            or kind == EntryKind.INFERRED.value
        ):
            # inferred entries on normal topics still need purpose
            # declared; sensitive topics need the full V4-21.04 gate.
            if topic["sensitivity"] == Sensitivity.SENSITIVE.value:
                if not self._inference_ok(
                    conn, scope_id, subject_id, topic, purpose
                ):
                    _deny()
            elif kind == EntryKind.INFERRED.value:
                allowed = set(
                    _schema.loads(topic["inference_purposes_json"], [])
                )
                if (
                    topic["inference"] != InferenceMode.ON.value
                    or purpose not in allowed
                ):
                    _deny()

    def _normalize_supports(
        self, supports: Iterable[Any]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for s in supports:
            if isinstance(s, dict):
                raw = dict(s)
                rel = str(raw.pop("relation", "supports"))
                role = raw.pop("claim_role", None)
                ref = derivations.normalize_ref(raw)
            else:
                rel = "supports"
                role = None
                ref = derivations.normalize_ref(s)
            kind, oid, rev = ref
            if rev is None:
                _validation("support refs require an explicit revision")
            if rel not in SUPPORT_RELATIONS:
                _validation(f"support relation must be {sorted(SUPPORT_RELATIONS)}")
            out.append(
                {
                    "kind": kind,
                    "id": oid,
                    "revision": int(rev),
                    "relation": rel,
                    "claim_role": role,
                }
            )
        if len(out) > MAX_SUPPORT_REFS:
            _validation(f"support refs capped at {MAX_SUPPORT_REFS}")
        # Deterministic order — digest stability.
        out.sort(key=lambda s: (s["relation"], s["kind"], s["id"], s["revision"]))
        return out

    def _check_supports_live(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        supports: list[dict[str, Any]],
    ) -> None:
        """Write-time gate: every named parent must be same-scope and
        currently viable — a derivation may not cite dead or foreign
        evidence."""
        for s in supports:
            status, ref_scope = self._support_status(
                conn, s["kind"], s["id"], s["revision"]
            )
            if ref_scope is not None and ref_scope != scope_id:
                _validation(
                    f"support {s['kind']}:{s['id']} lives in another "
                    "scope — cross-scope derivation is not permitted"
                )
            if status == "gone":
                _validation(
                    f"support {s['kind']}:{s['id']}@{s['revision']} does "
                    "not resolve to live evidence"
                )
            if status == "held":
                _deny()

    def _write_entry(
        self,
        conn: sqlite3.Connection,
        *,
        caller: CallerV3,
        scope_id: str,
        topic: dict[str, Any],
        subject_id: str,
        kind: str,
        value: Any,
        supports: list[dict[str, Any]],
        confidence: Optional[float],
        basis: Optional[str],
        audience: Optional[Iterable[str]],
        effective_us: Optional[int],
        expires_us: Optional[int],
        entry_id: Optional[str],
        profile_id: Optional[str],
        purpose: Optional[str],
        operation_id: str,
        producer_id: str,
        now: int,
        force_state: Optional[str] = None,
        reconcile: bool = True,
    ) -> dict:
        """Append one revision (new entry or revision of an existing id),
        register it in the v4 object registry, and record derivation
        parents. Returns the V4-21.05 update report."""
        profile_id = profile_id or self._profile_id(conn, scope_id)
        entry_id = entry_id or f"pe:{new_id()}"
        require_id(entry_id, "entry_id")
        head = _schema.entry_head_row(conn, entry_id)
        if head is not None and head["scope_id"] != scope_id:
            _deny()  # foreign-scope id — indistinguishable denial
        revision = (int(head["revision"]) + 1) if head else 1
        prev_revision = int(head["revision"]) if head else None
        if head and head["state"] == EntryState.TOMBSTONED.value:
            _validation("tombstoned entries cannot be revised")
        eff = int(effective_us) if effective_us is not None else now
        exp = expires_us
        if exp is None and kind == EntryKind.TASK_LOCAL.value:
            exp = eff + int(topic["expiry_s"])
        if exp is not None:
            exp = int(exp)
            if exp <= eff:
                _validation("expires_us must be after effective_us")
        conf = (
            float(confidence)
            if confidence is not None
            else (1.0 if kind == EntryKind.EXPLICIT.value else 0.5)
        )
        if not (0.0 <= conf <= 1.0):
            _validation("confidence must be within [0, 1]")
        aud = sorted(
            {require_id(str(a), "audience") for a in (audience or ())}
        )
        value_json = _canonical(value)
        state = force_state or EntryState.ACTIVE.value
        changed = self._changed_fields(head, value_json, conf, aud, state,
                                       exp)
        row = {
            "entry_id": entry_id,
            "scope_id": scope_id,
            "profile_id": profile_id,
            "subject_id": subject_id,
            "topic_key": topic["topic_key"],
            "entry_kind": kind,
            "revision": revision,
            "state": state,
            "value_json": value_json,
            "confidence": conf,
            "basis": basis,
            "conflict_group": head["conflict_group"] if head else None,
            "audience_json": _schema.dumps(aud),
            "effective_us": eff,
            "expires_us": exp,
            "recorded_us": now,
            "actor_id": caller.principal_id,
            "purpose": purpose,
            "operation_id": operation_id,
            "producer_id": producer_id,
            "supersedes_entry_id": None,
            "prev_revision": prev_revision,
            "changed_fields_json": _schema.dumps(changed),
            "digest": "",
        }
        row["digest"] = _entry_digest(row, supports)
        _schema.insert_entry_row(conn, row)
        _schema.insert_support_rows(conn, entry_id, revision, supports)
        self._register_object(conn, scope_id, entry_id, revision, state,
                              row["digest"], now, producer_id, kind)
        self._record_parents(
            conn, scope_id, entry_id, revision, supports,
            producer_id, operation_id,
        )
        self._log(
            conn, scope_id, "entry_write", caller.principal_id,
            entry_id=entry_id, purpose=purpose, operation_id=operation_id,
            detail={
                "revision": revision,
                "prev_revision": prev_revision,
                "changed_fields": changed,
                "kind": kind,
                "state": state,
            },
            us=now,
        )
        if reconcile:
            self._reconcile_slot(
                conn, scope_id=scope_id, profile_id=profile_id,
                subject_id=subject_id, topic=topic, trigger_id=entry_id,
                caller=caller, purpose=purpose, operation_id=operation_id,
                now=now, producer_id=producer_id,
            )
        return {
            "entry_id": entry_id,
            "revision": revision,
            "prev_revision": prev_revision,
            "changed_fields": changed,
            "effective_us": eff,
            "state": state,
            "supports": [
                {"kind": s["kind"], "id": s["id"],
                 "revision": s["revision"], "relation": s["relation"]}
                for s in supports
            ],
            "digest": row["digest"],
            "operation_id": operation_id,
        }

    @staticmethod
    def _changed_fields(
        head: Optional[dict[str, Any]],
        value_json: str,
        confidence: float,
        audience: list[str],
        state: str,
        expires_us: Optional[int],
    ) -> list[str]:
        if head is None:
            return [
                "state", "value", "confidence", "audience", "expires_us",
            ]
        changed: list[str] = []
        if head["value_json"] != value_json:
            changed.append("value")
        if float(head["confidence"]) != float(confidence):
            changed.append("confidence")
        if _schema.loads(head["audience_json"], []) != audience:
            changed.append("audience")
        if head["state"] != state:
            changed.append("state")
        if head["expires_us"] != expires_us:
            changed.append("expires_us")
        return changed

    def _register_object(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        entry_id: str,
        revision: int,
        state: str,
        digest: str,
        now: int,
        producer_id: str,
        entry_kind: str,
    ) -> None:
        """Mirror the entry into the v4 object registry so kernel
        resolution + disposition gating apply to profile objects."""
        disposition = {
            EntryState.ACTIVE.value: "active",
            EntryState.CONFLICTED.value: "active",
            EntryState.SUPERSEDED.value: "superseded",
            EntryState.WITHHELD.value: "held",
            EntryState.TOMBSTONED.value: "erased",
            EntryState.EXPIRED.value: "archived",
        }[state]
        existing = repos_v4.get(
            conn, "objects", {"kind": PROFILE_KIND, "object_id": entry_id}
        )
        if existing is None:
            repos_v4.insert(
                conn,
                "objects",
                {
                    "object_id": entry_id,
                    "kind": PROFILE_KIND,
                    "scope_id": scope_id,
                    "current_revision": revision,
                    "disposition": disposition,
                    "created_event": int(now),
                },
            )
        else:
            repos_v4.update(
                conn,
                "objects",
                {
                    "current_revision": revision,
                    "disposition": disposition,
                },
                {"kind": PROFILE_KIND, "object_id": entry_id},
            )
        repos_v4.insert(
            conn,
            "object_revisions",
            {
                "kind": PROFILE_KIND,
                "object_id": entry_id,
                "revision": revision,
                "digest": digest,
                "recorded_from": int(now),
                "recorded_until": None,
                "producer_ref": producer_id,
                "metadata_json": _schema.dumps(
                    {"entry_kind": entry_kind}
                ),
            },
        )
        # Keep the producer manifest honest: registered before its first
        # artifacts (V4-16), idempotent on later writes.
        if repos_v4.get(
            conn, "producer_manifests", {"producer_id": producer_id}
        ) is None:
            repos_v4.insert(
                conn,
                "producer_manifests",
                {
                    "producer_id": producer_id,
                    "kind": "profile",
                    "artifact_digest": None,
                    "rubric_digest": None,
                    "config_digest": None,
                    "schema_version": 1,
                    "license_ref": None,
                    "health": "unverified",
                    "registered_us": int(now),
                },
            )

    def _record_parents(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        entry_id: str,
        revision: int,
        supports: list[dict[str, Any]],
        producer_id: str,
        operation_id: str,
    ) -> None:
        """Immutable provenance: derivations edges (closure-visible) +
        dependency_edges (kernel ancestry for quarantine cascade)."""
        for seq, s in enumerate(supports):
            derivations.record_edge(
                conn,
                (PROFILE_KIND, entry_id, revision),
                (s["kind"], s["id"], s["revision"]),
                "service",
                producer_id,
                scope_id,
                seq,
            )
            repos_v4.insert(
                conn,
                "dependency_edges",
                {
                    "child_kind": PROFILE_KIND,
                    "child_id": entry_id,
                    "child_revision": revision,
                    "parent_kind": s["kind"],
                    "parent_id": s["id"],
                    "parent_revision": s["revision"],
                    "role": s["relation"],
                    "producer_id": producer_id,
                    "operation_id": operation_id,
                    "seq": seq,
                },
            )

    # ------------------------------------------------------------------
    # conflict reconciliation (V4-21.05)
    # ------------------------------------------------------------------

    def _reconcile_slot(
        self,
        conn: sqlite3.Connection,
        *,
        scope_id: str,
        profile_id: str,
        subject_id: str,
        topic: dict[str, Any],
        trigger_id: str,
        caller: CallerV3,
        purpose: Optional[str],
        operation_id: str,
        now: int,
        producer_id: str = PRODUCER_UPSERT,
    ) -> None:
        """After a write, mark/supersede co-slot heads per the topic's
        multiplicity + conflict policy. Every mark is a new revision —
        nothing is silently overwritten."""
        heads = _schema.head_rows(
            conn,
            scope_id,
            profile_id=profile_id,
            subject_id=subject_id,
            topic_key=topic["topic_key"],
            states=_ENTRY_VISIBLE_STATES,
            limit=int(topic["max_entries"]) * 4 + 16,
        )
        if topic["multiplicity"] != Multiplicity.SINGLE.value:
            self._cap_set_slot(
                conn, heads, topic, caller, purpose, operation_id, now,
                producer_id,
            )
            return
        group = _slot_group(scope_id, subject_id, topic["topic_key"])
        distinct = {h["value_json"] for h in heads}
        if len(distinct) <= 1:
            # Alternatives collapsed (superseded/withheld/deleted): an
            # open group resolves against the sole survivor, and the
            # survivor drops back to ``active``.
            crow = _schema.get_conflict_row(conn, scope_id, group)
            if crow is not None and crow["state"] == "open":
                winner = heads[0]["entry_id"] if heads else None
                for h in heads:
                    if h["state"] == EntryState.CONFLICTED.value:
                        self._revise_state(
                            conn, h, EntryState.ACTIVE.value, caller,
                            purpose, operation_id, now,
                            conflict_group=None,
                            extra={"reason": "conflict_resolved"},
                            producer_id=producer_id,
                        )
                if winner:
                    self._close_conflict(conn, scope_id, group, winner, now)
            return
        policy = topic["conflict_policy"]
        if policy == ConflictPolicy.EXPLICIT_ONLY.value:
            return  # non-explicit writes were rejected already
        if policy == ConflictPolicy.LATEST_WINS.value:
            winner = max(heads, key=lambda h: (h["effective_us"], h["entry_id"]))
            for h in heads:
                if h["entry_id"] == winner["entry_id"]:
                    continue
                self._revise_state(
                    conn, h, EntryState.SUPERSEDED.value, caller,
                    purpose, operation_id, now,
                    extra={"superseded_by": winner["entry_id"]},
                    producer_id=producer_id,
                )
            self._close_conflict(conn, scope_id, group, winner["entry_id"], now)
            return
        # keep_both: every distinct live value stays, all marked.
        members = sorted(h["entry_id"] for h in heads)
        for h in heads:
            if h["state"] == EntryState.CONFLICTED.value and (
                h["conflict_group"] == group
            ):
                continue
            self._revise_state(
                conn, h, EntryState.CONFLICTED.value, caller, purpose,
                operation_id, now, conflict_group=group,
                producer_id=producer_id,
            )
        _schema.put_conflict_row(
            conn,
            {
                "conflict_group": group,
                "scope_id": scope_id,
                "profile_id": profile_id,
                "topic_key": topic["topic_key"],
                "subject_id": subject_id,
                "members_json": _schema.dumps(members),
                "state": "open",
                "opened_us": (
                    _schema.get_conflict_row(conn, scope_id, group)
                    or {"opened_us": now}
                )["opened_us"],
                "resolved_us": None,
                "resolution_entry_id": None,
            },
        )
        self._log(
            conn, scope_id, "conflict_open", caller.principal_id,
            purpose=purpose, operation_id=operation_id,
            detail={"conflict_group": group, "members": members},
            us=now,
        )

    def _cap_set_slot(
        self,
        conn: sqlite3.Connection,
        heads: list[dict[str, Any]],
        topic: dict[str, Any],
        caller: CallerV3,
        purpose: Optional[str],
        operation_id: str,
        now: int,
        producer_id: str = PRODUCER_UPSERT,
    ) -> None:
        cap = int(topic["max_entries"])
        if len(heads) <= cap:
            return
        heads_sorted = sorted(
            heads, key=lambda h: (h["confidence"], h["effective_us"],
                                  h["entry_id"])
        )
        for h in heads_sorted[: len(heads_sorted) - cap]:
            self._revise_state(
                conn, h, EntryState.SUPERSEDED.value, caller, purpose,
                operation_id, now, extra={"reason": "slot_cap"},
                producer_id=producer_id,
            )

    def _revise_state(
        self,
        conn: sqlite3.Connection,
        head: dict[str, Any],
        new_state: str,
        caller: CallerV3,
        purpose: Optional[str],
        operation_id: str,
        now: int,
        *,
        conflict_group: Any = "__keep__",
        extra: Optional[dict] = None,
        producer_id: str = PRODUCER_UPSERT,
    ) -> dict:
        """Append a state-only revision to an existing entry."""
        revision = int(head["revision"]) + 1
        group = (
            head["conflict_group"]
            if conflict_group == "__keep__"
            else conflict_group
        )
        changed = ["state"]
        if group != head["conflict_group"]:
            changed.append("conflict_group")
        supports = [
            {
                "kind": s["support_kind"],
                "id": s["support_id"],
                "revision": s["support_revision"],
                "relation": s["relation"],
                "claim_role": s["claim_role"],
            }
            for s in _schema.support_rows(conn, head["entry_id"],
                                          int(head["revision"]))
        ]
        row = dict(head)
        row.update(
            {
                "revision": revision,
                "state": new_state,
                "conflict_group": group,
                "recorded_us": now,
                "actor_id": caller.principal_id,
                "purpose": purpose,
                "operation_id": operation_id,
                "prev_revision": int(head["revision"]),
                "changed_fields_json": _schema.dumps(changed),
                "digest": "",
            }
        )
        if extra:
            row["changed_fields_json"] = _schema.dumps(
                changed + [f"meta:{k}" for k in sorted(extra)]
            )
        row["digest"] = _entry_digest(row, supports)
        _schema.insert_entry_row(conn, row)
        _schema.insert_support_rows(conn, head["entry_id"], revision, supports)
        self._register_object(
            conn, head["scope_id"], head["entry_id"], revision, new_state,
            row["digest"], now, producer_id, head["entry_kind"],
        )
        self._record_parents(
            conn, head["scope_id"], head["entry_id"], revision, supports,
            producer_id, operation_id,
        )
        self._log(
            conn, head["scope_id"], "entry_revise", caller.principal_id,
            entry_id=head["entry_id"], purpose=purpose,
            operation_id=operation_id,
            detail={
                "revision": revision,
                "prev_revision": head["revision"],
                "changed_fields": changed,
                "state": new_state,
                **(extra or {}),
            },
            us=now,
        )
        return row

    def _close_conflict(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        group: str,
        resolution_entry_id: str,
        now: int,
    ) -> None:
        row = _schema.get_conflict_row(conn, scope_id, group)
        if row is None:
            return
        _schema.put_conflict_row(
            conn,
            {
                **row,
                "state": "resolved",
                "resolved_us": now,
                "resolution_entry_id": resolution_entry_id,
            },
        )

    # ------------------------------------------------------------------
    # entry reads — kernel-gated, audience-closed (V4-21.03/10)
    # ------------------------------------------------------------------

    def get_entry(
        self,
        caller: CallerV3,
        entry_id: str,
        *,
        purpose: str = "recall",
    ) -> dict:
        require_id(entry_id, "entry_id")
        with self._store.read() as conn:
            if not _schema.tables_present(conn):
                _deny()
            head = _schema.entry_head_row(conn, entry_id)
            if head is None:
                _deny()
            return self._serve_entry(conn, caller, head, purpose=purpose)

    def entries(
        self,
        caller: CallerV3,
        scope_id: str,
        *,
        subject_id: Optional[str] = None,
        topic_key: Optional[str] = None,
        profile_id: Optional[str] = None,
        purpose: str = "recall",
        include_states: Optional[Iterable[str]] = None,
        limit: int = MAX_ENTRIES_PER_READ,
    ) -> dict:
        """Audience-gated profile listing. Never widens: scope is pinned,
        subjects/audience filter, sensitive topics require their declared
        purpose, and every entry's derivation parents must resolve under
        the caller's lease."""
        require_id(scope_id, "scope_id")
        with self._store.read() as conn:
            if not _schema.tables_present(conn):
                return {"entries": [], "withheld": 0, "scope_id": scope_id}
            self._lease(
                conn, caller, Verb.READ.value, purpose, (scope_id,)
            )
            states = tuple(include_states or _ENTRY_VISIBLE_STATES)
            fetch = min(int(limit), MAX_ENTRIES_PER_READ) * 4
            heads = _schema.head_rows(
                conn,
                scope_id,
                profile_id=profile_id,
                subject_id=subject_id,
                topic_key=topic_key,
                states=states,
                limit=fetch,
            )
            out: list[dict] = []
            withheld = 0
            cap = min(int(limit), MAX_ENTRIES_PER_READ)
            for h in heads:
                view = self._serve_entry(
                    conn, caller, h, purpose=purpose,
                    raise_on_deny=False,
                )
                if view is None:
                    withheld += 1
                    continue
                out.append(view)
                if len(out) >= cap:
                    break
            return {
                "scope_id": scope_id,
                "entries": out,
                "withheld": withheld,
                "scanned": len(heads),
                "truncated": len(heads) >= fetch or len(out) >= cap,
            }

    def _serve_entry(
        self,
        conn: sqlite3.Connection,
        caller: CallerV3,
        head: dict[str, Any],
        *,
        purpose: str,
        raise_on_deny: bool = True,
    ) -> Optional[dict]:
        """One entry through the full disclosure gate:
        lease → servable state → audience → sensitive-purpose →
        kernel-verified derivation parents (all-or-nothing)."""
        scope_id = head["scope_id"]
        try:
            self._lease(
                conn, caller, Verb.READ.value, purpose, (scope_id,)
            )
            admin = self._is_admin(conn, caller, scope_id)
            if head["state"] not in _ENTRY_VISIBLE_STATES:
                _deny()  # history() is the inspection path for these
            if (
                head["expires_us"] is not None
                and int(head["expires_us"]) <= now_us()
            ):
                _deny()  # expiry is a read-time gate, not just a sweep
            if not self._audience_ok(caller, head, admin):
                _deny()
            topic = _schema.get_topic_row(conn, scope_id, head["topic_key"])
            if not self._sensitive_ok(topic, purpose):
                _deny()
            supports = _schema.support_rows(
                conn, head["entry_id"], int(head["revision"])
            )
            if head["entry_kind"] in _DERIVED_KINDS and not supports:
                _deny()  # a parentless derivation is fabrication
            refs = [
                (s["support_kind"], s["support_id"], s["support_revision"])
                for s in supports
            ]
            refs.append((PROFILE_KIND, head["entry_id"], int(head["revision"])))
            lease = self._kernel.resolve_access(
                conn, caller, Verb.READ.value, purpose, (scope_id,), refs
            )
            if lease.denied:
                _deny()
        except VerbatimError as exc:
            if exc.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED and not raise_on_deny:
                return None
            raise
        return self._entry_view(conn, head, supports, caller, admin,
                                purpose)

    def _audience_ok(
        self, caller: CallerV3, head: dict[str, Any], admin: bool
    ) -> bool:
        """Closed audience: subject always sees own entries; everyone else
        needs explicit audience membership; admin inspects via admin verb.
        An empty audience set is *subject-only*, never public."""
        if admin:
            return True
        if caller.principal_id == head["subject_id"]:
            return True
        return caller.principal_id in set(
            _schema.loads(head["audience_json"], [])
        )

    def _sensitive_ok(
        self, topic: Optional[dict[str, Any]], purpose: str
    ) -> bool:
        """Sensitive delivery requires the *declared purpose*, full stop —
        authority level is orthogonal (admin inspects via history())."""
        if topic is None:
            return False  # deleted topic serves nothing on the read path
        if topic["sensitivity"] != Sensitivity.SENSITIVE.value:
            return True
        return purpose in set(
            _schema.loads(topic["required_purposes_json"], [])
        )

    def _entry_view(
        self,
        conn: sqlite3.Connection,
        head: dict[str, Any],
        supports: list[dict[str, Any]],
        caller: CallerV3,
        admin: bool,
        purpose: str,
    ) -> dict:
        resolved: list[dict[str, Any]] = []
        hidden = 0
        for s in supports:
            ref = (s["support_kind"], s["support_id"], s["support_revision"])
            ok = self._allowed(
                conn, caller, Verb.READ.value, purpose,
                (head["scope_id"],), (ref,),
            )
            if ok:
                resolved.append(
                    {
                        "kind": s["support_kind"],
                        "id": s["support_id"],
                        "revision": s["support_revision"],
                        "relation": s["relation"],
                    }
                )
            else:
                hidden += 1
        conflict = None
        if head["conflict_group"]:
            crow = _schema.get_conflict_row(
                conn, head["scope_id"], head["conflict_group"]
            )
            if crow:
                all_members = _schema.loads(crow["members_json"], [])
                # Non-admin callers learn the count, not sibling entry
                # ids — a conflicted sibling may sit behind a narrower
                # audience (§21.03: no widening via the group record).
                members = (
                    list(all_members)
                    if admin
                    else [
                        m for m in all_members
                        if m == head["entry_id"]
                    ]
                )
                conflict = {
                    "group": crow["conflict_group"],
                    "state": crow["state"],
                    "members": members,
                    "member_count": len(all_members),
                    "opened_us": crow["opened_us"],
                    "resolved_us": crow["resolved_us"],
                }
        return {
            "entry_id": head["entry_id"],
            "scope_id": head["scope_id"],
            "profile_id": head["profile_id"],
            "subject_id": head["subject_id"],
            "topic_key": head["topic_key"],
            "entry_kind": head["entry_kind"],
            "revision": head["revision"],
            "state": head["state"],
            "value": _schema.loads(head["value_json"]),
            "confidence": head["confidence"],
            "basis": head["basis"],
            "audience": _schema.loads(head["audience_json"], []),
            "effective_us": head["effective_us"],
            "expires_us": head["expires_us"],
            "recorded_us": head["recorded_us"],
            "actor_id": head["actor_id"],
            "purpose": head["purpose"],
            "prev_revision": head["prev_revision"],
            "changed_fields": _schema.loads(head["changed_fields_json"], []),
            "supersedes_entry_id": head["supersedes_entry_id"],
            "digest": head["digest"],
            "support": resolved,
            "support_withheld": hidden,
            "conflict": conflict,
        }

    def resolve_support(
        self,
        caller: CallerV3,
        entry_id: str,
        *,
        purpose: str = "recall",
    ) -> dict:
        """Resolve an entry's derivation parents through the kernel —
        the only path that may disclose support identifiers (V4-21.10)."""
        require_id(entry_id, "entry_id")
        with self._store.read() as conn:
            if not _schema.tables_present(conn):
                _deny()
            head = _schema.entry_head_row(conn, entry_id)
            if head is None:
                _deny()
            admin = self._is_admin(conn, caller, head["scope_id"])
            if not self._audience_ok(caller, head, admin):
                _deny()
            topic = _schema.get_topic_row(
                conn, head["scope_id"], head["topic_key"]
            )
            if not self._sensitive_ok(topic, purpose):
                _deny()  # support identifiers follow the delivery gate
            supports = _schema.support_rows(
                conn, entry_id, int(head["revision"])
            )
            refs = [
                (s["support_kind"], s["support_id"], s["support_revision"])
                for s in supports
            ]
            lease = self._kernel.resolve_access(
                conn, caller, Verb.READ.value, purpose,
                (head["scope_id"],), refs,
            )
            if lease.denied:
                _deny()
            return {
                "entry_id": entry_id,
                "revision": int(head["revision"]),
                "lease_id": lease.lease_id,
                "resolved": [
                    {"kind": s["support_kind"], "id": s["support_id"],
                     "revision": s["support_revision"],
                     "relation": s["relation"]}
                    for s in supports
                ],
                "epoch_vector": dict(lease.epoch_vector),
            }

    def history(
        self,
        caller: CallerV3,
        entry_id: str,
        *,
        purpose: str = "admin",
    ) -> dict:
        """Owner inspection (V4-21.07): every revision, every support row,
        raw — admin-gated so hidden identifiers never reach audience."""
        require_id(entry_id, "entry_id")
        with self._store.read() as conn:
            if not _schema.tables_present(conn):
                _deny()
            head = _schema.entry_head_row(conn, entry_id)
            if head is None:
                _deny()
            self._lease(
                conn, caller, Verb.ADMIN.value, purpose, (head["scope_id"],)
            )
            revisions = []
            for r in _schema.entry_revisions(conn, entry_id):
                revisions.append(
                    {
                        "revision": int(r["revision"]),
                        "state": r["state"],
                        "value": _schema.loads(r["value_json"]),
                        "confidence": r["confidence"],
                        "changed_fields": _schema.loads(
                            r["changed_fields_json"], []
                        ),
                        "prev_revision": r["prev_revision"],
                        "effective_us": r["effective_us"],
                        "expires_us": r["expires_us"],
                        "recorded_us": r["recorded_us"],
                        "actor_id": r["actor_id"],
                        "purpose": r["purpose"],
                        "operation_id": r["operation_id"],
                        "producer_id": r["producer_id"],
                        "digest": r["digest"],
                        "supports": [
                            {
                                "kind": s["support_kind"],
                                "id": s["support_id"],
                                "revision": s["support_revision"],
                                "relation": s["relation"],
                            }
                            for s in _schema.support_rows(
                                conn, entry_id, int(r["revision"])
                            )
                        ],
                    }
                )
            return {
                "entry_id": entry_id,
                "scope_id": head["scope_id"],
                "profile_id": head["profile_id"],
                "subject_id": head["subject_id"],
                "topic_key": head["topic_key"],
                "entry_kind": head["entry_kind"],
                "head_revision": int(head["revision"]),
                "head_state": head["state"],
                "revisions": revisions,
            }

    # ------------------------------------------------------------------
    # deletion / invalidation (V4-21.07 + evidence-purge closure)
    # ------------------------------------------------------------------

    def delete_entry(
        self,
        caller: CallerV3,
        entry_id: str,
        *,
        purpose: str = "admin",
    ) -> dict:
        """Owner delete: terminal tombstone revision + registry
        disposition ``erased``. Provenance edges stay (they describe a
        real derivation); the *value* is what dies."""
        require_id(entry_id, "entry_id")
        now = now_us()
        with self._store.tx() as conn:
            self._prepare(conn)
            head = _schema.entry_head_row(conn, entry_id)
            if head is None:
                _deny()
            self._lease(
                conn, caller, Verb.ADMIN.value, purpose, (head["scope_id"],)
            )
            if head["state"] == EntryState.TOMBSTONED.value:
                return {"entry_id": entry_id, "state": "tombstoned",
                        "changed": False}
            self._revise_state(
                conn, head, EntryState.TOMBSTONED.value, caller, purpose,
                f"op:{new_id()}", now,
            )
            return {"entry_id": entry_id, "state": "tombstoned",
                    "changed": True}

    def resolve_conflict(
        self,
        caller: CallerV3,
        scope_id: str,
        conflict_group: str,
        keep_entry_id: str,
        *,
        purpose: str = "admin",
    ) -> dict:
        """Owner resolves an open contradiction: the kept entry returns
        to ``active``, every other member is superseded — all recorded as
        new revisions, never a silent overwrite (V4-21.05/07)."""
        require_id(scope_id, "scope_id")
        require_id(conflict_group, "conflict_group")
        require_id(keep_entry_id, "keep_entry_id")
        now = now_us()
        with self._store.tx() as conn:
            self._prepare(conn)
            self._lease(conn, caller, Verb.ADMIN.value, purpose, (scope_id,))
            crow = _schema.get_conflict_row(conn, scope_id, conflict_group)
            if crow is None:
                _deny()
            members = list(_schema.loads(crow["members_json"], []))
            if keep_entry_id not in members:
                _validation("keep_entry_id is not a conflict member")
            op = f"op:{new_id()}"
            superseded = []
            for mid in members:
                h = _schema.entry_head_row(conn, mid)
                if h is None or h["scope_id"] != scope_id:
                    continue
                if mid == keep_entry_id:
                    if h["state"] == EntryState.CONFLICTED.value:
                        self._revise_state(
                            conn, h, EntryState.ACTIVE.value, caller,
                            purpose, op, now, conflict_group=None,
                            extra={"reason": "conflict_resolved"},
                        )
                    continue
                if h["state"] in _ENTRY_VISIBLE_STATES:
                    self._revise_state(
                        conn, h, EntryState.SUPERSEDED.value, caller,
                        purpose, op, now, conflict_group=None,
                        extra={
                            "reason": "conflict_resolved",
                            "superseded_by": keep_entry_id,
                        },
                    )
                    superseded.append(mid)
            self._close_conflict(
                conn, scope_id, conflict_group, keep_entry_id, now
            )
            self._log(
                conn, scope_id, "conflict_resolved", caller.principal_id,
                entry_id=keep_entry_id, purpose=purpose, operation_id=op,
                detail={"conflict_group": conflict_group,
                        "superseded": superseded},
                us=now,
            )
            return {
                "conflict_group": conflict_group,
                "kept": keep_entry_id,
                "superseded": superseded,
                "state": "resolved",
            }

    def delete_topic(
        self,
        caller: CallerV3,
        scope_id: str,
        topic_key: str,
        *,
        purpose: str = "admin",
    ) -> dict:
        """Delete a topic with complete derivative invalidation: the
        config row tombstones and every entry under it tombstones too
        (V4-21.07)."""
        require_id(scope_id, "scope_id")
        require_id(topic_key, "topic_key")
        now = now_us()
        with self._store.tx() as conn:
            self._prepare(conn)
            self._lease(conn, caller, Verb.ADMIN.value, purpose, (scope_id,))
            topic = _schema.get_topic_row(conn, scope_id, topic_key)
            if topic is None:
                _deny()
            topic = dict(topic)
            topic["state"] = "deleted"
            topic["updated_us"] = now
            topic["updated_by"] = caller.principal_id
            _schema.put_topic_row(conn, topic)
            tombstoned = []
            for h in _schema.head_rows(
                conn, scope_id, topic_key=topic_key,
                states=None, limit=MAX_ENTRIES_PER_READ * 4,
            ):
                if h["state"] == EntryState.TOMBSTONED.value:
                    continue
                self._revise_state(
                    conn, h, EntryState.TOMBSTONED.value, caller, purpose,
                    f"op:{new_id()}", now,
                    extra={"reason": "topic_deleted"},
                )
                tombstoned.append(h["entry_id"])
            self._log(
                conn, scope_id, "topic_deleted", caller.principal_id,
                purpose=purpose,
                detail={"topic_key": topic_key, "tombstoned": tombstoned},
                us=now,
            )
            return {
                "topic_key": topic_key,
                "state": "deleted",
                "entries_tombstoned": tombstoned,
            }

    def refresh(
        self,
        caller: CallerV3,
        scope_id: str,
        *,
        purpose: str = "admin",
    ) -> dict:
        """Reconcile stored entries against current evidence viability.

        * task-local entries past expiry → ``expired``
        * entries whose derivation parents are held/purged → ``withheld``
        * entries whose parents are all gone → ``tombstoned``
        * support pointers to provably-erased objects are scrubbed (no
          surviving reference may name erased material)
        * withheld entries whose parents recovered → revived to their
          pre-withhold visibility.
        """
        require_id(scope_id, "scope_id")
        now = now_us()
        report: dict[str, Any] = {
            "withheld": [], "tombstoned": [], "expired": [],
            "revived": [], "scrubbed_refs": 0, "scanned": 0,
        }
        with self._store.tx() as conn:
            self._prepare(conn)
            self._lease(conn, caller, Verb.ADMIN.value, purpose, (scope_id,))
            heads = _schema.head_rows(
                conn, scope_id, states=None, limit=MAX_ENTRIES_PER_READ * 8
            )
            for h in heads:
                if h["state"] in (
                    EntryState.TOMBSTONED.value,
                    EntryState.SUPERSEDED.value,
                ):
                    continue
                report["scanned"] += 1
                op = f"op:{new_id()}"
                if (
                    h["expires_us"] is not None
                    and int(h["expires_us"]) <= now
                    and h["state"] != EntryState.EXPIRED.value
                ):
                    self._revise_state(
                        conn, h, EntryState.EXPIRED.value, caller, purpose,
                        op, now, producer_id=PRODUCER_REFRESH,
                    )
                    report["expired"].append(h["entry_id"])
                    continue
                supports = _schema.support_rows(
                    conn, h["entry_id"], int(h["revision"])
                )
                statuses = []
                for s in supports:
                    status, _scope = self._support_status(
                        conn, s["support_kind"], s["support_id"],
                        int(s["support_revision"]),
                    )
                    statuses.append((s, status))
                gone = [s for s, st in statuses if st == "gone"]
                held = [s for s, st in statuses if st == "held"]
                live_support = [
                    s for s, st in statuses
                    if st == "live" and s["relation"] == "supports"
                ]
                for s in gone:
                    _schema.delete_support_row(
                        conn, h["entry_id"], int(h["revision"]),
                        int(s["seq"]),
                    )
                    report["scrubbed_refs"] += 1
                if h["state"] in _ENTRY_VISIBLE_STATES:
                    if h["entry_kind"] in _DERIVED_KINDS and not supports:
                        # Closure stripped every derivation parent — a
                        # parentless derivation is fabrication, and this
                        # one can never re-derive its provenance.
                        self._revise_state(
                            conn, h, EntryState.TOMBSTONED.value, caller,
                            purpose, op, now,
                            extra={"reason": "provenance_stripped"},
                            producer_id=PRODUCER_REFRESH,
                        )
                        report["tombstoned"].append(h["entry_id"])
                    elif supports and not live_support and gone:
                        # every derivation parent is provably erased —
                        # the entry cannot ever revive.
                        self._revise_state(
                            conn, h, EntryState.TOMBSTONED.value, caller,
                            purpose, op, now,
                            extra={"reason": "support_erased"},
                            producer_id=PRODUCER_REFRESH,
                        )
                        report["tombstoned"].append(h["entry_id"])
                    elif gone or held:
                        self._revise_state(
                            conn, h, EntryState.WITHHELD.value, caller,
                            purpose, op, now,
                            extra={
                                "reason": "support_invalid",
                                "gone": len(gone), "held": len(held),
                            },
                            producer_id=PRODUCER_REFRESH,
                        )
                        report["withheld"].append(h["entry_id"])
                elif h["state"] == EntryState.WITHHELD.value:
                    if not gone and not held:
                        # support recovered (e.g. quarantine released)
                        prev = _schema.entry_revision_row(
                            conn, h["entry_id"], int(h["prev_revision"] or 0)
                        )
                        target = (
                            prev["state"]
                            if prev and prev["state"] in _ENTRY_VISIBLE_STATES
                            else EntryState.ACTIVE.value
                        )
                        self._revise_state(
                            conn, h, target, caller, purpose, op, now,
                            extra={"reason": "support_recovered"},
                            producer_id=PRODUCER_REFRESH,
                        )
                        report["revived"].append(h["entry_id"])
            self._log(
                conn, scope_id, "refresh", caller.principal_id,
                purpose=purpose, detail=report, us=now,
            )
        return report

    def _support_status(
        self,
        conn: sqlite3.Connection,
        kind: str,
        oid: str,
        revision: int,
    ) -> tuple[str, Optional[str]]:
        """Lifecycle viability of a derivation parent — mirrors the
        kernel's ``_check_object`` semantics (lifecycle + quarantine
        cascade + purge suppression) without consuming a caller lease.
        Returns (status, scope_id): ``live`` / ``held`` / ``gone``."""
        scope: Optional[str] = None
        exists = False
        held = False
        if kind == "claim":
            r = conn.execute(
                "SELECT scope_id FROM claims WHERE claim_id=?", (oid,)
            ).fetchone()
            if r is None:
                return "gone", None
            scope = r[0]
            rev = conn.execute(
                "SELECT state FROM claim_revisions"
                " WHERE claim_id=? AND revision=?",
                (oid, revision),
            ).fetchone()
            if rev is None:
                return "gone", scope
            state = rev[0]
            if state == "erased":
                return "gone", scope
            exists = state in _CLAIM_VISIBLE
            held = (
                not exists
                or _quarantine.is_quarantined(conn, "claim", oid, revision)
            )
            if exists and not held:
                held = self._claim_evidence_held(conn, oid, revision)
        elif kind == "span":
            r = conn.execute(
                "SELECT s.revision, src.scope_id, s.source_id"
                " FROM spans s JOIN sources src"
                "  ON src.source_id=s.source_id WHERE s.span_id=?",
                (oid,),
            ).fetchone()
            if r is None:
                return "gone", None
            scope = r[1]
            exists = int(r[0]) == int(revision)
            if exists:
                held = (
                    _quarantine.is_quarantined(conn, "span", oid, revision)
                    or _quarantine.is_quarantined(
                        conn, "source", r[2], int(r[0])
                    )
                    or self._envelope_held(conn, r[2], int(r[0]))
                )
        elif kind == "source":
            r = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id=?", (oid,)
            ).fetchone()
            if r is None:
                return "gone", None
            scope = r[0]
            rev = conn.execute(
                "SELECT length(payload) FROM source_revisions"
                " WHERE source_id=? AND revision=?",
                (oid, revision),
            ).fetchone()
            if rev is None or int(rev[0] or 0) == 0:
                return "gone", scope  # scrubbed payload = erased
            exists = True
            held = (
                _quarantine.is_quarantined(conn, "source", oid, revision)
                or self._envelope_held(conn, oid, revision)
            )
        elif kind == "source_revision":
            r = conn.execute(
                "SELECT s.scope_id, length(sr.payload)"
                " FROM source_revisions sr JOIN sources s"
                "  ON s.source_id=sr.source_id"
                " WHERE sr.source_id=? AND sr.revision=?",
                (oid, revision),
            ).fetchone()
            if r is None or int(r[1] or 0) == 0:
                return "gone", None if r is None else r[0]
            scope = r[0]
            exists = True
            held = (
                _quarantine.is_quarantined(
                    conn, "source_revision", oid, revision
                )
                or _quarantine.is_quarantined(conn, "source", oid, revision)
                or self._envelope_held(conn, oid, revision)
            )
        elif kind == "profile":
            r = _schema.entry_head_row(conn, oid)
            if r is None:
                return "gone", None
            scope = r["scope_id"]
            if r["state"] == EntryState.TOMBSTONED.value:
                return "gone", scope
            exists = True
            held = r["state"] not in _ENTRY_VISIBLE_STATES or (
                _quarantine.is_quarantined(conn, "profile", oid, revision)
            )
        else:
            reg = repos_v4.get(
                conn, "objects", {"kind": kind, "object_id": oid}
            )
            if reg is not None:
                scope = reg["scope_id"]
                if reg["disposition"] == "erased":
                    return "gone", scope
                exists = True
                held = reg["disposition"] != "active" or (
                    _quarantine.is_quarantined(conn, kind, oid, revision)
                )
            else:
                return "held", None  # unverifiable — fail closed
        pstate = self._purge_state(conn, kind, oid)
        if pstate == "completed":
            return "gone", scope  # closure ran — the object was erased
        if exists and pstate is not None:
            held = True
        if not exists:
            return "held", scope  # present-but-not-visible, or ambiguous
        return ("held" if held else "live"), scope

    def _claim_evidence_held(
        self, conn: sqlite3.Connection, claim_id: str, revision: int
    ) -> bool:
        if not has_table(conn, "claim_evidence"):
            return False
        spans = conn.execute(
            "SELECT span_id FROM claim_evidence"
            " WHERE claim_id=? AND revision=? LIMIT ?",
            (claim_id, revision, 512),
        ).fetchall()
        for (span_id,) in spans:
            srow = conn.execute(
                "SELECT revision, source_id FROM spans WHERE span_id=?",
                (span_id,),
            ).fetchone()
            if srow is None:
                continue
            srev = int(srow[0])
            if _quarantine.is_quarantined(conn, "span", span_id, srev):
                return True
            if _quarantine.is_quarantined(
                conn, "source", srow[1], srev
            ):
                return True
            if self._envelope_held(conn, srow[1], srev):
                return True
        return False

    def _envelope_held(
        self, conn: sqlite3.Connection, source_id: str, revision: int
    ) -> bool:
        if not has_table(conn, "source_envelopes"):
            return False
        for (env_id,) in conn.execute(
            "SELECT envelope_id FROM source_envelopes"
            " WHERE source_id=? AND revision=?",
            (source_id, revision),
        ).fetchall():
            if _quarantine.is_quarantined(
                conn, "source_envelope", env_id, revision
            ):
                return True
        return False

    def _purge_state(
        self, conn: sqlite3.Connection, kind: str, oid: str
    ) -> Optional[str]:
        """Most advanced purge state naming this object: ``completed``
        means the closure already erased it (gone), ``suppressed``/
        ``purging`` mean the hold is live (held)."""
        if not (
            has_table(conn, "purge_targets") and has_table(conn, "purges")
        ):
            return None
        r = conn.execute(
            "SELECT p.state FROM purge_targets pt JOIN purges p"
            "  ON p.purge_id=pt.purge_id"
            " WHERE pt.object_kind=? AND pt.object_id=?"
            "   AND p.state IN ('suppressed','purging','completed')"
            " ORDER BY CASE p.state"
            "   WHEN 'completed' THEN 0 WHEN 'purging' THEN 1 ELSE 2 END"
            " LIMIT 1",
            (kind, oid),
        ).fetchone()
        return r[0] if r else None

    # ------------------------------------------------------------------
    # compile — on-demand derivation (durable job deferred; see docstring)
    # ------------------------------------------------------------------

    def compile(
        self,
        caller: CallerV3,
        scope_id: str,
        *,
        subject_id: Optional[str] = None,
        topics: Optional[Iterable[str]] = None,
        purpose: str = "derive",
    ) -> dict:
        """Derive profile entries from live claim evidence.

        Bounded scan: at most ``MAX_COMPILE_CLAIMS`` claim heads are
        considered and ``truncated`` reports the cap honestly. Each claim
        is re-authorized through the kernel before it may contribute —
        inaccessible evidence is skipped and counted, never silently
        derived from."""
        require_id(scope_id, "scope_id")
        now = now_us()
        report: dict[str, Any] = {
            "scope_id": scope_id,
            "claims_scanned": 0,
            "claims_matched": 0,
            "skipped_unauthorized": 0,
            "skipped_min_support": 0,
            "skipped_sensitive": 0,
            "skipped_policy": 0,
            "safety_events": [],
            "entries_created": [],
            "entries_revised": [],
            "superseded": [],
            "conflicts": [],
            "truncated": False,
            "durable_job": False,
            "durable_job_reason": (
                "no profile_compile JobKind is registered — the jobs "
                "table CHECK constraint is schema-owned; compilation "
                "runs on-demand inside this transaction"
            ),
        }
        with self._store.tx() as conn:
            self._prepare(conn)
            lease = self._lease(
                conn, caller, Verb.DERIVE.value, purpose, (scope_id,)
            )
            topic_rows = _schema.topic_rows(conn, scope_id, state="active")
            if topics is not None:
                wanted = {str(t) for t in topics}
                topic_rows = [t for t in topic_rows if t["topic_key"] in wanted]
            if not topic_rows:
                report["topics"] = []
                return report
            claim_ids = self._scan_claim_ids(conn, scope_id, subject_id)
            if len(claim_ids) > MAX_COMPILE_CLAIMS:
                claim_ids = claim_ids[:MAX_COMPILE_CLAIMS]
                report["truncated"] = True
            report["claims_scanned"] = len(claim_ids)
            # group -> (topic, subject, canonical value) -> claims
            groups: dict[tuple[str, str, str], dict[str, Any]] = {}
            totals: dict[tuple[str, str], int] = {}
            for cid in claim_ids:
                head = read_claim_head(conn, cid)
                if head is None or head.state.value not in _CLAIM_VISIBLE:
                    report["skipped_unauthorized"] += 1
                    continue
                if not self._allowed(
                    conn, caller, Verb.DERIVE.value, purpose, (scope_id,),
                    (("claim", cid, int(head.revision)),),
                ):
                    report["skipped_unauthorized"] += 1
                    continue
                # X5: safety-bearing events are named in the report no
                # matter which topics they inform — a steering policy can
                # never make them invisible (V45-09.05, V4-16.06, D17).
                category = safety_event_category(head)
                if category is not None:
                    report["safety_events"].append(
                        {"claim_id": cid, "category": category}
                    )
                for topic in topic_rows:
                    if not self._matches(topic, head):
                        continue
                    tsub = head.subject_id or subject_id
                    if tsub is None:
                        continue
                    kind = (
                        EntryKind.INFERRED.value
                        if topic["sensitivity"] == Sensitivity.SENSITIVE.value
                        else EntryKind.OBSERVED.value
                    )
                    if kind == EntryKind.INFERRED.value and not (
                        self._inference_ok(conn, scope_id, tsub, topic,
                                           purpose)
                    ):
                        report["skipped_sensitive"] += 1
                        continue
                    if (
                        topic["conflict_policy"]
                        == ConflictPolicy.EXPLICIT_ONLY.value
                    ):
                        report["skipped_policy"] += 1
                        continue
                    value = self._claim_value(head)
                    key = (topic["topic_key"], tsub, _canonical(value))
                    g = groups.setdefault(
                        key,
                        {
                            "topic": topic,
                            "subject_id": tsub,
                            "value": value,
                            "kind": kind,
                            "supports": [],
                            "effective": 0,
                        },
                    )
                    g["supports"].append(
                        {
                            "kind": "claim",
                            "id": cid,
                            "revision": int(head.revision),
                            "relation": "supports",
                            "claim_role": "derived_from",
                        }
                    )
                    g["effective"] = max(
                        g["effective"], int(head.recorded_from or 0)
                    )
                    totals[(topic["topic_key"], tsub)] = (
                        totals.get((topic["topic_key"], tsub), 0) + 1
                    )
                    report["claims_matched"] += 1
            operation_id = f"compile:{new_id()}"
            for (tkey, tsub, _), g in sorted(groups.items()):
                topic = g["topic"]
                if len(g["supports"]) < int(topic["min_support"]):
                    report["skipped_min_support"] += 1
                    continue
                total = totals[(tkey, tsub)]
                conf = round(
                    min(1.0, len(g["supports"]) / max(1, total)), 4
                )
                preferred = set(
                    _schema.loads(topic["preferred_evidence_json"], [])
                )
                basis = (
                    f"{len(g['supports'])} claim(s); "
                    f"predicate-preferred="
                    f"{g['value'].get('predicate') in preferred}"
                )
                entry_id = self._derived_entry_id(
                    scope_id, tsub, tkey, g["kind"], g["value"]
                )
                res = self._write_entry(
                    conn,
                    caller=caller,
                    scope_id=scope_id,
                    topic=topic,
                    subject_id=tsub,
                    kind=g["kind"],
                    value=g["value"],
                    supports=g["supports"],
                    confidence=conf,
                    basis=basis,
                    audience=None,
                    effective_us=g["effective"] or now,
                    expires_us=None,
                    entry_id=entry_id,
                    profile_id=None,
                    purpose=purpose,
                    operation_id=operation_id,
                    producer_id=PRODUCER_COMPILE,
                    now=now,
                )
                if res["revision"] == 1:
                    report["entries_created"].append(entry_id)
                else:
                    report["entries_revised"].append(entry_id)
                if res["state"] == EntryState.CONFLICTED.value:
                    report["conflicts"].append(entry_id)
            self._supersede_stale(
                conn, scope_id, caller, purpose, operation_id, now,
                groups, subject_id, topic_rows, report,
            )
            report["operation_id"] = operation_id
            report["topics"] = [t["topic_key"] for t in topic_rows]
        return report

    def _scan_claim_ids(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        subject_id: Optional[str],
    ) -> list[str]:
        if subject_id is not None:
            rows = conn.execute(
                "SELECT claim_id FROM claims"
                " WHERE scope_id=? AND subject_id=? ORDER BY claim_id"
                " LIMIT ?",
                (scope_id, subject_id, MAX_COMPILE_CLAIMS + 1),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT claim_id FROM claims WHERE scope_id=?"
                " AND subject_id IS NOT NULL ORDER BY claim_id LIMIT ?",
                (scope_id, MAX_COMPILE_CLAIMS + 1),
            ).fetchall()
        return [r[0] for r in rows]

    def _matches(self, topic: dict[str, Any], head: Any) -> bool:
        # X5 (V45-09.05, V4-16.06): a safety-bearing claim — deletion,
        # consent withdrawal, correction/invalidation, security — can
        # never be filtered out by a topic's positive match spec. It may
        # inform any active topic and is reported under ``safety_events``
        # by compile; a narrow allowlist is never a suppression tool.
        if safety_event_category(head) is not None:
            return True
        match = _schema.loads(topic["match_json"], {})
        subjects = match.get("subjects") or []
        if subjects and (head.subject_id or "") not in subjects:
            return False
        preds = match.get("predicates") or []
        if preds and (head.predicate or "") not in preds:
            return False
        keywords = [str(k).lower() for k in (match.get("keywords") or [])]
        if not keywords and not preds:
            return True  # unconfigured match surface = all claims pass
        if not keywords:
            return True  # predicate allowlist already matched
        text = f"{head.predicate or ''} {head.object_json or ''}".lower()
        return any(k in text for k in keywords)

    def _claim_value(self, head: Any) -> dict[str, Any]:
        try:
            obj = json.loads(head.object_json) if head.object_json else None
        except (TypeError, ValueError):
            obj = head.object_json
        return {"predicate": head.predicate, "object": obj}

    def _derived_entry_id(
        self,
        scope_id: str,
        subject_id: str,
        topic_key: str,
        kind: str,
        value: Any,
    ) -> str:
        """Deterministic derived-entry id — same evidence re-derives the
        same identity so recompiles are idempotent revisions, not dupes."""
        h = hashlib.sha256(
            _canonical(
                ["pe", scope_id, subject_id, topic_key, kind, value]
            ).encode()
        ).hexdigest()
        return f"pe:{h[:24]}"

    def _supersede_stale(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        caller: CallerV3,
        purpose: Optional[str],
        operation_id: str,
        now: int,
        groups: dict,
        subject_id: Optional[str],
        topic_rows: list[dict[str, Any]],
        report: dict[str, Any],
    ) -> None:
        """Derived heads the evidence no longer produces are superseded —
        the profile reflects *current* derivations, not history."""
        live_keys = set(groups.keys())
        tkeys = {t["topic_key"] for t in topic_rows}
        heads = _schema.head_rows(
            conn, scope_id, subject_id=subject_id,
            states=_ENTRY_VISIBLE_STATES, limit=MAX_ENTRIES_PER_READ * 4,
        )
        for h in heads:
            if h["entry_kind"] not in _DERIVED_KINDS:
                continue
            if h["topic_key"] not in tkeys:
                continue
            key = (h["topic_key"], h["subject_id"], h["value_json"])
            if key in live_keys:
                continue
            self._revise_state(
                conn, h, EntryState.SUPERSEDED.value, caller, purpose,
                operation_id, now,
                extra={"reason": "no_longer_derived"},
            )
            report["superseded"].append(h["entry_id"])

    # ------------------------------------------------------------------
    # perspective packs (V4-21.03/06/08)
    # ------------------------------------------------------------------

    def build_perspective_pack(
        self,
        caller: CallerV3,
        scope_id: str,
        *,
        subjects: Iterable[str] = (),
        audience: Iterable[str] = (),
        topics: Iterable[str] = (),
        verbs: Iterable[str] = ("read", "quote"),
        purpose: str = "recall",
        perspective_id: Optional[str] = None,
        persist: bool = False,
    ) -> dict:
        """Declare a retrieval perspective. ``persist=True`` writes a
        governance ``perspectives`` row (asserter=caller, subjects,
        audience) under the ``derive`` verb; ephemeral packs are marked
        ``persisted: false`` honestly."""
        require_id(scope_id, "scope_id")
        subs = tuple(require_id(str(s), "subjects") for s in subjects)
        aud = tuple(require_id(str(a), "audience") for a in audience)
        tkeys = tuple(require_id(str(t), "topics") for t in topics)
        try:
            vset = tuple(Verb(str(v)).value for v in verbs)
        except ValueError:
            _validation("verbs must be known governance verbs")
        now = now_us()
        with self._store.tx() as conn:
            self._prepare(conn)
            self._lease(conn, caller, Verb.READ.value, purpose, (scope_id,))
            if persist:
                self._lease(
                    conn, caller, Verb.DERIVE.value, purpose, (scope_id,)
                )
                pid = perspective_id or f"persp:{new_id()}"
                _perspectives.create_perspective(
                    conn,
                    scope_id=scope_id,
                    asserter=caller.principal_id,
                    subjects=subs,
                    audience=aud,
                    perspective_id=pid,
                    created_event=int(now),
                )
            else:
                pid = perspective_id or f"persp:{new_id()}"
            pack = PerspectivePack(
                perspective_id=pid,
                scope_id=scope_id,
                caller_id=caller.principal_id,
                subjects=subs,
                audience=aud,
                topics=tkeys,
                verbs=vset,
                purpose=purpose,
                persisted=persist,
                issued_us=now,
                expires_us=now + _PACK_TTL_US,
            )
            self._log(
                conn, scope_id, "perspective_pack", caller.principal_id,
                purpose=purpose,
                detail={"perspective_id": pid, "persisted": persist,
                        "verbs": list(vset)},
                us=now,
            )
        return pack.to_dict()

    def apply_perspective(
        self,
        caller: CallerV3,
        pack: dict[str, Any],
        candidates: Iterable[dict[str, Any]],
        *,
        purpose: Optional[str] = None,
        now_us_: Optional[int] = None,
    ) -> dict:
        """Rank/narrow the caller's own candidates under the pack.

        Attenuation only: output ⊆ input; a pack whose caller lacks
        ``read`` yields nothing; ``liftable`` marks whether the caller's
        ``quote`` verb resolved — the pack never lifts bytes itself."""
        try:
            p = PerspectivePack.from_dict(pack)
        except (KeyError, TypeError, ValueError) as exc:
            _validation(f"malformed perspective pack: {exc}")
        purpose = purpose or p.purpose
        now = int(now_us_ if now_us_ is not None else now_us())
        result: dict[str, Any] = {
            "perspective_id": p.perspective_id,
            "scope_id": p.scope_id,
            "authorized_verbs": [],
            "denied_verbs": [],
            "items": [],
            "dropped": 0,
            "profile_hints": [],
            "attenuated": True,
        }
        if p.caller_id != caller.principal_id:
            result["reason"] = "pack_not_transferable"
            return result
        if p.expires_us and now >= p.expires_us:
            result["reason"] = "pack_expired"
            return result
        with self._store.read() as conn:
            if not _schema.tables_present(conn):
                result["reason"] = "profiles_unavailable"
                return result
            for v in p.verbs:
                if self._allowed(
                    conn, caller, v, purpose, (p.scope_id,)
                ):
                    result["authorized_verbs"].append(v)
                else:
                    result["denied_verbs"].append(v)
            if Verb.READ.value not in result["authorized_verbs"]:
                result["reason"] = "unauthorized"
                return result
            lift = Verb.QUOTE.value in result["authorized_verbs"]
            hints = self._pack_hints(conn, caller, p, purpose)
            result["profile_hints"] = hints
            items: list[dict[str, Any]] = []
            dropped = 0
            for cand in candidates:
                c = dict(cand)
                cscope = c.get("scope_id")
                if cscope is not None and cscope != p.scope_id:
                    dropped += 1
                    continue  # scope isolation — foreign candidates drop
                ctopics = {str(t) for t in (c.get("topics") or ())}
                csubs = {str(s) for s in (c.get("subjects") or ())}
                matched_topics = sorted(ctopics & set(p.topics))
                matched_subjects = sorted(csubs & set(p.subjects))
                boost = 0.0
                boost += 0.25 * len(matched_topics)
                boost += 0.25 * len(matched_subjects)
                text = str(c.get("value") or c.get("text") or "").lower()
                for h in hints:
                    v = _canonical(h["value"]).lower()
                    if h["topic_key"] in ctopics or (
                        text and any(
                            tok and tok in text
                            for tok in self._value_tokens(h["value"])
                        )
                    ):
                        boost += 0.5 * float(h["confidence"])
                base = float(c.get("score") or 0.0)
                c["perspective_score"] = round(base + boost, 6)
                c["matched_topics"] = matched_topics
                c["matched_subjects"] = matched_subjects
                c["liftable"] = lift
                items.append(c)
            items.sort(
                key=lambda c: (-c["perspective_score"],
                               str(c.get("id") or ""))
            )
            result["items"] = items
            result["dropped"] = dropped
            return result

    def _pack_hints(
        self,
        conn: sqlite3.Connection,
        caller: CallerV3,
        pack: PerspectivePack,
        purpose: str,
    ) -> list[dict[str, Any]]:
        """Profile entries usable for ranking under this pack — same
        disclosure gate as ``entries()`` (audience + sensitivity +
        kernel-verified support)."""
        heads = _schema.head_rows(
            conn,
            pack.scope_id,
            states=_ENTRY_VISIBLE_STATES,
            limit=MAX_ENTRIES_PER_READ,
        )
        wanted_topics = set(pack.topics)
        wanted_subjects = set(pack.subjects)
        hints: list[dict[str, Any]] = []
        for h in heads:
            if wanted_topics and h["topic_key"] not in wanted_topics:
                continue
            if wanted_subjects and h["subject_id"] not in wanted_subjects:
                continue
            view = self._serve_entry(
                conn, caller, h, purpose=purpose,
                raise_on_deny=False,
            )
            if view is None:
                continue
            hints.append(view)
        return hints

    @staticmethod
    def _value_tokens(value: Any) -> list[str]:
        if isinstance(value, dict):
            obj = value.get("object")
            if isinstance(obj, dict):
                parts = [str(v) for v in obj.values()]
            else:
                parts = [str(obj)] if obj is not None else []
            if value.get("predicate"):
                parts.append(str(value["predicate"]))
        else:
            parts = [str(value)]
        tokens: list[str] = []
        for p in parts:
            tokens.extend(t for t in p.lower().split() if len(t) > 2)
        return tokens

    # ------------------------------------------------------------------
    # context selection (V4-21.06) — stable profile + recent events inside
    # the *serialized* byte budget, formatting and provenance included.
    # ------------------------------------------------------------------

    #: Kind priority inside the stable section — declared preference
    #: outranks observation, observation outranks inference.
    _CONTEXT_KIND_ORDER = {
        EntryKind.EXPLICIT.value: 0,
        EntryKind.SENSITIVE.value: 1,
        EntryKind.OBSERVED.value: 2,
        EntryKind.INFERRED.value: 3,
        EntryKind.TASK_LOCAL.value: 4,
    }

    def context_pack(
        self,
        caller: CallerV3,
        scope_id: str,
        *,
        subject_id: Optional[str] = None,
        topics: Iterable[str] = (),
        recent_limit: int = 16,
        max_bytes: int = 8192,
        purpose: str = "recall",
    ) -> dict:
        """Build a serialized context bundle within ``max_bytes``.

        The budget is charged against the *final serialized form* —
        envelope, formatting, and provenance all cost bytes. The stable
        profile section fills first but reserves a quarter of the budget
        for recent in-scope claims (the ``recent`` section); leftover room
        from either side is released to the other. Every packed item was
        disclosed through the same gates as ``entries()`` — a withheld
        entry cannot occupy budget. Returns a plain dict; ``truncated``
        reports whether items were dropped for budget (never for
        authorization — those are simply absent)."""
        require_id(scope_id, "scope_id")
        if int(max_bytes) <= 0:
            _validation("max_bytes must be positive")
        tkeys = {require_id(str(t), "topics") for t in topics}
        now = now_us()
        with self._store.read() as conn:
            if not _schema.tables_present(conn):
                return {
                    "format": "v4-context-pack/1",
                    "scope_id": scope_id,
                    "state": "unavailable",
                    "reason": "profile tables absent",
                    "stable": [],
                    "recent": [],
                    "bytes_used": 0,
                    "bytes_budget": int(max_bytes),
                }
            self._lease(
                conn, caller, Verb.READ.value, purpose, (scope_id,)
            )
            heads = _schema.head_rows(
                conn,
                scope_id,
                subject_id=subject_id,
                states=_ENTRY_VISIBLE_STATES,
                limit=MAX_ENTRIES_PER_READ,
            )
            stable_pool: list[dict[str, Any]] = []
            for h in heads:
                if tkeys and h["topic_key"] not in tkeys:
                    continue
                view = self._serve_entry(
                    conn, caller, h, purpose=purpose, raise_on_deny=False
                )
                if view is None:
                    continue
                stable_pool.append(
                    {
                        "entry_id": view["entry_id"],
                        "topic_key": view["topic_key"],
                        "entry_kind": view["entry_kind"],
                        "state": view["state"],
                        "value": view["value"],
                        "confidence": view["confidence"],
                        "effective_us": view["effective_us"],
                        "expires_us": view["expires_us"],
                        "revision": int(view["revision"]),
                        "support": view["support"],
                        "support_withheld": view["support_withheld"],
                    }
                )
            stable_pool.sort(
                key=lambda it: (
                    self._CONTEXT_KIND_ORDER.get(it["entry_kind"], 9),
                    -float(it["confidence"] or 0.0),
                    it["topic_key"],
                    it["entry_id"],
                )
            )
            # Recent events: newest in-scope claim revisions, optionally
            # subject-filtered — each verified through the kernel.
            recent_pool: list[dict[str, Any]] = []
            if has_table(conn, "claims") and has_table(
                conn, "claim_revisions"
            ):
                sql = (
                    "SELECT c.claim_id, c.subject_id, c.predicate,"
                    "       r.revision, r.object_json, r.recorded_from,"
                    "       r.state"
                    " FROM claims c"
                    " JOIN claim_revisions r ON r.claim_id=c.claim_id"
                    "   AND r.revision = ("
                    "     SELECT MAX(revision) FROM claim_revisions"
                    "     WHERE claim_id=c.claim_id)"
                    " WHERE c.scope_id=?"
                    + (" AND c.subject_id=?" if subject_id else "")
                    + " ORDER BY r.recorded_from DESC, c.claim_id"
                    " LIMIT ?"
                )
                params: list[Any] = [scope_id]
                if subject_id:
                    params.append(subject_id)
                params.append(int(recent_limit))
                for row in conn.execute(sql, params).fetchall():
                    cid, subj, pred, rev, obj_json, rec_from, st = row
                    if st not in _CLAIM_VISIBLE:
                        continue
                    if not self._allowed(
                        conn, caller, Verb.READ.value, purpose,
                        (scope_id,), (("claim", cid, int(rev)),),
                    ):
                        continue
                    recent_pool.append(
                        {
                            "claim_id": cid,
                            "revision": int(rev),
                            "subject_id": subj,
                            "predicate": pred,
                            "object": _schema.loads(obj_json, obj_json),
                            "recorded_from": int(rec_from or 0),
                        }
                    )
            pack = {
                "format": "v4-context-pack/1",
                "scope_id": scope_id,
                "subject_id": subject_id,
                "purpose": purpose,
                "generated_us": now,
                "stable": [],
                "recent": [],
                "provenance": {"entries": [], "claims": []},
            }

            def _measure() -> int:
                return len(_canonical(pack).encode("utf-8"))

            budget = int(max_bytes)
            recent_floor = budget // 4
            stable_cap = budget - recent_floor
            dropped_stable: list[dict[str, Any]] = []

            def _pack_stable(item: dict[str, Any], cap: int) -> bool:
                pack["stable"].append(item)
                pack["provenance"]["entries"].append(
                    {"entry_id": item["entry_id"],
                     "revision": item["revision"]}
                )
                if _measure() <= cap:
                    return True
                pack["stable"].pop()
                pack["provenance"]["entries"].pop()
                return False

            def _pack_recent(item: dict[str, Any]) -> bool:
                pack["recent"].append(item)
                pack["provenance"]["claims"].append(
                    {"claim_id": item["claim_id"],
                     "revision": item["revision"]}
                )
                if _measure() <= budget:
                    return True
                pack["recent"].pop()
                pack["provenance"]["claims"].pop()
                return False

            for item in stable_pool:
                if not _pack_stable(item, stable_cap):
                    dropped_stable.append(item)
            for item in recent_pool:
                if not _pack_recent(item):
                    break  # ordered by recency — stop at first miss
            # Stable slack release: leftover items retry the full budget
            # once the recent section has taken its floor.
            for item in dropped_stable:
                if not _pack_stable(item, budget):
                    break  # stable list is sorted — first miss is final
            pack["bytes_used"] = _measure()
            pack["bytes_budget"] = budget
            pack["counts"] = {
                "stable": len(pack["stable"]),
                "recent": len(pack["recent"]),
                "stable_dropped": len(stable_pool) - len(pack["stable"]),
                "recent_dropped": len(recent_pool) - len(pack["recent"]),
            }
            pack["truncated"] = bool(
                pack["counts"]["stable_dropped"]
                or pack["counts"]["recent_dropped"]
            )
            return pack

    # ------------------------------------------------------------------
    # quality (V4-21.09) — adaptation, wrongness, correction burden;
    # never profile size.
    # ------------------------------------------------------------------

    def quality_report(
        self,
        caller: CallerV3,
        scope_id: str,
        *,
        purpose: str = "evaluate",
    ) -> dict:
        require_id(scope_id, "scope_id")
        with self._store.read() as conn:
            if not _schema.tables_present(conn):
                return {
                    "scope_id": scope_id,
                    "state": "unavailable",
                    "reason": "profile tables absent",
                }
            self._lease(conn, caller, Verb.READ.value, purpose, (scope_id,))
            heads = _schema.head_rows(
                conn, scope_id, states=None,
                limit=MAX_ENTRIES_PER_READ * 8,
            )
            by_kind: dict[str, int] = {}
            by_state: dict[str, int] = {}
            corrections = 0
            rework_revisions = 0
            for h in heads:
                by_kind[h["entry_kind"]] = by_kind.get(h["entry_kind"], 0) + 1
                by_state[h["state"]] = by_state.get(h["state"], 0) + 1
                rework_revisions += int(h["revision"]) - 1
                if h["supersedes_entry_id"]:
                    corrections += 1
            open_conflicts = len(
                _schema.conflict_rows(conn, scope_id, state="open")
            )
            live = sum(
                by_state.get(s, 0) for s in _ENTRY_VISIBLE_STATES
            )
            incorrect = by_state.get(EntryState.WITHHELD.value, 0) + by_state.get(
                EntryState.TOMBSTONED.value, 0
            )
            total = len(heads)
            return {
                "scope_id": scope_id,
                "state": "healthy",
                "entries_total": total,
                "entries_live": live,
                "by_kind": by_kind,
                "by_state": by_state,
                "open_conflicts": open_conflicts,
                # V4-21.09 measures — proxies honestly labelled:
                "beneficial_adaptations": live,
                "incorrect_assumptions": incorrect,
                "correction_burden": round(
                    rework_revisions / max(1, total), 4
                ),
                "explicit_corrections": corrections,
                "note": (
                    "profile size is reported for context only — it is "
                    "not a quality signal (V4-21.09)"
                ),
            }
