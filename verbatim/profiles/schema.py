"""Profile-plane storage (SPEC_V4 §21).

Module-owned tables, created idempotently inside the caller's write
transaction before any row is written (register-before-write, V4-19.09).
They are deliberately *not* added to ``storage/schema_v4.py`` — that file
is owned by the v4 schema worker; when it lands registration there, this
DDL stays byte-identical so ``CREATE TABLE IF NOT EXISTS`` converges.

Every profile entry also registers in the v4 ``objects`` /
``object_revisions`` / ``dependency_edges`` registry (kind ``profile``)
so the kernel's resolution, disposition gating, and parent-ancestry
quarantine cascade apply — profiles ride the same rails as every other
derived object instead of inventing a parallel authority.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Optional

from ..storage.repos import has_table

DDL = """
-- §21 configured topics: multiplicity, sensitivity, expiry, preferred
-- evidence classes, conflict handling (V4-21.01).
CREATE TABLE IF NOT EXISTS profile_topics (
    scope_id TEXT NOT NULL,
    topic_key TEXT NOT NULL,
    value_kind TEXT NOT NULL DEFAULT 'freeform',
    multiplicity TEXT NOT NULL DEFAULT 'single'
        CHECK (multiplicity IN ('single','set')),
    sensitivity TEXT NOT NULL DEFAULT 'normal'
        CHECK (sensitivity IN ('normal','sensitive')),
    preferred_evidence_json TEXT NOT NULL DEFAULT '[]',
    required_purposes_json TEXT NOT NULL DEFAULT '[]',
    inference TEXT NOT NULL DEFAULT 'off'
        CHECK (inference IN ('off','on')),
    inference_purposes_json TEXT NOT NULL DEFAULT '[]',
    conflict_policy TEXT NOT NULL DEFAULT 'keep_both'
        CHECK (conflict_policy IN ('keep_both','latest_wins','explicit_only')),
    expiry_s INTEGER,
    max_entries INTEGER NOT NULL DEFAULT 8,
    min_support INTEGER NOT NULL DEFAULT 1,
    match_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'active'
        CHECK (state IN ('active','disabled','deleted')),
    created_us INTEGER NOT NULL,
    updated_us INTEGER NOT NULL,
    updated_by TEXT NOT NULL,
    PRIMARY KEY (scope_id, topic_key)
);

-- Per-subject owner policy (V4-21.04/07): inference consent is an
-- explicit opt-in row; absence of a row is *not* consent.
CREATE TABLE IF NOT EXISTS profile_policies (
    scope_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    inference TEXT NOT NULL DEFAULT 'off'
        CHECK (inference IN ('off','on')),
    updated_us INTEGER NOT NULL,
    updated_by TEXT NOT NULL,
    PRIMARY KEY (scope_id, subject_id)
);

-- Append-only entry revisions. Identity = (entry_id); head = max
-- revision. ``supersedes_entry_id``/``prev_revision`` make every update
-- attributable to the revision it replaced (V4-21.05).
CREATE TABLE IF NOT EXISTS profile_entries (
    entry_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    topic_key TEXT NOT NULL,
    entry_kind TEXT NOT NULL
        CHECK (entry_kind IN
            ('explicit','observed','inferred','task_local','sensitive')),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    state TEXT NOT NULL
        CHECK (state IN
            ('active','conflicted','superseded','withheld',
             'tombstoned','expired')),
    value_json TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    basis TEXT,
    conflict_group TEXT,
    audience_json TEXT NOT NULL DEFAULT '[]',
    effective_us INTEGER NOT NULL,
    expires_us INTEGER,
    recorded_us INTEGER NOT NULL,
    actor_id TEXT NOT NULL,
    purpose TEXT,
    operation_id TEXT,
    producer_id TEXT,
    supersedes_entry_id TEXT,
    prev_revision INTEGER,
    changed_fields_json TEXT NOT NULL DEFAULT '[]',
    digest TEXT NOT NULL,
    PRIMARY KEY (entry_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_profile_entries_head
    ON profile_entries(scope_id, profile_id, topic_key, state);
CREATE INDEX IF NOT EXISTS idx_profile_entries_subject
    ON profile_entries(scope_id, subject_id);

-- Derivation-parent support per revision (V4-21.05). These rows are
-- provenance pointers only — disclosure always re-verifies through the
-- kernel, and refresh() scrubs pointers to erased evidence.
CREATE TABLE IF NOT EXISTS profile_entry_support (
    entry_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    support_kind TEXT NOT NULL,
    support_id TEXT NOT NULL,
    support_revision INTEGER NOT NULL,
    relation TEXT NOT NULL DEFAULT 'supports'
        CHECK (relation IN ('supports','contradicts')),
    claim_role TEXT,
    PRIMARY KEY (entry_id, revision, seq)
);
CREATE INDEX IF NOT EXISTS idx_profile_support_target
    ON profile_entry_support(support_kind, support_id);

-- Open contradiction groups (V4-21.05): both alternatives stay
-- inspectable; resolution is a new revision, never a silent overwrite.
CREATE TABLE IF NOT EXISTS profile_conflicts (
    conflict_group TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    topic_key TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    members_json TEXT NOT NULL DEFAULT '[]',
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open','resolved')),
    opened_us INTEGER NOT NULL,
    resolved_us INTEGER,
    resolution_entry_id TEXT,
    PRIMARY KEY (scope_id, conflict_group)
);

-- Bounded audit trail: topic changes, entry writes, compile runs,
-- refresh actions, perspective-pack declarations.
CREATE TABLE IF NOT EXISTS profile_events (
    seq INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    event TEXT NOT NULL,
    entry_id TEXT,
    actor_id TEXT NOT NULL,
    purpose TEXT,
    operation_id TEXT,
    us INTEGER NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (scope_id, seq)
);
"""

_TABLES = (
    "profile_topics",
    "profile_policies",
    "profile_entries",
    "profile_entry_support",
    "profile_conflicts",
    "profile_events",
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the profile tables if absent. Idempotent; runs inside the
    caller's transaction so registration precedes the first write.
    ``executescript`` would issue an implicit COMMIT — statements run
    individually so the caller's transaction boundary is preserved.
    Full-line ``--`` comments are stripped before splitting so a
    semicolon inside a comment cannot split a statement."""
    cleaned = "\n".join(
        line for line in DDL.splitlines()
        if not line.lstrip().startswith("--")
    )
    for stmt in cleaned.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        conn.execute(stmt + ";")


def tables_present(conn: sqlite3.Connection) -> bool:
    return all(has_table(conn, t) for t in _TABLES)


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    r = cur.fetchone()
    if r is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, r))


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def loads(text: Any, default: Any = None) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------
# topics
# ----------------------------------------------------------------------


def put_topic_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO profile_topics(scope_id,topic_key,value_kind,"
        "multiplicity,sensitivity,preferred_evidence_json,"
        "required_purposes_json,inference,inference_purposes_json,"
        "conflict_policy,expiry_s,max_entries,min_support,match_json,"
        "state,created_us,updated_us,updated_by) VALUES"
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(scope_id,topic_key) DO UPDATE SET"
        " value_kind=excluded.value_kind,"
        " multiplicity=excluded.multiplicity,"
        " sensitivity=excluded.sensitivity,"
        " preferred_evidence_json=excluded.preferred_evidence_json,"
        " required_purposes_json=excluded.required_purposes_json,"
        " inference=excluded.inference,"
        " inference_purposes_json=excluded.inference_purposes_json,"
        " conflict_policy=excluded.conflict_policy,"
        " expiry_s=excluded.expiry_s,"
        " max_entries=excluded.max_entries,"
        " min_support=excluded.min_support,"
        " match_json=excluded.match_json,"
        " state=excluded.state,"
        " updated_us=excluded.updated_us,"
        " updated_by=excluded.updated_by",
        (
            row["scope_id"], row["topic_key"], row["value_kind"],
            row["multiplicity"], row["sensitivity"],
            row["preferred_evidence_json"], row["required_purposes_json"],
            row["inference"], row["inference_purposes_json"],
            row["conflict_policy"], row["expiry_s"], row["max_entries"],
            row["min_support"], row["match_json"], row["state"],
            row["created_us"], row["updated_us"], row["updated_by"],
        ),
    )


def get_topic_row(
    conn: sqlite3.Connection, scope_id: str, topic_key: str
) -> Optional[dict[str, Any]]:
    return _row(
        conn.execute(
            "SELECT * FROM profile_topics WHERE scope_id=? AND topic_key=?",
            (scope_id, topic_key),
        )
    )


def topic_rows(
    conn: sqlite3.Connection, scope_id: str, *, state: Optional[str] = None
) -> list[dict[str, Any]]:
    if state is None:
        return _rows(
            conn.execute(
                "SELECT * FROM profile_topics WHERE scope_id=?"
                " ORDER BY topic_key",
                (scope_id,),
            )
        )
    return _rows(
        conn.execute(
            "SELECT * FROM profile_topics WHERE scope_id=? AND state=?"
            " ORDER BY topic_key",
            (scope_id, state),
        )
    )


# ----------------------------------------------------------------------
# policies
# ----------------------------------------------------------------------


def get_policy_row(
    conn: sqlite3.Connection, scope_id: str, subject_id: str
) -> Optional[dict[str, Any]]:
    return _row(
        conn.execute(
            "SELECT * FROM profile_policies WHERE scope_id=? AND subject_id=?",
            (scope_id, subject_id),
        )
    )


def put_policy_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO profile_policies(scope_id,subject_id,inference,"
        "updated_us,updated_by) VALUES(?,?,?,?,?)"
        " ON CONFLICT(scope_id,subject_id) DO UPDATE SET"
        " inference=excluded.inference,"
        " updated_us=excluded.updated_us,"
        " updated_by=excluded.updated_by",
        (
            row["scope_id"], row["subject_id"], row["inference"],
            row["updated_us"], row["updated_by"],
        ),
    )


# ----------------------------------------------------------------------
# entries
# ----------------------------------------------------------------------


def insert_entry_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO profile_entries(entry_id,scope_id,profile_id,"
        "subject_id,topic_key,entry_kind,revision,state,value_json,"
        "confidence,basis,conflict_group,audience_json,effective_us,"
        "expires_us,recorded_us,actor_id,purpose,operation_id,"
        "producer_id,supersedes_entry_id,prev_revision,"
        "changed_fields_json,digest) VALUES"
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            row["entry_id"], row["scope_id"], row["profile_id"],
            row["subject_id"], row["topic_key"], row["entry_kind"],
            row["revision"], row["state"], row["value_json"],
            row["confidence"], row["basis"], row["conflict_group"],
            row["audience_json"], row["effective_us"], row["expires_us"],
            row["recorded_us"], row["actor_id"], row["purpose"],
            row["operation_id"], row["producer_id"],
            row["supersedes_entry_id"], row["prev_revision"],
            row["changed_fields_json"], row["digest"],
        ),
    )


def entry_head_row(
    conn: sqlite3.Connection, entry_id: str
) -> Optional[dict[str, Any]]:
    return _row(
        conn.execute(
            "SELECT * FROM profile_entries WHERE entry_id=?"
            " ORDER BY revision DESC LIMIT 1",
            (entry_id,),
        )
    )


def entry_revision_row(
    conn: sqlite3.Connection, entry_id: str, revision: int
) -> Optional[dict[str, Any]]:
    return _row(
        conn.execute(
            "SELECT * FROM profile_entries WHERE entry_id=? AND revision=?",
            (entry_id, revision),
        )
    )


def entry_revisions(
    conn: sqlite3.Connection, entry_id: str
) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            "SELECT * FROM profile_entries WHERE entry_id=?"
            " ORDER BY revision",
            (entry_id,),
        )
    )


def head_rows(
    conn: sqlite3.Connection,
    scope_id: str,
    *,
    profile_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    topic_key: Optional[str] = None,
    states: Optional[Iterable[str]] = None,
    limit: int,
) -> list[dict[str, Any]]:
    """Latest revision per entry_id under the filters (scope pinned)."""
    sql = (
        "SELECT e.* FROM profile_entries e"
        " JOIN (SELECT entry_id, MAX(revision) AS mr"
        "       FROM profile_entries WHERE scope_id=? GROUP BY entry_id) h"
        "   ON h.entry_id=e.entry_id AND h.mr=e.revision"
        " WHERE e.scope_id=?"
    )
    params: list[Any] = [scope_id, scope_id]
    if profile_id is not None:
        sql += " AND e.profile_id=?"
        params.append(profile_id)
    if subject_id is not None:
        sql += " AND e.subject_id=?"
        params.append(subject_id)
    if topic_key is not None:
        sql += " AND e.topic_key=?"
        params.append(topic_key)
    if states is not None:
        st = tuple(states)
        sql += f" AND e.state IN ({','.join('?' for _ in st)})"
        params.extend(st)
    sql += " ORDER BY e.topic_key, e.entry_id LIMIT ?"
    params.append(int(limit))
    return _rows(conn.execute(sql, params))


def insert_support_rows(
    conn: sqlite3.Connection,
    entry_id: str,
    revision: int,
    supports: Iterable[dict[str, Any]],
) -> None:
    for seq, s in enumerate(supports):
        conn.execute(
            "INSERT INTO profile_entry_support(entry_id,revision,seq,"
            "support_kind,support_id,support_revision,relation,claim_role)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (
                entry_id, revision, seq,
                s["kind"], s["id"], int(s["revision"]),
                s.get("relation", "supports"), s.get("claim_role"),
            ),
        )


def support_rows(
    conn: sqlite3.Connection, entry_id: str, revision: int
) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            "SELECT * FROM profile_entry_support"
            " WHERE entry_id=? AND revision=? ORDER BY seq",
            (entry_id, revision),
        )
    )


def delete_support_row(
    conn: sqlite3.Connection, entry_id: str, revision: int, seq: int
) -> None:
    conn.execute(
        "DELETE FROM profile_entry_support"
        " WHERE entry_id=? AND revision=? AND seq=?",
        (entry_id, revision, seq),
    )


def entries_with_support_target(
    conn: sqlite3.Connection, scope_id: str, kind: str, oid: str
) -> list[dict[str, Any]]:
    """Entry heads whose support rows name the object — used by refresh."""
    return _rows(
        conn.execute(
            "SELECT DISTINCT s.entry_id FROM profile_entry_support s"
            " JOIN profile_entries e ON e.entry_id=s.entry_id"
            " WHERE e.scope_id=? AND s.support_kind=? AND s.support_id=?",
            (scope_id, kind, oid),
        )
    )


# ----------------------------------------------------------------------
# conflicts + events
# ----------------------------------------------------------------------


def get_conflict_row(
    conn: sqlite3.Connection, scope_id: str, group: str
) -> Optional[dict[str, Any]]:
    return _row(
        conn.execute(
            "SELECT * FROM profile_conflicts"
            " WHERE scope_id=? AND conflict_group=?",
            (scope_id, group),
        )
    )


def put_conflict_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO profile_conflicts(conflict_group,scope_id,profile_id,"
        "topic_key,subject_id,members_json,state,opened_us,resolved_us,"
        "resolution_entry_id) VALUES(?,?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(scope_id,conflict_group) DO UPDATE SET"
        " members_json=excluded.members_json,"
        " state=excluded.state,"
        " resolved_us=excluded.resolved_us,"
        " resolution_entry_id=excluded.resolution_entry_id",
        (
            row["conflict_group"], row["scope_id"], row["profile_id"],
            row["topic_key"], row["subject_id"], row["members_json"],
            row["state"], row["opened_us"], row["resolved_us"],
            row["resolution_entry_id"],
        ),
    )


def conflict_rows(
    conn: sqlite3.Connection, scope_id: str, *, state: Optional[str] = None
) -> list[dict[str, Any]]:
    if state is None:
        return _rows(
            conn.execute(
                "SELECT * FROM profile_conflicts WHERE scope_id=?"
                " ORDER BY conflict_group",
                (scope_id,),
            )
        )
    return _rows(
        conn.execute(
            "SELECT * FROM profile_conflicts WHERE scope_id=? AND state=?"
            " ORDER BY conflict_group",
            (scope_id, state),
        )
    )


def next_event_seq(conn: sqlite3.Connection, scope_id: str) -> int:
    r = conn.execute(
        "SELECT COALESCE(MAX(seq),0)+1 FROM profile_events WHERE scope_id=?",
        (scope_id,),
    ).fetchone()
    return int(r[0])


def insert_event(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO profile_events(seq,scope_id,event,entry_id,actor_id,"
        "purpose,operation_id,us,detail_json) VALUES(?,?,?,?,?,?,?,?,?)",
        (
            row["seq"], row["scope_id"], row["event"], row["entry_id"],
            row["actor_id"], row["purpose"], row["operation_id"],
            row["us"], row["detail_json"],
        ),
    )
