"""Honest erasure accounting for vault entries (SPEC_V3 §35.05, §36, §37.08).

``erase_entry`` performs *live-store logical erasure*: it sets
``erased_event`` and purges the searchable ``vault_refs`` rows so the
placeholder no longer resolves. It deliberately does NOT report
cryptographic erasure — deleting an entry and its current wrapped key
leaves backup/WAL copies of the wrapped key decryptable while the scope
wrap key exists (§35.05).

``report`` therefore defaults to ``cryptographic_erasure="unproven"`` and
returns ``"proven"`` only when the caller attests
``verified_key_destruction=True`` — meaning every copy of the covering
wrap-key version, including backups, was verifiably destroyed or an
equivalent external per-entry revocation mechanism applied. The engine
cannot verify backup state itself, so it never upgrades the claim on its
own (§35.05, §37.08).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable, Optional

from ..core.types import new_id, require_id
from ..storage import repos_v3


def erase_entry(
    conn: sqlite3.Connection,
    entry_id: str,
    *,
    erased_event: int = 0,
    hmac_fn: Optional[Callable[[bytes], bytes]] = None,
    purge_id: Optional[str] = None,
    erasure_epoch: int = 0,
) -> bool:
    """Logically erase one vault entry; returns True when newly erased.

    Idempotent: an already-erased or absent entry returns False without
    error so purge replays converge. Erasure tombstones the entry row
    (never deletes it — outstanding handles and refs fail closed against
    ``erased_event``), deletes its searchable ``vault_refs`` rows, and —
    when ``hmac_fn`` is supplied — writes an opaque erasure-ledger
    tombstone so a restored backup cannot resurrect it (§36.03).
    """
    require_id(entry_id, "entry_id")
    row = repos_v3.get(conn, "vault_entries", {"entry_id": entry_id})
    if row is None or row.get("erased_event") is not None:
        return False
    marked = repos_v3.update(
        conn,
        "vault_entries",
        {"erased_event": int(erased_event)},
        {"entry_id": entry_id, "erased_event": None},
    )
    if marked != 1:
        return False
    repos_v3.delete(conn, "vault_refs", {"entry_id": entry_id})
    # Outstanding unconsumed handles stay on the table but can never
    # resolve: resolve_handle → vault.open fences on erased_event.
    if hmac_fn is not None:
        digest = hmac_fn(f"vault_entry:{entry_id}".encode("utf-8"))
        conn.execute(
            "INSERT INTO erasure_ledger"
            " (erasure_id, scope_id, object_kind, object_digest, purge_id,"
            "  erased_event, erasure_epoch)"
            " VALUES (?, ?, 'vault_entry', ?, ?, ?, ?)",
            (
                new_id(),
                row["scope_id"],
                digest,
                purge_id,
                int(erased_event),
                int(erasure_epoch),
            ),
        )
    return True


def report(
    conn: sqlite3.Connection,
    entry_id: str,
    *,
    verified_key_destruction: bool = False,
) -> dict[str, Any]:
    """Erasure posture for one entry — honest by construction (§35.05).

    ``cryptographic_erasure`` is ``"proven"`` only under a caller's
    verified-key-destruction attestation; the engine's default is
    ``"unproven"`` because backups may retain decryptable wrapped-key
    copies while the covering wrap key exists.
    """
    require_id(entry_id, "entry_id")
    row = repos_v3.get(conn, "vault_entries", {"entry_id": entry_id})
    if row is None:
        return {
            "entry_id": entry_id,
            "present": False,
            "logical_erasure": False,
            "refs_purged": True,
            "cryptographic_erasure": (
                "proven" if verified_key_destruction else "unproven"
            ),
            "basis": (
                "entry absent; wrap-key destruction attested by caller"
                if verified_key_destruction
                else "entry absent; backup copies of the wrapped key may "
                "remain decryptable while the wrap key exists"
            ),
        }
    logical = row.get("erased_event") is not None
    refs = repos_v3.query(conn, "vault_refs", {"entry_id": entry_id})
    if verified_key_destruction:
        crypto = "proven"
        basis = (
            "caller attests every copy of wrap-key version "
            f"{row['wrap_key_version']} — including backups — was "
            "verifiably destroyed or externally revoked"
        )
    else:
        crypto = "unproven"
        basis = (
            "logical erasure only: backup/WAL copies of the wrapped key "
            "remain decryptable while wrap-key version "
            f"{row['wrap_key_version']} exists"
        )
    return {
        "entry_id": entry_id,
        "present": True,
        "scope_id": row["scope_id"],
        "wrap_key_version": int(row["wrap_key_version"]),
        "logical_erasure": logical,
        "refs_purged": not refs,
        "cryptographic_erasure": crypto,
        "basis": basis,
    }
