"""Seeded corpus for the v4.5 measured-ablation harnesses (SPEC_V4_5 §03
I1 / §05 I3; D01/D02, D05/D06).

Two slices, both disposable real ``Store.create`` databases — every arm
runs the production ``recall_v3`` path (lanes → union → fusion → groups
→ abstention → packs). No lane, gate, or packer is shimmed; the seeder
writes the same tables the real ingest would (the retrieval tests use
the identical pattern, including real profile-keyed content HMACs so
read-time digest verification passes).

**Stale-state slice (I1/D02)** — per topic: a state-asserting singleton
claim that outranks everything (the "answer"), an open two-member
conflict group whose members overlap the query weakly (the known-held
contradiction), plus a filler claim. Under an item-bound budget the
assertion ships and the atomic conflict group crowds out — the textbook
false-current failure. The manifest arm admits the conflict group first
under the SAME bound, or labels its omission.

**Long-history slice (I3/D05)** — per topic: an identifier-carrying fix
claim (``src/v45_mod/i.py`` is a hard identifier), a condition-bearing
constraint claim, a safety note, and several depth claims that share
query terms. The flat arm ships everything at L2; the sufficiency arm
ships coverage/mandatory groups at L2 and depth groups as bound L0
expansion stubs. Tokens, identifier coverage, condition coverage, and
utility are measured — nothing is asserted from fixture intent, only
from the returned packs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Optional

from verbatim.governance import (
    create_grant,
    register_principal,
    seed_purposes,
)
from verbatim.storage.store import Store

# Pinned profile HMAC key — identical role to the retrieval tests' key:
# seeded digests must match store.hmac() output for read verification.
HMAC_KEY = b"v45-hmac-key-0123456789abcdef012"


def _h(data: bytes) -> bytes:
    return hmac.new(HMAC_KEY, data, hashlib.sha256).digest()


def make_store(directory: str, name: str = "v45.db") -> Store:
    """A disposable real Store with the pinned profile key."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name + ".key"), "wb") as fh:
        fh.write(HMAC_KEY)
    return Store.create(os.path.join(directory, name))


def seed_scope(conn: Any, scope_id: str, principal: str = "p1",
               conv: str = "c1") -> None:
    conn.execute(
        "INSERT INTO scopes(scope_id,profile_id,principal_id,"
        "workspace_id,conversation_id,visibility,acl_revision)"
        " VALUES(?,?,?,?,?,'conversation',0)",
        (scope_id, "prof", principal, "ws", conv),
    )


def seed_auth(conn: Any, scope_id: str, pid: str = "human:alice",
              purposes=("recall",), verbs=("read", "quote")) -> str:
    """Principal + grant; returns the grant_id for revocation probes."""
    seed_purposes(conn)
    register_principal(conn, kind="human", principal_id=pid)
    return create_grant(
        conn, scope_id=scope_id, principal_id=pid, verbs=set(verbs),
        issuer_id=pid, purposes=list(purposes),
    )


def add_source(conn: Any, source_id: str, scope_id: str, payload: bytes,
               speaker: str = "u1", provenance: str = "direct_user") -> None:
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, speaker),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,"
        "metadata_json) VALUES(?,1,?,?,1,1,'UTC',?,'{}')",
        (source_id, payload, _h(payload), provenance),
    )


def add_span(conn: Any, span_id: str, source_id: str, start: int, end: int,
             rev: int = 1) -> None:
    payload = bytes(
        conn.execute(
            "SELECT payload FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (source_id, rev),
        ).fetchone()[0]
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,"
        "end_byte,excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (span_id, source_id, rev, start, end, _h(payload[start:end])),
    )


def add_fts(conn: Any, claim_id: str, rev: int, scope_id: str, text: str,
            gen: int) -> None:
    cur = conn.execute(
        "INSERT INTO fts_rows(claim_id,claim_revision,scope_id,"
        "projection_generation) VALUES(?,?,?,?)",
        (claim_id, rev, scope_id, gen),
    )
    conn.execute(
        "INSERT INTO facts_fts(fts_row_id,text) VALUES(?,?)",
        (cur.lastrowid, text),
    )


def seed_claim(conn: Any, claim_id: str, scope_id: str, source_id: str,
               span_id: str, text: str, gen: int, *,
               state: str = "active", recorded_from: int = 1,
               recorded_until: Optional[int] = None, rev: int = 1,
               condition: Optional[str] = None,
               freshness: Optional[str] = None) -> None:
    payload = text.encode("utf-8")
    add_source(conn, source_id, scope_id, payload)
    add_span(conn, span_id, source_id, 0, len(payload))
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,?,1)",
        (claim_id, scope_id, recorded_from),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,"
        "condition_json,recorded_from,recorded_until,perspective_id,"
        "freshness) VALUES(?,?,?,?,?,?,NULL,?)",
        (claim_id, rev, state, condition, recorded_from, recorded_until,
         freshness),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,?,?,'primary',NULL)",
        (claim_id, rev, span_id),
    )
    add_fts(conn, claim_id, rev, scope_id, text, gen)


def add_edge(conn: Any, edge_id: str, scope_id: str, source: str,
             target: str, edge_type: str) -> None:
    conn.execute(
        "INSERT INTO edges(edge_id,scope_id,source_kind,source_id,"
        "target_kind,target_id,edge_type,created_event)"
        " VALUES(?,?,'claim',?,'claim',?,?,1)",
        (edge_id, scope_id, source, target, edge_type),
    )


def open_conflict(conn: Any, group_id: str, scope_id: str,
                  member_ids: list) -> None:
    conn.execute(
        "INSERT INTO conflict_groups(group_id,scope_id,status)"
        " VALUES(?,?,'open')",
        (group_id, scope_id),
    )
    for cid in member_ids:
        conn.execute(
            "INSERT INTO conflict_members(group_id,claim_id)"
            " VALUES(?,?)",
            (group_id, cid),
        )


# ---------------------------------------------------------------------------
# I1 — the stale-state slice (D02)
# ---------------------------------------------------------------------------

#: Per-topic record: the assertion claim id, the two conflict members,
#: the filler, and the query. ``assertion_text`` is the state the answer
#: asserts when the singleton ships alone.
def seed_stale_slice(store: Store, scope_id: str = "sA",
                     topics: int = 8, caller: str = "human:alice") -> dict:
    """Per topic i (0-based):

    - ``clSum{i}`` — "service alpha{i} mode status is ENABLED steady
      state confirmed" — every query term, minimal size → outranks.
    - ``clOld{i}``/``clNew{i}`` — open conflict group ``cg{i}``: "alpha{i}
      mode was set to FAST" vs "alpha{i} mode is set to SAFE since the
      incident". Weak term overlap; the pair ships atomically or not at
      all (V3-30.06) — the unit a top-k budget can silently drop.
    - ``clFill{i}`` — an unrelated claim sharing "service"/"status"
      terms; occupies leftover budget slots honestly.
    """
    gen = store.projection_generation()
    records: list = []
    with store.tx() as conn:
        seed_scope(conn, scope_id)
        seed_auth(conn, scope_id, pid=caller)
        for i in range(topics):
            tag = f"alpha{i}"
            ident = f"svc/{tag}.cfg"   # hard identifier (path token)
            sid = f"clSum{i}"
            seed_claim(
                conn, sid, scope_id, f"srcSum{i}", f"spSum{i}",
                f"service {tag} mode status is ENABLED and the steady"
                f" state is confirmed in {ident}",
                gen,
            )
            old_id, new_id = f"clOld{i}", f"clNew{i}"
            seed_claim(
                conn, old_id, scope_id, f"srcOld{i}", f"spOld{i}",
                f"{tag} mode was set to FAST for the launch window"
                f" in {ident}",
                gen,
            )
            seed_claim(
                conn, new_id, scope_id, f"srcNew{i}", f"spNew{i}",
                f"{tag} mode is set to SAFE since the incident review"
                f" of {ident}",
                gen,
            )
            open_conflict(conn, f"cg{i}", scope_id, [old_id, new_id])
            fill_id = f"clFill{i}"
            seed_claim(
                conn, fill_id, scope_id, f"srcFill{i}", f"spFill{i}",
                f"service status bulletin {i}: routine operational"
                f" notes for the week",
                gen,
            )
            records.append({
                "topic": tag,
                "identifier": ident,
                "query": f"service {tag} mode status {ident}",
                "assertion_claim": sid,
                "assertion_marker": "ENABLED",
                "conflict_members": [old_id, new_id],
                "filler": fill_id,
            })
    return {"scope_id": scope_id, "caller": caller, "topics": records}


# ---------------------------------------------------------------------------
# I3 — the long-history slice (D05)
# ---------------------------------------------------------------------------

def seed_history_slice(store: Store, scope_id: str = "sA",
                       topics: int = 6, caller: str = "human:alice",
                       depth: int = 5) -> dict:
    """Per topic i (0-based):

    - ``clFix{i}`` — carries the hard identifier ``src/v45_mod/{i}.py``
      and the checkable fix text (``rebind CONFIG_{i}``).
    - ``clCond{i}`` — a condition-bearing constraint claim
      (``condition_json`` env gate) sharing the query's "loader" term —
      under ``sufficiency`` its ``cond_keys`` make it mandatory.
    - ``clSafe{i}`` — a second mandatory unit: conflicts_with-linked to
      the fix claim (the safety caveat that must not be dropped for
      tokens, V45-05.03).
    - ``depth`` × ``clDep{i}_{j}`` — long depth claims sharing the query
      terms ("loader", "crash", the module tag) — the material that
      becomes L0 expansion stubs under the sufficiency arm.
    """
    gen = store.projection_generation()
    records: list = []
    with store.tx() as conn:
        seed_scope(conn, scope_id)
        seed_auth(conn, scope_id, pid=caller)
        for i in range(topics):
            ident = f"src/v45_mod/{i}.py"
            fix_id = f"clFix{i}"
            seed_claim(
                conn, fix_id, scope_id, f"srcFix{i}", f"spFix{i}",
                f"the loader crash fix for {ident} is to rebind"
                f" CONFIG_{i} before the module import runs",
                gen,
            )
            cond_id = f"clCond{i}"
            seed_claim(
                conn, cond_id, scope_id, f"srcCond{i}", f"spCond{i}",
                f"loader rollout constraint for {ident}: apply only"
                f" when env is prod-eu and config_version >= 7",
                gen,
                condition=json.dumps({
                    "all": [
                        {"eq": ["env", "prod-eu"]},
                        {"eq": ["config_version", 7]},
                    ]
                }),
            )
            safe_id = f"clSafe{i}"
            seed_claim(
                conn, safe_id, scope_id, f"srcSafe{i}", f"spSafe{i}",
                f"loader safety caveat for {ident}: the rebind drops"
                f" in-flight sessions — drain the worker first",
                gen,
            )
            add_edge(conn, f"eSafe{i}", scope_id, fix_id, safe_id,
                     "conflicts_with")
            dep_ids = []
            for j in range(depth):
                did = f"clDep{i}_{j}"
                dep_ids.append(did)
                seed_claim(
                    conn, did, scope_id, f"srcDep{i}_{j}",
                    f"spDep{i}_{j}",
                    f"loader crash investigation note {j} for module"
                    f" work: stack walk of the failing thread, heap"
                    f" profile diff between the last good deploy and"
                    f" the regression, allocator arena tuning knobs"
                    f" tried and rejected, cache flush ordering under"
                    f" the old eviction policy, retry budget math for"
                    f" the client backoff, scheduler lease timings,"
                    f" and the full annotated investigation history"
                    f" of this component's incidents",
                    gen,
                )
            records.append({
                "topic": f"mod{i}",
                "identifier": ident,
                "query": f"loader crash fix {ident}",
                "fix_claim": fix_id,
                "fix_marker": f"rebind CONFIG_{i}",
                "condition_claim": cond_id,
                "condition_marker": "prod-eu",
                "safety_claim": safe_id,
                "safety_marker": "drain the worker",
                "depth_claims": dep_ids,
            })
    return {"scope_id": scope_id, "caller": caller, "topics": records}


__all__ = [
    "HMAC_KEY",
    "make_store",
    "seed_scope",
    "seed_auth",
    "add_source",
    "add_span",
    "add_fts",
    "seed_claim",
    "add_edge",
    "open_conflict",
    "seed_stale_slice",
    "seed_history_slice",
]
