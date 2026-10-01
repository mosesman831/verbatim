"""Operator CLI: `python -m verbatim` and `hermes verbatim`.

Thin frontend over Engine + repos — no alternative business logic (SPEC §36).
Exit codes: 0 ok, 2 usage/config, 3 unavailable dependency, 4 authorization,
5 retryable operational, 6 integrity failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

from .config import VerbatimConfig, load_config_file
from .core.time import now_us, parse_rfc3339, rfc3339
from .core.identity import scope_key
from .core.types import (
    EffectProposal,
    ErrorCode,
    Mode,
    Provenance,
    RecallMode,
    RecallRequest,
    Scope,
    SourceEnvelope,
    SourceKind,
    TransitionCommand,
    VerbatimError,
    Visibility,
)
from .host import LocalHost

_EXIT = {
    ErrorCode.CONFIG_INVALID: 2,
    ErrorCode.VALIDATION: 2,
    ErrorCode.ENCODER_UNAVAILABLE: 3,
    ErrorCode.SCHEMA_UNSUPPORTED: 3,
    ErrorCode.NOT_FOUND_OR_FORBIDDEN: 4,
    ErrorCode.EGRESS_DISABLED: 4,
    ErrorCode.CAPTURE_DISABLED: 4,
    ErrorCode.STORE_BUSY: 5,
    ErrorCode.BACKPRESSURE: 5,
    ErrorCode.REMOTE_BUSY: 5,
    ErrorCode.STORE_CORRUPT: 6,
    ErrorCode.STORE_WRITE_FAILED: 6,
}


def _host(args: argparse.Namespace) -> LocalHost:
    return LocalHost(
        profile_id=args.profile,
        principal_id=args.principal or "local-owner",
        conversation_id=args.conversation or "cli",
        workspace_id=args.workspace,
        allow_env_secrets=False,
    )


def _scope(args: argparse.Namespace, eng) -> Scope:
    """The partition the command addresses: host default narrowed by flags.

    ``--visibility`` relocates the addressed partition (an operator may own
    evidence under several visibilities); it never widens who may see it.
    """
    base = eng.host.default_scope()
    return Scope(
        profile_id=base.profile_id,
        principal_id=base.principal_id,
        workspace_id=base.workspace_id,
        conversation_id=base.conversation_id,
        visibility=Visibility(args.visibility) if args.visibility else base.visibility,
    )


def _cfg(args: argparse.Namespace) -> VerbatimConfig:
    if args.config:
        return load_config_file(args.config)
    return VerbatimConfig()


def _engine(args: argparse.Namespace):
    from .api import open_store

    cfg = _cfg(args)
    data_dir = args.data_dir or os.path.join(os.getcwd(), cfg.data_dir)
    return open_store(data_dir, cfg, _host(args), create=True)


def _print(result: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    elif isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def cmd_status(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        _print(eng.status(), args.json)
        return 0
    finally:
        eng.close()


def cmd_doctor(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        report = eng.store.check_integrity()
        report["mode"] = eng.cfg.mode.value
        _print(report, args.json)
        return 0
    finally:
        eng.close()


def cmd_ingest(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        with open(args.file, "rb") as fh:
            payload = fh.read()
        env = SourceEnvelope(
            origin="cli:ingest",
            source_kind=SourceKind(args.kind),
            scope=_scope(args, eng),
            speaker_id=_scope(args, eng).principal_id or "local-owner",
            payload=payload,
            event_us=now_us(),
            captured_us=now_us(),
            provenance=Provenance(args.provenance),
        )
        receipt = eng.ingest(env)
        drained = eng.run_pending()
        _print(
            {
                "accepted": list(receipt.accepted),
                "duplicate": receipt.duplicate,
                "jobs": list(receipt.job_ids),
                "jobs_drained": drained,
            },
            args.json,
        )
        return 0
    finally:
        eng.close()


def cmd_search(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        req = RecallRequest(
            query=args.query,
            scope=_scope(args, eng),
            mode=RecallMode(args.mode),
            limit=args.limit,
            valid_at_us=parse_rfc3339(args.valid_at) if args.valid_at else None,
            known_at_seq=args.known_at,
            max_bytes=24000,
        )
        res = eng.recall(req)
        if args.json:
            _print(
                {
                    "items": [
                        {
                            "claim_id": i.claim_id, "text": i.text,
                            "lifecycle": i.lifecycle.value, "valid": i.valid_label,
                            "historical": i.historical, "disputed": i.disputed,
                            "reasons": list(i.reasons),
                        }
                        for i in res.items
                    ],
                    "omitted": res.omitted, "warnings": list(res.warnings),
                },
                True,
            )
        else:
            for i in res.items:
                tags = [i.lifecycle.value]
                if i.historical:
                    tags.append("hist")
                if i.disputed:
                    tags.append("DISPUTED")
                print(f"[{','.join(tags)}] {i.text}  (claim {i.claim_id[:12]})")
            if not res.items:
                print("no evidence", file=sys.stderr)
            for w in res.warnings:
                print(f"warning: {w}", file=sys.stderr)
        return 0
    finally:
        eng.close()


def cmd_inspect(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        _print(eng.inspect(args.claim_id, _scope(args, eng)), args.json)
        return 0
    finally:
        eng.close()


def cmd_explain(args: argparse.Namespace) -> int:
    """Why a claim reads the way it does: lineage + decisions + relations."""
    eng = _engine(args)
    try:
        scope = _scope(args, eng)
        detail = eng.inspect(args.claim_id, scope)
        # Resolve any decision ids referenced by the claim's live edges —
        # decision payloads carry the recorded rationale (SPEC §31).
        from .storage.repos import DecisionsRepo

        decisions = []
        seen = set()
        for edge in detail.get("edges", ()):
            did = edge.get("decision")
            if did and did not in seen:
                seen.add(did)
                row = DecisionsRepo(eng.store).get(did)
                if row is not None:
                    decisions.append(row)
        detail["decisions"] = decisions
        _print(detail, args.json)
        return 0
    finally:
        eng.close()


def cmd_spans(args: argparse.Namespace) -> int:
    """List the exact evidence spans behind a claim."""
    eng = _engine(args)
    try:
        detail = eng.inspect(args.claim_id, _scope(args, eng))
        _print(
            {"claim_id": detail["claim_id"], "evidence": detail["evidence"]},
            args.json,
        )
        return 0
    finally:
        eng.close()


def cmd_remember(args: argparse.Namespace) -> int:
    """Grounded remember: quote a byte range of an accepted source (SPEC §35)."""
    eng = _engine(args)
    try:
        claim_id = eng.remember(
            args.source_id,
            args.start_byte,
            args.end_byte,
            _scope(args, eng),
            predicate_suggestion=args.predicate,
            revision=args.revision,
        )
        _print({"claim_id": claim_id}, args.json)
        return 0
    finally:
        eng.close()


def _proposal_for_review(
    review: dict[str, Any], operation_id: str, actor_id: str
) -> EffectProposal:
    """Translate a stored review row into the unified EffectProposal.

    Two payload shapes exist today (v1 gap — incompatible producers):
    admission reviews carry ``{"effect": "admit", "claim_id": …}`` while
    supersession reviews carry ``{"effect": "supersede", "predecessor_id": …,
    "successor_id": …}``. Both map onto the same proposal schema
    (V2-20.01); ``params.review_id`` lets the engine resolve the review in
    the same transaction as the effect (V2-20.04).
    """
    effect = review.get("proposed_effect") or {}
    expected = review.get("expected_versions") or {}
    kind = effect.get("effect")
    params: dict[str, Any] = {"review_id": review["review_id"]}
    successor = None
    if kind == "supersede":
        target = effect.get("predecessor_id") or effect.get("claim_id")
        successor = effect.get("successor_id") or effect.get("successor_claim_id")
        if not target or not successor:
            raise VerbatimError(
                ErrorCode.VALIDATION, "supersede review lacks predecessor/successor"
            )
        targets = ((str(target), int(expected.get(str(target), 1))),)
        pinned = expected.get(str(successor))
        if pinned is not None:
            params["successor_expected_revision"] = int(pinned)
        if isinstance(effect.get("interval"), dict):
            # The recorded applicability cut rides inside the proposal so
            # interval truncation replays deterministically (V2-19.11).
            params["interval"] = effect["interval"]
    else:
        target = effect.get("claim_id")
        if not target:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"review effect {kind!r} has no claim target"
            )
        if kind == "dispute" and effect.get("conflict_with_claim_id"):
            params["conflict_with_claim_id"] = str(
                effect["conflict_with_claim_id"]
            )
        targets = ((str(target), int(expected.get(str(target), 1))),)
    return EffectProposal(
        version=1,
        operation_id=operation_id,
        effect=str(kind),
        targets=targets,
        actor_id=actor_id,
        reason=str(effect.get("reason") or "operator review"),
        successor_claim_id=str(successor) if successor else None,
        params=params,
    )


def cmd_reviews(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        from .storage.repos import ReviewsRepo

        scope = _scope(args, eng)
        repo = ReviewsRepo(eng.store)
        if args.review_cmd == "list":
            rows = repo.list_open(scope_key(scope))
            _print(rows, args.json)
            return 0
        review = repo.get(args.review_id)
        if review is None:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown review")
        if args.review_cmd == "show":
            _print(review, args.json)
            return 0
        if args.review_cmd == "approve":
            if not args.yes:
                print("re-run with --yes to confirm applying this effect", file=sys.stderr)
                return 2
            # Effect + review resolution commit in ONE transaction — a stale
            # expected revision fails atomically and the review stays open
            # (V2-19.08, V2-20.04). The operation key is stable so a retried
            # approval replays its receipt instead of re-applying.
            proposal = _proposal_for_review(
                review, f"review-approve:{args.review_id}", args.principal or "local-owner"
            )
            receipt = eng.apply_proposal(proposal)
            _print(
                {"review_id": args.review_id, "state": "approved", "receipt": receipt},
                args.json,
            )
            return 0
        # reject: record the refusal — never an effect on the evidence
        # itself (V2-20.05).
        with eng.store.tx() as conn:
            from .storage.repos import EventsRepo

            seq = EventsRepo(eng.store).append(
                conn, scope_key(scope),
                "review_rejected", args.principal or "local-owner",
                {"review_id": args.review_id},
                "cli-1",
            )
            repo.resolve(conn, args.review_id, "rejected", seq)
        _print({"review_id": args.review_id, "state": "rejected"}, args.json)
        return 0
    finally:
        eng.close()


def cmd_jobs(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        from .jobs.queue import JobQueue

        scope = _scope(args, eng)
        if args.job_cmd == "run":
            drained = eng.run_pending(limit=args.limit)
            _print({"jobs_drained": drained}, args.json)
            return 0
        q = JobQueue(eng.store)
        if args.job_cmd == "stats":
            _print(q.stats(scope_key(scope)), args.json)
        else:
            _print(q.list(scope_key(scope), state=args.state), args.json)
        return 0
    finally:
        eng.close()


def cmd_migrate(args: argparse.Namespace) -> int:
    """Explicitly migrate a database file to the current schema version.

    ``Store.open`` already upgrades older schemas on a writable open, so
    this subcommand is the operator-visible way to force the upgrade (and
    to resume a crash-interrupted rebuild) without starting the engine.
    """
    from .storage import migrations
    from .storage.store import Store

    store = Store.open(args.path)
    try:
        version = migrations.apply(store)
        _print(
            {
                "path": args.path,
                "schema_version": version,
                "fts_enabled": store.fts_enabled,
            },
            args.json,
        )
        return 0
    finally:
        store.close()


def cmd_mcp(args: argparse.Namespace) -> int:
    """Launch the stdio MCP adapter bound to this store + caller identity."""
    eng = _engine(args)
    try:
        from .core.types import CallerContext
        from .mcp import MCP_GRANTS, serve_stdio

        scope = eng.host.default_scope()
        caller = CallerContext(
            profile_id=scope.profile_id,
            principal_id=scope.principal_id or "local-owner",
            agent_id="mcp",
            session_id=args.conversation or "mcp-stdio",
            workspace_id=scope.workspace_id,
            conversation_id=scope.conversation_id,
            grants=MCP_GRANTS,
        )
        return serve_stdio(eng, caller, instream=sys.stdin, outstream=sys.stdout)
    finally:
        eng.close()


def cmd_consent(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        from .storage.repos import ConsentsRepo

        repo = ConsentsRepo(eng.store)
        profile = scope_key(eng.host.default_scope())
        if args.consent_cmd == "show":
            _print(repo.active(profile, args.processor, args.purpose), args.json)
            return 0
        if args.consent_cmd == "grant":
            with eng.store.tx() as conn:
                cid = repo.grant(conn, profile, args.processor, args.purpose, "cli-grant-1")
            _print({"granted": cid}, args.json)
            return 0
        if args.consent_cmd == "revoke":
            row = repo.active(profile, args.processor, args.purpose)
            if row is None:
                raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "no active consent")
            with eng.store.tx() as conn:
                repo.revoke(conn, row["consent_id"])
            _print({"revoked": args.purpose}, args.json)
            return 0
        return 2
    finally:
        eng.close()


def cmd_grants(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        scope = eng.host.default_scope()
        sid = scope_key(scope)
        if args.grants_cmd == "list":
            from .storage.repos_v2 import GrantsRepo

            with eng.store.read() as conn:
                repo = GrantsRepo(eng.store)
                if args.principal:
                    rows = {
                        p: ("revoked" if p in repo.revoked(conn, sid, args.principal) else "active")
                        for p in repo.active(conn, sid, args.principal)
                        | repo.revoked(conn, sid, args.principal)
                    }
                else:
                    raw = conn.execute(
                        "SELECT principal_id, permission, revoked_event"
                        " FROM scope_grants WHERE scope_id = ?"
                        " ORDER BY principal_id, permission",
                        (sid,),
                    ).fetchall()
                    rows = [
                        {
                            "principal_id": r[0],
                            "permission": r[1],
                            "state": "revoked" if r[2] is not None else "active",
                        }
                        for r in raw
                    ]
            _print(rows, args.json)
            return 0
        if not args.principal or not args.permission:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "grant/revoke require <principal> and <permission>",
            )
        if args.grants_cmd == "grant":
            _print(
                eng.grant_permission(args.principal, args.permission),
                args.json,
            )
            return 0
        if args.grants_cmd == "revoke":
            _print(
                eng.revoke_permission(args.principal, args.permission),
                args.json,
            )
            return 0
        return 2
    finally:
        eng.close()


def cmd_backup(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        eng.store.backup(args.dest)
        _print({"backup": args.dest}, args.json)
        return 0
    finally:
        eng.close()


def cmd_purge(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        if args.purge_cmd in ("preview", "suppress"):
            if not args.target:
                print(f"purge {args.purge_cmd} requires an object_id",
                      file=sys.stderr)
                return 2
            fn = eng.plan_purge if args.purge_cmd == "preview" else eng.suppress
            _print(fn([(args.kind, args.target)]), args.json)
            return 0
        if args.purge_cmd in ("execute", "lift"):
            if not args.target:
                print(f"purge {args.purge_cmd} requires a purge_id",
                      file=sys.stderr)
                return 2
            if args.purge_cmd == "execute":
                if not args.yes:
                    print("re-run with --yes to confirm physical erasure",
                          file=sys.stderr)
                    return 2
                _print(eng.execute_purge(args.target), args.json)
            else:
                _print(eng.lift_suppression(args.target), args.json)
            return 0
        return 2
    finally:
        eng.close()


def cmd_export(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        bundle = eng.export_scope(
            args.dest, portable=not args.metadata_only
        )
        if args.dest is None:
            _print(bundle, True)
        else:
            _print({"export": args.dest,
                    "records": bundle["manifest"].get("counts")}, args.json)
        return 0
    finally:
        eng.close()


def cmd_import(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        receipt = eng.import_bundle(args.file)
        _print(receipt, args.json)
        return 0
    finally:
        eng.close()


def cmd_serve(args: argparse.Namespace) -> int:
    """Optional loopback HTTP API (SPEC_V4 §45) — bearer tokens from
    --token-file or VERBATIM_SERVICE_TOKEN_FILE / VERBATIM_SERVICE_TOKENS."""
    eng = _engine(args)
    try:
        from .service.httpd import HttpConfig
        from .service.server import serve

        return serve(
            eng,
            HttpConfig(host=args.bind, port=args.port, allow_remote=args.allow_remote),
            token_file=args.token_file,
        )
    finally:
        eng.close()


def cmd_workbench(args: argparse.Namespace) -> int:
    """Operator workbench UI (SPEC_V4 §49) — authenticated loopback only."""
    eng = _engine(args)
    try:
        from .service.httpd import HttpConfig
        from .workbench.server import serve

        return serve(
            eng,
            HttpConfig(host=args.bind, port=args.port, allow_remote=args.allow_remote),
            token_file=args.token_file,
        )
    finally:
        eng.close()


def cmd_share(args: argparse.Namespace) -> int:
    eng = _engine(args)
    try:
        if args.share_cmd == "create":
            if not args.recipient or not args.object_id:
                print("share create requires --recipient and --object-id",
                      file=sys.stderr)
                return 2
            refs = [(args.kind, args.object_id)]
            _print(
                eng.share(
                    args.recipient, refs,
                    permission=args.permission,
                    expires_us=parse_rfc3339(args.expires) if args.expires else None,
                ),
                args.json,
            )
            return 0
        if args.share_cmd == "consume":
            _print(eng.consume_handoff(args.capsule_id), args.json)
            return 0
        if args.share_cmd == "revoke":
            _print(eng.revoke_handoff(args.capsule_id), args.json)
            return 0
        return 2
    finally:
        eng.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verbatim", description="Verbatim evidence-first memory engine")
    p.add_argument("--data-dir", help="store directory (default: ./verbatim)")
    p.add_argument("--config", help="JSON config file")
    p.add_argument("--profile", default="local", help="profile id (default: local)")
    p.add_argument("--principal", help="principal id (default: local-owner)")
    p.add_argument("--conversation", help="conversation id")
    p.add_argument("--workspace", help="workspace id")
    p.add_argument("--visibility", choices=[v.value for v in Visibility],
                   help="addressed partition visibility (default: conversation)")
    p.add_argument("--json", action="store_true", help="versioned JSON output")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="mode, capabilities, counts, queue health")
    sub.add_parser("doctor", help="read-only integrity + capability diagnostics")

    pi = sub.add_parser("ingest", help="import a text file as evidence")
    pi.add_argument("file")
    pi.add_argument("--kind", default="import", choices=[k.value for k in SourceKind])
    pi.add_argument("--provenance", default="legacy_import",
                    choices=[p.value for p in Provenance])

    ps = sub.add_parser("search", help="recall evidence")
    ps.add_argument("query")
    ps.add_argument("--mode", default="current", choices=[m.value for m in RecallMode])
    ps.add_argument("--limit", type=int, default=8)
    ps.add_argument("--valid-at", dest="valid_at")
    ps.add_argument("--known-at", dest="known_at", type=int)

    pi2 = sub.add_parser("inspect", help="claim evidence + interpretation lineage")
    pi2.add_argument("claim_id")

    pe = sub.add_parser("explain", help="why a claim reads this way (lineage + decisions)")
    pe.add_argument("claim_id")

    psp = sub.add_parser("spans", help="exact evidence spans behind a claim")
    psp.add_argument("claim_id")

    pm = sub.add_parser("remember", help="quote a byte range of an accepted source")
    pm.add_argument("source_id")
    pm.add_argument("start_byte", type=int)
    pm.add_argument("end_byte", type=int)
    pm.add_argument("--predicate")
    pm.add_argument("--revision", type=int, default=1)

    pr = sub.add_parser("reviews", help="operator review workflow")
    pr.add_argument("review_cmd", choices=["list", "show", "approve", "reject"])
    pr.add_argument("review_id", nargs="?")
    pr.add_argument("--yes", action="store_true")

    pj = sub.add_parser("jobs", help="durable job queue")
    pj.add_argument("job_cmd", choices=["list", "stats", "run"])
    pj.add_argument("--state")
    pj.add_argument("--limit", type=int, default=64)

    pmg = sub.add_parser(
        "migrate",
        help="migrate a database file to the current schema version",
    )
    pmg.add_argument("path", help="database file path")

    sub.add_parser("mcp", help="serve the stdio MCP JSON-RPC adapter")

    pc = sub.add_parser("consent", help="remote-processing consent")
    pc.add_argument("consent_cmd", choices=["show", "grant", "revoke"])
    pc.add_argument("--processor", default="typesafe")
    pc.add_argument("--purpose", default="candidate_curation")

    pg = sub.add_parser("grants", help="scope permission administration")
    pg.add_argument("grants_cmd", choices=["list", "grant", "revoke"])
    pg.add_argument("principal", nargs="?")
    pg.add_argument("permission", nargs="?")

    pb = sub.add_parser("backup", help="consistent snapshot backup")
    pb.add_argument("dest")

    pp = sub.add_parser("purge", help="erasure preview/execute + suppression")
    pp.add_argument("purge_cmd", choices=["preview", "execute", "suppress", "lift"])
    pp.add_argument("target", nargs="?",
                    help="object_id for preview/suppress, purge_id for execute/lift")
    pp.add_argument("--kind", default="claim",
                    choices=["claim", "span", "source", "source_revision", "artifact"])
    pp.add_argument("--yes", action="store_true")

    px = sub.add_parser("export", help="versioned evidence bundle")
    px.add_argument("dest", nargs="?", help="output path (default: stdout)")
    px.add_argument("--metadata-only", action="store_true",
                    help="manifest + digests only, no payload bytes")

    pim = sub.add_parser("import", help="import a bundle into this scope")
    pim.add_argument("file")

    psh = sub.add_parser("share", help="handoff capsules between principals")
    psh.add_argument("share_cmd", choices=["create", "consume", "revoke"])
    psh.add_argument("capsule_id", nargs="?",
                     help="capsule id for consume/revoke")
    psh.add_argument("--kind", default="claim",
                     help="object kind for create")
    psh.add_argument("--object-id", dest="object_id",
                     help="object id for create")
    psh.add_argument("--recipient", help="recipient principal/agent id")
    psh.add_argument("--permission", default="read_evidence")
    psh.add_argument("--expires")

    for _name, _help in (
        ("serve", "optional loopback HTTP API (bearer-token auth)"),
        ("workbench", "operator workbench UI (authenticated loopback)"),
    ):
        _psv = sub.add_parser(_name, help=_help)
        _psv.add_argument("--bind", default="127.0.0.1")
        _psv.add_argument("--port", type=int, default=0)
        _psv.add_argument("--token-file", dest="token_file",
                          help="operator-provisioned bearer token JSON file")
        _psv.add_argument("--allow-remote", action="store_true",
                          help="accept a non-loopback cleartext bind (TLS unimplemented)")

    return p


_HANDLERS = {
    "status": cmd_status,
    "doctor": cmd_doctor,
    "ingest": cmd_ingest,
    "search": cmd_search,
    "inspect": cmd_inspect,
    "explain": cmd_explain,
    "spans": cmd_spans,
    "remember": cmd_remember,
    "reviews": cmd_reviews,
    "jobs": cmd_jobs,
    "migrate": cmd_migrate,
    "mcp": cmd_mcp,
    "consent": cmd_consent,
    "grants": cmd_grants,
    "backup": cmd_backup,
    "purge": cmd_purge,
    "export": cmd_export,
    "import": cmd_import,
    "share": cmd_share,
    "serve": cmd_serve,
    "workbench": cmd_workbench,
}


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _HANDLERS[args.cmd](args)
    except VerbatimError as exc:
        print(f"error[{exc.code.value}]: {exc.message}", file=sys.stderr)
        return _EXIT.get(exc.code, 5)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
