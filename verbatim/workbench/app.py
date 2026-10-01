"""Operator workbench — an authenticated loopback HTML UI (SPEC_V4 §49).

One honest core shared with the service API (``verbatim.service.httpd``
transport + ``service.auth`` credentials). The workbench is an OPERATOR
surface: every request — reads included — requires a token whose
principal holds ``admin`` on the bound scope through
``governance.authorize``, then all reads/mutations go through the same
``Engine`` verified paths (inspect quote gating, review proposals,
purge closure). Nothing here touches evidence bytes except through the
engine's own gating; stored text is HTML-escaped and no page emits
scripts (V4-49.08).

Views follow §49's operator-question table:

- ``/``           index + partition summary
- ``/status``     mode, capabilities, integrity, queue health
- ``/objects``    authorized inventory — sources vs derived claims,
                  suppressed rows rendered withheld (V4-49.03/49.05)
- ``/claims/{id}`` evidence beside interpretation, with a correction form
- ``/reviews``    risk-ordered open queue + approve/reject (real review path)
- ``/jobs``       queue age/states + explicit drain
- ``/readiness``  per-receipt obligation DAG state
- ``/forget``     preview → confirm → execute, suppression lift,
                  suppression/closure state list
- ``/search``     operator recall debugging (why-did-you-return-this)

A small ``/api/*`` mirror keeps the same views machine-readable.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Optional

from ..core.identity import scope_key
from ..core.types import (
    CallerContext,
    ErrorCode,
    Scope,
    VerbatimError,
    Visibility,
)
from ..core.types_v3 import Verb
from .. import governance
from ..service.api import _targets, apply_review
from ..service.auth import TokenAuthenticator, TokenCredential
from ..service.httpd import (
    Application,
    Request,
    Response,
    Router,
    denial,
)
from ..storage.repos import has_table

_e = html.escape

_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem;max-width:72rem;color:#111;background:#fff}
nav a{margin-right:1rem}
table{border-collapse:collapse;margin:.5rem 0}
th,td{border:1px solid #999;padding:.25rem .6rem;text-align:left;vertical-align:top}
th{background:#eee}
code{background:#f4f4f4;padding:0 .2rem}
.err{border:2px solid #a00;background:#fee;padding:.5rem}
.ok{border:2px solid #060;background:#efe;padding:.5rem}
.withheld{color:#555;font-style:italic}
label{display:block;margin:.4rem 0}
fieldset{border:1px solid #999;margin:.6rem 0}
"""

_NAV = (
    '<nav aria-label="workbench">'
    '<a href="/">home</a><a href="/status">status</a>'
    '<a href="/objects">objects</a><a href="/reviews">reviews</a>'
    '<a href="/jobs">jobs</a><a href="/readiness">readiness</a>'
    '<a href="/forget">forget</a><a href="/search">search</a></nav>'
)

_PURGE_STATE_LABEL = {
    "previewed": "previewed",
    "suppressed": "suppressed (reversible, withheld)",
    "purging": "cleaning in progress",
    "completed": "verified erased (local authority)",
}


def _page(title: str, body: str) -> Response:
    doc = (
        "<!doctype html><html lang=\"en\"><head>"
        '<meta charset="utf-8">'
        f"<title>{_e(title)} — verbatim workbench</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{_NAV}<main><h1>{_e(title)}</h1>{body}</main></body></html>"
    )
    return Response(status=200, body=doc, content_type="text/html; charset=utf-8")


def _redirect(location: str) -> Response:
    return Response(status=303, body=b"", headers={"Location": location})


def _err_page(title: str, exc: BaseException) -> Response:
    if isinstance(exc, VerbatimError):
        msg = f"error[{exc.code.value}]: {exc.message}"
    else:
        msg = "internal error"
    return _page(title, f'<p class="err" role="alert">{_e(msg)}</p>')


def _fstr(body: Any, key: str) -> Optional[str]:
    """A trimmed string field from a form dict (or JSON object)."""
    if not isinstance(body, dict):
        return None
    v = body.get(key)
    if not isinstance(v, str):
        return None
    v = v.strip()
    return v or None


def _fint(body: Any, key: str) -> Optional[int]:
    v = _fstr(body, key)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"{key} must be an integer"
        ) from None


class WorkbenchApp(Application):
    """Operator workbench bound to one Engine + operator credentials."""

    realm = "verbatim-workbench"

    def __init__(self, engine: Any, credentials: Iterable[TokenCredential]) -> None:
        self._engine = engine
        self._auth = TokenAuthenticator(credentials)
        self._router = Router()
        self._routes()

    # -- auth ------------------------------------------------------------

    def authenticate(self, request: Request) -> Optional[TokenCredential]:
        cred = self._auth.authenticate(request.header("authorization"))
        if cred is None:
            return None
        if Verb.ADMIN.value not in cred.verbs:
            # The workbench is an operator surface (§49): a credential that
            # was never provisioned the admin verb class gets the same
            # 401 as an unknown token — the surface does not exist for it.
            return None
        return cred

    def _operator_caller(self, cred: TokenCredential, scope: Scope) -> CallerContext:
        """Authorize ``admin`` through the grant table, then build the
        operator CallerContext the engine's operator paths require."""
        caller_v3 = governance.CallerV3(
            principal_id=cred.principal_id, session_id="workbench",
            host_id="workbench",
        )
        with self._engine.store.read() as conn:
            if not has_table(conn, "grants_v3"):
                raise denial()
            governance.authorize(
                conn, caller_v3, scope_key(scope), Verb.ADMIN.value
            )
        return CallerContext(
            profile_id=scope.profile_id,
            principal_id=cred.principal_id,
            agent_id="workbench",
            session_id="workbench",
            workspace_id=scope.workspace_id,
            conversation_id=scope.conversation_id,
            grants=frozenset(),
            is_operator=True,
        )

    def _scope(self, cred: TokenCredential, fields: dict) -> Scope:
        base = self._engine.host.default_scope()
        pin = cred.scope or {}
        vis = (
            fields.get("visibility")
            or pin.get("visibility")
            or base.visibility
        )
        return Scope(
            profile_id=base.profile_id,
            principal_id=cred.principal_id,
            workspace_id=(
                fields.get("workspace_id") or pin.get("workspace_id") or base.workspace_id
            ),
            conversation_id=(
                fields.get("conversation_id")
                or pin.get("conversation_id")
                or base.conversation_id
            ),
            visibility=vis if isinstance(vis, Visibility) else Visibility(str(vis)),
        )

    def dispatch(self, request: Request, cred: TokenCredential) -> Response:
        handler, params = self._router.match(request.method, request.path)
        fields = dict(request.query)
        if isinstance(request.body, dict):
            fields.update(request.body)
        scope = self._scope(cred, fields)
        try:
            caller = self._operator_caller(cred, scope)
        except VerbatimError as exc:
            return _err_page("not authorized", exc)
        try:
            return handler(self, request, cred, scope, caller, params)
        except VerbatimError as exc:
            return _err_page(request.path, exc)

    # -- routes ------------------------------------------------------------

    def _routes(self) -> None:
        r = self._router.add
        r("GET", "/", WorkbenchApp._index)
        r("GET", "/status", WorkbenchApp._status)
        r("GET", "/objects", WorkbenchApp._objects)
        r("GET", "/claims/{claim_id}", WorkbenchApp._claim)
        r("POST", "/claims/{claim_id}/correct", WorkbenchApp._claim_correct)
        r("GET", "/reviews", WorkbenchApp._reviews)
        r("GET", "/reviews/{review_id}", WorkbenchApp._review)
        r("POST", "/reviews/{review_id}/approve", WorkbenchApp._review_approve)
        r("POST", "/reviews/{review_id}/reject", WorkbenchApp._review_reject)
        r("GET", "/jobs", WorkbenchApp._jobs)
        r("POST", "/jobs/drain", WorkbenchApp._jobs_drain)
        r("GET", "/readiness", WorkbenchApp._readiness)
        r("GET", "/search", WorkbenchApp._search)
        r("GET", "/forget", WorkbenchApp._forget)
        r("POST", "/forget/preview", WorkbenchApp._forget_preview)
        r("POST", "/forget/execute", WorkbenchApp._forget_execute)
        r("POST", "/forget/suppress", WorkbenchApp._forget_suppress)
        r("POST", "/forget/lift", WorkbenchApp._forget_lift)
        # machine-readable mirrors of the same views
        r("GET", "/api/status", WorkbenchApp._api_status)
        r("GET", "/api/objects", WorkbenchApp._api_objects)
        r("GET", "/api/claims/{claim_id}", WorkbenchApp._api_claim)
        r("GET", "/api/reviews", WorkbenchApp._api_reviews)

    # -- helpers ------------------------------------------------------------

    def _suppressed_objects(self) -> set:
        """(kind, object_id) under a live suppression/purge — the withheld set."""
        sid = scope_key(self._engine.host.default_scope())
        out = set()
        with self._engine.store.read() as conn:
            rows = conn.execute(
                "SELECT pt.object_kind, pt.object_id FROM purge_targets pt"
                " JOIN purges p ON p.purge_id = pt.purge_id"
                " WHERE p.state IN ('suppressed','purging','completed')"
                " AND p.scope_id = ?",
                (sid,),
            ).fetchall()
        for kind, oid in rows:
            out.add((kind, oid))
        return out

    def _purges(self) -> list[dict[str, Any]]:
        sid = scope_key(self._engine.host.default_scope())
        with self._engine.store.read() as conn:
            rows = conn.execute(
                "SELECT p.purge_id, p.state, p.requested_us, p.completed_us,"
                " pt.object_kind, pt.object_id"
                " FROM purges p LEFT JOIN purge_targets pt"
                "   ON pt.purge_id = p.purge_id"
                " WHERE p.scope_id = ? ORDER BY p.requested_us",
                (sid,),
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for pid, state, req, done, kind, oid in rows:
            entry = out.setdefault(pid, {
                "purge_id": pid, "state": state, "requested_us": req,
                "completed_us": done, "targets": [],
            })
            if kind is not None:
                entry["targets"].append({"object_kind": kind, "object_id": oid})
        return sorted(out.values(), key=lambda p: p["purge_id"])

    # -- pages --------------------------------------------------------------

    def _index(self, request, cred, scope, caller, params) -> Response:
        sid = scope_key(scope)
        with self._engine.store.read() as conn:
            n_claims = conn.execute(
                "SELECT COUNT(*) FROM claims WHERE scope_id = ?", (sid,)
            ).fetchone()[0]
            n_sources = conn.execute(
                "SELECT COUNT(*) FROM sources WHERE scope_id = ?", (sid,)
            ).fetchone()[0]
            n_reviews = conn.execute(
                "SELECT COUNT(*) FROM reviews WHERE scope_id = ? AND state='open'",
                (sid,),
            ).fetchone()[0]
        suppressed = len(self._suppressed_objects())
        body = (
            f"<p>partition <code>{_e(sid)}</code> — "
            f"{n_claims} claims · {n_sources} sources · "
            f"{n_reviews} open reviews · {suppressed} withheld objects</p>"
            "<p>Everything here runs through the engine's authorized paths; "
            "the workbench holds no independent authority.</p>"
        )
        return _page("operator workbench", body)

    def _status(self, request, cred, scope, caller, params) -> Response:
        st = self._engine.status(caller=caller)
        caps = st["capabilities"]
        rows = "".join(
            f"<tr><th scope=\"row\">{_e(k)}</th><td>{_e(v['state'])}</td>"
            f"<td>{_e(str(v.get('degraded_reason') or '—'))}</td></tr>"
            for k, v in sorted(caps.items())
        )
        integ = st.get("integrity") or {}
        integ_rows = "".join(
            f"<tr><th scope=\"row\">{_e(str(k))}</th><td>{_e(str(v))}</td></tr>"
            for k, v in sorted(integ.items())
            if isinstance(v, (str, int, float, bool)) or v is None
        )
        body = (
            f"<p>mode <code>{_e(st['mode'])}</code> · "
            f"capture_enabled={_e(str(st['capture_enabled']))} · "
            f"projection_generation={_e(str(st['projection_generation']))} · "
            f"policy_epoch={_e(str(st['policy_epoch']))}</p>"
            f"<h2>capabilities</h2><table>"
            f"<tr><th>capability</th><th>state</th><th>degraded reason</th></tr>{rows}</table>"
            f"<h2>integrity</h2><table>{integ_rows}</table>"
        )
        return _page("status", body)

    def _objects(self, request, cred, scope, caller, params) -> Response:
        """Authorized inventory: sources (evidence) vs claims (derived)."""
        sid = scope_key(scope)
        suppressed = self._suppressed_objects()
        with self._engine.store.read() as conn:
            claims = conn.execute(
                "SELECT c.claim_id, cr.revision, cr.state, c.predicate"
                " FROM claims c JOIN claim_revisions cr"
                "  ON cr.claim_id = c.claim_id"
                "  AND cr.revision = (SELECT MAX(revision) FROM claim_revisions"
                "                     WHERE claim_id = c.claim_id)"
                " WHERE c.scope_id = ? ORDER BY c.claim_id LIMIT 500",
                (sid,),
            ).fetchall()
            sources = conn.execute(
                "SELECT source_id, source_kind, speaker_id, created_us"
                " FROM sources WHERE scope_id = ? ORDER BY created_us LIMIT 500",
                (sid,),
            ).fetchall()
        crows = []
        for cid, rev, state, pred in claims:
            if ("claim", cid) in suppressed:
                crows.append(
                    f"<tr><td><code>{_e(cid[:16])}…</code></td>"
                    f"<td colspan=\"3\" class=\"withheld\">withheld — suppressed</td></tr>"
                )
            else:
                crows.append(
                    f'<tr><td><a href="/claims/{_e(cid)}"><code>{_e(cid[:16])}…</code></a></td>'
                    f"<td>rev {_e(str(rev))}</td><td>{_e(state)}</td>"
                    f"<td>{_e(str(pred or ''))}</td></tr>"
                )
        srows = []
        for src, kind, speaker, created in sources:
            if ("source", src) in suppressed:
                srows.append(
                    f"<tr><td><code>{_e(src[:16])}…</code></td>"
                    f"<td colspan=\"3\" class=\"withheld\">withheld — suppressed</td></tr>"
                )
            else:
                srows.append(
                    f"<tr><td><code>{_e(src[:16])}…</code></td><td>{_e(kind)}</td>"
                    f"<td>{_e(str(speaker or ''))}</td><td>{_e(str(created))}</td></tr>"
                )
        body = (
            "<h2>derived objects (claims)</h2>"
            f"<table><tr><th>claim</th><th>rev</th><th>state</th><th>predicate</th></tr>"
            f"{''.join(crows) or '<tr><td colspan=4>none</td></tr>'}</table>"
            "<h2>evidence (sources)</h2>"
            f"<table><tr><th>source</th><th>kind</th><th>speaker</th><th>captured</th></tr>"
            f"{''.join(srows) or '<tr><td colspan=4>none</td></tr>'}</table>"
        )
        return _page("objects", body)

    def _claim(self, request, cred, scope, caller, params) -> Response:
        claim_id = params["claim_id"]
        suppressed = ("claim", claim_id) in self._suppressed_objects()
        try:
            detail = self._engine.inspect(claim_id, scope, caller=caller)
        except VerbatimError as exc:
            if exc.code in (
                ErrorCode.NOT_FOUND_OR_FORBIDDEN,
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            ):
                # Held/missing are one shape — the page says "withheld",
                # never "exists but you may not see it" (§09.09).
                label = "withheld — suppressed" if suppressed else "not found or withheld"
                return _page("claim", f'<p class="withheld">{_e(label)}</p>')
            return _err_page("claim", exc)
        revs = "".join(
            f"<tr><td>{_e(str(r['revision']))}</td><td>{_e(str(r['state']))}</td>"
            f"<td>{_e(str(r.get('polarity')))}</td><td>{_e(str(r.get('modality')))}</td>"
            f"<td><code>{_e(str(r.get('interpretation') or ''))[:400]}</code></td></tr>"
            for r in detail.get("revisions", ())
        )
        evs = "".join(
            f"<tr><td><code>{_e(str(e['span_id'])[:20])}</code></td>"
            f"<td>{_e(str(e.get('role') or ''))}</td>"
            f"<td><code>{_e(str(e['source_id'])[:20])}@{_e(str(e['source_revision']))}</code></td>"
            f"<td>{_e(str(e.get('text') or '[unavailable]'))}</td></tr>"
            for e in detail.get("evidence", ())
        )
        edges = "".join(
            f"<tr><td>{_e(str(ed.get('edge_type') or ed.get('type') or ''))}</td>"
            f"<td><code>{_e(str(ed.get('source_id') or ''))[:20]}</code></td>"
            f"<td><code>{_e(str(ed.get('target_id') or ''))[:20]}</code></td></tr>"
            for ed in detail.get("edges", ())
        )
        body = (
            f"<p>claim <code>{_e(claim_id)}</code> · predicate "
            f"<code>{_e(str(detail.get('predicate') or ''))}</code></p>"
            "<h2>revisions (interpretation)</h2>"
            f"<table><tr><th>rev</th><th>state</th><th>polarity</th><th>modality</th><th>object</th></tr>{revs}</table>"
            "<h2>evidence (exact quotes — held spans render [unavailable])</h2>"
            f"<table><tr><th>span</th><th>role</th><th>source</th><th>text</th></tr>{evs}</table>"
            "<h2>relations</h2>"
            f"<table><tr><th>edge</th><th>source</th><th>target</th></tr>{edges or '<tr><td colspan=3>none</td></tr>'}</table>"
            "<h2>propose a correction</h2>"
            f'<form method="post" action="/claims/{_e(claim_id)}/correct">'
            '<fieldset><legend>transition proposal</legend>'
            '<label>effect <select name="effect">'
            '<option value="correct">correct</option>'
            '<option value="dispute">dispute</option>'
            '<option value="archive">archive</option>'
            '<option value="supersede">supersede</option></select></label>'
            '<label>successor claim id (for correct/supersede) '
            '<input name="successor_claim_id" size="40"></label>'
            '<label>reason <input name="reason" size="60"></label>'
            '<button type="submit">propose (creates a review)</button>'
            "</fieldset></form>"
            "<h2>forget</h2>"
            '<form method="post" action="/forget/suppress">'
            f'<input type="hidden" name="object_kind" value="claim">'
            f'<input type="hidden" name="object_id" value="{_e(claim_id)}">'
            '<button type="submit">suppress (reversible)</button></form>'
            '<form method="post" action="/forget/preview">'
            f'<input type="hidden" name="object_kind" value="claim">'
            f'<input type="hidden" name="object_id" value="{_e(claim_id)}">'
            '<button type="submit">preview erasure</button></form>'
        )
        return _page("claim detail", body)

    def _claim_correct(self, request, cred, scope, caller, params) -> Response:
        """Correction proposals ride the real review path — a review row
        is created; the queue approves it through apply_proposal."""
        from ..core.types import TransitionCommand

        body = request.body if isinstance(request.body, dict) else {}
        effect = _fstr(body, "effect") or "correct"
        if effect not in ("correct", "dispute", "archive", "supersede", "admit", "restore"):
            raise VerbatimError(ErrorCode.VALIDATION, f"unsupported effect {effect!r}")
        cmd = TransitionCommand(
            claim_id=params["claim_id"],
            expected_revision=_fint(body, "expected_revision") or 0,
            effect=effect,
            actor_id=cred.principal_id,
            reason=_fstr(body, "reason") or "operator correction",
            successor_claim_id=_fstr(body, "successor_claim_id"),
        )
        review_id = self._engine.propose_transition(cmd, scope, caller=caller)
        return _redirect(f"/reviews/{review_id}")

    # -- reviews -------------------------------------------------------------

    def _reviews(self, request, cred, scope, caller, params) -> Response:
        from ..storage.repos import ReviewsRepo

        rows = ReviewsRepo(self._engine.store).list_open(scope_key(scope))
        trs = []
        for r in rows:
            effect = r.get("proposed_effect") or {}
            claim = effect.get("claim_id") or effect.get("predecessor_id") or ""
            trs.append(
                f'<tr><td><a href="/reviews/{_e(r["review_id"])}">'
                f'<code>{_e(r["review_id"][:16])}…</code></a></td>'
                f"<td>{_e(str(effect.get('effect') or ''))}</td>"
                f"<td><code>{_e(str(claim)[:20])}</code></td>"
                f"<td>{_e(str(effect.get('reason') or ''))}</td></tr>"
            )
        body = (
            f"<p>{len(rows)} open review(s). Approval applies the proposed "
            "effect and resolves the review atomically; rejection records "
            "the refusal without touching evidence.</p>"
            "<table><tr><th>review</th><th>effect</th><th>claim</th><th>reason</th></tr>"
            f"{''.join(trs) or '<tr><td colspan=4>queue empty</td></tr>'}</table>"
        )
        return _page("review queue", body)

    def _review(self, request, cred, scope, caller, params) -> Response:
        from ..storage.repos import ReviewsRepo

        review = ReviewsRepo(self._engine.store).get(params["review_id"])
        if review is None or review.get("scope_id") != scope_key(scope):
            raise denial()
        import json as _json

        effect = review.get("proposed_effect") or {}
        rows = "".join(
            f"<tr><th scope=\"row\">{_e(str(k))}</th><td><code>{_e(str(v))}</code></td></tr>"
            for k, v in sorted(effect.items())
        )
        actions = ""
        if review.get("state") == "open":
            actions = (
                f'<form method="post" action="/reviews/{_e(params["review_id"])}/approve">'
                '<button type="submit">approve — apply effect</button></form>'
                f'<form method="post" action="/reviews/{_e(params["review_id"])}/reject">'
                '<button type="submit">reject — no effect</button></form>'
            )
        body = (
            f"<p>review <code>{_e(params['review_id'])}</code> · "
            f"state <strong>{_e(str(review.get('state')))}</strong></p>"
            f"<h2>proposed effect</h2><table>{rows}</table>"
            f"<h2>expected versions</h2><code>{_e(_json.dumps(review.get('expected_versions') or {}))}</code>"
            f"{actions}"
        )
        return _page("review", body)

    def _review_approve(self, request, cred, scope, caller, params) -> Response:
        apply_review(
            self._engine,
            params["review_id"],
            scope,
            f"workbench-approve:{params['review_id']}",
            actor_id=cred.principal_id,
            caller=caller,
        )
        return _redirect(f"/reviews/{params['review_id']}")

    def _review_reject(self, request, cred, scope, caller, params) -> Response:
        from ..storage.repos import EventsRepo, ReviewsRepo

        rid = params["review_id"]
        repo = ReviewsRepo(self._engine.store)
        review = repo.get(rid)
        if review is None or review.get("scope_id") != scope_key(scope) \
                or review.get("state") != "open":
            raise denial()
        with self._engine.store.tx() as conn:
            seq = EventsRepo(self._engine.store).append(
                conn, scope_key(scope), "review_rejected",
                cred.principal_id, {"review_id": rid}, "workbench-1",
            )
            repo.resolve(conn, rid, "rejected", seq)
        return _redirect(f"/reviews/{rid}")

    # -- jobs / readiness ------------------------------------------------------

    def _jobs(self, request, cred, scope, caller, params) -> Response:
        from ..jobs.queue import JobQueue

        q = JobQueue(self._engine.store)
        sid = scope_key(scope)
        stats = q.stats(sid)
        rows = q.list(sid, limit=100)
        trs = "".join(
            f"<tr><td><code>{_e(str(j.get('job_id'))[:16])}…</code></td>"
            f"<td>{_e(str(j.get('kind')))}</td><td>{_e(str(j.get('state')))}</td>"
            f"<td>{_e(str(j.get('attempts')))}</td>"
            f"<td>{_e(str(j.get('error_code') or ''))}</td></tr>"
            for j in rows
        )
        stats_rows = "".join(
            f"<tr><th scope=\"row\">{_e(str(k))}</th><td>{_e(str(v))}</td></tr>"
            for k, v in sorted(stats.items())
        )
        body = (
            f"<h2>queue health</h2><table>{stats_rows}</table>"
            '<form method="post" action="/jobs/drain">'
            '<button type="submit">drain pending jobs now</button></form>'
            f"<h2>jobs (newest 100)</h2><table><tr><th>job</th><th>kind</th>"
            f"<th>state</th><th>attempts</th><th>error</th></tr>"
            f"{trs or '<tr><td colspan=5>none</td></tr>'}</table>"
        )
        return _page("jobs", body)

    def _jobs_drain(self, request, cred, scope, caller, params) -> Response:
        report = self._engine.drain_report(limit=256, caller=caller)
        rows = "".join(
            f"<tr><th scope=\"row\">{_e(str(k))}</th><td>{_e(str(v))}</td></tr>"
            for k, v in sorted(report.items())
            if isinstance(v, (str, int, float, bool))
        )
        body = (
            f'<p class="ok">drain complete — {_e(str(report.get("succeeded")))} '
            f"succeeded, {_e(str(report.get('failed')))} failed, "
            f"{_e(str(report.get('still_pending', report.get('deferred', 0))))} "
            "still outstanding</p>"
            f"<table>{rows}</table>"
            '<p><a href="/jobs">back to jobs</a></p>'
        )
        return _page("drain report", body)

    def _readiness(self, request, cred, scope, caller, params) -> Response:
        receipt_id = _fstr(request.query, "receipt_id")
        form = (
            '<form method="get" action="/readiness">'
            '<label>receipt id <input name="receipt_id" size="48" '
            'placeholder="rc_ingest:…"></label>'
            '<button type="submit">inspect receipt DAG</button></form>'
        )
        detail = ""
        if receipt_id:
            try:
                state = self._engine.receipt_state(
                    receipt_id, scope=scope, caller=caller
                )
            except VerbatimError as exc:
                detail = f'<p class="err" role="alert">{_e(exc.code.value)}: {_e(exc.message)}</p>'
            else:
                rows = "".join(
                    f"<tr><td>{_e(str(k))}</td><td>{_e(str(v))}</td></tr>"
                    for k, v in sorted(state.items())
                )
                detail = f"<h2>receipt <code>{_e(receipt_id)}</code></h2><table>{rows}</table>"
        return _page("readiness", form + detail)

    # -- search ------------------------------------------------------------

    def _search(self, request, cred, scope, caller, params) -> Response:
        from ..core.types import RecallRequest

        query = _fstr(request.query, "query")
        form = (
            '<form method="get" action="/search">'
            '<label>query <input name="query" size="48"></label>'
            '<button type="submit">recall</button></form>'
        )
        out = ""
        if query:
            res = self._engine.recall(
                RecallRequest(query=query, scope=scope, max_bytes=24000),
                caller=caller,
            )
            trs = "".join(
                f'<tr><td><a href="/claims/{_e(i.claim_id)}"><code>'
                f"{_e(i.claim_id[:16])}…</code></a></td>"
                f"<td>{_e(i.text)}</td><td>{_e(i.lifecycle.value)}</td>"
                f"<td>{_e(i.valid_label)}</td>"
                f"<td>{_e(', '.join(i.reasons))}</td></tr>"
                for i in res.items
            )
            warns = "".join(
                f'<p class="err">{_e(w)}</p>' for w in res.warnings
            )
            out = (
                f"<p>{len(res.items)} item(s), {_e(str(res.omitted))} omitted</p>"
                f"{warns}<table><tr><th>claim</th><th>quote</th><th>state</th>"
                f"<th>valid</th><th>signals</th></tr>"
                f"{trs or '<tr><td colspan=5>no evidence</td></tr>'}</table>"
            )
        return _page("search", form + out)

    # -- forget --------------------------------------------------------------

    def _forget(self, request, cred, scope, caller, params) -> Response:
        purges = self._purges()
        trs = []
        for p in purges:
            label = _PURGE_STATE_LABEL.get(p["state"], p["state"])
            targets = ", ".join(
                f"{t['object_kind']}:{t['object_id'][:16]}" for t in p["targets"]
            )
            lift = ""
            if p["state"] in ("previewed", "suppressed"):
                lift = (
                    '<form method="post" action="/forget/lift" style="display:inline">'
                    f'<input type="hidden" name="purge_id" value="{_e(p["purge_id"])}">'
                    '<button type="submit">lift</button></form>'
                )
            trs.append(
                f'<tr><td><code>{_e(p["purge_id"][:16])}…</code></td>'
                f"<td>{_e(label)}</td><td>{_e(targets)}</td><td>{lift}</td></tr>"
            )
        body = (
            "<h2>start a forget</h2>"
            '<form method="post" action="/forget/preview">'
            "<fieldset><legend>erasure preview (no mutation)</legend>"
            '<label>object kind <select name="object_kind">'
            '<option value="claim">claim</option><option value="source">source</option>'
            '<option value="source_revision">source_revision</option>'
            '<option value="span">span</option></select></label>'
            '<label>object id <input name="object_id" size="48"></label>'
            '<label>revision (optional) <input name="revision" size="6"></label>'
            '<button type="submit">preview closure</button></fieldset></form>'
            '<form method="post" action="/forget/suppress">'
            "<fieldset><legend>suppress (reversible tombstone)</legend>"
            '<label>object kind <select name="object_kind">'
            '<option value="claim">claim</option><option value="source">source</option>'
            '<option value="span">span</option></select></label>'
            '<label>object id <input name="object_id" size="48"></label>'
            '<button type="submit">suppress now</button></fieldset></form>'
            "<h2>suppression &amp; erasure states</h2>"
            "<table><tr><th>purge</th><th>state</th><th>targets</th><th></th></tr>"
            f"{''.join(trs) or '<tr><td colspan=4>none</td></tr>'}</table>"
            "<p>External copies (exports, projections) are a separate "
            "obligation — the erasure receipt reports them under "
            "<code>unhandled</code>/<code>derived</code> and never claims "
            "remote erasure it cannot prove.</p>"
        )
        return _page("forget", body)

    def _forget_preview(self, request, cred, scope, caller, params) -> Response:
        body = request.body if isinstance(request.body, dict) else {}
        targets = self._form_targets(body)
        plan = self._engine.plan_purge(
            targets, scope=scope, caller=caller, actor=cred.principal_id
        )
        def _t(t):
            return (t[0], t[1], t[2] if len(t) > 2 else None)

        rows = "".join(
            f"<tr><td>{_e(str(k))}</td><td><code>{_e(str(oid))}</code></td>"
            f"<td>{_e(str(rev)) if rev is not None else ''}</td></tr>"
            for k, oid, rev in (_t(t) for t in plan.get("targets", ()))
        )
        collateral = "".join(
            f"<tr><td>{_e(str(k))}</td><td><code>{_e(str(oid))}</code></td></tr>"
            for k, oid, _r in (_t(t) for t in plan.get("collateral", ()))
        )
        out = (
            f"<p>purge <code>{_e(plan['purge_id'])}</code> · state "
            f"<strong>{_e(plan['state'])}</strong> — nothing deleted yet.</p>"
            "<h2>selection</h2>"
            f"<table><tr><th>kind</th><th>object</th><th>rev</th></tr>{rows}</table>"
            "<h2>collateral dependents</h2>"
            f"<table><tr><th>kind</th><th>object</th></tr>{collateral or '<tr><td colspan=2>none</td></tr>'}</table>"
            '<form method="post" action="/forget/execute">'
            f'<input type="hidden" name="purge_id" value="{_e(plan["purge_id"])}">'
            '<label><input type="checkbox" name="confirm" value="yes"> '
            "I understand this physically erases the objects above and "
            "cannot be undone</label>"
            '<button type="submit">execute erasure</button></form>'
            '<p><a href="/forget">cancel — back to forget</a></p>'
        )
        return _page("erasure preview", out)

    def _forget_execute(self, request, cred, scope, caller, params) -> Response:
        body = request.body if isinstance(request.body, dict) else {}
        if _fstr(body, "confirm") not in ("yes", "true", "on", "1"):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "execute requires the explicit confirmation checkbox",
            )
        purge_id = _fstr(body, "purge_id")
        if not purge_id:
            raise VerbatimError(ErrorCode.VALIDATION, "purge_id required")
        result = self._engine.execute_purge(purge_id, scope=scope, caller=caller)
        rows = "".join(
            f"<tr><th scope=\"row\">{_e(str(k))}</th><td>{_e(str(v))}</td></tr>"
            for k, v in sorted(result.items())
            if isinstance(v, (str, int, float, bool))
        )
        unhandled = "".join(
            f"<li>{_e(str(u))}</li>" for u in result.get("unhandled", ())
        )
        return _page(
            "erasure complete",
            f'<p class="ok">purge {_e(purge_id)} completed — '
            f'{_e(str(result.get("payloads", 0)))} payload(s) scrubbed</p>'
            f"<table>{rows}</table>"
            + (f"<h2>unhandled / external obligations</h2><ul>{unhandled}</ul>" if unhandled else "")
            + '<p><a href="/forget">back to forget</a></p>',
        )

    def _forget_suppress(self, request, cred, scope, caller, params) -> Response:
        body = request.body if isinstance(request.body, dict) else {}
        targets = self._form_targets(body)
        self._engine.suppress(
            targets, scope=scope, caller=caller, actor=cred.principal_id
        )
        return _redirect("/forget")

    def _forget_lift(self, request, cred, scope, caller, params) -> Response:
        body = request.body if isinstance(request.body, dict) else {}
        purge_id = _fstr(body, "purge_id")
        if not purge_id:
            raise VerbatimError(ErrorCode.VALIDATION, "purge_id required")
        self._engine.lift_suppression(purge_id, scope=scope, caller=caller)
        return _redirect("/forget")

    @staticmethod
    def _form_targets(body: dict) -> list:
        kind = _fstr(body, "object_kind") or "claim"
        oid = _fstr(body, "object_id")
        if not oid:
            raise VerbatimError(ErrorCode.VALIDATION, "object_id required")
        rev = _fint(body, "revision")
        item: dict[str, Any] = {"object_kind": kind, "object_id": oid}
        if rev is not None:
            item["revision"] = rev
        return _targets([item])

    # -- JSON mirrors -----------------------------------------------------

    def _api_status(self, request, cred, scope, caller, params) -> Response:
        return Response(body=self._engine.status(caller=caller))

    def _api_objects(self, request, cred, scope, caller, params) -> Response:
        sid = scope_key(scope)
        suppressed = self._suppressed_objects()
        with self._engine.store.read() as conn:
            claims = conn.execute(
                "SELECT c.claim_id, cr.revision, cr.state FROM claims c"
                " JOIN claim_revisions cr ON cr.claim_id = c.claim_id"
                " AND cr.revision = (SELECT MAX(revision) FROM claim_revisions"
                "                   WHERE claim_id = c.claim_id)"
                " WHERE c.scope_id = ? LIMIT 500",
                (sid,),
            ).fetchall()
            sources = conn.execute(
                "SELECT source_id, source_kind FROM sources"
                " WHERE scope_id = ? LIMIT 500",
                (sid,),
            ).fetchall()
        return Response(body={
            "claims": [
                {
                    "claim_id": cid, "revision": rev, "state": state,
                    "withheld": ("claim", cid) in suppressed,
                }
                for cid, rev, state in claims
            ],
            "sources": [
                {
                    "source_id": src, "kind": kind,
                    "withheld": ("source", src) in suppressed,
                }
                for src, kind in sources
            ],
        })

    def _api_claim(self, request, cred, scope, caller, params) -> Response:
        return Response(
            body=self._engine.inspect(params["claim_id"], scope, caller=caller)
        )

    def _api_reviews(self, request, cred, scope, caller, params) -> Response:
        from ..storage.repos import ReviewsRepo

        return Response(
            body={"reviews": ReviewsRepo(self._engine.store).list_open(scope_key(scope))}
        )


__all__ = ["WorkbenchApp"]
