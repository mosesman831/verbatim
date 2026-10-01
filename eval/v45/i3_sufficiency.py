"""I3 minimal-sufficient progressive context measurer (SPEC_V4_5 §05;
D05, D06; V45-05.01–05.04).

Two arms over the seeded long-history slice, same ``Store``, same
request budgets:

* **flat** — ordinary v3 delivery at the requested detail tier (L2):
  every admitted group serializes its full body.
* **progressive** — ``pack_mode="sufficiency"``: the coverage/mandatory
  set (required identifiers, condition-bearing, contradiction/safety
  units) ships at L2; remaining groups ship their navigational L0
  projection with caller/scope/purpose/revision/expiry-bound expansion
  refs (V45-05.01/05.02).

Token accounting uses the packer's own estimator —
``ContextPack.tokens`` (≈4 chars/token, the same number budget
enforcement consumed) — reported per pack and totaled, never
re-estimated here.

Measured per topic: flat/progressive tokens, required-identifier
coverage, condition-marker coverage, safety-marker coverage, and
utility — the fraction of the topic's DECLARED expectation markers
(fix, condition, safety) present in the delivered bodies. The declared
expectation set is what the minimal-sufficient contract owes; depth
text is intentionally deferred to expansion and measured separately
(``depth_text_shipped``) rather than hidden.

Non-inferiority (V45-05.04): mean progressive utility ≥ mean flat
utility − 0.01 (one percentage point on the 0..1 expectation fraction).
Success needs BOTH ≥20% token reduction AND non-inferiority — a token
win bought by dropping identifiers/conditions fails by definition
(V45-05.03) and is reported as such.

D06 probe: an expansion ref minted on the progressive arm is re-asked
via the real ``expand_item`` path (roundtrip measured), then the grant
is revoked and the same ref is re-asked — the indistinguishable denial
is the measured outcome. The durable anchor is
``tests/retrieval/test_disclosure_tiers.py`` (expand-MAC revocation/
expiry/forgery tests).

CLI: ``python -m eval.v45.i3_sufficiency --out <dir> [--topics N]``
→ ``i3_sufficiency_report.json`` + ``.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import RecallRequestV3
from verbatim.governance import revoke_grant
from verbatim.retrieval.manifest import (
    PACK_MODE_SUFFICIENCY,
    SUFFICIENCY_RULE,
    required_identifiers,
)
from verbatim.retrieval.v3.recall import expand_item, recall_v3

from . import corpus


def _body(item: Any) -> dict:
    text = item.text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return {}


def _delivered_bodies(result: Any) -> list:
    return [_body(i) for p in result.packs for i in p.items]


def _blob(bodies: list) -> str:
    """All text-bearing fields across delivered bodies (any tier)."""
    parts = []
    for b in bodies:
        for k in ("text", "summary", "title"):
            if b.get(k):
                parts.append(str(b[k]))
    return " ".join(parts)


def _ids_blob(bodies: list) -> str:
    """Identifier surfaces: text fields + every id/locator field — a
    hard identifier 'ships' if the pack names it anywhere."""
    parts = []
    for b in bodies:
        for k in (
            "text", "summary", "title", "claim_id", "object_kind",
            "detail_tier",
        ):
            if b.get(k) is not None:
                parts.append(str(b[k]))
        span = b.get("span") or {}
        for k in ("span_id", "source_id"):
            if span.get(k):
                parts.append(str(span[k]))
    return " ".join(parts)


def _request(topic: dict, scope_id: str, caller: str, max_items: int,
             pack_mode: str = "standard") -> RecallRequestV3:
    return RecallRequestV3(
        query=topic["query"],
        scope_id=scope_id,
        caller_id=caller,
        purpose="recall",
        max_items=max_items,
        pack_mode=pack_mode,
    )


def _topic_outcome(topic: dict, result: Any,
                   expectations: list) -> dict:
    bodies = _delivered_bodies(result)
    blob = _blob(bodies)
    idblob = _ids_blob(bodies)
    marker_hits = {
        name: (marker in blob or marker in idblob)
        for name, marker in expectations
    }
    utility = (
        sum(marker_hits.values()) / len(marker_hits)
        if marker_hits else 0.0
    )
    depth_ids = set(topic["depth_claims"])
    depth_bodies = [
        b for b in bodies if b.get("claim_id") in depth_ids
    ]
    depth_text = sum(1 for b in depth_bodies if b.get("text"))
    expand_refs = [
        b["expand"] for b in bodies if b.get("expand")
    ]
    return {
        "bodies": bodies,
        "marker_hits": marker_hits,
        "utility": utility,
        "tokens": sum(int(p.tokens) for p in result.packs),
        "serialized_bytes": sum(
            int(p.serialized_bytes) for p in result.packs
        ),
        "per_pack_tokens": {
            (p.kind.value if hasattr(p.kind, "value") else str(p.kind)):
                int(p.tokens)
            for p in result.packs
        },
        "items": sum(len(p.items) for p in result.packs),
        "depth_items": len(depth_bodies),
        "depth_text_shipped": depth_text,
        "expand_refs": expand_refs,
        "abstained": bool(result.abstained),
        "coverage": (result.capabilities or {}).get("sufficiency"),
        "warnings": sorted(set(str(w) for w in result.warnings)),
    }


def run_i3(topics: int = 6, max_items: int = 16, depth: int = 5,
           workdir: Optional[str] = None) -> dict:
    """Run flat vs sufficiency arms over the long-history slice."""
    import tempfile

    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="v45_i3_")
    store = corpus.make_store(workdir)
    seeded = corpus.seed_history_slice(store, topics=topics, depth=depth)
    scope_id = seeded["scope_id"]
    caller = seeded["caller"]
    rows: list = []
    all_idents: set = set()
    flat_ident_hits = 0
    prog_ident_hits = 0
    flat_cond = 0
    prog_cond = 0
    expand_ref_for_probe: Optional[str] = None
    expand_roundtrips = 0
    for topic in seeded["topics"]:
        ident = topic["identifier"]
        all_idents.add(ident)
        expectations = [
            ("fix", topic["fix_marker"]),
            ("condition", topic["condition_marker"]),
            ("safety", topic["safety_marker"]),
        ]
        flat = recall_v3(
            store, _request(topic, scope_id, caller, max_items)
        )
        prog = recall_v3(
            store,
            _request(topic, scope_id, caller, max_items,
                     pack_mode=PACK_MODE_SUFFICIENCY),
        )
        flat_o = _topic_outcome(topic, flat, expectations)
        prog_o = _topic_outcome(topic, prog, expectations)
        # identifier coverage — the hard identifier must be named in the
        # delivered pack (any field/tier), V45-05.03.
        flat_cov = ident in _ids_blob(flat_o["bodies"])
        prog_cov = ident in _ids_blob(prog_o["bodies"])
        flat_ident_hits += int(bool(flat_cov))
        prog_ident_hits += int(bool(prog_cov))
        # condition-bearing claim's text present → conditions kept
        flat_cond += int(
            any(b.get("claim_id") == topic["condition_claim"]
                and b.get("text")
                for b in flat_o["bodies"])
        )
        prog_cond += int(
            any(b.get("claim_id") == topic["condition_claim"]
                and b.get("text")
                for b in prog_o["bodies"])
        )
        # expansion roundtrip on the progressive arm (D05's progressive
        # leg, V45-05.01): an L0 stub's bound ref expands to full text —
        # one measured roundtrip per topic, counted honestly.
        if prog_o["expand_refs"]:
            ref = prog_o["expand_refs"][0]
            try:
                expanded = expand_item(
                    store, ref, caller_id=caller, detail_tier="l2"
                )
                ebodies = _delivered_bodies(expanded)
                if any(b.get("text") for b in ebodies):
                    expand_roundtrips += 1
                    if expand_ref_for_probe is None:
                        expand_ref_for_probe = ref
            except VerbatimError:
                pass
        rows.append({
            "topic": topic["topic"],
            "query": topic["query"],
            "identifier": ident,
            "flat": {k: v for k, v in flat_o.items()
                     if k not in ("bodies", "expand_refs")},
            "progressive": {k: v for k, v in prog_o.items()
                            if k not in ("bodies", "expand_refs")},
            "identifier_covered": {
                "flat": bool(flat_cov), "progressive": bool(prog_cov),
            },
            "condition_text_shipped": {
                "flat": bool(flat_o["marker_hits"].get("condition")),
                "progressive": bool(
                    prog_o["marker_hits"].get("condition")
                ),
            },
        })
    n = len(rows)
    flat_tokens = sum(r["flat"]["tokens"] for r in rows)
    prog_tokens = sum(r["progressive"]["tokens"] for r in rows)
    reduction = (
        (flat_tokens - prog_tokens) / flat_tokens if flat_tokens else 0.0
    )
    flat_util = (
        sum(r["flat"]["utility"] for r in rows) / n if n else 0.0
    )
    prog_util = (
        sum(r["progressive"]["utility"] for r in rows) / n if n else 0.0
    )
    margin = 0.01
    non_inferior = prog_util >= flat_util - margin
    # ---- D06 probe: expand ref after revocation must deny ------------
    d06 = {
        "probe": "expand_ref_post_revocation",
        "anchor": "tests/retrieval/test_disclosure_tiers.py",
        "ref_obtained": expand_ref_for_probe is not None,
        "revocation_denied": None,
        "denial_code": None,
    }
    if expand_ref_for_probe is not None:
        gid = None
        with store.read() as conn:
            row = conn.execute(
                "SELECT grant_id FROM grants_v3 WHERE scope_id = ?"
                " AND principal_id = ? AND revoked_us IS NULL"
                " LIMIT 1",
                (scope_id, caller),
            ).fetchone()
            gid = row[0] if row else None
        if gid is not None:
            with store.tx() as conn:
                revoke_grant(conn, gid)
            try:
                expand_item(
                    store, expand_ref_for_probe,
                    caller_id=caller, detail_tier="l2",
                )
                d06["revocation_denied"] = False
            except VerbatimError as exc:
                d06["revocation_denied"] = True
                d06["denial_code"] = str(exc.code)
            except Exception:
                d06["revocation_denied"] = False
        else:
            d06["revocation_denied"] = None
            d06["note"] = "no live grant row found to revoke"
    required = sorted(all_idents)
    report = {
        "experiment": "i3_minimal_sufficient_progressive",
        "spec": {
            "requirements": [
                "V45-05.01", "V45-05.02", "V45-05.03", "V45-05.04",
            ],
            "acceptance": ["D05", "D06"],
        },
        "sample_size": {
            "topics": n,
            "arms": 2,
            "runs": 2 * n,
            "depth_claims_per_topic": depth,
            "note": "registered long-history slice; n is the realized"
                    " topic count actually executed",
        },
        "coverage_rule": SUFFICIENCY_RULE,
        "token_estimator": "approx_chars_per_token:4 "
                           "(ContextPack.tokens — the packer's own "
                           "estimate, not re-derived here)",
        "matched_bound": {
            "max_items": max_items,
            "note": "identical RecallRequestV3 budgets on both arms",
        },
        "definitions": {
            "utility": (
                "per-topic fraction of DECLARED expectation markers"
                " (fix, condition, safety) present in delivered bodies;"
                " depth text is deferred to expansion by design and"
                " measured separately"
            ),
            "non_inferiority": (
                f"mean progressive utility ≥ mean flat utility −"
                f" {margin} (one percentage point, V45-05.04)"
            ),
        },
        "topics": rows,
        "tokens": {
            "flat_total": flat_tokens,
            "progressive_total": prog_tokens,
            "reduction_ratio": round(reduction, 6),
            "reduction_pct": round(100.0 * reduction, 2),
            "per_topic": [
                {
                    "topic": r["topic"],
                    "flat": r["flat"]["tokens"],
                    "progressive": r["progressive"]["tokens"],
                    "flat_per_pack": r["flat"]["per_pack_tokens"],
                    "progressive_per_pack":
                        r["progressive"]["per_pack_tokens"],
                }
                for r in rows
            ],
        },
        "coverage": {
            "required_identifiers": required,
            "flat_covered": flat_ident_hits,
            "progressive_covered": prog_ident_hits,
            "condition_text_topics": {
                "flat": flat_cond, "progressive": prog_cond,
            },
            "per_topic_sufficiency": [
                r["progressive"]["coverage"] for r in rows
            ],
        },
        "utility": {
            "flat_mean": round(flat_util, 6),
            "progressive_mean": round(prog_util, 6),
            "delta": round(prog_util - flat_util, 6),
            "margin": margin,
            "non_inferior": non_inferior,
        },
        "expansion": {
            "roundtrips_measured": expand_roundtrips,
            "d06": d06,
        },
        "d05": {
            "token_reduction_met": reduction >= 0.20,
            "non_inferior_met": non_inferior,
            "identifiers_kept": prog_ident_hits >= flat_ident_hits,
            "conditions_kept": prog_cond >= flat_cond,
            "measured": (
                f"tokens flat={flat_tokens} progressive={prog_tokens}"
                f" (−{round(100.0 * reduction, 2)}%), utility"
                f" {flat_util:.3f}→{prog_util:.3f}"
            ),
        },
    }
    report["met"] = bool(
        report["d05"]["token_reduction_met"]
        and report["d05"]["non_inferior_met"]
        and report["d05"]["identifiers_kept"]
        and report["d05"]["conditions_kept"]
    )
    store.close()
    return report


def _md(report: dict) -> str:
    t = report["tokens"]
    u = report["utility"]
    d = report["d05"]
    lines = [
        "# I3 — minimal-sufficient progressive context (D05/D06)",
        "",
        f"- sample: {report['sample_size']['topics']} topics × 2 arms"
        f" ({report['sample_size']['runs']} runs),"
        f" depth={report['sample_size']['depth_claims_per_topic']}/topic",
        f"- tokens flat/progressive: **{t['flat_total']} /"
        f" {t['progressive_total']}** (−{t['reduction_pct']}%)",
        f"- utility flat→progressive: {u['flat_mean']} →"
        f" {u['progressive_mean']} (Δ{u['delta']}, margin"
        f" {u['margin']}) → non-inferior: {u['non_inferior']}",
        f"- identifiers kept: {d['identifiers_kept']}, conditions kept:"
        f" {d['conditions_kept']}",
        f"- D05 met: {d['token_reduction_met'] and d['non_inferior_met']}"
        f" → overall met: {report['met']}",
        f"- expansion roundtrips: {report['expansion']['roundtrips_measured']};"
        f" D06 revoke-denied: {report['expansion']['d06']['revocation_denied']}",
        "",
        "| topic | flat tok | prog tok | flat util | prog util |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in report["topics"]:
        lines.append(
            f"| {r['topic']} | {r['flat']['tokens']} |"
            f" {r['progressive']['tokens']} |"
            f" {r['flat']['utility']:.2f} |"
            f" {r['progressive']['utility']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "i3_sufficiency_report.json")
    mpath = os.path.join(out_dir, "i3_sufficiency_report.md")
    with open(jpath, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
    with open(mpath, "w") as fh:
        fh.write(_md(report))
    return {"json": jpath, "md": mpath}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="I3 minimal-sufficient progressive measurer (D05/D06)"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--topics", type=int, default=6)
    ap.add_argument("--max-items", type=int, default=16)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)
    report = run_i3(topics=args.topics, max_items=args.max_items,
                    depth=args.depth, workdir=args.workdir)
    if args.out:
        paths = write_reports(report, args.out)
        print(f"wrote {paths['json']}")
    else:
        print(json.dumps(report["d05"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
