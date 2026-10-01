"""I1 counterevidence-first ablation runner (SPEC_V4_5 §03; D01, D02;
V45-03.01–03.04).

Two arms over the seeded stale-state slice, same ``Store``, same
``RecallRequestV3`` budgets — the matched-token bound (V45-03.04):

* **topk** — ordinary v3 delivery (``manifest=None``). Greedy group
  order; a crowded-out conflict group leaves a generic
  ``budget_omitted_group`` warning that names nothing.
* **manifest** — ``manifest="counterevidence_first"``: contrary groups
  admit first under the same bound, and ``capabilities["manifest"]``
  records every contrary ref — included, or omitted with its reason —
  plus presence-level insufficiency labels for what never survived
  admission.

Measured, not asserted: per topic the runner parses the delivered
items' serialized bodies and records whether the answer *asserts* the
topic state while *disclosing no contrary evidence* — the false-current
definition below. All numbers in the report come from the returned
results; a topic that behaves unexpectedly is still counted, with its
delivered ids disclosed.

``false_current`` (declared): a delivered item asserts the topic's
state marker AND (a) no contrary-side claim is delivered, AND (b) no
contrary-specific signal rides the result (a contrary ref on the
manifest, or a ``contrary_*`` warning/label). Generic omission warnings
(``budget_omitted_group``, ``no_authorized_evidence``) are recorded but
do NOT count — they name no counterevidence, which is exactly the
silent-omission gap the manifest closes.

CLI: ``python -m eval.v45.i1_manifest --out <dir> [--topics N]
[--max-items 2]`` → ``i1_manifest_report.json`` + ``.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

from verbatim.core.types_v3 import RecallRequestV3
from verbatim.retrieval.manifest import (
    COUNTEREVIDENCE_FIRST,
    MANIFEST_REVISION,
)
from verbatim.retrieval.v3.recall import recall_v3

from . import corpus


def _body(item: Any) -> dict:
    """The serialized JSON body of one delivered item."""
    text = item.text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return {}


def _delivered(result: Any) -> list:
    """(body, handle) for every item in the result's packs."""
    return [
        (_body(i), i.handle)
        for p in result.packs
        for i in p.items
    ]


def _request(topic: dict, scope_id: str, caller: str, max_items: int,
             manifest: Optional[str] = None) -> RecallRequestV3:
    return RecallRequestV3(
        query=topic["query"],
        scope_id=scope_id,
        caller_id=caller,
        purpose="recall",
        max_items=max_items,
        manifest=manifest,
    )


def _arm_outcome(topic: dict, result: Any) -> dict:
    """One arm's measured outcome on one topic."""
    delivered = _delivered(result)
    bodies = [b for b, _h in delivered]
    delivered_ids = {h.object_id for _b, h in delivered}
    text_blob = " ".join(
        str(b.get("text") or "") + " " + str(b.get("summary") or "")
        + " " + str(b.get("title") or "")
        for b in bodies
    )
    asserted = topic["assertion_marker"] in text_blob
    contrary_ids = set(topic["conflict_members"])
    contrary_delivered = sorted(delivered_ids & contrary_ids)
    manifest_doc = (result.capabilities or {}).get("manifest") or {}
    labels = list(manifest_doc.get("insufficiency_labels") or ())
    contrary_labels = [
        w for w in result.warnings
        if str(w).startswith("contrary_")
    ] + [
        l for l in labels if str(l).startswith("contrary_")
    ]
    contrary_refs = manifest_doc.get("contrary_refs") or []
    contrary_included_refs = [
        r["object_id"] for r in contrary_refs
        if r.get("disposition") == "included"
    ]
    contrary_omitted_refs = [
        {"object_id": r["object_id"], "reason": r.get("reason")}
        for r in contrary_refs
        if r.get("disposition") == "omitted"
    ]
    contrary_disclosed = bool(
        contrary_delivered or contrary_included_refs or contrary_labels
    )
    ident = topic.get("identifier") or ""
    ident_hit = bool(
        ident
        and (
            ident in text_blob
            or any(ident in str(v) for b in bodies for v in b.values())
        )
    )
    return {
        "delivered_ids": sorted(delivered_ids),
        "asserted": asserted,
        "contrary_delivered": contrary_delivered,
        "contrary_included_refs": contrary_included_refs,
        "contrary_omitted_refs": contrary_omitted_refs,
        "contrary_labels": sorted(set(contrary_labels)),
        "contrary_disclosed": contrary_disclosed,
        "false_current": bool(asserted and not contrary_disclosed),
        "identifier_hit": ident_hit,
        "tokens": sum(int(p.tokens) for p in result.packs),
        "abstained": bool(result.abstained),
        "warnings": sorted(set(str(w) for w in result.warnings)),
    }


def run_i1(topics: int = 8, max_items: int = 2,
           workdir: Optional[str] = None) -> dict:
    """Run both arms over the seeded stale-state slice.

    ``max_items`` bounds the shared item budget — the registered slice's
    pressure point: the state-asserting singleton plus a filler fill two
    slots; the atomic conflict group (2 items) can then only ship under
    the manifest arm's contrary-first admission.
    """
    import tempfile

    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="v45_i1_")
    store = corpus.make_store(workdir)
    try:
        seeded = corpus.seed_stale_slice(store, topics=topics)
    finally:
        pass
    scope_id = seeded["scope_id"]
    caller = seeded["caller"]
    rows: list = []
    for topic in seeded["topics"]:
        base = recall_v3(
            store, _request(topic, scope_id, caller, max_items)
        )
        man = recall_v3(
            store,
            _request(topic, scope_id, caller, max_items,
                     manifest=COUNTEREVIDENCE_FIRST),
        )
        base_o = _arm_outcome(topic, base)
        man_o = _arm_outcome(topic, man)
        man_doc = (man.capabilities or {}).get("manifest") or {}
        rows.append({
            "topic": topic["topic"],
            "query": topic["query"],
            "arms": {"topk": base_o, "manifest": man_o},
            "manifest_producer": man_doc.get("producer_kind"),
            "manifest_labels": man_doc.get("insufficiency_labels"),
            "manifest_coverage": man_doc.get("coverage"),
        })
    n = len(rows)
    base_fc = sum(1 for r in rows if r["arms"]["topk"]["false_current"])
    man_fc = sum(1 for r in rows if r["arms"]["manifest"]["false_current"])
    man_incl = sum(
        1 for r in rows
        if r["arms"]["manifest"]["contrary_included_refs"]
        or r["arms"]["manifest"]["contrary_delivered"]
    )
    man_labeled = sum(
        1 for r in rows if r["arms"]["manifest"]["contrary_labels"]
    )
    base_ident = sum(
        1 for r in rows if r["arms"]["topk"]["identifier_hit"]
    )
    man_ident = sum(
        1 for r in rows if r["arms"]["manifest"]["identifier_hit"]
    )
    report = {
        "experiment": "i1_counterevidence_first",
        "spec": {
            "requirements": [
                "V45-03.01", "V45-03.02", "V45-03.03", "V45-03.04",
            ],
            "acceptance": ["D01", "D02"],
            "manifest_revision": MANIFEST_REVISION,
        },
        "sample_size": {
            "topics": n,
            "arms": 2,
            "runs": 2 * n,
            "note": "registered stale-state slice; n is the realized"
                    " topic count actually executed",
        },
        "matched_bound": {
            "max_items": max_items,
            "note": "identical RecallRequestV3 budgets on both arms —"
                    " the matched serialized-token bound (V45-03.04);"
                    " realized token totals reported per arm",
        },
        "definitions": {
            "false_current": (
                "a delivered item asserts the topic's state marker AND"
                " no contrary-side claim is delivered AND no"
                " contrary-specific signal rides the result (manifest"
                " contrary ref or contrary_* label). Generic omission"
                " warnings are recorded but never count as disclosure —"
                " they name no counterevidence."
            ),
            "contrary_disclosed": (
                "a conflict-side claim delivered, or a manifest"
                " contrary ref included, or a contrary_* label present"
            ),
        },
        "topics": rows,
        "totals": {
            "topk_false_current": base_fc,
            "manifest_false_current": man_fc,
            "manifest_contrary_included": man_incl,
            "manifest_contrary_labeled": man_labeled,
            "identifier_hits": {"topk": base_ident, "manifest": man_ident},
            "tokens": {
                "topk": sum(r["arms"]["topk"]["tokens"] for r in rows),
                "manifest": sum(
                    r["arms"]["manifest"]["tokens"] for r in rows
                ),
            },
        },
        "d01": {
            "measured": (
                "every manifest delivery carries the inspectable"
                " manifest: supporting/contrary refs, as_of, coverage,"
                " producer_kind, obligations, insufficiency labels"
            ),
            "topics_with_manifest": sum(
                1 for r in rows if r["manifest_producer"]
            ),
            "topics_contrary_accounted": sum(
                1 for r in rows
                if r["arms"]["manifest"]["contrary_included_refs"]
                or r["arms"]["manifest"]["contrary_omitted_refs"]
                or r["arms"]["manifest"]["contrary_labels"]
            ),
        },
        "d02": {
            "false_current_fell": man_fc < base_fc,
            "topk_false_current": base_fc,
            "manifest_false_current": man_fc,
            "exact_identifier_hits_kept": man_ident >= base_ident,
            "measured": (
                f"top-k false-current {base_fc}/{n} vs manifest"
                f" {man_fc}/{n} on the seeded stale-state slice"
            ),
        },
    }
    report["met"] = bool(
        report["d02"]["false_current_fell"]
        and report["d02"]["exact_identifier_hits_kept"]
    )
    store.close()
    return report


def _md(report: dict) -> str:
    t = report["totals"]
    d = report["d02"]
    lines = [
        "# I1 — counterevidence-first manifest ablation (D01/D02)",
        "",
        f"- sample: {report['sample_size']['topics']} topics × 2 arms"
        f" ({report['sample_size']['runs']} runs),"
        f" max_items={report['matched_bound']['max_items']}",
        f"- false-current (top-k): **{t['topk_false_current']}**",
        f"- false-current (manifest): **{t['manifest_false_current']}**",
        f"- manifest contrary included: {t['manifest_contrary_included']},"
        f" contrary labeled: {t['manifest_contrary_labeled']}",
        f"- identifier hits top-k/manifest:"
        f" {t['identifier_hits']['topk']}/"
        f"{t['identifier_hits']['manifest']}",
        f"- tokens top-k/manifest:"
        f" {t['tokens']['topk']}/{t['tokens']['manifest']}",
        f"- D02 fell: {d['false_current_fell']}, identifier kept:"
        f" {d['exact_identifier_hits_kept']} → met: {report['met']}",
        "",
        "| topic | top-k fc | manifest fc | contrary refs (incl/omit) |"
        " labels |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in report["topics"]:
        a = r["arms"]
        labels = ";".join(a["manifest"]["contrary_labels"]) or "-"
        incl = len(a["manifest"]["contrary_included_refs"])
        omit = len(a["manifest"]["contrary_omitted_refs"])
        lines.append(
            f"| {r['topic']} | {a['topk']['false_current']} |"
            f" {a['manifest']['false_current']} | {incl}/{omit} |"
            f" {labels} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "i1_manifest_report.json")
    mpath = os.path.join(out_dir, "i1_manifest_report.md")
    with open(jpath, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
    with open(mpath, "w") as fh:
        fh.write(_md(report))
    return {"json": jpath, "md": mpath}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="I1 counterevidence-first ablation (D01/D02)"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--topics", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=2)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)
    report = run_i1(topics=args.topics, max_items=args.max_items,
                    workdir=args.workdir)
    if args.out:
        paths = write_reports(report, args.out)
        print(f"wrote {paths['json']}")
    else:
        print(json.dumps(report["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
