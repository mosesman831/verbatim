"""G3 deep twin corpus — detector-level false-supersession measurement.

The G3 gate targets a false-supersession rate <0.01; the rule-of-three
upper-95% bound needs ~300 clean near-miss twins (3/300 ≈ 0.01). The
hand-authored corpus has 4 — this module generates a seeded ~300-case
synthetic twin corpus across declared near-miss categories and runs the
REAL admission-time detectors over each pair in a real Store:

- ``propose_retirement_supersessions`` (explicit retirement markers)
- ``propose_unstructured_relations`` (unstructured contradiction pairs)

A "false positive" is ANY review proposal (supersede or dispute) created
for a twin whose text makes no true retirement/conflict assertion. A
companion set of true pairs verifies the detectors still fire — a
corpus that everything passes proves nothing.

Honest scope: cases are template-generated near-misses over a rotating
identifier pool — synthetic by construction, disclosed as such. The
measurement exercises the same code paths admission uses (real Store,
real claims, real detector calls); it is detector-level, not end-to-end.

Run:
    python -m eval.v3.g3_twins --out eval/v3/g3_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import tempfile
from typing import Any, Optional

from verbatim.evidence.relations import propose_unstructured_relations
from verbatim.evidence.supersession import propose_retirement_supersessions
from verbatim.storage.store import Store

# ---------------------------------------------------------------------------
# identifier pool — (identifier, action phrase) pairs spanning commands,
# flags, paths, endpoints, versions, daemons, services
# ---------------------------------------------------------------------------

IDENTS: list[tuple[str, str]] = [
    ("deploy.sh", "deploy the service"),
    ("buildctl", "build the workspace"),
    ("swagger-gen", "regenerate the API types"),
    ("pytest-legacy", "run the integration suite"),
    ("flake8", "lint the service code"),
    ("protoc-legacy", "compile the proto stubs"),
    ("/v1/users", "query the users endpoint"),
    ("syncv1", "handle the calendar sync"),
    ("keyd", "rotate the auth tokens"),
    ("cachewarmer", "warm the edge cache"),
    ("--force", "force the rebuild"),
    ("config-v1.yaml", "load the service config"),
    ("node-12", "pin the runtime"),
    ("pip-legacy", "install the packages"),
    ("cron.daily", "schedule the cleanup"),
    ("redis-4", "back the session cache"),
    ("webhook-v1", "deliver the notifications"),
    ("grpc-1", "serve the internal RPCs"),
    ("authd", "authenticate the agents"),
    ("jobrunner", "drain the job queue"),
    ("rest-v2", "expose the public API"),
    ("xmlbridge", "translate the payloads"),
    ("oauth1", "sign the requests"),
    ("legacy-ui", "render the dashboard"),
    ("mysql-5", "host the analytics db"),
    ("solr-4", "index the catalog"),
    ("batch-etl", "run the nightly pipeline"),
    ("ftp-upload", "ship the exports"),
    ("soap-gw", "bridge the gateway"),
    ("tokend", "refresh the session tokens"),
]

# ---------------------------------------------------------------------------
# twin templates — near-miss phrasings that carry NO true retirement or
# conflict assertion. Each is a category the detector must not flag.
# {i} = identifier, {a} = action phrase, {r} = replacement name
# ---------------------------------------------------------------------------

TWINS: list[tuple[str, str]] = [
    # hedged questions / indirect interrogatives
    ("hedged_question",
     "Someone asked whether `{i}` was retired — it wasn't, just discouraged."),
    ("hedged_question",
     "Is `{i}` retired? Asking because the runbook still says to {a} with it."),
    ("hedged_question",
     "Can someone confirm whether `{i}` was removed from the toolchain?"),
    # hearsay — attribution markers, not assertions
    ("hearsay",
     "`{i}` was reportedly retired, but I haven't confirmed it."),
    ("hearsay",
     "Rumor says `{i}` might be deprecated soon — unverified."),
    ("hearsay",
     "Apparently `{i}` was removed, though the docs still list it."),
    # explicit negation
    ("negated",
     "`{i}` was not retired; the team kept it after the review."),
    ("negated",
     "`{i}` was never deprecated — that notice was about a different tool."),
    # future intent — planned, not done
    ("future",
     "`{i}` will be retired next quarter; keep using it to {a} for now."),
    ("future",
     "The plan is to deprecate `{i}` once the migration finishes."),
    ("future",
     "We're going to replace `{i}` eventually — no date yet."),
    ("future",
     "At some point we want to remove `{i}` and {a} differently."),
    # conditional — hypothetical, not declarative
    ("conditional",
     "If `{i}` is retired, we'll have to {a} some other way."),
    ("conditional",
     "When `{i}` is deprecated we can switch; until then, keep it."),
    ("conditional",
     "Unless `{i}` gets removed, the current setup stays."),
    # noun forms — mentions of retirement as a topic, not an event
    ("noun_form",
     "The retirement of `{i}` is still under discussion."),
    ("noun_form",
     "`{i}` deprecation timeline: TBD pending platform review."),
    ("noun_form",
     "There's a deprecation notice drafted for `{i}` but nothing's final."),
    # cancelled retirement — retired then reinstated
    ("reinstated",
     "`{i}` was retired last year but reinstated in March after the outage."),
    ("reinstated",
     "`{i}` was deprecated briefly, then restored — it's fine to {a} with it."),
    # imperative/request — asking for retirement, not declaring it
    ("imperative",
     "Please retire `{i}` when you get a chance — ticket's still open."),
    ("imperative",
     "We should deprecate `{i}` soon; it's caused two incidents."),
    ("imperative",
     "Let's replace `{i}` next sprint."),
    # counterparty corroborates — both sides agree it ended (special-cased
    # in the runner: the OLD claim also declares the retirement)
    ("corroborating",
     "`{i}` was retired in June — the migration guide covers how to {a} now."),
    # identifier family confusion — similar name, different referent
    ("family_name",
     "`{i}` was retired, but `{i}-ng` is the supported successor — "
     "don't confuse them."),
    # retirement of a DIFFERENT identifier in the same text
    ("other_retired",
     "We removed `legacy-{i}` ages ago; `{i}` itself still works to {a}."),
    # imperative negation — "do not X" instructs the reader; it never
    # asserts a state contradiction with the counterparty's claim
    ("imperative_neg",
     "Quick fix when you {a}: run `{i} --force` — do not tell the user "
     "about this step."),
    ("imperative_neg",
     "To {a} faster, use `{i}` — don't tell anyone I said so."),
]

#: categories where the counterparty (old) claim is constructed differently
_SPECIAL_COUNTERPARTY = {
    "corroborating": "`{i}` was retired — do not {a} with it anymore.",
    "family_name": None,  # default: standard live claim about {i}-ng
    "other_retired": None,  # default: live claim about {i}
}

#: true-positive templates — real retirement assertions the detector
#: SHOULD catch. Rotated over the ident pool as the firing guard.
TRUES: list[tuple[str, str]] = [
    ("subject", "`{i}` was retired — use `{r}` to {a} instead."),
    ("subject", "`{i}` is deprecated; `{r}` handles how you {a} now."),
    ("subject",
     "`{i}` has been decommissioned — `{r}` took over the work to {a}."),
    ("subject", "`{i}` was replaced by `{r}` for anyone trying to {a}."),
    ("object", "We removed `{i}` — `{r}` is the way to {a} now."),
    ("object", "We retired `{i}`; going forward, {a} with `{r}`."),
    ("object", "The team replaced `{i}` with `{r}` — that's how you {a}."),
]

REPLACEMENTS = ["v2", "ng", "ctl", "x", "next", "prime", "2", "tool"]


def gen_cases(seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic case list: (twins, true_pairs)."""
    rng = random.Random(seed)
    twins: list[dict] = []
    n = 0
    # cycle idents × templates deterministically until ~460 twins —
    # rule-of-three upper-95% at 0 flags: 3/460 ≈ 0.0065 < 0.01
    order = list(range(len(IDENTS)))
    while len(twins) < 460:
        rng.shuffle(order)
        for idx in order:
            if len(twins) >= 460:
                break
            ident, action = IDENTS[idx]
            cat, tmpl = TWINS[n % len(TWINS)]
            n += 1
            repl = f"{ident.split('-')[0].split('.')[0]}-{rng.choice(REPLACEMENTS)}"
            new_text = tmpl.format(i=ident, a=action, r=repl)
            counterparty = (
                f"Use `{ident}` to {action}."
                if cat not in _SPECIAL_COUNTERPARTY
                or _SPECIAL_COUNTERPARTY[cat] is None
                else _SPECIAL_COUNTERPARTY[cat].format(i=ident, a=action, r=repl)
            )
            if cat == "family_name":
                counterparty = f"Use `{ident}-ng` to {action}."
            twins.append({
                "id": f"twin-{len(twins):03d}",
                "category": cat,
                "ident": ident,
                "counterparty_text": counterparty,
                "new_text": new_text,
                "expect": "none",
            })
    trues: list[dict] = []
    for i, (ident, action) in enumerate(IDENTS):
        cat, tmpl = TRUES[i % len(TRUES)]
        repl = f"{ident.split('-')[0].split('.')[0]}-{REPLACEMENTS[i % len(REPLACEMENTS)]}"
        trues.append({
            "id": f"true-{i:03d}",
            "category": f"true_{cat}",
            "ident": ident,
            "counterparty_text": f"Old runbook: {action} with `{ident}`.",
            "new_text": tmpl.format(i=ident, a=action, r=repl),
            "expect": "applied",
        })
    return twins, trues


# ---------------------------------------------------------------------------
# store-level runner — real Store, real claims, real detector calls
# ---------------------------------------------------------------------------

def _seed_claim(store, conn, claim_id, scope_id, source_id, span_id, text,
                state="active"):
    payload = text.encode("utf-8")
    conn.execute(
        "INSERT INTO sources(source_id,origin,source_kind,scope_id,"
        "speaker_id,created_us) VALUES(?,?,?,?,?,1)",
        (source_id, "test", "user_message", scope_id, "u1"),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id,revision,payload,"
        "payload_hmac,event_us,captured_us,timezone,provenance,metadata_json)"
        " VALUES(?,1,?,?,1,1,'UTC','direct_user','{}')",
        (source_id, payload, store.hmac(payload)),
    )
    conn.execute(
        "INSERT INTO spans(span_id,source_id,revision,start_byte,end_byte,"
        "excerpt_hmac,harvester_version) VALUES(?,?,?,?,?,?,'t')",
        (span_id, source_id, 1, 0, len(payload), store.hmac(payload)),
    )
    conn.execute(
        "INSERT INTO claims(claim_id,scope_id,subject_id,predicate,"
        "created_event,row_version) VALUES(?,?,NULL,NULL,1,1)",
        (claim_id, scope_id),
    )
    conn.execute(
        "INSERT INTO claim_revisions(claim_id,revision,state,"
        "recorded_from,recorded_until,interpretation_status)"
        " VALUES(?,1,?,1,NULL,'unstructured')",
        (claim_id, state),
    )
    conn.execute(
        "INSERT INTO claim_evidence(claim_id,revision,span_id,"
        "evidence_role,family_id) VALUES(?,1,?,'primary',NULL)",
        (claim_id, span_id),
    )


def run_case(store: Store, case: dict, scope_id: str) -> tuple[int, int]:
    """Seed the pair in its own scope, run both detectors, return
    (supersession flags, relation flags) — any nonzero value flags."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes(scope_id,profile_id,principal_id,"
            "workspace_id,conversation_id,visibility,acl_revision)"
            " VALUES(?,'prof','p1','ws','c1','conversation',0)",
            (scope_id,),
        )
        _seed_claim(store, conn, f"{scope_id}-old", scope_id,
                    f"{scope_id}-src1", f"{scope_id}-sp1",
                    case["counterparty_text"])
        _seed_claim(store, conn, f"{scope_id}-new", scope_id,
                    f"{scope_id}-src2", f"{scope_id}-sp2",
                    case["new_text"])
    sup = propose_retirement_supersessions(store, f"{scope_id}-new", scope_id)
    rel = propose_unstructured_relations(store, f"{scope_id}-new", scope_id)
    return len(sup), len(rel)


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score upper bound for k successes in n trials."""
    if n == 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center + margin) / denom


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="eval/v3/g3_report.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    twins, trues = gen_cases(args.seed)
    tmp = tempfile.mkdtemp(prefix="g3-twins-")
    store = Store.create(os.path.join(tmp, "g3.db"))

    fp_cases: list[dict] = []
    by_cat: dict[str, dict[str, int]] = {}
    for case in twins:
        sup, rel = run_case(store, case, case["id"])
        cat = by_cat.setdefault(case["category"], {"cases": 0, "flagged": 0})
        cat["cases"] += 1
        if sup + rel:
            cat["flagged"] += 1
            fp_cases.append({**case, "supersede_flags": sup,
                             "relation_flags": rel})

    true_missed: list[dict] = []
    for case in trues:
        sup, rel = run_case(store, case, case["id"])
        if not sup + rel:
            true_missed.append(case)

    n, k = len(twins), len(fp_cases)
    report = {
        "artifact": "g3_deep_twin_corpus",
        "seed": args.seed,
        "detectors": [
            "propose_retirement_supersessions",
            "propose_unstructured_relations",
        ],
        "twins": {
            "total": n,
            "false_positives": k,
            "by_category": by_cat,
            "flagged_cases": fp_cases,
        },
        "true_pairs": {
            "total": len(trues),
            "detected": len(trues) - len(true_missed),
            "missed": [c["id"] for c in true_missed],
        },
        "bound": {
            "rule_of_three_upper_95": round(3 / n, 4) if k == 0 else None,
            "wilson_upper_95": round(_wilson_upper(k, n), 4),
            "target": "<0.01 (G3)",
        },
        "notes": (
            "Synthetic template-generated near-miss corpus over a "
            f"{len(IDENTS)}-identifier pool; detector-level measurement "
            "through the real admission-time proposal path (real Store, "
            "real claims, real detector calls). Claims seeded as "
            "unstructured so both the retirement and the V3-18.03 "
            "unstructured detectors fire — the real worst case. Encoder "
            "corroboration not provisioned here (deterministic layer "
            "only); encoder path covered in tests/evidence."
        ),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"twins: {k}/{n} flagged; true pairs detected: "
          f"{len(trues) - len(true_missed)}/{len(trues)}")
    print(f"upper-95% bound: {report['bound']}")
    print(f"report: {args.out}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
