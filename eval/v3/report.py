"""Markdown report for the v3 eval harness (SPEC_V3 §54, §56).

Layout contract:

* every number rendered is a **measurement on this corpus and
  configuration** — the header states both and the comparison table
  repeats it;
* spec targets live in the *Gates vs measured* section only, cited as
  targets — a measured value is never silently upgraded to a gate pass;
* missing capabilities render as ``capability unavailable`` rows, never
  absent rows;
* ``None`` metrics render ``n/a (no denominator)`` — an empty
  denominator is not a 0 and not a 1.
"""

from __future__ import annotations

import datetime
from typing import Any, Iterable, List, Optional, Sequence

from .baselines import SuiteRun
from .corpus import Corpus, corpus_stats


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


#: The detector surface the committed G3 deep-twin artifact declares it
#: measured — detector function name → the module that must still define
#: it.  Mirrors ``g3_twins.py``'s report writer; an artifact naming a
#: different set, or naming functions the tree no longer has, predates
#: the current detector surface and is flagged in the gate row.
_G3_DETECTOR_MODULES: dict[str, str] = {
    "propose_retirement_supersessions": "verbatim.evidence.supersession",
    "propose_unstructured_relations": "verbatim.evidence.relations",
}


def _g3_artifact_provenance(path: Any, rep: dict) -> dict:
    """Provenance binding for a committed ``g3_report.json`` artifact.

    The artifact schema carries ``seed`` and ``detectors`` but no
    ``generated_us``/rules-revision field — so the honest generation
    timestamp is the file's own mtime (labeled as such), and staleness is
    assessed by comparing the artifact's declared detector names against
    the detector functions the tree currently defines, plus whether the
    detector sources changed after the artifact was written.
    """
    import datetime
    import importlib
    import pathlib

    prov: dict[str, Any] = {
        "seed": rep.get("seed"),
        "detectors": list(rep.get("detectors") or []),
        "generated": None,
        "stale_notes": [],
    }
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if mtime is not None:
        prov["generated"] = datetime.datetime.fromtimestamp(
            mtime, datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%SZ")
    declared = prov["detectors"]
    if sorted(declared) != sorted(_G3_DETECTOR_MODULES):
        prov["stale_notes"].append(
            f"declared detectors {sorted(declared)} differ from the "
            f"current set {sorted(_G3_DETECTOR_MODULES)}"
        )
    for name, mod_name in _G3_DETECTOR_MODULES.items():
        if name not in declared:
            continue
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            prov["stale_notes"].append(
                f"detector module {mod_name} unimportable "
                f"({type(exc).__name__})"
            )
            continue
        if not hasattr(mod, name):
            prov["stale_notes"].append(
                f"detector {name} no longer exists in {mod_name}"
            )
            continue
        if mtime is not None:
            try:
                src_mtime = pathlib.Path(mod.__file__).stat().st_mtime
            except (OSError, TypeError):
                continue
            if src_mtime > mtime:
                prov["stale_notes"].append(
                    f"{mod_name} modified after artifact was generated — "
                    "bound may predate current detector logic"
                )
    return prov


def _deep_twin_bound() -> Optional[dict]:
    """The G3 deep twin corpus report, when present — the real bound
    measurement behind the in-corpus twin smoke check. Never fabricated:
    absent file means the gate row reports the corpus slice only.

    The returned dict gains a ``_provenance`` key binding the artifact to
    the code that produced it (declared seed/detectors, file generation
    time, staleness notes) — the gate row renders it so a stale committed
    artifact cannot be read as a fresh measurement.
    """
    import json
    import pathlib

    path = pathlib.Path(__file__).with_name("g3_report.json")
    try:
        rep = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    rep["_provenance"] = _g3_artifact_provenance(path, rep)
    return rep


def _metric(run: SuiteRun, name: str) -> Any:
    m = run.metrics.get(name)
    if isinstance(m, dict):
        return m.get("value")
    return m


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

_RETRIEVAL_ROWS = [
    ("evidence_recall_at_k", "evidence recall@k"),
    ("precision_at_k", "precision@k"),
    ("hit_rate", "full-set hit rate"),
    ("abstain_precision", "abstain precision"),
    ("abstain_recall", "abstain recall"),
    ("spurious_answer_rate", "spurious answer rate"),
    ("grounded_support_rate", "grounded support rate"),
    ("attack_retrieval_rate", "poison exposure rate"),
    ("disclosure_violations", "disclosure violations"),
]

_GROUNDING_ROWS = [
    ("grounded_support_rate", "grounded support rate"),
    ("fabricated_item_rate", "fabricated item rate"),
    ("abstain_recall", "abstain recall"),
    ("spurious_answer_rate", "spurious answer rate"),
    ("evidence_recall_at_k", "evidence recall@k"),
]

_SECURITY_ROWS = [
    ("screening_flag_rate", "screen flag rate (poisoned)"),
    ("benign_instructional_pass_rate", "benign pass rate"),
    ("attack_retrieval_rate", "poison exposure (pre-quarantine)"),
    ("poisoning_block_rate", "poisoning block rate (pre-quarantine)"),
    ("post_quarantine_exposure", "poison exposure (post-quarantine)"),
    ("disclosure_violations", "disclosure violations"),
    ("grounded_support_rate", "grounded support rate"),
]


def _section_header(title: str) -> List[str]:
    return ["", f"## {title}", ""]


def _capability_table(runs: Sequence[SuiteRun]) -> List[str]:
    lines = _section_header("Capability report")
    lines.append("| capability | status | detail |")
    lines.append("|---|---|---|")
    seen: dict[str, Any] = {}
    order: List[str] = []
    for run in runs:
        for name, cap in run.capabilities.items():
            if name not in seen:
                seen[name] = cap
                order.append(name)
    for name in order:
        cap = seen[name]
        status = "available" if cap.available else "**capability unavailable**"
        lines.append(f"| `{name}` | {status} | {cap.detail} |")
    for run in runs:
        for lane in run.unavailable_lanes:
            lines.append(
                f"| `{run.suite}` arm `{lane}` | **capability unavailable** "
                f"| reported per-record; records retained in denominators |"
            )
    return lines


def _suite_table(run: SuiteRun, rows: list[tuple[str, str]]) -> List[str]:
    lines = [f"### suite `{run.suite}` — baseline `{run.baseline}`", ""]
    lines.append("| metric | measured |")
    lines.append("|---|---|")
    for key, label in rows:
        lines.append(f"| {label} | {_fmt(_metric(run, key))} |")
    lines.append(f"| scored records | {len(run.records)} |")
    lines.append(f"| errors | {run.errors} |")
    if run.unavailable_lanes:
        lines.append(
            f"| unavailable lanes | {', '.join(run.unavailable_lanes)} |"
        )
    return lines


def _tasks_table(run: SuiteRun) -> List[str]:
    lines = [f"### suite `tasks` — baseline `{run.baseline}`", ""]
    pd = run.metrics.get("paired_delta", {})
    mode = _metric(run, "execution_mode") or "paired"
    lines.append("| metric | measured |")
    lines.append("|---|---|")
    lines.append(f"| execution mode | {mode} |")
    lines.append(f"| paired trials | {_fmt(pd.get('trials'))} |")
    lines.append(f"| memory-arm success | {_fmt(pd.get('memory_rate'))} |")
    lines.append(f"| control-arm success | {_fmt(pd.get('control_rate'))} |")
    lines.append(f"| paired delta (memory − control) | {_fmt(pd.get('value'))} |")
    ci = pd.get("ci") or [None, None]
    lines.append(f"| paired 95% interval | [{_fmt(ci[0])}, {_fmt(ci[1])}] |")
    lines.append(f"| wins memory-only | {_fmt(pd.get('wins_memory'))} |")
    lines.append(f"| wins control-only | {_fmt(pd.get('wins_control'))} |")
    lines.append(f"| negative transfer | {_fmt(_metric(run, 'negative_transfer'))} |")
    lines.append(f"| oracle ceiling | {_fmt(_metric(run, 'oracle_ceiling'))} |")
    if mode == "shadow":
        lines.append("")
        lines.append(
            "> **Shadow run** — no task executed; this table is not paired "
            "evidence and cannot support a learned-controller claim (G8)."
        )
    return lines


def _comparison_table(runs: Sequence[SuiteRun]) -> List[str]:
    lines = _section_header("Baseline comparison (measured on this corpus)")
    retr = [r for r in runs if r.suite == "retrieval"]
    lines.append(
        "All values below are measurements on the seed corpus stated in "
        "the header under the offline_rules configuration. They describe "
        "this harness's fixtures only — they are **not** a superiority or "
        "competitiveness claim (G7 requires licensed corpora and "
        "preregistered protocols, neither of which this table provides)."
    )
    lines.append("")
    header = ["metric"] + [r.baseline for r in retr]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for key, label in _RETRIEVAL_ROWS:
        row = [label]
        for r in retr:
            row.append(_fmt(_metric(r, key)))
        lines.append("| " + " | ".join(row) + " |")
    sec = [r for r in runs if r.suite == "security"]
    if sec:
        lines.append("")
        lines.append("Security lane:")
        lines.append("")
        header = ["metric"] + [r.baseline for r in sec]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for key, label in _SECURITY_ROWS:
            row = [label]
            for r in sec:
                row.append(_fmt(_metric(r, key)))
            lines.append("| " + " | ".join(row) + " |")
    return lines


def _post_quarantine_table(run: SuiteRun) -> List[str]:
    """Per-poisoning-case exposure before vs after quarantine holds."""
    cases = [t for t in run.records if t.kind == "poisoning"]
    pq_metric = run.metrics.get("post_quarantine_exposure") or {}
    post = pq_metric.get("cases") or {}
    if not cases and not post:
        return []
    lines = ["", "#### Poisoning cases: exposure before vs after quarantine", ""]
    lines.append("| case | poisoned sources | pre-quarantine exposed | post-quarantine exposed |")
    lines.append("|---|---|---|---|")
    for rec in cases:
        tid = rec.task_id
        pq = post.get(tid)
        lines.append(
            f"| `{tid}` | {rec.poisoned_total} | {rec.poisoned_returned} | "
            f"{pq['poisoned_returned'] if pq else '—'} |"
        )
    return lines


def _conformance_table(run: SuiteRun) -> List[str]:
    """Surface matrix: per-surface case counts, unavailable lanes, and
    every non-passing case named (§53.05 — nothing drops out)."""
    m = run.metrics
    lines = [
        "",
        f"cases: **{m.get('cases_total', 0)}** — passed "
        f"**{m.get('cases_passed', 0)}**, failed "
        f"**{m.get('cases_failed', 0)}**, skipped "
        f"{m.get('cases_skipped', 0)} "
        f"({m.get('elapsed_s', '?')}s)",
        "",
        "| surface | total | passed | failed | skipped |",
        "|---|---|---|---|---|",
    ]
    for surface, s in sorted((m.get("surfaces") or {}).items()):
        lines.append(
            f"| `{surface}` | {s['total']} | {s['passed']} | "
            f"{s['failed']} | {s['skipped']} |"
        )
    for u in m.get("unavailable_surfaces") or []:
        lines.append(
            f"| `{u['surface']}` | — | — | — | — |"
        )
    if m.get("unavailable_surfaces"):
        lines.append("")
        for u in m["unavailable_surfaces"]:
            lines.append(f"- `{u['surface']}` unavailable: {u['reason']}")
    bad = [r for r in run.records if r["result"] != "passed"]
    if bad:
        lines += ["", "#### Non-passing conformance cases", ""]
        lines.append("| case | surface | result | requirements |")
        lines.append("|---|---|---|---|")
        for rec in bad:
            reqs = ", ".join(rec["requirement_ids"]) or "—"
            lines.append(
                f"| `{rec['node']}` | `{rec['surface']}` | "
                f"{rec['result']} | {reqs} |"
            )
    return lines


def _governance_table(run: SuiteRun) -> List[str]:
    gov = run.metrics.get("governance")
    if not isinstance(gov, dict):
        return []
    lines = ["", "#### Governance checks", ""]
    if gov.get("unavailable_reason"):
        lines.append(f"- {gov['unavailable_reason']}")
        return lines
    checks = gov.get("checks", {})
    cc = checks.get("consent_cycle", {})
    if cc:
        if "error" in cc:
            lines.append(f"- consent cycle: error — {cc['error']}")
        else:
            ok = "pass" if cc.get("ok") else "**fail**"
            lines.append(
                f"- consent cycle: {ok} "
                f"(before={cc.get('before')}, issued={cc.get('during')}, "
                f"after_revoke={cc.get('after_revoke')})"
            )
    pr = checks.get("post_revocation_recall", {})
    if pr:
        if "unavailable" in pr:
            lines.append(f"- post-revocation recall: {pr['unavailable']}")
        else:
            denied = pr.get("denied_or_empty")
            lines.append(
                f"- post-revocation recall denied/empty: "
                f"{'yes' if denied else '**no**'}"
                + (f" (raised {pr['error_type']})" if pr.get("raised") else "")
            )
    return lines


# Gate rows: (gate, what the spec requires, how to read observed value).
# Values are pulled from the runs; nothing here asserts a pass.
def _gates_table(runs: Sequence[SuiteRun]) -> List[str]:
    lines = _section_header("Spec gates vs measured values")
    lines.append(
        "Spec text is quoted as **targets** (§54: design targets, not "
        "results). The measured column shows only what this harness "
        "actually ran; ``pending`` means the required evidence does not "
        "exist in this run — a missing arm or an unavailable capability "
        "leaves the gate pending, never failed-over to a mock success "
        "(§58.09)."
    )
    lines.append("")
    lines.append("| gate | spec target (design target) | measured on this run | status |")
    lines.append("|---|---|---|---|")

    def _sut(suite: str) -> Optional[SuiteRun]:
        """The system-under-test arm for gate evidence: verbatim_v3 first,
        then verbatim_v2, then the first available run — a gate is never
        evidenced by a control arm's trivially-empty results."""
        candidates = [r for r in runs if r.suite == suite]
        for preferred in ("verbatim_v3", "verbatim_v2"):
            for r in candidates:
                if r.baseline == preferred:
                    return r
        return candidates[0] if candidates else None

    sec = _sut("security")
    tasks = _sut("tasks")
    retr = [r for r in runs if r.suite == "retrieval"]

    # G1 — zero disclosure (zero-tolerance, deterministic)
    dv = _metric(sec, "disclosure_violations") if sec else None
    dv_note = f" (baseline `{sec.baseline}`)" if sec else ""
    status = "measured" if dv is not None else "pending"
    lines.append(
        "| G1 — authorization | zero unauthorized disclosures (zero-tolerance) "
        f"| disclosure_violations = {_fmt(dv)}{dv_note} | {status} |"
    )

    # G3 — update automation
    sup_m = None
    for r in retr:
        m = r.metrics.get("supersession")
        if m and m.get("checked"):
            sup_m = (r.baseline, m)
            break
    if sup_m is None and tasks is not None:
        m = tasks.metrics.get("supersession")
        if m and m.get("checked"):
            sup_m = (tasks.baseline, m)
    if sup_m is not None:
        arm, m = sup_m
        # rule-of-three honest bound: 0 false proposals in n twins bounds
        # the true rate under ~3/n — the <0.01 target needs n≳300 twins,
        # reported rather than claimed
        none_str = m.get("none", "0/0")
        n_none = int(none_str.split("/")[1]) if "/" in none_str else 0
        bound = f"{3.0 / n_none:.2f}" if n_none else "n/a"
        deep = _deep_twin_bound()
        if deep:
            prov = deep.get("_provenance") or {}
            twins = deep.get("twins") or {}
            tp = deep.get("true_pairs") or {}
            stale = prov.get("stale_notes") or []
            deep_note = (
                f"; deep corpus {twins.get('total', '?')} synthetic twins "
                f"{twins.get('false_positives', '?')}/"
                f"{twins.get('total', '?')} flagged, upper-95% "
                f"{(deep.get('bound') or {}).get('wilson_upper_95', '?')}"
                f", true pairs detected "
                f"{tp.get('detected', '?')}/{tp.get('total', '?')} "
                f"[artifact seed {prov.get('seed', '?')}, generated "
                f"{prov.get('generated') or 'unknown'} (file mtime), "
                f"detectors: "
                f"{', '.join(prov.get('detectors') or ['?'])}"
                + (
                    "; **stale artifact?** " + "; ".join(stale)
                    if stale
                    else ""
                )
                + "]"
            )
        else:
            deep_note = ""
        lines.append(
            "| G3 — update automation | false-supersession upper 95% bound "
            f"< 0.01; coverage ≥ 0.50 | on `{arm}`: supersede applied "
            f"{m.get('applied', '?')} of true pairs, twins clean "
            f"{none_str} (upper-95% ≈ {bound} at this twin count)"
            f"{deep_note} | measured (update-pair slice) |"
        )
    else:
        lines.append(
            "| G3 — update automation | false-supersession upper 95% bound < 0.01; "
            "coverage ≥ 0.50 | not measured — no update-pair fixtures in this "
            "corpus slice | pending |"
        )

    # G5 — experience benefit
    if tasks is not None:
        pd = tasks.metrics.get("paired_delta", {})
        mode = _metric(tasks, "execution_mode")
        if mode == "shadow" or not pd.get("trials"):
            m = f"no paired trials (mode={mode})"
            st = "pending — shadow/no paired evidence"
        else:
            m = (
                f"delta {_fmt(pd.get('value'))} "
                f"CI [{_fmt(pd.get('ci', [None, None])[0])}, "
                f"{_fmt(pd.get('ci', [None, None])[1])}] "
                f"over {pd.get('trials')} trials"
            )
            st = "measured (harness-level paired execution)"
        lines.append(
            "| G5 — experience benefit | task-success gain ≥ 0.05 over "
            f"no-memory, positive paired interval | {m} | {st} |"
        )
    else:
        lines.append(
            "| G5 — experience benefit | task-success gain ≥ 0.05 over "
            "no-memory | tasks suite not run | pending |"
        )

    lines.append(
        "| G7 — competitive claim | +0.05 macro over strongest baseline on "
        "licensed corpora | not estimable — no licensed competitor corpus "
        "in this harness | pending |"
    )
    if tasks is not None and tasks.metrics.get("paired_delta", {}).get("trials"):
        lines.append(
            "| G8 — learned controller | success lower interval > −0.01 with "
            "≥10% token reduction, real paired executions | paired harness "
            "executed (deterministic controller only — no learned policy "
            "exists to evaluate) | pending |"
        )
    else:
        lines.append(
            "| G8 — learned controller | actual paired policy executions; "
            "shadow logs insufficient | no paired execution in this run | pending |"
        )

    if sec is not None:
        bp = _metric(sec, "benign_instructional_pass_rate")
        ar = _metric(sec, "attack_retrieval_rate")
        lines.append(
            "| G9 — attack/privacy | benign retained ≥ 0.95; attack-induced "
            f"rate ≤ 0.05; disclosure ≤ 0.02 | benign pass {_fmt(bp)}; "
            f"poison exposure {_fmt(ar)}; disclosure {_fmt(dv)}{dv_note} "
            "| measured |"
        )
    else:
        lines.append(
            "| G9 — attack/privacy | benign ≥ 0.95; attack ≤ 0.05; "
            "disclosure ≤ 0.02 | security suite not run | pending |"
        )

    # retrieval workload — a measurement, not a gate
    if retr:
        cells = ", ".join(
            f"`{r.baseline}` recall {_fmt(_metric(r, 'evidence_recall_at_k'))}"
            for r in retr
        )
        lines.append(
            "| retrieval workload (§53) | — (workload measurement, not a "
            f"gate) | {cells} | measured |"
        )
    return lines


def _notes(runs: Sequence[SuiteRun]) -> List[str]:
    notes: List[str] = []
    for run in runs:
        for n in run.notes:
            if n not in notes:
                notes.append(n)
    if not notes:
        return []
    lines = _section_header("Run notes & capability fallbacks")
    lines.extend(f"- {n}" for n in notes)
    return lines


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def render_report(
    runs: Sequence[SuiteRun],
    corpus: Corpus,
    *,
    elapsed_s: Optional[float] = None,
) -> str:
    stats = corpus_stats(corpus)
    now = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%SZ"
    )
    L: List[str] = [
        "# Verbatim v3 evaluation report",
        "",
        f"- corpus: `{stats['name']}` — {stats['tasks']} tasks, "
        f"{stats['setup_sources']} setup sources, "
        f"{stats['runnable']} runnable task fixtures",
        f"- corpus digest (sha256): `{stats['digest']}`",
        f"- workload kinds: "
        + ", ".join(f"{k}={v}" for k, v in sorted(stats["kinds"].items())),
        f"- poison patterns covered: "
        + (", ".join(sorted(stats["poison_patterns"])) or "none"),
        f"- generated: {now}"
        + (f"; elapsed {elapsed_s:.1f}s" if elapsed_s else ""),
        "",
        "**Reading this report.** Every number under a *suite* heading is a "
        "measurement taken against fresh SQLite stores ingested through the "
        "public pipeline, on the corpus above. Values in the *gates* table "
        "quote §54 design targets next to the measurement — nothing in this "
        "document asserts a gate pass, production readiness, or superiority "
        "over any other system.",
    ]
    L += _capability_table(runs)

    for run in runs:
        L += _section_header(f"Suite `{run.suite}` — baseline `{run.baseline}`")
        if run.suite == "retrieval":
            L += _suite_table(run, _RETRIEVAL_ROWS)
        elif run.suite == "grounding":
            L += _suite_table(run, _GROUNDING_ROWS)
        elif run.suite == "security":
            L += _suite_table(run, _SECURITY_ROWS)
            L += _post_quarantine_table(run)
            L += _governance_table(run)
        elif run.suite == "tasks":
            L += _tasks_table(run)
        elif run.suite == "conformance":
            L += _conformance_table(run)

    L += _comparison_table(runs)
    L += _gates_table(runs)
    L += _notes(runs)
    L.append("")
    return "\n".join(L)


__all__ = ["render_report"]
