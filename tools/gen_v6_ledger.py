#!/usr/bin/env python3
"""Generate the v6 requirements ledger (SPEC_V6, V6-00.06/04.01).

Reads every ``V6-NN.MM`` definition out of SPEC_V6.md plus the
F01–F32 scenario table (§07), the G6-00–G6-08 gate table (§08), and the
P0–P7 phase table (§09); applies the curated overlay in
``eval/v6/dispositions_v6.json``; reads the carried V5 tail from
``eval/v5/dispositions_v5.json`` (read-only — V6-04.02/03); then emits:

* ``eval/v6/ledger_v6.json`` — machine-readable per-requirement rows:
  profile/capability, owner, inferred phase + provenance, code
  surfaces, applicable F scenarios, bound G6 gates, envelopes,
  inherited V4/V5 ids, evidence labels, named executed evidence,
  implementation vs qualification status, derived rollup status,
  blocker — plus the ``carried`` block listing every V5 row still
  ``planned`` or ``implemented_unmeasured``;
* ``eval/v6/summary.md`` — counts by section/status/phase plus the
  scenario, gate, and carried-tail coverage tables.

Usage:

* ``python tools/gen_v6_ledger.py --write`` — refresh the dispositions
  spec-hash pin (entries are preserved verbatim; unknown keys fail),
  regenerate both artifacts, then validate.
* ``python tools/gen_v6_ledger.py --check`` — validate only; exits
  nonzero when anything is stale or invalid: duplicate/unknown ids,
  dangling spec references to requirements/scenarios/gates, ids
  outside the declared F01–F32 / G6-00–G6-08 registries, overlay rows
  for undefined requirements, invalid statuses or evidence labels,
  ``qualified``/``locally_measured`` without named executed evidence,
  missing code surfaces, or generated artifacts that drifted.

Default (no flag) is ``--check``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.v6 import ledger as L  # noqa: E402


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _pipeline(root: str):
    """Shared front half for both modes: parse spec + dispositions."""
    try:
        idx = L.load_specs(root)
    except L.LedgerError as e:
        print(f"error: {e}", file=sys.stderr)
        return None
    disp_path = os.path.join(root, L.DEFAULT_DISPOSITIONS_PATH)
    if os.path.exists(disp_path):
        try:
            disp = L.load_dispositions(disp_path)
        except L.LedgerError as e:
            print(f"error: {e}", file=sys.stderr)
            return None
    else:
        disp = L.Dispositions(source_path=disp_path)
    try:
        carried = L.load_carried(root)
    except L.LedgerError as e:
        print(f"error: {e}", file=sys.stderr)
        return None
    return idx, disp, carried


def _render_all(idx, disp, carried):
    registry = L.build_registry(idx, disp)
    ledger_json = json.dumps(
        L.render_ledger(idx, registry, disp, carried=carried),
        indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n"
    summary_md = L.render_summary(idx, registry, disp, carried=carried)
    return registry, ledger_json, summary_md


def cmd_check(root: str) -> int:
    pipe = _pipeline(root)
    if pipe is None:
        return 2
    idx, disp, carried = pipe
    issues = L.validate(idx, disp, root=root)

    disp_path = os.path.join(root, L.DEFAULT_DISPOSITIONS_PATH)
    if not os.path.exists(disp_path):
        issues.append(
            f"{L.DEFAULT_DISPOSITIONS_PATH} is missing — run --write"
        )

    # freshness: generated artifacts must equal what --write would emit
    registry, ledger_json, summary_md = _render_all(idx, disp, carried)
    json_path = os.path.join(root, L.DEFAULT_JSON_PATH)
    if _read(json_path) != ledger_json:
        issues.append(f"{L.DEFAULT_JSON_PATH} is stale — run --write")
    md_path = os.path.join(root, L.DEFAULT_SUMMARY_PATH)
    if _read(md_path) != summary_md:
        issues.append(f"{L.DEFAULT_SUMMARY_PATH} is stale — run --write")

    if issues:
        print(f"v6 ledger check FAILED ({len(issues)} issue(s)):")
        for i in issues:
            print(f"  - {i}")
        return 1
    summary = L.render_ledger(idx, registry, disp, carried=carried)["summary"]
    print(
        "v6 ledger OK: {total} requirements "
        "(qualified={q} locally_measured={m} implemented_unmeasured={u} "
        "in_progress={p} planned={pl} failed={f} not_applicable={n}); "
        "{sc} scenarios, {gates} gates; carried V5 tail {ct}".format(
            total=summary["total"],
            q=summary["by_status"]["qualified"],
            m=summary["by_status"]["locally_measured"],
            u=summary["by_status"]["implemented_unmeasured"],
            p=summary["by_status"]["in_progress"],
            pl=summary["by_status"]["planned"],
            f=summary["by_status"]["failed"],
            n=summary["by_status"]["not_applicable"],
            sc=summary["scenarios_total"],
            gates=summary["gates_total"],
            ct=summary["carried_total"],
        )
    )
    return 0


def cmd_write(root: str) -> int:
    pipe = _pipeline(root)
    if pipe is None:
        return 2
    idx, disp, carried = pipe

    # entries are human-curated and preserved verbatim; only the spec
    # hash pin is refreshed. Unknown ids/keys are caught by validate()
    # below, after the write, so nothing is silently dropped.
    disp.spec_sha256 = dict(idx.spec_sha256)

    registry, ledger_json, summary_md = _render_all(idx, disp, carried)

    disp_path = os.path.join(root, L.DEFAULT_DISPOSITIONS_PATH)
    _write(disp_path, L.render_dispositions(disp))
    _write(os.path.join(root, L.DEFAULT_JSON_PATH), ledger_json)
    _write(os.path.join(root, L.DEFAULT_SUMMARY_PATH), summary_md)

    # the written artifacts must validate against the reloaded overlay
    disp2 = L.load_dispositions(disp_path)
    issues = L.validate(idx, disp2, root=root)
    print(
        f"wrote {L.DEFAULT_DISPOSITIONS_PATH}, {L.DEFAULT_JSON_PATH}, "
        f"{L.DEFAULT_SUMMARY_PATH} "
        f"({len(registry)} requirements, {len(disp2.entries)} "
        f"disposition overrides, {len(carried.rows)} carried V5 rows)"
    )
    if issues:
        print(f"post-write validation FAILED ({len(issues)} issue(s)):")
        for i in issues:
            print(f"  - {i}")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="validate dispositions + artifacts (default)")
    ap.add_argument("--write", action="store_true",
                    help="regenerate dispositions pin, ledger_v6.json, "
                         "summary.md")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), help="repo root (default: auto)")
    args = ap.parse_args(argv)
    if args.write:
        return cmd_write(args.root)
    return cmd_check(args.root)


if __name__ == "__main__":
    sys.exit(main())
