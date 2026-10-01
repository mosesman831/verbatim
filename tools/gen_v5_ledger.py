#!/usr/bin/env python3
"""Generate the v5 requirements ledger (SPEC_V5 §27.2, V5-27.*).

Reads every ``V5-NN.MM`` definition out of SPEC_V5.md plus the E01–E96
scenario table (§24/§36), the G5-00–G5-14 gate table (§25), and the
P0–P6 stage table (§26); applies the curated overlay in
``eval/v5/dispositions_v5.json``; then emits:

* ``eval/v5/ledger_v5.json`` — machine-readable per-requirement rows:
  profile/capability, owner, inferred stage + provenance, code
  surfaces, applicable E/C/D scenarios, bound gates, evidence labels,
  named executed evidence, implementation vs qualification status,
  derived rollup status, blocker (V5-27.01/27.05);
* ``eval/v5/summary.md`` — counts by section/status/stage plus the
  scenario and gate coverage tables.

Usage:

* ``python tools/gen_v5_ledger.py --write`` — refresh the dispositions
  spec-hash pin (entries are preserved verbatim; unknown keys fail),
  regenerate both artifacts, then validate.
* ``python tools/gen_v5_ledger.py --check`` — validate only; exits
  nonzero when anything is stale or invalid: duplicate/unknown ids,
  dangling spec references to requirements/scenarios/gates, overlay
  rows for undefined requirements, invalid statuses or evidence
  labels, ``qualified``/``locally_measured`` without named executed
  evidence, missing code surfaces, or generated artifacts that
  drifted.

Default (no flag) is ``--check``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.v5 import ledger as L  # noqa: E402


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
    return idx, disp


def _render_all(idx, disp):
    registry = L.build_registry(idx, disp)
    ledger_json = json.dumps(
        L.render_ledger(idx, registry, disp), indent=2, sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    summary_md = L.render_summary(idx, registry, disp)
    return registry, ledger_json, summary_md


def cmd_check(root: str) -> int:
    pipe = _pipeline(root)
    if pipe is None:
        return 2
    idx, disp = pipe
    issues = L.validate(idx, disp, root=root)

    disp_path = os.path.join(root, L.DEFAULT_DISPOSITIONS_PATH)
    if not os.path.exists(disp_path):
        issues.append(
            f"{L.DEFAULT_DISPOSITIONS_PATH} is missing — run --write"
        )

    # freshness: generated artifacts must equal what --write would emit
    registry, ledger_json, summary_md = _render_all(idx, disp)
    json_path = os.path.join(root, L.DEFAULT_JSON_PATH)
    if _read(json_path) != ledger_json:
        issues.append(f"{L.DEFAULT_JSON_PATH} is stale — run --write")
    md_path = os.path.join(root, L.DEFAULT_SUMMARY_PATH)
    if _read(md_path) != summary_md:
        issues.append(f"{L.DEFAULT_SUMMARY_PATH} is stale — run --write")

    if issues:
        print(f"v5 ledger check FAILED ({len(issues)} issue(s)):")
        for i in issues:
            print(f"  - {i}")
        return 1
    summary = L.render_ledger(idx, registry, disp)["summary"]
    print(
        "v5 ledger OK: {total} requirements "
        "(qualified={q} locally_measured={m} implemented_unmeasured={u} "
        "in_progress={p} planned={pl} failed={f} not_applicable={n}); "
        "{sc} scenarios, {gates} gates".format(
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
        )
    )
    return 0


def cmd_write(root: str) -> int:
    pipe = _pipeline(root)
    if pipe is None:
        return 2
    idx, disp = pipe

    # entries are human-curated and preserved verbatim; only the spec
    # hash pin is refreshed. Unknown ids/keys are caught by validate()
    # below, after the write, so nothing is silently dropped.
    disp.spec_sha256 = dict(idx.spec_sha256)

    registry, ledger_json, summary_md = _render_all(idx, disp)

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
        f"disposition overrides)"
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
                    help="regenerate dispositions pin, ledger_v5.json, "
                         "summary.md")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), help="repo root (default: auto)")
    args = ap.parse_args(argv)
    if args.write:
        return cmd_write(args.root)
    return cmd_check(args.root)


if __name__ == "__main__":
    sys.exit(main())
