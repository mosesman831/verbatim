#!/usr/bin/env python3
"""Generate the v4 executable-traceability ledger (SPEC_V4 §61, V4-02.*).

Reads every ``V4-NN.MM`` / ``V45-NN.MM`` definition out of SPEC_V4.md /
SPEC_V4_5.md, merges it with the curated evidence overlay in
``eval/v4/requirement_map.yaml`` and the anchors discovered in
``tests/``, then emits:

* ``REQUIREMENTS_V4.md`` — human registry, grouped by spec section;
* ``eval/v4/ledger.json`` — machine-readable per-requirement status,
  tests, scenarios, and V4-02.07 evidence labels.

Usage:

* ``python tools/gen_v4_ledger.py --write`` — refresh the map's
  discovered fields (tests/scenarios/spec hashes; human-set statuses
  are preserved), regenerate both artifacts, then validate.
* ``python tools/gen_v4_ledger.py --check`` — validate only; exits
  nonzero when the map is stale or invalid: unknown/duplicate ids,
  dangling spec references, mapped tests pytest cannot collect, stale
  spec hashes, ``verified`` without a named check, discovered checks
  missing from the map, or generated artifacts that drifted.

Default (no flag) is ``--check``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.v4 import ledger as L  # noqa: E402


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
    """Shared front half for both modes: parse, discover, collect, merge."""
    idx = L.load_specs(root)
    disc = L.discover_evidence(root, idx)
    collected = L.collect_pytest_nodes(root)
    map_path = os.path.join(root, L.DEFAULT_MAP_PATH)
    if os.path.exists(map_path):
        try:
            rm = L.load_map(map_path)
        except L.LedgerError as e:
            print(f"error: {e}", file=sys.stderr)
            return None
    else:
        rm = L.RequirementMap(source_path=map_path)
    merged = L.merge_discovery(rm, idx, disc)
    return idx, disc, collected, rm, merged


def _render_all(idx, disc, merged):
    registry = L.build_registry(idx, merged, disc)
    map_yaml = L.render_map_yaml(merged)
    md = L.render_markdown(idx, registry)
    ledger_json = json.dumps(
        L.render_ledger(idx, registry, disc), indent=2, sort_keys=True
    ) + "\n"
    return registry, map_yaml, md, ledger_json


def cmd_check(root: str) -> int:
    pipe = _pipeline(root)
    if pipe is None:
        return 2
    idx, disc, collected, rm, merged = pipe
    issues = L.validate(idx, rm, disc, collected)

    # freshness: the on-disk map and generated artifacts must equal what
    # --write would emit from the current spec + tree
    registry, map_yaml, md, ledger_json = _render_all(idx, disc, merged)
    map_path = os.path.join(root, L.DEFAULT_MAP_PATH)
    if _read(map_path) != map_yaml:
        issues.append(
            f"{L.DEFAULT_MAP_PATH} is stale — discovered evidence or "
            f"spec hashes drifted; run --write"
        )
    md_path = os.path.join(root, L.DEFAULT_MD_PATH)
    if _read(md_path) != md:
        issues.append(f"{L.DEFAULT_MD_PATH} is stale — run --write")
    json_path = os.path.join(root, L.DEFAULT_JSON_PATH)
    if _read(json_path) != ledger_json:
        issues.append(f"{L.DEFAULT_JSON_PATH} is stale — run --write")

    if issues:
        print(f"v4 ledger check FAILED ({len(issues)} issue(s)):")
        for i in issues:
            print(f"  - {i}")
        return 1
    summary = L.render_ledger(idx, registry, disc)["summary"]
    print(
        "v4 ledger OK: {total} requirements "
        "(verified={v} partial={p} implemented_unverified={u} "
        "deferred={d} not_applicable={n} unimplemented={x})".format(
            total=summary["total"],
            v=summary["by_status"]["verified"],
            p=summary["by_status"]["partial"],
            u=summary["by_status"]["implemented_unverified"],
            d=summary["by_status"]["deferred"],
            n=summary["by_status"]["not_applicable"],
            x=summary["by_status"]["unimplemented"],
        )
    )
    return 0


def cmd_write(root: str) -> int:
    pipe = _pipeline(root)
    if pipe is None:
        return 2
    idx, disc, collected, rm, merged = pipe

    # prune mapped test nodes pytest no longer collects (renamed or
    # deleted checks); the drop is printed, never silent
    for rid, entry in merged.entries.items():
        dead = [t for t in entry.tests
                if t.split("[", 1)[0] not in collected]
        for t in dead:
            entry.tests.remove(t)
            print(f"warning: {rid}: dropped uncollectable test {t}",
                  file=sys.stderr)

    _registry, map_yaml, md, ledger_json = _render_all(idx, disc, merged)

    map_path = os.path.join(root, L.DEFAULT_MAP_PATH)
    _write(map_path, map_yaml)
    _write(os.path.join(root, L.DEFAULT_MD_PATH), md)
    _write(os.path.join(root, L.DEFAULT_JSON_PATH), ledger_json)

    # the written artifacts must validate against the reloaded map
    rm2 = L.load_map(map_path)
    issues = L.validate(idx, rm2, disc, collected)
    n = len(rm2.entries)
    print(
        f"wrote {L.DEFAULT_MAP_PATH}, {L.DEFAULT_MD_PATH}, "
        f"{L.DEFAULT_JSON_PATH} ({n} mapped requirements)"
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
                    help="validate map + artifacts (default)")
    ap.add_argument("--write", action="store_true",
                    help="regenerate map, REQUIREMENTS_V4.md, ledger.json")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), help="repo root (default: auto)")
    args = ap.parse_args(argv)
    if args.write:
        return cmd_write(args.root)
    return cmd_check(args.root)


if __name__ == "__main__":
    sys.exit(main())
