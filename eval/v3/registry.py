"""V3 requirement registry: seed ``registry/ledger.json`` from SPEC_V3.md.

Parses every ``V3-NN.MM`` requirement id from the spec (§56.10: the
frozen evidence matrix enumerates all requirements), records the owning
section from the ``## NN.`` headers, assigns the frozen required-evidence
set per :func:`eval.v3.ledger.required_evidence_for_section`, and writes
the seed ledger that every run registers evidence into.

Usage::

    python -m eval.v3.registry            # writes eval/v3/registry/ledger.json
    python -m eval.v3.registry --check    # verify seed is current

The seed marks every requirement ``pending`` (unverified) — the registry
never invents evidence; statuses advance only through Ledger.register().
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Iterable, Optional

from .ledger import Ledger, load_ledger

#: ``- V3-53.01: <text>`` — the numbered-requirement bullet form used
#: throughout SPEC_V3.md. Anchored to line start so mentions inside prose
#: (e.g. "see V3-56.01") do not create phantom requirements.
REQ_RE = re.compile(r"^\s*-\s*(V3-\d{2}\.\d{2})\s*[:.]\s*(.*)$")
SECTION_RE = re.compile(r"^##\s+(\d{2})\.")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPEC_PATH = os.path.join(REPO_ROOT, "SPEC_V3.md")
LEDGER_PATH = os.path.join(HERE, "registry", "ledger.json")


def parse_requirements(spec_path: str = SPEC_PATH) -> list[dict]:
    """Extract ``{req_id, section, title}`` for every numbered
    requirement in SPEC_V3.md, in file order, deduped by id."""
    out: list[dict] = []
    seen: set[str] = set()
    section = 0
    with open(spec_path, "r", encoding="utf-8") as f:
        for line in f:
            m = SECTION_RE.match(line)
            if m:
                section = int(m.group(1))
                continue
            m = REQ_RE.match(line)
            if m:
                rid, title = m.group(1), m.group(2).strip()
                if rid in seen:
                    continue
                seen.add(rid)
                out.append({"req_id": rid, "section": section, "title": title})
    return out


def seed_ledger(
    spec_path: str = SPEC_PATH,
    *,
    ledger: Optional[Ledger] = None,
) -> Ledger:
    """Build the seed ledger: every V3-NN.MM requirement, all pending,
    with required-evidence sets frozen by section (§56.01)."""
    lg = ledger or Ledger()
    reqs = parse_requirements(spec_path)
    if not reqs:
        raise ValueError(f"no V3-NN.MM requirements found in {spec_path}")
    lg.spec = f"{os.path.basename(spec_path)} R1"
    lg.meta.setdefault("seed", "all requirements pending — no evidence registered")
    for r in reqs:
        lg.add_requirement(r["req_id"], section=r["section"], title=r["title"])
    return lg


def write_seed(
    spec_path: str = SPEC_PATH, out_path: str = LEDGER_PATH
) -> str:
    """Parse the spec and write the seed ledger.json."""
    lg = seed_ledger(spec_path)
    return lg.write(out_path)


def check_current(
    spec_path: str = SPEC_PATH, ledger_path: str = LEDGER_PATH
) -> list[str]:
    """Return drift between the spec's requirement ids and the seed file:
    ids in the spec but missing from the ledger, and stale ids in the
    ledger that no longer appear in the spec."""
    spec_ids = {r["req_id"] for r in parse_requirements(spec_path)}
    lg = load_ledger(ledger_path)
    ledger_ids = set(lg.requirements)
    drift = []
    for rid in sorted(spec_ids - ledger_ids):
        drift.append(f"missing from ledger: {rid}")
    for rid in sorted(ledger_ids - spec_ids):
        drift.append(f"stale in ledger: {rid}")
    return drift


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default=SPEC_PATH)
    ap.add_argument("--out", default=LEDGER_PATH)
    ap.add_argument(
        "--check",
        action="store_true",
        help="report drift between SPEC_V3.md and the seed ledger",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.check:
        drift = check_current(args.spec, args.out)
        for d in drift:
            print(d)
        print(f"{len(drift)} drift item(s)")
        return 1 if drift else 0
    path = write_seed(args.spec, args.out)
    n = len(parse_requirements(args.spec))
    print(f"wrote {n} requirements -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
