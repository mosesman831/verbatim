"""CLI driver for the V5 eval portfolio.

Usage::

    python -m eval.v5.run                       # quick portfolio
    python -m eval.v5.run --full                # larger local scale
    python -m eval.v5.run --suite quality       # one suite
    python -m eval.v5.run --out-prefix /tmp/v5  # artifact location

Artifacts: ``<prefix>.json`` (full results), ``<prefix>.md`` (rendered
report), ``<prefix>.repro.json`` (reproduction manifest).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="eval.v5.run")
    p.add_argument("--full", action="store_true",
                   help="larger local scale (still locally measured)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--suite", action="append",
                   help="restrict to named suite(s); repeatable")
    p.add_argument("--out-prefix", default="eval/v5/report_v5",
                   help="artifact path prefix")
    p.add_argument("--workdir", default=None,
                   help="shared workdir (default: per-suite tempdirs)")
    args = p.parse_args(argv)

    from .portfolio import run_portfolio
    from .report import write_report
    from .reproduction import write_manifest

    results = run_portfolio(
        quick=not args.full, seed=args.seed,
        suites=args.suite, workdir=args.workdir)

    out = args.out_prefix
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json_path = out + ".json"
    md_path = out + ".md"
    write_report(results, json_path, md_path)
    manifest = write_manifest(
        results, out + ".repro.json",
        command="python -m eval.v5.run " + " ".join(sys.argv[1:]),
        artifact_paths=(json_path, md_path))

    verdict = results.get("verdict", "unknown")
    print(f"[v5] suites: {list((results.get('suites') or {}).keys())}")
    print(f"[v5] verdict: {verdict} "
          f"(qualification={results.get('qualification')})")
    for name, suite in (results.get("suites") or {}).items():
        st = suite.get("verdict") or suite.get("status")
        print(f"  {name}: {st}")
    print(f"[v5] wrote {json_path}, {md_path}, {out + '.repro.json'}")
    return 0 if verdict in ("passed", "inconclusive", "failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
