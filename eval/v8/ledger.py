"""Paired-run keep-decision ledger for SPEC_V8 (V8-15.04, V8-00.05/00.06).

``eval/v8/ledger.jsonl`` holds one append-only JSON record per landed
V8 requirement — the evidence behind the V8-00.06 keep rule.  Every
requirement that can change retrieval output lands with a paired
LoCoMo-dev Track R run against the head *immediately before it* (never
the wave start — V8-00.05, scenario K04), and the record carries both
manifest digests so the pairing is auditable (V8-15.04, scenario K02).

Record shape (``v8_ledger/v1``)::

    {
      "schema": "v8_ledger/v1",
      "requirement_ids": ["V8-06.01"],          # non-empty, V8-NN.MM
      "defect_ids": ["D8-05"],                  # may be empty
      "base_commit": "<git rev before>",        # paired baseline head
      "head_commit": "<git rev after>",         # the landed head
      "base_manifest_digest": "<sha256 hex>",
      "candidate_manifest_digest": "<sha256 hex>",
      "metrics": {
        "base":      <METRIC_SET>,              # run at base_commit
        "candidate": <METRIC_SET>,              # run at head_commit
      },
      "comparison": {<eval.v8.stats output>} | null,
      "decision": "default_on | flag_off | structural | reverted",
      "rejecting_metric": "<metric>" | null,    # required when flag_off
                                                # or reverted (V8-00.06)
      "decision_record": "eval/v8/decisions/<file>",
      "recorded_at": "<ISO-8601>" | null,
      "notes": [],
    }

``<METRIC_SET>`` is the V8-00.05 set for one side of the pair::

    {
      "answerable": {                            # cats 1-4, item gran.
        "overall":     {any@10, any@20, all@10, prop@10, mrr@10, ndcg@10},
        "categories":  {"<cat>": {same six}},    # per category
      },
      "cat5_premise": {                          # beside, never inside
        "delivery_any@10": x,                    # V8-22.02 delivery
        "correct_refusal": x,                    # V8-22.04 refusal
      },
      "false_insufficient": x,                   # V8-22.03
      "latency_500ms": {"p50": x, "p95": x},     # 500 ms profile
      "sql_statements_per_query_median": x,      # V8-22.07 census
    }

Metric values are numbers or the literal ``"not_run"`` — an unmeasured
slot is labeled, never zero-filled (V8-15.05 honesty rule carried to
the write side).

Append-time validation is hard: a record missing any required field,
carrying an unknown decision, or missing either manifest digest is
refused (``LedgerError``) — an unpinned ledger record is worse than no
record (K01: a retrieval-changing land without a ledger record fails
the check).  CLI: ``python -m eval.v8.ledger --check`` validates the
file and prints a per-record summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Iterable, List, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Record schema tag — versioned like the manifest tags.
RECORD_SCHEMA = "v8_ledger/v1"

#: Default location (relative to repo root).
DEFAULT_LEDGER_PATH = os.path.join("eval", "v8", "ledger.jsonl")

#: Keep decisions (V8-00.06 / V8-15.04).  ``structural`` is the
#: owner-marked exemption from rule (a) — never from rule (b);
#: ``reverted`` names a landed-then-removed requirement.
DECISIONS = ("default_on", "flag_off", "structural", "reverted")

#: Decisions that must name the metric that rejected them (V8-00.06:
#: "a decision record naming the metric that rejected it"; V8-15.04:
#: "the rejecting metric when flagged off").
REJECTING_DECISIONS = ("flag_off", "reverted")

#: The six answerable item-level metrics (V8-00.05 / V8-22.01) — each
#: required at overall and inside every reported category row.
ANSWERABLE_METRICS = (
    "any@10", "any@20", "all@10", "prop@10", "mrr@10", "ndcg@10",
)

#: The cat-5 premise row reports delivery and refusal separately
#: (V8-22.02 / V8-22.04, V8-03.05 "beside, never inside").
CAT5_METRICS = ("delivery_any@10", "correct_refusal")

#: Id shapes (V8-00.03 namespaces).
_REQUIREMENT_ID_RE = re.compile(r"^V8-\d{2}\.\d{2}$")
_DEFECT_ID_RE = re.compile(r"^D8-\d{2}$")

#: Manifest digests are sha256 hex (eval.v7.manifest convention); an
#: 8-char minimum admits a recorded prefix without accepting prose.
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{8,64}$")

#: The required metric-set keys beyond the answerable block.
_METRIC_SET_KEYS = (
    "answerable",
    "cat5_premise",
    "false_insufficient",
    "latency_500ms",
    "sql_statements_per_query_median",
)


class LedgerError(ValueError):
    """Raised when a record or file fails the V8-15.04 contract."""


# ---------------------------------------------------------------------------
# canonical JSON (eval.v7.manifest convention)
# ---------------------------------------------------------------------------


def _json_dumps(value: Any) -> str:
    """Canonical JSON via the repo serialization seam (V5-06.01)."""
    try:
        from verbatim.core.serialize import json_dumps
        return json_dumps(value)
    except Exception:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False)


def _json_loads(text: str) -> Any:
    try:
        from verbatim.core.serialize import json_loads
        return json_loads(text, max_bytes=4 << 20, max_depth=64)
    except ImportError:
        return json.loads(text)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _metric_value_ok(v: Any) -> bool:
    """A metric slot holds a number or the honest ``not_run`` label."""
    return _is_num(v) or v == "not_run"


def _check_id_list(
    rec: Mapping[str, Any], key: str, rx: re.Pattern,
    problems: List[str], *, required: bool,
) -> None:
    val = rec.get(key)
    if val is None:
        if required:
            problems.append(f"{key}: required, missing")
        return
    if not isinstance(val, (list, tuple)):
        problems.append(f"{key}: must be a list")
        return
    if required and not val:
        problems.append(f"{key}: must be non-empty")
        return
    for v in val:
        if not isinstance(v, str) or not rx.match(v):
            problems.append(f"{key}: bad id {v!r}")


def _check_non_empty_str(
    rec: Mapping[str, Any], key: str, problems: List[str],
) -> None:
    v = rec.get(key)
    if not isinstance(v, str) or not v.strip():
        problems.append(f"{key}: required non-empty string")


def _check_digest(
    rec: Mapping[str, Any], key: str, problems: List[str],
) -> None:
    """A record missing a manifest digest is refused outright."""
    v = rec.get(key)
    if not isinstance(v, str) or not v:
        problems.append(f"{key}: manifest digest required")
    elif not _DIGEST_RE.match(v):
        problems.append(f"{key}: not a hex digest ({v!r})")


def _check_answerable_block(
    blk: Any, path: str, problems: List[str],
) -> None:
    if not isinstance(blk, Mapping):
        problems.append(f"{path}.answerable: must be a mapping")
        return
    overall = blk.get("overall")
    if not isinstance(overall, Mapping):
        problems.append(f"{path}.answerable.overall: must be a mapping")
    else:
        for m in ANSWERABLE_METRICS:
            v = overall.get(m)
            if v is None:
                problems.append(
                    f"{path}.answerable.overall.{m}: required")
            elif not _metric_value_ok(v):
                problems.append(
                    f"{path}.answerable.overall.{m}: number or "
                    f"'not_run', got {v!r}")
    cats = blk.get("categories")
    if cats is None:
        problems.append(f"{path}.answerable.categories: required")
    elif not isinstance(cats, Mapping):
        problems.append(
            f"{path}.answerable.categories: must be a mapping")
    else:
        for cat, row in cats.items():
            if not isinstance(row, Mapping):
                problems.append(
                    f"{path}.answerable.categories.{cat}: must be a "
                    "mapping")
                continue
            for m in ANSWERABLE_METRICS:
                v = row.get(m)
                if v is None:
                    problems.append(
                        f"{path}.answerable.categories.{cat}.{m}: "
                        "required")
                elif not _metric_value_ok(v):
                    problems.append(
                        f"{path}.answerable.categories.{cat}.{m}: "
                        f"number or 'not_run', got {v!r}")


def _check_metric_set(ms: Any, path: str, problems: List[str]) -> None:
    if not isinstance(ms, Mapping):
        problems.append(f"{path}: must be a metric-set mapping")
        return
    for key in _METRIC_SET_KEYS:
        if key not in ms:
            problems.append(f"{path}.{key}: required")
    _check_answerable_block(ms.get("answerable"), path, problems)
    cat5 = ms.get("cat5_premise")
    if isinstance(cat5, Mapping):
        for m in CAT5_METRICS:
            v = cat5.get(m)
            if v is None:
                problems.append(f"{path}.cat5_premise.{m}: required")
            elif not _metric_value_ok(v):
                problems.append(
                    f"{path}.cat5_premise.{m}: number or 'not_run', "
                    f"got {v!r}")
    elif cat5 is not None:
        problems.append(f"{path}.cat5_premise: must be a mapping")
    lat = ms.get("latency_500ms")
    if isinstance(lat, Mapping):
        for p in ("p50", "p95"):
            v = lat.get(p)
            if v is None:
                problems.append(f"{path}.latency_500ms.{p}: required")
            elif not _metric_value_ok(v):
                problems.append(
                    f"{path}.latency_500ms.{p}: number or 'not_run', "
                    f"got {v!r}")
    elif lat is not None:
        problems.append(f"{path}.latency_500ms: must be a mapping")
    for key in ("false_insufficient", "sql_statements_per_query_median"):
        v = ms.get(key)
        if v is not None and not _metric_value_ok(v):
            problems.append(f"{path}.{key}: number or 'not_run'")


def record_problems(record: Any) -> List[str]:
    """All V8-15.04 contract violations in ``record`` (empty = valid)."""
    problems: List[str] = []
    if not isinstance(record, Mapping):
        return ["record must be a mapping"]

    schema = record.get("schema")
    if schema is not None and schema != RECORD_SCHEMA:
        problems.append(
            f"schema: expected {RECORD_SCHEMA!r}, got {schema!r}")

    _check_id_list(record, "requirement_ids", _REQUIREMENT_ID_RE,
                   problems, required=True)
    _check_id_list(record, "defect_ids", _DEFECT_ID_RE,
                   problems, required=False)
    if isinstance(record.get("defect_ids"), (list, tuple)) and \
            len(record["defect_ids"]) != len(set(record["defect_ids"])):
        problems.append("defect_ids: duplicates")
    if isinstance(record.get("requirement_ids"), (list, tuple)) and \
            len(record["requirement_ids"]) != \
            len(set(record["requirement_ids"])):
        problems.append("requirement_ids: duplicates")

    _check_non_empty_str(record, "base_commit", problems)
    _check_non_empty_str(record, "head_commit", problems)
    _check_digest(record, "base_manifest_digest", problems)
    _check_digest(record, "candidate_manifest_digest", problems)

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        problems.append("metrics: required mapping with "
                        "'base' and 'candidate' metric sets")
    else:
        for side in ("base", "candidate"):
            if side not in metrics:
                problems.append(f"metrics.{side}: required")
            else:
                _check_metric_set(
                    metrics[side], f"metrics.{side}", problems)

    decision = record.get("decision")
    if decision not in DECISIONS:
        problems.append(
            f"decision: one of {DECISIONS}, got {decision!r}")
    else:
        rejecting = record.get("rejecting_metric")
        if decision in REJECTING_DECISIONS:
            if not isinstance(rejecting, str) or not rejecting.strip():
                problems.append(
                    f"rejecting_metric: required when decision is "
                    f"{decision!r} (V8-00.06)")
        elif rejecting is not None and (
                not isinstance(rejecting, str)):
            problems.append("rejecting_metric: must be a string or null")

    _check_non_empty_str(record, "decision_record", problems)

    cmp_ = record.get("comparison")
    if cmp_ is not None and not isinstance(cmp_, Mapping):
        problems.append("comparison: must be a mapping or null")
    notes = record.get("notes")
    if notes is not None and (
            not isinstance(notes, (list, tuple))
            or not all(isinstance(n, str) for n in notes)):
        problems.append("notes: must be a list of strings")
    rat = record.get("recorded_at")
    if rat is not None and not isinstance(rat, str):
        problems.append("recorded_at: must be a string or null")
    return problems


def validate_record(record: Mapping[str, Any]) -> dict:
    """Normalize + validate one record; raises ``LedgerError``.

    Defaults filled: ``schema``, ``defect_ids: []``,
    ``rejecting_metric: None``, ``comparison: None``, ``notes: []``,
    ``recorded_at: None``.  Unknown extra keys pass through — the
    ledger is append-only evidence, not a closed struct.
    """
    problems = record_problems(record)
    if problems:
        raise LedgerError(
            "ledger record rejected: " + "; ".join(problems))
    rec = dict(record)
    rec["schema"] = RECORD_SCHEMA
    rec.setdefault("defect_ids", [])
    rec.setdefault("rejecting_metric", None)
    rec.setdefault("comparison", None)
    rec.setdefault("notes", [])
    rec.setdefault("recorded_at", None)
    return rec


# ---------------------------------------------------------------------------
# append / read
# ---------------------------------------------------------------------------


def append_record(path: str, record: Mapping[str, Any]) -> dict:
    """Validate and append one record to ``ledger.jsonl``.

    The canonical serialization is a single sorted-key JSON line —
    byte-stable so the file diffs cleanly and digests reproducibly.
    """
    rec = validate_record(record)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json_dumps(rec) + "\n")
    return rec


def read_ledger(path: str, *, strict: bool = False) -> List[dict]:
    """Read every record.  A missing file is honest emptiness; a
    malformed line is a ``LedgerError`` — evidence is never skipped
    silently.  ``strict`` re-validates each record's contract."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise LedgerError(f"ledger unreadable: {exc}") from exc
    out: List[dict] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = _json_loads(line)
        except Exception as exc:
            raise LedgerError(
                f"ledger line {i}: malformed JSON ({exc})") from exc
        if strict:
            problems = record_problems(rec)
            if problems:
                raise LedgerError(
                    f"ledger line {i}: " + "; ".join(problems))
        out.append(rec)
    return out


def latest(path: str) -> Optional[dict]:
    """The most recent record, or None on an empty/missing ledger."""
    recs = read_ledger(path)
    return recs[-1] if recs else None


def check_chain(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """K04 pairing audit: each record's ``base_commit`` should equal the
    previous record's ``head_commit`` — the paired baseline is the head
    immediately before, never the wave start.  Returns violation
    strings (empty = clean chain)."""
    violations: List[str] = []
    prev = None
    for i, rec in enumerate(records):
        if prev is not None:
            b = rec.get("base_commit")
            h = prev.get("head_commit")
            if b is not None and h is not None and b != h:
                violations.append(
                    f"record {i} ({rec.get('requirement_ids')}): "
                    f"base_commit {b!r} != prior head_commit {h!r}")
        prev = rec
    return violations


# ---------------------------------------------------------------------------
# metric helpers (report + keep-rule consumers)
# ---------------------------------------------------------------------------


def delta_any10(record: Mapping[str, Any]) -> Optional[float]:
    """Candidate − base answerable ``any@10`` (the V8-00.06 (a) axis);
    ``None`` when either side is ``not_run``/absent."""
    try:
        c = record["metrics"]["candidate"]["answerable"]["overall"]["any@10"]
        b = record["metrics"]["base"]["answerable"]["overall"]["any@10"]
    except (KeyError, TypeError):
        return None
    if _is_num(c) and _is_num(b):
        return float(c) - float(b)
    return None


def category_recall_deltas(record: Mapping[str, Any],
                           metric: str = "any@10") -> dict:
    """``{category: candidate − base}`` over the union of reported
    categories — the V8-00.06 (b) recall-regression axis.  Categories
    missing on one side or ``not_run`` are skipped, never zeroed."""
    m = record.get("metrics") or {}
    cb = ((m.get("candidate") or {}).get("answerable") or {})
    bb = ((m.get("base") or {}).get("answerable") or {})
    ccats = cb.get("categories") or {}
    bcats = bb.get("categories") or {}
    out = {}
    for cat in sorted(set(ccats) | set(bcats)):
        c = (ccats.get(cat) or {}).get(metric)
        b = (bcats.get(cat) or {}).get(metric)
        if _is_num(c) and _is_num(b):
            out[cat] = float(c) - float(b)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v8.ledger",
        description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", default=DEFAULT_LEDGER_PATH,
                    help=f"ledger path (default {DEFAULT_LEDGER_PATH})")
    ap.add_argument("--check", action="store_true",
                    help="validate every record + the K04 pairing chain")
    args = ap.parse_args(argv)
    recs = read_ledger(args.ledger, strict=True)
    violations = check_chain(recs)
    print(f"{args.ledger}: {len(recs)} record(s), "
          f"{len(violations)} pairing violation(s)")
    for r in recs:
        ids = ",".join(r.get("requirement_ids") or [])
        d = delta_any10(r)
        print(f"  {ids}: {r.get('decision')} "
              f"Δany@10={'not_run' if d is None else f'{d:+.4f}'}")
    for v in violations:
        print(f"  PAIRING: {v}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
