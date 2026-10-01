"""LoCoMo observation oracle — eval-only labeled fact set (SPEC_V7_5
V75-04.06; SPEC_V7 V7-13.19, V7-22.18).

``locomo10.json`` carries, per conversation, an ``observation`` object:
``{"session_N_observation": {speaker: [[assertion, ref], ...]}}`` where
each pair is an assertion text plus the dialog ref(s) it was derived
from (``"D<s>:<t>"`` — the released file also emits lists
(``["D15:3", "D15:5"]``) and comma-joined strings
(``"D26:14, D26:34, D26:42"``); :func:`corpora._parse_evidence`
normalizes all three).

This module turns that layer into a typed :class:`ObservationOracle`.
Its only uses — by spec — are:

* estimating fact coverage ``p`` for the V7-13.19 unit ablation
  (what fraction of labeled assertions does a retrieval-unit set
  retain?), and
* typed-lane coverage scoring (slice the fact set by speaker or
  session and measure coverage per slice).

The oracle is **gold**: an oracle ceiling, disclosed as such, never
reported as the system, and never fed back into lexicons, prompts, or
policy (V7-13.19).  It lives under ``eval/`` only; the V7-22.18
tripwire (``tests/eval/test_v75_eval_infra.py``) fails the build if any
runtime ``verbatim/`` module imports it.  Runtime code may not branch
on dataset identity, so nothing in ``verbatim/`` may know this module
exists.

Loading path: :func:`eval.v7.corpora.load_locomo` carries the raw
``conv["observation"]`` dicts into ``Corpus.metadata`` under
:data:`eval.v7.corpora.LOCOMO_ORACLE_META_KEY`, so
:func:`oracle_from_corpus` builds the fact set without re-reading the
(gated, local-only) file.  :func:`oracle_from_conversations` builds it
directly from parsed JSON for fixture-level tests.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from eval.v7 import corpora


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleFact:
    """One labeled oracle assertion (V75-04.06).

    ``assertion`` is the released observation text; ``speaker`` the
    per-speaker table key the pair was filed under; ``session_n`` the
    ``N`` of its ``session_N_observation`` key.  ``evidence_refs`` are
    the normalized ``D<s>:<t>`` dialog refs; ``evidence_item_ids`` are
    the same refs qualified into the corpus item-id space
    (``{sample_id}:D<s>:<t>``) so coverage can be checked directly
    against a unit/item-id set.  ``unresolved_refs`` keeps the tokens
    that did not parse — recorded, never guessed (the loader's
    ``unresolved_evidence`` convention).  ``raw_ref`` preserves the
    released second element verbatim for audit.
    """

    fact_id: str
    sample_id: str
    session_n: int
    speaker: str
    assertion: str
    evidence_refs: Tuple[str, ...] = ()
    evidence_item_ids: Tuple[str, ...] = ()
    unresolved_refs: Tuple[str, ...] = ()
    raw_ref: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "sample_id": self.sample_id,
            "session_n": self.session_n,
            "speaker": self.speaker,
            "assertion": self.assertion,
            "evidence_refs": list(self.evidence_refs),
            "evidence_item_ids": list(self.evidence_item_ids),
            "unresolved_refs": list(self.unresolved_refs),
            "raw_ref": self.raw_ref,
        }


@dataclass(frozen=True)
class FactCoverage:
    """Coverage estimate ``p`` over an oracle fact set.

    ``p`` is ``covered / total``; ``None`` when the fact set is empty —
    an undefined denominator is reported, not silently scored 0
    (metrics.py convention).  ``covered_ids`` / ``uncovered_ids`` name
    the per-fact verdicts so a report can show *which* assertions a unit
    set retains.
    """

    total: int
    covered: int
    p: Optional[float]
    covered_ids: Tuple[str, ...] = ()
    uncovered_ids: Tuple[str, ...] = ()
    mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "covered": self.covered,
            "p": self.p,
            "mode": self.mode,
            "covered_ids": list(self.covered_ids),
            "uncovered_ids": list(self.uncovered_ids),
        }


@dataclass(frozen=True)
class ObservationOracle:
    """The labeled oracle fact set for one LoCoMo load (V75-04.06).

    ``facts`` is every parsed ``[assertion, ref]`` pair, in released
    order.  ``dropped`` holds the entries that could not be parsed at
    all — the honest count of unusable labels (an oracle that silently
    loses facts inflates coverage denominators).
    """

    dataset_id: str
    facts: Tuple[OracleFact, ...]
    dropped: Tuple[Any, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.facts)

    def __iter__(self):
        return iter(self.facts)

    def for_sample(self, sample_id: str) -> Tuple[OracleFact, ...]:
        """The fact slice of one conversation (a split group)."""
        return tuple(f for f in self.facts if f.sample_id == sample_id)

    def speakers(self) -> Tuple[str, ...]:
        return tuple(sorted({f.speaker for f in self.facts}))

    def digest(self) -> str:
        """Content fingerprint — pinned beside any coverage number so a
        report proves which oracle file produced it."""
        canon = {
            "dataset_id": self.dataset_id,
            "facts": [f.to_dict() for f in self.facts],
            "dropped": [str(d) for d in self.dropped],
        }
        blob = json.dumps(
            canon, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

_OBS_KEY = re.compile(r"session_(\d+)_observation")


def _pair_refs(raw_ref: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Normalize the released ref element to ``D<s>:<t>`` refs.

    The file emits a bare string, a list of strings, or a comma/space-
    joined multi-ref string — all handled by the loader's evidence
    parser so oracle refs normalize identically to QA evidence.
    """
    if raw_ref is None:
        return (), ()
    pieces: Iterable[Any] = (
        raw_ref if isinstance(raw_ref, (list, tuple)) else (raw_ref,)
    )
    return corpora._parse_evidence(pieces)


def parse_observation_table(
    sample_id: str, observation: Optional[Mapping[str, Any]]
) -> Tuple[Tuple[OracleFact, ...], Tuple[Any, ...]]:
    """Parse one conversation's ``observation`` dict into oracle facts.

    Returns ``(facts, dropped)``: entries that are not a
    ``[assertion, ref-ish]`` pair are collected into ``dropped`` for the
    audit count rather than crashing or being silently skipped.
    """
    facts: List[OracleFact] = []
    dropped: List[Any] = []
    for key in sorted(
        (observation or {}).keys(),
        key=lambda k: (
            int(_OBS_KEY.fullmatch(k).group(1))
            if _OBS_KEY.fullmatch(k)
            else 1 << 30,
            k,
        ),
    ):
        m = _OBS_KEY.fullmatch(key)
        table = (observation or {}).get(key)
        if not m or not isinstance(table, Mapping):
            dropped.append({key: table})
            continue
        session_n = int(m.group(1))
        for speaker in sorted(table.keys()):
            pairs = table.get(speaker)
            if not isinstance(pairs, (list, tuple)):
                dropped.append({key: {speaker: pairs}})
                continue
            for pair in pairs:
                if (
                    not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                ):
                    dropped.append({key: {speaker: pair}})
                    continue
                assertion, raw_ref = pair
                refs, unresolved = _pair_refs(raw_ref)
                idx = len(facts)
                facts.append(
                    OracleFact(
                        fact_id=f"{sample_id}#obs{idx:04d}",
                        sample_id=sample_id,
                        session_n=session_n,
                        speaker=str(speaker),
                        assertion=assertion,
                        evidence_refs=refs,
                        evidence_item_ids=tuple(
                            f"{sample_id}:{r}" for r in refs
                        ),
                        unresolved_refs=unresolved,
                        raw_ref=raw_ref,
                    )
                )
    return tuple(facts), tuple(dropped)


def oracle_from_conversations(
    data: Iterable[Mapping[str, Any]], *, dataset_id: str = "locomo"
) -> ObservationOracle:
    """Build the oracle from parsed ``locomo10.json`` content.

    ``data`` is the released top-level list of conversation objects;
    each contributes its ``observation`` dict.  Conversations without an
    observation layer contribute nothing (recorded, not invented).
    """
    facts: List[OracleFact] = []
    dropped: List[Any] = []
    seen: List[str] = []
    for conv in data:
        sid = (conv or {}).get("sample_id", "unknown")
        seen.append(sid)
        fs, ds = parse_observation_table(
            sid, (conv or {}).get("observation")
        )
        facts.extend(fs)
        dropped.extend(ds)
    return ObservationOracle(
        dataset_id=dataset_id,
        facts=tuple(facts),
        dropped=tuple(dropped),
        metadata={"sample_ids": seen},
    )


def oracle_from_corpus(corpus: Any) -> ObservationOracle:
    """Build the oracle from a loaded LoCoMo :class:`corpora.Corpus`.

    Reads the raw observation layer the loader carried into
    ``Corpus.metadata[LOCOMO_ORACLE_META_KEY]`` — no second read of the
    gated file.  A corpus without the layer (a non-LoCoMo corpus, or a
    load predating V75-04.06) yields an empty oracle rather than a
    fabricated one.
    """
    raw = (getattr(corpus, "metadata", None) or {}).get(
        corpora.LOCOMO_ORACLE_META_KEY
    ) or {}
    facts: List[OracleFact] = []
    dropped: List[Any] = []
    for sid in sorted(raw.keys()):
        fs, ds = parse_observation_table(sid, raw.get(sid))
        facts.extend(fs)
        dropped.extend(ds)
    return ObservationOracle(
        dataset_id=getattr(corpus, "dataset_id", "locomo"),
        facts=tuple(facts),
        dropped=tuple(dropped),
        metadata={"sample_ids": sorted(raw.keys())},
    )


# ---------------------------------------------------------------------------
# fact coverage (V7-13.19 p estimate; typed-lane coverage)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset:
    """Lowercase alphanumeric token set — the overlap unit."""
    return frozenset(_TOKEN.findall(text.lower()))


def fact_coverage(
    facts: Iterable[OracleFact],
    *,
    covered_item_ids: Optional[Iterable[str]] = None,
    covered_texts: Optional[Iterable[str]] = None,
    min_token_overlap: float = 1.0,
) -> FactCoverage:
    """Estimate ``p``: the fraction of oracle facts a unit set retains.

    A fact is **covered** when either holds (the two modes union):

    * *id mode* — at least one of the fact's ``evidence_item_ids``
      appears in ``covered_item_ids``.  This is the turn-unit question:
      does the unit set still contain a dialog turn the assertion was
      derived from?  (Any-one-is-enough is the lenient form; a fact
      listing several refs is covered when the unit set retains any of
      them.)
    * *text mode* — at least one covered text's token set contains ≥
      ``min_token_overlap`` of the assertion's token set
      (``|assert ∩ text| / |assert| ≥ θ``).  ``θ = 1.0`` is verbatim
      containment — the strict form for quote-verified extracted-line
      units; lower θ is the caller's documented fuzziness, never a
      default smuggled in here.

    This is a *lower bound* on true coverage: paraphrase-level matches
    are not attempted (no semantic scorer — determinism rule).  Facts
    with no resolvable refs and no text match count uncovered, honestly.
    """
    if not (0.0 < min_token_overlap <= 1.0):
        raise ValueError(
            f"min_token_overlap must be in (0,1], got {min_token_overlap}"
        )
    fact_list = tuple(facts)
    id_set = frozenset(covered_item_ids or ())
    # Inverted index token -> covered-text token sets, so each fact
    # inspects only texts sharing ≥1 assertion token.
    text_index: Dict[str, List[frozenset]] = {}
    text_sets: List[frozenset] = []
    for t in covered_texts or ():
        ts = _tokens(t)
        if not ts:
            continue
        text_sets.append(ts)
        for tok in ts:
            text_index.setdefault(tok, []).append(ts)

    covered: List[str] = []
    uncovered: List[str] = []
    for f in fact_list:
        hit = False
        if id_set and any(r in id_set for r in f.evidence_item_ids):
            hit = True
        if not hit and text_sets:
            want = _tokens(f.assertion)
            if want:
                cands: List[frozenset] = []
                seen: set = set()
                for tok in want:
                    for ts in text_index.get(tok, ()):
                        if id(ts) not in seen:
                            seen.add(id(ts))
                            cands.append(ts)
                need = min_token_overlap * len(want)
                for ts in cands:
                    if len(want & ts) >= need:
                        hit = True
                        break
        (covered if hit else uncovered).append(f.fact_id)
    total = len(fact_list)
    return FactCoverage(
        total=total,
        covered=len(covered),
        p=(len(covered) / total) if total else None,
        covered_ids=tuple(covered),
        uncovered_ids=tuple(uncovered),
        mode="+".join(
            s
            for s, on in (
                ("evidence_ids", bool(id_set)),
                ("token_overlap", bool(text_sets)),
            )
            if on
        ),
    )
