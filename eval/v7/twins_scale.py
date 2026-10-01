"""Parameterized scale twin generator for latency envelopes (V7-18, H77).

``generate(seed, n_units, vocab_size, df_skew)`` produces an invented
message corpus with a *controlled* term/document-frequency distribution:
a Zipf-weighted vocabulary (term rank ``r`` drawn with weight
``r ** -df_skew``) plus a small set of dominant speaker canons injected
into a large share of unit texts. The result reproduces the D7-07
speaker-dominance pathology at scale — a flat per-entity bonus would
promote thousands of speaker-name-bearing units over the one evidence
turn that actually answers the question — and sizes to 1K / 10K / 100K
units for envelope runs (B1/B3; T-1M ≈ 20K units per V7-18.01).

Planted probe tasks (``speaker_topic_probe``) give the scale corpora a
retrieval-quality check beside raw latency: each probe picks a dominant
speaker and a unique rare two-word topic, plants exactly one evidence
unit spoken by that speaker containing the topic, and asks
"What did <speaker> say about <topic>?". A ranker without IDF-weighted
entity signals (V7-08.06) should drown it in speaker-only units — the
fixture records ``df`` of both the speaker canon and the topic so the
pathology is measurable, not anecdotal.

Deterministic per ``(seed, n_units, vocab_size, df_skew, ...)``; stdlib
only; no benchmark text. Generation is O(n_units × avg_len); 100K units
is seconds-scale in pure Python.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

GENERATOR_ID = "twins_scale/v1"
CONSTANTS_TAG = "provisional/v7-r0"
CORPUS_NAME = "owned_scale"
DEFAULT_SEED = 20260924

BASE_US = 1_704_067_200_000_000  # 2024-01-01T00:00:00Z
DAY_US = 86_400_000_000
HOUR_US = 3_600_000_000

#: Invented speaker display names — the dominant-speaker pool (D7-07).
SPEAKERS: Tuple[str, ...] = (
    "Vesper", "Odell", "Romer", "Sagan", "Tull", "Ines", "Bram", "Cato",
)

#: Per-speaker share of authorship, Zipf-lite (index 0 is the heavy talker).
_SPEAKER_WEIGHTS: Tuple[float, ...] = (
    0.46, 0.24, 0.12, 0.07, 0.04, 0.03, 0.02, 0.02,
)

#: Probability a unit's text name-drops a *different* speaker, and the
#: probability it carries its own speaker's canon — together these inflate
#: the dominant canons' document frequency (the D7-07 pathology).
_SPEAKER_MENTION_P = 0.38
_SPEAKER_SELF_P = 0.50

#: Invented rare probe topics — each is used by exactly one probe so its
#: df is 1 by construction.
PROBE_TOPICS: Tuple[str, ...] = (
    "orchid pressing", "tidal cartography", "kiln glazing", "kite rigging",
    "lichen survey", "bell tuning", "moss gardening", "reef sketching",
    "dune mapping", "fermenting kraut", "canal lockpicking", "ember firing",
    "cloud cataloguing", "saddle stitching", "tide pooling", "ash glazing",
    "wind rosing", "loom warping", "cairn building", "flint knapping",
    "reed weaving", "sap tapping", "ice auditing", "grove grafting",
)

_PROBE_VERBS: Tuple[str, ...] = (
    "started", "finished", "documented", "photographed", "mapped",
    "repaired", "measured", "catalogued",
)

_PROBE_DETAILS: Tuple[str, ...] = (
    "over the long weekend", "during the quiet sprint", "with good results",
    "and logged the outcome", "before the weather turned", "on my day off",
)


def corpus_digest(corpus: Dict[str, Any]) -> str:
    canon = {
        "name": corpus.get("name"),
        "generator": corpus.get("generator"),
        "seed": corpus.get("seed"),
        "params": corpus.get("params"),
        "units": corpus.get("units"),
        "tasks": corpus.get("tasks"),
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _term(rank0: int) -> str:
    return f"w{rank0:05d}"


def _build_sampler(
    rng: random.Random, vocab_size: int, df_skew: float
) -> Tuple[List[float], List[str]]:
    """Cumulative-weight table over vocab ranks; sample via bisect."""
    terms = [_term(i) for i in range(vocab_size)]
    cum: List[float] = []
    total = 0.0
    for i in range(vocab_size):
        total += (i + 1) ** (-df_skew)
        cum.append(total)
    return cum, terms


def _sample_term(rng: random.Random, cum: List[float], terms: List[str]) -> str:
    x = rng.random() * cum[-1]
    return terms[bisect.bisect_left(cum, x)]


def _pick_speaker(rng: random.Random, n_speakers: int) -> Tuple[str, int]:
    x = rng.random()
    acc = 0.0
    for i in range(min(n_speakers, len(_SPEAKER_WEIGHTS))):
        acc += _SPEAKER_WEIGHTS[i]
        if x <= acc:
            return SPEAKERS[i], i
    return SPEAKERS[min(n_speakers, len(SPEAKERS)) - 1], min(
        n_speakers, len(SPEAKERS)
    ) - 1


def generate(
    seed: int = DEFAULT_SEED,
    n_units: int = 10_000,
    vocab_size: int = 2_000,
    df_skew: float = 1.2,
    *,
    avg_len: int = 40,
    n_probes: int = 12,
    n_speakers: int = 8,
    session_len: int = 48,
) -> Dict[str, Any]:
    """Generate a scale twin corpus.

    ``df_skew`` is the Zipf exponent applied to vocabulary ranks: larger
    values steepen the head/tail split. The dominant speaker canons are
    overlaid on top (own-canon and cross-speaker mention probabilities
    times speaker share × ``n_units``) so the corpus carries the
    speaker-dominance pathology by construction.
    """
    if not isinstance(seed, int):
        raise ValueError(f"seed must be int, got {seed!r}")
    if n_units < 1:
        raise ValueError(f"n_units must be >= 1, got {n_units!r}")
    if vocab_size < 8:
        raise ValueError(f"vocab_size must be >= 8, got {vocab_size!r}")
    if not (0.2 <= df_skew <= 4.0):
        raise ValueError(f"df_skew out of range [0.2, 4.0]: {df_skew!r}")
    if avg_len < 4:
        raise ValueError(f"avg_len must be >= 4, got {avg_len!r}")
    if n_probes < 0 or n_probes > len(PROBE_TOPICS):
        raise ValueError(
            f"n_probes must be 0..{len(PROBE_TOPICS)}, got {n_probes!r}"
        )
    n_speakers = max(2, min(n_speakers, len(SPEAKERS)))

    rng = random.Random(seed)
    cum, terms = _build_sampler(rng, vocab_size, df_skew)

    # Probe placements: one evidence unit per (speaker, unique topic),
    # assigned to deterministic unit slots spread through the timeline.
    probe_slots: Dict[int, Tuple[int, str, str]] = {}
    for j in range(n_probes):
        speaker = SPEAKERS[j % max(2, n_speakers // 2)]  # dominant half
        topic = PROBE_TOPICS[j]
        slot = rng.randrange(n_units)
        while slot in probe_slots:
            slot = (slot + 1) % n_units
        probe_slots[slot] = (j, speaker, topic)

    units: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    df: Dict[str, int] = {}

    sess_seq = -1
    session_id = ""
    us = BASE_US
    for seq in range(n_units):
        if seq % session_len == 0:
            sess_seq += 1
            session_id = f"sess-{sess_seq:05d}"
            us += rng.randint(6, 30) * HOUR_US if seq else 0
        us += rng.randint(90, 2400) * 1_000_000

        speaker, sidx = _pick_speaker(rng, n_speakers)

        if seq in probe_slots:
            _pj, spk, topic = probe_slots[seq]
            text = (
                f"I {_PROBE_VERBS[seq % len(_PROBE_VERBS)]} the {topic} "
                f"{_PROBE_DETAILS[seq % len(_PROBE_DETAILS)]}."
            )
            speaker = spk
            kind = "turn"
            meta: Optional[Dict[str, Any]] = {"probe_evidence": True}
        else:
            length = max(
                4, int(avg_len * rng.expovariate(1.0)) + avg_len // 2
            )
            bag = set()
            for _ in range(length):
                bag.add(_sample_term(rng, cum, terms))
            # speaker-name drop-ins drive dominant-canon df (D7-07):
            # own speaker canon plus occasional references to others
            if rng.random() < _SPEAKER_SELF_P:
                bag.add(f"name:{speaker.lower()}")
            if rng.random() < _SPEAKER_MENTION_P:
                _, midx = _pick_speaker(rng, n_speakers)
                bag.add(f"name:{SPEAKERS[midx].lower()}")
            text = " ".join(sorted(bag))
            kind = "turn"
            meta = None

        # df bookkeeping over text tokens (probe topics count too)
        for tok in set(text.split()):
            df[tok] = df.get(tok, 0) + 1

        uid = f"u-{seq:07d}"
        units.append(
            {
                "id": uid,
                "kind": kind,
                "speaker": speaker,
                "session_id": session_id,
                "text": text,
                "occurred_us": us,
                "perspective": "user_stated",
                **({"meta": meta} if meta else {}),
            }
        )
        if seq in probe_slots:
            probe_j, spk, topic = probe_slots[seq]
            tasks.append(
                {
                    "task_id": f"scale-probe-{probe_j:04d}",
                    "kind": "speaker_topic_probe",
                    "query": f"What did {spk} say about {topic}?",
                    "task_text": f"What did {spk} say about {topic}?",
                    "gold_unit_ids": [uid],
                    "expected_abstain": False,
                    "meta": {
                        "speaker": spk,
                        "topic": topic,
                        "speaker_df_planned": True,
                        "topic_df_planned": 1,
                    },
                }
            )

    # realized-df summary over vocab terms + speaker canons
    vocab_dfs = sorted(
        (df.get(t, 0) for t in terms), reverse=True
    )
    speaker_df = {
        s: df.get(f"name:{s.lower()}", 0) for s in SPEAKERS[:n_speakers]
    }
    n = len(units)
    median_df = vocab_dfs[len(vocab_dfs) // 2] if vocab_dfs else 0
    corpus = {
        "name": CORPUS_NAME,
        "generator": GENERATOR_ID,
        "constants": CONSTANTS_TAG,
        "seed": seed,
        "params": {
            "n_units": n_units,
            "vocab_size": vocab_size,
            "df_skew": df_skew,
            "avg_len": avg_len,
            "n_probes": n_probes,
            "n_speakers": n_speakers,
            "session_len": session_len,
        },
        "units": units,
        "tasks": tasks,
        "stats": {
            "n_units": n,
            "n_sessions": sess_seq + 1,
            "vocab_size": vocab_size,
            "df_top10": vocab_dfs[:10],
            "df_median": median_df,
            "df_p90_rank_pct": (
                sum(1 for d in vocab_dfs if d <= max(1, n // 100))
                / max(1, len(vocab_dfs))
            ),
            "speaker_df": speaker_df,
            "span_days": (
                (units[-1]["occurred_us"] - units[0]["occurred_us"]) // DAY_US
                if units
                else 0
            ),
        },
    }
    corpus["digest"] = corpus_digest(corpus)
    return corpus


__all__ = [
    "GENERATOR_ID",
    "CONSTANTS_TAG",
    "CORPUS_NAME",
    "DEFAULT_SEED",
    "PROBE_TOPICS",
    "SPEAKERS",
    "corpus_digest",
    "generate",
]
