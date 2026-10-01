"""Durable tests for the V7.5 eval infrastructure — SPEC_V7_5
V75-04.06 (LoCoMo observation oracle), V75-04.07 (dev/test partition
machinery + registry path extension); acceptance scenarios J15, J16,
J20; SPEC_V7 V7-22.08/22.09/22.18 and the V7-13.19 oracle ceiling.

Coverage:

* **Oracle (V75-04.06)** — ``eval.v7.locomo_oracle`` extracts the
  released ``session_N_observation`` ``[assertion, "D<s>:<t>"]`` pairs
  into typed :class:`OracleFact`s (speaker / session / normalized refs
  / corpus-qualified item ids), normalizing the file's messy ref forms
  (lists, comma-joined strings) through the same parser QA evidence
  uses; unparseable entries land in ``dropped``, never guessed.
  ``fact_coverage`` estimates ``p`` in both modes.
* **Tripwire (J15 / V7-22.18)** — the oracle is eval-only: no runtime
  ``verbatim/`` module may import it (or any ``eval`` module), and no
  runtime source may name it or a benchmark dataset.
* **Partition (V75-04.07 / V7-22.08)** — deterministic hash partition
  over synthetic ``sample_id``s with a recorded digest; nothing here
  materializes a partition of the gated file (O1 pending — ids only).
* **Integrity (J16 / V7-22.09)** — a tuning artifact declaring a
  test-partition ``sample_id`` fails ``assert_not_tuned_on_test``.
* **Registry (J20)** — the extended ``locomo10.json`` path list
  resolves and the pinned sha256 is verified before loading; a
  wrong-hash file is rejected.

The bundle copy of ``locomo10.json`` is local-only (CC BY-NC 4.0,
gitignored): file-touching tests skip when it is absent, and nothing
here ever touches the network.
"""

from __future__ import annotations

import ast
import json
import os
import re

import pytest

from eval.v7 import corpora, dataset_registry as reg
from eval.v7 import locomo_oracle  # noqa: F401 — J15: loadable in eval/


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
VERBATIM_DIR = os.path.join(REPO_ROOT, "verbatim")
BUNDLE_REL = "research/v7_formula_search/locomo10.json"
BUNDLE_PATH = os.path.join(REPO_ROOT, BUNDLE_REL)
BUNDLE_PRESENT = os.path.isfile(BUNDLE_PATH)
LOCOMO_ENV = {"VERBATIM_EVAL_LOCOMO": "1"}

ORACLE_MODULE = "eval.v7.locomo_oracle"


# ---------------------------------------------------------------------------
# synthetic conversation fixture — released-file shape, messy refs included
# ---------------------------------------------------------------------------

SYNTH_CONV = {
    "sample_id": "conv-t1",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I run daily."},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice habit."},
            {"dia_id": "D1:3", "speaker": "Alice", "text": "I joined a club."},
        ],
        "session_1_date_time": "1:00 pm on 1 May, 2023",
        "session_2": [
            {"dia_id": "D2:1", "speaker": "Alice", "text": "Marathon soon."},
            {"dia_id": "D2:2", "speaker": "Bob", "text": "Good luck."},
        ],
        "session_2_date_time": "2:00 pm on 8 May, 2023",
    },
    "observation": {
        "session_1_observation": {
            "Alice": [
                ["Alice runs every day.", "D1:1"],
                # released list-of-refs form
                ["Alice joined a running club.", ["D1:3", "D1:1"]],
            ],
            "Bob": [["Bob encouraged Alice.", "D1:2"]],
        },
        "session_2_observation": {
            # released comma-joined ref form
            "Alice": [["Alice entered a marathon.", "D2:1, D2:2"]],
            # malformed entries → dropped, never guessed
            "Carol": "not-a-list",
            "Dave": [["missing ref element"]],
            "Eve": [["Eve bikes to work.", "not-a-ref"]],
        },
        "not_an_observation_key": {"Zed": [["lost.", "D1:1"]]},
    },
    "qa": [],
}


def _synth_corpus_with_oracle() -> corpora.Corpus:
    """A minimal Corpus carrying the raw observation layer — what
    ``load_locomo`` produces in ``metadata`` (V75-04.06)."""
    items = tuple(
        corpora.CorpusItem(
            id=f"conv-t1:D{s}:{t}",
            text="x",
            speaker="Alice",
            session_id=f"conv-t1/session_{s}",
            group_id="conv-t1",
        )
        for s, t in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2))
    )
    return corpora.Corpus(
        name="synth",
        dataset_id="locomo",
        items=items,
        tasks=(),
        metadata={
            corpora.LOCOMO_ORACLE_META_KEY: {
                "conv-t1": SYNTH_CONV["observation"]
            }
        },
    )


# ---------------------------------------------------------------------------
# oracle extraction (V75-04.06)
# ---------------------------------------------------------------------------


class TestOracleExtraction:
    def test_pairs_extracted_typed(self):
        oracle = locomo_oracle.oracle_from_conversations([SYNTH_CONV])
        facts = {f.fact_id: f for f in oracle.facts}
        # Alice s1 (2) + Bob s1 (1) + Alice s2 (1) + Eve (1, unresolved)
        assert len(oracle.facts) == 5
        assert oracle.dataset_id == "locomo"
        a0 = oracle.facts[0]
        assert a0.sample_id == "conv-t1"
        assert a0.session_n == 1
        assert a0.speaker == "Alice"
        assert a0.assertion == "Alice runs every day."
        assert a0.evidence_refs == ("D1:1",)
        assert a0.evidence_item_ids == ("conv-t1:D1:1",)
        assert a0.fact_id.startswith("conv-t1#obs")

    def test_messy_ref_forms_normalized(self):
        oracle = locomo_oracle.oracle_from_conversations([SYNTH_CONV])
        by_assert = {f.assertion: f for f in oracle.facts}
        club = by_assert["Alice joined a running club."]
        assert club.evidence_refs == ("D1:3", "D1:1")  # list form
        mar = by_assert["Alice entered a marathon."]
        assert mar.evidence_refs == ("D2:1", "D2:2")  # comma-joined form
        assert mar.session_n == 2
        eve = by_assert["Eve bikes to work."]
        assert eve.evidence_refs == ()
        assert eve.unresolved_refs == ("not-a-ref",)  # recorded, loud

    def test_unparseable_entries_dropped_not_guessed(self):
        oracle = locomo_oracle.oracle_from_conversations([SYNTH_CONV])
        # Carol (non-list table value), Dave (1-element pair),
        # not_an_observation_key (key outside the session_N pattern)
        assert len(oracle.dropped) == 3

    def test_oracle_from_corpus_metadata(self):
        corpus = _synth_corpus_with_oracle()
        oracle = locomo_oracle.oracle_from_corpus(corpus)
        direct = locomo_oracle.oracle_from_conversations([SYNTH_CONV])
        assert [f.fact_id for f in oracle.facts] == [
            f.fact_id for f in direct.facts
        ]
        assert oracle.digest() == direct.digest()

    def test_oracle_from_corpus_absent_layer_is_empty(self):
        corpus = corpora.Corpus(
            name="x", dataset_id="owned_scale", items=(), tasks=()
        )
        oracle = locomo_oracle.oracle_from_corpus(corpus)
        assert oracle.facts == () and oracle.dropped == ()

    def test_digest_deterministic_and_content_sensitive(self):
        o1 = locomo_oracle.oracle_from_conversations([SYNTH_CONV])
        o2 = locomo_oracle.oracle_from_conversations(
            [json.loads(json.dumps(SYNTH_CONV))]
        )
        assert o1.digest() == o2.digest()
        alt = json.loads(json.dumps(SYNTH_CONV))
        alt["observation"]["session_1_observation"]["Alice"][0][0] = (
            "Alice runs weekly."
        )
        o3 = locomo_oracle.oracle_from_conversations([alt])
        assert o3.digest() != o1.digest()

    def test_for_sample_and_speakers(self):
        oracle = locomo_oracle.oracle_from_conversations([SYNTH_CONV])
        assert len(oracle.for_sample("conv-t1")) == len(oracle.facts)
        assert oracle.for_sample("nobody") == ()
        assert set(oracle.speakers()) >= {"Alice", "Bob", "Eve"}


class TestFactCoverage:
    def _oracle(self):
        return locomo_oracle.oracle_from_conversations([SYNTH_CONV])

    def test_id_mode_coverage(self):
        oracle = self._oracle()
        # retaining D1:1 covers "Alice runs every day." and the
        # multi-ref club fact (any-one-ref-is-enough)
        cov = locomo_oracle.fact_coverage(
            oracle.facts, covered_item_ids={"conv-t1:D1:1"}
        )
        assert cov.total == 5 and cov.covered == 2 and cov.p == 0.4
        assert cov.mode == "evidence_ids"
        assert set(cov.covered_ids) | set(cov.uncovered_ids) == {
            f.fact_id for f in oracle.facts
        }

    def test_text_mode_verbatim_containment(self):
        oracle = self._oracle()
        cov = locomo_oracle.fact_coverage(
            oracle.facts,
            covered_texts=["Alice runs every day without fail."],
            min_token_overlap=1.0,
        )
        # assertion tokens ⊆ text tokens for exactly one fact
        assert cov.covered == 1 and cov.p == 0.2
        assert cov.mode == "token_overlap"

    def test_text_mode_partial_threshold(self):
        oracle = self._oracle()
        strict = locomo_oracle.fact_coverage(
            oracle.facts,
            covered_texts=["Alice runs daily."],  # missing "every"
            min_token_overlap=1.0,
        )
        loose = locomo_oracle.fact_coverage(
            oracle.facts,
            covered_texts=["Alice runs daily."],
            min_token_overlap=0.5,
        )
        assert strict.covered == 0
        assert loose.covered == 1  # 3/4 assertion tokens ≥ 0.5

    def test_modes_union(self):
        oracle = self._oracle()
        cov = locomo_oracle.fact_coverage(
            oracle.facts,
            covered_item_ids={"conv-t1:D1:2"},
            covered_texts=["Alice entered a marathon."],
        )
        assert cov.covered == 2 and cov.mode == "evidence_ids+token_overlap"

    def test_empty_fact_set_reports_undefined_not_zero(self):
        cov = locomo_oracle.fact_coverage(
            (), covered_item_ids={"x:D1:1"}
        )
        assert cov.total == 0 and cov.covered == 0 and cov.p is None

    def test_bad_threshold_rejected(self):
        with pytest.raises(ValueError):
            locomo_oracle.fact_coverage(
                self._oracle().facts,
                covered_texts=["x"],
                min_token_overlap=1.5,
            )


# ---------------------------------------------------------------------------
# J15 / V7-22.18 tripwire — the oracle never reaches runtime code
# ---------------------------------------------------------------------------


def _runtime_py_files():
    for dp, _dirs, fns in os.walk(VERBATIM_DIR):
        for fn in sorted(fns):
            if fn.endswith(".py"):
                yield os.path.join(dp, fn)


def _import_roots(path: str):
    """Module names imported by a source file (AST — imports only)."""
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module


#: Benchmark dataset names that must never appear in runtime code
#: (V7-22.18: "dataset names inside runtime (non-eval) code" fail the
#: build; runtime may not branch on dataset identity).
_DATASET_NAME_RE = re.compile(
    r"\b(locomo10?|longmemeval|dolphinbench|halumem|memoryagentbench|"
    r"convomem|locomo_oracle)\b",
    re.IGNORECASE,
)


class TestOracleIsolation:
    """J15: oracle loadable under ``eval/`` (imported at this module's
    top) and never importable from runtime modules."""

    def test_no_runtime_module_imports_eval(self):
        offenders = []
        for path in _runtime_py_files():
            for mod in _import_roots(path):
                if mod == "eval" or mod.startswith("eval."):
                    offenders.append((path, mod))
        assert offenders == [], (
            "runtime modules importing eval code (V7-22.18): "
            f"{offenders}"
        )

    def test_oracle_module_never_named_in_runtime(self):
        """Catches the string form too — ``import_module("eval.v7.
        locomo_oracle")`` and friends bypass the AST import scan."""
        offenders = []
        for path in _runtime_py_files():
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            if "locomo_oracle" in src or ORACLE_MODULE in src:
                offenders.append(path)
        assert offenders == [], f"runtime files naming the oracle: {offenders}"

    def test_no_dataset_names_in_runtime_code(self):
        offenders = []
        for path in _runtime_py_files():
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            m = _DATASET_NAME_RE.search(src)
            if m:
                offenders.append((path, m.group(0)))
        assert offenders == [], (
            "benchmark dataset names in runtime code (V7-22.18): "
            f"{offenders}"
        )

    def test_oracle_module_is_eval_only_on_disk(self):
        import eval.v7.locomo_oracle as mod

        assert os.sep + "eval" + os.sep in os.path.abspath(mod.__file__)


# ---------------------------------------------------------------------------
# V75-04.07 partition machinery (V7-22.08) — synthetic ids, O1-safe
# ---------------------------------------------------------------------------


class TestPartition:
    IDS = tuple(f"conv-{i:03d}" for i in range(50))

    def test_deterministic_same_input_same_split_and_digest(self):
        p1 = reg.materialize_partition("locomo", self.IDS)
        p2 = reg.materialize_partition("locomo", self.IDS)
        assert p1.dev == p2.dev and p1.test == p2.test
        assert p1.digest == p2.digest

    def test_order_insensitive(self):
        p1 = reg.materialize_partition("locomo", self.IDS)
        p2 = reg.materialize_partition("locomo", tuple(reversed(self.IDS)))
        assert p1.digest == p2.digest and p1.dev == p2.dev

    def test_partition_complete_and_disjoint(self):
        p = reg.materialize_partition("locomo", self.IDS)
        assert set(p.dev) & set(p.test) == set()
        assert set(p.groups) == set(self.IDS)
        assert p.dev and p.test  # 40/60 over 50 ids lands both sides
        assert p.digest == reg.split_digest("locomo", self.IDS)

    def test_consistent_with_split_for_group(self):
        p = reg.materialize_partition("locomo", self.IDS)
        for g in self.IDS:
            assert p.assignment(g) == reg.split_for_group("locomo", g)
        assert p.assignment("never-partitioned") is None

    def test_different_seed_repartitions(self):
        spec_alt = reg.SplitSpec(
            unit="conversation", dev_pct=40, tag="v7-split-alt-seed"
        )
        p1 = reg.materialize_partition("locomo", self.IDS)
        p2 = reg.materialize_partition("locomo", self.IDS, spec=spec_alt)
        assert p1.digest != p2.digest
        # the seed actually changes assignments on this fixture set
        assert p1.dev != p2.dev

    def test_fraction_monotone_in_dev_pct(self):
        # bucket < dev_pct is monotone: a larger dev fraction can only
        # grow the dev set — deterministic, no hash luck involved.
        spec40 = reg.SplitSpec(unit="conversation", dev_pct=40)
        spec80 = reg.SplitSpec(unit="conversation", dev_pct=80)
        p40 = reg.materialize_partition("locomo", self.IDS, spec=spec40)
        p80 = reg.materialize_partition("locomo", self.IDS, spec=spec80)
        assert set(p40.dev) <= set(p80.dev)
        assert p40.digest != p80.digest

    def test_custom_spec_without_registry_entry(self):
        spec = reg.SplitSpec(
            unit="conversation", dev_pct=50, tag="synthetic-seed"
        )
        p = reg.materialize_partition("synthetic-ds", self.IDS, spec=spec)
        assert p.dataset_id == "synthetic-ds"
        assert set(p.groups) == set(self.IDS)

    def test_round_trip_dict(self):
        p = reg.materialize_partition("locomo", self.IDS)
        d = p.to_dict()
        assert d["digest"] == p.digest
        assert d["spec"]["unit"] == "conversation"
        assert sorted(d["dev"] + d["test"]) == sorted(self.IDS)


# ---------------------------------------------------------------------------
# J16 / V7-22.09 — tuning-artifact integrity
# ---------------------------------------------------------------------------


class TestTuningIntegrity:
    def _partition(self):
        return reg.materialize_partition(
            "locomo", [f"conv-{i:03d}" for i in range(50)]
        )

    def test_contaminated_manifest_fails(self):
        p = self._partition()
        assert p.test, "fixture: expected a non-empty test split"
        manifest = {
            "artifact": "bm25f/v2",
            "fitted_groups": list(p.dev) + [p.test[0]],
            "dev_split_digest": p.digest,
        }
        with pytest.raises(reg.SplitIntegrityError) as ei:
            reg.assert_not_tuned_on_test(manifest, p)
        assert p.test[0] in str(ei.value)

    def test_clean_manifest_passes(self):
        p = self._partition()
        reg.assert_not_tuned_on_test(
            {"fitted_groups": list(p.dev), "dev_split_digest": p.digest},
            p,
        )
        # digest-only provenance also satisfies V7-22.09
        reg.assert_not_tuned_on_test({"dev_split_digest": p.digest}, p)

    def test_wrong_partition_digest_fails(self):
        p = self._partition()
        other = reg.materialize_partition(
            "locomo",
            ["conv-000"],
            spec=reg.SplitSpec(unit="conversation", dev_pct=40),
        )
        with pytest.raises(reg.SplitIntegrityError):
            reg.assert_not_tuned_on_test(
                {"dev_split_digest": other.digest}, p
            )

    def test_group_outside_partition_universe_fails(self):
        p = self._partition()
        with pytest.raises(reg.SplitIntegrityError) as ei:
            reg.assert_not_tuned_on_test(
                {"fitted_groups": ["conv-999-not-in-corpus"]}, p
            )
        assert "outside" in str(ei.value)

    def test_no_provenance_fails_closed(self):
        p = self._partition()
        with pytest.raises(reg.SplitIntegrityError) as ei:
            reg.assert_not_tuned_on_test({"artifact": "rerank/v2"}, p)
        assert "provenance" in str(ei.value)


# ---------------------------------------------------------------------------
# J20 — registry resolves the extended locomo10 path, sha256 verified
# ---------------------------------------------------------------------------


class TestRegistryPaths:
    def test_path_list_extended_not_replaced(self):
        e = reg.get("locomo")
        assert "/tmp/locomo/locomo10.json" in e.paths
        assert BUNDLE_REL in e.paths
        assert e.paths[0] == "/tmp/locomo/locomo10.json"  # original kept

    def test_wrong_hash_file_rejected(self, tmp_path):
        bad = tmp_path / "locomo10.json"
        bad.write_text(json.dumps([{"sample_id": "fake"}]))
        env = {
            "VERBATIM_EVAL_LOCOMO": "1",
            "VERBATIM_EVAL_LOCOMO_PATH": str(bad),
        }
        assert reg.status("locomo", env=env) == reg.unavailable(
            "sha256-mismatch"
        )

    def test_env_override_missing_is_missing(self):
        env = {
            "VERBATIM_EVAL_LOCOMO": "1",
            "VERBATIM_EVAL_LOCOMO_PATH": "/nonexistent/locomo10.json",
        }
        assert reg.status("locomo", env=env) == reg.STATUS_MISSING

    @pytest.mark.skipif(not BUNDLE_PRESENT, reason="bundle copy absent")
    def test_extended_path_resolves_and_hash_verifies(self):
        e = reg.get("locomo")
        resolved = reg.resolved_path(e, LOCOMO_ENV)
        assert resolved is not None
        # whichever configured path won, the pin is enforced before use
        assert reg.file_sha256(resolved) == e.sha256
        if not os.path.isfile("/tmp/locomo/locomo10.json"):
            assert resolved.endswith(BUNDLE_REL)
        assert reg.status("locomo", env=LOCOMO_ENV) == reg.STATUS_AVAILABLE

    @pytest.mark.skipif(not BUNDLE_PRESENT, reason="bundle copy absent")
    def test_oracle_builds_end_to_end_from_loaded_corpus(self):
        """J15's loadable half + V75-04.06 on the real gated file:
        registry resolution → loader → metadata carry → typed oracle."""
        corpus = corpora.load_corpus("locomo", env=LOCOMO_ENV)
        assert corpora.LOCOMO_ORACLE_META_KEY in corpus.metadata
        oracle = locomo_oracle.oracle_from_corpus(corpus)
        assert len(oracle.facts) >= 2500
        assert oracle.dropped == ()
        assert oracle.digest()
        # every resolvable ref lands in the corpus item-id space
        item_ids = set(corpus.item_by_id())
        n_refs = 0
        for f in oracle.facts:
            for r in f.evidence_item_ids:
                assert r in item_ids
                n_refs += 1
        assert n_refs > 2500
        # coverage smoke: retaining all items covers every ref'd fact
        cov = locomo_oracle.fact_coverage(
            oracle.facts, covered_item_ids=item_ids
        )
        assert cov.covered == sum(
            1 for f in oracle.facts if f.evidence_item_ids
        )
