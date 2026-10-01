"""Tests for the w-twin-extra owned generators (V7-22.05, V7-20.06,
V7-24.10): ``twins_actions``, ``twins_prefs``, ``twins_scale``.

Covers: seeded determinism (identical inputs → byte-identical digest),
planted-rule cue sanity (§32.13), gold referential integrity, the
``preference`` annotation contract (V7-24.11), and the scale generator's
controlled df skew (D7-07 fixture at size).
"""

from __future__ import annotations

import json
import re
import unittest

from eval.v7 import twins_actions, twins_prefs, twins_scale


def _units_by_id(corpus):
    return {u["id"]: u for u in corpus["units"]}


def _word_hit(text: str, lexeme: str) -> bool:
    return re.search(r"\b" + re.escape(lexeme) + r"\b", text, re.I) is not None


class TestActionsTwin(unittest.TestCase):
    def setUp(self):
        self.corpus = twins_actions.generate(
            seed=7, n_implicit=20, n_compliance=10, n_procedure=8,
            n_action_lookup=6, filler_sessions=5,
        )

    def test_deterministic(self):
        a = twins_actions.generate(
            seed=7, n_implicit=20, n_compliance=10, n_procedure=8,
            n_action_lookup=6, filler_sessions=5,
        )
        self.assertEqual(a["digest"], self.corpus["digest"])
        self.assertEqual(
            json.dumps(a["units"], sort_keys=True),
            json.dumps(self.corpus["units"], sort_keys=True),
        )

    def test_seed_varies(self):
        other = twins_actions.generate(seed=8, n_implicit=20)
        self.assertNotEqual(other["digest"], self.corpus["digest"])

    def test_json_serializable(self):
        json.dumps(self.corpus)

    def test_gold_refs_resolve(self):
        units = _units_by_id(self.corpus)
        for task in self.corpus["tasks"]:
            for uid in task["gold_unit_ids"]:
                self.assertIn(uid, units, task["task_id"])

    def test_task_ids_unique(self):
        ids = [t["task_id"] for t in self.corpus["tasks"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_planted_rules_carry_cues(self):
        """Every rule unit text matches ≥1 §32.13 cue lexeme."""
        rule_units = [
            u for u in self.corpus["units"]
            if "standing_rule" in u.get("meta", {})
        ]
        self.assertGreater(len(rule_units), 0)
        for u in rule_units:
            self.assertTrue(
                any(_word_hit(u["text"], cue) for cue in twins_actions.RULE_CUES),
                f"rule without §32.13 cue: {u['text']!r}",
            )
            meta = u["meta"]["standing_rule"]
            self.assertIn("rule_id", meta)
            self.assertIn("pattern", meta)
            self.assertTrue(meta["trigger_entities"])
            # rules are user/document perspective only (§32.13)
            self.assertEqual(u["perspective"], "user_stated")
            self.assertEqual(u["speaker"], "user")

    def test_filler_is_cue_free(self):
        """No non-rule unit may carry a §32.13 cue — filler would become
        false-positive normative statements."""
        for u in self.corpus["units"]:
            if "standing_rule" in u.get("meta", {}):
                continue
            for cue in twins_actions.RULE_CUES:
                self.assertFalse(
                    _word_hit(u["text"], cue),
                    f"{cue!r} leaked into {u['id']}: {u['text']!r}",
                )

    def test_implicit_tasks_shape(self):
        imp = [t for t in self.corpus["tasks"]
               if t["kind"] == "implicit_rule_recall"]
        self.assertEqual(len(imp), 20)
        units = _units_by_id(self.corpus)
        for t in imp:
            self.assertEqual(len(t["gold_unit_ids"]), 1)
            self.assertNotEqual(t["naive_action"], t["expected_action"])
            self.assertTrue(t["needs_history"])
            # cue-free: task text never contains the rule's condition phrase
            rule_meta = units[t["gold_unit_ids"][0]]["meta"]["standing_rule"]
            cond = rule_meta.get("condition")
            if cond:
                self.assertNotIn(cond, t["task_text"])
            # channel scoping is what the task must recall — absent from text
            for ent in rule_meta["trigger_entities"]:
                if ent.startswith("#"):
                    self.assertNotIn(ent, t["task_text"])

    def test_compliance_tasks_have_rule_and_action(self):
        cmp_ = [t for t in self.corpus["tasks"]
                if t["kind"] == "rule_compliance"]
        self.assertEqual(len(cmp_), 10)
        units = _units_by_id(self.corpus)
        kinds = set()
        for t in cmp_:
            self.assertEqual(len(t["gold_unit_ids"]), 2)
            rule_u = units[t["gold_unit_ids"][0]]
            act_u = units[t["gold_unit_ids"][1]]
            self.assertIn("standing_rule", rule_u["meta"])
            self.assertEqual(act_u["perspective"], "agent_action")
            self.assertGreater(
                act_u["occurred_us"], rule_u["occurred_us"],
                "the audited action must postdate its rule",
            )
            self.assertIsInstance(t["expect"]["compliant"], bool)
            kinds.add(t["expect"]["compliant"])
        # both outcomes represented
        self.assertEqual(kinds, {True, False})

    def test_procedure_tasks(self):
        procs = [t for t in self.corpus["tasks"]
                 if t["kind"] == "procedure_recall"]
        self.assertEqual(len(procs), 8)
        units = _units_by_id(self.corpus)
        for t in procs:
            self.assertGreaterEqual(len(t["gold_unit_ids"]), 3)
            steps = sorted(
                units[u]["meta"]["step"] for u in t["gold_unit_ids"]
            )
            self.assertEqual(steps, list(range(len(steps))))

    def test_tool_sequences_present(self):
        kinds = {u["kind"] for u in self.corpus["units"]}
        self.assertIn("tool_call", kinds)
        self.assertIn("tool_result", kinds)

    def test_default_counts_meet_spec(self):
        full = twins_actions.generate()
        self.assertGreaterEqual(len(full["tasks"]), 500)  # V7-24.10
        imp = [t for t in full["tasks"]
               if t["kind"] == "implicit_rule_recall"]
        self.assertGreaterEqual(len(imp), 200)  # V7-20.06


class TestPrefsTwin(unittest.TestCase):
    def setUp(self):
        self.corpus = twins_prefs.generate(
            seed=11, n_recall=20, n_apply=12, n_update=10,
            n_comparative=8, n_negative=6, filler_sessions=4,
        )

    def test_deterministic(self):
        a = twins_prefs.generate(
            seed=11, n_recall=20, n_apply=12, n_update=10,
            n_comparative=8, n_negative=6, filler_sessions=4,
        )
        self.assertEqual(a["digest"], self.corpus["digest"])
        self.assertEqual(
            json.dumps(a, sort_keys=True),
            json.dumps(self.corpus, sort_keys=True),
        )

    def test_seed_varies(self):
        other = twins_prefs.generate(seed=12, n_recall=20)
        self.assertNotEqual(other["digest"], self.corpus["digest"])

    def test_every_task_has_preference_annotation(self):
        for t in self.corpus["tasks"]:
            self.assertIn("preference", t, t["task_id"])
            self.assertEqual(t["preference"]["subject"], "user")

    def test_gold_refs_resolve(self):
        units = _units_by_id(self.corpus)
        for t in self.corpus["tasks"]:
            for uid in t["gold_unit_ids"]:
                self.assertIn(uid, units, t["task_id"])
            for uid in t.get("historical_ids", ()):
                self.assertIn(uid, units, t["task_id"])
            for uid in t.get("distractor_ids", ()):
                self.assertIn(uid, units, t["task_id"])

    def test_pref_unit_metadata(self):
        pref_units = [
            u for u in self.corpus["units"] if "pref" in u.get("meta", {})
        ]
        self.assertGreater(len(pref_units), 0)
        for u in pref_units:
            p = u["meta"]["pref"]
            self.assertIn(p["slot"], twins_prefs.SLOTS)
            self.assertIn(
                p["strength"],
                twins_prefs.STRENGTHS + ("excluded",),
            )
            self.assertTrue(p["object_text"])

    def test_filler_is_lexeme_free(self):
        """Filler must not carry §32.12 lexemes — it would read as a
        preference the extractor should have caught."""
        for u in self.corpus["units"]:
            if "pref" in u.get("meta", {}):
                continue
            for lex in twins_prefs.PREF_LEXEMES:
                self.assertFalse(
                    _word_hit(u["text"], lex),
                    f"{lex!r} leaked into {u['id']}: {u['text']!r}",
                )

    def test_update_tasks_supersede(self):
        upd = [t for t in self.corpus["tasks"]
               if t["kind"] == "preference_update"]
        self.assertEqual(len(upd), 10)
        units = _units_by_id(self.corpus)
        for t in upd:
            self.assertEqual(len(t["gold_unit_ids"]), 1)
            self.assertEqual(len(t["historical_ids"]), 1)
            new_u = units[t["gold_unit_ids"][0]]
            old_u = units[t["historical_ids"][0]]
            self.assertGreater(new_u["occurred_us"], old_u["occurred_us"])
            self.assertNotEqual(
                new_u["meta"]["pref"]["object_text"],
                old_u["meta"]["pref"]["object_text"],
            )
            self.assertEqual(
                t["expect"]["current_object"],
                new_u["meta"]["pref"]["object_text"],
            )

    def test_negative_tasks_abstain(self):
        neg = [t for t in self.corpus["tasks"]
               if t["kind"] == "preference_negative"]
        self.assertEqual(len(neg), 6)
        for t in neg:
            self.assertTrue(t["expected_abstain"])
            self.assertEqual(t["gold_unit_ids"], [])
            self.assertEqual(len(t["distractor_ids"]), 1)
            self.assertIn(
                t["preference"]["excluded_form"],
                ("hypothetical", "quoted", "hedged", "negated_hypothetical"),
            )

    def test_apply_tasks_multi_gold(self):
        app = [t for t in self.corpus["tasks"]
               if t["kind"] == "preference_apply"]
        self.assertEqual(len(app), 12)
        for t in app:
            self.assertGreaterEqual(len(t["gold_unit_ids"]), 1)
            self.assertTrue(t["preference"]["slots"])

    def test_default_counts_meet_spec(self):
        full = twins_prefs.generate()
        self.assertGreaterEqual(len(full["tasks"]), 300)  # V7-24.10


class TestScaleTwin(unittest.TestCase):
    def test_requested_counts(self):
        for n in (100, 1000, 5000):
            corpus = twins_scale.generate(
                seed=3, n_units=n, vocab_size=300, df_skew=1.2, n_probes=4,
            )
            self.assertEqual(len(corpus["units"]), n)
            self.assertEqual(len(corpus["tasks"]), 4)

    def test_deterministic(self):
        a = twins_scale.generate(seed=3, n_units=800, vocab_size=200)
        b = twins_scale.generate(seed=3, n_units=800, vocab_size=200)
        self.assertEqual(a["digest"], b["digest"])
        c = twins_scale.generate(seed=4, n_units=800, vocab_size=200)
        self.assertNotEqual(a["digest"], c["digest"])

    def test_vocab_size_respected(self):
        corpus = twins_scale.generate(seed=5, n_units=400, vocab_size=50)
        pat = re.compile(r"^w(\d{5})$")
        for u in corpus["units"]:
            for tok in u["text"].split():
                m = pat.match(tok)
                if m:
                    self.assertLess(int(m.group(1)), 50)

    def test_df_skewed(self):
        """A few terms dominate; the long tail sits at df≈1 (D7-07)."""
        n = 3000
        corpus = twins_scale.generate(
            seed=9, n_units=n, vocab_size=600, df_skew=1.3,
            n_probes=0, avg_len=30,
        )
        stats = corpus["stats"]
        top = stats["df_top10"][0]
        median = stats["df_median"]
        self.assertGreaterEqual(top, n // 5)
        self.assertLessEqual(median, n // 20)
        self.assertGreater(top, median * 10)
        # ≥ 55% of vocab terms appear in ≤ 1% of units (long tail)
        self.assertGreater(stats["df_p90_rank_pct"], 0.55)
        # dominant speaker canons saturate a large share of units
        speaker_df = stats["speaker_df"]
        self.assertGreater(max(speaker_df.values()), n // 5)

    def test_skew_parameter_monotone(self):
        """Steeper df_skew ⇒ heavier head relative to tail."""
        n = 2000
        flat = twins_scale.generate(
            seed=13, n_units=n, vocab_size=400, df_skew=0.4, n_probes=0,
        )
        steep = twins_scale.generate(
            seed=13, n_units=n, vocab_size=400, df_skew=2.0, n_probes=0,
        )
        # with a steep exponent the head term saturates while the flat
        # distribution spreads occurrences more evenly
        self.assertGreater(
            steep["stats"]["df_top10"][0], flat["stats"]["df_top10"][0]
        )
        self.assertGreaterEqual(
            steep["stats"]["df_p90_rank_pct"],
            flat["stats"]["df_p90_rank_pct"],
        )

    def test_probes_planted_and_gold_resolves(self):
        corpus = twins_scale.generate(
            seed=17, n_units=600, vocab_size=150, n_probes=6,
        )
        units = _units_by_id(corpus)
        probes = [t for t in corpus["tasks"]
                  if t["kind"] == "speaker_topic_probe"]
        self.assertEqual(len(probes), 6)
        for t in probes:
            (uid,) = t["gold_unit_ids"]
            u = units[uid]
            self.assertIn(t["meta"]["topic"], u["text"])
            self.assertEqual(u["speaker"], t["meta"]["speaker"])
            self.assertTrue(u["meta"]["probe_evidence"])
            # topic df is exactly 1 — the probe is findable only via it
            df = {}
            for uu in corpus["units"]:
                for tok in set(uu["text"].split()):
                    df[tok] = df.get(tok, 0) + 1
            for tok in t["meta"]["topic"].split():
                self.assertEqual(df.get(tok, 0), 1, tok)

    def test_param_validation(self):
        with self.assertRaises(ValueError):
            twins_scale.generate(seed=1, n_units=0, vocab_size=50)
        with self.assertRaises(ValueError):
            twins_scale.generate(seed=1, n_units=10, vocab_size=4)
        with self.assertRaises(ValueError):
            twins_scale.generate(
                seed=1, n_units=10, vocab_size=50, df_skew=9.0
            )


if __name__ == "__main__":
    unittest.main()
