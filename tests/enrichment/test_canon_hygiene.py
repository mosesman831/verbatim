"""V8 §13 write-path and canon hygiene — scenarios K74–K78 (SPEC_V8 §13,
§24, §25).

Everything here exercises the real machinery, never stubs:

* **K74** drives the real ``verbatim_claims`` evaluation arm — a real
  ``Memory`` built with ``admission.require_review=False``, real
  ``add`` → durable-queue drain (``harvest``/``source_project``/
  ``source_embed``/``admit`` jobs succeed) → real ``claims``/
  ``claim_revisions``/``reviews`` rows. The product default
  ``require_review=True`` is asserted unchanged, the override is
  asserted in the arm's recorded ``config_overrides``, and the
  observable admission difference is asserted on review rows: with the
  blanket gate off, no review may carry the ``review_required`` reason
  (``policy.py`` — that park is unreachable when the flag is off).
* **K75** runs ``consolidate_scope_v7`` over a real store seeded with
  real ``units``/``state_facts``/``preferences``/``events_v7`` rows and
  byte-pinned ``source_revisions`` payloads; the V8-13.02 per-run
  ``stats["trace"]`` is asserted structurally — slots considered,
  candidates, gate-named rejections, writes — and the emitted
  ``observations_v7`` rows carry quote-pinned support refs.
* **K76** exercises ``entities_v2`` canon hygiene (V8-13.03,
  ``EXTRACTOR_ID = "entities/v2.1"``): apostrophe-contraction tails end
  capitalized runs ("Can't"/"You're"/"Don't" never mint ``can t``/
  ``you re``/``don t`` canons), owned vocatives never join a run
  ("Thanks Nate" → ``nate``), and the whole thing is re-verified at row
  level through the real ``project_units_v7`` write path.
* **K77** asserts the V8-13.04 subject-key contract on real extraction
  output AND on persisted rows: known speaker → ``<canon>/<slot>``
  (``speaker:alice/home_city`` at the extractor contract level;
  ``alice/home_city`` for a source declared speaker), unattributed
  input → ``me/<slot>``.
* **K78** replays ``project_units_v7`` on the same
  ``(source_id, revision, generation)`` slice and asserts every content
  table is byte-identical — including the consolidation-derived
  ``observations_v7``/``profiles_v7`` rows (the replay regression the
  ``consolidation_seen_v7`` un-seen in ``_delete_slice`` fixes). The
  only tolerated deltas are named volatile structures: the
  consolidation cursor's run bookkeeping (``run_seq``/timestamps),
  ``consolidation_seen_v7.seen_run``, and FTS5 shadow pages —
  SQLite-internal index structures, not logical content.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.core.time import now_us  # noqa: E402
from verbatim.core.types_v7 import IntervalUs  # noqa: E402
from verbatim.enrichment.entities_v2 import (  # noqa: E402
    EXTRACTOR_ID,
    extract_mentions,
)
from verbatim.enrichment.prefs_state import (  # noqa: E402
    extract_preferences,
    extract_state_facts,
)
from verbatim.observations.consolidate_v7 import (  # noqa: E402
    consolidate_scope_v7,
    ensure_consolidation_v7,
)
from verbatim.storage.repos import has_table  # noqa: E402
from verbatim.storage.schema_v7 import ensure_v7_additive  # noqa: E402
from verbatim.text.norm_v2 import analyze  # noqa: E402

GEN = 1
SCOPE = "s1"
NOW = 1_800_000_000_000_000

#: FTS5 shadow-table suffixes — internal index pages, not logical rows.
_FTS5_SHADOW = re.compile(r"_(data|idx|seg|segdir|docsize|stat|config)$")


# ---------------------------------------------------------------------------
# K74 — the claims-plane evaluation arm (V8-13.01)
# ---------------------------------------------------------------------------


class TestK74ClaimsArm:
    """The ``verbatim_claims`` arm pins ``admission.require_review=False``
    through the real config mapping; the product default stays ``True``;
    a real ingest + external-worker drain exercises the actual job chain
    and lands real claim/review rows."""

    ITEMS = [
        {"id": "d1",
         "text": "Caroline adopted a rescue dog named Biscuit last "
                 "March.",
         "speaker": "Caroline", "session_id": "s1",
         "when": "2023-05-01"},
        {"id": "d2",
         "text": "Melanie paints watercolors on weekends.",
         "speaker": "Melanie", "session_id": "s1",
         "when": "2023-05-01"},
    ]
    TASKS = [{"task_id": "q1", "query": "what dog did caroline adopt",
              "category": "single_hop", "gold_evidence": ["d1"]}]

    def test_k74_registry_and_product_default(self):
        from eval.v7.arms import VerbatimClaimsArm, arm_name, make_arm
        from verbatim.config import AdmissionConfig, VerbatimConfig

        arm = make_arm("verbatim_claims")
        try:
            assert isinstance(arm, VerbatimClaimsArm)
            assert arm_name(arm) == "verbatim_claims"
            assert arm._policy_overrides["admission"]["require_review"] \
                is False
            # V8-13.01 invariant 1 — the eval pin never moves product
            # defaults.
            assert AdmissionConfig().require_review is True
            assert VerbatimConfig().admission.require_review is True
            assert any("require_review" in n for n in arm.notes)
        finally:
            arm.close()

    def test_k74_real_ingest_drain_observable_claims(self, tmp_path):
        from eval.v7.arms import DictCorpus, VerbatimClaimsArm

        corpus = DictCorpus(
            self.ITEMS, self.TASKS, name="k74", dataset_id="k74-dict")
        arm = VerbatimClaimsArm(
            workdir=str(tmp_path / "k74"), settle_timeout_s=60.0)
        try:
            rep = arm.ingest(corpus)

            # The override is recorded in the ingest manifest block and
            # the built Memory really runs it.
            cfg = rep["config_overrides"]
            assert cfg["admission"]["require_review"] is False
            assert arm._mem._cfg.admission.require_review is False

            # The real external-worker drain ran the actual job chain.
            drain = rep["drain"]
            assert drain["processed"] > 0, drain
            assert drain["failed"] == 0, drain
            assert drain["still_pending"] == 0, drain

            with arm._mem._store.read() as conn:
                jobs = conn.execute(
                    "SELECT kind, COUNT(*) FROM jobs"
                    " WHERE state='succeeded' GROUP BY kind"
                ).fetchall()
                kinds = {k for k, _ in jobs}
                # harvest → source_project (+ embed/admit) — the real
                # write chain, not a stub.
                assert {"harvest", "source_project"} <= kinds, jobs
                n_claims = conn.execute(
                    "SELECT COUNT(*) FROM claims").fetchone()[0]
                assert n_claims > 0
                states = conn.execute(
                    "SELECT state, COUNT(*) FROM claim_revisions"
                    " GROUP BY state").fetchall()
                assert sum(n for _s, n in states) > 0
                # The observable difference the override makes: the
                # blanket ``review_required`` park is unreachable when
                # the flag is off — review rows may carry real ladder
                # reasons (e.g. ``abstained``), never this one.
                rr = conn.execute(
                    "SELECT COUNT(*) FROM reviews"
                    " WHERE proposed_effect_json LIKE '%review_required%'"
                ).fetchone()[0]
                assert rr == 0
        finally:
            arm.close()


# ---------------------------------------------------------------------------
# K75 — observation consolidation trace (V8-13.02)
# ---------------------------------------------------------------------------


def _bootstrap(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO scopes(scope_id, profile_id, visibility)"
        " VALUES (?,?,?)",
        (SCOPE, "prof", "conversation"),
    )
    ensure_v7_additive(conn)
    ensure_consolidation_v7(conn)


def _src(conn, source_id: str, payload: str, scope: str = SCOPE) -> None:
    conn.execute(
        "INSERT INTO sources(source_id, origin, source_kind, scope_id,"
        " created_us) VALUES (?,?,?,?,?)",
        (source_id, "test", "user_message", scope, 0),
    )
    pb = payload.encode("utf-8")
    conn.execute(
        "INSERT INTO source_revisions(source_id, revision, payload,"
        " payload_hmac, event_us, captured_us, provenance)"
        " VALUES (?,?,?,?,?,?,?)",
        (source_id, 1, pb, b"h", 0, 0, "direct_user"),
    )


def _unit(conn, unit_id: str, payload: str, *, occurred: int) -> None:
    _src(conn, f"src-{unit_id}", payload)
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " session_id, seq, recorded_at_us, occurred_start_us,"
        " occurred_end_us, byte_start, byte_end, generation)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (unit_id, f"src-{unit_id}", 1, SCOPE, "turn", "se1", 1,
         occurred, occurred, occurred, 0, len(payload.encode("utf-8")),
         GEN),
    )


def _state(conn, unit_id: str, state_key: str, value: str,
           *, valid_from: int) -> None:
    conn.execute(
        "INSERT INTO state_facts(scope_id, state_key, unit_id,"
        " generation, value_text, value_norm, valid_from_us, status,"
        " producer, pins_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (SCOPE, state_key, unit_id, GEN, value, value.lower(),
         valid_from, "current", "t0", "{}"),
    )


def _event(conn, event_id: str, unit_id: str, subject: str,
           predicate: str, obj: str, *, occurred: int) -> None:
    conn.execute(
        "INSERT INTO events_v7(event_id, unit_id, scope_id,"
        " subject_canon, predicate_lemma, object_text, polarity,"
        " occurred_start_us, occurred_end_us, precision, pins_json,"
        " rule_id, generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (event_id, unit_id, SCOPE, subject, predicate, obj, "affirm",
         occurred, occurred, "day", "{}", f"{predicate}/test", GEN),
    )


def _pref(conn, unit_id: str, subject: str, obj: str,
          *, occurred: int) -> None:
    conn.execute(
        "INSERT INTO preferences(scope_id, subject_canon, unit_id,"
        " generation, object_text, polarity, strength,"
        " occurred_start_us, pins_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (SCOPE, subject, unit_id, GEN, obj, "affirm", "", occurred, "{}"),
    )


def _obs_rows(conn):
    cur = conn.execute(
        "SELECT obs_id, slot, text, proof_count, support_refs_json,"
        " stale, generation FROM observations_v7"
        " WHERE scope_id=? AND generation=? ORDER BY obs_id",
        (SCOPE, GEN),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


class TestK75ConsolidationTrace:
    """``stats["trace"]`` (V8-13.02) on a real pass: candidate slots,
    per-gate rejections, writes — over real seeded fact rows."""

    @pytest.fixture()
    def conn(self, store):
        with store.tx() as c:
            _bootstrap(c)
        return store._conn

    def test_k75_trace_accepts_corroborated_observation(self, conn):
        # Two distinct units, two distinct families, one slot+value —
        # corroborated → written observation with pinned quotes.
        _unit(conn, "u1", "I live in Berlin.", occurred=100)
        _unit(conn, "u2", "I moved to Berlin.", occurred=200)
        _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
        _event(conn, "e1", "u2", "alice", "move", "Berlin", occurred=200)

        st = consolidate_scope_v7(
            conn, scope_id=SCOPE, generation=GEN, now_us=NOW)
        assert st["status"] == "ok"
        trace = st["trace"]
        assert trace["run_seq"] == 1
        assert trace["units_processed"] == 2
        assert trace["units_with_assertions"] == 2
        # The touched slot is reported with its assertion/candidate
        # counts.
        slots = {s["slot"]: s for s in trace["slots"]}
        assert "alice/home_city" in slots
        assert slots["alice/home_city"]["assertions"] == 2
        assert slots["alice/home_city"]["candidates"] == 1
        assert trace["candidates"] == 1
        assert trace["written"] == 1
        assert trace["rejected"] == []

        obs = _obs_rows(conn)
        assert len(obs) == 1
        o = obs[0]
        assert o["slot"] == "alice/home_city"
        assert o["proof_count"] == 2
        refs = json.loads(o["support_refs_json"])
        assert {r["unit_id"] for r in refs} == {"u1", "u2"}
        assert {r["quote"] for r in refs} == {
            "I live in Berlin.", "I moved to Berlin."}
        assert {f for r in refs for f in r["families"]} == {
            "state", "event"}

    def test_k75_trace_gate_named_rejections(self, conn):
        # u3: single unit asserting favorite_color — fails
        # min_support_units; u4+u5: same family (state) asserting sport —
        # fails min_support_families.
        _unit(conn, "u3", "my favorite color is blue.", occurred=100)
        _state(conn, "u3", "alice/favorite_color", "blue",
               valid_from=100)
        _unit(conn, "u4", "i play tennis.", occurred=100)
        _unit(conn, "u5", "i play tennis on tuesdays.", occurred=200)
        _state(conn, "u4", "alice/sport", "tennis", valid_from=100)
        _state(conn, "u5", "alice/sport", "tennis", valid_from=200)

        st = consolidate_scope_v7(
            conn, scope_id=SCOPE, generation=GEN, now_us=NOW)
        trace = st["trace"]
        assert st["observations_written"] == 0
        assert trace["written"] == 0
        gates = {}
        for rej in trace["rejected"]:
            gates.setdefault(rej["gate"], []).append(rej)
            assert rej["slot"].startswith("alice/")
            assert rej["reason"]
            assert rej["support_units"] >= 0
            assert rej["families"]
        # Each failed gate lands its own enumerable entry.
        assert "min_support_units" in gates
        assert "min_support_families" in gates
        assert any(r["slot"] == "alice/favorite_color"
                   for r in gates["min_support_units"])
        assert any(r["slot"] == "alice/sport"
                   for r in gates["min_support_families"])
        assert _obs_rows(conn) == []

    def test_k75_second_run_continues_from_cursor(self, conn):
        _unit(conn, "u1", "I live in Berlin.", occurred=100)
        _state(conn, "u1", "alice/home_city", "Berlin", valid_from=100)
        st1 = consolidate_scope_v7(
            conn, scope_id=SCOPE, generation=GEN, now_us=NOW)
        # Second pass over no new units: cursor-incremental, honest zero.
        st2 = consolidate_scope_v7(
            conn, scope_id=SCOPE, generation=GEN, now_us=NOW + 1)
        assert st2["trace"]["run_seq"] == 2
        assert st2["processed"] == 0
        assert st2["trace"]["units_processed"] == 0
        assert st1["trace"]["candidates"] == 1


# ---------------------------------------------------------------------------
# K76 — canon hygiene (V8-13.03, entities/v2.1)
# ---------------------------------------------------------------------------


def _norm_with_surface(text: str):
    """The norm object the write path builds: real ``analyze()`` terms
    with ``text`` carrying the raw surface (``units_jobs`` substitutes
    it via ``dataclasses.replace`` — mirrored, not stubbed)."""
    from dataclasses import replace

    return replace(analyze(text), text=text)


def _canons(text: str) -> set[str]:
    return {m.canon for m in extract_mentions(_norm_with_surface(text), "u-x")}


class TestK76CanonHygiene:
    def test_k76_extractor_id_bumped(self):
        # V8-13.03 lands under entities/v2.1 — the rule-snapshot pin on
        # every derived artifact.
        assert EXTRACTOR_ID == "entities/v2.1"

    @pytest.mark.parametrize("text,fragments", [
        ("Can't reach Tom today.", {"can t", "cant", "can"}),
        ("You're late for dinner.", {"you re", "youre", "you"}),
        ("Don't forget the keys.", {"don t", "dont", "don"}),
        ("We've seen worse.", {"we ve", "weve"}),
        ("I'll call Sam.", {"i ll", "ill"}),
        ("It'd be fine.", {"it d", "itd"}),
        ("I'm leaving now.", {"i m", "im"}),
    ])
    def test_k76_contraction_never_mints_fragment(self, text, fragments):
        # The whole contraction token is excluded: no folded fragment
        # canon AND no pronoun-stem head canon.
        assert _canons(text).isdisjoint(fragments)

    def test_k76_contraction_ends_run(self):
        # "Nate Can't Go" — the contraction ends the capitalized run:
        # the names on either side survive as their own canons, never
        # fused through the contraction.
        cs = _canons("Nate Can't Go")
        assert cs == {"nate", "go"}

    @pytest.mark.parametrize("text,expect,absent", [
        ("Thanks Nate, see you soon.", {"nate"}, {"thanks", "thanks nate"}),
        ("Hey Gina, how are you?", {"gina"}, {"hey", "hey gina"}),
        ("Hello Marco Polo.", {"marco polo"}, {"hello", "hello marco polo"}),
        ("Wow, Caroline did it.", {"caroline"}, {"wow", "wow caroline"}),
        ("Please tell Jonas.", {"jonas"}, {"please", "please jonas"}),
    ])
    def test_k76_vocative_never_joins_run(self, text, expect, absent):
        cs = _canons(text)
        assert expect <= cs
        assert cs.isdisjoint(absent)

    def test_k76_possessive_still_canonizes(self):
        # 's is the one tail kept at run level — strip_possessive folds
        # "Caroline's" to canon "caroline" with the surface byte-exact.
        ms = extract_mentions(_norm_with_surface("Caroline's dog won."),
                              "u-x")
        by_canon = {m.canon: m for m in ms}
        assert "caroline" in by_canon
        assert by_canon["caroline"].surface == "Caroline's"

    def test_k76_sentence_boundary_blocks_fusion(self):
        # "Jon: Gina went…" — the colon ends the speaker prefix; a run
        # never crosses . ! ? newline or colon.
        cs = _canons("Jon: Gina went home.")
        assert "jon gina" not in cs
        assert "gina" in cs

    def test_k76_row_level_real_projection(self, tmp_path):
        """Same hygiene, asserted on persisted entity_mentions /
        entity_canon rows written by the real ``project_units_v7``."""
        from verbatim import Memory
        from verbatim.jobs.units_jobs import project_units_v7

        m = Memory(path=str(tmp_path / "m.db"), worker="external")
        try:
            r = m.add(
                "Thanks Nate, I live in Berlin. Can't wait. "
                "You're invited. Don't worry.",
                speaker="Alice",
            )
            sid = r.memory_id
            with m._store.tx() as conn:
                srow = conn.execute(
                    "SELECT source_id, origin, external_id, source_kind,"
                    " scope_id, speaker_id, created_us FROM sources"
                    " WHERE source_id=?",
                    (sid,),
                ).fetchone()
                rrow = conn.execute(
                    "SELECT source_id, revision, payload, event_us,"
                    " captured_us, timezone, provenance, metadata_json"
                    " FROM source_revisions"
                    " WHERE source_id=? AND revision=1",
                    (sid,),
                ).fetchone()
                source_row = dict(zip(
                    ("source_id", "origin", "external_id", "source_kind",
                     "scope_id", "speaker_id", "created_us"), srow))
                revision_row = dict(zip(
                    ("source_id", "revision", "payload", "event_us",
                     "captured_us", "timezone", "provenance",
                     "metadata_json"), rrow))
                stats = project_units_v7(
                    conn, source_row=source_row, revision_row=revision_row,
                    add_args={}, generation=GEN, scope_id=m._namespace)
                assert stats["units"] >= 1, stats
                assert stats["mentions"] >= 2, stats

                canons = {
                    r2[0] for r2 in conn.execute(
                        "SELECT canon FROM entity_mentions"
                        " WHERE scope_id=? AND generation=?",
                        (m._namespace, GEN))
                }
                canon_rows = {
                    r2[0] for r2 in conn.execute(
                        "SELECT canon FROM entity_canon"
                        " WHERE scope_id=? AND generation=?",
                        (m._namespace, GEN))
                }
            all_canons = canons | canon_rows
            # Real names survived…
            assert "nate" in all_canons
            assert "berlin" in all_canons
            assert "alice" in all_canons  # role=speaker mention
            # …and no hygiene-failure canon exists at row level.
            bad = {c for c in all_canons
                   if c in {"can t", "you re", "don t", "we ve",
                            "thanks", "thanks nate", "you", "can", "don"}
                   or c.endswith((" t", " re", " ve", " ll"))}
            assert bad == set(), bad
        finally:
            m.close()


# ---------------------------------------------------------------------------
# K77 — subject-key contract on real extraction + persisted rows (V8-13.04)
# ---------------------------------------------------------------------------


class TestK77SubjectKeys:
    def test_k77_known_speaker_keys(self):
        (f,) = extract_state_facts(
            analyze("i live in Berlin"), "u-1", "speaker:alice",
            IntervalUs(start_us=1, end_us=2), raw_text="i live in Berlin")
        assert f.state_key == "speaker:alice/home_city"
        (p,) = extract_preferences(
            analyze("i love pizza"), "u-1", "speaker:alice",
            raw_text="i love pizza")
        assert p.subject_canon == "speaker:alice"

    def test_k77_unattributed_binds_me(self):
        (f,) = extract_state_facts(
            analyze("i live in Berlin"), "u-1", "",
            IntervalUs(start_us=1, end_us=2), raw_text="i live in Berlin")
        assert f.state_key == "me/home_city"
        (p,) = extract_preferences(
            analyze("i love pizza"), "u-1", "", raw_text="i love pizza")
        assert p.subject_canon == "me"

    def test_k77_row_level_via_projection(self, tmp_path):
        """Persisted ``state_facts``/``preferences`` rows carry the same
        contract: declared speaker → ``<canon>/<slot>``; a source with no
        speaker at all → ``me/<slot>``."""
        from verbatim import Memory
        from verbatim.jobs.units_jobs import project_units_v7

        m = Memory(path=str(tmp_path / "m.db"), worker="external")
        try:
            # (a) declared speaker — ``mem.add(speaker=…)`` persists it
            # through metadata_json → unit speaker_canon → subject key.
            r = m.add("i live in Berlin. i love pizza.", speaker="Alice")
            sid = r.memory_id
            # (b) a speaker-less source row — first-person binds "me".
            with m._store.tx() as conn:
                conn.execute(
                    "INSERT INTO sources(source_id, origin, source_kind,"
                    " scope_id, created_us) VALUES (?,?,?,?,?)",
                    ("src-nospeaker", "test", "user_message",
                     m._namespace, now_us()))
                payload = "i live in Lisbon. i love sushi.".encode()
                conn.execute(
                    "INSERT INTO source_revisions(source_id, revision,"
                    " payload, payload_hmac, event_us, captured_us,"
                    " provenance) VALUES (?,?,?,?,?,?,?)",
                    ("src-nospeaker", 1, payload,
                     m._store.hmac(payload), now_us(), now_us(),
                     "direct_user"))

            def _project(source_id: str):
                with m._store.tx() as conn:
                    srow = conn.execute(
                        "SELECT source_id, origin, external_id,"
                        " source_kind, scope_id, speaker_id, created_us"
                        " FROM sources WHERE source_id=?",
                        (source_id,),
                    ).fetchone()
                    rrow = conn.execute(
                        "SELECT source_id, revision, payload, event_us,"
                        " captured_us, timezone, provenance,"
                        " metadata_json FROM source_revisions"
                        " WHERE source_id=? AND revision=1",
                        (source_id,),
                    ).fetchone()
                    source_row = dict(zip(
                        ("source_id", "origin", "external_id",
                         "source_kind", "scope_id", "speaker_id",
                         "created_us"), srow))
                    revision_row = dict(zip(
                        ("source_id", "revision", "payload", "event_us",
                         "captured_us", "timezone", "provenance",
                         "metadata_json"), rrow))
                    return project_units_v7(
                        conn, source_row=source_row,
                        revision_row=revision_row, add_args={},
                        generation=GEN, scope_id=m._namespace)

            _project(sid)
            _project("src-nospeaker")

            with m._store.read() as conn:
                keys = {
                    r2[0] for r2 in conn.execute(
                        "SELECT state_key FROM state_facts"
                        " WHERE scope_id=? AND generation=?",
                        (m._namespace, GEN))
                }
                subjs = {
                    r2[0] for r2 in conn.execute(
                        "SELECT subject_canon FROM preferences"
                        " WHERE scope_id=? AND generation=?",
                        (m._namespace, GEN))
                }
            # Known speaker → its canon keys; never "me".
            assert "alice/home_city" in keys
            # Unattributed source → me/ keys.
            assert "me/home_city" in keys
            assert {"alice", "me"} <= subjs
            with m._store.read() as conn:
                lisbon = conn.execute(
                    "SELECT state_key, value_text FROM state_facts"
                    " WHERE value_text='Lisbon'").fetchall()
            assert lisbon and all(k == "me/home_city" for k, _v in lisbon)
        finally:
            m.close()


# ---------------------------------------------------------------------------
# K78 — write-path replay is byte-identical (V8-13.05)
# ---------------------------------------------------------------------------


def _fts5_vtables(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND sql LIKE '%fts5%'")
    }


def _snapshot_content(conn: sqlite3.Connection) -> dict:
    """``{table: (cols, rows)}`` over every real content table.

    Excluded by rule, and asserted separately:

    * FTS5 shadow tables (``<vtable>_{data,idx,seg,segdir,docsize,stat,
      config}``) — SQLite-internal index pages whose byte layout legit-
      imately differs after delete+reinsert; the logical ``unit_fts*``
      content tables ARE compared;
    * ``consolidation_cursor_v7`` — run bookkeeping (``run_seq``,
      timestamps) compared on content columns only;
    * ``consolidation_seen_v7`` — the ``seen_run`` column is which run
      observed the row; coverage (scope, unit, generation) is compared.
    """
    shadows: set[str] = set()
    for v in _fts5_vtables(conn):
        for sfx in ("data", "idx", "seg", "segdir", "docsize", "stat",
                    "config"):
            shadows.add(f"{v}_{sfx}")

    out: dict = {}
    volatile_cols = {
        "consolidation_cursor_v7": {
            "run_seq", "last_run_us", "prev_run_us", "last_processed",
            "total_processed"},
        "consolidation_seen_v7": {"seen_run"},
    }
    tabs = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " ORDER BY name")
    ]
    for t in tabs:
        if t.startswith("sqlite_") or t in shadows:
            continue
        cur = conn.execute(f"SELECT * FROM [{t}]")
        cols = [d[0] for d in cur.description]
        drop = volatile_cols.get(t, set()) | {"rowid"}
        keep = [i for i, c in enumerate(cols) if c not in drop]
        rows = sorted(
            repr(tuple(
                (v.tobytes() if isinstance(v, memoryview) else v)
                for i, v in enumerate(r) if i in keep))
            for r in cur.fetchall()
        )
        out[t] = (tuple(c for c in cols if c not in drop), tuple(rows))
    return out


class TestK78ReplayIdempotent:
    def _project(self, m, sid: str) -> dict:
        from verbatim.jobs.units_jobs import project_units_v7

        with m._store.tx() as conn:
            srow = conn.execute(
                "SELECT source_id, origin, external_id, source_kind,"
                " scope_id, speaker_id, created_us FROM sources"
                " WHERE source_id=?",
                (sid,),
            ).fetchone()
            rrow = conn.execute(
                "SELECT source_id, revision, payload, event_us,"
                " captured_us, timezone, provenance, metadata_json"
                " FROM source_revisions"
                " WHERE source_id=? AND revision=1",
                (sid,),
            ).fetchone()
            source_row = dict(zip(
                ("source_id", "origin", "external_id", "source_kind",
                 "scope_id", "speaker_id", "created_us"), srow))
            revision_row = dict(zip(
                ("source_id", "revision", "payload", "event_us",
                 "captured_us", "timezone", "provenance",
                 "metadata_json"), rrow))
            return project_units_v7(
                conn, source_row=source_row, revision_row=revision_row,
                add_args={}, generation=GEN, scope_id=m._namespace)

    def test_k78_reprojection_byte_identical(self, tmp_path):
        from verbatim import Memory

        m = Memory(path=str(tmp_path / "m.db"), worker="external")
        try:
            r = m.add(
                "i live in Berlin. i love pizza. my dog is Rex.")
            sid = r.memory_id

            s1 = self._project(m, sid)
            assert s1["units"] >= 1, s1
            with m._store.read() as conn:
                snap1 = _snapshot_content(conn)
            assert "units" in snap1 and snap1["units"][1]
            # Consolidation ran inside the projection — profiles exist.
            assert snap1.get("profiles_v7", (None, ()))[1]

            s2 = self._project(m, sid)
            with m._store.read() as conn:
                snap2 = _snapshot_content(conn)

            # Same projection stats on replay (deterministic output).
            for k in ("units", "turns", "sentence_windows", "fts_rows",
                      "mentions", "canons"):
                assert s2.get(k) == s1.get(k), (k, s1.get(k), s2.get(k))

            diffs = [
                t for t in sorted(set(snap1) | set(snap2))
                if snap1.get(t) != snap2.get(t)
            ]
            assert diffs == [], diffs

            # The K78 regression pinned: unit-pinning consolidation
            # artifacts are re-derived on replay, not stranded by the
            # incremental cursor.
            assert snap2.get("profiles_v7", (None, ()))[1] == \
                snap1["profiles_v7"][1]

            # The consolidation cursor legitimately advanced (run
            # bookkeeping, not content).
            with m._store.read() as conn:
                cur = conn.execute(
                    "SELECT run_seq, last_unit_rowid"
                    " FROM consolidation_cursor_v7"
                    " WHERE scope_id=? AND generation=?",
                    (m._namespace, GEN),
                ).fetchone()
            assert cur is not None and cur[0] >= 2
        finally:
            m.close()
