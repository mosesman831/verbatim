"""Dedup-by-linking tests (SPEC_V5 §30.2; E74/E75).

Real on-disk ``Store`` + the §3 contract tables (see conftest). Covers
exact-digest linking, MinHash-class near linking, the V5-30.07 hard
guards, group reads (members/corroboration/representative), pack-time
collapse, the per-namespace dedupe policy, and deletion closure.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import VerbatimError
from verbatim.dedup import links
from verbatim.dedup.links import (
    DedupFields,
    collapse_for_hit,
    corroboration_count,
    drop_member,
    get_dedupe_policy,
    group_members,
    link_exact,
    link_near,
    member_refs,
    representative,
    set_dedupe_policy,
)

from .conftest import (
    add_revision,
    erase_source,
    norm_digest,
    payload_hmac_hex,
    seed_source,
    signature_of,
)

NS = "ns-main"
T0 = 1_700_000_000_000_000


def _links_rows(conn, source_id):
    cur = conn.execute(
        "SELECT source_id, revision, group_id, method, score"
        " FROM duplicate_links WHERE source_id = ? ORDER BY method",
        (source_id,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _all_group_ids(conn):
    return {
        r[0] for r in conn.execute(
            "SELECT DISTINCT group_id FROM duplicate_links"
        ).fetchall()
    }


# ---------------------------------------------------------------------------
# exact_digest linking (V5-30.05, E74)
# ---------------------------------------------------------------------------


class TestExactLink:
    def test_second_identical_copy_links(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1",
                        text="the linter is ruff",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2",
                        text="the linter is ruff",
                        namespace=NS, created_us=T0 + 1)
            out = link_exact(
                conn, source_id="s2", revision=1, namespace=NS,
                digest=payload_hmac_hex(store, "the linter is ruff"),
            )
        assert out.linked is True
        assert out.method == "exact_digest"
        assert out.score == 1.0
        assert out.group_id == "s1"          # earliest live member anchors
        assert out.linked_to == ("s1", 1)
        with store.read() as conn:
            members = group_members(conn, "s1")
        assert {(m["source_id"], m["revision"]) for m in members} == {
            ("s1", 1), ("s2", 1)}
        assert all(m["live"] for m in members)

    def test_both_records_retained_with_own_attribution(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="same bytes",
                        namespace=NS, speaker="alice",
                        provenance="direct_user", created_us=T0)
            seed_source(conn, store, source_id="s2", text="same bytes",
                        namespace=NS, speaker="bob",
                        provenance="operator", created_us=T0 + 1)
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=payload_hmac_hex(store, "same bytes"))
        with store.read() as conn:
            # Both payloads and attribution rows intact — linking never
            # merges or deletes (V5-30.03/05).
            rows = conn.execute(
                "SELECT source_id, speaker_id FROM sources"
                " WHERE source_id IN ('s1','s2') ORDER BY source_id"
            ).fetchall()
            assert rows == [("s1", "alice"), ("s2", "bob")]
            payloads = conn.execute(
                "SELECT source_id, payload FROM source_revisions"
                " ORDER BY source_id"
            ).fetchall()
            assert payloads == [("s1", b"same bytes"), ("s2", b"same bytes")]

    def test_third_copy_joins_existing_group(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="x",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="x",
                        namespace=NS, created_us=T0 + 1)
            seed_source(conn, store, source_id="s3", text="x",
                        namespace=NS, created_us=T0 + 2)
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=payload_hmac_hex(store, "x"))
            out3 = link_exact(conn, source_id="s3", revision=1,
                              namespace=NS,
                              digest=payload_hmac_hex(store, "x"))
        assert out3.group_id == "s1"
        with store.read() as conn:
            assert len(group_members(conn, "s1")) == 3
            assert _all_group_ids(conn) == {"s1"}

    def test_idempotent_relink(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="x",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="x",
                        namespace=NS, created_us=T0 + 1)
            d = payload_hmac_hex(store, "x")
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=d)
            out = link_exact(conn, source_id="s2", revision=1,
                             namespace=NS, digest=d)
        assert out.linked is True
        with store.read() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) FROM duplicate_links"
            ).fetchone()[0]
        assert rows == 2  # one row per member per method — no dup rows

    def test_different_content_no_link(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="alpha",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="beta",
                        namespace=NS, created_us=T0 + 1)
            out = link_exact(conn, source_id="s2", revision=1,
                             namespace=NS,
                             digest=payload_hmac_hex(store, "beta"))
        assert out.linked is False
        assert out.reason == "no_match"
        with store.read() as conn:
            assert _links_rows(conn, "s2") == []

    def test_namespace_isolation(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="x",
                        namespace="ns-other", created_us=T0)
            seed_source(conn, store, source_id="s2", text="x",
                        namespace=NS, created_us=T0 + 1)
            out = link_exact(conn, source_id="s2", revision=1,
                             namespace=NS,
                             digest=payload_hmac_hex(store, "x"))
        assert out.linked is False  # same bytes, foreign namespace

    def test_normalized_method_links_casefolded_text(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1",
                        text="Deploy V1 finished.",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2",
                        text="deploy v1 finished",
                        namespace=NS, created_us=T0 + 1)
            out = link_exact(
                conn, source_id="s2", revision=1, namespace=NS,
                digest=norm_digest("deploy v1 finished"),
                method="normalized",
            )
        assert out.linked is True
        assert out.method == "normalized"
        assert out.group_id == "s1"

    def test_exact_digest_requires_byte_identity(self, store):
        """Folded-equal but byte-different payloads are NOT exact_digest
        duplicates — only the normalized path may link them (V5-30.05)."""
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="Deploy V1.",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="deploy v1",
                        namespace=NS, created_us=T0 + 1)
            out = link_exact(
                conn, source_id="s2", revision=1, namespace=NS,
                digest=payload_hmac_hex(store, "deploy v1"),
            )
        assert out.linked is False


# ---------------------------------------------------------------------------
# near-duplicate linking (V5-30.06)
# ---------------------------------------------------------------------------


class TestNearLink:
    def test_similar_texts_link_minhash(self, store):
        a = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1")
        b = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1 overall")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1)
            out = link_near(conn, source_id="s2", revision=1,
                            namespace=NS, signature=signature_of(b))
        assert out.linked is True
        assert out.method == "minhash"
        assert 0.85 <= out.score < 1.0
        assert out.group_id == "s1"
        with store.read() as conn:
            rows = _links_rows(conn, "s2")
        assert rows[0]["method"] == "minhash"
        assert rows[0]["score"] == pytest.approx(out.score)

    def test_dissimilar_no_link(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1",
                        text="the database backup runs every night",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2",
                        text="a pizza recipe needs fresh basil",
                        namespace=NS, created_us=T0 + 1)
            out = link_near(conn, source_id="s2", revision=1,
                            namespace=NS,
                            signature=signature_of(
                                "a pizza recipe needs fresh basil"))
        assert out.linked is False
        assert out.reason == "no_match"

    def test_threshold_is_configurable(self, store):
        a = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1")
        b = a + " overall"   # high overlap but < 1.0
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1)
            strict = link_near(conn, source_id="s2", revision=1,
                               namespace=NS, signature=signature_of(b),
                               threshold=0.999)
            assert strict.linked is False
            loose = link_near(conn, source_id="s2", revision=1,
                              namespace=NS, signature=signature_of(b),
                              threshold=0.50)
        assert loose.linked is True
        assert 0.50 <= loose.score < 0.999

    def test_candidate_scan_is_bounded_recent(self, store):
        near = ("the release pipeline for our staging environment in this "
                "repository stays pinned to the artifact version 1")
        with store.tx() as conn:
            # Oldest member is the true near-duplicate.
            seed_source(conn, store, source_id="old", text=near,
                        namespace=NS, created_us=T0)
            # 5 newer unrelated members push it past a scan_limit of 5.
            for i in range(5):
                seed_source(conn, store, source_id=f"f{i}",
                            text=f"unrelated filler content number {i}"
                                 " with distinct vocabulary entirely",
                            namespace=NS, created_us=T0 + 1 + i)
            seed_source(conn, store, source_id="new", text=near,
                        namespace=NS, created_us=T0 + 9)
            bounded = link_near(conn, source_id="new", revision=1,
                                namespace=NS, signature=signature_of(near),
                                scan_limit=5)
            assert bounded.linked is False
            assert bounded.scanned == 5
            wide = link_near(conn, source_id="new", revision=1,
                             namespace=NS, signature=signature_of(near),
                             scan_limit=50)
        assert wide.linked is True
        assert wide.linked_to == ("old", 1)

    def test_invalid_threshold_rejected(self, store):
        with store.tx() as conn:
            with pytest.raises(VerbatimError):
                link_near(conn, source_id="s1", revision=1, namespace=NS,
                          signature={"a"}, threshold=0.0)


# ---------------------------------------------------------------------------
# V5-30.07 hard guards — twins that must never link (E75)
# ---------------------------------------------------------------------------


class TestHardGuards:
    def test_version_twin_never_links(self, store):
        """deploy-v1 vs deploy-v2 at >=0.85 similarity — the spec's own
        example — vetoed on the identifier dimension."""
        a = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1")
        b = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 2")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1)
            out = link_near(conn, source_id="s2", revision=1,
                            namespace=NS, signature=signature_of(b))
        assert out.linked is False
        assert out.reason == "guard_veto"
        assert out.vetoes[0]["dimension"] in {"identifiers", "numbers"}
        assert out.vetoes[0]["score"] >= 0.85  # similarity WAS high

    def test_polarity_twin_never_links(self, store):
        """"I like X" vs "I no longer like X" — polarity veto even when
        the shingle similarity clears a lowered bar."""
        a = ("the team reviewed the rollout plan for the billing "
             "service and i like that approach")
        b = ("the team reviewed the rollout plan for the billing "
             "service and i no longer like that approach")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1)
            out = link_near(conn, source_id="s2", revision=1,
                            namespace=NS, signature=signature_of(b),
                            threshold=0.3)
        assert out.linked is False
        assert out.vetoes[0]["dimension"] == "polarity"

    def test_identifier_twin_never_links(self, store):
        a = ("the migration checklist for the billing service database "
             "cluster ends with the checksum"
             " a94a8fe5ccb19ba61c4c0873d391e987982fbbd3")
        b = ("the migration checklist for the billing service database "
             "cluster ends with the checksum"
             " b94a8fe5ccb19ba61c4c0873d391e987982fbbd4")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1)
            out = link_near(conn, source_id="s2", revision=1,
                            namespace=NS, signature=signature_of(b),
                            threshold=0.5)
        assert out.linked is False
        assert out.vetoes[0]["dimension"] == "identifiers"

    def test_number_twin_never_links(self, store):
        a = ("capacity planning set the staging database connection pool "
             "for the billing service at 42")
        b = ("capacity planning set the staging database connection pool "
             "for the billing service at 43")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1)
            out = link_near(conn, source_id="s2", revision=1,
                            namespace=NS, signature=signature_of(b),
                            threshold=0.5)
        assert out.linked is False
        assert out.vetoes[0]["dimension"] == "numbers"

    def test_time_twin_never_links(self, store):
        a = ("the weekly release train for the billing service database "
             "cluster leaves the depot on monday")
        b = ("the weekly release train for the billing service database "
             "cluster leaves the depot on tuesday")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1)
            out = link_near(conn, source_id="s2", revision=1,
                            namespace=NS, signature=signature_of(b),
                            threshold=0.5)
        assert out.linked is False
        assert out.vetoes[0]["dimension"] == "time"

    def test_type_twin_never_links(self, store):
        a = ("after reviewing every rollout option for the billing "
             "service pipeline i prefer static deployments")
        b = ("after reviewing every rollout option for the billing "
             "service pipeline we decided static deployments")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1)
            out = link_near(conn, source_id="s2", revision=1,
                            namespace=NS, signature=signature_of(b),
                            threshold=0.5)
        assert out.linked is False
        assert out.vetoes[0]["dimension"] == "type"

    def test_guard_veto_blocks_normalized_link(self, store):
        """Normalized-equal digests still pass through guards — a quoted
        record and an unquoted one are not duplicates (V5-30.07)."""
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1",
                        text='"the linter is ruff"',  # quoted → quoted
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2",
                        text="the linter is ruff",
                        namespace=NS, created_us=T0 + 1)
            out = link_exact(
                conn, source_id="s2", revision=1, namespace=NS,
                digest=norm_digest("the linter is ruff"),
                method="normalized",
            )
        assert out.linked is False
        assert out.vetoes[0]["dimension"] == "polarity"


# ---------------------------------------------------------------------------
# per-namespace dedupe policy (V5-30.08)
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_default_is_link(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="x",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="x",
                        namespace=NS, created_us=T0 + 1)
            assert get_dedupe_policy(conn, NS) == "link"
            out = link_exact(conn, source_id="s2", revision=1,
                             namespace=NS,
                             digest=payload_hmac_hex(store, "x"))
        assert out.linked is True

    def test_policy_none_disables_linking(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="x",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="x",
                        namespace=NS, created_us=T0 + 1)
            set_dedupe_policy(conn, NS, "none")
            exact = link_exact(conn, source_id="s2", revision=1,
                               namespace=NS,
                               digest=payload_hmac_hex(store, "x"))
            near = link_near(conn, source_id="s2", revision=1,
                             namespace=NS, signature=signature_of("x"))
        assert exact.linked is False and exact.reason == "policy_none"
        assert near.linked is False and near.reason == "policy_none"
        with store.read() as conn:
            assert _links_rows(conn, "s2") == []

    def test_policy_is_per_namespace(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="a1", text="x",
                        namespace="ns-a", created_us=T0)
            seed_source(conn, store, source_id="a2", text="x",
                        namespace="ns-a", created_us=T0 + 1)
            seed_source(conn, store, source_id="b1", text="x",
                        namespace="ns-b", created_us=T0)
            seed_source(conn, store, source_id="b2", text="x",
                        namespace="ns-b", created_us=T0 + 1)
            set_dedupe_policy(conn, "ns-a", "none")
            off = link_exact(conn, source_id="a2", revision=1,
                             namespace="ns-a",
                             digest=payload_hmac_hex(store, "x"))
            on = link_exact(conn, source_id="b2", revision=1,
                            namespace="ns-b",
                            digest=payload_hmac_hex(store, "x"))
        assert off.linked is False
        assert on.linked is True

    def test_invalid_policy_rejected(self, store):
        with store.tx() as conn:
            with pytest.raises(VerbatimError):
                set_dedupe_policy(conn, NS, "merge")
            with pytest.raises(VerbatimError):
                set_dedupe_policy(conn, NS, "delete")


# ---------------------------------------------------------------------------
# group reads: members, corroboration, representative
# ---------------------------------------------------------------------------


class TestGroupReads:
    def _three_member_group(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="shared",
                        namespace=NS, speaker="alice", created_us=T0)
            seed_source(conn, store, source_id="s2", text="shared",
                        namespace=NS, speaker="bob", created_us=T0 + 1)
            seed_source(conn, store, source_id="s3", text="shared",
                        namespace=NS, speaker="carol", created_us=T0 + 2)
            d = payload_hmac_hex(store, "shared")
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=d)
            link_exact(conn, source_id="s3", revision=1, namespace=NS,
                       digest=d)

    def test_corroboration_counts_independent_submitters(self, store):
        self._three_member_group(store)
        with store.read() as conn:
            assert corroboration_count(conn, "s1") == 3

    def test_copies_never_count_twice(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="shared",
                        namespace=NS, speaker="alice", created_us=T0)
            # Same submitter re-submitting identical bytes = a copy.
            seed_source(conn, store, source_id="s2", text="shared",
                        namespace=NS, speaker="alice", created_us=T0 + 1)
            seed_source(conn, store, source_id="s3", text="shared",
                        namespace=NS, speaker="alice", created_us=T0 + 2)
            d = payload_hmac_hex(store, "shared")
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=d)
            link_exact(conn, source_id="s3", revision=1, namespace=NS,
                       digest=d)
        with store.read() as conn:
            assert corroboration_count(conn, "s1") == 1

    def test_representative_is_earliest_live(self, store):
        self._three_member_group(store)
        with store.read() as conn:
            rep = representative(conn, "s1")
        assert rep["source_id"] == "s1"

    def test_representative_prefers_non_superseded(self, store):
        self._three_member_group(store)
        with store.tx() as conn:
            conn.execute(
                "UPDATE source_state SET disposition = 'superseded'"
                " WHERE source_id = 's1'"
            )
        with store.read() as conn:
            rep = representative(conn, "s1")
        # s1 stays live (retained) but the current record is preferred.
        assert rep["source_id"] == "s2"
        with store.read() as conn:
            members = group_members(conn, "s1")
        assert all(m["live"] for m in members)

    def test_member_refs_reachable_for_inspect(self, store):
        self._three_member_group(store)
        with store.read() as conn:
            refs = member_refs(conn, "s1")
        assert refs == [("s1", 1), ("s2", 1), ("s3", 1)]


# ---------------------------------------------------------------------------
# pack-time collapse (V5-30.09)
# ---------------------------------------------------------------------------


class TestCollapse:
    def test_collapse_reports_representative_and_counts(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="shared",
                        namespace=NS, speaker="alice", created_us=T0)
            seed_source(conn, store, source_id="s2", text="shared",
                        namespace=NS, speaker="bob", created_us=T0 + 1)
            seed_source(conn, store, source_id="s3", text="shared",
                        namespace=NS, speaker="carol", created_us=T0 + 2)
            d = payload_hmac_hex(store, "shared")
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=d)
            link_exact(conn, source_id="s3", revision=1, namespace=NS,
                       digest=d)
        with store.read() as conn:
            view = collapse_for_hit(conn, "s2", 1)
        assert view["group_id"] == "s1"
        assert view["representative"]["source_id"] == "s1"
        assert view["collapsed_duplicates"] == 2
        assert view["corroboration"] == 3
        # Member refs stay reachable for inspect (V5-30.09).
        assert view["member_refs"] == [("s1", 1), ("s2", 1), ("s3", 1)]

    def test_ungrouped_hit_has_no_collapse(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="solo",
                        namespace=NS, created_us=T0)
        with store.read() as conn:
            assert collapse_for_hit(conn, "s1", 1) is None


# ---------------------------------------------------------------------------
# deletion closure (V5-30.08): forgetting a member never forgets siblings
# ---------------------------------------------------------------------------


class TestDeletionClosure:
    def test_forget_one_member_leaves_siblings(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="shared",
                        namespace=NS, speaker="alice", created_us=T0)
            seed_source(conn, store, source_id="s2", text="shared",
                        namespace=NS, speaker="bob", created_us=T0 + 1)
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=payload_hmac_hex(store, "shared"))
            erase_source(conn, store, "s1")
            drop_member(conn, "s1", 1)
        with store.read() as conn:
            # Sibling bytes/attribution untouched by the member's forget.
            row = conn.execute(
                "SELECT payload FROM source_revisions"
                " WHERE source_id = 's2'"
            ).fetchone()
            assert row[0] == b"shared"
            spk = conn.execute(
                "SELECT speaker_id FROM sources WHERE source_id = 's2'"
            ).fetchone()
            assert spk[0] == "bob"
            # Group dissolved to a single member — no rows remain.
            assert _links_rows(conn, "s2") == []
            assert corroboration_count(conn, "s1") == 0
            assert representative(conn, "s1") is None

    def test_forgotten_member_not_live_without_drop(self, store):
        """Even if closure never calls drop_member, stale link rows must
        not resurrect a forgotten member (erasure-signal liveness)."""
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="shared",
                        namespace=NS, speaker="alice", created_us=T0)
            seed_source(conn, store, source_id="s2", text="shared",
                        namespace=NS, speaker="bob", created_us=T0 + 1)
            seed_source(conn, store, source_id="s3", text="shared",
                        namespace=NS, speaker="carol", created_us=T0 + 2)
            d = payload_hmac_hex(store, "shared")
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=d)
            link_exact(conn, source_id="s3", revision=1, namespace=NS,
                       digest=d)
            erase_source(conn, store, "s1")   # no drop_member call
        with store.read() as conn:
            members = group_members(conn, "s1")
            dead = [m for m in members if m["source_id"] == "s1"]
            assert dead and dead[0]["live"] is False
            rep = representative(conn, "s1")
            assert rep["source_id"] == "s2"      # earliest LIVE member
            assert corroboration_count(conn, "s1") == 2  # bob + carol

    def test_anchor_forget_reanchors_group(self, store):
        """group_id re-anchors to the new earliest live member when the
        anchor is forgotten via drop_member."""
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="shared",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="shared",
                        namespace=NS, created_us=T0 + 1)
            seed_source(conn, store, source_id="s3", text="shared",
                        namespace=NS, created_us=T0 + 2)
            d = payload_hmac_hex(store, "shared")
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=d)
            link_exact(conn, source_id="s3", revision=1, namespace=NS,
                       digest=d)
            erase_source(conn, store, "s1")
            drop_member(conn, "s1", 1)
        with store.read() as conn:
            assert _all_group_ids(conn) == {"s2"}
            members = group_members(conn, "s2")
            assert {m["source_id"] for m in members} == {"s2", "s3"}
            assert representative(conn, "s2")["source_id"] == "s2"

    def test_new_link_anchors_to_live_not_forgotten(self, store):
        """A fresh identical submission groups with live members — the
        forgotten earliest source never anchors new links."""
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="shared",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="shared",
                        namespace=NS, created_us=T0 + 1)
            link_exact(conn, source_id="s2", revision=1, namespace=NS,
                       digest=payload_hmac_hex(store, "shared"))
            erase_source(conn, store, "s1")
            drop_member(conn, "s1", 1)
            seed_source(conn, store, source_id="s4", text="shared",
                        namespace=NS, created_us=T0 + 3)
            out = link_exact(conn, source_id="s4", revision=1,
                             namespace=NS,
                             digest=payload_hmac_hex(store, "shared"))
        assert out.linked is True
        assert out.group_id == "s2"   # earliest LIVE member, not dead s1

    def test_non_live_source_cannot_link(self, store):
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="x",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="x",
                        namespace=NS, created_us=T0 + 1)
            erase_source(conn, store, "s2")
            out = link_exact(conn, source_id="s2", revision=1,
                             namespace=NS,
                             digest=payload_hmac_hex(store, "x"))
        assert out.linked is False
        assert out.reason == "not_live"


# ---------------------------------------------------------------------------
# field extraction sanity (guard inputs)
# ---------------------------------------------------------------------------


class TestFields:
    def test_fields_from_text_extracts_guard_dimensions(self):
        f = links.fields_from_text("Deploy-v1 finished on monday")
        assert f.identifiers  # version identifier extracted (folded)
        assert any("v1" in v for v in f.identifiers | f.number_tokens)
        assert f.number_tokens  # version/number tokens present
        assert f.time_expressions  # explicit time expr present
        assert f.polarity == "affirmative"

    def test_negated_polarity_detected(self):
        f = links.fields_from_text("I no longer like kale")
        assert f.polarity == "negated"

    def test_text_fields_dict_coercion(self):
        f = links._coerce_fields({"polarity": "NEGATED",
                                  "identifiers": ["Deploy-V1"]})
        assert f.polarity == "negated"          # folded at construction
        # norm/v1 folds punctuation/case — "Deploy-V1" → "deploy v1"
        assert f.identifiers == frozenset({"deploy v1"})

    def test_vacuous_fields_flagged(self):
        assert DedupFields().vacuous() is True


# ---------------------------------------------------------------------------
# edge cases: merge across methods, self-revision, caller fields, namespace
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_exact_link_merges_into_existing_minhash_group(self, store):
        """s3 byte-identical to s2 joins s2's existing group — groups
        merge on the earliest live member, methods coexist on rows."""
        near_a = ("the release pipeline for our staging environment in "
                  "this repository stays pinned to the artifact version 1")
        near_b = ("the release pipeline for our staging environment in "
                  "this repository stays pinned to the artifact version 1 "
                  "overall")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=near_a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=near_b,
                        namespace=NS, created_us=T0 + 1)
            link_near(conn, source_id="s2", revision=1, namespace=NS,
                      signature=signature_of(near_b))
            # s3 carries s2's exact bytes — joins the same group.
            seed_source(conn, store, source_id="s3", text=near_b,
                        namespace=NS, created_us=T0 + 2)
            out = link_exact(conn, source_id="s3", revision=1,
                             namespace=NS,
                             digest=payload_hmac_hex(store, near_b))
        assert out.linked is True
        assert out.group_id == "s1"   # earliest live member anchors all
        with store.read() as conn:
            members = group_members(conn, "s1")
            assert {m["source_id"] for m in members} == {"s1", "s2", "s3"}
            assert _all_group_ids(conn) == {"s1"}
            methods = {m["method"] for m in members}
        assert methods <= {"exact_digest", "minhash"}

    def test_same_source_revisions_are_not_duplicates(self, store):
        """A source's own revision chain never joins a duplicate group —
        revisions are one logical memory's history, not copies."""
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="same bytes",
                        namespace=NS, created_us=T0)
            add_revision(conn, store, source_id="s1", text="same bytes",
                         namespace=NS, revision=2, created_us=T0 + 1)
            out = link_exact(conn, source_id="s1", revision=2,
                             namespace=NS,
                             digest=payload_hmac_hex(store, "same bytes"))
        assert out.linked is False
        with store.read() as conn:
            assert _links_rows(conn, "s1") == []

    def test_link_near_uses_caller_text_fields(self, store):
        """Caller-supplied text_fields are honored even when the new
        record's enrichment row has not landed yet."""
        a = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1")
        b = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1 overall")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            # No enrichment row for s2 — the job calls link_near before
            # (or without) enrichment persistence.
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1, enrich=False)
            out = link_near(
                conn, source_id="s2", revision=1, namespace=NS,
                signature=signature_of(b),
                text_fields={"polarity": "affirmative", "type": "untyped",
                             "identifiers": [], "number_tokens": ["1"],
                             "time_expressions": []},
            )
        assert out.linked is True

    def test_caller_fields_still_veto(self, store):
        """Explicit text_fields carrying a differing dimension veto even
        though the stored enrichment rows would not differ."""
        a = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1")
        b = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1 overall")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 1)
            out = link_near(
                conn, source_id="s2", revision=1, namespace=NS,
                signature=signature_of(b),
                text_fields={"polarity": "negated"},
            )
        assert out.linked is False
        assert out.vetoes[0]["dimension"] == "polarity"

    def test_foreign_namespace_source_rejected(self, store):
        """A source that is not a member of the claimed namespace cannot
        link into it — fail closed, never a cross-namespace edge."""
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="x",
                        namespace=NS, created_us=T0)
            seed_source(conn, store, source_id="s2", text="x",
                        namespace="ns-other", created_us=T0 + 1)
            out = link_exact(conn, source_id="s2", revision=1,
                             namespace=NS,
                             digest=payload_hmac_hex(store, "x"))
        assert out.linked is False
        assert out.reason == "not_in_namespace"

    def test_scope_fallback_authority(self, store):
        """When the control artifact manages no rows for the namespace,
        membership falls back to sources.scope_id and links still form
        (pre-v5 store shape)."""
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="same bytes",
                        namespace=NS, created_us=T0, state=False)
            seed_source(conn, store, source_id="s2", text="same bytes",
                        namespace=NS, created_us=T0 + 1, state=False)
            out = link_exact(conn, source_id="s2", revision=1,
                             namespace=NS,
                             digest=payload_hmac_hex(store, "same bytes"))
        assert out.linked is True
        assert out.group_id == "s1"

    def test_scope_fallback_still_isolates_namespaces(self, store):
        """The scope fallback cannot widen visibility: identical bytes in
        a different scope are not candidates."""
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text="same bytes",
                        namespace=NS, created_us=T0, state=False)
            seed_source(conn, store, source_id="s2", text="same bytes",
                        namespace="ns-other", created_us=T0 + 1,
                        state=False)
            out = link_exact(conn, source_id="s2", revision=1,
                             namespace=NS,
                             digest=payload_hmac_hex(store, "same bytes"))
        assert out.linked is False
        assert out.reason == "not_in_namespace"

    def test_scan_bound_is_namespace_scoped(self, store):
        """Foreign-namespace rows must not consume the bounded scan —
        a near candidate inside the namespace is found even when newer
        foreign projections crowd the recency window."""
        a = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1")
        b = ("the release pipeline for our staging environment in this "
             "repository stays pinned to the artifact version 1 overall")
        with store.tx() as conn:
            seed_source(conn, store, source_id="s1", text=a,
                        namespace=NS, created_us=T0)
            # 5 newer foreign-namespace projections would push s1 out of
            # a global recent-3 window — the bound applies per namespace.
            for i in range(5):
                seed_source(conn, store, source_id=f"f{i}",
                            text=f"foreign filler entry number {i} here",
                            namespace="ns-other", created_us=T0 + 10 + i)
            seed_source(conn, store, source_id="s2", text=b,
                        namespace=NS, created_us=T0 + 20)
            out = link_near(conn, source_id="s2", revision=1,
                            namespace=NS, signature=signature_of(b),
                            scan_limit=3)
        assert out.linked is True
        assert out.linked_to == ("s1", 1)

    def test_missing_links_table_is_typed_error(self, tmp_path):
        """On a store without the v5 link table, linking reports
        CAPABILITY_UNAVAILABLE rather than a bare SQL error."""
        from verbatim.storage.store import Store

        s = Store.create(str(tmp_path / "v.db"))
        try:
            with s.tx() as conn:
                conn.execute("DROP TABLE duplicate_links")
                with pytest.raises(VerbatimError) as ei:
                    link_exact(conn, source_id="s1", revision=1,
                               namespace=NS, digest="00")
            assert ei.value.code.value == "CAPABILITY_UNAVAILABLE"
        finally:
            s.close()
