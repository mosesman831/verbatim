"""V8-23 flag/arm paired ablation driver — SPEC_V8 W3 keep-rule harness.

One ingested ``Memory`` store (``ForensicVerbatimArm``), one query loop
per :class:`ArmSpec` — the tightest pairing available (same items, same
store bytes, only the ``policy.params`` knob differs).  Per-question rows
feed ``eval/v8/stats.py`` paired bootstrap / exact McNemar for the
keep-rule (V8-00.06: a default-on feature must add ≥ +0.01 answerable
any@10, or be a cost-free speed/correctness fix; otherwise it ships off).

Each ablation toggles ONE §23 arm to its "off" / pre-V8 value; the
``v8_all_on`` spec is the shared baseline (all §23 priors).  Write-path
arms (``source_coalesce``, ``dense.embed_batch``, ``subject_source``
backfill) are ingest-time — they are NOT here; they need a paired
*ingest* run, not a shared-store query ablation, and are measured by the
envelope/starvation suites instead.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..v7.corpora import load_corpus
from ..v7.forensics._common import (
    ArmSpec,
    PolicyPatch,
    arm_report_kwargs,
    paired_run,
    write_report,
)

#: The W3 ablation matrix — name → the spec that realizes "off" for that
#: §23 arm on the shared store.  ``params`` values are the pre-V8 / off
#: settings; ``lanes_disabled`` removes the lane from the policy tuple
#: (the §02.3-correct ablation, never a config.v3.retrieval gate).
#: V8.5 earners-only default tuple (policy.LANES_V85) — the ``with_*``
#: arms add one removed lane back through a declared ``lanes`` list.
_V85_LANES = ["lex", "fuzzy", "dense", "time"]

ABLATIONS: Dict[str, Dict[str, Any]] = {
    # ---- V8.5 ship-set keep-rule arms (each vs the V85 baseline) ----
    # V85-05.02 co-occurrence nomination — off restores the union flood.
    "no_cooc": {"params": {"lexical.nom_cooc": False}},
    # V85-05.03 rare-term exemption — 0 disables (rare terms then need
    # the same ≥2-term co-occurrence as common ones).
    "no_rare_df": {"params": {"lexical.rare_df": 0}},
    # V85-05.01 earners-only — each removed lane added back singly.
    "with_ent": {"policy_doc_lanes": _V85_LANES + ["ent"]},
    "with_graph": {"policy_doc_lanes": _V85_LANES + ["graph"]},
    "with_typed": {"policy_doc_lanes": _V85_LANES + ["typed"]},
    "with_obs": {"policy_doc_lanes": _V85_LANES + ["obs"]},
    # V8-05 graph lane removal on the OLD tuple — kept for reference.
    "no_graph_legacy": {"lanes_disabled": ["graph"]},
    # V85-03.04 pack-measured context bundle (w=0.9, W=2, M_ctx=25) as
    # one arm vs the current priors (0.7/1/50).
    "ctx_ship": {"params": {"context.w": 0.9, "context.W": 2,
                            "context.M_ctx": 25}},
    # V85-05.07 — the two-phase keep-rule decision arm.
    "two_phase_off": {"params": {"scheduler.two_phase": False}},
    # V85-05.06 — global as_of re-anchor (pack measured −0.029).
    "as_of_global": {"params": {"temporal.as_of_scope": "global"}},
    # V85-05.04 — b4 measured any positive sm weight costing cat-5
    # (0.413→0.291 @0.4) for +0.011 answerable; sm=0 keeps the feature
    # dead-weight-free.  The keep-rule arm decides the default.
    "sm0": {"params": {"rerank.speaker_match_weight": 0.0}},
    # ---- carried V8 arms still meaningful on the V85 baseline ----
    # V8-06 context (propagation + injection).  off ⇒ pre-V8 ranking.
    "no_context": {"params": {"context.mode": "off"}},
    # V8-07.03 bounded rescue — 0 disables (pre-V8 thin-nomination miss).
    "no_rescue": {"params": {"lexical.K_rescue": 0}},
    # V8-07.02 df prefetch gate — "off" lets flood terms through.
    "no_df_gate": {"params": {"lexical.df_floor": "off"}},
    # V8-07.05 stem LRU — 0 disables the memo (latency arm).
    "no_stem_lru": {"params": {"lexical.stem_lru": 0}},
    # V8-14.01 eligibility snapshot cache — 0 disables (latency arm).
    "no_elig_cache": {"params": {"elig.cache_size": 0}},
    # V8-12.01 premise_speaker — armed (default is advisory-off).
    "premise_speaker_on": {"params": {"verdict.premise_speaker": True}},
    # V8-08.04 dense top-N sweep.  ``dense_N10`` is the exact pre-V8.5
    # default — the ledger record for the 10→0 don't-ship change pairs
    # against it.
    "dense_N5": {"params": {"dense.N_d": 5}},
    "dense_N10": {"params": {"dense.N_d": 10}},
    "dense_N20": {"params": {"dense.N_d": 20}},
    # V8-14.03/14.05 scheduler arms.
    "rpost30": {"params": {"scheduler.R_post": 30.0}},
    # V8-11.03 lexical anchor (default off → on measures its MRR effect).
    "lex_anchor_on": {"params": {"fusion.lex_anchor": True}},
    # V8-10.05 group maxpool pack variant (default off → on).
    "group_maxpool_on": {"params": {"pack.group_maxpool": True}},
    # V8-10.02 facet whole-share sweep.
    "facets_share0.3": {"params": {"facets.whole_share": 0.3}},
}


def build_specs(names: Sequence[str]) -> List[ArmSpec]:
    """Baseline + one ArmSpec per named ablation (shared-store query)."""
    specs: List[ArmSpec] = [
        ArmSpec(
            label="v8_all_on",
            patch=None,
            notes=["baseline — HEAD defaults (V8.5 ship-set priors)"],
        )
    ]
    for name in names:
        cfg = ABLATIONS.get(name)
        if cfg is None:
            raise KeyError(
                f"unknown ablation {name!r}; registered: {sorted(ABLATIONS)}"
            )
        doc: Dict[str, Any] = {}
        if cfg.get("params"):
            doc["params"] = dict(cfg["params"])
        if cfg.get("policy_doc_lanes"):
            doc["lanes"] = list(cfg["policy_doc_lanes"])
        patch = PolicyPatch(
            lanes_disabled=cfg.get("lanes_disabled"),
            policy_doc=doc or None,
        )
        specs.append(
            ArmSpec(
                label=name,
                patch=patch,
                notes=[f"§23 ablation {name!r}: {json.dumps(cfg)}"],
            )
        )
    return specs


def _reuse_ingest(arm: Any, corpus: Any, store_path: str) -> Dict[str, Any]:
    """``ForensicVerbatimArm.ingest`` replacement for a prebuilt store.

    Copies ``store_path`` (+ its ``.key`` sibling) into the arm workdir
    and opens it — no adds, no drain.  The attribution maps are rebuilt
    by byte-exact payload match against ``source_revisions.payload``
    (the V7-22 arm-parity contract: ``item_document`` bytes ARE the
    ingested payload).  Items whose document never appears in the store
    stay out of ``_indexed`` — attribution reports them ``not_indexed``
    rather than pretending coverage.  Byte-identical documents dedupe to
    the first ref, matching the live-ingest collision rule.
    """
    from verbatim import Memory

    from ..v7.arms import _group_anchors, corpus_items, item_document, item_ref

    items = corpus_items(corpus)
    t0 = time.perf_counter()
    dst = os.path.join(arm._workdir, "mem.db")
    shutil.copy2(store_path, dst)
    key_src = store_path + ".key"
    if os.path.exists(key_src):
        shutil.copy2(key_src, dst + ".key")

    mapping = dict(arm._config_mapping)
    for key, val in arm._policy_overrides.items():
        if isinstance(val, Mapping) and isinstance(
            mapping.get(key), Mapping
        ):
            mapping[key] = {**mapping[key], **val}
        else:
            mapping[key] = val
    arm._mem = Memory(
        dst,
        user_id=arm.user_id,
        worker=arm.worker,
        encoder=arm.encoder,
        config=mapping or None,
        **arm._memory_kwargs,
    )
    arm._group_as_of = _group_anchors(items)

    want: Dict[bytes, List[str]] = {}
    for i, item in enumerate(items):
        ref = item_ref(item, i)
        try:
            doc = item_document(item)
        except Exception:  # noqa: BLE001 — degenerate item, never indexed
            continue
        want.setdefault(doc.encode("utf-8"), []).append(ref)

    matched: set = set()
    collisions = 0
    with arm._mem._store.read() as conn:
        rows = conn.execute(
            "SELECT source_id, payload FROM source_revisions"
        ).fetchall()
    for sid, payload in rows:
        pb = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        refs = want.get(pb)
        if not refs:
            continue
        if arm._source_ref.get(sid) not in (None, refs[0]):
            collisions += 1
        arm._source_ref.setdefault(sid, refs[0])
        for ref in refs:
            arm._ref_source[ref] = sid
            matched.add(ref)
    arm._indexed = [
        item_ref(it, i) for i, it in enumerate(items)
        if item_ref(it, i) in matched
    ]
    unmatched = [r for refs in want.values() for r in refs
                 if r not in matched]
    if collisions:
        arm.notes.append(
            f"{collisions} byte-identical adds deduped to an earlier "
            "item's source (first ref wins)"
        )
    arm.notes.append(
        f"reuse_store: opened {store_path!r} — no adds, no drain; "
        f"attribution rebuilt by payload match"
    )
    arm.last_ingest = {
        "reuse_store": store_path,
        "indexed": len(arm._indexed),
        "unmatched_items": unmatched,
        "dedup_collisions": collisions,
        "ingest_ms": (time.perf_counter() - t0) * 1000.0,
        "config_overrides": mapping,
    }
    return dict(arm.last_ingest)


@contextlib.contextmanager
def _reuse_store(store_path: str):
    """Scope ``ForensicVerbatimArm.ingest`` to the prebuilt store — the
    paired specs still share one store byte-for-byte, just not a fresh
    ingest.  Restored unconditionally on exit."""
    from ..v7.forensics._common import ForensicVerbatimArm

    orig = ForensicVerbatimArm.ingest
    ForensicVerbatimArm.ingest = (
        lambda self, corpus: _reuse_ingest(self, corpus, store_path)
    )
    try:
        yield
    finally:
        ForensicVerbatimArm.ingest = orig


def run_ablations(
    corpus: Any,
    names: Sequence[str],
    *,
    k_list: Sequence[int] = (10, 20),
    arm_kwargs: Optional[Mapping[str, Any]] = None,
    full_explain: bool = False,
    census: bool = False,
    out: Optional[str] = None,
    reuse_store: Optional[str] = None,
) -> Dict[str, Any]:
    specs = build_specs(names)
    manifest = {
        "requirement": "V8-23 keep-rule paired ablation",
        "ablations": list(names),
        "shared_store": True,
        "keep_rule": "default-on needs ≥ +0.01 answerable any@10 "
                     "or a cost-free speed/correctness fix (V8-00.06)",
    }
    if reuse_store:
        manifest["reuse_store"] = reuse_store
    with (_reuse_store(reuse_store) if reuse_store
          else contextlib.nullcontext()):
        report = paired_run(
            corpus,
            specs,
            k_list=k_list,
            arm_kwargs=arm_kwargs,
            full_explain=full_explain,
            need_census=census,
            tool="v8_flag_ablation",
            extra_manifest=manifest,
        )
    write_report(report, out)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v8.ablate",
        description="V8-23 §23 flag/arm paired ablation (shared store)",
    )
    ap.add_argument("--dataset", required=True,
                    help="registry id or corpus JSON path")
    ap.add_argument("--split", default=None)
    ap.add_argument("--ablations", default=",".join(ABLATIONS),
                    help="comma list (default: every registered ablation)")
    ap.add_argument("--list", action="store_true",
                    help="print the ablation registry and exit")
    ap.add_argument("--k", default="10,20")
    ap.add_argument("--timeout-ms", type=float, default=None)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--settle-timeout", type=float, default=None)
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--full-explain", action="store_true")
    ap.add_argument("--store", default=None,
                    help="reuse a prebuilt verbatim store (mem.db) "
                         "instead of a fresh ingest — attribution maps "
                         "are rebuilt by byte-exact payload match")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.list:
        for name, cfg in ABLATIONS.items():
            print(f"{name:22} {json.dumps(cfg)}")
        return 0

    names = [n.strip() for n in args.ablations.split(",") if n.strip()]
    corpus = load_corpus(args.dataset, args.split)
    ks = tuple(int(x) for x in args.k.split(",") if x.strip())
    rep = run_ablations(
        corpus,
        names,
        k_list=ks,
        arm_kwargs=arm_report_kwargs(
            timeout_ms=args.timeout_ms,
            pool_limit=args.pool_limit,
            settle_timeout_s=args.settle_timeout,
        ),
        full_explain=args.full_explain,
        census=args.census,
        out=args.out,
        reuse_store=args.store,
    )
    # Per-spec headline: answerable any@10 / mrr@10 delta vs the shared
    # baseline (the keep-rule read).  ``overall`` is the all-task item
    # aggregate; ``categories`` carries the answerable split.
    specs = rep.get("specs") or {}
    base = (specs.get("v8_all_on") or {}).get("overall") or {}
    base_any = base.get("any@10")
    sys.stderr.write(
        f"status={rep.get('status')} not_run={rep.get('not_run')}\n"
    )
    sys.stderr.write(
        f"{'spec':22} {'any@10':>8} {'mrr@10':>8} {'Δany':>8} "
        f"{'applied':>7}\n"
    )
    for label, blk in specs.items():
        ov = blk.get("overall") or {}
        any10 = ov.get("any@10")
        delta = (
            round(any10 - base_any, 4)
            if (any10 is not None and base_any is not None
                and label != "v8_all_on")
            else None
        )
        sys.stderr.write(
            f"{label:22} "
            f"{any10 if any10 is not None else float('nan'):>8.4f} "
            f"{ov.get('mrr@10', float('nan')):>8.4f} "
            f"{delta if delta is not None else '—':>8} "
            f"{str(blk.get('applied')):>7}\n"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
