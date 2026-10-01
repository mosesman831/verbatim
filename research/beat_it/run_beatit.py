"""Fix-wave delta measurement on the owned_locomo_like twin.

Runs the verbatim arm (eval-quality profile, timeout_ms=2000 default)
and the flat_bm25 reference arm sequentially in ONE process via
eval/v7/track_r.py machinery, writing fresh reports under
eval/v7/reports/ (committed v7_5 JSONs are untouched).

Also captures per-task lane statuses (coverage.lanes via
QueryOutcome.diag) so we can verify the dense lane never reports
PARTIAL post-fix.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from eval.v7 import corpora, track_r  # noqa: E402
from eval.v7.arms import FlatBM25Arm, VerbatimArm  # noqa: E402

OUT_DIR = os.path.join("eval", "v7", "reports")
K_LIST = (10, 20)


def write_report(report, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    js = os.path.join(OUT_DIR, f"{stem}.json")
    md = os.path.join(OUT_DIR, f"{stem}.md")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    with open(md, "w", encoding="utf-8") as fh:
        fh.write(track_r.render_markdown(report))
    return js, md


def lane_scan_summary(captured):
    """Aggregate captured coverage.lanes across all measured queries."""
    per_lane = {}
    partial_tasks = []
    for rec in captured:
        lanes = rec.get("lanes") or {}
        for name, status in lanes.items():
            st = str(status)
            per_lane.setdefault(name, {}).setdefault(st, 0)
            per_lane[name][st] += 1
            if name.endswith("dense") and st != "ok":
                partial_tasks.append((rec["task_id"], st))
    return {"per_lane_status": per_lane, "dense_non_ok_tasks": partial_tasks}


def main():
    corpus = corpora.load_corpus("owned_locomo_like")
    print(f"corpus: {corpus.name} items={len(corpus.items)} "
          f"tasks={len(corpus.tasks)} digest={corpus.digest()[:16]}")

    # --- verbatim (flat fusion, eval-quality deadline) -------------------
    arm = VerbatimArm()  # timeout_ms defaults to 2000.0 eval profile
    captured = []
    orig_query = arm.query

    def capturing_query(task, k):
        out = orig_query(task, k)
        captured.append({
            "task_id": getattr(task, "task_id", None),
            "lanes": out.diag.get("lanes"),
            "status": out.status,
            "warnings": list(out.warnings),
        })
        return out

    arm.query = capturing_query
    t0 = time.time()
    rep_v = track_r.run_track_r(corpus, [arm], K_LIST, seed=0)
    rep_v["lane_status_scan"] = lane_scan_summary(captured)
    js, md = write_report(rep_v, "beatit_flat")
    print(f"verbatim done in {time.time()-t0:.1f}s -> {js} , {md}")
    arm.close()

    # --- flat_bm25 reference ---------------------------------------------
    arm_b = FlatBM25Arm()
    t0 = time.time()
    rep_b = track_r.run_track_r(corpus, [arm_b], K_LIST, seed=0)
    js, md = write_report(rep_b, "beatit_bm25")
    print(f"flat_bm25 done in {time.time()-t0:.1f}s -> {js} , {md}")
    arm_b.close()

    # --- console summary --------------------------------------------------
    for stem, rep in (("verbatim", rep_v), ("flat_bm25", rep_b)):
        ov = rep["arms"][stem]["overall"]
        lat = ov["latency_ms"]
        print(f"\n== {stem} ==")
        for k in ("any@10", "all@10", "prop@10", "any@20", "all@20",
                  "prop@20", "ndcg@10", "mrr@10", "zero_rate",
                  "abstain_rate", "false_abstention", "correct_refusal"):
            print(f"  {k:18s} {ov.get(k)}")
        print(f"  p50 {lat.get('p50'):.1f}ms  p95 {lat.get('p95'):.1f}ms")
    print("\nlane scan:", json.dumps(rep_v["lane_status_scan"], indent=1))


if __name__ == "__main__":
    main()
