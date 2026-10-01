"""Like-for-like latency/quality check: verbatim arm at timeout_ms=500
(the product default the committed baseline ran under), same corpus.
Writes eval/v7/reports/beatit_flat_500ms.json."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from eval.v7 import corpora, track_r  # noqa: E402
from eval.v7.arms import VerbatimArm  # noqa: E402

OUT_DIR = os.path.join("eval", "v7", "reports")


def main():
    corpus = corpora.load_corpus("owned_locomo_like")
    arm = VerbatimArm(timeout_ms=500.0)
    captured = []
    orig = arm.query

    def cap(task, k):
        out = orig(task, k)
        captured.append({"task_id": getattr(task, "task_id", None),
                         "lanes": out.diag.get("lanes")})
        return out

    arm.query = cap
    t0 = time.time()
    rep = track_r.run_track_r(corpus, [arm], (10, 20), seed=0)
    per_lane = {}
    for rec in captured:
        for n, s in (rec.get("lanes") or {}).items():
            per_lane.setdefault(n, {}).setdefault(str(s), 0)
            per_lane[n][str(s)] += 1
    rep["lane_status_scan"] = {"per_lane_status": per_lane}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "beatit_flat_500ms.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    ov = rep["arms"]["verbatim"]["overall"]
    lat = ov["latency_ms"]
    print(f"done {time.time()-t0:.0f}s")
    for k in ("any@10", "all@10", "prop@10", "any@20", "prop@20",
              "ndcg@10", "mrr@10", "correct_refusal", "abstain_rate",
              "false_abstention"):
        print(f"  {k:18s} {ov.get(k)}")
    print(f"  p50 {lat['p50']:.1f}  p95 {lat['p95']:.1f}")
    print("lanes:", json.dumps(per_lane))
    arm.close()


if __name__ == "__main__":
    main()
