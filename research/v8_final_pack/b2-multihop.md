# V8-10 multi-hop / aggregation arms — measured on the real dev store

Store: `/tmp/v7w/mem.db` (locomo dev, 2,877 items / 990 tasks / 4,143 units / 13,683 entity_mentions / 853 canons — built once by `eval.v7.track_r --arms verbatim,flat_bm25`, then every arm re-scored against the frozen store via a replay arm; `base` replay reproduces the build-run verbatim arm **bit-for-bit** — any@10 0.5414141414141415 — so all deltas below are attributable to the arm switch, nothing else).

All retrieval mechanics are env-gated (`V810_*` flags read at call time; result cache disabled), prototypes live only under `/tmp/proto` — nothing committed.

Denominators: overall n=990 scored (gold non-empty); multi_hop = cat-1, n=126, gold_mean=3.57 refs; "p50" column = wall ms over all 990 tasks; "mc-p50" = p50 over the 936 multi-canon queries from the mechanics probe.

## Headline table

| variant (env flags)                            | any@10 | all@10 | prop@10 | tok_mean | p50ms | mc-p50 | multi_hop any/all/prop@10 |
|------------------------------------------------|--------|--------|---------|----------|-------|--------|---------------------------|
| base (all off)                                 | .5414  | .4596  | .4952   | 1226.8   | 673   | 661    | .4206 / .0873 / .2206     |
| flat_bm25 (build run)                          | .5869  | .5030  | .5404   | —        | 4.3   | —      | .4127 / .0635 / .2032     |
| decomp (10.01)                                 | .5364  | .4535  | .4895   | 1227.5   | 680   | —      | .4286 / .0794 / .2206     |
| slots (10.02, share=.5)                        | .5444  | .4606  | .4981   | 1224.3   | 672   | —      | .4206 / .0794 / .2230     |
| decomp+slots                                   | .5424  | .4596  | .4965   | 1224.0   | 682   | —      | .4365 / .0794 / .2248     |
| **decomp+slots+scope (01+02+06, lex,ent)**     | **.5465** | **.4667** | **.5026** | 1222.0 | 658 | **662** | .4286 / **.0873** / **.2294** |
| scope (10.06 alone)                            | .5404  | .4586  | .4942   | 1226.1   | 663   | —      | .4206 / .0873 / .2206     |
| joint (10.03)                                  | .5394  | .4576  | .4932   | 1226.2   | 680   | 658    | .4206 / .0873 / .2206     |
| joint+spk (10.03 + speaker-pair)               | .5394  | .4586  | .4937   | 1225.7   | 661   | —      | .4206 / .0873 / .2206     |
| spk (speaker-pair alone)                       | .5384  | .4586  | .4929   | 1225.1   | 676   | —      | .4206 / .0873 / .2179     |
| quota50 (10.04)                                | .5455  | .4596  | .4967   | 1224.2   | 654   | 644    | .4206 / **.0556** / .2008 |
| grouppool (10.05, expand=0)                    | .4364  | .3677  | .3968   | 1185.9   | 682   | 657    | .3413 / .0317 / .1488     |
| grouppool+expand6                              | .2737  | .2303  | .2489   | 1114.2   | 690   | —      | .1825 / .0000 / .0726     |
| all-arms combo                                 | .2929  | .2424  | .2636   | 1115.6   | 685   | —      | .2143 / .0000 / .0747     |

## Per-category detail (selected)

### decomp+slots+scope (ship candidate) vs base

| cat         | n   | any@10 Δ  | all@10 Δ  | prop@10 Δ |
|-------------|-----|-----------|-----------|-----------|
| adversarial | 213 | .4178→.4225 +.005 | .3991→.4085 +.009 | .4085→.4155 +.007 |
| multi_hop   | 126 | .4206→.4286 +.008 | .0873→.0873 ±.000 | .2206→.2294 +.009 |
| open_domain | 47  | .2979→.2979 ±.000 | .1702→.1915 +.021 | .2122→.2376 +.025 |
| single_hop  | 429 | .5944→.6014 +.007 | .5478→.5548 +.007 | .5719→.5789 +.007 |
| temporal    | 175 | .7143→.7143 ±.000 | .6629→.6686 +.006 | .6867→.6895 +.003 |

Task-level flips vs base (prop@10): multi_hop 4+/0−, single_hop 4+/1−, temporal 1+/1−, open_domain 2+/2− — one-directional gains, near-zero collateral.

### quota50 (the churn case)

| cat | n | any Δ | all Δ | prop Δ | flipped tasks |
|-----|---|-------|-------|--------|----------------|
| adversarial | 213 | **+.066** | +.066 | +.066 | moderate |
| multi_hop   | 126 | ±.000 | **−.032** | −.020 | 42/126 churned |
| single_hop  | 429 | −.009 | −.002 | −.006 | 121/429 churned |
| temporal    | 175 | −.040 | −.046 | −.044 | 56/175 churned |

High churn both directions — quota reordering shuffles ~30% of answerable tasks with net-negative coverage outside adversarial.

### grouppool — collapse evidence

multi_hop examples (prop@10): conv-30/q005 0.67→0.00 — base pack D18:12,D18:1,D5:17,D2:4 (3 sessions) → arm D18:1,D18:18,D18:7,D18:13 (single session floods). conv-30/q025, conv-42/q011 same pattern: best group’s siblings displace the other half of the question. 481/429 single_hop and 86/126 multi_hop tasks flipped, mostly down.

## Mechanics probe (936 multi-canon queries, `_search_v7` limit=10)

| variant | p50ms | facet produced | facet kept | rolled over | joint emitted-1st | delivered joint | delivered facet | quota rows dropped | gp groups |
|---------|-------|---------------|-----------|-------------|-------------------|-----------------|-----------------|--------------------|-----------|
| base    | 661   | (≈71,854 intent facets; stats uncounted) | — | — | — | — | 208 | — | — |
| d+s+scope | 662 (+1ms) | 129,351 | 37,369 | 2,631 | — | — | 1,753 | — | — |
| joint   | 658   | 71,854 | legacy-append | — | 22,034 | 4,424 | 205 | — | — |
| grouppool | 657 | 71,854 | legacy-append | — | — | — | 144 | — | 34,440 |
| quota50 | 644   | 50,507 | legacy-append | — | — | — | 726 | 594,727 | — |

## Findings per SPEC arm

**V8-10.01 structural decomposition.** 202/990 dev questions decompose (≥2 resolved canons + coordination marker): multi_hop 30/126, adversarial 43/213, single_hop 82/429, open_domain 14/47, temporal 33/175. (936/990 resolve ≥2 canons — nearly every question mentions speaker+object — so the coordination gate is what keeps this selective.) Alone: overall −0.005, multi_hop any +0.008 / all −0.008 — facets without reserved slots still die at `merged[:capv]`.

**V8-10.02 reserved facet slots (share=0.5).** With decomp: whole 100 / facets split 100 equally, roll-over works (2,631 slots rolled back to the whole tail). Facet-originated items in delivered packs go 208→1,753 (8.4×). Only arm that lifts multi_hop any@10 (+0.016 for 01+02) — needs scope (below) to stop the all@10 bleed.

**V8-10.06 facet lane scope (lex,ent).** Removes facet work on every other lane. mc-p50 = +1ms (≤ +10ms bound, essentially free); on the full run it *saves* −15ms p50. Combined 01+02+06 is the only variant that beats base on every overall metric and holds multi_hop all@10 at parity while prop@10 rises +0.009.

**V8-10.03 conjunctive joint.** The probe IS cheap and effective mechanically — 22,034 co-mentioning units promoted to the head, 4,424 reach packs (~47% of multi-canon packs). But delivered sets are identical to base (0 multi_hop flips; identical any/all/prop): the eligible pool already contains co-mentioning units and qcov≥2's `ln()` boost already ranks them top. The r6 "co-mention blindness" premise doesn't bind at this df (scan row-limit covers all 13,683 mentions). Speaker-pair adds nothing (mh prop −0.003). No measured gain → don't ship.

**V8-10.04 per-canon quota 50.** 594,727 rows dropped, −17ms p50 — but the quota poisons multi_hop (all@10 0.087→0.056, prop −0.020) while *helping* adversarial (+0.066 any). df-truncation cuts the co-mentioning tail where aggregation evidence lives; adversarial gains are hot-canon-thinning luck, not the target. Don't ship at 50.

**V8-10.05 group→member max-pool.** Catastrophic, and worse with expansion (expand6: overall any 0.274, mh all@10 = 0.000). At ~equal delivered tokens (1,186 vs 1,227) prop@10 −0.098 — the mechanism itself, not budget. Root cause: one session's siblings flood top-10 (see q005 above) — gold is split across sessions, and assigning group-max score to every member destroys intra-group ranking. Do not pursue this shape; if grouping is ever wanted it needs per-member scores preserved + a same-session pack cap, not score homogenization.

**Contamination cases.** (a) quota50: 121 single_hop + 56 temporal tasks churn — conv-30/q024 loses D10:7/D14:3 for D8:12/D5:22; (b) grouppool: session-flood losses — conv-30/q005, conv-30/q025, conv-42/q011 all 0.5–0.67→0.00; (c) decomp-alone temporal: 3 all− flips — whole-query tail truncation costs the shared predicate. Ship arm's flip profile is clean.

## Verdicts

**Ship (one switch set):** structural decomposition + reserved slots + scoped facet lanes —
`DECOMP` on queries with ≥2 resolved canons AND a coordination marker; `WHOLE_SHARE=0.5` (floor(cap×0.5) whole, facets split remainder equally, sequential rollover + whole-tail backfill); `FACET_LANES=lex,ent`.
Measured: overall any +0.005 / all +0.007 / prop +0.007; multi_hop any +0.008, all ±0.000, prop +0.009; mc-p50 +1ms, full p50 −15ms. No category regresses.

**Don't ship:** `JOINT`/`JOINT_SPK` (zero delta — co-mention pool already reachable; 22k promotions change nothing), `CANON_QUOTA` (50-canon cap truncates the aggregation tail; multi_hop all −0.032), `GROUPPOOL`/`EXPAND` (score-homogenized session blocks collapse coverage — mh all@10 0.087→0.032/0.000).

**Targets not met — honest bound:** cat-1 all@10 stays 0.087 (target ≥0.20), prop@10 0.229 (target ≥0.40). No candidate-order arm closes it. gold_mean=3.57 vs top-10 with per-turn granularity means full coverage needs ~3.6 distinct gold-bearing refs — the miss is *unit granularity / cross-session packaging*, not ranking. The productive next lever is probably wider units (sentence_window/session kinds — currently 0 entity_mentions, so invisible to the ent lane) or explicit session-aggregation packs; NOT more postings reordering, and NOT group scoring as specified (it actively hurts).

**Repro:** `/tmp/proto` (verbatim@1c86f4f + edits in `verbatim/v810.py`, `querying/query_view.py`, `retrieval/v7/{pipeline,entity}.py`, `eval/v7/{arms,v810_replay,v810_probe}.py`); store `/tmp/v7w`; per-variant JSON `/tmp/v810_{a..d}.json`, mechanics `/tmp/v810_probe.json`, baseline `/tmp/v7_baseline.{json,md}`. Command shape: `VERBATIM_EVAL_LOCOMO=1 python -m eval.v7.v810_replay --workdir /tmp/v7w --variant <combo> --pool 10` (flags: decomp, slots[=share], joint, spk, quota=N, scope, grouppool, expand=N).
