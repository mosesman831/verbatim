# V8-05 graph-lane verification — measured results

Repo `mosesman831/verbatim` @ `1c86f4f` (read-only clone; rescue = patched copy at `/tmp/vb_rescue`). Store `/tmp/v7w/mem.db`: units=4143 (latest gen), graph_edges=144,995 by type `{co_mention: 56580, same_session: 40725, temporal_near: 28948, adjacent_turn: 17879, causal: 863}`. Corpus: locomo dev split, 990 tasks (777 answerable, 213 adversarial-premise), 2877 items. Two measurement harnesses (probe = raw `run_search` internals; score = full `VerbatimArm.query` → pack → §33 metrics), sqlite trace-callback stmt counting, all runs on the real store. Baseline reproduced: any@10 0.542 vs claimed 0.541, session 0.845 vs 0.852 — within run noise.

## Mechanism (why the lane emits noise)

`lane_graph` (verbatim/retrieval/v7/graph.py:351):
1. Per-node `unit_row` SELECT at :401-430, called per seed (:443), per edge row in the BFS loop via `is_eligible(other)` (:509), and per emitted candidate (:631) — measured **units stmts p50=411, mean=509/query** (the reported ~486 N+1; slightly higher here since trace counts every units hit).
2. BFS saturates `FRONTIER_CAP_V1=400` (:506) in round 1 — hop_hist p50 `{0:19, 1:380, 2:1}`; `frontier_truncated` on 990/990.
3. The W-subgraph query (:529-538, `src IN (400 ids) AND dst IN (400 ids)`) has **no deadline check before it** — it pulls ~7.5K edge rows after the budget is already gone.
4. Deadline trips at the iteration top (:564) → `iters_done=0 on all 990` → `r` stays at its init `r=p` (:559-560) where non-seeds score 0.0 → `order = sorted(key=(-r, unit_id))` (:625-628) collapses to **raw unit_id order** → the 200 emitted candidates are an arbitrary prefix of the visited set, not a ranking.
5. Slice accounting: GRAPH cost 40.0 in `LANE_COSTS_V1` (policy.py:158-164) → slice p50=148.3ms of the 2000ms budget; lane's measured elapsed p50=305.1ms (2.06× slice — the deadline machinery inside the lane checks too late, and `run_one` at lanes_base.py:196-200 downgrades elapsed>slice to `partial`/`deadline_overrun` post-hoc).
6. Dead seed plumbing: `_update_seeds` (pipeline.py:422-431) writes `ctx.manifest["seeds"]` (top-10 ENT ∪ top-10 LEX) as those lanes complete, but the lane dispatch at pipeline.py:876 calls `fn(ctx, qv, slice_)` — the seeds are never read. Graph self-seeds via `_derive_seeds` (graph.py:271) regardless.

## Task 1 — paired ablation, graph ON vs OFF (dev split, timeout_ms=2000)

OFF implemented by removing `LaneName.GRAPH` from `POL.LANES_V1` (policy.py:80-89) — not `lanes_disabled` (config.v3, doesn't gate V7 — verified trap).

| metric | graph ON | graph OFF | Δ (ON−OFF) |
|---|---|---|---|
| item any@10 | 0.542 | 0.553 | **-0.010** |
| item any@20 | 0.660 | 0.669 | -0.009 |
| item all@10 | 0.461 | 0.469 | -0.008 |
| mrr@10 | 0.282 | 0.298 | -0.017 |
| session any@10 | 0.845 | 0.855 | -0.009 |
| multi_hop any@10 (n=126) | 0.452 | 0.468 | -0.016 |
| multi_hop all@10 | 0.056 | 0.056 | 0.000 |
| adversarial (cat-5 premise) any@10 (n=213) | 0.399 | 0.408 | -0.009 |
| answerable-only any@10 (n=777) | 0.582 | 0.592 | -0.010 |
| answerable-only any@20 | 0.696 | 0.708 | -0.012 |
| p50 latency ms | 658.2 | 352.4 | +305.8 |
| p95 latency ms | 1531.1 | 617.0 | +914.1 |
| total stmts/q p50 / mean | 4905 / 5887 | 1692 / 2371 | -3213 / -3516 |
| graph stmts/q p50 / mean | 3251 / 3516 | 0 / 0 | — |

Paired any@10 flips (delivered-set, per task):
- all: graph was the deciding lane on **9** tasks (ON hit / OFF miss); flipped the other way on **19**; both 528, neither 434 → net **-10**
- multi_hop: +3 / -5; adversarial: +1 / -3; single_hop: +4 / -8; temporal: +0 / -3

Lane status @2000ms ON: `graph: partial 990/990` (0 ok). Sibling lanes: lex 990 ok, dense 990 ok, ent 976 ok, fuzzy 181 ok/767 skipped, obs 620 ok, time 205 ok, typed 364 ok — **graph is the only lane that never completes**.

Contribution probe (uid→ref mapped): graph emits ≥1 gold in **258/990** lane outputs, but only **48** of those survive into the fused top-10 via graph rank — and **0** tasks have a gold unit carried *exclusively* by graph.

**Verdict: the lane does NOT earn its slot — it is net-negative, not neutral.** Every recall metric improves with it off (the ID-ordered noise displaces real candidates in RRF), while p50 drops 306ms and ~3.2K stmts/q disappear.

## Task 2 — rescue prototype (/tmp/vb_rescue patch)

Patch at `/tmp/vb_rescue/verbatim/retrieval/v7/graph.py`: batched `_fetch_unit_rows(unit_id IN (...) AND generation<=?)` chunked ≤256, used for seeds, BFS-round dst nodes, and emission; `K_FAN_V1=8` per-node edge cap ordered `(-weight, dst_unit_id)`; deadline checks before each of the two edge SELECTs per round + every 64 rows; W subgraph skipped entirely when deadline already flagged.

Probe @2000ms:

| metric | baseline | rescue |
|---|---|---|
| graph elapsed p50 / p95 ms | 305.1 / 366.8 | **174.1 / 211.9** (-43%) |
| slice p50 ms | 148.3 | 148.3 |
| status ok / partial | 0 / 990 | **93 / 897** |
| units stmts p50 / mean | 411 / 509.2 | **6 / 6.9** (≤8 target ✓) |
| visited mean | 398.5 | 149.4 |
| W edges mean | 7600 | 1001 |
| power iters ran on | 0/990 | 121/990 |
| n_cand mean | 199.7 | 138.4 |
| deadline_flag tasks | 990 | 878 |

Probe @500ms: units stmts 78.0→3.5 mean ✓; status still 981 partial (traversal doesn't fit in a 37ms slice); **n_cand mean 0.1** — rescue emits nothing when it can't finish (honest under-budget) vs baseline's ~44 noise candidates.

Score arms:

| arm | any@10 | any@20 | all@10 | mrr@10 | p50 | p95 | session any@10 |
|---|---|---|---|---|---|---|---|
| baseline @2000 | 0.542 | 0.660 | 0.461 | 0.282 | 658 | 1531 | 0.845 |
| **graph OFF @2000** | **0.553** | **0.669** | **0.469** | **0.298** | **352** | **617** | **0.855** |
| rescue @2000 | 0.552 | 0.664 | 0.466 | 0.291 | 523 | 954 | 0.848 |
| baseline @500 | 0.552 | 0.662 | 0.468 | 0.316 | 397 | 648 | 0.846 |
| rescue @500 | 0.557 | 0.662 | 0.473 | 0.329 | 351 | 610 | 0.847 |

**Verdict: the rescue fixes the mechanics (stmts -98.6%, -43% lane time, ok status finally reachable, honest-empty at 500ms) and closes most of the harm — but it ties OFF (0.552 vs 0.553), it does not beat it.** The residual bottleneck moved to the Python-side PPR over ~150 visited × ~1000 W-edges (~110ms inside a 148ms slice), so 90% of queries still flag deadline during scoring/emission even though traversal now completes. A faster lane is still buying zero exclusive contribution.

## Task 3 — per-edge-type traversal arms (@2000ms probe, `DECLARED_EDGE_TYPES=(t,)`)

| arm | status | visited mean | W edges mean | n_cand | elapsed p50 | iters ran | gold∩gcand | fused top-10 via graph (excl.) | fused_any_top-10 |
|---|---|---|---|---|---|---|---|---|---|
| all types (baseline) | partial 990 | 398.5 | 7600 | 199.7 | 305.1 | 0 | 258/990 | 48 (0) | 496 |
| only co_mention (56.6K edges) | ok 863 | 399.9 | 3961 | 200.0 | 109.5 | 918 | 281/990 | 81 (0) | 497 |
| only same_session (40.7K) | ok 954 | 382.7 | 3617 | 199.7 | 120.7 | 955 | 469/990 | **95** (0) | 526 |
| only temporal_near (28.9K) | ok 963 | 395.1 | 2633 | 199.9 | 86.5 | 964 | 478/990 | 93 (0) | 532 |
| only adjacent_turn (17.9K) | ok 966 | 382.4 | 1722 | 199.7 | 86.0 | 973 | **479**/990 | 67 (0) | **541** |
| only causal (863) | ok 990 | 63.2 | 85 | 55.3 | 12.2 | 990 | 39/990 | 12 (0) | 531 |

Read: it is the **type mix**, not traversal itself, that kills the lane — every single-type arm completes ≥96% and actually runs the power iterations. Reach into gold is roughly proportional to how tight the edge type is: adjacent_turn/temporal_near/same_session (turn/session-local edges) hit gold on ~47-48% of tasks and produce 3-9× more surviving top-10 contribution than the full mix, at one-third the cost — while `co_mention` (the largest family) is the worst signal-to-cost and `causal` is nearly free but nearly useless (859 edges, 12 via-graph). Note `adjacent_turn`-only also posts the best fused-level recall of any arm including baseline (541 vs 496).

## Task 4 — need-gate coverage (V8-05.05, evaluated on probe_on_2000)

| gate | fires |
|---|---|
| (a) primary intent ∈ {multi_hop, comparison} | 529/990 (53.4%) |
| (a) intent-classes contain multi_hop/comparison | identical |
| (a) full: (a-intent) ∨ canons ≥ 2 | **938/990 (94.7%)** — canons≥2 alone covers 936 |
| (b) core_union < 4·limit (limit=10 → 40; =20 → 80) | **1/990 (0.1%)** — non-graph lane union p50=489, p10=431 |
| (a) ∧ (b) — spec's compound gate | **0/990 (0.0%)** |
| (a) ∨ (b) | 939/990 (94.8%) |

**Verdict: the gate as specced is degenerate on this corpus.** Under the natural AND reading it silently ablates the lane (never fires); under an OR reading it admits ~95% of queries (canons≥2 is nearly universal — the corpus is entity-dense). A scarcity condition built on `core_union < 4·limit` cannot fire while seven sibling lanes each emit ~200 candidates (union p50=489 ≫ 80). And even where the gate would fire (multi_hop, its target class), OFF still beats ON (paired +3/−5) — gating cannot rescue a lane that's net-negative on its best population.

## Task 5 — seed selection arms (@2000ms probe)

| arm | seed_source | seeds p50 | elapsed p50 | status | gold∩gcand | via-graph top-10 (excl.) |
|---|---|---|---|---|---|---|
| derived (baseline `_derive_seeds`: ent-postings IDF ∪ unit_fts) | derived 990 | 19.0 | 305.1 | partial 990 | 258/990 | 48 (0) |
| manifest union (ENT∪LEX top-10 — the dead-plumbing path, made live) | provided 990 | 18.9 | 297.1 | partial 990 | 277/990 | 30 (0) |
| fused S2a top-20 (re-ran pre-graph lanes → rrf_fuse → top-20) | provided 990 | 20.0 | 490.3 | partial 990 | 289/990 | 10 (0) |

**Verdict: seed choice is not the bottleneck.** All three variants still saturate the 400-frontier, still run zero power iterations, still show 0 exclusive top-k gold — and contribution *falls* with "better" seeds (fused-seeded traversal reaches marginally more gold in its candidate pool, 289 vs 258, but far less of it survives fusion, 10 vs 48). Traversal breadth and the emission-collapse bug dominate; seeding is noise-level.

## Verdicts

**Recommendation: ABLATE (drop `LaneName.GRAPH` from `LANES_V1`) — measured OFF strictly dominates ON.**

- Does graph earn its slot? **No.** any@10 −0.010, mrr@10 −0.017, session −0.009, multi_hop any@10 −0.016 (its raison-d'être class), cat-5 premise −0.009 — all negative, plus +306ms p50, +914ms p95, +3,213 stmts/q p50 for the privilege. Paired net −10 tasks. Zero graph-exclusive gold at top-10 across every arm tested (baseline, rescue, 5 edge types, 3 seedings — all `fused_via_graph_only = 0`).
- Root cause (constants): `FRONTIER_CAP_V1=400` saturates in round 1; the 400×400 W query runs *after* the deadline; `iters_done=0` → `r=p` → `sorted(-r, unit_id)` = arbitrary 200-prefix into RRF; `unit_row` N+1 at graph.py:401 (call sites :443, :509, :631); lane overruns its own slice 2.06×. Fix list verified working: batched `unit_id IN (…)` + `K_FAN_V1=8` + deadline-before-SQL at /tmp/vb_rescue (units stmts 509→6.9, elapsed −43%, but recall only ties OFF).
- Ship-gated? **Non-starter** — the V8-05.05 gate is degenerate on locomo dev: a∧b fires 0/990, a∨b fires 94.8%.
- Rewrite path (only if graph stays on the roadmap): single/dual-type traversal — `adjacent_turn` first (479 gold reach, best fused recall 541), `temporal_near`/`same_session` second; drop `co_mention` (worst signal-to-cost) and `causal` (12 via-graph); keep the rescue's batching/K_fan/deadline patch; re-spec the need-gate on a discriminator that actually varies (core_union scarcity cannot fire when 7 lanes emit 200 each). Success bar for any rewrite: **beat graph-OFF (any@10 ≥0.553, mrr@10 ≥0.298, p50 ≤350ms), not baseline-ON.**
- Constants for the record: `K_FAN_V1=8`, `FRONTIER_CAP_V1=400`, `EXPANSION_ROUNDS_V1=3`, `POWER_ITERS_V1=3`, `DAMPING_V1=0.5`, lane cost GRAPH=40.0 (slice p50=148.3ms@2000), lane_cap=200, seed cap=19-20.
- Housekeeping found in passing: `ctx.manifest["seeds"]` written at pipeline.py:422-431 is dead plumbing (dispatch at :876 never passes it) — either wire it or delete it; `causal_candidate` absent from `EDGE_WEIGHTS_V1` but 863 `causal` rows exist in the store.
