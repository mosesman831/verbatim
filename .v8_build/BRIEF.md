# V8 BUILD — shared worker brief (read SPEC_V8.md first; it is normative)

Repo: /workspace/memorysys (branch main, HEAD 1c86f4f). Spec: SPEC_V8.md in repo root.
Python: use `.venv/bin/python` and `.venv/bin/pytest` (pytest 8.4.2, py 3.12). Do NOT use system python3 for tests.
Run tests: `.venv/bin/pytest tests/<area> -x -q`. Check neighboring test files for fixtures/conventions first.

## Ownership rule (hard)
You own ONLY the files listed in your task + NEW files you create inside your listed dirs + test files for your area.
If you need a change in a file you don't own, DO NOT edit it — record `HANDOFF: <file>: <what you need>` in your report.
Multiple workers are editing this tree concurrently. Never revert or "fix" changes you didn't make.
Do not commit. Do not push. Do not run git commands that modify state.

## Invariants (violating any = failed work)
1. Product defaults unchanged: `Memory.search` default `timeout_ms=500`, `admission.require_review=True`. Eval arms may override via `VerbatimArm(timeout_ms=…, policy_overrides=…)`.
2. Additive only: no existing column/table changes type or meaning; new DDL is idempotent (`CREATE TABLE IF NOT EXISTS` / the `ensure_v7_additive` pattern in verbatim/storage/schema_v7.py).
3. Generation fencing: every reader of units-derived tables filters `generation <= <pinned>` and takes latest row per unit (see graph.py:401-411 for the pattern).
4. Fail-closed eligibility: a lane that cannot apply eligibility returns error + zero candidates — never unfiltered output.
5. No new dependencies, no network, no model downloads, no Postgres/vector/graph servers. stdlib + existing deps only.
6. Determinism: same inputs → identical outputs; only `t_*`/`t_ms`/`slice_ms` fields are exempt from determinism comparison.
7. Closure: every new artifact must be purge-covered and verifier-checked (SPEC_V8 V8-19.03 table).
8. Exact byte pins: delivered units keep their pinned byte spans; injected context never replaces the delivered unit.
9. Honest coverage: every declared flag/coverage key must be actually read/written — declared-but-dead flags are defects (V8-23.02).
10. Follow existing code conventions: spec-ID comments (`# V8-NN.MM`) are the codebase norm; keep them. Compact code, no gratuitous comments.

## Flag/config mechanism
Arms/flags live in SPEC_V8 §23 and flow through `load_policy()` in verbatim/retrieval/v7/policy.py (line ~419) via `VerbatimArm(policy_overrides={...})`. Read `load_policy`, `GatedPolicyV7`, and `LANE_COSTS_V1` before adding a knob — reuse the existing params/overrides plumbing; don't invent a parallel config path. Query-time knobs that must reach lanes travel on the policy object the pipeline already threads.

## Coverage/explain contract (V8-20.03/20.04)
New coverage keys and per-item explain fields are specified verbatim in §20 — implement the keys your area owns; emit `null` where not applicable; never emit a key your code didn't compute.

## Cross-worker contract (use these EXACT names — other workers consume them)

Candidate `signals` dict keys (set by the producing lane/stage, consumed by fusion+explain):
- `signals["joint"]=True` — entity lane, unit mentions ALL resolved canons (V8-10.03)
- `signals["facet"]=<int>` — facet index the candidate was produced for (V8-10.02); whole-query candidates carry no key
- `signals["ctx_from"]=<unit_id>` + `signals["pre_ctx_score"]=<float>` — context injection/boost (V8-06)
- `signals["dense_slot"]=True` — candidate filled a guaranteed dense slot (V8-08.04)
- `signals["rescue"]=True` — lexical rescue-produced (V8-07.03)
- `signals["mention_interval"]=(start_us, end_us, precision)` — temporal mention match (V8-09.08)

Flags/arms (§23 — implement via load_policy/policy_overrides plumbing, defaults as in §23):
`verdict.premise_speaker`, `graph.gate`, `graph.contain_N_g`, `graph.K_fan`, `graph.M_seed`, `graph.hop_policy`, `graph.edge_types`, `context.mode`, `context.w`, `context.W`, `context.M_ctx`, `lexical.df_floor`, `lexical.K_rescue`, `lexical.rescue_rows`, `lexical.stem_lru`, `lexical.single_match`, `dense.B_max`, `dense.embed_batch`, `dense.N_d`, `fusion.dense_form`, `temporal.events_weights`, `facets.whole_share`, `facets.dense_per_facet`, `ent.per_canon_quota`, `pack.group_maxpool`, `rerank.speaker_match_weight`, `fusion.lex_anchor`, `scheduler.R_post`, `scheduler.post_pool`, `elig.cache_size`.

Schema (§19 — DDL lands in schema_v7.py via another worker): if your tests need `unit_time_mentions`/`unit_doclen` before it lands, create the tables IN YOUR TEST FIXTURE ONLY using the §19 SQL verbatim; production readers MUST use `has_table`-style guards and degrade honestly when absent (lane runs, labels `partial`, never crashes).

`coverage.*` keys per V8-20.03 — emit exactly those names for your area.

## Report format (required in your final message)
- `FILES:` every file created/modified (paths)
- `TESTS:` commands run + pass/fail counts
- `REQS:` V8-NN.MM requirement IDs implemented
- `FLAGS:` new flags/arms added and their defaults
- `HANDOFF:` needs outside your ownership (or `none`)
- `NOTES:` design decisions, deviations, risks
