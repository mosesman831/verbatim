# V8.5 evaluation runbook — cloud agent

Everything needed to reproduce the V8.5 measurements and run the real
AMB benchmark. All commands run from the repository root unless noted.

**Machine prerequisites**: Python ≥3.11; `uv` for the AMB harness
(`curl -LsSf https://astral.sh/uv/install.sh | sh`); ≥4 cores and ≥8 GB
RAM recommended (ingest is CPU-bound; a dev store is ~250 MB on disk —
keep ~2 GB free for stores + WAL churn). Git for the harness clone.

Setup (venv or uv env recommended):

```bash
pip install -e '.[dev,semantic]'   # cryptography + pytest + numpy
# — or the minimal eval set: pip install pytest pyyaml cryptography numpy
```

The eval modules also run in-place from the repo root without install
(`verbatim/` and `eval/` are top-level packages on sys.path); the AMB
harness inserts `AMB_VERBATIM_REPO` itself, so no install is needed for
the provider path either.

Pre-flight correctness smoke (~seconds, run before the long jobs):

```bash
python -c "from eval.amb.provider import VerbatimAMBProvider; print('provider OK')"
python -m eval.v8.ablate --dataset locomo --list   # needs VERBATIM_EVAL_LOCOMO=1
python -m pytest tests -q -n auto --dist loadfile   # full unit suite (~20min on 4 cores)
```

Faster iteration during bring-up:

```bash
python -m pytest --lf -q                          # rerun only last-failed tests
python -m pytest tests/memory tests/retrieval/v7 -q -n 4 --dist loadfile
```

`pytest-xdist` is already in the dev extras. Serial runs (~80min) remain
the authoritative gate — `-n` parallelism is the iteration shortcut.

`tests/memory/test_write_starvation.py` was historically load-sensitive
(>250ms scheduling holes defeat the SQLite busy cap — spec-pinned
V4-40.03); the tests now tolerate single scheduling holes while still
catching real starvation. If it ever fails, rerun the file in isolation
before treating it as a regression.

## Dataset

`locomo10.json` is committed at `research/v7_formula_search/locomo10.json`
(sha256 `79fa87e9…`, pinned in `eval/v7/dataset_registry.py`). Every
LoCoMo command needs the license gate flag:

```bash
export VERBATIM_EVAL_LOCOMO=1
```

## 1. Internal evals (no LLM needed)

### Track R — retrieval + delivery on locomo dev (990 tasks)

```bash
python -m eval.v7.track_r \
  --dataset locomo --split dev \
  --arms verbatim,verbatim_msg,flat_bm25 \
  --k 10,20 \
  --out eval/v8/results/trackr_dev_v85.json \
  --md  eval/v8/results/trackr_dev_v85.md
```

~2×80 min serial (each verbatim arm ingests fresh). Reference numbers
(f5b6fcd): verbatim any@10 0.696 / mrr 0.392; verbatim_msg any@10 0.702 /
mrr 0.414 / GEIC@4.5K any 0.920 / all 0.816.

### GEIC proxy — delivery-only gate (ingests once, ~35 min)

```bash
python -m eval.v8.amb_proxy --dataset locomo --split dev \
  --out eval/v8/results/amb_proxy_dev.json \
  --md  eval/v8/results/amb_proxy_dev.md
```

Legacy-shape diagnosis reproduction — the ~494-token/no-expansion
delivery the v2-era AMB adapter produced. `--provider-budget 494`
shapes the provider's own retrieve; `--budgets 494,…` adds the matching
GEIC counterfactual row (delivered-gold coverage at exactly 494 tokens):

```bash
python -m eval.v8.amb_proxy --dataset locomo --split dev \
  --provider-budget 494 --window 0 --budgets 494,1000,4500 \
  --out eval/v8/results/amb_proxy_dev_legacyshape.json \
  --md  eval/v8/results/amb_proxy_dev_legacyshape.md
```

Caveat: this isolates the *delivery* variable on V8-quality hits — the
pack's 8% gold-string coverage also involved the old `Engine.recall`
retrieval path. A fully faithful G85-0 reproduction would need that
path re-run; the delivery-shape proxy bounds the delivery half of the
deficit honestly.

### Paired ablation matrix (23 arms — the keep-rule evidence)

One shared ingest per invocation. Two suggested shards (run in parallel
on ≥4 cores; ~5 h each):

```bash
python -m eval.v8.ablate --dataset locomo --split dev --k 10,20 \
  --ablations no_cooc,no_rare_df,with_ent,with_graph,with_typed,with_obs,no_graph_legacy,ctx_ship,two_phase_off,as_of_global,sm0 \
  --out eval/v8/results/ablate_v85_dev_a.json

python -m eval.v8.ablate --dataset locomo --split dev --k 10,20 \
  --ablations no_context,no_rescue,no_df_gate,no_stem_lru,no_elig_cache,premise_speaker_on,dense_N5,dense_N10,dense_N20,rpost30,lex_anchor_on,group_maxpool_on,facets_share0.3 \
  --out eval/v8/results/ablate_v85_dev_b.json
```

Analyze (bootstrap CI + McNemar + keep-rule per arm):

```bash
python -m eval.v8.analyze_ablate \
  eval/v8/results/ablate_v85_dev_a.json \
  eval/v8/results/ablate_v85_dev_b.json
```

`--store <path>` reuses a previously-built store instead of re-ingesting
(see `--help`); omit it on a fresh cloud agent.

Resume safety: a report is written only when its whole spec list
finishes. On a preemptible box prefer more, smaller invocations — each
`--ablations` subset still carries its own `v8_all_on` baseline, so
pairing is preserved per file:

```bash
python -m eval.v8.ablate --dataset locomo --split dev --k 10,20 \
  --ablations ctx_ship,sm0 --out eval/v8/results/ablate_ctx_sm.json
```

## 2. Real AMB run (needs an OpenAI-compatible gateway)

Owner decisions first: copy `eval/amb/authorizations.example.json` to
`eval/amb/authorizations.json` and set `granted: true` only where the
owner actually decided — `O1` (LoCoMo local eval), `O5` (reader/judge
spend), plus `amb_pin.commit` = the AMB checkout sha and
`amb_license.verified`. The in-repo runner (`eval.amb.runner`) reports
`blocked_on_authorization` for anything ungranted — by design.

```bash
# one-time: clone the pinned harness and install the provider + patches
eval/amb/harness/setup_harness.sh ./agent-memory-benchmark "$PWD"

# dataset: pin the harness to the repo's committed locomo10.json —
# sha256-identical to what the internal evals ingest (fair comparison),
# and skips the harness's GitHub download entirely.
export LOCOMO_DATA_PATH="$PWD/research/v7_formula_search/locomo10.json"

# per-run: gateway creds + model ids
export OPENAI_BASE_URL=<endpoint>/v1
export OPENAI_API_KEY=<key>
export OMB_ANSWER_MODEL=<reader model id>
export OMB_JUDGE_MODEL=<judge model id>        # defaults to reader

# full locomo10 — 10 parallel shards
eval/amb/harness/run_verbatim_locomo10.sh ./agent-memory-benchmark /tmp/amb-shards

# merge when all shard logs say done
python eval/amb/harness/merge_shards.py /tmp/amb-shards
```

G85-3 pass bar: accuracy ≥ 0.75, no category < 0.60, zero query errors,
on the **full locomo10 split** — reader/judge ids must be recorded
beside the result (`merged_locomo10.json` carries them).

Single-unit smoke before the full run (cheap sanity check):

```bash
cd agent-memory-benchmark && uv run amb run \
  --dataset locomo --split locomo10 --memory verbatim --unit conv-30
```

## Reporting rules (SPEC_V8_5 §07 — non-negotiable)

- Missing/unmeasurable metrics record `not_run`, never zero-filled.
- Reader/judge model ids ride every accuracy number.
- No "beats Hindsight/Mem0/…" claim without a same-harness run of that
  system under the identical reader+judge.
- `eval/v8/results/` is gitignored — upload artifacts separately if
  they need to persist.
