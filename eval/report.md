# Verbatim v2 — synthetic-corpus baseline report

This report is **measured** output from `eval/harness.py` driving the public `verbatim.api.Engine` on a fully synthetic, locally generated corpus (no external benchmark data). Numbers below are observations of this configuration, not specification targets.

## Configuration

| setting | value |
|---|---|
| approve | `True` |
| mode | `offline_rules` |
| rebuild_fts | `True` |
| require_review | `True` |
| top_k | `5` |
| corpus_sha256 | `2884b134e96b31d2…` |
| corpus_seed | `42` |
| elapsed_s | 53.616 |

## Corpus

| metric | value |
|---|---|
| statements | 1000 |
| queries | 850 |
| update pairs | 300 |
| contradiction pairs | 20 |

## Ingestion and processing

| metric | value |
|---|---|
| envelopes accepted | 1000 / 1000 |
| duplicate receipts | 0 |
| claims created | 1000 |
| structured claims (predicate set) | 368 |
| pending after processing | 1000 |
| active (after operator approvals) | 700 |
| superseded | 300 |
| admit failures | 0 |
| supersede failures | 0 |
| job states | {"succeeded": 2000} |

## Extraction coverage

| metric | value |
|---|---|
| statements producing ≥1 claim | 1000 |
| extraction coverage | 1.0 |
| claims per statement | 1.0 |

## Recall (post-rebuild where applicable)

| query kind | n | hits | hit_rate | items_returned | notes |
|---|---|---|---|---|---|
| point | 250 | 180 | 0.72 | 1012 |  |
| current | 200 | 173 | 0.865 | 943 |  |
| historical | 200 | 178 | 0.89 | 994 |  |
| no_answer | 200 | 0 | 0.0 | 0 | false_positive_rate=0.0 |

**Latency (ms):** p50=3.566 p95=10.387 p99=12.392 max=14.247 (n=850)

### Pre-rebuild recall (generation-regression check)

| query kind | n | hits | hit_rate | items_returned |
|---|---|---|---|---|
| point | 250 | 177 | 0.708 | 1012 |
| current | 200 | 173 | 0.865 | 943 |
| historical | 200 | 116 | 0.58 | 994 |
| no_answer | 200 | 0 | 0.0 | 0 |

## FTS projection state

| metric | value |
|---|---|
| projection generation | 1 |
| claims indexed at current generation | 1000 |
| rebuild used | True |
| claims re-indexed by rebuild | 1000 |

## Store

| metric | value |
|---|---|
| db size (bytes) | 15839232 |

## Notes

- rebuild_fts re-indexed every previously indexed claim (active and superseded — HISTORICAL recall needs both) at the current projection generation via the ingester's index_claim path, simulating the projection-rebuild job the generation design anticipates.

## Specification targets vs measurement

The spec defines *targets*; this report records *measurements*. The two are shown side by side — agreement is a coincidence of this run, not a guarantee.

| quantity | spec target | measured |
|---|---|---|
| corpus statements | ~1000 | 1000 |
| update pairs | ~300 | 300 |
| no-answer probes | ~200 | 200 |
| unstructured evidence preserved | preserved (A01) | extraction coverage 1.0 |
| temporal update semantics | current→new / historical→old (A07) | current hit_rate 0.865, historical hit_rate 0.89 — measured against gold source links |
| abstention on unanswerable probes | no fabricated evidence | false_positive_rate 0.0 |

Scenario-level acceptance status lives in `tests/test_v2_acceptance.py` (A01–A25); each test asserts the spec behavior and marks genuinely missing capabilities with strict xfail.

## Honesty footer

- These numbers describe the `offline_rules` configuration on a synthetic corpus only. They do not measure semantic-encoder configurations, real conversational data, or any other system.
- The operator-assisted admission path (`apply_transition`) was used for approvals because the default configuration requires review; unassisted auto-admission is not measured here.
- Failed or errored queries remain in every denominator; nothing is dropped for looking bad.
