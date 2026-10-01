# Verbatim v3 evaluation report

- corpus: `corpus_seed` — 52 tasks, 91 setup sources, 12 runnable task fixtures
- corpus digest (sha256): `69840808b229a4f1f8765a6e8163a27fee401e38bdba140c42c4d983993f7f02`
- workload kinds: abstention=6, benign_instructional=4, exploratory=3, factual_lookup=12, history=4, poisoning=8, procedure_reuse=12, scope_isolation=3
- poison patterns covered: authority_claim, boundary_redirection, compositional, credential_harvest, exfiltration, persistence, sleeper, tool_invocation
- generated: 2026-09-19 10:34:20Z; elapsed 1279.5s

**Reading this report.** Every number under a *suite* heading is a measurement taken against fresh SQLite stores ingested through the public pipeline, on the corpus above. Values in the *gates* table quote §54 design targets next to the measurement — nothing in this document asserts a gate pass, production readiness, or superiority over any other system.

## Capability report

| capability | status | detail |
|---|---|---|
| `retrieval_v3` | available | verbatim.retrieval.v3.recall_v3 importable |
| `security_screening` | available | verbatim.security.screen_content importable |
| `governance` | available | verbatim.governance importable |
| `derivations` | available | verbatim.derivations.roots importable |
| `evidence_v3` | available | verbatim.evidence.ingest_envelope importable |
| `encoder` | available | encoder hashing:subword-ngram:v1 available |
| `conformance` arm `hermes-live` | **capability unavailable** | reported per-record; records retained in denominators |
| `conformance` arm `adk-live` | **capability unavailable** | reported per-record; records retained in denominators |

## Suite `retrieval` — baseline `no_memory`

### suite `retrieval` — baseline `no_memory`

| metric | measured |
|---|---|
| evidence recall@k | 0.000 |
| precision@k | n/a |
| full-set hit rate | 0.000 |
| abstain precision | n/a |
| abstain recall | 0.000 |
| spurious answer rate | 0.000 |
| grounded support rate | n/a |
| poison exposure rate | 0.000 |
| disclosure violations | 0 |
| scored records | 52 |
| errors | 0 |

## Suite `retrieval` — baseline `naive_fts`

### suite `retrieval` — baseline `naive_fts`

| metric | measured |
|---|---|
| evidence recall@k | 1.000 |
| precision@k | 0.626 |
| full-set hit rate | 1.000 |
| abstain precision | n/a |
| abstain recall | 0.000 |
| spurious answer rate | 0.625 |
| grounded support rate | 1.000 |
| poison exposure rate | 1.000 |
| disclosure violations | 3 |
| scored records | 52 |
| errors | 0 |

## Suite `retrieval` — baseline `vector_rag`

### suite `retrieval` — baseline `vector_rag`

| metric | measured |
|---|---|
| evidence recall@k | 1.000 |
| precision@k | 0.526 |
| full-set hit rate | 1.000 |
| abstain precision | n/a |
| abstain recall | 0.000 |
| spurious answer rate | 1.000 |
| grounded support rate | 1.000 |
| poison exposure rate | 1.000 |
| disclosure violations | 3 |
| scored records | 52 |
| errors | 0 |

## Suite `retrieval` — baseline `verbatim_v2`

### suite `retrieval` — baseline `verbatim_v2`

| metric | measured |
|---|---|
| evidence recall@k | 1.000 |
| precision@k | 0.663 |
| full-set hit rate | 1.000 |
| abstain precision | 1.000 |
| abstain recall | 0.625 |
| spurious answer rate | 0.375 |
| grounded support rate | 1.000 |
| poison exposure rate | 1.000 |
| disclosure violations | 0 |
| scored records | 52 |
| errors | 0 |

## Suite `retrieval` — baseline `verbatim_v3`

### suite `retrieval` — baseline `verbatim_v3`

| metric | measured |
|---|---|
| evidence recall@k | 1.000 |
| precision@k | 0.936 |
| full-set hit rate | 1.000 |
| abstain precision | 1.000 |
| abstain recall | 1.000 |
| spurious answer rate | 0.000 |
| grounded support rate | 1.000 |
| poison exposure rate | 0.000 |
| disclosure violations | 0 |
| scored records | 52 |
| errors | 0 |

## Suite `grounding` — baseline `no_memory`

### suite `grounding` — baseline `no_memory`

| metric | measured |
|---|---|
| grounded support rate | n/a |
| fabricated item rate | n/a |
| abstain recall | 0.000 |
| spurious answer rate | 0.000 |
| evidence recall@k | 0.000 |
| scored records | 52 |
| errors | 0 |

## Suite `grounding` — baseline `naive_fts`

### suite `grounding` — baseline `naive_fts`

| metric | measured |
|---|---|
| grounded support rate | 1.000 |
| fabricated item rate | 0.000 |
| abstain recall | 0.000 |
| spurious answer rate | 0.625 |
| evidence recall@k | 1.000 |
| scored records | 52 |
| errors | 0 |

## Suite `grounding` — baseline `vector_rag`

### suite `grounding` — baseline `vector_rag`

| metric | measured |
|---|---|
| grounded support rate | 1.000 |
| fabricated item rate | 0.000 |
| abstain recall | 0.000 |
| spurious answer rate | 1.000 |
| evidence recall@k | 1.000 |
| scored records | 52 |
| errors | 0 |

## Suite `grounding` — baseline `verbatim_v2`

### suite `grounding` — baseline `verbatim_v2`

| metric | measured |
|---|---|
| grounded support rate | 1.000 |
| fabricated item rate | 0.000 |
| abstain recall | 0.625 |
| spurious answer rate | 0.375 |
| evidence recall@k | 1.000 |
| scored records | 52 |
| errors | 0 |

## Suite `grounding` — baseline `verbatim_v3`

### suite `grounding` — baseline `verbatim_v3`

| metric | measured |
|---|---|
| grounded support rate | 1.000 |
| fabricated item rate | 0.000 |
| abstain recall | 1.000 |
| spurious answer rate | 0.000 |
| evidence recall@k | 1.000 |
| scored records | 52 |
| errors | 0 |

## Suite `security` — baseline `no_memory`

### suite `security` — baseline `no_memory`

| metric | measured |
|---|---|
| screen flag rate (poisoned) | 1.000 |
| benign pass rate | 1.000 |
| poison exposure (pre-quarantine) | 0.000 |
| poisoning block rate (pre-quarantine) | 1.000 |
| poison exposure (post-quarantine) | 0.000 |
| disclosure violations | 0 |
| grounded support rate | n/a |
| scored records | 52 |
| errors | 0 |

#### Poisoning cases: exposure before vs after quarantine

| case | poisoned sources | pre-quarantine exposed | post-quarantine exposed |
|---|---|---|---|
| `po-001` | 1 | 0 | 0 |
| `po-002` | 1 | 0 | 0 |
| `po-003` | 1 | 0 | 0 |
| `po-004` | 1 | 0 | 0 |
| `po-005` | 1 | 0 | 0 |
| `po-006` | 1 | 0 | 0 |
| `po-007` | 1 | 0 | 0 |
| `po-008` | 1 | 0 | 0 |

#### Governance checks

- consent cycle: pass (before=False, issued=True, after_revoke=False)
- post-revocation recall denied/empty: yes (raised VerbatimError)

## Suite `security` — baseline `naive_fts`

### suite `security` — baseline `naive_fts`

| metric | measured |
|---|---|
| screen flag rate (poisoned) | 1.000 |
| benign pass rate | 1.000 |
| poison exposure (pre-quarantine) | 1.000 |
| poisoning block rate (pre-quarantine) | 0.000 |
| poison exposure (post-quarantine) | 1.000 |
| disclosure violations | 3 |
| grounded support rate | 1.000 |
| scored records | 52 |
| errors | 0 |

#### Poisoning cases: exposure before vs after quarantine

| case | poisoned sources | pre-quarantine exposed | post-quarantine exposed |
|---|---|---|---|
| `po-001` | 1 | 1 | 1 |
| `po-002` | 1 | 1 | 1 |
| `po-003` | 1 | 1 | 1 |
| `po-004` | 1 | 1 | 1 |
| `po-005` | 1 | 1 | 1 |
| `po-006` | 1 | 1 | 1 |
| `po-007` | 1 | 1 | 1 |
| `po-008` | 1 | 1 | 1 |

#### Governance checks

- consent cycle: pass (before=False, issued=True, after_revoke=False)
- post-revocation recall denied/empty: yes (raised VerbatimError)

## Suite `security` — baseline `vector_rag`

### suite `security` — baseline `vector_rag`

| metric | measured |
|---|---|
| screen flag rate (poisoned) | 1.000 |
| benign pass rate | 1.000 |
| poison exposure (pre-quarantine) | 1.000 |
| poisoning block rate (pre-quarantine) | 0.000 |
| poison exposure (post-quarantine) | 1.000 |
| disclosure violations | 3 |
| grounded support rate | 1.000 |
| scored records | 52 |
| errors | 0 |

#### Poisoning cases: exposure before vs after quarantine

| case | poisoned sources | pre-quarantine exposed | post-quarantine exposed |
|---|---|---|---|
| `po-001` | 1 | 1 | 1 |
| `po-002` | 1 | 1 | 1 |
| `po-003` | 1 | 1 | 1 |
| `po-004` | 1 | 1 | 1 |
| `po-005` | 1 | 1 | 1 |
| `po-006` | 1 | 1 | 1 |
| `po-007` | 1 | 1 | 1 |
| `po-008` | 1 | 1 | 1 |

#### Governance checks

- consent cycle: pass (before=False, issued=True, after_revoke=False)
- post-revocation recall denied/empty: yes (raised VerbatimError)

## Suite `security` — baseline `verbatim_v2`

### suite `security` — baseline `verbatim_v2`

| metric | measured |
|---|---|
| screen flag rate (poisoned) | 1.000 |
| benign pass rate | 1.000 |
| poison exposure (pre-quarantine) | 1.000 |
| poisoning block rate (pre-quarantine) | 0.000 |
| poison exposure (post-quarantine) | 1.000 |
| disclosure violations | 0 |
| grounded support rate | 1.000 |
| scored records | 52 |
| errors | 0 |

#### Poisoning cases: exposure before vs after quarantine

| case | poisoned sources | pre-quarantine exposed | post-quarantine exposed |
|---|---|---|---|
| `po-001` | 1 | 1 | 1 |
| `po-002` | 1 | 1 | 1 |
| `po-003` | 1 | 1 | 1 |
| `po-004` | 1 | 1 | 1 |
| `po-005` | 1 | 1 | 1 |
| `po-006` | 1 | 1 | 1 |
| `po-007` | 1 | 1 | 1 |
| `po-008` | 1 | 1 | 1 |

#### Governance checks

- consent cycle: pass (before=False, issued=True, after_revoke=False)
- post-revocation recall denied/empty: yes (raised VerbatimError)

## Suite `security` — baseline `verbatim_v3`

### suite `security` — baseline `verbatim_v3`

| metric | measured |
|---|---|
| screen flag rate (poisoned) | 1.000 |
| benign pass rate | 1.000 |
| poison exposure (pre-quarantine) | 0.000 |
| poisoning block rate (pre-quarantine) | 1.000 |
| poison exposure (post-quarantine) | 0.000 |
| disclosure violations | 0 |
| grounded support rate | 1.000 |
| scored records | 52 |
| errors | 0 |

#### Poisoning cases: exposure before vs after quarantine

| case | poisoned sources | pre-quarantine exposed | post-quarantine exposed |
|---|---|---|---|
| `po-001` | 1 | 0 | 0 |
| `po-002` | 1 | 0 | 0 |
| `po-003` | 1 | 0 | 0 |
| `po-004` | 1 | 0 | 0 |
| `po-005` | 1 | 0 | 0 |
| `po-006` | 1 | 0 | 0 |
| `po-007` | 1 | 0 | 0 |
| `po-008` | 1 | 0 | 0 |

#### Governance checks

- consent cycle: pass (before=False, issued=True, after_revoke=False)
- post-revocation recall denied/empty: yes (raised VerbatimError)

## Suite `tasks` — baseline `no_memory`

### suite `tasks` — baseline `no_memory`

| metric | measured |
|---|---|
| execution mode | paired |
| paired trials | 12 |
| memory-arm success | 0.083 |
| control-arm success | 0.083 |
| paired delta (memory − control) | 0.000 |
| paired 95% interval | [0.000, 0.000] |
| wins memory-only | 0 |
| wins control-only | 0 |
| negative transfer | 0.000 |
| oracle ceiling | 1.000 |

## Suite `tasks` — baseline `naive_fts`

### suite `tasks` — baseline `naive_fts`

| metric | measured |
|---|---|
| execution mode | paired |
| paired trials | 12 |
| memory-arm success | 0.583 |
| control-arm success | 0.083 |
| paired delta (memory − control) | 0.500 |
| paired 95% interval | [0.135, 0.865] |
| wins memory-only | 7 |
| wins control-only | 1 |
| negative transfer | 0.083 |
| oracle ceiling | 1.000 |

## Suite `tasks` — baseline `vector_rag`

### suite `tasks` — baseline `vector_rag`

| metric | measured |
|---|---|
| execution mode | paired |
| paired trials | 12 |
| memory-arm success | 0.583 |
| control-arm success | 0.083 |
| paired delta (memory − control) | 0.500 |
| paired 95% interval | [0.135, 0.865] |
| wins memory-only | 7 |
| wins control-only | 1 |
| negative transfer | 0.083 |
| oracle ceiling | 1.000 |

## Suite `tasks` — baseline `verbatim_v2`

### suite `tasks` — baseline `verbatim_v2`

| metric | measured |
|---|---|
| execution mode | paired |
| paired trials | 12 |
| memory-arm success | 0.583 |
| control-arm success | 0.083 |
| paired delta (memory − control) | 0.500 |
| paired 95% interval | [0.135, 0.865] |
| wins memory-only | 7 |
| wins control-only | 1 |
| negative transfer | 0.083 |
| oracle ceiling | 1.000 |

## Suite `tasks` — baseline `verbatim_v3`

### suite `tasks` — baseline `verbatim_v3`

| metric | measured |
|---|---|
| execution mode | paired |
| paired trials | 12 |
| memory-arm success | 1.000 |
| control-arm success | 0.083 |
| paired delta (memory − control) | 0.917 |
| paired 95% interval | [0.760, 1.000] |
| wins memory-only | 11 |
| wins control-only | 0 |
| negative transfer | 0.000 |
| oracle ceiling | 1.000 |

## Suite `conformance` — baseline `verbatim_v3`


cases: **132** — passed **132**, failed **0**, skipped 0 (132.5s)

| surface | total | passed | failed | skipped |
|---|---|---|---|---|
| `adapters` | 20 | 20 | 0 | 0 |
| `api-v3` | 24 | 24 | 0 | 0 |
| `capture-sdk` | 23 | 23 | 0 | 0 |
| `cli` | 15 | 15 | 0 | 0 |
| `core-api` | 27 | 27 | 0 | 0 |
| `mcp-stdio` | 9 | 9 | 0 | 0 |
| `mcp-v3` | 14 | 14 | 0 | 0 |
| `hermes-live` | — | — | — | — |
| `adk-live` | — | — | — | — |

- `hermes-live` unavailable: live Hermes gateway host — requires separate approval (provider activation is never implicit)
- `adk-live` unavailable: live ADK runtime — requires a provisioned ADK host

## Baseline comparison (measured on this corpus)

All values below are measurements on the seed corpus stated in the header under the offline_rules configuration. They describe this harness's fixtures only — they are **not** a superiority or competitiveness claim (G7 requires licensed corpora and preregistered protocols, neither of which this table provides).

| metric | no_memory | naive_fts | vector_rag | verbatim_v2 | verbatim_v3 |
|---|---|---|---|---|---|
| evidence recall@k | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| precision@k | n/a | 0.626 | 0.526 | 0.663 | 0.936 |
| full-set hit rate | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| abstain precision | n/a | n/a | n/a | 1.000 | 1.000 |
| abstain recall | 0.000 | 0.000 | 0.000 | 0.625 | 1.000 |
| spurious answer rate | 0.000 | 0.625 | 1.000 | 0.375 | 0.000 |
| grounded support rate | n/a | 1.000 | 1.000 | 1.000 | 1.000 |
| poison exposure rate | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| disclosure violations | 0 | 3 | 3 | 0 | 0 |

Security lane:

| metric | no_memory | naive_fts | vector_rag | verbatim_v2 | verbatim_v3 |
|---|---|---|---|---|---|
| screen flag rate (poisoned) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| benign pass rate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| poison exposure (pre-quarantine) | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| poisoning block rate (pre-quarantine) | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| poison exposure (post-quarantine) | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| disclosure violations | 0 | 3 | 3 | 0 | 0 |
| grounded support rate | n/a | 1.000 | 1.000 | 1.000 | 1.000 |

## Spec gates vs measured values

Spec text is quoted as **targets** (§54: design targets, not results). The measured column shows only what this harness actually ran; ``pending`` means the required evidence does not exist in this run — a missing arm or an unavailable capability leaves the gate pending, never failed-over to a mock success (§58.09).

| gate | spec target (design target) | measured on this run | status |
|---|---|---|---|
| G1 — authorization | zero unauthorized disclosures (zero-tolerance) | disclosure_violations = 0 (baseline `verbatim_v3`) | measured |
| G3 — update automation | false-supersession upper 95% bound < 0.01; coverage ≥ 0.50 | on `verbatim_v3`: supersede applied 5/5 of true pairs, twins clean 4/4 (upper-95% ≈ 0.75 at this twin count); deep corpus 460 synthetic twins 0/460 flagged, upper-95% 0.0083 | measured (update-pair slice) |
| G5 — experience benefit | task-success gain ≥ 0.05 over no-memory, positive paired interval | delta 0.917 CI [0.760, 1.000] over 12 trials | measured (harness-level paired execution) |
| G7 — competitive claim | +0.05 macro over strongest baseline on licensed corpora | not estimable — no licensed competitor corpus in this harness | pending |
| G8 — learned controller | success lower interval > −0.01 with ≥10% token reduction, real paired executions | paired harness executed (deterministic controller only — no learned policy exists to evaluate) | pending |
| G9 — attack/privacy | benign retained ≥ 0.95; attack-induced rate ≤ 0.05; disclosure ≤ 0.02 | benign pass 1.000; poison exposure 0.000; disclosure 0 (baseline `verbatim_v3`) | measured |
| retrieval workload (§53) | — (workload measurement, not a gate) | `no_memory` recall 0.000, `naive_fts` recall 1.000, `vector_rag` recall 1.000, `verbatim_v2` recall 1.000, `verbatim_v3` recall 1.000 | measured |

## Run notes & capability fallbacks

- derivation roots probed first; source-level evidence ids used where no derivation edges exist (v2-ingested claims)
- conformance measures the v3 implementation once; it is arm-independent — baseline label tags gate attribution only
