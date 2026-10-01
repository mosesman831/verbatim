# Verbatim workspace context

Released source for the Verbatim memory engine (`verbatim-memory`).

- Spec corpus, requirements ledgers and the development log live in the private
  repo `mosesman831/verbatim-specs` (SPEC v1-v8.5, REQUIREMENTS, THREAT_MODEL).
  They are not duplicated here.

## Layout

- `verbatim/` - engine (storage, retrieval, evidence, governance, privacy, service, MCP)
- `tests/` - test suite (`python -m pytest tests/`)
- `eval/` - internal eval harnesses and measured reports
- `docs/*_contracts.md` - frozen worker/surface interfaces per version
- `clients/ts/memory-client` - TypeScript client
- `RELEASE_MANIFEST_V4.json` - declared capability and gate status plus
  `known_limitations`; read it before making any claim about this system

## Claims policy

Capability and gate status is declared with evidence, never asserted. Failing
measurements stay visible in the manifest rather than being quietly dropped.
No leadership or comparative-quality claims are made.

## Development

```bash
python -m pytest tests/
python -m eval.v3.run
```

Benchmark environment recipes are in `eval/amb/RECOMMENDED_ENV.md`. Endpoint and
credential material is operator-supplied via environment variables - never
committed to this repository.