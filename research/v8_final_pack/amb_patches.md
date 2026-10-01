# AMB harness patches — apply to a fresh vectorize-io/agent-memory-benchmark clone

The AMB clone lives in /tmp and does NOT survive restarts. These three patches make
it run verbatim on an OpenAI-compatible gateway whose models may (a) leak special
tokens into content, (b) lack structured outputs, (c) be flaky. Apply verbatim.

## 1. `src/memory_bench/llm/openai.py` — full replacement of `generate()`

```python
import json
import os
import re
import time

from .base import LLM, Schema

_MAX_RETRIES = 6
_RETRY_BASE_DELAY = 5


class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        schema_json = {
            "type": "object",
            "properties": schema.properties,
            "required": schema.required,
            "additionalProperties": False,
        }
        delay = _RETRY_BASE_DELAY
        last_exc = None
        # rf_mode: "json_schema" (strict) → "json_object" → "prompt" (no
        # response_format; JSON requested in the text). Downgraded lazily on
        # the gateway's 400 INVALID_REQUEST_BODY, cached on the instance.
        rf_mode = getattr(self, "_rf_mode", "json_schema")
        for attempt in range(_MAX_RETRIES):
            try:
                effective_prompt = prompt
                rf_kwargs: dict = {}
                if rf_mode == "json_schema":
                    rf_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": schema_json, "strict": True},
                    }
                elif rf_mode == "json_object":
                    rf_kwargs["response_format"] = {"type": "json_object"}
                else:
                    keys = ", ".join(schema.required or schema.properties.keys())
                    effective_prompt = (
                        prompt + f"\n\nRespond ONLY with a valid JSON object "
                        f"containing exactly these keys: {keys}. No markdown fences, no prose."
                    )
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": effective_prompt}],
                    **rf_kwargs,
                )
                text = response.choices[0].message.content or ""
                # Some OpenAI-compatible gateways leak eos/special tokens into content
                text = re.sub(r"<\|[^|]*\|>", "", text).strip()
                # Extract the first JSON object — tolerate leading prose and
                # trailing junk (repeated objects, markdown fences, etc.)
                start = text.find("{")
                if start == -1:
                    raise ValueError(f"no JSON object in response: {text[:200]!r}")
                try:
                    obj, _ = json.JSONDecoder().raw_decode(text[start:])
                except json.JSONDecodeError as e:
                    raise ValueError(f"unparseable JSON in response: {e}: {text[:300]!r}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"response is not a JSON object: {text[:200]!r}")
                return obj
            except Exception as e:
                last_exc = e
                msg = str(e)
                # Gateway/model doesn't accept the current response_format —
                # downgrade once and retry immediately (no sleep): strict
                # json_schema → json_object → plain-prompt JSON.
                if ("structured outputs" in msg or "INVALID_REQUEST_BODY" in msg
                        or ("400" in msg and "response_format" in msg)):
                    self._rf_mode = "json_object" if rf_mode == "json_schema" else "prompt"
                    rf_mode = self._rf_mode
                    continue
                # Retry transient rate-limit/parse failures — model output is
                # nondeterministic, so a malformed response often self-heals.
                if attempt < _MAX_RETRIES - 1 and (
                    "429" in msg or "rate" in msg.lower() or isinstance(e, ValueError)
                ):
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError(f"OpenAI request failed after {_MAX_RETRIES} retries: {last_exc}")
```

## 2. `src/memory_bench/runner.py` — non-fatal per-query errors

Replace `_process_one` (~line 169) so one bad query can't kill the run:

```python
        async def _process_one(q) -> QueryResult:
            for _attempt in range(4):
                try:
                    return await _process_one_attempt(q)
                except Exception as exc:
                    msg = str(exc)
                    if _attempt < 3:
                        transient = any(code in msg for code in ("502", "503", "529", "429", "overloaded", "quota"))
                        wait = 15 * (2 ** _attempt) if transient else 5
                        logger.warning("[query:%s] %s error (attempt %d/4), retrying in %ds: %s", q.id,
                                       "transient" if transient else "non-transient", _attempt + 1, wait, msg[:120])
                        await asyncio.sleep(wait)
                    else:
                        # One bad query must not kill the whole run — record it
                        # as a failed result and keep going.
                        logger.error("[query:%s] FAILED after retries: %s", q.id, msg[:200])
                        return QueryResult(
                            query_id=q.id, query=q.query, answer="", reasoning="",
                            context="", context_tokens=0, retrieve_time_ms=0.0,
                            gold_answers=q.gold_answers, correct=False,
                            judge_reason=f"query_error: {msg[:300]}", score=None,
                            meta=q.meta,
                            category_axes=dataset.get_result_categories(q.meta),
                        )
            raise RuntimeError("unreachable")
```

## 3. `src/memory_bench/memory/verbatim.py` — concurrency knob

In `VerbatimProvider` (from `verbatim_provider.py` in this dir), the class attr:

```python
    # Runner reads this attr for its asyncio.Semaphore — it bounds the
    # WHOLE per-query pipeline (retrieve + answer + judge), not just the
    # verbatim call. Retrieve holds self._lock around ~50ms of sqlite work,
    # so raising this parallelizes the LLM-bound portion safely.
    concurrency = int(os.environ.get("AMB_VERBATIM_CONCURRENCY", "4"))
```

## Env cheatsheet

| var | purpose |
|---|---|
| `GEMINI_API_KEY=dummy` | CLI gate demands it set; unused with openai LLMs |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | any OpenAI-compatible endpoint |
| `OMB_ANSWER_LLM=openai` `OMB_ANSWER_MODEL=<model>` | reader |
| `OMB_JUDGE_LLM=openai` `OMB_JUDGE_MODEL=<model>` | judge |
| `AMB_VERBATIM_REPO=/workspace/verbatim-new` | verbatim import path |
| `AMB_VERBATIM_CONCURRENCY=4` | per-shard query parallelism |
| `AMB_VERB_DEADLINE_MS / MAX_BYTES / TARGET_TOKENS` | verbatim recall knobs (4000/24000/4096 defaults) |

## Parallel shard runner (10 units, ~12 min for full locomo10)

```bash
for u in conv-26 conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50; do
  mkdir -p /tmp/amb-shards/$u
  (cd /tmp/amb-shards/$u && rm -rf outputs && \
   GEMINI_API_KEY=dummy OPENAI_BASE_URL=<ep>/v1 OPENAI_API_KEY=<key> \
   OMB_ANSWER_LLM=openai OMB_ANSWER_MODEL=<m> OMB_JUDGE_LLM=openai OMB_JUDGE_MODEL=<m> \
   AMB_VERBATIM_CONCURRENCY=4 nohup <clone>/.venv/bin/amb run \
     --dataset locomo --split locomo10 --memory verbatim --unit $u > run.log 2>&1 &)
done
# merge: concatenate "results" arrays from each /tmp/amb-shards/$u/outputs/locomo/verbatim/rag/locomo10.json
```
