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
        self._client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=float(os.environ.get("AMB_OPENAI_TIMEOUT", "240")),
        )
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
                    temperature=float(os.environ.get("AMB_OPENAI_TEMPERATURE", "0.0")),
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
