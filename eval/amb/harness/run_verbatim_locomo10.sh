#!/usr/bin/env bash
# Verbatim × AMB locomo10 — parallel shard runner (SPEC_V8_5 G85-3).
#
# Usage:
#   export OPENAI_BASE_URL=<endpoint>/v1
#   export OPENAI_API_KEY=<key>
#   export OMB_ANSWER_MODEL=<reader model>   # e.g. inclusionai/ling-3.0-flash-fin:free
#   export OMB_JUDGE_MODEL=<judge model>     # same as reader unless overriding
#   export AMB_VERBATIM_REPO=/path/to/memorysys
#   bash run_verbatim_locomo10.sh [AMB_DIR] [OUT_DIR]
#
#   AMB_DIR  — the agent-memory-benchmark checkout prepared by
#              setup_harness.sh (default: $AMB_DIR or ./agent-memory-benchmark)
#   OUT_DIR  — shard workspace (default: $AMB_OUT or /tmp/amb-shards)
#
# Runs the 10 locomo10 units as parallel shards (one process each), then
# merge with: python merge_shards.py [OUT_DIR]
set -uo pipefail

AMB="${1:-${AMB_DIR:-./agent-memory-benchmark}}"
OUT="${2:-${AMB_OUT:-/tmp/amb-shards}}"
UNITS="conv-26 conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50"

: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL to the OpenAI-compatible gateway}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
: "${OMB_ANSWER_MODEL:?set OMB_ANSWER_MODEL (reader model id)}"
: "${AMB_VERBATIM_REPO:?set AMB_VERBATIM_REPO to the verbatim repo checkout}"
OMB_JUDGE_MODEL="${OMB_JUDGE_MODEL:-$OMB_ANSWER_MODEL}"

export GEMINI_API_KEY=dummy          # CLI gate requires it set; unused
export OMB_ANSWER_LLM=openai
export OMB_JUDGE_LLM=openai
export AMB_VERBATIM_CONCURRENCY="${AMB_VERBATIM_CONCURRENCY:-4}"
export VERBATIM_AMB_TOKEN_BUDGET="${VERBATIM_AMB_TOKEN_BUDGET:-4500}"

mkdir -p "$OUT"
for u in $UNITS; do
  mkdir -p "$OUT/$u"
  (cd "$OUT/$u" && rm -rf outputs && \
   cd "$AMB" && nohup uv run amb run \
     --dataset locomo --split locomo10 --memory verbatim --unit "$u" \
     --output-dir "$OUT/$u/outputs" \
     > "$OUT/$u/run.log" 2>&1 &)
  echo "launched $u (log: $OUT/$u/run.log)"
done
echo "all shards launched; watch: tail -f $OUT/*/run.log"
echo "merge when done: python $(dirname "$0")/merge_shards.py $OUT"
