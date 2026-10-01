#!/usr/bin/env bash
# AMB harness setup — clones agent-memory-benchmark at the pinned commit
# and installs the Verbatim provider + the three harness patches.
#
# Usage:
#   eval/amb/harness/setup_harness.sh <target-dir> <verbatim-repo-path>
#
#   <target-dir>         where the AMB clone lands (created if absent)
#   <verbatim-repo-path> path to THIS repo's checkout (becomes
#                        AMB_VERBATIM_REPO — verbatim + eval.amb must be
#                        importable from it)
#
# Afterwards run the benchmark via run_verbatim_locomo10.sh (same dir).
set -euo pipefail

AMB_PIN="03c1d0f"   # verified upstream commit (fix(sdebench) #47)
TARGET="${1:?usage: setup_harness.sh <target-dir> <verbatim-repo>}"
VERBATIM_REPO="$(cd "${2:?usage: setup_harness.sh <target-dir> <verbatim-repo>}" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$TARGET/.git" ]; then
    git clone https://github.com/vectorize-io/agent-memory-benchmark "$TARGET"
fi
cd "$TARGET"
git fetch --quiet origin "$AMB_PIN" 2>/dev/null || true
git checkout --quiet "$AMB_PIN"

# --- install the provider adapter + patches ------------------------------
SRC="src/memory_bench"
cp "$HERE/verbatim.py"        "$SRC/memory/verbatim.py"
cp "$HERE/memory__init__.py"  "$SRC/memory/__init__.py"
cp "$HERE/openai.py"          "$SRC/llm/openai.py"
cp "$HERE/runner.py"          "$SRC/runner.py"
# MSC-MemFuse + MemBench adapters (not upstream): files + registry entries
cp "$HERE/msc_memfuse.py"     "$SRC/dataset/msc_memfuse.py"
cp "$HERE/membench.py"        "$SRC/dataset/membench.py"
cp "$HERE/dataset__init__.py" "$SRC/dataset/__init__.py"

# --- smoke: provider registers and imports -------------------------------
AMB_VERBATIM_REPO="$VERBATIM_REPO" uv run python - <<'PY'
import sys, os
sys.path.insert(0, os.environ["AMB_VERBATIM_REPO"])
from memory_bench.memory import get_memory_provider
p = get_memory_provider("verbatim")
print("verbatim provider:", type(p).__name__,
      "| impl:", type(p._impl).__module__)
PY

echo "harness ready at $TARGET (pin $AMB_PIN)"
echo "next: AMB_VERBATIM_REPO=$VERBATIM_REPO $HERE/run_verbatim_locomo10.sh $TARGET"
