#!/usr/bin/env bash
# CPU memory-mapping profiler (one perf call, no hidden loop).
# Usage: profile_exp2_cpu_memory.sh <block|cyclic> [threads]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT_DIR/bin/exp2_cpu_memory"

mapping="${1:-block}"
threads="${2:-8}"

exec perf stat -e cycles,instructions,cache-references,cache-misses \
  "$BIN" --mapping "$mapping" --threads "$threads" --size 100000000 \
  --warmup 0 --iters 1 --format human
