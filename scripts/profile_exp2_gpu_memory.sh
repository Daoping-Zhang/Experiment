#!/usr/bin/env bash
# GPU memory-mapping profiler — extracts coalescing / transaction metrics only.
# Usage: profile_exp2_gpu_memory.sh <stride>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT_DIR/bin/exp2_gpu_memory"

stride="${1:-1}"

exec python3 "$SCRIPT_DIR/ncu_profile.py" \
  --preset memory \
  --label "GPU memory evidence (stride $stride)" \
  -- "$BIN" --stride "$stride" --threads 65536 --size 16777216 --iters 1
