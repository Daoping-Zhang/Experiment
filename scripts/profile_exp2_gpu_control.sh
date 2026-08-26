#!/usr/bin/env bash
# GPU work/control profiler — extracts divergence / lane / branch metrics.
# Usage: profile_exp2_gpu_control.sh <grouped|mixed>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT_DIR/bin/exp2_gpu_control"

dist="${1:-grouped}"

exec python3 "$SCRIPT_DIR/ncu_profile.py" \
  --patterns "branch,divergent,warp,lane,active,uniform,inst_executed,gpu__time_duration" \
  --label "GPU work/control profile ($dist)" \
  -- "$BIN" --distribution "$dist" --threads 65536 --tasks 1048576 --iters 1
