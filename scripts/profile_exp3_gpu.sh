#!/usr/bin/env bash
# GPU scaling profiler — extracts occupancy / SM utilization metrics.
# Usage: profile_exp3_gpu.sh <threads>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT_DIR/bin/exp3_gpu"

threads="${1:-65536}"

exec python3 "$SCRIPT_DIR/ncu_profile.py" \
  --preset occupancy \
  --label "GPU scaling evidence (threads $threads)" \
  -- "$BIN" --threads "$threads" --size 1000000 --compute-iterations 500 --iters 1
