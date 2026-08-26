#!/usr/bin/env bash
# perf wrapper for Experiment 1 CPU — one profiler call, no hidden loop.
#
# Usage:
#   perf_exp1_cpu.sh dependent
#   perf_exp1_cpu.sh independent [chains]      # default chains=1
#   perf_exp1_cpu.sh branch [predictable|random]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT_DIR/bin/exp1_cpu"

case="${1:-dependent}"
arg="${2:-}"

EVENTS="cycles,instructions"
ARGS=(--case "$case")

case "$case" in
  dependent)
    ARGS+=(--iterations 100000000)
    ;;
  independent)
    ARGS+=(--chains "${arg:-1}" --iterations 100000000)
    ;;
  branch)
    ARGS+=(--data "${arg:-predictable}" --iterations 10000000)
    EVENTS="cycles,instructions,branches,branch-misses"
    ;;
  *)
    echo "usage: $0 dependent|independent [chains]|branch [predictable|random]" >&2
    exit 1
    ;;
esac

ARGS+=(--warmup 0 --iters 1 --format human)

exec perf stat -e "$EVENTS" "$BIN" "${ARGS[@]}"
