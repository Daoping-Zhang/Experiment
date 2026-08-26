#!/usr/bin/env bash
# Experiment 1 — strong CPU thread vs lightweight GPU thread (single thread).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RES="$ROOT_DIR/results/exp1"
mkdir -p "$RES"

BIN_CPU="$BIN_DIR/exp1_cpu"
BIN_GPU="$BIN_DIR/exp1_gpu"

ITER=100000000        # chain iterations (dependent / independent)
ITER_BRANCH=10000000  # elements for the branch workload

HDR="platform,workload,variant,threads,iterations,latency_ms,throughput_ops_s,gflops,cycles,instructions,ipc,branches,branch_misses,branch_miss_rate,kernel_latency_ms,end_to_end_ms"

MAP_CHAIN="cycles:cycles,instructions:instructions"
DERIVE_IPC="ipc:instructions/cycles"
MAP_BRANCH="cycles:cycles,instructions:instructions,branches:branches,branch_misses:branch-misses"
DERIVE_BRANCH="ipc:instructions/cycles,branch_miss_rate:branch_misses/branches"
EV_CHAIN="cycles,instructions"
EV_BRANCH="cycles,instructions,branches,branch-misses"

: > "$RES/dependent.csv"
: > "$RES/independent.csv"
: > "$RES/branch.csv"

echo "== Experiment 1: CPU single thread =="
[ -x "$BIN_CPU" ] || { echo "ERROR: $BIN_CPU not built (run 'make' first)"; exit 1; }

run_cpu "$RES/dependent.csv" "$HDR" "$MAP_CHAIN" "$DERIVE_IPC" "$EV_CHAIN" \
  "$BIN_CPU" --case dependent --iterations "$ITER"

for c in 1 2 4 8; do
  run_cpu "$RES/independent.csv" "$HDR" "$MAP_CHAIN" "$DERIVE_IPC" "$EV_CHAIN" \
    "$BIN_CPU" --case independent --chains "$c" --iterations "$ITER"
done

for d in predictable random; do
  run_cpu "$RES/branch.csv" "$HDR" "$MAP_BRANCH" "$DERIVE_BRANCH" "$EV_BRANCH" \
    "$BIN_CPU" --case branch --data "$d" --iterations "$ITER_BRANCH"
done

echo "== Experiment 1: GPU single thread =="
if [ -x "$BIN_GPU" ]; then
  "$BIN_GPU" --case dependent --iterations "$ITER" --csv-file "$RES/dependent.csv" >/dev/null
  for c in 1 2 4 8; do
    "$BIN_GPU" --case independent --chains "$c" --iterations "$ITER" --csv-file "$RES/independent.csv" >/dev/null
  done
  for d in predictable random; do
    "$BIN_GPU" --case branch --data "$d" --iterations "$ITER_BRANCH" --csv-file "$RES/branch.csv" >/dev/null
  done
else
  echo "[skip] GPU binary not built (no CUDA toolchain); GPU rows omitted"
fi

echo "Experiment 1 done -> $RES"
