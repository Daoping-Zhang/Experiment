#!/usr/bin/env bash
# Experiment 3 — few strong CPU threads vs massive lightweight GPU threads.
# Sweeps thread count only; workload (N, K) and mapping stay fixed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RES="$ROOT_DIR/results/exp3"
mkdir -p "$RES"

BIN_CPU="$BIN_DIR/exp3_cpu"
BIN_GPU="$BIN_DIR/exp3_gpu"

SIZE=1000000          # elements
K=500                 # compute iterations per element
HDR="platform,threads,size,k,latency_ms,throughput_elem_s,gflops"

: > "$RES/cpu_scaling.csv"
: > "$RES/gpu_scaling.csv"

# CPU thread sweep: 1,2,4,...,logical hardware threads (inclusive).
HW=$(cpu_threads)
CPU_THREADS="1"
t=1
while [ $((t * 2)) -le "$HW" ]; do
  t=$((t * 2))
  CPU_THREADS="$CPU_THREADS $t"
done
case " $CPU_THREADS " in
  *" $HW "*) ;;
  *) CPU_THREADS="$CPU_THREADS $HW" ;;
esac

# GPU thread sweep: 1,2,4,...,min(N, 1M).
MAX=$((SIZE < 1048576 ? SIZE : 1048576))
GPU_THREADS="1"
t=1
while [ $((t * 2)) -le "$MAX" ]; do
  t=$((t * 2))
  GPU_THREADS="$GPU_THREADS $t"
done
case " $GPU_THREADS " in
  *" $MAX "*) ;;
  *) GPU_THREADS="$GPU_THREADS $MAX" ;;
esac

echo "== Experiment 3: CPU scaling (threads: $CPU_THREADS) =="
[ -x "$BIN_CPU" ] || { echo "ERROR: $BIN_CPU not built"; exit 1; }
for th in $CPU_THREADS; do
  "$BIN_CPU" --threads "$th" --size "$SIZE" --compute-iterations "$K" \
    --csv-file "$RES/cpu_scaling.csv" >/dev/null
done

echo "== Experiment 3: GPU scaling (threads: $GPU_THREADS) =="
if [ -x "$BIN_GPU" ]; then
  for th in $GPU_THREADS; do
    "$BIN_GPU" --threads "$th" --size "$SIZE" --compute-iterations "$K" \
      --csv-file "$RES/gpu_scaling.csv" >/dev/null
  done
else
  echo "[skip] GPU scaling binary not built"
fi

# Optional Nsight Compute occupancy/SM-utilization raw reports (best effort).
if command -v ncu >/dev/null 2>&1 && [ -x "$BIN_GPU" ]; then
  echo "== Optional: Nsight Compute raw reports =="
  mkdir -p "$RES/ncu"
  for th in 32 1024 65536; do
    ncu --csv -o "$RES/ncu/gpu_scaling_t${th}" "$BIN_GPU" --threads "$th" --size "$SIZE" --compute-iterations "$K" --iters 1 >/dev/null 2>&1 || true
  done
fi

echo "Experiment 3 done -> $RES"
