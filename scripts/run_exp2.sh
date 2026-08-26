#!/usr/bin/env bash
# Experiment 2 — same math, different execution mapping (fixed thread count).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RES="$ROOT_DIR/results/exp2"
mkdir -p "$RES"

BIN_CPU_MEM="$BIN_DIR/exp2_cpu_memory"
BIN_GPU_MEM="$BIN_DIR/exp2_gpu_memory"
BIN_CPU_CTRL="$BIN_DIR/exp2_cpu_control"
BIN_GPU_CTRL="$BIN_DIR/exp2_gpu_control"

CPU_SIZE=100000000   # vector-add elements (CPU)
GPU_SIZE=16777216    # vector-add elements (GPU, 2^24)
GPU_THREADS=65536
CTRL_TASKS=1048576   # work/control tasks (2^20)

HDR_CPU_MEM="platform,mapping,threads,size,latency_ms,elements_per_s,effective_bandwidth_gbs,cycles,instructions,cache_references,cache_misses,llc_loads,llc_load_misses"
HDR_GPU_MEM="platform,stride,threads,size,latency_ms,elements_per_s,effective_bandwidth_gbs"
HDR_CPU_CTRL="platform,distribution,threads,tasks,heavy_iters,light_iters,latency_ms,throughput_tasks_s,max_thread_ms,min_thread_ms,avg_thread_ms,load_imbalance_ratio"
HDR_GPU_CTRL="platform,distribution,threads,tasks,heavy_iters,light_iters,latency_ms,throughput_tasks_s"

: > "$RES/cpu_memory_mapping.csv"
: > "$RES/gpu_memory_mapping.csv"
: > "$RES/cpu_work_mapping.csv"
: > "$RES/gpu_work_mapping.csv"

# --- cache events (LLC events are optional and hardware-dependent) ----------
EV_CACHE="cycles,instructions,cache-references,cache-misses"
MAP_CACHE="cycles:cycles,instructions:instructions,cache_references:cache-references,cache_misses:cache-misses"
if perf_has_event "LLC-loads" && perf_has_event "LLC-load-misses"; then
  EV_CACHE="$EV_CACHE,LLC-loads,LLC-load-misses"
  MAP_CACHE="$MAP_CACHE,llc_loads:LLC-loads,llc_load_misses:LLC-load-misses"
fi

echo "== Experiment 2A: CPU memory mapping =="
[ -x "$BIN_CPU_MEM" ] || { echo "ERROR: $BIN_CPU_MEM not built"; exit 1; }
for m in block cyclic; do
  run_cpu "$RES/cpu_memory_mapping.csv" "$HDR_CPU_MEM" "$MAP_CACHE" "" "$EV_CACHE" \
    "$BIN_CPU_MEM" --mapping "$m" --size "$CPU_SIZE"
done

echo "== Experiment 2A: GPU memory mapping =="
if [ -x "$BIN_GPU_MEM" ]; then
  for s in 1 2 4 8 16 32; do
    "$BIN_GPU_MEM" --stride "$s" --threads "$GPU_THREADS" --size "$GPU_SIZE" \
      --csv-file "$RES/gpu_memory_mapping.csv" >/dev/null
  done
else
  echo "[skip] GPU memory binary not built"
fi

echo "== Experiment 2B: CPU work/control mapping =="
[ -x "$BIN_CPU_CTRL" ] || { echo "ERROR: $BIN_CPU_CTRL not built"; exit 1; }
for d in grouped mixed; do
  run_cpu "$RES/cpu_work_mapping.csv" "$HDR_CPU_CTRL" "" "" "" \
    "$BIN_CPU_CTRL" --distribution "$d" --tasks "$CTRL_TASKS"
done

echo "== Experiment 2B: GPU work/control mapping =="
if [ -x "$BIN_GPU_CTRL" ]; then
  for d in grouped mixed; do
    "$BIN_GPU_CTRL" --distribution "$d" --threads "$GPU_THREADS" --tasks "$CTRL_TASKS" \
      --csv-file "$RES/gpu_work_mapping.csv" >/dev/null
  done
else
  echo "[skip] GPU control binary not built"
fi

# --- optional NVIDIA Nsight Compute raw reports (best effort) ---------------
if command -v ncu >/dev/null 2>&1 && [ -x "$BIN_GPU_MEM" ]; then
  echo "== Optional: Nsight Compute raw reports =="
  mkdir -p "$RES/ncu"
  ncu --csv -o "$RES/ncu/gpu_memory_stride1" "$BIN_GPU_MEM" --stride 1 --threads "$GPU_THREADS" --size "$GPU_SIZE" --iters 1 >/dev/null 2>&1 || true
  ncu --csv -o "$RES/ncu/gpu_memory_stride32" "$BIN_GPU_MEM" --stride 32 --threads "$GPU_THREADS" --size "$GPU_SIZE" --iters 1 >/dev/null 2>&1 || true
  ncu --csv -o "$RES/ncu/gpu_control_grouped" "$BIN_GPU_CTRL" --distribution grouped --threads "$GPU_THREADS" --tasks "$CTRL_TASKS" --iters 1 >/dev/null 2>&1 || true
  ncu --csv -o "$RES/ncu/gpu_control_mixed" "$BIN_GPU_CTRL" --distribution mixed --threads "$GPU_THREADS" --tasks "$CTRL_TASKS" --iters 1 >/dev/null 2>&1 || true
fi

echo "Experiment 2 done -> $RES"
