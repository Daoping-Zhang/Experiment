#!/usr/bin/env bash
# AI validation — GEMM (optimized CPU BLAS vs cuBLAS).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RES="$ROOT_DIR/results/gemm"
mkdir -p "$RES"

BIN_CPU="$BIN_DIR/ai_gemm_cpu"
BIN_GPU="$BIN_DIR/ai_gemm_gpu"
SIZES="${SIZES:-128,256,512,1024,2048,4096}"

: > "$RES/cpu_gemm.csv"
: > "$RES/gpu_gemm.csv"

echo "== GEMM: CPU =="
[ -x "$BIN_CPU" ] || { echo "ERROR: $BIN_CPU not built"; exit 1; }
"$BIN_CPU" --sizes "$SIZES" --csv-file "$RES/cpu_gemm.csv" >/dev/null

echo "== GEMM: GPU =="
if [ -x "$BIN_GPU" ]; then
  "$BIN_GPU" --sizes "$SIZES" --csv-file "$RES/gpu_gemm.csv" >/dev/null
else
  echo "[skip] GPU GEMM binary not built"
fi

echo "GEMM done -> $RES"
