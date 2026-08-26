#!/usr/bin/env bash
# Detect a usable CBLAS and print compiler flags for ai_gemm_cpu.
# Prints: "-DUSE_CBLAS -lopenblas" / "-DUSE_CBLAS -lcblas -lblas" /
#         "-DUSE_CBLAS -framework Accelerate" / "" (no BLAS -> naive fallback).
set -euo pipefail

probe='#include <cblas.h>
int main(){float a=0;cblas_sgemm(101,111,111,1,1,1,1.0f,&a,1,&a,1,0.0f,&a,1);return 0;}'

if [ "$(uname -s)" = "Darwin" ]; then
  echo "-DUSE_CBLAS -framework Accelerate"
  exit 0
fi

if printf '%s\n' "$probe" | gcc -x c - -lopenblas -o /dev/null 2>/dev/null; then
  echo "-DUSE_CBLAS -lopenblas"
elif printf '%s\n' "$probe" | gcc -x c - -lcblas -lblas -o /dev/null 2>/dev/null; then
  echo "-DUSE_CBLAS -lcblas -lblas"
else
  echo ""
fi
