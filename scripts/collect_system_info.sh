#!/usr/bin/env bash
# Collect hardware / software environment info into results/system_info.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="$ROOT_DIR/results/system_info.txt"
mkdir -p "$ROOT_DIR/results"

{
  echo "=== System Info ==="
  echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo

  echo "--- OS ---"
  uname -a
  if [ -f /etc/os-release ]; then
    echo "--- /etc/os-release ---"
    cat /etc/os-release
  fi
  echo

  echo "--- CPU ---"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu
  else
    echo "model:          $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "physical cores: $(sysctl -n hw.physicalcpu 2>/dev/null || echo unknown)"
    echo "logical cores:  $(sysctl -n hw.logicalcpu 2>/dev/null || echo unknown)"
  fi
  echo

  echo "--- GPU ---"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "no NVIDIA GPU (nvidia-smi not found)"
  fi
  echo

  echo "--- CUDA ---"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
  else
    echo "nvcc not found"
  fi
  echo

  echo "--- Compilers ---"
  echo "gcc: $(gcc --version 2>/dev/null | head -1 || echo 'not found')"
  echo "g++: $(g++ --version 2>/dev/null | head -1 || echo 'not found')"
  echo

  echo "--- Python ---"
  echo "python3: $(python3 --version 2>/dev/null || echo 'not found')"
  python3 -c 'import pandas, matplotlib; print("pandas", pandas.__version__, "matplotlib", matplotlib.__version__)' 2>/dev/null \
    || echo "pandas / matplotlib NOT installed (pip install pandas matplotlib)"
  echo
} > "$OUT"

echo "system info -> $OUT"
