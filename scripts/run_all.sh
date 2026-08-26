#!/usr/bin/env bash
# Run the whole Lecture 01 tutorial end-to-end:
#   system info -> exp1 -> exp2 -> exp3 -> GEMM -> figures -> summary.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================ Lecture 01 Tutorial ================"
echo "Working dir: $ROOT_DIR"

"$SCRIPT_DIR/collect_system_info.sh"
"$SCRIPT_DIR/run_exp1.sh"
"$SCRIPT_DIR/run_exp2.sh"
"$SCRIPT_DIR/run_exp3.sh"
"$SCRIPT_DIR/run_gemm.sh"

echo "================ Plotting ================"
python3 "$SCRIPT_DIR/plot_results.py"
python3 "$SCRIPT_DIR/make_summary.py"

echo "================ Done ================"
echo "Results:  $ROOT_DIR/results/"
echo "Figures:  $ROOT_DIR/figures/"
echo "Summary:  $ROOT_DIR/results/summary.md"
