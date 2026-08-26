#!/usr/bin/env bash
# Shared helpers for the Lecture 01 tutorial run scripts.
# Source this file (it is not meant to be executed directly).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$ROOT_DIR/bin"
PERF_MERGE="$SCRIPT_DIR/perf_merge.py"

# Detect perf. Require it to actually work (some kernels restrict counters).
HAVE_PERF=0
if command -v perf >/dev/null 2>&1 && perf stat -e cycles true >/dev/null 2>&1; then
  HAVE_PERF=1
fi

perf_has_event() {
  [ "$HAVE_PERF" = "1" ] && perf stat -e "$1" true >/dev/null 2>&1
}

# number of logical CPU threads
cpu_threads() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.logicalcpu >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu
  else
    echo 1
  fi
}

# run_cpu <csv> <header> <map> <derive> <events> <binary> [args...]
#
# Runs <binary> once (optionally under `perf stat`) and appends exactly one CSV
# row to <csv>. Hardware counters are merged in by perf_merge.py when perf is
# available; otherwise hardware columns stay "NA".
run_cpu() {
  local csv="$1" header="$2" map="$3" derive="$4" events="$5"
  shift 5

  if [ "$HAVE_PERF" = "1" ] && [ -n "$events" ]; then
    local sidecar fmt=csv outfile
    sidecar="$(mktemp)"
    outfile="$(mktemp)"
    if perf stat -j -e "$events" true >/dev/null 2>&1; then
      fmt=json
    fi
    # Capture the binary's CSV row (stdout) to a file; let its stderr (the
    # [check]/summary lines) pass through so a FAILED correctness check is
    # visible before `set -e` aborts.
    if [ "$fmt" = json ]; then
      perf stat -j -e "$events" -o "$sidecar" "$@" > "$outfile"
    else
      perf stat -x, -e "$events" -o "$sidecar" "$@" > "$outfile"
    fi
    local row
    row="$(cat "$outfile")"
    python3 "$PERF_MERGE" --row "$row" --sidecar "$sidecar" --csv "$csv" \
      --header "$header" --map "$map" --derive "$derive" --format "$fmt"
    rm -f "$sidecar" "$outfile"
  else
    "$@" --csv-file "$csv" >/dev/null
  fi
}
