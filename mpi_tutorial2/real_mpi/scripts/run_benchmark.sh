#!/bin/bash
# Real MPI benchmark sweep -> results.csv (ranks,bytes,time_ms).
# Ranks default 1 2 4 8 (machines with fewer cores: export RANKS="1 2 4").
# Sizes in BYTES (int32): 16 B .. 16 MB.
set -e
cd "$(dirname "$0")/.."
make allreduce_benchmark
RANKS="${RANKS:-1 2 4 8}"
SIZES="${SIZES:-16 1024 16384 262144 4194304 16777216}"
OUT="${1:-results.csv}"
echo "ranks,bytes,time_ms" > "$OUT"
for NP in $RANKS; do
  for B in $SIZES; do
    ELEM=$((B / 4))
    out=$(mpirun -n "$NP" ./allreduce_benchmark "$ELEM")
    ms=$(echo "$out" | sed -n 's/.*: \([0-9.]*\) ms$/\1/p')
    echo "$NP,$B,$ms" >> "$OUT"
    echo "ranks=$NP bytes=$B -> ${ms} ms"
  done
done
echo "results written to $OUT"
