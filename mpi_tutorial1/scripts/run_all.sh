#!/usr/bin/env bash
# Run every Tutorial 1 demo, one at a time, with clear section headers.
# Stdout is NOT hidden: the class watches these outputs live.
#
# Usage:  ./scripts/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Binaries must exist first.
if [ ! -x bin/hello_mpi ] || [ ! -x bin/send_recv ] || [ ! -x bin/ping_pong ]; then
    echo "binaries missing — run 'make' first" >&2
    exit 1
fi

# Runner: prefer mpiexec, fall back to mpirun.
RUNNER="$(command -v mpiexec || command -v mpirun || true)"
[ -n "$RUNNER" ] || { echo "mpiexec/mpirun not found" >&2; exit 1; }

# OpenMPI specifics: as root it needs --allow-run-as-root, and on small hosts
# it may need --oversubscribe to launch more processes than physical cores.
OPTS=""
if "$RUNNER" --version 2>&1 | grep -qiE "open ?mpi|openrte"; then
    OPTS="--oversubscribe"
    [ "$(id -u)" = "0" ] && OPTS="--allow-run-as-root $OPTS"
fi

heading() {
    echo
    echo "========================================"
    echo "$1"
    echo "========================================"
}

heading "Demo 1: Hello MPI with 1 process"
"$RUNNER" $OPTS -n 1 ./bin/hello_mpi

heading "Demo 1: Hello MPI with 2 processes"
"$RUNNER" $OPTS -n 2 ./bin/hello_mpi

heading "Demo 1: Hello MPI with 4 processes"
"$RUNNER" $OPTS -n 4 ./bin/hello_mpi

heading "Demo 2: Send / Recv (2 processes)"
"$RUNNER" $OPTS -n 2 ./bin/send_recv

heading "Demo 3 / Exercise: Ping Pong (2 processes)"
"$RUNNER" $OPTS -n 2 ./bin/ping_pong

echo
echo "All demos done."
