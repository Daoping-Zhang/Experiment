#!/usr/bin/env bash
# Check the MPI environment before class starts.
#
# Checks: mpicc, mpiexec/mpirun, versions, and a minimal compile + run.
# Exit code 0 = ready, non-zero = something is missing.
#
# Works with OpenMPI or MPICH; as root in a container, OpenMPI needs
# --allow-run-as-root, which is handled automatically here.
set -u

fails=0
say_pass() { echo "[PASS] $1"; }
say_fail() { echo "[FAIL] $1"; fails=1; }

echo "========================================"
echo "MPI Tutorial 1 — environment check"
echo "========================================"
echo

# --- 1. mpicc ---------------------------------------------------------------
if command -v mpicc >/dev/null 2>&1; then
    say_pass "mpicc found: $(command -v mpicc)"
    echo "       $(mpicc --version 2>&1 | head -1)"
else
    say_fail "mpicc not found"
fi

# --- 2. mpiexec / mpirun (prefer mpiexec) ------------------------------------
RUNNER=""
for c in mpiexec mpirun; do
    if command -v "$c" >/dev/null 2>&1; then
        RUNNER="$c"
        say_pass "$c found: $(command -v "$c")"
        echo "       $("$c" --version 2>&1 | head -1)"
        break
    fi
done
[ -n "$RUNNER" ] || say_fail "neither mpiexec nor mpirun found"

# --- 3. compile a minimal MPI program ----------------------------------------
TMPD="$(mktemp -d 2>/dev/null || echo /tmp/mpi_check.$$)"
mkdir -p "$TMPD"
cat > "$TMPD/check.c" <<'EOF'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) printf("ok\n");
    MPI_Finalize();
    return 0;
}
EOF

if mpicc -O2 -Wall "$TMPD/check.c" -o "$TMPD/check" 2>/dev/null; then
    say_pass "MPI compile test"
else
    say_fail "MPI compile test failed"
fi

# --- 4. run the minimal program ----------------------------------------------
if [ -n "$RUNNER" ]; then
    OPTS=""
    if [ "$(id -u)" = "0" ] && "$RUNNER" --version 2>&1 | grep -qiE "open ?mpi|openrte"; then
        OPTS="--allow-run-as-root"
    fi
    if "$RUNNER" $OPTS -n 2 "$TMPD/check" >/dev/null 2>&1; then
        say_pass "MPI run test ($RUNNER -n 2)"
    else
        say_fail "MPI run test failed"
    fi
fi

rm -rf "$TMPD"

echo
if [ "$fails" = "0" ]; then
    echo "Environment OK."
    exit 0
else
    echo "Environment check FAILED."
    exit 1
fi
