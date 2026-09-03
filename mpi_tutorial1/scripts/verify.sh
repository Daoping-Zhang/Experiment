#!/usr/bin/env bash
# Automatic acceptance check for MPI Tutorial 1.
#
#   ./scripts/verify.sh
#
# Covers: build, hello_mpi with 1/2/4 processes, send_recv, ping_pong,
# and the "too few processes" guard. Exits 0 only if every test passes.
set -u

cd "$(dirname "$0")/.."

RUNNER="$(command -v mpiexec || command -v mpirun || true)"
if [ -z "$RUNNER" ]; then
    echo "[FAIL] mpiexec/mpirun not found"
    exit 1
fi

# OpenMPI as root (containers) needs --allow-run-as-root.
OPTS=""
if [ "$(id -u)" = "0" ] && "$RUNNER" --version 2>&1 | grep -qiE "open ?mpi|openrte"; then
    OPTS="--allow-run-as-root"
fi

# Portable timeout: run a command, kill it if it exceeds N seconds.
# (Polling with kill -0 — no background killer process, so no "Terminated"
#  noise on stderr.)
run_cap() {  # run_cap <outfile> <seconds> <cmd...>
    local out="$1" secs="$2"
    shift 2
    "$@" > "$out" 2>&1 &
    local pid=$!
    local i=0
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt "$secs" ]; do
        sleep 1
        i=$((i + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null   # exceeded the timeout
        wait "$pid" 2>/dev/null
        return 124
    fi
    wait "$pid" 2>/dev/null
    return $?
}

TMP="$(mktemp -d 2>/dev/null || echo /tmp/mpi_t1.$$)"
mkdir -p "$TMP"

pass=0
fail=0
note_pass() { echo "[PASS] $1"; pass=$((pass + 1)); }
note_fail() { echo "[FAIL] $1"; fail=$((fail + 1)); }

echo "========================================"
echo "MPI Tutorial 1 Verification"
echo "========================================"
echo

# ---------------------------------------------------------------------------
# Test 1 — Build
# ---------------------------------------------------------------------------
echo "---- Test 1: build ----"
if make clean >/dev/null 2>&1 && make > "$TMP/build.log" 2>&1; then
    note_pass "build"
else
    note_fail "build"
    echo "----- make log -----"
    tail -30 "$TMP/build.log"
fi

# ---------------------------------------------------------------------------
# Test 2 — Hello MPI
# ---------------------------------------------------------------------------
echo "---- Test 2: hello_mpi ----"

check_hello() {  # check_hello <n> <expected ranks...>
    local n="$1"; shift
    local out="$TMP/hello_$n.log"
    if run_cap "$out" 60 "$RUNNER" $OPTS -n "$n" ./bin/hello_mpi; then
        local ok=1
        for r in "$@"; do grep -q "rank $r " "$out" || ok=0; done
        grep -q "of $n" "$out" || ok=0
        if [ "$ok" = "1" ]; then
            note_pass "hello n=$n"
        else
            note_fail "hello n=$n (missing expected output)"
            echo "----- output -----"; cat "$out"
        fi
    else
        note_fail "hello n=$n (exit code $?)"
        echo "----- output -----"; cat "$out"
    fi
}

check_hello 1 "0"
check_hello 2 "0" "1"
check_hello 4 "0" "1" "2" "3"

# ---------------------------------------------------------------------------
# Test 3 — Send / Recv
# ---------------------------------------------------------------------------
echo "---- Test 3: send_recv (n=2) ----"
out="$TMP/send_recv.log"
if run_cap "$out" 60 "$RUNNER" $OPTS -n 2 ./bin/send_recv; then
    if grep -q "42" "$out"; then
        note_pass "send_recv"
    else
        note_fail "send_recv (value 42 not found)"
        echo "----- output -----"; cat "$out"
    fi
else
    note_fail "send_recv (exit code $?)"
    echo "----- output -----"; cat "$out"
fi

# ---------------------------------------------------------------------------
# Test 4 — Ping Pong
# ---------------------------------------------------------------------------
echo "---- Test 4: ping_pong (n=2) ----"
out="$TMP/ping_pong.log"
if run_cap "$out" 60 "$RUNNER" $OPTS -n 2 ./bin/ping_pong; then
    if grep -q "43" "$out"; then
        note_pass "ping_pong"
    else
        note_fail "ping_pong (value 43 not found)"
        echo "----- output -----"; cat "$out"
    fi
else
    note_fail "ping_pong (exit code $?)"
    echo "----- output -----"; cat "$out"
fi

# ---------------------------------------------------------------------------
# Test 5 — Invalid process count (n=1) must exit cleanly, no crash/deadlock
# ---------------------------------------------------------------------------
echo "---- Test 5: invalid process count ----"
out="$TMP/invalid.log"
if run_cap "$out" 60 "$RUNNER" $OPTS -n 1 ./bin/send_recv; then
    if grep -q "requires at least 2" "$out"; then
        note_pass "invalid process count"
    else
        note_fail "invalid process count (no friendly message)"
        echo "----- output -----"; cat "$out"
    fi
else
    note_fail "invalid process count (exit code $?)"
    echo "----- output -----"; cat "$out"
fi

rm -rf "$TMP"

echo
echo "========================================"
echo "Results: $pass passed, $fail failed"
echo "========================================"
if [ "$fail" = "0" ]; then
    echo "ALL TESTS PASSED"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi
