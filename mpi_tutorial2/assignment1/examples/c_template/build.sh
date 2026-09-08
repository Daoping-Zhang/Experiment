#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mpicc -O2 \
    "$SCRIPT_DIR/solution.c" \
    -o "$SCRIPT_DIR/solution"
