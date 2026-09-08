#!/bin/bash
# Real MPI server demo: compile + run reduce_allreduce_demo.
# Expects mpicc/mpirun (set PATH or export MPIRUN/MPICC).
set -e
cd "$(dirname "$0")/.."
make reduce_allreduce_demo
NP="${1:-4}"
mpirun -n "$NP" ./reduce_allreduce_demo
