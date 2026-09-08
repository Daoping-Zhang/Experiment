#!/bin/bash
# compile and run every native MPI collective example with NP ranks
set -e
cd "$(dirname "$0")"
make
NP="${1:-4}"
for ex in bcast_example scatter_example gather_example \
          reduce_example allreduce_example barrier_example; do
    echo "===== $ex (np=$NP) ====="
    mpirun -n "$NP" "./$ex"
done
