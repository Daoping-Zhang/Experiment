"""python_template/solution.py — Assignment 1 SKELETON (fill in the TODOs).

Run by the grader:  mpirun -n P ./run.sh <input.txt>
Allowed: direct MPI_Reduce / MPI_Allreduce, or your own Send/Recv design.
"""
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# TODO 1 — read THIS rank's local vector from sys.argv[1]
#   Input format: first line "P N", then P lines (line r = vector of rank r).
#   local = [...]
local = None

# TODO 2 — Reduce (SUM -> root 0). Only rank 0 prints:
#   "REDUCE rank=0: v0 v1 ... vN-1"
reduced = None
# reduce_out = ...  (implement)

if rank == 0:
    print("REDUCE rank=0: %s" % " ".join(map(str, reduced)))

# TODO 3 — AllReduce (SUM -> every rank). Every rank prints:
#   "ALLREDUCE rank=<rank>: v0 v1 ... vN-1"
all_out = None
# all_out = ...  (implement)

print("ALLREDUCE rank=%d: %s" % (rank, " ".join(map(str, all_out))))
sys.stdout.flush()
