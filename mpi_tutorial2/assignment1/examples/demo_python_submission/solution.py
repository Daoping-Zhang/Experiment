"""demo_python_submission — MPI Reduce + AllReduce (SUM, root=0).

Run by the grader: mpirun -n P ./run.sh <input.txt>
"""
import sys
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    path = sys.argv[1]
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    P = int(lines[0].split()[0])
    local = [int(x) for x in lines[1 + rank].split()]

    reduce_out = [0] * len(local) if rank == 0 else None
    comm.Reduce(local, reduce_out, root=0, op=MPI.SUM)
    if rank == 0:
        print("REDUCE rank=0: %s" % " ".join(map(str, reduce_out)))

    all_out = [0] * len(local)
    comm.Allreduce(local, all_out, op=MPI.SUM)
    print("ALLREDUCE rank=%d: %s" % (rank, " ".join(map(str, all_out))))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
