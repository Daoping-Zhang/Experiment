"""naive_allreduce.py — STUDENT-FACING: Reduce + Broadcast (baseline).

Two logical rounds:
    Round 1  all-to-one  : everyone -> root, root combines (SUM)
    Round 2  one-to-all  : root -> everyone
Every rank ends with the same reduced value (MPI_Allreduce semantics).

This is the BASELINE that Tree / Ring are compared against.
"""
from minimpi.communicator import ANY_SOURCE, combine

TAG = 201


def naive_allreduce(comm, value, op="sum", root=0):
    size = comm.Get_size()

    # ---- Round 1: all-to-one reduce -------------------------------------
    comm.begin_round(1, "all-to-one")
    if comm.Get_rank() == root:
        acc = value if isinstance(value, (bytes, bytearray)) else list(value)
        for _ in range(1, size):
            received = comm.recv(source=ANY_SOURCE, tag=TAG)
            acc = combine(acc, received, op, comm.fmt)
        comm.sync_round(1)
    else:
        comm.send(value, dest=root, tag=TAG)
        comm.sync_round(1)

    # ---- Round 2: one-to-all broadcast -----------------------------------
    comm.begin_round(2, "one-to-all")
    if comm.Get_rank() == root:
        for dst in range(size):
            if dst != root:
                comm.send(acc, dest=dst, tag=TAG)
        comm.sync_round(2)
        return acc
    result = comm.recv(source=root, tag=TAG)
    comm.sync_round(2)
    return result
