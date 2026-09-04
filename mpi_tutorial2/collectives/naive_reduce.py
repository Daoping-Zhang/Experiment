"""naive_reduce.py — all-to-one reduce to root (Rank 0 is the hotspot).

    Rank 1 -----------\
    Rank 2 ------------\
    Rank 3 -------------> Rank 0 (root)
    ...

One logical round: every non-root rank sends its local value to the root;
the root receives all of them and combines (default op: SUM).
"""
from minimpi.communicator import ANY_SOURCE, combine

TAG = 101


def naive_reduce(rt, value, op="sum", fmt="i32", root=0):
    comm = rt.comm
    rnd = 1
    if comm.rank == root:
        acc = list(value) if fmt != "raw" else value
        for _ in range(1, comm.size):
            v = comm.recv(source=ANY_SOURCE, tag=TAG, fmt=fmt,
                          algo="naive_reduce", phase="all-to-one", rnd=rnd)
            acc = combine(acc, v, op, fmt)
        rt.sync_round(rnd)
        return acc
    comm.send(value, dest=root, tag=TAG, fmt=fmt, algo="naive_reduce",
              phase="all-to-one", rnd=rnd)
    rt.sync_round(rnd)
    return value
