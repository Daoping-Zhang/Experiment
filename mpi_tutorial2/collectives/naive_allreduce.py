"""naive_allreduce.py — all-to-one reduce (round 1) + one-to-all broadcast
(round 2). Everyone ends with the global result. This is the BASELINE that
Tree / Ring are compared against.

    Phase 1 (all-to-one):  everyone -> root
    Phase 2 (one-to-all):  root -> everyone

No tree code is reused here on purpose — it is the naive baseline.
"""
from minimpi.communicator import ANY_SOURCE, combine

TAG = 201


def naive_allreduce(rt, value, op="sum", fmt="i32", root=0):
    comm = rt.comm
    if comm.rank == root:
        acc = list(value) if fmt != "raw" else value
        for _ in range(1, comm.size):
            v = comm.recv(source=ANY_SOURCE, tag=TAG, fmt=fmt,
                          algo="naive_allreduce", phase="all-to-one", rnd=1)
            acc = combine(acc, v, op, fmt)
        rt.sync_round(1)
        for dst in range(comm.size):
            if dst != root:
                comm.send(acc, dest=dst, tag=TAG, fmt=fmt,
                          algo="naive_allreduce", phase="one-to-all", rnd=2)
        rt.sync_round(2)
        return acc
    comm.send(value, dest=root, tag=TAG, fmt=fmt, algo="naive_allreduce",
              phase="all-to-one", rnd=1)
    rt.sync_round(1)
    result = comm.recv(source=root, tag=TAG, fmt=fmt, algo="naive_allreduce",
                       phase="one-to-all", rnd=2)
    rt.sync_round(2)
    return result
