"""naive_reduce.py — STUDENT-FACING: All-to-One Reduce to the root.

Read this file top to bottom: it is the whole algorithm.

    every rank that is not the root   ->   send(value, dest=root)
    root (rank 0)                     ->   recv() from everyone, combine (SUM)

Pattern to learn:
    Everyone -> Root     (root becomes a communication hotspot)
"""
from minimpi.communicator import ANY_SOURCE, combine

TAG = 101


def naive_reduce(comm, value, op="sum", root=0):
    rnd = 1
    if comm.Get_rank() == root:
        acc = value if isinstance(value, (bytes, bytearray)) else list(value)
        comm.begin_round(rnd, "all-to-one")
        for _ in range(1, comm.Get_size()):
            received = comm.recv(source=ANY_SOURCE, tag=TAG)
            acc = combine(acc, received, op, comm.fmt)   # local reduce
            comm.note_operation_complete("sum")          # real SUM finished
        comm.sync_round(rnd)
        return acc
    comm.begin_round(rnd, "all-to-one")
    comm.send(value, dest=root, tag=TAG)
    comm.sync_round(rnd)
    return value
