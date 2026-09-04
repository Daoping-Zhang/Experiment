"""barrier.py — teaching-mode round barrier over the DATA plane only.

Between two logical rounds, every rank must prove it finished round r before
anyone starts round r+1. Instead of a control-plane handshake, this is done
NCCL-style: a tiny allreduce that passes the value 1.

    non-root: send([1] -> root)  then  recv([1] <- root)
    root:     recv [1] from every rank  ->  [teacher prints round r, ENTER]
              send([1] -> every rank)

So rank 0 (the teacher) only completes the barrier after it has shown the
global view of round r and released the class manually. Built on the same
comm.send/comm.recv primitives as every collective.
"""
from . import protocol as P

BASE_TAG = 7000   # control-plane sync tags (CTRL_TAG_BASE + rnd)


def barrier(comm, rnd, on_root_ready=None):
    """Blocking round barrier. `on_root_ready(rnd)` runs on rank 0 between
    the gather and the broadcast legs (that is where the teacher pauses)."""
    root = 0
    tag = BASE_TAG + rnd
    if comm.rank == root:
        for _ in range(1, comm.size):
            comm.recv(source=P.ANY_SOURCE, tag=tag, fmt="i32",
                      algo="teaching-barrier", phase="sync-wait", rnd=rnd)
        if on_root_ready is not None:
            on_root_ready(rnd)
        for dst in range(1, comm.size):
            comm.send([1], dest=dst, tag=tag, fmt="i32",
                      algo="teaching-barrier", phase="sync-go", rnd=rnd)
    else:
        comm.send([1], dest=root, tag=tag, fmt="i32",
                  algo="teaching-barrier", phase="sync-wait", rnd=rnd)
        comm.recv(source=root, tag=tag, fmt="i32",
                  algo="teaching-barrier", phase="sync-go", rnd=rnd)
    return True
