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

BASE_TAG = P.BARRIER_TAG_BASE   # barrier tag = BARRIER_TAG_BASE + rnd


def barrier(comm, rnd, on_root_ready=None, on_root_gathered=None):
    """Blocking data-plane barrier (allreduce-of-1, MiniMPI teaching impl).

    non-root: send([1] -> root)  then  recv([1] <- root)
    root:     recv [1] from every rank
                  -> on_root_gathered(rnd)   (all ranks finished this round)
                  -> on_root_ready(rnd)      (teacher display / ENTER pause)
              send([1] -> every rank)

    Teaching pauses between on_root_gathered and on_root_ready; that pause is
    AFTER the gather time, so it never enters round timing. Performance mode
    never calls this barrier.
    """
    root = 0
    tag = BASE_TAG + rnd
    if comm.rank == root:
        for _ in range(1, comm.size):
            comm.recv(source=P.ANY_SOURCE, tag=tag, fmt="i32",
                      algo="teaching-barrier", phase="sync-wait", rnd=rnd,
                      kind=P.KIND_BARRIER)
        if on_root_gathered is not None:
            on_root_gathered(rnd)
        if on_root_ready is not None:
            on_root_ready(rnd)
        for dst in range(1, comm.size):
            comm.send([1], dest=dst, tag=tag, fmt="i32",
                      algo="teaching-barrier", phase="sync-go", rnd=rnd,
                      kind=P.KIND_BARRIER)
    else:
        comm.send([1], dest=root, tag=tag, fmt="i32",
                  algo="teaching-barrier", phase="sync-wait", rnd=rnd,
                  kind=P.KIND_BARRIER)
        comm.recv(source=root, tag=tag, fmt="i32",
                  algo="teaching-barrier", phase="sync-go", rnd=rnd,
                  kind=P.KIND_BARRIER)
    return True
