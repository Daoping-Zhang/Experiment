"""barrier.py — teaching-mode round barrier over the DATA plane only.

MiniMPI implements Barrier internally as an AllReduce-of-1:

    every rank contributes [1]
              ↓  (reduce, sum)
    rank 0 (root) gets  world_size  = 1 + 1 + ... + 1
              ↓  (broadcast)
    every rank ends with world_size

So finishing the barrier means: I received `world_size`, i.e. every rank
arrived. Between two logical rounds this is what proves round r is complete
before anyone starts round r+1 — built on the same comm.send/comm.recv
primitives as every collective (never calling the student collectives, which
would recurse through their own sync_round).

    non-root: send([1] -> root)  then  recv([world_size] <- root)
    root:     recv [1] from every rank  ->  sum = world_size
                  -> on_root_gathered(rnd)   (all ranks finished this round)
                  -> on_root_ready(rnd)      (teacher display / ENTER pause)
              send([world_size] -> every rank)

So rank 0 (the teacher) only finishes the barrier after it has shown the
global view of round r and released the class manually. Teaching pauses
between on_root_gathered and on_root_ready are AFTER the gather time, so they
never enter round timing. Performance mode never calls this barrier.

Barrier messages use their own tag region (BARRIER_TAG_BASE + rnd) and
kind=barrier, so they can never match — or be shown as — algorithm traffic.
"""
from . import protocol as P

BASE_TAG = P.BARRIER_TAG_BASE   # barrier tag = BARRIER_TAG_BASE + rnd


def barrier(comm, rnd, on_root_gathered=None, on_root_ready=None):
    """Blocking data-plane AllReduce-of-1 barrier (MiniMPI teaching impl).

    Every rank contributes 1; rank 0 reduces them to world_size and
    broadcasts it back. Returns the reduced total (= world_size when every
    rank arrived).

    root: recv [1] from every rank
              -> on_root_gathered(rnd)   (all ranks finished this round)
              -> on_root_ready(rnd)      (teacher display / ENTER pause)
          broadcast [world_size] to every rank
    non-root: send([1] -> root)  then  recv([world_size] <- root)
    """
    root = 0
    tag = BASE_TAG + rnd
    if comm.rank == root:
        total = 1                     # root's own contribution
        for _ in range(1, comm.size):
            token = comm.recv(source=P.ANY_SOURCE, tag=tag, fmt="i32",
                              algo="teaching-barrier", phase="sync-wait",
                              rnd=rnd, kind=P.KIND_BARRIER)
            total += int(token[0]) if token else 1
        if on_root_gathered is not None:
            on_root_gathered(rnd)
        if on_root_ready is not None:
            on_root_ready(rnd)
        for dst in range(1, comm.size):
            comm.send([total], dest=dst, tag=tag, fmt="i32",
                      algo="teaching-barrier", phase="sync-go", rnd=rnd,
                      kind=P.KIND_BARRIER)
        return total
    comm.send([1], dest=root, tag=tag, fmt="i32",
              algo="teaching-barrier", phase="sync-wait", rnd=rnd,
              kind=P.KIND_BARRIER)
    token = comm.recv(source=root, tag=tag, fmt="i32",
                      algo="teaching-barrier", phase="sync-go", rnd=rnd,
                      kind=P.KIND_BARRIER)
    return int(token[0]) if token else 1
