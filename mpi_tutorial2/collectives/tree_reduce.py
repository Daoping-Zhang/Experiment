"""tree_reduce.py — binomial-tree reduce to root (parallel log(P) rounds).

For P = 8 the pattern is:

    Round 1:  1->0  3->2  5->4  7->6
    Round 2:  2->0  6->4
    Round 3:  4->0

A rank sends exactly once (on the round of its lowest set bit), then becomes
inactive; the receiving rank reduces the message into its local value.

Requires a power-of-two world size (checked with a clear error).
"""
from minimpi.communicator import combine

TAG = 301


def _is_pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


def tree_reduce(rt, value, op="sum", fmt="i32", root=0):
    comm = rt.comm
    P = comm.size
    if not _is_pow2(P):
        raise ValueError("tree_reduce requires a power-of-two world size, got %d" % P)
    if root != 0:
        raise ValueError("tree_reduce first version supports root == 0 only")

    # force the root's value into position 0 by index arithmetic below; with
    # root == 0 the tree is plain: partner = rank ^ (1 << k), smaller wins.
    R = P.bit_length() - 1
    rank = comm.rank
    local = value
    active = True

    for k in range(R):
        rnd = k + 1
        low = (rank & ((1 << k) - 1)) == 0
        bit_set = (rank >> k) & 1 == 1
        if active and bit_set and low:
            # this is the round of this rank's lowest set bit -> send once
            partner = rank ^ (1 << k)
            comm.send(local, dest=partner, tag=TAG, fmt=fmt, algo="tree_reduce",
                      phase="reduce", rnd=rnd)
            active = False
        elif active and not bit_set and low and (rank + (1 << k)) < P:
            partner = rank + (1 << k)
            v = comm.recv(source=partner, tag=TAG, fmt=fmt, algo="tree_reduce",
                          phase="reduce", rnd=rnd)
            local = combine(local, v, op, fmt)
        rt.sync_round(rnd)
    return local
