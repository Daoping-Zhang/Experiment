"""tree_allreduce.py — tree reduce to root, then a REVERSE tree broadcast.

    Phase 1 (rounds 1..L):  binomial tree reduce  -> root holds the sum
    Phase 2 (rounds L+1..2L): reverse the tree, root fans the result out

Only comm.send / comm.recv are used — never a second socket path.
Requires a power-of-two world size.
"""
from minimpi.communicator import combine

TAG = 401


def _is_pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


def tree_allreduce(rt, value, op="sum", fmt="i32", root=0):
    comm = rt.comm
    P = comm.size
    if not _is_pow2(P):
        raise ValueError("tree_allreduce requires a power-of-two world size, got %d" % P)
    if root != 0:
        raise ValueError("tree_allreduce first version supports root == 0 only")

    R = P.bit_length() - 1
    rank = comm.rank
    local = value
    active = True

    # ---- Phase 1: reduce (rounds 1..R) -----------------------------------
    for k in range(R):
        rnd = k + 1
        low = (rank & ((1 << k) - 1)) == 0
        bit_set = (rank >> k) & 1 == 1
        if active and bit_set and low:
            partner = rank ^ (1 << k)
            comm.send(local, dest=partner, tag=TAG, fmt=fmt,
                      algo="tree_allreduce", phase="reduce", rnd=rnd)
            active = False
        elif active and not bit_set and low and (rank + (1 << k)) < P:
            partner = rank + (1 << k)
            v = comm.recv(source=partner, tag=TAG, fmt=fmt,
                          algo="tree_allreduce", phase="reduce", rnd=rnd)
            local = combine(local, v, op, fmt)
        rt.sync_round(rnd)

    # ---- Phase 2: broadcast (rounds R+1..2R) -----------------------------
    # Reverse of the reduce tree. Level k senders are the parents that
    # received at reduce level k: {p | low bits clear, bit k clear, p+2^k<P}.
    # Level k receivers are the original reduce senders S_k (children):
    #   {r | low bits clear AND bit k set}.  A rank with lower bits set is a
    # child of an earlier level and must NOT receive here (that was the bug:
    # 'bit_set' alone made e.g. rank 3 wait at level 1 forever).
    for k in range(R - 1, -1, -1):
        rnd = R + (R - k)
        low = (rank & ((1 << k) - 1)) == 0
        bit_set = (rank >> k) & 1 == 1
        if low and not bit_set and (rank + (1 << k)) < P:
            partner = rank + (1 << k)
            comm.send(local, dest=partner, tag=TAG, fmt=fmt,
                      algo="tree_allreduce", phase="broadcast", rnd=rnd)
        elif low and bit_set:
            partner = rank ^ (1 << k)
            local = comm.recv(source=partner, tag=TAG, fmt=fmt,
                              algo="tree_allreduce", phase="broadcast", rnd=rnd)
        rt.sync_round(rnd)
    return local
