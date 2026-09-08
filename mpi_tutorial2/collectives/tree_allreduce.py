"""tree_allreduce.py — STUDENT-FACING: Tree Reduce + Reverse-Tree Broadcast.

    Phase 1 (rounds 1..L,   L=log2 P):  tree reduce      -> root has the sum
    Phase 2 (rounds L+1..2L):           reverse the tree -> everyone gets it

Every rank ends with the same reduced value. Only comm.send/comm.recv.
Requires a power-of-two world size.
"""
from minimpi.communicator import combine

TAG = 401


def _pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


def tree_allreduce(comm, value, op="sum", root=0):
    P = comm.Get_size()
    if not _pow2(P):
        raise ValueError("tree_allreduce requires a power-of-two world size, got %d" % P)
    if root != 0:
        raise ValueError("tree_allreduce (first version) supports root == 0 only")

    rank = comm.Get_rank()
    local = value if isinstance(value, (bytes, bytearray)) else list(value)
    L = P.bit_length() - 1
    active = True

    # ---- Phase 1: reduce -------------------------------------------------
    for k in range(L):
        rnd = k + 1
        bit = 1 << k
        low_bits_clear = (rank & (bit - 1)) == 0
        is_sender = ((rank >> k) & 1) == 1
        comm.begin_round(rnd, "reduce")
        if active and is_sender and low_bits_clear:
            parent = rank ^ bit
            comm.send(local, dest=parent, tag=TAG)
            active = False
        elif active and not is_sender and low_bits_clear and (rank + bit) < P:
            child = rank + bit
            received = comm.recv(source=child, tag=TAG)
            local = combine(local, received, op, comm.fmt)
            comm.note_operation_complete("sum")   # real SUM finished
        comm.sync_round(rnd)

    # ---- Phase 2: reverse-tree broadcast ---------------------------------
    for k in range(L - 1, -1, -1):
        rnd = L + (L - k)
        bit = 1 << k
        low_bits_clear = (rank & (bit - 1)) == 0
        is_sender = ((rank >> k) & 1) == 1
        comm.begin_round(rnd, "broadcast")
        if low_bits_clear and not is_sender and (rank + bit) < P:
            child = rank + bit
            comm.send(local, dest=child, tag=TAG)
        elif low_bits_clear and is_sender:
            parent = rank ^ bit
            local = comm.recv(source=parent, tag=TAG)
            comm.note_operation_complete("copy")  # real COPY (result)
        comm.sync_round(rnd)
    return local
