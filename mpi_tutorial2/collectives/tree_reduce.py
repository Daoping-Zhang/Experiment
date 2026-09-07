"""tree_reduce.py — STUDENT-FACING: Binomial-Tree Reduce (log2(P) rounds).

For P = 8 the pattern is:

    Round 1:  1->0  3->2  5->4  7->6
    Round 2:  2->0  6->4
    Round 3:  4->0

A rank sends exactly ONCE (on the round of its lowest set bit) and then goes
inactive; the receiving parent combines the message into its local value.
Fewer steps than naive reduce -> less root hotspot (but harder to write).

Requires a power-of-two world size (clear error otherwise).
"""
from minimpi.communicator import combine

TAG = 301


def _pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


def tree_reduce(comm, value, op="sum", root=0):
    P = comm.Get_size()
    if not _pow2(P):
        raise ValueError("tree_reduce requires a power-of-two world size, got %d" % P)
    if root != 0:
        raise ValueError("tree_reduce (first version) supports root == 0 only")

    rank = comm.Get_rank()
    local = value if isinstance(value, (bytes, bytearray)) else list(value)
    active = True

    for k in range(P.bit_length() - 1):      # log2(P) rounds
        rnd = k + 1
        bit = 1 << k
        low_bits_clear = (rank & (bit - 1)) == 0
        is_sender = ((rank >> k) & 1) == 1

        comm.begin_round(rnd, "reduce")
        if active and is_sender and low_bits_clear:
            parent = rank ^ bit                 # clear bit k -> smaller rank
            comm.send(local, dest=parent, tag=TAG)
            active = False                     # sent once, becomes inactive
        elif active and not is_sender and low_bits_clear and (rank + bit) < P:
            child = rank + bit
            received = comm.recv(source=child, tag=TAG)
            local = combine(local, received, op, comm.fmt)
        comm.sync_round(rnd)
    return local
