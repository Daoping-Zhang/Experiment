"""recursive_doubling_allreduce.py — STUDENT-FACING: Recursive Doubling
AllReduce (P must be a power of two).

Instead of reduce + broadcast, every round is a PAIRWISE EXCHANGE:

    P=4, L = log2(P) = 2 rounds:

      Round 1:  (0,1) and (2,3) exchange
      Round 2:  (0,2) and (1,3) exchange

    round d: partner = rank ^ (1 << d)

    partner sends its local result, we send ours, and BOTH combine
    (SUM). After L rounds every rank holds the total — no rank is ever
    idle, there is no root, no broadcast phase.

Send/recv order avoids a deadlock between two blocking sides:

    if rank < partner:  send -> recv
    else:               recv -> send

Only comm.send / comm.recv / combine — same primitives as everywhere.
"""
from minimpi.communicator import combine

TAG = 401


def _pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


def recursive_doubling_allreduce(comm, value, op="sum", root=0):
    P = comm.Get_size()
    if not _pow2(P):
        raise ValueError("recursive_doubling_allreduce requires a "
                         "power-of-two world size, got %d" % P)

    rank = comm.Get_rank()
    data = value if isinstance(value, (bytes, bytearray)) else list(value)
    L = P.bit_length() - 1                      # log2(P) rounds

    for d in range(L):
        rnd = d + 1
        partner = rank ^ (1 << d)               # exchange partner
        comm.begin_round(rnd, "exchange+reduce")
        if rank < partner:                      # ordered: smaller sends first
            comm.send(data, dest=partner, tag=TAG)
            incoming = comm.recv(source=partner, tag=TAG)
        else:
            incoming = comm.recv(source=partner, tag=TAG)
            comm.send(data, dest=partner, tag=TAG)
        data = combine(data, incoming, op, comm.fmt)   # real SUM
        comm.note_operation_complete("sum")
        comm.sync_round(rnd)
    return data
