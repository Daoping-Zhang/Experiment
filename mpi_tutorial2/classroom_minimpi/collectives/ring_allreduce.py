"""ring_allreduce.py — STUDENT-FACING: Reduce-Scatter + AllGather.

Real ring allreduce, NOT "the whole vector travels around once":

    Phase 1 reduce-scatter (P-1 rounds): split into P chunks; every round each
        rank sends one chunk to next and receives one from prev, combining.
        Afterwards rank r holds the fully reduced chunk (r+1) % P.
    Phase 2 allgather (P-1 rounds): those reduced chunks rotate around the
        ring until every rank has all of them.

    next = (rank + 1) % P ;  prev = (rank - 1 + P) % P

Limitation (see README): vector length must be divisible by world size.
"""
from minimpi.communicator import combine

TAG = 501


def ring_allreduce(comm, value, op="sum", root=0):
    P = comm.Get_size()
    rank = comm.Get_rank()
    if P < 2:
        return value

    n = len(value)
    if n % P:
        raise ValueError("ring_allreduce requires payload length divisible by "
                         "world size (%d %% %d != 0)" % (n, P))

    chunk_len = n // P
    chunks = [value[i * chunk_len:(i + 1) * chunk_len] for i in range(P)]
    nxt = (rank + 1) % P
    prv = (rank - 1 + P) % P

    # ---- Phase 1: reduce-scatter -----------------------------------------
    for step in range(P - 1):
        rnd = step + 1
        send_idx = (rank - step) % P
        recv_idx = (send_idx - 1) % P
        comm.begin_round(rnd, "reduce-scatter")
        comm.send(chunks[send_idx], dest=nxt, tag=TAG)
        got = comm.recv(source=prv, tag=TAG)
        chunks[recv_idx] = combine(chunks[recv_idx], got, op, comm.fmt)
        comm.note_operation_complete("sum")      # real SUM on the chunk
        comm.sync_round(rnd)

    # ---- Phase 2: allgather ----------------------------------------------
    owned = (rank + 1) % P
    final = [None] * P
    final[owned] = chunks[owned]
    cur = chunks[owned]
    cur_idx = owned
    for step in range(P - 1):
        rnd = P + step
        comm.begin_round(rnd, "allgather")
        comm.send(cur, dest=nxt, tag=TAG)
        cur = comm.recv(source=prv, tag=TAG)
        cur_idx = (cur_idx - 1) % P
        final[cur_idx] = cur
        comm.note_operation_complete("copy")     # real COPY of the chunk
        comm.sync_round(rnd)

    if isinstance(value, (bytes, bytearray)):
        out = bytearray()
        for c in final:
            out.extend(c)
        return bytes(out)
    out = []
    for c in final:
        out.extend(c)
    return out
