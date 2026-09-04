"""ring_allreduce.py — real ring allreduce: Reduce-Scatter + AllGather.

NOT "the whole vector travels around the ring once". The vector is split
into `size` chunks; every round each rank sends one chunk to its next
neighbour and receives one from its previous neighbour:

    next = (rank + 1) % size
    prev = (rank - 1 + size) % size

Phase 1 reduce-scatter (size-1 rounds): chunk partials combine as they travel
around the ring; afterwards rank r holds the fully reduced chunk (r+1)%size.
Phase 2 allgather (size-1 rounds): the reduced chunks rotate around the ring
until every rank has all of them.

Limitation (documented in README): payload element/byte count must be
divisible by world size.
"""
from minimpi.communicator import combine

TAG = 501


def ring_allreduce(rt, value, op="sum", fmt="i32"):
    comm = rt.comm
    P = comm.size
    rank = comm.rank
    if P < 2:
        return value

    if fmt == "i32":
        n = len(value)
    else:
        n = len(bytes(value))
    if n % P != 0:
        raise ValueError("ring_allreduce requires payload length divisible by "
                         "world size (%d %% %d != 0)" % (n, P))

    chunk = n // P
    chunks = [value[i * chunk:(i + 1) * chunk] for i in range(P)]
    nxt = (rank + 1) % P
    prv = (rank - 1 + P) % P

    # ---- Phase 1: reduce-scatter (rounds 1..P-1) --------------------------
    for step in range(P - 1):
        rnd = step + 1
        send_idx = (rank - step) % P
        recv_idx = (send_idx - 1) % P
        comm.send(chunks[send_idx], dest=nxt, tag=TAG, fmt=fmt,
                  algo="ring_allreduce", phase="reduce-scatter", rnd=rnd)
        got = comm.recv(source=prv, tag=TAG, fmt=fmt, algo="ring_allreduce",
                        phase="reduce-scatter", rnd=rnd)
        chunks[recv_idx] = combine(chunks[recv_idx], got, op, fmt)
        rt.sync_round(rnd)

    # ---- Phase 2: allgather (rounds P..2P-2) ------------------------------
    owned = (rank + 1) % P
    final = [None] * P
    final[owned] = chunks[owned]
    cur = chunks[owned]
    cur_idx = owned
    for step in range(P - 1):
        rnd = P + step
        comm.send(cur, dest=nxt, tag=TAG, fmt=fmt,
                  algo="ring_allreduce", phase="allgather", rnd=rnd)
        cur = comm.recv(source=prv, tag=TAG, fmt=fmt,
                        algo="ring_allreduce", phase="allgather", rnd=rnd)
        cur_idx = (cur_idx - 1) % P
        final[cur_idx] = cur
        rt.sync_round(rnd)

    if fmt == "i32":
        out = []
        for c in final:
            out.extend(c)
        return out
    b = bytearray()
    for c in final:
        b.extend(c)
    return bytes(b)
