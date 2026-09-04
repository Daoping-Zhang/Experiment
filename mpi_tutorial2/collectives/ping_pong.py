"""ping_pong.py — Demo 0: point-to-point refresher (Tutorial 1 content).

Rank 1 sends its local value to Rank 0; Rank 0 receives and prints it.
Built only on comm.send/comm.recv. Requires size >= 2.
"""
TAG = 0   # TAG_DATA: plain point-to-point payload (data plane)


def ping_pong(rt, value):
    comm = rt.comm
    if comm.size < 2:
        raise ValueError("ping_pong requires at least 2 ranks")
    if comm.rank == 0:
        v = comm.recv(source=1, tag=TAG, fmt="i32", algo="ping_pong",
                      phase="send-recv", rnd=1)
        rt.sync_round(1)
        return v
    if comm.rank == 1:
        comm.send(value, dest=0, tag=TAG, fmt="i32", algo="ping_pong",
                  phase="send-recv", rnd=1)
        rt.sync_round(1)
        return value
    rt.sync_round(1)
    return value
