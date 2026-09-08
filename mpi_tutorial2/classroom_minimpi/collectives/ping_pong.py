"""ping_pong.py — STUDENT-FACING: point-to-point refresher (Demo 0).

    Rank 1 sends its value to Rank 0; Rank 0 receives it (like Tutorial 1's
    MPI_Send / MPI_Recv). Requires size >= 2.
"""
TAG = 0      # MPI.TAG_DATA: plain point-to-point payload (data plane)


def ping_pong(comm, value):
    if comm.Get_size() < 2:
        raise ValueError("ping_pong requires at least 2 ranks")
    comm.begin_round(1, "send-recv")
    if comm.Get_rank() == 0:
        v = comm.recv(source=1, tag=TAG)
        comm.sync_round(1)
        return v
    if comm.Get_rank() == 1:
        comm.send(value, dest=0, tag=TAG)
        comm.sync_round(1)
        return value
    comm.sync_round(1)
    return value
