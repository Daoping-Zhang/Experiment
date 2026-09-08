"""minimpi — a teaching MiniMPI runtime (stdlib only, not production MPI).

Layers:

    minimpi.transport       TCP transport (only module touching sockets)
    minimpi.communicator    send/recv + value encoding + reduce kernels
    minimpi.collectives_*   collective algorithms on send/recv only
    minimpi.runtime         per-rank identity, events, round sync
"""
