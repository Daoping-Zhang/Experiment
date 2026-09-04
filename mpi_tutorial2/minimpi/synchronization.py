"""synchronization.py — Teaching vs Performance hook.

DEPRECATED in favour of minimpi.barrier: teaching-mode round sync is now a
data-plane allreduce-of-1 barrier (NCCL style), not a control-plane handshake.
This module is kept only so older imports still resolve; runtime.sync_round
handles both modes directly.
"""

