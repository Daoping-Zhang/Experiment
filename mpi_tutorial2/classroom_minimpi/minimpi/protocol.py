"""protocol.py — wire protocol and control-plane message vocabulary.

Data plane (worker <-> worker)
    [4B header_len][header JSON utf-8][raw payload]
    header fields: ver, kind, src, dst, tag, fmt, plen, rnd, phase, algo, ts
    payload is RAW bytes (never JSON/base64) so benchmarks stay clean.

Control plane (worker <-> teacher)
    one JSON object per line (newline-delimited), fields begin with `t` (type).
"""
import json
import socket
import sys
import struct
import threading

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANY_SOURCE = -1
ANY_TAG = -1

# ---------------------------------------------------------------------------
# Version guard: worker and teacher MUST be the same version.
#   MINIMPI_VERSION  : bump on any classroom behaviour change
#   PROTOCOL_VERSION : bump when the control/data-plane message shape changes
# A worker whose version differs is rejected at join with a clear message
# telling the student to update the copy (git pull) and restart.
# ---------------------------------------------------------------------------
MINIMPI_VERSION = "2.2.0"
PROTOCOL_VERSION = 2

# tag meanings inside the MiniMPI DATA plane (all via comm.send/comm.recv):
#   ALGO_TAG_BASE   : upper bound of the algorithm tag region. Every collective
#                     module owns ONE small dedicated channel tag below it
#                     (ping_pong=0, naive_reduce=101, naive_allreduce=201,
#                     tree_reduce=301, recursive_doubling_allreduce=401, ring_allreduce=501),
#                     so within one run messages of different algorithms can
#                     never be confused.
#   BARRIER_TAG_BASE: teaching sync / start-barrier messages begin here;
#                     the barrier of round r uses tag BARRIER_TAG_BASE + r.
#                     A barrier is implemented as an allreduce-of-1 — that is
#                     a MiniMPI TEACHING implementation, NOT a statement about
#                     real MPI (production MPI may use dedicated barrier
#                     algorithms).
# Algorithm tags (< ALGO_TAG_BASE) and barrier tags (>= BARRIER_TAG_BASE) live
# in disjoint regions, so an algorithm message can never satisfy a barrier
# recv_match and vice versa.
# Classroom control (RUN / DONE / SHUTDOWN) is NOT an MPI message and has no
# MPI tag — it travels over its own control channel.
ALGO_TAG_BASE = 1000
BARRIER_TAG_BASE = 7000

TAG_DATA = 0     # plain point-to-point payload tag (ping_pong, Tutorial-1 style)

# Event / message classification: what a message is FOR.
KIND_ALGO = "algorithm"    # real collective payload communication
KIND_BARRIER = "barrier"   # teaching synchronization traffic (hide from view)
ELEMENT_BYTES = 4          # one int32 element (current i32 data type)

FMT_INT32 = "i32"     # struct int32 vector (list of ints)
FMT_FLOAT64 = "f64"   # struct float64 vector (list of floats)
FMT_RAW = "raw"       # opaque bytes (benchmark payload / big-int XOR reduce)

KIND_DATA = "data"    # payload-carrying data-plane message
KIND_PING = "ping"    # data-plane liveness probe (peer connectivity check)

# control message types
C_JOIN = "join"                 # worker -> teacher: register
C_WELCOME = "welcome"           # teacher -> worker: rank/size/peers
C_HELLO = "hello"               # worker -> teacher: control link ready
C_CHECK = "check"               # teacher -> worker: run p2p connectivity check
C_CHECK_REPORT = "check_report" # worker -> teacher: p2p probe results
C_RUN = "run"                   # teacher -> worker: start a demo/algorithm
C_EVENT = "event"               # worker -> teacher: one communication event
C_ROUND_DONE = "round_done"     # worker -> teacher: finished logical round r
C_RELEASE = "release"           # teacher -> worker: teaching-mode round release
C_DONE = "collective_done"      # worker -> teacher: algorithm finished (+ value)
C_SUMMARY = "summary"           # teacher -> worker: benchmark summary (display)
C_ABORT = "abort"                # teacher -> worker: cancel current RUN
C_HEARTBEAT = "heartbeat"        # worker -> teacher: "I am alive and waiting"
C_KICK = "kick"                  # teacher -> worker: you were removed from the class
# A worker that stops sending heartbeats is frozen (asleep laptop, dead VM,
# stopped process) — the only reliable way to tell it apart from a rank that is
# simply blocked in a barrier, since both look silent on the data plane.
HEARTBEAT_S = 2.0
HEARTBEAT_STALE_S = 6.0
C_SHUTDOWN = "shutdown"
C_ERROR = "error"

# ---------------------------------------------------------------------------
# Data-plane framing
# ---------------------------------------------------------------------------
def send_frame(sock, header, payload):
    """Send one length-prefixed frame: [4B json-header len][header][payload]."""
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!I", len(hb)))
    sock.sendall(hb)
    sock.sendall(bytes(payload))


def recv_frame(sock):
    """Read one frame from `sock`. Returns (header dict, payload bytes)."""
    hl = _recv_exact(sock, 4)
    (hlen,) = struct.unpack("!I", hl)
    hb = _recv_exact(sock, hlen)
    header = json.loads(hb.decode("utf-8"))
    plen = int(header["plen"])
    payload = _recv_exact(sock, plen)
    return header, payload


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed connection mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def make_header(**kw):
    h = {"ver": 1, "kind": KIND_ALGO, "src": ANY_SOURCE, "dst": ANY_SOURCE,
         "tag": 0, "fmt": FMT_RAW, "plen": 0, "rnd": 0, "phase": "", "algo": "", "ts": 0}
    h.update(kw)
    return h

# ---------------------------------------------------------------------------
# Control-plane helpers (newline-delimited JSON)
# ---------------------------------------------------------------------------
# Each endpoint reads control messages from a DEDICATED blocking reader
# thread, and sends are guarded by the caller's own lock — so no settimeout
# is ever applied to a socket that a concurrent sendall uses.


def ctrl_send(sock, obj):
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))


def ctrl_recv_line(sock):
    """Blocking read of one newline-delimited JSON object. Raises
    ConnectionError when the peer closes the connection."""
    buf = bytearray()
    while True:
        byte = sock.recv(1)
        if not byte:
            raise ConnectionError("control connection closed")
        if byte == b"\n":
            return json.loads(buf.decode("utf-8"))
        buf.extend(byte)


# ---------------------------------------------------------------------------
# Socket liveness
# ---------------------------------------------------------------------------
def enable_keepalive(sock, idle=8, interval=2, probes=3):
    """Best-effort TCP keepalive on a MiniMPI socket.

    A student closing the laptop lid or pulling the cable sends no FIN: the
    peer's reads simply block forever and the class would wait for a rank that
    can no longer answer (until the stall watchdog fires). Keepalive turns
    that into a normal disconnect, which every robustness path already
    handles. Option names differ per platform, so every step is best-effort.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        return False
    for name, value in (("TCP_KEEPIDLE", idle),       # Linux
                        ("TCP_KEEPALIVE", idle),      # macOS: idle seconds
                        ("TCP_KEEPINTVL", interval),
                        ("TCP_KEEPCNT", probes)):
        opt = getattr(socket, name, None)
        if opt is None:
            continue
        try:
            sock.setsockopt(socket.IPPROTO_TCP, opt, value)
        except OSError:
            pass
    if sys.platform == "darwin" and getattr(socket, "TCP_KEEPALIVE", None) is None:
        # CPython on macOS does not expose Darwin's TCP_KEEPALIVE (idle
        # seconds, option 0x10), so set it through the raw number: without it
        # the first probe would only go out after the OS default (2 hours).
        try:
            sock.setsockopt(socket.IPPROTO_TCP, 0x10, idle)
        except OSError:
            pass
    return True
