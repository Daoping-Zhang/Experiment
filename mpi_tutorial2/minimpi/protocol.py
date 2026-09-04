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
import struct
import threading

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANY_SOURCE = -1
ANY_TAG = -1

# The teaching story maps the two trailing MPI arguments straight onto the
# two planes:
#   tag  -> which CHANNEL this message uses:
#             TAG_DATA    (0) = data-plane payload      (collective payloads)
#             TAG_CONTROL (1) = control-plane / sync    (teaching sync)
#   comm -> WHICH GROUP the message belongs to. MPI_COMM_WORLD is the
#           membership table the teacher's CONTROL plane built during
#           registration (rank, size, peer endpoints); you pass it to every
#           send/recv to say "within this world".
# Algorithm-phase-rounds use distinct tags above TAG_CONTROL for MPI-style
# message matching; sync/barrier messages live in CTRL_TAG_BASE.. (control
# plane over the data channel).
TAG_DATA = 0
TAG_CONTROL = 1
CTRL_TAG_BASE = 7000       # teaching barrier / sync tags (control plane)

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
    h = {"ver": 1, "kind": KIND_DATA, "src": ANY_SOURCE, "dst": ANY_SOURCE,
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
