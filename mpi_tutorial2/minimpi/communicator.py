"""communicator.py — MPI-like API on top of PeerTransport.

Provides the familiar mental model:

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.send(value, dest=1, tag=0)
    value = comm.recv(source=1, tag=0)

Values:
    fmt "i32"   -> python list of ints   (struct int32 vector)
    fmt "f64"   -> python list of floats (struct float64 vector)
    fmt "raw"   -> bytes / bytearray     (opaque; used by big benchmarks)

Reduction kernels (combine) live here so every collective reuses the exact
same code path. Collective modules must call only these send/recv/combine
methods — never the transport.
"""
import struct

from . import protocol as P
from .metrics import CommunicationEvent, bandwidth_mbps, now_ns

ANY_SOURCE = P.ANY_SOURCE
ANY_TAG = P.ANY_TAG


def encode(value, fmt):
    if fmt == P.FMT_INT32:
        vals = list(value)
        return struct.pack("!%di" % len(vals), *vals)
    if fmt == P.FMT_FLOAT64:
        vals = list(value)
        return struct.pack("!%dd" % len(vals), *vals)
    return bytes(value)   # FMT_RAW


def decode(payload, fmt):
    if fmt == P.FMT_INT32:
        return list(struct.unpack("!%di" % (len(payload) // 4), payload))
    if fmt == P.FMT_FLOAT64:
        return list(struct.unpack("!%dd" % (len(payload) // 8), payload))
    return payload       # FMT_RAW


def combine(a, b, op, fmt):
    """Combine two local partials of the same format.

    op "sum" : elementwise addition  (i32/f64)
    op "xor" : bytewise xor          (raw — fast big-int reduce for benchmarks)
    """
    if op == "sum":
        if fmt == P.FMT_RAW:
            return _xor_bytes(a, b)
        return [x + y for x, y in zip(a, b)]
    if op == "xor":
        return _xor_bytes(a if fmt == P.FMT_RAW else encode(a, fmt),
                          b if fmt == P.FMT_RAW else encode(b, fmt))
    raise ValueError("unknown op: %s" % op)


def _xor_bytes(a, b):
    ai = int.from_bytes(bytes(a), "big")
    bi = int.from_bytes(bytes(b), "big")
    n = max(len(a), len(b))
    return (ai ^ bi).to_bytes(n, "big")


def zero_like(value, fmt):
    if fmt == P.FMT_RAW:
        return bytes(len(value))
    return [0] * len(value)


class Communicator:
    """Send/Recv + message matching + one event log. One per rank."""

    def __init__(self, transport, rank, size, events=None, algo_hook=None):
        self.transport = transport
        self.rank = rank
        self.size = size
        self.events = events  # metrics.EventLog or None
        self.algo_hook = None  # minimpi.synchronization hook (round sync)

    # ------------------------------------------------------------------ info
    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    # ------------------------------------------------------------------ send
    def send(self, value, dest, tag=0, fmt=P.FMT_INT32, algo="", phase="",
             rnd=0):
        """Blocking send. Returns after the frame is written to the socket."""
        payload = encode(value, fmt)
        header = P.make_header(src=self.rank, dst=dest, tag=tag, fmt=fmt,
                               plen=len(payload), rnd=rnd, phase=phase,
                               algo=algo)
        t0 = now_ns()
        self.transport.send_to(dest, header, payload)
        t1 = now_ns()
        self._emit(side="send", header=header, value_before=value, t0=t0, t1=t1)
        return len(payload)

    # ------------------------------------------------------------------ recv
    def recv(self, source=ANY_SOURCE, tag=ANY_TAG, timeout=None, fmt=None,
             algo="", phase="", rnd=0):
        """Blocking receive matched by (source, tag). Returns the decoded
        value. A CommunicationEvent is recorded on success."""
        t0 = now_ns()
        result = self.transport.recv_match(source=source, tag=tag, timeout=timeout)
        if result is None:
            raise TimeoutError("Receive timeout: source=%s tag=%s" % (source, tag))
        header, payload = result
        t1 = now_ns()
        value = decode(payload, header.get("fmt", P.FMT_RAW))
        if header.get("fmt") == P.FMT_RAW and payload is not value:
            pass  # decode returns payload itself for raw
        self._emit(side="recv", header=header, value_after=value,
                   t0=t0, t1=t1)
        return value

    # ------------------------------------------------------------- events
    def _emit(self, side, header, value_before=None, value_after=None,
              received_value=None, t0=0, t1=0):
        if self.events is None:
            return
        dt = (t1 - t0) / 1e6 if t1 > t0 else 0.0
        ev = CommunicationEvent(
            algorithm=header.get("algo", ""),
            phase=header.get("phase", ""),
            logical_round=header.get("rnd", 0),
            source=header.get("src", self.rank),
            destination=header.get("dst", self.rank),
            tag=header.get("tag", 0),
            payload_bytes=header.get("plen", 0),
            transfer_time_ms=round(dt, 4),
            effective_bandwidth_mbps=round(bandwidth_mbps(header.get("plen", 0), dt / 1e3), 3),
            ts_start_ns=t0, ts_end_ns=t1,
            value_before=value_before, received_value=received_value,
            value_after=value_after, side=side)
        self.events.record(ev)
