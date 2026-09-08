"""runtime.py — one per rank: identity, communicator, events, round sync.

A rank is either:
  * a student worker (owns a PeerTransport + control connection to teacher), or
  * the teacher's rank 0 (owns a PeerTransport; control is in-process).

`run_algorithm(params)` dispatches to the collectives package, so teacher and
worker execute byte-for-byte the same algorithm code.
"""
import json
import socket
import threading

from . import protocol as P
from .transport import PeerTransport
from .communicator import Communicator
from .metrics import EventLog


class SendRecvControl:
    """Newline-delimited JSON control channel client (worker side).

    Reads happen on one dedicated reader thread (blocking). All sends go
    through `send()`, which is guarded by a lock so concurrent algorithm
    threads can report without racing.
    """

    def __init__(self, host, port, timeout=8.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(None)
        self.lock = threading.Lock()

    def send(self, obj):
        with self.lock:
            P.ctrl_send(self.sock, obj)

    def recv(self):
        return P.ctrl_recv_line(self.sock)

    def send_round_done(self, rnd, events=None):
        self.send({"t": P.C_ROUND_DONE, "rnd": rnd,
                   "events": [e.to_dict() for e in events] if events else []})

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class MiniRuntime:
    def __init__(self, name="worker", rank=None, size=None, peers=None,
                 control=None, mode="performance", bind_host="0.0.0.0",
                 transport=None):
        self.name = name
        self.rank = rank
        self.size = size
        self.control = control
        self.mode = mode
        self.advertise_ip = None
        self.events = EventLog()
        self.local_value = None
        # teaching-mode hooks (set by worker / teacher before a run):
        # teaching ranks enter the barrier FIRST (arrival ASAP); the student
        # local view + C_ROUND_DONE upload run inside the barrier's
        # `on_arrived` window — see worker.py's round_barrier().
        self._barrier = None  # callable(rnd) — data-plane barrier (arrival first)
        self._report = None   # callable(rnd, round_events) — fallback path only
        self._on_round = None # callable(rnd) — fallback path only
        self.show_ui = False  # only the student worker renders local views
        self.run_meta = {}    # algorithm / data_size / vector_len / fmt

        self.transport = transport if transport is not None else PeerTransport(name, bind_host)
        self.comm = Communicator(self.transport, rank if rank is not None else 0,
                                 size if size is not None else 1, events=self.events)
        self.done_ok = False
        self.done_error = None

    # ------------------------------------------------------------------ join
    def register_with_teacher(self, server, name=None):
        """Worker path: connect control, join, receive rank/size/peers."""
        if ":" in server:
            host, port = server.rsplit(":", 1)
            port = int(port)
        else:
            host, port = server, 9000
        self.control = SendRecvControl(host, port)
        ip = _detect_ip(host)
        self.advertise_ip = ip
        self.control.send({"t": P.C_JOIN, "host": ip,
                           "port": self.transport.port})
        welcome = self.control.recv()
        if welcome is None or welcome.get("t") != P.C_WELCOME:
            raise RuntimeError("did not receive welcome from teacher")
        self.rank = int(welcome["rank"])
        self.size = int(welcome["size"])
        peers = {int(k): (v["host"], v["port"]) for k, v in welcome["peers"].items()}
        self.transport.set_peers(self.rank, peers)
        self.transport.warm_to(peers.keys())   # real persistent sockets, no msg
        self.comm.rank = self.rank
        self.comm.size = self.size
        return welcome

    def sync_round(self, rnd, events=None):
        """Called by collectives after finishing logical round rnd.

        Performance mode: label only, no barrier.
        Teaching mode: report barrier ARRIVAL first — a rank sends its [1]
        the moment its algorithm work finishes, so Rank 0's gather time (=
        Round Finished At) never includes UI printing or control-plane event
        upload. The student local view + C_ROUND_DONE upload run inside the
        barrier's `on_arrived` window (while the rank waits for the release);
        worker.py wires that through `_barrier`.
        """
        if self.mode == "teaching":
            # test-only: simulate a rank whose LOCAL work is slow — the sleep
            # sits between the local-work-done mark and the barrier ARRIVAL,
            # so rank 0 observes this rank reaching the sync point late
            # (see tests/test_timing_worker.py O).
            import os as _os, time as _time
            d = _os.environ.get("MINIMPI_LOCAL_WORK_DELAY")
            if d:
                _time.sleep(float(d))
            if self._barrier is not None:
                self._barrier(rnd)
            else:
                # no custom barrier: keep the plain arrival/release barrier
                from . import barrier as B
                B.barrier(self.comm, rnd)
        return True

    # ------------------------------------------------------------------ run
    def run_algorithm(self, params, value=None, barrier=True):
        """Execute the collective named in params; store result."""
        from . import collectives_dispatch
        self.events.clear()
        self.local_value = collectives_dispatch.run(self, params, value=value,
                                                    barrier=barrier)
        return self.local_value

    def close(self):
        """MPI session shutdown: close control + peer transport.
        (Not a collective completion — no fake C_DONE is sent here.)"""
        if self.control is not None:
            try:
                self.control.close()
            except Exception:
                pass
        self.transport.close()


def _detect_ip(server_host):
    """Best-effort local IP visible from the teacher (UDP trick, no traffic)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((server_host, 9))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
