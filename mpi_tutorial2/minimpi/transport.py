"""transport.py — the ONLY module allowed to touch sockets directly.

PeerTransport is a per-rank object:
  * opens one TCP listener (a background accept/read loop fills an inbox),
  * lazily opens outbound TCP connections to other ranks and caches them,
  * reads length-prefixed frames (never assumes one recv == one message),
  * lets the caller pull the next frame that matches a (source, tag) matcher.

Collective code must never import this directly — use minimpi.Communicator.
"""
import socket
import threading
import time

from . import protocol as P


def _now():
    return time.perf_counter_ns()


class PeerTransport:
    """One data-plane endpoint per rank."""

    def __init__(self, name, bind_host="0.0.0.0", listen_timeout=5.0):
        self.name = name
        self.peers = {}            # rank -> (host, port)
        self.rank = None
        self._ln = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._ln.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._ln.bind((bind_host, 0))
        self.host, self.port = self._ln.getsockname()[:2]
        self._ln.listen(128)
        self._ln.settimeout(0.5)

        self._inbox = []           # list of (header, payload)
        self._inbox_cv = threading.Condition()
        self._out = {}             # rank -> socket
        self._out_lock = threading.Lock()
        self._closed = False
        self._reader_threads = []

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    # ------------------------------------------------------------------ peers
    def set_peers(self, rank, peers):
        self.rank = rank
        self.peers = dict(peers)   # rank -> (host, port)

    def endpoint_str(self):
        return "%s:%d" % (self.host, self.port)

    # ------------------------------------------------------------------ accept
    def _accept_loop(self):
        while not self._closed:
            try:
                conn, _ = self._ln.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            t = threading.Thread(target=self._reader_loop, args=(conn,), daemon=True)
            t.start()
            self._reader_threads.append(t)

    def _reader_loop(self, conn):
        try:
            while not self._closed:
                header, payload = P.recv_frame(conn)
                with self._inbox_cv:
                    self._inbox.append((header, payload))
                    self._inbox_cv.notify_all()
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ------------------------------------------------------------------ send
    def send_to(self, rank, header, payload):
        """Send one frame to a peer (lazily connects and caches the socket)."""
        host, port = self.peers[rank]
        conn = self._get_out(rank, host, port)
        header = dict(header)
        header["src"] = header.get("src", self.rank)
        header["dst"] = rank
        header["plen"] = len(payload)
        header["ts"] = _now()
        with self._out_lock:
            P.send_frame(conn, header, payload)

    def _get_out(self, rank, host, port):
        with self._out_lock:
            conn = self._out.get(rank)
            if conn is not None:
                return conn
            conn = socket.create_connection((host, port), timeout=8.0)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._out[rank] = conn
            return conn

    # ------------------------------------------------------------------ recv
    def recv_match(self, source=P.ANY_SOURCE, tag=P.ANY_TAG, timeout=None):
        """Block until a frame whose header matches source/tag arrives.

        Matches MPI's queue mental model: arrival order over TCP does not
        imply the order in which the application posts receives.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._inbox_cv:
            while True:
                for i, (header, payload) in enumerate(self._inbox):
                    if (source == P.ANY_SOURCE or header.get("src") == source) and \
                       (tag == P.ANY_TAG or header.get("tag") == tag):
                        del self._inbox[i]
                        return header, payload
                if deadline is None:
                    self._inbox_cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._inbox_cv.wait(remaining)

    # ------------------------------------------------------------------ close
    def close(self):
        self._closed = True
        try:
            self._ln.close()
        except OSError:
            pass
        with self._out_lock:
            for conn in self._out.values():
                try:
                    conn.close()
                except OSError:
                    pass
            self._out.clear()
