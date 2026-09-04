#!/usr/bin/env python3
"""teacher.py — Coordinator + Rank 0 of the MiniMPI classroom runtime.

Control plane only (join/registration, world size, peer endpoints, demo/mode
selection, metrics collection, completion tracking). Collective payload never
passes through the coordinator: workers talk to workers over peer TCP.

Teaching-mode pacing is NOT a control-plane handshake: after each logical
round every rank takes part in a data-plane allreduce-of-1 barrier
(minimpi.barrier); rank 0 completes its barrier leg only after it prints the
round's global view and (interactively) the class presses ENTER.

Usage:
    python3 teacher.py --size 4 --host 0.0.0.0 --port 9000
    python3 teacher.py --size 8 --benchmark
    python3 teacher.py --size 4 --demo tree_allreduce --mode teaching --auto
"""
import argparse
import base64
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minimpi import protocol as P, collectives_dispatch  # noqa: E402
from minimpi import barrier as BarrierMod                # noqa: E402
from minimpi.transport import PeerTransport             # noqa: E402
from minimpi.communicator import Communicator           # noqa: E402
from minimpi.metrics import EventLog                    # noqa: E402


def detect_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Aggregator:
    """Thread-safe run state shared between control readers and the menu."""

    def __init__(self, size, teaching, auto):
        self.size = size
        self.teaching = teaching
        self.auto = auto
        self.cv = threading.Condition()
        self.round_reports = {}     # rnd -> set(rank)   (events coverage)
        self.round_events = {}      # rnd -> [event dicts] (worker events)
        self.done = set()           # ranks that finished
        self.results = {}           # rank -> final value
        self.errors = {}            # rank -> error string
        self.active = False

    def worker_round_done(self, rank, rnd, events):
        with self.cv:
            self.round_reports.setdefault(rnd, set()).add(rank)
            self.round_events.setdefault(rnd, []).extend(events)
            self.cv.notify_all()

    def worker_done(self, rank, value=None, error=None):
        with self.cv:
            if error:
                self.errors[rank] = error
            self.done.add(rank)
            if value is not None:
                self.results[rank] = value
            self.cv.notify_all()

    def wait_round_events(self, rnd, need, timeout=30):
        with self.cv:
            return self.cv.wait_for(
                lambda: len(self.round_reports.get(rnd, set())) >= need,
                timeout=timeout)

    def wait_all_done(self, timeout=600):
        with self.cv:
            return self.cv.wait_for(lambda: len(self.done) >= self.size,
                                    timeout=timeout)

    def reset(self):
        with self.cv:
            self.round_reports.clear()
            self.round_events.clear()
            self.done.clear()
            self.results.clear()
            self.errors.clear()
            self.active = True


class Coordinator:
    def __init__(self, size, host, port, advertise, auto=False):
        self.size = size
        self.host = host
        self.port = port
        self.advertise = advertise or (host if host != "0.0.0.0" else detect_ip())
        self.auto = auto

        self.workers = {}            # rank -> control socket
        self.peers = {0: {"host": self.advertise, "port": None}}
        self.lock = threading.Lock()
        self.next_rank = 1
        self.closed = False
        self.agg = None

        self.transport = PeerTransport("teacher", bind_host=self.host)
        self.peers[0]["port"] = self.transport.port
        self.events0 = EventLog()
        self.comm0 = Communicator(self.transport, 0, size, events=self.events0)

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(64)
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while not self.closed:
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle_worker, args=(conn,),
                             daemon=True).start()

    def _handle_worker(self, conn):
        try:
            msg = P.ctrl_recv_line(conn)
            if msg is None or msg.get("t") != P.C_JOIN:
                return
            with self.lock:
                if self.next_rank >= self.size:
                    P.ctrl_send(conn, {"t": P.C_ERROR, "why": "world already full"})
                    return
                rank = self.next_rank
                self.next_rank += 1
                self.workers[rank] = conn
                self.peers[rank] = {"host": msg["host"], "port": int(msg["port"])}
            welcome = {"t": P.C_WELCOME, "rank": rank, "size": self.size,
                       "peers": self.peers}
            P.ctrl_send(conn, welcome)
            while not self.closed:
                m = P.ctrl_recv_line(conn)
                self._dispatch(rank, m)
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _dispatch(self, rank, m):
        t = m.get("t")
        agg = self.agg
        if t == P.C_CHECK_REPORT:
            with self.lock:
                self._check_ok.setdefault(rank, set(m.get("pass", [])))
                self._check_fail.setdefault(rank, set(m.get("fail", [])))
        elif t == P.C_ROUND_DONE and agg:
            agg.worker_round_done(rank, int(m["rnd"]), m.get("events", []))
        elif t == P.C_DONE and agg:
            val = m.get("final")
            if isinstance(val, dict) and "vec" in val:
                val = val["vec"]
            elif isinstance(val, dict) and "raw" in val:
                val = base64.b64decode(val["raw"])
            agg.worker_done(rank, value=val, error=m.get("error"))
        elif t == P.C_ERROR:
            print("[ERROR from rank %d] %s" % (rank, m.get("why", "")))

    def connectivity_check(self):
        print("\n========================================\nPeer Connectivity "
              "Check\n========================================")
        self._check_ok, self._check_fail = {}, {}
        msg = {"t": P.C_CHECK, "peers": self.peers}
        for conn in self.workers.values():
            P.ctrl_send(conn, msg)
        self._self_check()
        deadline = time.time() + 30
        while len(self._check_ok) < self.size and time.time() < deadline:
            time.sleep(0.1)

        fails = 0
        for a in range(self.size):
            ok = self._check_ok.get(a, set())
            fail = self._check_fail.get(a, set())
            for b in range(self.size):
                if a == b:
                    continue
                status = "FAIL" if b in fail else ("PASS" if b in ok else "?")
                if status == "FAIL":
                    fails += 1
                print("Rank %d -> Rank %d  %s" % (a, b, status))
        if fails:
            print("\nP2P Network: NOT READY (%d failed edges)" % fails)
            return False
        print("\nP2P Network: READY")
        return True

    def _self_check(self):
        ok, fail = [], []
        for dst, ep in self.peers.items():
            if dst == 0:
                continue
            try:
                s = socket.create_connection((ep["host"], ep["port"]), timeout=3)
                s.close()
                ok.append(dst)
            except OSError:
                fail.append(dst)
        self._check_ok[0] = set(ok)
        self._check_fail[0] = set(fail)

    def send_to_workers(self, obj):
        for rank, conn in list(self.workers.items()):
            try:
                P.ctrl_send(conn, obj)
            except OSError:
                pass

    def run_demo(self, params, mode):
        self.transport.set_peers(0, {r: (ep["host"], ep["port"])
                                     for r, ep in self.peers.items() if r != 0})
        agg = Aggregator(self.size, teaching=(mode == "teaching"), auto=self.auto)
        self.agg = agg
        agg.reset()
        params = dict(params, mode=mode, size=self.size)
        params["peers"] = self.peers

        print("\nRunning...")
        self.send_to_workers({"t": P.C_RUN, "params": params})
        self._start_rank0(agg, params)
        if not agg.wait_all_done(timeout=600):
            print("[TIMEOUT] demo did not finish; done=%s" % sorted(agg.done))
        if agg.errors:
            print("Errors:", agg.errors)
        self.agg = None
        return agg

    def _start_rank0(self, agg, params):
        def work():
            try:
                rt = _Rank0Rt(self, agg)
                result = collectives_dispatch.run(rt, params)
                if params.get("payload"):
                    agg.worker_done(0)   # no multi-MB result over the wire
                else:
                    agg.worker_done(0, value=result)
            except Exception as e:  # noqa: BLE001
                agg.worker_done(0, error=str(e))
        threading.Thread(target=work, daemon=True).start()

    def on_round_ready(self, agg, rnd):
        """Rank-0 barrier leg: show the global view of round rnd, then let the
        class proceed. Runs on the teacher's rank-0 thread."""
        agg.wait_round_events(rnd, need=self.size - 1, timeout=30)
        remote = agg.round_events.get(rnd, [])
        local = [e.to_dict() for e in self.events0.by_round(rnd)]
        self._print_round(rnd, remote + local)
        if not self.auto:
            input("\nPress ENTER for next round.")

    def _print_round(self, rnd, events):
        print("\n---------------- Round %d ----------------" % rnd)
        seen = set()
        for e in events:
            key = (e.get("source"), e.get("destination"))
            if key in seen:
                continue
            seen.add(key)
            dt = e.get("transfer_time_ms", 0)
            print("Rank %d -> Rank %d   %d B   %.3f ms" %
                  (e["source"], e["destination"], e.get("payload_bytes", 0), dt))

    def shutdown(self):
        self.closed = True
        try:
            self.send_to_workers({"t": P.C_SHUTDOWN})
        except Exception:
            pass
        try:
            self.server.close()
        except OSError:
            pass
        self.transport.close()


class _Rank0Rt:
    """Minimal runtime facade so collectives code is identical for rank 0."""

    def __init__(self, coord, agg):
        self.rank = 0
        self.size = coord.size
        self.name = "teacher"
        self.comm = coord.comm0
        self.mode = "teaching" if agg.teaching else "performance"
        self.events = coord.events0
        self._coord = coord
        self._agg = agg

    def sync_round(self, rnd):
        if self.mode == "teaching":
            BarrierMod.barrier(
                self.comm, rnd,
                on_root_ready=lambda r: self._coord.on_round_ready(self._agg, r))
        return True


def _pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


def _params_for(algo, mode, payload=0, vector_len=0):
    p = {"algorithm": algo, "mode": mode, "op": "sum"}
    if payload:
        p.update(payload=payload, fmt="raw", op="xor")
    elif vector_len:
        p.update(vector_len=vector_len, fmt="i32")
    else:
        p.update(n_value=7, fmt="i32")
    return p


def _fmt_value(v):
    if isinstance(v, list):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def run_demo(coord, algo, mode, payload=0, vector_len=0, show=True):
    if algo in ("tree_reduce", "tree_allreduce") and not _pow2(coord.size):
        print("[skip] %s requires a power-of-two world size (got %d)" %
              (algo, coord.size))
        return None
    if not payload and vector_len == 0:
        if algo == "naive_reduce":
            vector_len = 1
        elif algo == "ring_allreduce":
            vector_len = coord.size
        else:
            vector_len = 4
    if algo == "ring_allreduce" and vector_len % coord.size:
        print("[skip] ring needs vector length divisible by size")
        return None

    params = _params_for(algo, mode, payload=payload, vector_len=vector_len)
    print("\n== Demo: %s  mode=%s ==" % (algo, mode))
    if payload:
        print("payload: %d bytes (op=xor, fmt=raw)" % payload)
    t0 = time.time()
    agg = coord.run_demo(params, mode)
    dt = time.time() - t0
    if show and agg:
        print("\nCollective complete.  Total wall time: %.3f s" % dt)
        for r in sorted(agg.results):
            print("  Rank %d final = %s" % (r, _fmt_value(agg.results[r])))
    if agg and agg.errors:
        print("Errors:", agg.errors)
    return agg


MENU = [
    ("Point-to-Point Send / Recv", "ping_pong"),
    ("Naive Reduce", "naive_reduce"),
    ("Naive AllReduce", "naive_allreduce"),
    ("Tree Reduce", "tree_reduce"),
    ("Tree AllReduce", "tree_allreduce"),
    ("Ring AllReduce", "ring_allreduce"),
    ("Performance Benchmark", "benchmark"),
    ("Network Check", "network"),
    ("Exit", "exit"),
]


def run_benchmark(coord):
    algs = ["naive_allreduce", "tree_allreduce", "ring_allreduce"]
    sizes = [8, 1024, 16 * 1024, 256 * 1024, 4 * 1024 * 1024]
    print("\n========================================\nCollective Benchmark\n"
          "World Size: %d\n========================================" % coord.size)
    header = "Size".ljust(10) + "".join(a.replace("_", " ").ljust(18)
                                        for a in algs)
    print(header)
    print("-" * len(header))
    for sz in sizes:
        row = str(sz).ljust(10)
        for a in algs:
            if a == "ring_allreduce" and sz % coord.size:
                row += "n/a".ljust(18)
                continue
            t0 = time.time()
            run_demo(coord, a, "performance", payload=sz, show=False)
            row += ("%7.2f ms" % ((time.time() - t0) * 1000)).ljust(18)
        print(row)


def wait_ready(coord, size):
    print("\nWaiting for workers (%d/%d)..." % (len(coord.workers) + 1, size))
    while len(coord.workers) + 1 < size:
        time.sleep(0.4)
    print("Ready: %d / %d\n" % (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--advertise", default="")
    ap.add_argument("--auto", action="store_true", help="no ENTER prompts")
    ap.add_argument("--demo", default="", help="run one demo then exit")
    ap.add_argument("--mode", default="teaching")
    ap.add_argument("--payload", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    if args.size < 1:
        sys.exit("world size must be >= 1")

    print("========================================\nMiniMPI Classroom\n"
          "========================================")
    coord = Coordinator(args.size, args.host, args.port, args.advertise,
                        auto=args.auto)
    print("Coordinator: %s:%d" % (coord.advertise, coord.port))
    print("Rank 0: Teacher (advertised %s:%d)" % (coord.advertise,
                                                  coord.transport.port))
    print("Expected World Size: %d" % args.size)

    wait_ready(coord, args.size)
    coord.connectivity_check()

    if args.benchmark:
        run_benchmark(coord)
        coord.shutdown()
        return
    if args.demo:
        run_demo(coord, args.demo, args.mode, payload=args.payload)
        coord.shutdown()
        return

    while True:
        print("\n========================================\nMiniMPI Classroom\n"
              "========================================")
        print("World Size: %d   Workers Ready: %d / %d\n"
              % (args.size, args.size, args.size))
        for i, (label, _) in enumerate(MENU, 1):
            print("%d. %s" % (i, label))
        try:
            choice = input("\nSelect: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if choice in ("9", "exit"):
            break
        if choice in ("8", "network"):
            coord.connectivity_check()
            continue
        if choice in ("7", "benchmark"):
            run_benchmark(coord)
            continue
        try:
            label, algo = MENU[int(choice) - 1]
        except (ValueError, IndexError):
            print("invalid choice")
            continue
        mode = args.mode
        if mode not in ("teaching", "performance"):
            mode = input("Mode (teaching/performance) [teaching]: ").strip() or "teaching"
        vector = coord.size if algo == "ring_allreduce" else 0
        run_demo(coord, algo, mode, vector_len=vector)

    coord.shutdown()
    print("Bye.")


if __name__ == "__main__":
    main()
