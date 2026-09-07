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
from minimpi.metrics import EventLog, fmt_bytes, now_ns      # noqa: E402


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
        self.timing = {}            # rnd -> {rank: {"send": ms, "work": ms}}
        self.active = False

    def worker_round_done(self, rank, rnd, events, send_ms=None, work_ms=None):
        with self.cv:
            self.round_reports.setdefault(rnd, set()).add(rank)
            self.round_events.setdefault(rnd, []).extend(events)
            if send_ms is not None:
                self.timing.setdefault(rnd, {})[rank] = {"send": send_ms,
                                                         "work": work_ms}
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
        # teacher-clock timings (ns)
        self._timing = {"start": None, "gather": {}, "release": {},
                        "col_start": None, "col_end": None}

        self.transport = PeerTransport("teacher", bind_host=self.host)
        self.peers[0]["port"] = self.transport.port
        self.events0 = EventLog()
        self.comm0 = Communicator(self.transport, 0, size, events=self.events0)
        self.rank0 = _Rank0Rt(self)          # persistent Rank 0 participant

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
            agg.worker_round_done(rank, int(m["rnd"]), m.get("events", []),
                                  send_ms=m.get("send_ms"),
                                  work_ms=m.get("work_ms"))
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
        self.events0.clear()   # rank0 events belong to this RUN only
        params = dict(params, mode=mode, size=self.size)
        params["peers"] = self.peers
        params["value0"] = params.pop("value0", 1)

        print("\nRunning...")
        self.send_to_workers({"t": P.C_RUN, "params": params})
        self._start_rank0(agg, params)
        if not agg.wait_all_done(timeout=600):
            print("[TIMEOUT] demo did not finish; done=%s" % sorted(agg.done))
        # Collective Time ends when ALL ranks reported C_DONE (the collective
        # really finished everywhere) — student input / RUN control / Start
        # Barrier waiting all happened before `col_start` and stay outside it.
        self._timing["col_end"] = now_ns()
        if agg.errors:
            print("Errors:", agg.errors)
        self.agg = None
        return agg

    def _start_rank0(self, agg, params):
        def work():
            try:
                rt = self.rank0
                rt._agg = agg
                rt.mode = "teaching" if agg.teaching else "performance"
                result = collectives_dispatch.run(rt, params,
                                                  value=params.get("value0"))
                if params.get("payload"):
                    agg.worker_done(0)   # no multi-MB result over the wire
                else:
                    agg.worker_done(0, value=result)
            except Exception as e:  # noqa: BLE001
                agg.worker_done(0, error=str(e))
        threading.Thread(target=work, daemon=True).start()

    def on_round_ready(self, agg, rnd):
        """Rank-0 barrier leg: show the global view of round rnd + timing,
        then let the class proceed (ENTER). The gather time is already fixed
        before this pause, so ENTER never enters round timing."""
        agg.wait_round_events(rnd, need=self.size - 1, timeout=30)
        # teaching-barrier traffic is kind="barrier" and hidden; only the
        # collective's real algorithm communication is shown.
        remote = [e for e in agg.round_events.get(rnd, [])
                  if e.get("kind") != "barrier"]
        local = [e.to_dict() for e in self.events0.by_round(rnd)
                 if e.kind != "barrier"]
        self._print_round(rnd, remote + local)

        # ---- Timing (Audit 7/8/10) ------------------------------------
        print("\nTiming")
        timing = agg.timing.get(rnd, {})
        # Rank 0 (teacher) timings from its own World snapshot
        if self.rank0.comm_world is not None:
            send0, _work0 = self.rank0.comm_world.snapshot_ms()
            timing.setdefault(0, {"send": send0})
        for rk in range(self.size):
            t = timing.get(rk)
            send = None if t is None else t.get("send")
            send_txt = "N/A" if send is None else "%.2f ms" % send
            print("  Rank %d send finished %s" % (rk, send_txt))
        rms = self.round_ms(rnd)
        print("Round Finished At: %s" %
              ("%.2f ms" % rms if rms is not None else "N/A"))
        print("  (compare My Send Finished At per rank vs the round total)")
        print("\nAll ranks completed Round %d." % rnd)

        import os as _os, time as _time
        if not self.auto:
            input("\nPress ENTER for next round.")
        elif _os.environ.get("MINIMPI_TEACH_PAUSE"):
            _time.sleep(float(_os.environ["MINIMPI_TEACH_PAUSE"]))
        # gather time was fixed BEFORE the pause, so round timing excludes it
        self.record_release(rnd)   # baseline for the NEXT round (after pause)

    def _print_round(self, rnd, events):
        print("\n--------------- Round %d ----------------" % rnd)
        seen = set()
        for e in events:
            src, dst = e.get("source"), e.get("destination")
            key = (src, dst)
            if key in seen:
                continue
            seen.add(key)
            dt = e.get("transfer_time_ms", 0)
            print("Rank %d -> Rank %d    %s   %.3f ms" %
                  (src, dst, fmt_bytes(e.get("payload_bytes", 0)), dt))

    # ---- round timing (teacher clock; teaching pauses excluded) ---------
    def record_start(self):
        """Start Barrier: EVERY rank has arrived (gather complete, release
        not yet sent). This is the baseline for Round 1 AND for the
        Performance-mode Collective Time — anything before it (C_RUN send,
        student input, vector build, waiting for slow ranks) is NOT part of
        the algorithm measurement."""
        ns = now_ns()
        self._timing["start"] = ns
        self._timing["col_start"] = ns

    def collective_ms(self):
        """Performance Collective Time = Start Barrier complete (all ranks
        ready) -> all ranks done. None if never started / never finished."""
        cs, ce = self._timing["col_start"], self._timing["col_end"]
        if cs is None or ce is None:
            return None
        return (ce - cs) / 1e6

    def record_gather(self, rnd):
        self._timing["gather"][rnd] = now_ns()

    def record_release(self, rnd):
        self._timing["release"][rnd] = now_ns()

    def round_ms(self, rnd):
        g = self._timing["gather"].get(rnd)
        if g is None:
            return None
        base = self._timing["release"].get(rnd - 1,
                                            self._timing["start"])
        if base is None:
            return None
        return (g - base) / 1e6

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
    """Rank 0 participant: created once per teacher session and reused by
    every RUN, so Teacher uses one session-level COMM_WORLD like workers."""

    def __init__(self, coord):
        self.rank = 0
        self.size = coord.size
        self.comm = coord.comm0
        self.comm_world = None           # MPI.COMM_WORLD (set once)
        # Start Barrier fires this when every rank has ARRIVED (gather done,
        # before rank 0 sends the release broadcast) — record_start() then
        # happens exactly at "all ranks ready", never late.
        self.on_start_gathered = lambda rnd: coord.record_start()
        self.events = coord.events0
        self.mode = "performance"
        self._coord = coord
        self._agg = None                 # current RUN aggregator

    def sync_round(self, rnd):
        if self.mode == "teaching":
            BarrierMod.barrier(
                self.comm, rnd,
                on_root_gathered=lambda r: self._coord.record_gather(r),
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
    """Show ONE number (the vector is [base]*N, so element 0 carries the
    result). Avoids printing thousands of elements on the terminal."""
    if isinstance(v, list):
        return "%d" % v[0] if len(v) else "[]"
    return str(v)


def run_demo(coord, algo, mode, payload=0, vector_len=0, show=True, value0=1):
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
        vector_len = ((vector_len + coord.size - 1) // coord.size) * coord.size
        print("(ring requires Data Size divisible by World Size; "
              "using Data Size %d)" % vector_len)

    params = _params_for(algo, mode, payload=payload, vector_len=vector_len)
    params["value0"] = value0
    print("\n== Demo: %s  mode=%s ==" % (algo, mode))
    if payload:
        # Benchmark payload is LOCAL per rank: every rank holds `payload`
        # bytes; how much a single message carries depends on the algorithm
        # (e.g. ring sends chunks of N/P) and is shown by the round events.
        print("Local Payload per Rank: %s  (op=xor, fmt=raw)" % fmt_bytes(payload))
    else:
        params["data_size"] = vector_len
        print("Data Size: %d elements  =  Local Data per Rank %s (int32)"
              % (vector_len, fmt_bytes(vector_len * P.ELEMENT_BYTES)))
        if algo == "ring_allreduce":
            print("  (ring: one message = one chunk = Data Size / World Size "
                  "elements; per-message bytes shown by the events)")
    t0 = time.time()
    agg = coord.run_demo(params, mode)
    dt = time.time() - t0
    if show and agg:
        cms = coord.collective_ms()
        print("\nCollective complete.")
        print("  Collective Time: %s  (Start Barrier complete -> all ranks done)"
              % ("%.2f ms" % cms if cms is not None else "N/A"))
        print("  Session wall time: %.3f s  (incl. student input / control / UI)"
              % dt)
        for r in sorted(agg.results):
            print("  Rank %d final = %s" % (r, _fmt_value(agg.results[r])))
        if algo in ("naive_reduce", "tree_reduce"):
            print("\n(note) %s: only Rank 0 (root) received the reduced result; "
                  "other ranks keep their own local values." % algo)
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
          "World Size: %d\n"
          "rows = Local Payload per Rank (bytes; each rank holds that much)\n"
          "========================================" % coord.size)
    header = "Local Payload".ljust(15) + "".join(a.replace("_", " ").ljust(18)
                                                 for a in algs)
    print(header)
    print("-" * len(header))
    for sz in sizes:
        row = str(sz).ljust(15)
        for a in algs:
            if a == "ring_allreduce" and sz % coord.size:
                row += "n/a".ljust(18)
                continue
            run_demo(coord, a, "performance", payload=sz, show=False)
            cms = coord.collective_ms()      # Start Barrier -> all ranks done
            row += ("%7.2f ms" % cms if cms is not None else "n/a").ljust(18)
        print(row)


def wait_ready(coord, size):
    print("\nWaiting for ranks (%d/%d)..." % (len(coord.workers) + 1, size))
    while len(coord.workers) + 1 < size:
        time.sleep(0.4)
    print("MPI World Ready: %d / %d ranks (%d student workers)\n"
          % (size, size, size - 1))


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
    ap.add_argument("--data-size", type=int, default=0,
                    help="elements per rank (i32); menu default 16")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    if args.size < 1:
        sys.exit("world size must be >= 1")

    from minimpi import mpi as M

    M.Init()                          # one MPI session for this process (Rank 0)
    coord = Coordinator(args.size, args.host, args.port, args.advertise,
                        auto=args.auto)
    try:
        print("========================================\nMiniMPI Classroom\n"
              "========================================")
        print("Coordinator: %s:%d" % (coord.advertise, coord.port))
        print("Rank 0: Teacher (a real collective participant)")
        print("Expected World Size: %d" % args.size)

        wait_ready(coord, args.size)
        coord.connectivity_check()

        # Rank 0 COMM_WORLD is created ONCE and reused by every RUN.
        comm0 = M.World(coord.rank0)
        coord.rank0.comm_world = comm0
        M.COMM_WORLD = comm0

        if args.benchmark:
            run_benchmark(coord)
            return
        if args.demo:
            run_demo(coord, args.demo, args.mode, payload=args.payload,
                     vector_len=args.data_size)
            return

        while True:
            print("\n========================================\nMiniMPI Classroom\n"
                  "========================================")
            print("World Size: %d   MPI World Ready: %d / %d ranks\n"
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

            # --- interactive run setup: data size -> rank0 value -> mode ---
            print("\nAlgorithm: %s" % algo)
            ds_raw = input("\nData Size (elements, int32 = %d B/elem, default 16):\n> "
                           % P.ELEMENT_BYTES).strip()
            try:
                ds = int(ds_raw) if ds_raw.isdigit() and int(ds_raw) > 0 else 16
            except ValueError:
                ds = 16
            print("\nMode:")
            print("1. Teaching")
            print("2. Performance")
            m = input("\nSelect: ").strip()
            mode = "teaching" if m != "2" else "performance"
            v = input("\nYour value (Rank 0, default 1):\n> ").strip()
            try:
                v0 = int(v)
            except ValueError:
                v0 = 1
            run_demo(coord, algo, mode, vector_len=ds, value0=v0)

        print("Bye.")
    finally:
        coord.shutdown()              # notify workers + close control server
        M.Finalize()                  # every exit path ends the MPI session


if __name__ == "__main__":
    main()
