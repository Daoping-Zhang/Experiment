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
    python3 teacher.py --size 4 --demo recursive_doubling_allreduce --mode teaching --auto
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
from minimpi import teaching as T                        # noqa: E402
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
                        "col_start": None, "col_end": None,
                        "rank0": {}, "ready": {}}

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
        value0 = params.get("value0", 1)
        # value0 is ONLY rank 0's own input — workers read theirs themselves.
        # (In benchmark cases value0 may be raw BYTES, which must never go
        # over the JSON control channel.)
        params.pop("value0", None)

        print("\nRunning...")
        # Rank-0 (teacher) per-run teaching state: real initial data + a
        # per-round semantic context, so the Rank 0 Local View shows the
        # teacher's ACTUAL local data, not a UI-side mock.
        self.rank0._alg = params.get("algorithm", "")
        self.rank0._local_ctx = None
        if mode == "teaching":
            ds = int(params.get("data_size") or params.get("vector_len") or 0)
            if params.get("fmt", "i32") == "i32" and ds > 0:
                v0 = int(value0) if not isinstance(value0, (bytes, bytearray)) \
                    else 1
                self.rank0._local_ctx = T.RoundCtx(self.rank0._alg, self.size,
                                                   0, [v0] * ds)
        params["value0"] = value0      # rank-0 dispatch arg (local only)
        self.send_to_workers({"t": P.C_RUN, "params": _without_value0(params)})
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
        """Rank-0 barrier leg after every rank arrived (Round Finished At is
        already fixed by record_gather). Shows the fixed 4-part teaching view,
        then pauses and releases:

            1. Global Communication     2. Rank 0 Local View
            3. Timing                   4. Teacher Control
        """
        agg.wait_round_events(rnd, need=self.size - 1, timeout=30)
        alg = getattr(self.rank0, "_alg", "")
        P = self.size
        total = T.total_rounds(alg, P)

        # ---- round header (Algorithm / Phase / Round x / y) ---------------
        phase_label, _op = T.phase_of(alg, P, rnd)
        print("\n" + "=" * 50)
        print(T.pretty_algorithm(alg))
        print(phase_label)
        print("Round %d / %d" % (rnd, total))
        print("=" * 50)

        # ---- 1. Global Communication --------------------------------------
        remote = [e for e in agg.round_events.get(rnd, [])
                  if e.get("kind") != "barrier"]
        local = [e.light_dict() for e in self.events0.by_round(rnd)
                 if e.kind != "barrier"]
        events = remote + local
        print("\nGLOBAL COMMUNICATION")
        if alg == "recursive_doubling_allreduce":
            # pairwise exchange: show "a ⇄ b" when both directions happened
            edges = {(e.get("source"), e.get("destination")): e
                     for e in events}
            printed = set()
            for (a, b) in sorted(edges):
                if (a, b) in printed:
                    continue
                if (b, a) in edges:
                    line = "Rank %d \u21c4 Rank %d    %s each direction" % (
                        min(a, b), max(a, b),
                        fmt_bytes(edges[(a, b)].get("payload_bytes", 0)))
                    printed.add((a, b)); printed.add((b, a))
                else:
                    line = "Rank %d -> Rank %d    %s" % (
                        a, b, fmt_bytes(edges[(a, b)].get("payload_bytes", 0)))
                    printed.add((a, b))
                print(line)
        else:
            seen = set()
            for e in events:
                src, dst = e.get("source"), e.get("destination")
                if (src, dst) in seen:
                    continue
                seen.add((src, dst))
                line = "Rank %d -> Rank %d" % (src, dst)
                if alg == "ring_allreduce":
                    chunk = T.ring_chunk_index(src, P, rnd, "send")
                    line += "    Chunk %d" % chunk
                line += "    %s" % fmt_bytes(e.get("payload_bytes", 0))
                print(line)

        # ---- OPERATIONS (data-changing operations; no vector dumps) -------
        print("\nOPERATIONS")
        if alg == "recursive_doubling_allreduce":
            partners = {}
            for e in events:
                if e.get("side") == "send":
                    partners.setdefault(e.get("source"), set()).add(
                        e.get("destination"))
                elif e.get("side") == "recv":
                    partners.setdefault(e.get("destination"), set()).add(
                        e.get("source"))
            for rk in sorted(partners):
                who = " (Teacher)" if rk == 0 else ""
                p = ", ".join(str(x) for x in sorted(partners[rk]))
                print("Rank %d%s: Exchange with %s + SUM" % (rk, who, p))
        else:
            ops_by = {}
            for e in events:
                if e.get("side") == "recv":
                    ops_by.setdefault(e.get("destination"),
                                      []).append(e.get("source"))
            if not ops_by:
                print("None")
            else:
                _lbl, op = T.phase_of(alg, P, rnd)
                opl = ("+ SUM" if op == "sum"
                       else "+ COPY" if op == "copy" else "")
                for rk in sorted(ops_by):
                    who = " (Teacher)" if rk == 0 else ""
                    peers = ", ".join(str(p) for p in sorted(set(ops_by[rk])))
                    print("Rank %d%s: Recv from %s %s" % (rk, who, peers, opl))

        # ---- 2. Rank 0 Local View (real rank-0 execution state) -----------
        print("\nRANK 0 LOCAL VIEW")
        ctx = getattr(self.rank0, "_local_ctx", None)
        evs0 = [e for e in self.events0.by_round(rnd) if e.kind == "algorithm"]
        if ctx is None:
            print("(scalar/payload run — no vector state to show)")
        else:
            view = T.describe_round(ctx, rnd, evs0)
            print(T.local_view_text(view))
        if self.rank0.comm_world is not None:
            lines = T.local_timeline_text(self.rank0.comm_world.local_timings())
            if lines:
                print("\nLOCAL TIMELINE\n" + "\n".join(lines))

        # ---- 3. Synchronization Window (Rank-0 single clock) ---------------
        # Arrivals are stamped on the ONE rank-0 clock when each rank's
        # barrier token reached rank 0 (tiny token transmission included).
        # We only report how UNEVEN the arrivals were: first-to-arrive is
        # +0.00 ms, each other rank's offset is from that first arrival, and
        # waited = window - offset. No 'whole round finished' claim — that
        # mixes baselines (release/scheduling/token observation) with the
        # student local clocks.
        rd = (self._timing.get("ready", {}).get(rnd) or {})
        if len(rd) >= 2:
            first = min(rd.values())
            last = max(rd.values())
            window = (last - first) / 1e6
            print("\nSYNCHRONIZATION — Rank 0 Observation")
            for rk, ns in sorted(rd.items(), key=lambda kv: kv[1]):
                off = (ns - first) / 1e6
                wait = (last - ns) / 1e6
                print("Rank %d arrived: +%.2f ms | waited %.2f ms"
                      % (rk, off, wait))
            print("\nSynchronization Window: %.2f ms" % window)

        # ---- 4. Teacher Control -------------------------------------------
        last = rnd >= total
        print("\nAll ranks finished Round %d / %d." % (rnd, total))
        if not self.auto:
            input("\n[ENTER] %s" % ("Close Collective" if last
                                    else "Next Round"))
        elif os.environ.get("MINIMPI_TEACH_PAUSE"):
            time.sleep(float(os.environ["MINIMPI_TEACH_PAUSE"]))
        # gather time was fixed BEFORE the pause, so round timing excludes it
        self.record_release(rnd)   # baseline for the NEXT round (after pause)

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

    # ---- barrier-arrival observation (ALL on the teacher's own clock) ----
    def note_rank0_ready(self, rnd):
        """Rank 0 finished its local algorithm work for this round."""
        self._timing["rank0"][rnd] = now_ns()

    def note_rank_arrival(self, rnd, src, arrival_ns):
        """Rank 0 observed `src`'s barrier token arrive (rank-0 clock)."""
        if arrival_ns is None:
            arrival_ns = now_ns()
        self._timing["ready"].setdefault(rnd, {})[src] = arrival_ns

    def note_gather_ready(self, rnd):
        """All barrier tokens gathered: record gather time and freeze the
        ready map (rank 0's own ready is its local-work-done instant)."""
        self.record_gather(rnd)
        rd = self._timing["ready"].setdefault(rnd, {})
        r0 = self._timing["rank0"].get(rnd)
        if r0 is None:                    # safety fallback
            r0 = now_ns()
        rd[0] = r0

    def _round_base_ns(self, rnd):
        rel = self._timing["release"].get(rnd - 1)
        if rel is not None:
            return rel
        return self._timing["start"]

    def ready_ms(self, rnd, rank):
        """Rank Ready At (ms from Round Start) on the teacher clock."""
        base = self._round_base_ns(rnd)
        ns = (self._timing.get("ready", {}).get(rnd) or {}).get(rank)
        if base is None or ns is None:
            return None
        return (ns - base) / 1e6

    def ready_table(self, rnd):
        """[(rank, ready_ms, wait_ms)] for this round, teacher clock."""
        rd = self._timing.get("ready", {}).get(rnd) or {}
        rows = []
        ms = {}
        for rk, ns in rd.items():
            m = self.ready_ms(rnd, rk)
            if m is not None:
                ms[rk] = m
        whole = max(ms.values()) if ms else None
        for rk in sorted(ms):
            rows.append((rk, ms[rk], 0.0 if whole is None
                         else max(0.0, whole - ms[rk])))
        return rows

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
        # before rank 0 sends the release broadcast). Teaching mode: make the
        # barrier visible, let the teacher ENTER, THEN record_start() — so the
        # pause never enters Round 1 / Collective Time.
        self.on_start_gathered = self._on_start_gathered
        self.events = coord.events0
        self.mode = "performance"
        self._coord = coord
        self._agg = None                 # current RUN aggregator
        self._alg = ""                   # current RUN algorithm
        self._local_ctx = None           # per-run RoundCtx (rank-0 local view)
        # World.sync_round() calls this the moment rank 0's local algorithm
        # work for the round finished (before the round barrier) — the
        # "rank 0 ready" instant on the teacher's own clock.
        self.on_local_work_done = lambda rnd: coord.note_rank0_ready(rnd)

    def _on_start_gathered(self, rnd):
        """All ranks have arrived at the Start Barrier (release not sent)."""
        coord = self._coord
        if self.mode == "teaching":
            print("\n" + "=" * 50)
            print("Start Barrier")
            print("=" * 50)
            print("\nWaiting for all ranks...")
            print("\nAll ranks ready.")
            if not coord.auto:
                input("\n[ENTER] Start Collective")
            elif os.environ.get("MINIMPI_TEACH_PAUSE"):
                time.sleep(float(os.environ["MINIMPI_TEACH_PAUSE"]))
        coord.record_start()              # baseline AFTER any teacher pause

    def sync_round(self, rnd):
        if self.mode == "teaching":
            coord = self._coord
            BarrierMod.barrier(
                self.comm, rnd,
                on_root_arrival=lambda src, ns: coord.note_rank_arrival(
                    rnd, src, ns),
                on_root_gathered=lambda r: coord.note_gather_ready(r),
                on_root_ready=lambda r: coord.on_round_ready(self._agg, r))
        return True


def _pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


def _without_value0(params):
    """Control-plane form of RUN params: rank 0's own value never crosses
    the JSON control channel (workers input their own)."""
    d = dict(params)
    d.pop("value0", None)
    return d


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


def run_demo(coord, algo, mode, payload=0, vector_len=0, show=True,
             value0=1, silent=False, kind=None):
    if algo == "recursive_doubling_allreduce" and not _pow2(coord.size):
        print("[skip] %s requires a power-of-two world size (got %d)" %
              (algo, coord.size))
        return None
    if algo in ("tree_reduce",) and not _pow2(coord.size):
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
    if kind is not None:
        params["kind"] = kind
    if not silent:
        print("\n== Demo: %s  mode=%s ==" % (algo, mode))
    if payload:
        # Benchmark payload is LOCAL per rank: every rank holds `payload`
        # bytes (Data Size = payload / 4 elements); how much a single
        # message carries depends on the algorithm (ring sends chunks).
        if not silent:
            print("Local Payload per Rank: %s  (op=xor, fmt=raw)"
                  % fmt_bytes(payload))
    else:
        params["data_size"] = vector_len
        if not silent:
            print("Data Size: %d elements  =  Local Data per Rank %s (int32)"
                  % (vector_len, fmt_bytes(vector_len * P.ELEMENT_BYTES)))
            if algo == "ring_allreduce":
                print("  (ring: one message = one chunk = Data Size / World "
                      "Size elements; per-message bytes shown by the events)")
    t0 = time.time()
    agg = coord.run_demo(params, mode)
    dt = time.time() - t0
    if show and agg:
        if mode == "teaching" and not payload:
            # per-rank final is a SINGLE number (never a vector dump)
            for r in sorted(agg.results):
                print("Rank %d final = %s" % (r, _fmt_value(agg.results[r])))
            _print_teaching_complete(agg, algo)
        else:
            cms = coord.collective_ms()
            print("\nCollective complete.")
            print("  Collective Time: %s  (Start Barrier complete -> all ranks done)"
                  % ("%.2f ms" % cms if cms is not None else "N/A"))
            print("  Session wall time: %.3f s  (incl. student input / control / UI)"
                  % dt)
            for r in sorted(agg.results):
                print("  Rank %d final = %s" % (r, _fmt_value(agg.results[r])))
    if agg and agg.errors:
        print("Errors:", agg.errors)
    return agg


BENCH_ALGORITHMS = ["naive_allreduce", "recursive_doubling_allreduce",
                    "ring_allreduce"]
# bytes per rank per case — all divisible by World Size (int32 elements):
#   16 B=4, 1 KB=256, 16 KB=4096, 256 KB=65536, 4 MB=1048576, 16 MB=4194304
BENCH_SIZES = [16, 1024, 16 * 1024, 256 * 1024,
               4 * 1024 * 1024, 16 * 1024 * 1024]
BENCH_RUNS = 3


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _elem_rows():
    """bytes -> (int32 elements)"""
    return {b: b // P.ELEMENT_BYTES for b in BENCH_SIZES}


def run_benchmark(coord):
    """Performance Benchmark session: each rank enters ONE value, then 3
    algorithms x 6 sizes x 3 runs run automatically; results are medians."""
    print("\n========================================\n"
          "Performance Benchmark\n"
          "========================================")
    print("\nAlgorithms:")
    for a in BENCH_ALGORITHMS:
        print("- %s" % T.pretty_algorithm(a))
    print("\nMessage Sizes (Local Data per Rank):")
    for b in BENCH_SIZES:
        print("  %s  = %d elements" % (fmt_bytes(b), b // P.ELEMENT_BYTES))
    print("\nRuns per case:\n%d" % BENCH_RUNS)
    print("\nEach rank will enter one integer once.")
    print("The same local value will be reused for all benchmark cases.")

    if sys.stdin.isatty():          # interactive menu: rank 0 types once
        raw = input("\nYour benchmark value (Rank 0, default 1):\n> ").strip()
        try:
            v0 = int(raw)
        except ValueError:
            v0 = 1
    else:
        v0 = 1
    # Benchmark Setup is NOT a collective run: one integer per rank, cached
    # for the whole session (no per-case inputs, no zero Data Size).
    coord.send_to_workers({"t": P.C_RUN, "params": {"kind": "benchmark_setup",
                                                    "mode": "performance"}})

    rows = {a: {} for a in BENCH_ALGORITHMS}
    total = len(BENCH_ALGORITHMS) * len(BENCH_SIZES) * BENCH_RUNS
    done = 0
    for a in BENCH_ALGORITHMS:
        for b in BENCH_SIZES:
            times = []
            # every real benchmark case is explicitly a benchmark_case (no
            # re-prompt on worker) and rank 0 carries its OWN payload bytes
            # derived from the value entered once at setup.
            payload0 = collectives_dispatch.make_benchmark_payload(v0, b)
            for k in range(BENCH_RUNS):
                agg = run_demo(coord, a, "performance", payload=b,
                               show=False, silent=True, kind="benchmark_case",
                               value0=payload0)
                cms = coord.collective_ms()
                ms = cms if cms is not None else 0.0
                times.append(ms)
                done += 1
                print("raw %s %d run%d %.3f ms  (%d / %d)" %
                      (a, b, k + 1, ms, done, total))
            rows[a][b] = _median(times)

    print("\n" + "=" * 78)
    print("Performance Benchmark Results")
    print("%d runs per case — median" % BENCH_RUNS)
    print("World Size: %d" % coord.size)
    print("=" * 78)
    cols = ["Local Data / Rank", "Naive", "Recursive Doubling", "Ring"]
    widths = [26, 10, 18, 12]
    header = "".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for b in BENCH_SIZES:
        line = ("%s  = %d elem" % (fmt_bytes(b), b // P.ELEMENT_BYTES)).ljust(
            widths[0])
        line += ("%7.2f ms" % rows["naive_allreduce"][b]).ljust(widths[1])
        line += ("%7.2f ms" % rows["recursive_doubling_allreduce"][b]
                 ).ljust(widths[2])
        line += ("%7.2f ms" % rows["ring_allreduce"][b]).ljust(widths[3])
        print(line)
    print("\nLower is better.")
    coord.send_to_workers({"t": P.C_RUN,
                           "params": {"kind": "benchmark_done",
                                      "mode": "performance"}})
    print("\nBenchmark Complete.\n\nReturning to Teacher menu...")


def _print_teaching_complete(agg, algo):
    """Teacher-side 'Collective Complete' block (teaching mode, §21)."""
    reduce_only = algo in ("naive_reduce", "tree_reduce")
    r0 = agg.results.get(0)
    print("\n" + "=" * 50)
    print("Collective Complete")
    print("=" * 50)
    if reduce_only:
        print("\nReduce Result")
        print("Rank 0 (root): %s" % _fmt_value(r0))
        print("Only the root owns the final reduced result.")
    elif r0 is not None:
        first = r0[0] if isinstance(r0, list) and r0 else r0
        print("\nResult: %s" % first)
        finals = {(v[0] if isinstance(v, list) and v else v)
                  for r, v in agg.results.items() if v is not None}
        if len(finals) == 1:
            print("All ranks received the same reduced result.")
        else:
            print("(per-rank finals: %s)"
                  % ", ".join("r%d=%s" % (r, _fmt_value(v))
                              for r, v in sorted(agg.results.items())))
    print()


MENU = [
    ("Point-to-Point Send / Recv", "ping_pong"),
    ("Naive Reduce", "naive_reduce"),
    ("Naive AllReduce", "naive_allreduce"),
    ("Tree Reduce", "tree_reduce"),
    ("Recursive Doubling AllReduce", "recursive_doubling_allreduce"),
    ("Ring AllReduce", "ring_allreduce"),
    ("Performance Benchmark", "benchmark"),
    ("Network Check", "network"),
    ("Exit", "exit"),
]


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
        print("\nTeacher Role")
        print("-" * 40)
        print("Classroom Coordinator")
        print("Global Communication View")
        print("MPI Participant: Rank 0")
        print("-" * 40)

        wait_ready(coord, args.size)
        # Warm the REAL persistent data-plane connections that collectives
        # will reuse (workers already warmed theirs after joining; teacher
        # warms its outbound legs). No message, no event, no timing.
        coord.transport.warm_to(list(range(1, coord.size)))
        print("MPI Data Plane Ready.")
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
