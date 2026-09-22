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
import queue
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
        self.aborted = None         # reason string if this RUN was aborted
        self.last_progress = time.monotonic()   # watchdog: last sign of life
        self.activity_fn = None     # optional counter of rank-0 data activity
        self._activity = 0
        self.last_seen = {}         # rank -> monotonic of its last sign of life
        self.arrived = {}           # rnd -> set(rank) seen at that round barrier

    def worker_round_done(self, rank, rnd, events, send_ms=None, work_ms=None):
        with self.cv:
            self.round_reports.setdefault(rnd, set()).add(rank)
            self.round_events.setdefault(rnd, []).extend(events)
            if send_ms is not None:
                self.timing.setdefault(rnd, {})[rank] = {"send": send_ms,
                                                         "work": work_ms}
            self.last_seen[rank] = time.monotonic()
            self.last_progress = self.last_seen[rank]
            self.cv.notify_all()

    def worker_done(self, rank, value=None, error=None):
        with self.cv:
            if error:
                self.errors[rank] = error
            self.done.add(rank)
            if value is not None:
                self.results[rank] = value
            self.last_seen[rank] = time.monotonic()
            self.last_progress = self.last_seen[rank]
            self.cv.notify_all()

    def heartbeat(self, rank):
        """A rank PROVED it is alive (control-plane heartbeat).

        Liveness only: a healthy rank blocked in a barrier must not look like
        run progress, or the stall watchdog would never fire for a frozen
        peer."""
        with self.cv:
            self.last_seen[rank] = time.monotonic()

    def silent_ranks(self, stale=None):
        """The suspects when the watchdog fires.

        A rank is a suspect when it completed nothing, reported nothing AND
        stopped sending heartbeats: that is a frozen student (asleep laptop,
        dead VM), which is the only way to tell it apart from a healthy rank
        that is simply blocked in a barrier. A rank that is merely waiting
        keeps heartbeating, so it is never dropped.
        """
        if stale is None:
            import minimpi.protocol as _P
            stale = _P.HEARTBEAT_STALE_S
        with self.cv:
            healthy = set(self.done)
            for ranks in self.round_reports.values():
                healthy |= ranks
            now = time.monotonic()
            return [r for r in range(1, self.size)
                    if r not in healthy
                    and now - self.last_seen.get(r, 0.0) >= stale]

    def touch(self, rank=None, rnd=None):
        """Something moved: the watchdog measures STALLS, not total wall time,
        so a teaching pause never looks like a hang while ranks keep reporting.

        `rank` (a rank that reached a round barrier) is remembered too, so the
        suspects named by a stalled RUN are the ranks that really went quiet —
        a healthy rank blocked in a barrier has said all it can say."""
        with self.cv:
            now = time.monotonic()
            self.last_progress = now
            if rank is not None:
                self.last_seen[rank] = now
                if rnd is not None:
                    self.arrived.setdefault(rnd, set()).add(rank)

    def wait_round_events(self, rnd, need, timeout=30):
        with self.cv:
            return self.cv.wait_for(
                lambda: len(self.round_reports.get(rnd, set())) >= need,
                timeout=timeout)

    def wait_all_done(self, timeout=600, idle=None):
        """Wait until every rank reported C_DONE (or the run was aborted).

        idle=None : plain wall-clock deadline (`timeout` seconds).
        idle=k    : watchdog on STALLS — give up only when nothing at all has
                    happened for k seconds (used by the classroom so that long
                    teaching pauses in front of the class are not mistaken for
                    a hang).
        """
        with self.cv:
            while True:
                if self.aborted is not None or len(self.done) >= self.size:
                    return True
                if idle is None:
                    if not self.cv.wait_for(
                            lambda: self.aborted is not None
                            or len(self.done) >= self.size,
                            timeout=timeout):
                        return False
                    continue
                if self.activity_fn is not None:
                    # rank 0's own data-plane traffic counts as progress even in
                    # performance mode, where workers never report mid-run.
                    # (Liveness is tracked separately, by heartbeats: a healthy
                    # rank blocked in a barrier must NOT look like progress.)
                    cur = self.activity_fn()
                    if cur != self._activity:
                        self._activity = cur
                        self.last_progress = time.monotonic()
                if time.monotonic() - self.last_progress >= idle:
                    return False
                self.cv.wait(0.5)

    def abort(self, reason):
        """End the current RUN early (a rank left / watchdog fired)."""
        with self.cv:
            if self.aborted is None:
                self.aborted = reason
                self.errors["run"] = reason
            self.cv.notify_all()

    def reset(self):
        with self.cv:
            self.round_reports.clear()
            self.round_events.clear()
            self.done.clear()
            self.results.clear()
            self.errors.clear()
            self.aborted = None
            self.last_progress = time.monotonic()
            self.last_seen = {}
            self.arrived = {}
            self.active = True


class Coordinator:
    """Control plane + dynamic roster.

    `size` is the CAPACITY (the world size the teacher opened, `--size`).
    The ACTIVE roster (rank 0 + the workers currently joined) may be smaller:
    ranks 1..N-1 are always kept contiguous, so when a worker leaves during
    the lesson every later rank is compacted down and re-welcomed before the
    next RUN. Each RUN therefore runs with the CURRENT active size.
    """

    def __init__(self, size, host, port, advertise, auto=False):
        self.size = size                 # capacity (max world size)
        self.host = host
        self.port = port
        self.advertise = advertise or (host if host != "0.0.0.0" else detect_ip())
        self.auto = auto

        self.workers = {}            # rank -> control socket (contiguous!)
        self.epid = {}               # rank -> {"host":..,"port":..} (data plane)
        self.peers = {0: {"host": self.advertise, "port": None}}
        # RLock: roster helpers are called from inside roster-critical
        # sections (a plain Lock deadlocked the joining thread against
        # itself the moment a helper re-read the roster size).
        self.lock = threading.RLock()
        # Control messages must never interleave: joins, leaves and runs are
        # handled by different threads, and two concurrent roster broadcasts
        # on one socket used to garble/reorder welcome messages — which left
        # workers with DIFFERENT world sizes and deadlocked the next run.
        self.ctrl_lock = threading.RLock()
        self.closed = False
        self.agg = None
        self.run_active = False      # True while a RUN is in flight
        self.roster_dirty = False    # roster changed during a RUN
        self.exclude_suspects = {}   # ranks to drop after a stalled RUN
        self.abort_epoch = 0         # bumped on every abort (cancels pauses)
        self.last_seen = {}          # rank -> monotonic of its last heartbeat
        self.pause_menu = threading.Event()   # ask a pause to open the roster
        self._roster_reset = True    # rank-0 peer cache needs rebuilding
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
        try:
            self.server.bind((host, port))
        except OSError as e:
            # Do not die with a traceback in front of the class: say what is
            # wrong and what to do (another program - Docker, a second teacher
            # - already holds this port).
            hint = ""
            if e.errno in (48, 98):          # EADDRINUSE (macOS / Linux)
                hint = ("\n[ERROR] port %d is already in use on this machine."
                        "\n[ERROR] Another program (or a second teacher) is "
                        "holding it. Start with a different port, e.g.:\n"
                        "        python3 teacher.py --size 4 --port %d"
                        % (port, port + 1))
            self.server.close()
            raise SystemExit("[ERROR] cannot listen on %s:%d (%s)%s"
                             % (host, port, e, hint))
        self.server.listen(64)
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True).start()

    # ---- roster bookkeeping ------------------------------------------------
    def active_size(self):
        """Current world size = rank 0 (teacher) + every joined worker."""
        with self.lock:
            return len(self.workers) + 1

    def liveness_age(self, rank):
        """Seconds since this rank's last heartbeat (None if never heard)."""
        with self.lock:
            t = self.last_seen.get(rank)
        return None if t is None else time.monotonic() - t

    def note_heartbeat(self, rank):
        with self.lock:
            self.last_seen[rank] = time.monotonic()

    def roster_snapshot(self):
        """(active world size, worker ranks) — one consistent read, so the
        menu / wait loop never iterate a dict another thread is editing."""
        with self.lock:
            return len(self.workers) + 1, sorted(self.workers)

    def _rebuild_roster_locked(self):
        """Keep worker ranks contiguous (1..N-1) after a join or a leave.

        A worker's data-plane endpoint never changes, only its rank, so the
        peers table is rebuilt from the per-worker endpoints.
        """
        pairs = sorted(self.workers.items())
        workers, epid, moves = {}, {}, []
        for new_rank, (old_rank, conn) in enumerate(pairs, start=1):
            workers[new_rank] = conn
            ep = self.epid.get(old_rank)
            if ep is not None:
                epid[new_rank] = ep
            if new_rank != old_rank:
                moves.append((old_rank, new_rank))
        self.workers, self.epid = workers, epid
        self.peers = {0: {"host": self.advertise, "port": self.transport.port}}
        self.peers.update({r: dict(ep) for r, ep in epid.items()})
        self._roster_reset = True     # rank 0 must drop its cached sockets
        return moves

    def _sync_rank0_peers(self):
        """Point rank 0's data plane at the CURRENT roster (keeping the warm
        sockets when nothing changed: benchmark timings must stay comparable)."""
        table = {r: (ep["host"], ep["port"])
                 for r, ep in self.peers.items() if r != 0}
        if self._roster_reset:
            self.transport.reset_peers(0, table)
            self._roster_reset = False
        else:
            self.transport.set_peers(0, table)

    def _welcome_msg(self, rank, size):
        # Same message type the JOIN path uses: a worker re-reads it at any
        # time to adopt a new rank / size / peer table.
        return {"t": P.C_WELCOME, "rank": rank, "size": size,
                "peers": self.peers,
                "version": P.MINIMPI_VERSION,
                "protocol": P.PROTOCOL_VERSION}

    def _apply_local_size(self, n=None):
        """Rank 0 (this process) adopts the current active world size.

        `comm0` is the raw communicator; the MPI.COMM_WORLD facade caches
        rank/size in plain attributes, and the start barrier reads the facade,
        so both must move together. Callers that already know the size (and
        may already hold the roster lock) pass it in."""
        if n is None:
            n = self.active_size()
        self.comm0.size = n
        cw = getattr(self.rank0, "comm_world", None)
        if cw is not None:
            cw.size = n
            cw.rank = 0
        return n

    def _broadcast_roster(self, size, reason, skip=None):
        """Tell the joined workers the current rank/size/peer table.

        `skip` is the rank that just received its welcome (a new joiner), so
        the log shows one line per actually-updated worker."""
        with self.ctrl_lock:
            with self.lock:
                targets = [(r, c, self._welcome_msg(r, size))
                           for r, c in sorted(self.workers.items())
                           if r != skip]
            for rank, conn, msg in targets:
                try:
                    P.ctrl_send(conn, msg)
                    print("[ROSTER] Rank %d re-welcomed (world size is now %d)"
                          % (rank, size))
                except OSError:
                    pass
        if reason:
            print("[ROSTER] %s  ->  World Size %d (%d student ranks)"
                  % (reason, size, size - 1))

    def _abort_run(self, reason, clean=True):
        """Unblock a RUN that can no longer finish.

        Workers get C_ABORT (their pending receives stop blocking); rank 0
        gives its own communicator a timeout so its local recv cannot hang.
        Collective algorithm code is untouched — this is a transport-level
        deadline, not a change to what collectives do.
        """
        print("\n[ABORT] %s" % reason)
        self.abort_epoch += 1            # cancels any ENTER pause in flight
        ag = self.agg
        if ag is not None:
            ag.abort(reason)
        self.comm0.default_timeout = 2.0
        self.transport.abort_pending(True)     # unblock rank 0's own recv
        try:
            self.send_to_workers({"t": P.C_ABORT, "why": reason,
                                  "clean": bool(clean)})
        except OSError:
            pass
        print("[ABORT] Collective aborted; the class returns to the menu "
              "(no rank stays blocked).")

    def _on_worker_left(self, rank, who="", why=""):
        """A worker disconnected: compact the roster and keep the class sane."""
        with self.ctrl_lock:
            with self.lock:
                running = self.run_active
                self.workers.pop(rank, None)
                self.epid.pop(rank, None)
            print("[LEAVE] Rank %d disconnected%s%s%s"
                  % (rank, ("  (during a RUN)" if running else ""),
                     ("  [%s]" % who) if who and who != "?" else "",
                     ("  (%s)" % why) if why else ""))
            if running:
                # Mid-RUN departure: the collective can never complete. (The
                # check is INSIDE the control lock, so the roster can never be
                # rewritten after a RUN has already been sent.)
                self.roster_dirty = True
                self._abort_run("Rank %d left during the run" % rank)
                # Remaining workers are re-welcomed once the RUN has unwound
                # (see _sync_roster_after_run).
                return
            size = self._apply_local_size()
            with self.lock:
                moves = self._rebuild_roster_locked()
            if moves:
                print("[ROSTER] rank compaction: %s"
                      % ", ".join("%d -> %d" % mv for mv in moves))
            self._broadcast_roster(size, "Rank %d left" % rank)

    def kick_rank(self, rank, why=""):
        """Remove one rank from the class ON PURPOSE.

        The teacher's manual tool: a student's machine is stuck, somebody has
        to leave, or a RUN is waiting for a rank that will never answer. The
        worker is told first (C_KICK) so it can report "removed by the teacher"
        instead of a socket error; then the control connection is shut down,
        which runs the normal LEAVE path (compaction + re-welcome, or an abort
        if a RUN is in flight). Works even for a frozen rank — it simply never
        reads the message and exits when it wakes up.
        """
        if rank == 0:
            print("[KICK] Rank 0 is the teacher — nothing to kick.")
            return False
        with self.lock:
            conn = self.workers.get(rank)
        if conn is None:
            print("[KICK] Rank %d is not in the class." % rank)
            return False
        why = why or "the teacher removed this rank from the class"
        print("\n[KICK] removing Rank %d — %s" % (rank, why))
        with self.ctrl_lock:
            try:
                P.ctrl_send(conn, {"t": P.C_KICK, "why": why})
            except OSError:
                pass
        time.sleep(0.2)                  # let the message arrive before we hang up
        try:
            conn.shutdown(socket.SHUT_RDWR)   # wakes its reader -> LEAVE path
        except OSError:
            pass
        try:
            conn.close()
        except OSError:
            pass
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self.lock:
                still = any(c is conn for c in self.workers.values())
            if not still:
                break
            time.sleep(0.05)
        print("[KICK] Rank %d removed. World size is now %d (%d student "
              "rank(s)); the class keeps running."
              % (rank, self.active_size(), self.active_size() - 1))
        return True

    def _exclude_suspects(self):
        """Drop the ranks a stalled RUN never heard from.

        A frozen student (asleep laptop, killed VM, dead Wi-Fi) leaves the
        socket open, so there is no LEAVE to detect. Rather than stalling every
        following RUN, the watchdog names the suspects and the class continues
        without them — exactly what the teacher would do by hand."""
        suspects, self.exclude_suspects = self.exclude_suspects, {}
        if not suspects:
            return
        for rank, why in sorted(suspects.items()):
            with self.lock:
                conn = self.workers.get(rank)
            if conn is None:
                continue
            print("[ROSTER] excluding Rank %d (%s) — the class continues "
                  "without it" % (rank, why))
            try:
                conn.shutdown(socket.SHUT_RDWR)   # its reader thread then
            except OSError:                       # reports the normal LEAVE
                pass
            try:
                conn.close()
            except OSError:
                pass
            with self.lock:
                self.workers.pop(rank, None)
                self.epid.pop(rank, None)
            self.roster_dirty = True

    def _sync_roster_after_run(self):
        """Apply roster changes that arrived while a RUN was in flight."""
        if not getattr(self, "roster_dirty", False):
            return
        self.roster_dirty = False
        size = self._apply_local_size()
        with self.lock:
            moves = self._rebuild_roster_locked()
        if moves:
            print("[ROSTER] rank compaction: %s"
                  % ", ".join("%d -> %d" % mv for mv in moves))
        self._broadcast_roster(size, "roster updated")

    def _accept_loop(self):
        while not self.closed:
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            # A vanished student (lid closed / cable pulled) sends no FIN; with
            # keepalive the reader fails and the normal LEAVE path runs.
            P.enable_keepalive(conn)
            threading.Thread(target=self._handle_worker, args=(conn,),
                             daemon=True).start()

    def _handle_worker(self, conn):
        rank = None
        why = "connection closed by the worker"      # reported in [LEAVE]
        try:
            who = "?"
            try:
                peer = conn.getpeername()
                who = "%s:%s" % (peer[0], peer[1])
            except OSError:
                pass
            msg = P.ctrl_recv_line(conn)
            if msg is None or msg.get("t") != P.C_JOIN:
                print("[JOIN] %s -> REJECTED: bad handshake (no join message)"
                      % who)
                return
            # ---- version guard: everyone must run the same MiniMPI --------
            wv = msg.get("version")
            wp = msg.get("protocol")
            if wv != P.MINIMPI_VERSION or wp != P.PROTOCOL_VERSION:
                why = ("version mismatch: worker=%s (protocol %s), "
                       "teacher=%s (protocol %s) — please UPDATE the student "
                       "copy (git pull) and restart"
                       % (wv or "unknown/old copy", wp,
                          P.MINIMPI_VERSION, P.PROTOCOL_VERSION))
                with self.ctrl_lock:
                    P.ctrl_send(conn, {"t": P.C_ERROR, "code": "version",
                                       "why": why})
                print("[VERSION] rejected a worker: %s\n"
                      "[JOIN] %s -> REJECTED: %s" % (why, who, why))
                return
            # Registering a worker, welcoming it and telling everybody else
            # must be ONE atomic control step: a RUN may not slip in between,
            # and two joins may not write to one socket at the same time.
            with self.ctrl_lock:
                with self.lock:
                    if self.run_active:
                        why = ("a collective run is in progress — wait for it "
                               "to finish, then join again")
                        P.ctrl_send(conn, {"t": P.C_ERROR, "code": "busy",
                                           "why": why})
                        print("[JOIN] %s -> REJECTED: %s" % (who, why))
                        return
                    free = [r for r in range(1, self.size)
                            if r not in self.workers]
                    if not free:
                        why = ("world already full (capacity %d) — the "
                               "teacher can raise --size" % self.size)
                        P.ctrl_send(conn, {"t": P.C_ERROR, "code": "full",
                                           "why": why})
                        print("[JOIN] %s -> REJECTED: %s" % (who, why))
                        return
                    rank = free[0]                # lowest free rank
                    self.workers[rank] = conn
                    self.epid[rank] = {"host": msg["host"],
                                       "port": int(msg["port"])}
                    moves = self._rebuild_roster_locked()
                    ready = len(self.workers) + 1
                    total = self.size
                    welcome = self._welcome_msg(rank, ready)
                    self._apply_local_size(ready)
                if moves:
                    print("[ROSTER] rank compaction: %s"
                          % ", ".join("%d -> %d" % mv for mv in moves))
                P.ctrl_send(conn, welcome)
                print("[JOIN] %s -> Rank %d  (data plane %s:%s)  "
                      "[%d/%d ranks ready%s]"
                      % (who, rank, msg.get("host"), msg.get("port"),
                         ready, total,
                         "" if ready == total else " - capacity"))
                # Everybody else (including ranks that just moved down) adopts
                # the new roster; the class keeps working without a restart.
                self._broadcast_roster(ready, None, skip=rank)
            while not self.closed:
                m = P.ctrl_recv_line(conn)
                if m is None:            # peer went away
                    break
                # A worker's rank may have changed by compaction, so resolve
                # it from the connection instead of trusting the join-time
                # number.
                with self.lock:
                    cur = next((r for r, c in self.workers.items()
                                if c is conn), None)
                if cur is None:
                    break
                self._dispatch(cur, m)
        except (ConnectionError, OSError) as e:
            why = type(e).__name__ if not str(e) else "%s: %s" % (
                type(e).__name__, e)
        except Exception as e:  # noqa: BLE001
            # never let an unexpected error hide a departure silently
            why = "internal error: %r" % e
        finally:
            try:
                conn.close()
            except OSError:
                pass
            # Find the worker by CONNECTION, never by its join-time rank:
            # compaction may have renumbered it, and a stale rank would make
            # the departure invisible — the class would then wait forever for
            # a rank that is already gone.
            with self.lock:
                cur = next((r for r, c in self.workers.items() if c is conn),
                           None)
            if cur is not None:
                self._on_worker_left(cur, who, why)

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
            if m.get("error") and self.run_active and not agg.aborted:
                # This rank cannot finish the RUN, so the others would sit in
                # a collective until the stall watchdog. End it now instead.
                self._abort_run("Rank %d failed: %s" % (rank, m.get("error")))
        elif t == P.C_HEARTBEAT:
            self.note_heartbeat(rank)    # roster view: who is really alive
            if agg:
                agg.heartbeat(rank)      # alive and waiting (not frozen)
        elif t == P.C_ERROR:
            print("[ERROR from rank %d] %s" % (rank, m.get("why", "")))

    def connectivity_check(self):
        print("\n========================================\nPeer Connectivity "
              "Check\n========================================")
        self._check_ok, self._check_fail = {}, {}
        n = self.active_size()
        msg = {"t": P.C_CHECK, "peers": self.peers}
        with self.ctrl_lock:
            for conn in list(self.workers.values()):
                try:
                    P.ctrl_send(conn, msg)
                except OSError:
                    pass
        self._self_check()
        deadline = time.time() + 30
        while len(self._check_ok) < n and time.time() < deadline:
            if self.active_size() != n:
                print("[ROSTER] membership changed during the check — "
                      "reporting what is known")
                break
            time.sleep(0.1)

        fails = 0
        unknown = 0
        for a in range(n):
            ok = self._check_ok.get(a, set())
            fail = self._check_fail.get(a, set())
            for b in range(n):
                if a == b:
                    continue
                status = "FAIL" if b in fail else ("PASS" if b in ok else "?")
                if status == "FAIL":
                    fails += 1
                elif status == "?":
                    unknown += 1
                print("Rank %d -> Rank %d  %s" % (a, b, status))
        if fails:
            print("\nP2P Network: NOT READY (%d failed edges)" % fails)
            return False
        if unknown:
            print("\nP2P Network: READY (%d edges unknown — a rank did not "
                  "report)" % unknown)
            return True
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

    def send_to_workers(self, obj, report=False):
        """Send one control message to every worker. With report=True the
        ranks whose connection refused the message are returned: a rank that
        cannot even receive a RUN is not part of the class any more."""
        failed = []
        with self.ctrl_lock:
            for rank, conn in list(self.workers.items()):
                try:
                    P.ctrl_send(conn, obj)
                except OSError as e:
                    failed.append(rank)
                    if report:
                        print("[ROSTER] Rank %d did not accept %s (%s)"
                              % (rank, obj.get("t"), e))
        return failed

    def run_demo(self, params, mode):
        # Current roster — a RUN always uses the ranks that are here NOW.
        n = self._apply_local_size()
        self._sync_rank0_peers()
        agg = Aggregator(n, teaching=(mode == "teaching"), auto=self.auto)
        self.agg = agg
        agg.reset()
        self.events0.clear()   # rank0 events belong to this RUN only
        params = dict(params, mode=mode, size=n)
        params["peers"] = self.peers
        value0 = params.get("value0", 1)
        # value0 is ONLY rank 0's own input — workers read theirs themselves.
        # (In benchmark cases value0 may be raw BYTES, which must never go
        # over the JSON control channel.)
        params.pop("value0", None)

        print("\nRunning...  (World Size %d)" % n)
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
                self.rank0._local_ctx = T.RoundCtx(self.rank0._alg, n,
                                                   0, [v0] * ds)
        params["value0"] = value0      # rank-0 dispatch arg (local only)
        self.comm0.default_timeout = None      # no deadline on a healthy RUN
        agg.activity_fn = lambda: len(self.events0.events)
        # rank 0 starts each RUN from a clean data plane too (stale frames from
        # an aborted RUN would be matched by this run's receives)
        self.transport.drop_pending()
        try:
            with self.ctrl_lock:
                # "a run is in flight" and the RUN itself are one step, so a
                # join or a leave can never rewrite the roster in between.
                self.run_active = True
                failed = self.send_to_workers(
                    {"t": P.C_RUN, "params": _without_value0(params)},
                    report=True)
            if failed:
                # Those ranks can never take part in this RUN: drop them now
                # (the normal leave path compacts the roster) instead of
                # waiting for a collective that can never complete.
                for r in failed:
                    self.exclude_suspects[r] = "could not receive the RUN"
                self._abort_run("rank(s) %s could not be reached for this RUN"
                                % ", ".join(str(r) for r in failed))
            self._start_rank0(agg, params)
            timeout = _run_timeout()
            while True:
                try:
                    finished = agg.wait_all_done(timeout=timeout, idle=timeout)
                    break
                except KeyboardInterrupt:
                    # The teacher sees a stuck class and presses Ctrl-C: never
                    # crash, offer the roster and the option to remove a rank.
                    # If a teaching pause is waiting for ENTER it owns stdin,
                    # so it serves the request instead (two threads must never
                    # read the keyboard at once).
                    if pause_active():
                        self.pause_menu.set()
                        print("\n[PAUSE] the RUN is still waiting — opening "
                              "the roster/kick prompt...")
                    else:
                        print("\n[PAUSE] the RUN is still waiting.\n"
                              "[PAUSE] remove a rank that will not answer and "
                              "the RUN is aborted; the class continues with "
                              "the rest.")
                        try:
                            roster_and_kick(self, allow_quit=True)
                        except KeyboardInterrupt:
                            print("\n(kick prompt cancelled)")
                    if not self.roster_snapshot()[1]:
                        finished = False
                        break
            if not finished:
                if agg.aborted:
                    print("[ABORTED] %s (done=%s)"
                          % (agg.aborted, sorted(agg.done)))
                else:
                    # Watchdog: a RUN that never finishes must not wedge the
                    # lesson — abort it, and remember who never reported so
                    # the class can continue without them.
                    silent = agg.silent_ranks()
                    self._abort_run(
                        "watchdog: no rank reported progress for %ds "
                        "(done=%s%s)" % (timeout, sorted(agg.done),
                                         ", silent=%s" % silent if silent
                                         else ""))
                    if not silent:
                        print("[ROSTER] no rank could be singled out — if this "
                              "repeats, ask a student to restart their worker")
                        print("[ROSTER] watchdog evidence: arrived=%s "
                              "reports=%s last_seen=%s"
                              % ({k: sorted(v) for k, v
                                  in sorted(agg.arrived.items())},
                                 {k: sorted(v) for k, v
                                  in sorted(agg.round_reports.items())},
                                 {r: "%.1fs ago" % (time.monotonic() - t)
                                  for r, t in sorted(agg.last_seen.items())}))
                    for r in silent:
                        self.exclude_suspects[r] = (
                            "no progress for %ds" % timeout)
        finally:
            with self.ctrl_lock:
                self.run_active = False
            self.agg = None
            # Let rank 0's algorithm thread unwind WHILE the abort is still
            # armed, then make the transport healthy again.
            self._join_rank0_thread()
            self.comm0.default_timeout = None
            self.transport.abort_pending(False)
            self._exclude_suspects()
            self._sync_roster_after_run()
        # Collective Time ends when ALL ranks reported C_DONE (the collective
        # really finished everywhere) — student input / RUN control / Start
        # Barrier waiting all happened before `col_start` and stay outside it.
        self._timing["col_end"] = now_ns()
        if agg.errors:
            print("Errors:", agg.errors)
        return agg

    def _join_rank0_thread(self, timeout=5.0):
        """Let rank 0's algorithm thread unwind before the next RUN.

        After an abort its pending receive has already been cancelled, so
        this normally returns at once — it just guarantees two runs never
        share one communicator."""
        t = getattr(self, "_rank0_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                print("[ABORT] rank 0 is still unwinding the previous run "
                      "(continuing; it will report its error)")

    def _start_rank0(self, agg, params):
        self._join_rank0_thread(timeout=5.0)

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
        t = threading.Thread(target=work, daemon=True)
        self._rank0_thread = t
        t.start()

    def on_round_ready(self, agg, rnd):
        """Rank-0 barrier leg after every rank arrived (Round Finished At is
        already fixed by record_gather). Shows the fixed 4-part teaching view,
        then pauses and releases:

            1. Global Communication     2. Rank 0 Local View
            3. Timing                   4. Teacher Control
        """
        agg.wait_round_events(rnd, need=agg.size - 1, timeout=30)
        alg = getattr(self.rank0, "_alg", "")
        P = agg.size                 # world size of THIS run (dynamic roster)
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
            ask_enter(self, "\n[ENTER] %s" % ("Close Collective" if last
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
        if self.agg is not None:
            self.agg.touch(src, rnd)   # this rank reached the barrier

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
        self._coord = coord
        self.comm = coord.comm0
        self.comm_world = None           # MPI.COMM_WORLD (set once)
        # Start Barrier fires this when every rank has ARRIVED (gather done,
        # before rank 0 sends the release broadcast). Teaching mode: make the
        # barrier visible, let the teacher ENTER, THEN record_start() — so the
        # pause never enters Round 1 / Collective Time.
        self.on_start_gathered = self._on_start_gathered
        self.events = coord.events0
        self.mode = "performance"
        self._agg = None                 # current RUN aggregator
        self._alg = ""                   # current RUN algorithm
        self._local_ctx = None           # per-run RoundCtx (rank-0 local view)
        # World.sync_round() calls this the moment rank 0's local algorithm
        # work for the round finished (before the round barrier) — the
        # "rank 0 ready" instant on the teacher's own clock.
        self.on_local_work_done = lambda rnd: coord.note_rank0_ready(rnd)
        # Start Barrier: rank 0 records every rank that really arrived, so a
        # stalled RUN can name the ranks that went quiet.
        self.on_start_arrival = (
            lambda src, ns: coord.note_rank_arrival(0, src, ns))

    @property
    def size(self):
        """World size of the CURRENT roster (rank 0 + joined workers)."""
        return self._coord.active_size()

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
                ask_enter(coord, "\n[ENTER] Start Collective")
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


_STDIN_Q = None
_STDIN_LOCK = threading.Lock()
# Who owns the keyboard right now. A teaching pause reads stdin on rank 0's
# algorithm thread while a menu/roster prompt reads it on the main thread; if
# both simply waited on the same queue, a line typed for one of them could be
# eaten by the other. Interactive prompts take priority; a pause then keeps
# waiting (it never steals) until no prompt is active.
_OWNER_LOCK = threading.Lock()
_INTERACTIVE_READERS = [0]
# Is a teaching pause waiting for ENTER right now? If it is, a Ctrl-C during a
# RUN must be handled BY THAT PAUSE: one thread owns stdin at a time, so a line
# can never be stolen by the other one.
_PAUSE_ACTIVE = [0]


def _owner_add(delta):
    with _OWNER_LOCK:
        _INTERACTIVE_READERS[0] += delta


def _owner_active():
    with _OWNER_LOCK:
        return _INTERACTIVE_READERS[0] > 0


def _pause_add(delta):
    with _OWNER_LOCK:
        _PAUSE_ACTIVE[0] += delta


def pause_active():
    with _OWNER_LOCK:
        return _PAUSE_ACTIVE[0] > 0


def _stdin_pump():
    """Read stdin forever, one line at a time, into a queue.

    ONE thread owns stdin for the whole teacher process. Both the menu and the
    teaching pauses read from that queue, so a line can never hide in Python's
    text buffer: the menu's input() used to swallow buffered ENTERs (piped
    lessons, or a teacher pressing ENTER twice), after which the pause waited
    forever while its input sat unread — exactly how a class ended up "unable
    to run".
    """
    while True:
        try:
            line = sys.stdin.readline()
        except (OSError, ValueError):
            line = ""
        if line == "":                    # EOF / closed stdin
            _STDIN_Q.put(None)
            return
        _STDIN_Q.put(line.rstrip("\n"))


def read_line(prompt, coord=None):
    """Print `prompt` and read one line. With `coord`, return "" instead of
    blocking when that RUN is aborted or the teacher is shutting down."""
    global _STDIN_Q
    with _STDIN_LOCK:
        if _STDIN_Q is None:
            _STDIN_Q = queue.Queue()
            threading.Thread(target=_stdin_pump, daemon=True).start()
    sys.stdout.write(prompt)
    sys.stdout.flush()
    epoch = getattr(coord, "abort_epoch", 0) if coord is not None else None
    interactive = coord is None
    if interactive:
        _owner_add(1)
    else:
        _pause_add(1)
    try:
        return _read_line_loop(coord, epoch, interactive)
    finally:
        if interactive:
            _owner_add(-1)
        else:
            _pause_add(-1)


def _read_line_loop(coord, epoch, interactive):
    while True:
        if not interactive and getattr(coord, "pause_menu", None) is not None \
                and coord.pause_menu.is_set():
            # the main thread asked for the teacher tool (Ctrl-C during a RUN):
            # this pause owns the keyboard, so it serves the request
            coord.pause_menu.clear()
            print("\n[PAUSE] the RUN is still waiting.\n"
                  "[PAUSE] remove a rank that will not answer and the RUN is "
                  "aborted; the class continues with the rest.")
            try:
                roster_and_kick(coord, allow_quit=True)
            except KeyboardInterrupt:
                print("\n(kick prompt cancelled)")
            return ""
        if not interactive and _owner_active():
            # a menu / roster prompt owns the keyboard: keep waiting, do not
            # consume the line the teacher is typing for it
            time.sleep(0.05)
            continue
        try:
            line = _STDIN_Q.get(timeout=0.2 if interactive else 0.05)
        except KeyboardInterrupt:
            if coord is None:
                # a menu / roster prompt: Ctrl-C means "leave the session"
                raise
            # Ctrl-C must never kill rank 0 (that would break the class). While
            # a RUN is in flight it opens the teacher tool instead: who is in
            # the class, and remove a rank that will not answer.
            print("\n[PAUSE] interrupted. The RUN keeps waiting; no rank was "
                  "dropped.")
            try:
                roster_and_kick(coord, allow_quit=True)
            except KeyboardInterrupt:
                print("\n(kick prompt cancelled)")
            return ""
        except queue.Empty:
            if coord is None:
                continue
            if getattr(coord, "abort_epoch", 0) != epoch:
                print("\n(pause cancelled: this RUN was aborted)")
                return ""
            if coord.closed:
                return ""
            continue
        if line is None:                  # EOF: never hang on a dead stdin
            return ""
        return line


def ask_enter(coord, prompt):
    """Teaching pause that can be CANCELLED (see read_line)."""
    return read_line(prompt, coord=coord)


def _run_timeout():
    """Stall watchdog for one RUN (seconds, MINIMPI_RUN_TIMEOUT to change).

    Measured as "no rank reported anything for this long", so a teacher
    explaining a round for a while is fine — a wedged rank is not."""
    try:
        v = int(os.environ.get("MINIMPI_RUN_TIMEOUT", "") or 600)
    except ValueError:
        v = 600
    return max(5, v)


def run_demo(coord, algo, mode, payload=0, vector_len=0, show=True,
             value0=1, silent=False, kind=None):
    n = coord.active_size()          # THIS run's world size (dynamic roster)
    if algo == "ping_pong" and n < 2:
        print("[skip] %s needs at least 2 ranks (World Size %d — every "
              "student has left, only Rank 0 is here)" % (algo, n))
        return None
    if algo == "recursive_doubling_allreduce" and not _pow2(n):
        print("[skip] %s requires a power-of-two world size (got %d)" %
              (algo, n))
        return None
    if algo in ("tree_reduce",) and not _pow2(n):
        print("[skip] %s requires a power-of-two world size (got %d)" %
              (algo, n))
        return None
    if not payload and vector_len == 0:
        if algo == "naive_reduce":
            vector_len = 1
        elif algo == "ring_allreduce":
            vector_len = n
        else:
            vector_len = 4
    if payload and n > 1 and payload % n and payload // n > 0:
        keep = (payload // n) * n
        print("[warn] Local Payload per Rank %s is not divisible by World Size "
              "%d (Ring sends equal chunks); using %s"
              % (fmt_bytes(payload), n, fmt_bytes(keep)))
        payload = keep
    if algo == "ring_allreduce" and vector_len % n:
        vector_len = ((vector_len + n - 1) // n) * n
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
#   default 16 B .. 4 MB (16 MB removed so a real-LAN classroom demo stays
#   short). Override with MINIMPI_BENCH_SIZES="16,1024,16384" etc.
def _bench_sizes():
    """Benchmark sizes (bytes per rank). A typo in MINIMPI_BENCH_SIZES must
    never kill the teacher, and a nonsense size must never be benchmarked:
    unparsable / below one int32 / not a multiple of 4 bytes is skipped with a
    warning, and if nothing is left the defaults are used."""
    default = [16, 1024, 16 * 1024, 256 * 1024, 4 * 1024 * 1024]
    raw = os.environ.get("MINIMPI_BENCH_SIZES", "")
    if not raw.strip():
        return default
    keep, bad = [], []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            bad.append(part)
            continue
        if v < P.ELEMENT_BYTES or v % P.ELEMENT_BYTES:
            bad.append(part)
            continue
        keep.append(v)
    if bad:
        print("[warn] MINIMPI_BENCH_SIZES ignored %s (need multiples of %d "
              "bytes)" % (", ".join(bad), P.ELEMENT_BYTES))
    return keep or default

BENCH_SIZES = _bench_sizes()
BENCH_RUNS = 3


def _bench_skip(algo, payload_bytes, size):
    """Cases the CURRENT roster cannot run honestly.

    Both are reachable in a real classroom now that the roster changes: a
    class of 3 ranks cannot run Recursive Doubling (power-of-two only), and
    its payloads are not all divisible by 3 (Ring sends equal chunks)."""
    if algo == "recursive_doubling_allreduce" and not _pow2(size):
        return ("requires a power-of-two world size (have %d)" % size)
    if algo == "ring_allreduce" and size and payload_bytes % size:
        return ("payload is not divisible by the world size %d" % size)
    return None


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
    algorithms x len(BENCH_SIZES) sizes x 3 runs run automatically;\n    results are medians and the summary is shown on EVERY rank."""
    if coord.active_size() < 2:
        print("\n[ROSTER] No student ranks are connected (World Size %d) — "
              "a benchmark needs at least one worker." % coord.active_size())
        print("[ROSTER] Start a student worker and choose 7 again.")
        return
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
        raw = read_line("\nYour benchmark value (Rank 0, default 1):\n> ").strip()
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
    skipped = []
    total = len(BENCH_ALGORITHMS) * len(BENCH_SIZES) * BENCH_RUNS
    done = 0
    cancelled = None
    size0 = coord.active_size()      # a stable world size is the whole point
    for a in BENCH_ALGORITHMS:
        for b in BENCH_SIZES:
            times = []
            if coord.active_size() != size0:
                cancelled = ("the world size changed during the session "
                             "(%d -> %d)" % (size0, coord.active_size()))
                break
            why = _bench_skip(a, b, size0)
            if why:
                # never record a time for a case this world cannot run: the
                # table would silently show meaningless numbers
                print("[skip] %s %s = %d elem: %s"
                      % (a, fmt_bytes(b), b // P.ELEMENT_BYTES, why))
                rows[a][b] = None
                skipped.append((a, b, why))
                continue
            # every real benchmark case is explicitly a benchmark_case (no
            # re-prompt on worker) and rank 0 carries its OWN payload bytes
            # derived from the value entered once at setup.
            payload0 = collectives_dispatch.make_benchmark_payload(v0, b)
            for k in range(BENCH_RUNS):
                agg = run_demo(coord, a, "performance", payload=b,
                               show=False, silent=True, kind="benchmark_case",
                               value0=payload0)
                if agg is not None and agg.aborted:
                    # a rank left / the watchdog fired: stop the session
                    # instead of filling the table with meaningless runs
                    cancelled = agg.aborted
                    break
                cms = coord.collective_ms()
                ms = cms if cms is not None else 0.0
                times.append(ms)
                done += 1
                print("raw %s %d run%d %.3f ms  (%d / %d)" %
                      (a, b, k + 1, ms, done, total))
            if cancelled:
                break
            rows[a][b] = _median(times)
        if cancelled:
            break

    if cancelled:
        print("\n[ABORT] benchmark session cancelled: %s" % cancelled)
        print("[ABORT] %d of %d raw runs had completed; no summary is "
              "printed because the world size changed mid-session." %
              (done, total))
        print("[ROSTER] World Size is now %d — run the benchmark again to get "
              "a consistent table." % coord.active_size())
        coord.send_to_workers(
            {"t": P.C_RUN, "params": {"kind": "benchmark_done",
                                      "mode": "performance",
                                      "summary": "Benchmark cancelled: %s"
                                                 % cancelled}})
        print("\nBenchmark Cancelled.\n\nReturning to Teacher menu...")
        return

    summary = ["\n" + "=" * 78,
               "Performance Benchmark Results",
               "%d runs per case — median" % BENCH_RUNS,
               "World Size: %d" % coord.active_size(),
               "=" * 78]
    cols = ["Local Data / Rank", "Naive", "Recursive Doubling", "Ring"]
    widths = [26, 10, 18, 12]
    header = "".join(c.ljust(w) for c, w in zip(cols, widths))
    summary.append(header)
    summary.append("-" * len(header))
    def cell(algo, b):
        v = rows[algo][b]
        return "%7.2f ms" % v if isinstance(v, (int, float)) else "n/a".rjust(10)

    for b in BENCH_SIZES:
        line = ("%s  = %d elem" % (fmt_bytes(b), b // P.ELEMENT_BYTES)).ljust(
            widths[0])
        line += cell("naive_allreduce", b).ljust(widths[1])
        line += cell("recursive_doubling_allreduce", b).ljust(widths[2])
        line += cell("ring_allreduce", b).ljust(widths[3])
        summary.append(line)
    summary.append("\nLower is better.")
    if skipped:
        summary.append("\nSkipped for World Size %d (the table shows n/a):"
                       % size0)
        for a, b, why in skipped:
            summary.append("  %s @ %s = %d elem: %s"
                           % (a, fmt_bytes(b), b // P.ELEMENT_BYTES, why))
    print("\n".join(summary))
    # show the same summary table on EVERY student rank
    coord.send_to_workers(
        {"t": P.C_RUN, "params": {"kind": "benchmark_done",
                                  "mode": "performance",
                                  "summary": "\n".join(summary)}})
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
    # teacher tools (10 is printed after Exit on purpose: 9 must stay Exit)
    ("Roster — who is in the class / kick a rank", "kick"),
]


def roster_and_kick(coord, allow_quit=False):
    """Teacher tool: show who is really in the class, and remove somebody.

    Joins and departures are automatic and live; this is for the cases the
    teacher must decide: a student's machine that stopped answering, somebody
    who has to leave, or a RUN blocked by a rank that will never reply.
    """
    print("\n========================================\n"
          "Class Roster\n"
          "========================================")
    n, ranks = coord.roster_snapshot()
    stale = P.HEARTBEAT_STALE_S
    print("Rank 0  teacher (this machine)          —")
    for r in ranks:
        ep = coord.peers.get(r, {})
        age = coord.liveness_age(r)
        if age is None:
            state = "no heartbeat yet"
        elif age > stale:
            state = "SILENT — no heartbeat for %.1f s" % age
        else:
            state = "alive (heartbeat %.1f s ago)" % age
        print("Rank %-2d %-32s %s" % (r, "%s:%s" % (ep.get("host", "?"),
                                                   ep.get("port", "?")), state))
    print("\nActive world size: %d (capacity %d)%s"
          % (n, coord.size,
             "   [a RUN is in flight]" if coord.run_active else ""))
    print("[ROSTER] joining and leaving are automatic — this tool is for "
          "decisions only the teacher can make.")
    hint = "\nKick which rank? (number, ENTER = cancel"
    if allow_quit and coord.run_active:
        hint += ", q = abort this RUN and go back to the menu"
    raw = read_line(hint + ")\n> ").strip().lower()
    if not raw:
        print("(cancelled)")
        return
    if raw in ("q", "quit", "abort") and allow_quit:
        coord._abort_run("the teacher aborted the RUN (Ctrl-C at the pause)")
        print("[PAUSE] RUN aborted — back to the menu.")
        return
    if not raw.isdigit() or int(raw) not in ranks:
        print("[KICK] %r is not an active student rank." % raw)
        return
    rank = int(raw)
    age = coord.liveness_age(rank)
    if age is not None and age > stale:
        print("[KICK] Rank %d has not sent a heartbeat for %.1f s — it may "
              "already be gone." % (rank, age))
    if coord.run_active:
        print("[KICK] a RUN is in flight: removing Rank %d ABORTS that RUN; "
              "the class then continues with the remaining ranks." % rank)
    if read_line("Kick Rank %d? [y/N] " % rank).strip().lower() not in ("y",
                                                                      "yes"):
        print("(cancelled)")
        return
    coord.kick_rank(rank)


def wait_ready(coord, size, require_full=True, quiesce=3.0):
    """Wait for the class to arrive, then report the ACTIVE world size.

    require_full=True  (demo / benchmark / --auto): wait for the full `--size`
                       so scripted runs and tests are reproducible.
    require_full=False (interactive lesson): wait for at least 2 ranks and
                       start once joins stop coming (quiescence). Latecomers
                       may still join and anyone may leave: the roster is
                       rebuilt live between RUNs.
    """
    if require_full:
        cur, _ = coord.roster_snapshot()
        last, last_note = cur, time.time()
        print("\nWaiting for ranks (%d/%d)..." % (cur, size))
        while cur < size:
            cur, have = coord.roster_snapshot()
            if cur != last:                      # live feedback per change
                if cur < last:                   # somebody left while waiting
                    print("[ROSTER] a rank left before the world was ready: "
                          "%d/%d — waiting for a replacement" % (cur, size))
                print("Waiting for ranks (%d/%d)..." % (cur, size))
                last, last_note = cur, time.time()
            elif time.time() - last_note > 20:
                # a demo / benchmark really needs the full size: say which
                # ranks are missing instead of looking hung
                print("[ROSTER] waiting for the full world (%d/%d): have rank "
                      "0%s; missing %s"
                      % (cur, size, "".join(", %d" % r for r in have),
                         ", ".join(str(r) for r in range(1, size)
                                   if r not in have)))
                last_note = time.time()
            time.sleep(0.4)
        print("MPI World Ready: %d / %d ranks (%d student workers)\n"
              % (size, size, size - 1))
        return size

    print("\nWaiting for students to join (%d rank minimum, capacity %d)..."
          % (2, size))
    print("[ROSTER] Start as soon as students stop joining — latecomers and "
          "leavers are both handled live.")
    last, last_change = -1, time.time()
    while True:
        n = coord.active_size()
        if n != last:
            print("Waiting for ranks (%d joined, capacity %d)..."
                  % (n, size))
            last, last_change = n, time.time()
        if n >= size:
            break
        if n >= 2 and time.time() - last_change >= quiesce:
            break
        time.sleep(0.3)
    n = coord.active_size()
    print("MPI World Ready: %d ranks (%d student workers)%s\n"
          % (n, n - 1,
             "" if n >= size else "  — capacity %d, more students may still "
             "join" % size))
    return n


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
        print("Version: %s (protocol %d)" % (P.MINIMPI_VERSION,
                                             P.PROTOCOL_VERSION))
        print("Expected World Size (capacity): %d" % args.size)
        print("\nTeacher Role")
        print("-" * 40)
        print("Classroom Coordinator")
        print("Global Communication View")
        print("MPI Participant: Rank 0")
        print("-" * 40)

        wait_ready(coord, args.size,
                   require_full=bool(args.benchmark or args.demo or args.auto))
        # Warm the REAL persistent data-plane connections that collectives
        # will reuse (workers already warmed theirs after joining; teacher
        # warms its outbound legs). No message, no event, no timing.
        coord.transport.warm_to(list(range(1, coord.active_size())))
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
            n, have = coord.roster_snapshot()
            print("World Size: %d (capacity %d, %d student ranks)"
                  "   MPI World Ready: %d ranks\n"
                  % (n, args.size, n - 1, n))
            if n < 2:
                print("[ROSTER] Nobody has joined yet — start a student "
                      "worker (scripts/start_worker_mac.command) and this "
                      "menu will pick it up. A RUN now would be Rank 0 alone "
                      "(World Size 1 is not a collective).")
            for i, (label, _) in enumerate(MENU, 1):
                if i == len(MENU):           # teacher tools come after Exit
                    print("\nTeacher tools:")
                print("%d. %s" % (i, label))
            try:
                choice = read_line("\nSelect: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if choice in ("9", "exit"):
                break
            if choice in ("8", "network"):
                coord.connectivity_check()
                continue
            if choice in ("10", "kick", "roster", "r"):
                roster_and_kick(coord)
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
            ds_raw = read_line(
                "\nData Size (elements, int32 = %d B/elem, default 16):\n> "
                % P.ELEMENT_BYTES).strip()
            try:
                ds = int(ds_raw) if ds_raw.isdigit() and int(ds_raw) > 0 else 16
            except ValueError:
                ds = 16
            print("\nMode:")
            print("1. Teaching")
            print("2. Performance")
            m = read_line("\nSelect: ").strip()
            mode = "teaching" if m != "2" else "performance"
            v = read_line("\nYour value (Rank 0, default 1):\n> ").strip()
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
