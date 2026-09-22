"""classroom_worker.py — teacher-control reading for one student worker.

Classroom infrastructure (NOT a fake MPI API): a single control reader
thread receives teacher commands and fills a run queue. The worker's MAIN
thread waits on wait_for_run() and executes each collective itself, one at a
time — so the main program reads like:

    while True:
        run = classroom.wait_for_run()     # blocks until teacher sends RUN
        if run is None:                    # SHUTDOWN sentinel
            break
        ...
"""
import dataclasses
import queue
import socket
import threading
import time

from . import protocol as P
from . import mpi as M
from .runtime import MiniRuntime  # noqa: F401  (documented dependency)


@dataclasses.dataclass
class RunSpec:
    """One RUN command from the teacher (a classroom object, not MPI)."""
    params: dict

    @property
    def kind(self):
        return self.params.get("kind", "run")

    @property
    def algorithm(self):
        return self.params.get("algorithm", "")

    @property
    def mode(self):
        return self.params.get("mode", "performance")

    @property
    def data_size(self):
        return int(self.params.get("data_size")
                   or self.params.get("vector_len") or 0)

    @property
    def payload(self):
        return int(self.params.get("payload") or 0)

    @property
    def fmt(self):
        return self.params.get("fmt", "i32")


class ClassroomWorker:
    """One instance per student worker: owns the control reader + run queue."""

    _instance = None

    @classmethod
    def attach_session(cls):
        """Attach to the MPI session runtime started by MPI.Init(server)."""
        if cls._instance is None:
            cls._instance = cls()
        inst = cls._instance
        rt = M._session["rt"]           # classroom infra, hidden from worker.py
        inst._rt = rt
        inst._q = queue.Queue()
        inst._reader = threading.Thread(target=inst._read_loop, daemon=True)
        inst._reader.start()
        inst._beat = threading.Thread(target=inst._heartbeat_loop, daemon=True)
        inst._beat.start()
        return inst

    def __init__(self):
        self._rt = None
        self._q = None
        self._reader = None
        self._beat = None
        self.kicked = False
        self._benchmark_value = None
        self._benchmark_announced = False

    # ------------------------------------------------------------------ API
    @property
    def runtime(self):
        """The MiniRuntime behind the session (used by helper code only)."""
        return self._rt

    def wait_for_run(self):
        """Blocking, meaningful wait: returns the next RunSpec, or None when
        the teacher shut the session down."""
        spec = self._q.get()
        return spec

    def apply_roster(self, rank, size, peers):
        """Adopt a new rank/size/peers from the teacher (between RUNs)."""
        if self._rt is not None:
            self._rt.apply_roster(rank, size, peers)

    def clear_abort(self):
        if self._rt is not None:
            self._rt.clear_abort()

    def drop_stale_frames(self):
        """Forget frames left over by the previous RUN (see transport)."""
        if self._rt is not None:
            self._rt.drop_stale_frames()

    def benchmark_value(self):
        """Value typed once at Performance Benchmark setup; None outside a
        benchmark session."""
        return self._benchmark_value

    def set_benchmark_value(self, v):
        self._benchmark_value = v
        self._benchmark_announced = False

    def benchmark_announced(self):
        return self._benchmark_announced

    def mark_benchmark_announced(self):
        self._benchmark_announced = True

    def reset_benchmark(self):
        """Benchmark session finished: clear the cached value so the next
        Benchmark selection asks for a fresh input."""
        self._benchmark_value = None
        self._benchmark_announced = False

    def close(self):
        self._rt = None                # also stops the heartbeat thread
        self._benchmark_value = None

    # ------------------------------------------------------------- control
    def _heartbeat_loop(self):
        """Tell the teacher we are alive while we wait.

        A rank blocked in a barrier looks exactly like a frozen rank on the
        data plane; the heartbeat is what separates "waiting" from "gone", so a
        stalled RUN can name the student whose machine stopped answering."""
        while (self._rt is not None and self._rt.control is not None
               and not self.kicked):
            try:
                self._rt.control.send({"t": P.C_HEARTBEAT,
                                       "rank": self._rt.rank})
            except OSError as e:
                # The link to the teacher is gone. The control reader may be
                # parked in a receive that never returns, so THIS is where we
                # notice: never let the worker sit idle in a session that no
                # longer exists — end it so the student can rejoin.
                print("\n[CONTROL LOST] %s" % e)
                print("[CONTROL LOST] the teacher's connection is gone — "
                      "leaving the session (restart to join again).")
                self._rt.abort_run("the teacher's control connection was lost")
                self._q.put(None)          # main loop: exit cleanly
                return
            time.sleep(P.HEARTBEAT_S)

    def _rt_abort(self, why):
        if self._rt is not None:
            self._rt.abort_run(why)

    def _read_loop(self):
        rt = self._rt
        why = "teacher closed the control connection"
        try:
            while True:
                try:
                    m = rt.control.recv()
                except ValueError as e:
                    # one malformed line must not kill the control reader (a
                    # dead reader means this rank never hears the next RUN)
                    print("[control] ignoring a malformed message: %s" % e)
                    continue
                t = m.get("t")
                if t == P.C_RUN:
                    self._q.put(RunSpec(params=m["params"]))
                elif t == P.C_WELCOME:
                    # teacher re-assigned our rank (someone joined/left)
                    self._q.put(RunSpec(params={"kind": "roster",
                                                "rank": m.get("rank"),
                                                "size": m.get("size"),
                                                "peers": m.get("peers", {}),
                                                "why": m.get("why", "")}))
                elif t == P.C_KICK:
                    # the teacher removed this rank: stop the heartbeat, end
                    # any RUN we are stuck in and shut down cleanly
                    self.kicked = True
                    self._rt_abort("removed from the class by the teacher")
                    self._q.put(RunSpec(params={
                        "kind": "kicked",
                        "why": m.get("why") or "the teacher removed this rank"}))
                elif t == P.C_ABORT:
                    # a rank left / teacher cancelled: unblock immediately
                    if self._rt is not None:
                        self._rt.abort_run(m.get("why", ""))
                    self._q.put(RunSpec(params={"kind": "abort",
                                                "why": m.get("why", "")}))
                elif t == P.C_CHECK:
                    self._answer_check(m)
                elif t == P.C_SHUTDOWN:
                    why = "session ended by the teacher"
                    break
        except (ConnectionError, OSError):
            why = "teacher control connection closed"
        # Never leave the main thread blocked inside a collective waiting for
        # a rank that can no longer answer.
        if self._rt is not None:
            self._rt.abort_run(why)
        self._q.put(None)               # shutdown sentinel for the main loop
        if self._reader is not None:
            pass

    def _answer_check(self, m):
        rt = self._rt
        ok, fail = [], []
        for dst, ep in m.get("peers", {}).items():
            dst = int(dst)
            if dst == rt.rank:
                continue
            try:
                s = socket.create_connection((ep["host"], int(ep["port"])),
                                             timeout=3)
                s.close()
                ok.append(dst)
            except OSError:
                fail.append(dst)
        try:
            rt.control.send({"t": P.C_CHECK_REPORT, "rank": rt.rank,
                             "pass": ok, "fail": fail})
        except OSError:
            pass
