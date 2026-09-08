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
        return inst

    def __init__(self):
        self._rt = None
        self._q = None
        self._reader = None
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

    def close(self):
        self._rt = None
        self._benchmark_value = None

    # ------------------------------------------------------------- control
    def _read_loop(self):
        rt = self._rt
        try:
            while True:
                m = rt.control.recv()
                t = m.get("t")
                if t == P.C_RUN:
                    self._q.put(RunSpec(params=m["params"]))
                elif t == P.C_CHECK:
                    self._answer_check(m)
                elif t == P.C_SHUTDOWN:
                    break
        except (ConnectionError, OSError):
            pass
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
