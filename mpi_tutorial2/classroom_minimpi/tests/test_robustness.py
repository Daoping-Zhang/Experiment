#!/usr/bin/env python3
"""test_robustness.py — classroom membership robustness (real processes).

A live lesson is not a batch job: students join late, close their laptop in
the middle of a collective, or restart. These tests drive the REAL teacher.py
and worker.py over TCP and check that the class never wedges:

  A  worker leaves while the class is idle  -> roster compacts, next RUN works
  B  latecomer joins after the lesson began -> gets a free rank, size grows
  C  worker leaves DURING a run             -> ABORT, no hang, teacher usable
  D  watchdog: a rank that never arrives    -> run aborted, teacher exits
  E  teacher disappears mid-run             -> worker unblocks and exits

Every scenario has a hard deadline: a hang is a FAIL, never an infinite wait.
"""
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # classroom_minimpi/
PY = sys.executable

_passed = []
_failed = []


def check(name, ok, extra=""):
    (_passed if ok else _failed).append(name)
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  (%s)" % extra) if extra else ""))


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def kill(p):
    if p is None or p.poll() is not None:
        return
    try:
        p.kill()
    except OSError:
        pass
    try:
        p.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


class Class:
    """One teacher + its workers, each logging to its own file.

    No pipes: a worker blocked mid-collective must never be stopped by a full
    stdout pipe, and every log must stay readable while the test runs.
    """

    def __init__(self, size, tmp, extra_teacher=(), teacher_env=None):
        self.size = size
        self.port = free_port()
        self.tmp = tmp
        self.extra_teacher = list(extra_teacher)
        self.teacher_env = dict(teacher_env or {})
        self.tag = "cap%d" % size
        self.teacher = None
        self.workers = {}                 # tag -> Popen
        self._f = {}
        self._p = {}

    # ------------------------------------------------------------- spawn
    def _log(self, tag):
        path = os.path.join(self.tmp, "%s_%s_%d.log" % (self.tag, tag, self.port))
        self._f[tag] = open(path, "w+")
        self._p[tag] = path
        return self._f[tag]

    def start_teacher(self, tag="teacher", interactive=True, extra=None):
        f = self._log(tag)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env.update(self.teacher_env)
        args = [PY, os.path.join(ROOT, "teacher.py"),
                "--size", str(self.size), "--host", "127.0.0.1",
                "--port", str(self.port), "--advertise", "127.0.0.1"]
        args += self.extra_teacher if extra is None else extra
        self.teacher = subprocess.Popen(
            args, stdin=subprocess.PIPE if interactive else subprocess.DEVNULL,
            stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
        # wait until the control socket listens: a worker started too early
        # dies with "connection refused" and would silently change the roster
        self.wait_log(r"Coordinator: 127\.0\.0\.1:%d" % self.port, timeout=20)
        return self.teacher

    def start_worker(self, tag, env_extra=None):
        f = self._log(tag)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env.update(env_extra or {})
        args = [PY, os.path.join(ROOT, "worker.py"),
                "--server", "127.0.0.1:%d" % self.port]
        self.workers[tag] = subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=f,
            stderr=subprocess.STDOUT, text=True, env=env)
        self.wait_log(r"MiniMPI Worker version", timeout=20, tag=tag)
        return self.workers[tag]

    def alive(self, tag):
        p = self.workers.get(tag)
        return p is not None and p.poll() is None

    # -------------------------------------------------------------- drive
    def menu(self, text):
        self.teacher.stdin.write(text)
        self.teacher.stdin.flush()

    def log(self, tag="teacher"):
        f = self._f.get(tag)
        if f is None:
            return ""
        f.flush()
        with open(self._p[tag]) as fh:
            return fh.read()

    def count(self, pattern, tag="teacher"):
        return len(re.findall(pattern, self.log(tag)))

    def wait_count(self, pattern, n=1, timeout=30, tag="teacher"):
        end = time.time() + timeout
        while time.time() < end:
            if self.count(pattern, tag) >= n:
                return True
            time.sleep(0.2)
        return False

    def wait_log(self, pattern, timeout=30, tag="teacher"):
        return self.wait_count(pattern, 1, timeout, tag)

    def wait_menu(self, n=1, timeout=40):
        """The interactive menu shows the CURRENT world size, so this pattern
        also proves which roster the teacher believes in."""
        return self.wait_count(r"\(capacity %d, " % self.size, n, timeout)

    # ------------------------------------------------------------- cleanup
    def close(self):
        for p in self.workers.values():
            kill(p)
        kill(self.teacher)
        for f in self._f.values():
            try:
                f.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# A. a worker leaves while the class is idle -> roster compacts, RUN still works
def scenario_leave_when_idle(tmp):
    c = Class(4, tmp)
    try:
        c.start_teacher()
        c.start_worker("w1")
        # join one worker at a time: the JOIN arrival order decides the rank,
        # so this keeps "w2 == Rank 2" true in the assertions below
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2")
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("A idle leave: teacher reaches its menu", False)
            return
        check("A idle leave: teacher reaches its menu", True)
        kill(c.workers["w2"])
        check("A idle leave: teacher notices the departure",
              c.wait_log(r"\[LEAVE\] Rank 2 disconnected", timeout=25))
        check("A idle leave: remaining worker is re-welcomed",
              c.wait_log(r"\[ROSTER\] Rank 1 re-welcomed", timeout=25))
        # the RUN must use the roster that is here NOW (rank 0 + w1 = 2)
        c.menu("3\n16\n2\n1\n")             # naive_allreduce / perf / value 1
        check("A idle leave: following RUN uses World Size 2",
              c.wait_log(r"Running\.\.\.  \(World Size 2\)", timeout=30))
        check("A idle leave: RUN completes (no hang)",
              c.wait_log(r"Collective complete", timeout=30))
        check("A idle leave: result correct (1+2=3)", "final = 3" in c.log())
    finally:
        c.close()


# ---------------------------------------------------------------------------
# B. a latecomer joins after the lesson started -> free rank, world grows
def scenario_latecomer_join(tmp):
    c = Class(4, tmp)
    try:
        c.start_teacher()
        c.start_worker("w1")
        if not c.wait_menu(1, timeout=45):
            check("B late join: teacher reaches its menu", False)
            return
        c.start_worker("w2")                 # joins while the menu is up
        check("B late join: latecomer is welcomed as Rank 2",
              c.wait_log(r"\[JOIN\] .* -> Rank 2 .*\[3/4 ranks ready",
                         timeout=25))
        c.menu("3\n16\n2\n1\n")
        check("B late join: RUN uses World Size 3",
              c.wait_log(r"Running\.\.\.  \(World Size 3\)", timeout=30))
        check("B late join: RUN completes, result = 6 (1+2+3)",
              c.wait_log(r"Collective complete", timeout=30)
              and "final = 6" in c.log())
    finally:
        c.close()


# ---------------------------------------------------------------------------
# C. a worker disappears DURING a run -> ABORT, teacher stays usable
def scenario_leave_during_run(tmp):
    c = Class(3, tmp)
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2", env_extra={"MINIMPI_INPUT_DELAY": "25"})
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("C mid-run leave: teacher reaches its menu", False)
            return
        if not (c.alive("w1") and c.alive("w2")):
            check("C mid-run leave: both workers joined", False)
            return
        check("C mid-run leave: both workers joined", True)
        c.menu("3\n16\n1\n1\n")              # naive_allreduce / TEACHING
        if not c.wait_log(r"Running\.\.\.  \(World Size 3\)", timeout=25):
            check("C mid-run leave: run started", False)
            return
        check("C mid-run leave: run started", True)
        time.sleep(2.0)
        kill(c.workers["w2"])                # never reaches the barrier
        check("C mid-run leave: departure detected during the RUN",
              c.wait_log(r"\[LEAVE\] Rank 2 disconnected  \(during a RUN\)",
                         timeout=25))
        check("C mid-run leave: RUN is aborted instead of hanging",
              c.wait_log(r"\[ABORT\] Rank 2 left during the run", timeout=25))
        # the teacher must come back to its menu with the compacted roster
        check("C mid-run leave: teacher returns to its menu (no wedge)",
              c.wait_count(r"World Size: 2 \(capacity 3, 1 student",
                           1, timeout=45))
        # ... and the class can simply run again
        c.menu("3\n16\n2\n1\n")
        check("C mid-run leave: next RUN works with the remaining ranks",
              c.wait_log(r"Running\.\.\.  \(World Size 2\)", timeout=30)
              and c.wait_log(r"Collective complete", timeout=30))
        check("C mid-run leave: result from the surviving ranks (1+2=3)",
              "final = 3" in c.log())
    finally:
        c.close()


# ---------------------------------------------------------------------------
# D. watchdog: a rank never arrives -> RUN aborted, teacher exits
def scenario_watchdog(tmp):
    c = Class(2, tmp, extra_teacher=["--demo", "naive_allreduce",
                                     "--mode", "performance"],
              teacher_env={"MINIMPI_RUN_TIMEOUT": "5"})
    try:
        c.start_teacher(interactive=False)
        # this worker sleeps 60 s before entering the barrier: the run stalls
        c.start_worker("w1", env_extra={"MINIMPI_INPUT_DELAY": "60"})
        end = time.time() + 60
        while time.time() < end and c.teacher.poll() is None:
            time.sleep(0.3)
        log = c.log()
        check("D watchdog: a stalled RUN does not hang forever",
              c.teacher.poll() is not None, "rc=%s" % c.teacher.poll())
        check("D watchdog: the stall is reported as an abort",
              "watchdog: no rank reported progress" in log
              or "[ABORTED]" in log)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# E. the teacher disappears mid-run -> the worker unblocks and exits
def scenario_teacher_gone(tmp):
    c = Class(2, tmp)
    try:
        c.start_teacher()
        c.start_worker("w1")
        if not c.wait_menu(1, timeout=45):
            check("E teacher gone: menu reached", False)
            return
        c.menu("3\n16\n1\n1\n")              # teaching run: pauses at the barrier
        if not c.wait_log(r"World Size: 2", timeout=25):
            check("E teacher gone: run started", False)
            return
        # teaching mode pauses inside the Start Barrier, so w1 is parked in a
        # blocking receive when the teacher disappears
        if not c.wait_log(r"Start Barrier", timeout=30):
            check("E teacher gone: worker reached the barrier", False)
            return
        kill(c.teacher)
        w1 = c.workers["w1"]
        end = time.time() + 40
        while time.time() < end and w1.poll() is None:
            time.sleep(0.3)
        check("E teacher gone: worker notices and exits (no hang)",
              w1.poll() is not None)
        check("E teacher gone: worker explains why",
              "teacher control connection closed" in c.log("w1")
              or "[ABORT]" in c.log("w1"))
    finally:
        c.close()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    tmp = tempfile.mkdtemp(prefix="minimpi_robustness_")
    print("Robustness logs: %s\n" % tmp)
    scenario_leave_when_idle(tmp)
    scenario_latecomer_join(tmp)
    scenario_leave_during_run(tmp)
    scenario_watchdog(tmp)
    scenario_teacher_gone(tmp)
    print("\nRobustness tests: %d passed, %d failed" % (len(_passed),
                                                        len(_failed)))
    if _failed:
        print("Failed: %s" % ", ".join(_failed))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
