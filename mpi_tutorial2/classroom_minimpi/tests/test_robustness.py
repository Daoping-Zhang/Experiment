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
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minimpi import protocol as _P       # noqa: E402  (version for the stub)


def _version():
    return _P.MINIMPI_VERSION


def _protocol():
    return _P.PROTOCOL_VERSION

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

    def __init__(self, name, size, tmp, extra_teacher=(), teacher_env=None):
        self.name = name
        self.size = size
        self.port = free_port()
        self.tmp = tmp
        self.extra_teacher = list(extra_teacher)
        self.teacher_env = dict(teacher_env or {})
        self.tag = name
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

    def freeze(self, tag):
        """Simulate a vanished machine: the process is alive but answers
        nothing (no FIN is sent — exactly like a closed laptop lid)."""
        p = self.workers.get(tag)
        if p is not None and p.poll() is None:
            os.kill(p.pid, signal.SIGSTOP)

    def join_stub(self, settle=0.05):
        """A control-plane client that joins and then dies without reading the
        welcome (a student who closes the laptop right after clicking join)."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        s.sendall((json.dumps({"t": "join", "host": "127.0.0.1", "port": 1,
                               "version": _version(), "protocol": _protocol()})
                   + "\n").encode())
        time.sleep(settle)
        s.close()

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
    c = Class("A", 4, tmp)
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
    c = Class("B", 4, tmp)
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
    c = Class("C", 3, tmp)
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
    c = Class("D", 2, tmp, extra_teacher=["--demo", "naive_allreduce",
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
    c = Class("E", 2, tmp)
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


# ---------------------------------------------------------------------------
# F. a burst of joins -> every rank must end up with the SAME world size
def scenario_burst_join(tmp):
    """Concurrent welcomes used to interleave on one control socket and leave
    workers with different world sizes; the next collective then paired ranks
    that disagreed and deadlocked. All three students join at once here."""
    c = Class("F", 4, tmp)
    try:
        c.start_teacher()
        for tag in ("w1", "w2", "w3"):
            c.start_worker(tag)              # no waiting: a burst of joins
        check("F burst join: all 4 ranks are ready",
              c.wait_log(r"\[4/4 ranks ready", timeout=45))
        if not c.wait_menu(1, timeout=45):
            check("F burst join: teacher reaches its menu", False)
            return
        ranks, sizes = [], []
        for tag in ("w1", "w2", "w3"):
            log = c.log(tag)
            last = re.findall(r"\[ROSTER\] You are now Rank (\d+) / (\d+)", log)
            if last:
                r, n = last[-1]
            else:                            # never re-welcomed: startup rank
                m = re.search(r"Rank: (\d+) / (\d+)", log)
                if m is None:
                    r, n = "?", "?"
                else:
                    r, n = m.groups()
            ranks.append(r)
            sizes.append(n)
        check("F burst join: every worker agrees on the world size",
              set(sizes) == {"4"}, "sizes=%s" % sizes)
        check("F burst join: ranks are unique 1..3",
              sorted(ranks) == ["1", "2", "3"], "ranks=%s" % ranks)
        c.menu("3\n16\n2\n1\n")
        check("F burst join: RUN completes with World Size 4",
              c.wait_log(r"Running\.\.\.  \(World Size 4\)", timeout=30)
              and c.wait_log(r"Collective complete", timeout=30))
        check("F burst join: result = 10 (1+2+3+4)",
              "final = 10" in c.log()
              and "Errors: {}" not in c.log().split("Collective complete")[-1])
    finally:
        c.close()


# ---------------------------------------------------------------------------
# G. a worker leaves while the teacher is paused at [ENTER]
def scenario_leave_during_pause(tmp):
    """Teaching mode pauses between rounds. A student leaving DURING that pause
    must abort the run and cancel the pause (a stale ENTER wait used to eat the
    next menu command)."""
    c = Class("G", 3, tmp)
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2")
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("G leave during pause: menu reached", False)
            return
        c.menu("3\n16\n1\n1\n")             # naive_allreduce / TEACHING
        if not c.wait_log(r"\[ENTER\]", timeout=30):
            check("G leave during pause: teacher reaches its ENTER pause",
                  False)
            return
        check("G leave during pause: teacher reaches its ENTER pause", True)
        kill(c.workers["w2"])               # student walks out mid-pause
        check("G leave during pause: departure aborts the RUN",
              c.wait_log(r"\[ABORT\] Rank 2 left during the run", timeout=25))
        check("G leave during pause: the pause is cancelled, not left hanging",
              c.wait_log(r"pause cancelled", timeout=25))
        check("G leave during pause: teacher returns to its menu (no wedge)",
              c.wait_count(r"World Size: 2 \(capacity 3, 1 student",
                           1, timeout=45))
        c.menu("3\n16\n2\n1\n")             # performance run, world size 2
        check("G leave during pause: next RUN works",
              c.wait_log(r"Running\.\.\.  \(World Size 2\)", timeout=30)
              and c.wait_log(r"Collective complete", timeout=30)
              and "final = 3" in c.log())
    finally:
        c.close()


# ---------------------------------------------------------------------------
# H. two workers leave back-to-back, then everybody leaves
def scenario_back_to_back_leaves(tmp):
    """Compaction renumbers ranks; the second departure must still be seen
    (the leave handler used to look the worker up by its join-time rank)."""
    c = Class("H", 4, tmp)
    try:
        c.start_teacher()
        for tag, rank in (("w1", 1), ("w2", 2), ("w3", 3)):
            c.start_worker(tag)
            c.wait_log(r"\[JOIN\] .* -> Rank %d " % rank, timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("H back-to-back leaves: menu reached", False)
            return
        kill(c.workers["w2"])
        kill(c.workers["w3"])               # both gone at (nearly) once
        # both must be released; the second one is named with its NEW rank
        # (it was compacted from 3 to 2 before its departure was noticed)
        check("H back-to-back leaves: both departures are seen",
              c.wait_count(r"\[LEAVE\] Rank \d+ disconnected", 2, timeout=30))
        if not c.wait_log(r"\[ROSTER\] Rank \d+ left  ->  World Size 2 "
                          r"\(1 student ranks\)", timeout=25):
            check("H back-to-back leaves: roster is compacted to 2", False)
            return
        check("H back-to-back leaves: roster is compacted to 2", True)
        c.menu("3\n16\n2\n1\n")             # world size 2 -> 1+2 = 3
        check("H back-to-back leaves: RUN works with the survivor",
              c.wait_log(r"Running\.\.\.  \(World Size 2\)", timeout=30)
              and c.wait_log(r"Collective complete", timeout=30)
              and "final = 3" in c.log())
        # ... and if EVERYBODY leaves, a run is Rank 0 alone (World Size 1)
        kill(c.workers["w1"])
        check("H back-to-back leaves: last departure is seen",
              c.wait_log(r"\[LEAVE\] Rank 1 disconnected", timeout=25))
        if not c.wait_log(r"\[ROSTER\] Rank \d+ left  ->  World Size 1 "
                          r"\(0 student ranks\)", timeout=25):
            check("H back-to-back leaves: empty class is reported", False)
            return
        check("H back-to-back leaves: empty class is reported", True)
        c.menu("3\n16\n2\n1\n")
        check("H empty class: a RUN as Rank 0 alone still completes",
              c.wait_log(r"Running\.\.\.  \(World Size 1\)", timeout=30)
              and c.wait_log(r"Collective complete", timeout=30))
    finally:
        c.close()


# ---------------------------------------------------------------------------
# I. a join that dies immediately (before reading its welcome)
def scenario_dead_on_join(tmp):
    c = Class("I", 3, tmp)
    try:
        c.start_teacher()
        c.join_stub()                        # joins, then vanishes
        check("I dead-on-join: the half-open joiner is released",
              c.wait_log(r"\[LEAVE\] Rank \d+ disconnected", timeout=25))
        c.start_worker("w1")
        check("I dead-on-join: a real student gets Rank 1",
              c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25))
        if not c.wait_menu(1, timeout=45):
            check("I dead-on-join: menu reached", False)
            return
        c.menu("3\n16\n2\n1\n")
        check("I dead-on-join: RUN works (world size 2, 1+2=3)",
              c.wait_log(r"Running\.\.\.  \(World Size 2\)", timeout=30)
              and c.wait_log(r"Collective complete", timeout=30)
              and "final = 3" in c.log())
    finally:
        c.close()


# ---------------------------------------------------------------------------
# J. a FROZEN worker (no FIN at all) -> watchdog + exclusion
def scenario_frozen_worker(tmp):
    """SIGSTOP'ing a worker keeps its socket open and its kernel answering, so
    no LEAVE can ever be detected — the stall watchdog must abort the run AND
    drop the silent rank, otherwise every following RUN would stall again."""
    c = Class("J", 3, tmp, teacher_env={"MINIMPI_RUN_TIMEOUT": "20"})
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2")
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("J frozen worker: menu reached", False)
            return
        c.menu("3\n16\n1\n1\n")           # teaching run; w2 is frozen below
        if not c.wait_log(r"Running\.\.\.  \(World Size 3\)", timeout=25):
            check("J frozen worker: run started", False)
            return
        # wait for the teacher's Start Barrier pause: w2 must freeze BEFORE the
        # release, so it never reaches round 1 while w1 does
        c.wait_log(r"\[ENTER\] Start Collective", timeout=25)
        c.freeze("w2")                       # no FIN, kernel still responsive
        c.menu("\n")                        # release the start barrier
        check("J frozen worker: the stall watchdog aborts the RUN",
              c.wait_log(r"watchdog: no rank reported progress", timeout=45))
        check("J frozen worker: the silent rank is named",
              c.wait_log(r"silent=\[2\]", timeout=10))
        check("J frozen worker: the silent rank is excluded",
              c.wait_log(r"\[ROSTER\] excluding Rank 2", timeout=20))
        if not c.wait_count(r"World Size: 2 \(capacity 3, 1 student",
                            1, timeout=45):
            check("J frozen worker: teacher is usable again", False)
            return
        check("J frozen worker: teacher is usable again", True)
        c.menu("3\n16\n2\n1\n")
        check("J frozen worker: the class continues without it",
              c.wait_log(r"Running\.\.\.  \(World Size 2\)", timeout=35)
              and c.wait_log(r"Collective complete", timeout=30)
              and "final = 3" in c.log())
    finally:
        c.close()


# ---------------------------------------------------------------------------
# J2. keepalive is really applied (a vanished machine sends no FIN)
def scenario_keepalive_options():
    """A vanished machine sends no FIN: keepalive is what turns "silently
    gone" into an ordinary disconnect (verified on a real TCP pair)."""
    from minimpi import protocol as P
    ln = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ln.bind(("127.0.0.1", 0))
    ln.listen(1)
    cl = socket.create_connection(ln.getsockname(), timeout=5)
    srv, _ = ln.accept()
    try:
        ok = P.enable_keepalive(srv)
        got = srv.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
        # macOS reports the raw value (not just 1) once keepalive is on
        check("J2 keepalive: SO_KEEPALIVE is set on MiniMPI sockets",
              bool(ok) and got != 0, "SO_KEEPALIVE=%s" % got)
        idle = None
        for opt in [getattr(socket, n, None)
                    for n in ("TCP_KEEPIDLE", "TCP_KEEPALIVE")] + [0x10]:
            if opt is None:
                continue
            try:
                v = srv.getsockopt(socket.IPPROTO_TCP, opt)
            except OSError:
                continue
            if 0 < v <= 60:
                idle = v
                break
        check("J2 keepalive: an idle timeout is configured (not the OS "
              "default 2 h)", idle is not None, "idle=%s" % idle)
    finally:
        for s_ in (srv, cl, ln):
            try:
                s_.close()
            except OSError:
                pass



# ---------------------------------------------------------------------------
# K. a worker leaves during the Performance Benchmark
def scenario_leave_during_benchmark(tmp):
    c = Class("K", 3, tmp, teacher_env={"MINIMPI_BENCH_SIZES": "16777216"})
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2")
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("K benchmark leave: menu reached", False)
            return
        c.menu("7\n1\n")                    # benchmark, rank-0 value 1
        if not c.wait_log(r"raw ", timeout=40):
            check("K benchmark leave: the session starts", False)
            return
        check("K benchmark leave: the session starts", True)
        kill(c.workers["w2"])               # leave in the middle of the cases
        check("K benchmark leave: the session stops instead of printing "
              "bogus rows",
              c.wait_log(r"\[ABORT\] benchmark session cancelled", timeout=60))
        check("K benchmark leave: the teacher returns to its menu",
              c.wait_count(r"World Size: 2 \(capacity 3, 1 student",
                           1, timeout=45))
    finally:
        c.close()


# ---------------------------------------------------------------------------
# L. joining WHILE a run is in flight -> refused, then the worker retries
def scenario_join_during_run(tmp):
    """A student who starts late (mid-run) must not die: the worker is refused,
    waits, and joins as a latecomer once the run finishes."""
    c = Class("L", 4, tmp)
    try:
        c.start_teacher()
        c.start_worker("w1", env_extra={"MINIMPI_INPUT_DELAY": "12"})
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("L join during run: menu reached", False)
            return
        c.menu("3\n16\n1\n1\n")             # teaching run; w1 is still "typing"
        if not c.wait_log(r"Running\.\.\.  \(World Size 2\)", timeout=25):
            check("L join during run: run started", False)
            return
        c.start_worker("w2")                # arrives in the middle of the run
        check("L join during run: the latecomer is refused, not killed",
              c.wait_log(r"\[JOIN REFUSED\]", timeout=30, tag="w2")
              and c.wait_log(r"run is in progress", timeout=5, tag="w2"))
        check("L join during run: the latecomer keeps trying",
              c.wait_log(r"retrying in", timeout=30, tag="w2"))
        # a teacher presses ENTER at EVERY pause: start barrier, each round,
        # and the final close
        c.menu("\n\n\n\n")
        check("L join during run: the latecomer joins after the run",
              c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=90))
        check("L join during run: teacher re-welcomes the class",
              c.wait_log(r"\[ROSTER\] Rank 1 re-welcomed", timeout=30))
        c.menu("3\n16\n2\n1\n")             # world size 3 -> 1+2+3 = 6
        check("L join during run: next RUN includes the latecomer",
              c.wait_log(r"Running\.\.\.  \(World Size 3\)", timeout=40)
              and c.wait_log(r"Collective complete", timeout=30)
              and "final = 6" in c.log())
    finally:
        c.close()


# ---------------------------------------------------------------------------
# M. the teacher KICKS an idle rank (manual removal)
def scenario_kick_idle(tmp):
    """Dynamic membership is automatic, but the teacher must also be able to
    remove somebody on purpose — a stuck machine, or a student who has to go."""
    c = Class("M", 4, tmp)
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2")
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("M kick idle: menu reached", False)
            return
        c.menu("10\n")                      # ask for the roster
        check("M kick idle: the roster lists every rank",
              c.wait_log(r"Class Roster", timeout=20)
              and c.wait_log(r"Rank 2\s+127\.0\.0\.1:", timeout=10)
              and c.wait_log(r"Active world size: 3", timeout=10))
        c.menu("2\ny\n")                  # kick Rank 2 and confirm
        check("M kick idle: the teacher says what it is doing",
              c.wait_log(r"\[KICK\] removing Rank 2", timeout=20))
        check("M kick idle: the kicked worker is told and exits cleanly",
              c.wait_log(r"\[KICKED\]", timeout=25, tag="w2"))
        check("M kick idle: the class continues with the rest",
              c.wait_log(r"\[KICK\] Rank 2 removed.*World size is now 2",
                         timeout=20)
              and c.wait_log(r"\[ROSTER\] Rank \d+ left  ->  World Size 2",
                             timeout=20))
        c.menu("3\n16\n2\n1\n")
        check("M kick idle: next RUN uses World Size 2 (1+2=3)",
              c.wait_log(r"Running\.\.\.  \(World Size 2\)", timeout=30)
              and c.wait_log(r"Collective complete", timeout=30)
              and "final = 3" in c.log())
    finally:
        c.close()


# ---------------------------------------------------------------------------
# N. the teacher KICKS the frozen rank that is stalling a RUN (Ctrl-C tool)
def scenario_kick_stuck_run(tmp):
    """The manual answer to a stuck class: Ctrl-C during a RUN shows the roster
    and lets the teacher remove the rank that stopped answering, so the class
    continues immediately (no waiting for any timeout)."""
    c = Class("N", 3, tmp, teacher_env={"MINIMPI_RUN_TIMEOUT": "120"})
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2")
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("N kick stuck RUN: menu reached", False)
            return
        c.menu("3\n16\n1\n1\n")           # teaching run
        if not c.wait_log(r"Running\.\.\.  \(World Size 3\)", timeout=25):
            check("N kick stuck RUN: run started", False)
            return
        c.freeze("w2")                      # its machine stops answering
        # the teacher watches the class hang until the heartbeats go stale
        time.sleep(8.0)
        os.kill(c.teacher.pid, signal.SIGINT)   # teacher presses Ctrl-C
        check("N kick stuck RUN: Ctrl-C shows the roster, not a crash",
              c.wait_log(r"\[PAUSE\] the RUN is still waiting", timeout=25)
              and c.wait_log(r"Class Roster", timeout=15))
        check("N kick stuck RUN: the frozen rank is called out",
              c.wait_log(r"Rank 2\s+127\.0\.0\.1:\d+\s+SILENT", timeout=20))
        c.menu("2\ny\n")
        check("N kick stuck RUN: the teacher removes it",
              c.wait_log(r"\[KICK\] Rank 2 removed", timeout=20))
        check("N kick stuck RUN: the RUN is aborted, not left waiting",
              c.wait_log(r"\[ABORT\] Rank 2 left during the run", timeout=20))
        if not c.wait_count(r"World Size: 2 \(capacity 3, 1 student",
                            1, timeout=45):
            check("N kick stuck RUN: the class is usable again", False)
            return
        check("N kick stuck RUN: the class is usable again", True)
        c.menu("3\n16\n2\n1\n")
        check("N kick stuck RUN: next RUN completes without it",
              c.wait_log(r"Running\.\.\.  \(World Size 2\)", timeout=35)
              and c.wait_log(r"Collective complete", timeout=30)
              and "final = 3" in c.log())
    finally:
        c.close()


# ---------------------------------------------------------------------------
# O. a rank FAILS inside a RUN -> the RUN ends at once, the class survives
def scenario_rank_error(tmp):
    """A rank that errors cannot finish the RUN; the others would sit in a
    collective until the stall watchdog. The teacher aborts immediately and
    the failing rank stays in the class for the next RUN."""
    c = Class("O", 3, tmp, teacher_env={"MINIMPI_RUN_TIMEOUT": "120"})
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2", env_extra={"MINIMPI_FAIL_RUN": "1"})
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("O rank error: menu reached", False)
            return
        c.menu("3\n16\n2\n1\n")           # performance naive_allreduce
        check("O rank error: the RUN is aborted as soon as a rank fails",
              c.wait_log(r"\[ABORT\] Rank 2 failed: .*forced failure",
                         timeout=30))
        check("O rank error: the class is usable again (no 120 s watchdog)",
              c.wait_log(r"Collective complete|\[ABORTED\]", timeout=20))
        # after a failed RUN the menu is printed again with the SAME roster
        if not c.wait_count(r"World Size: 3 \(capacity 3, 2 student",
                            2, timeout=45):
            check("O rank error: nobody was dropped", False)
            return
        check("O rank error: nobody was dropped", True)
        check("O rank error: both workers survive the failed RUN",
              c.alive("w1") and c.alive("w2"),
              "w1 rc=%s w2 rc=%s" % (c.workers["w1"].poll(),
                                     c.workers["w2"].poll()))
        # A real teacher takes a breath between two runs. NOTE: the assertion
        # below is deliberately about the CLASS, not about one specific rank —
        # see REVIEW_NOTES ("known transport anomaly"): under load the teacher
        # can see a spurious connection reset right after an aborted RUN, in
        # which case it drops that rank (loudly) instead of splitting the
        # world; the class must stay usable either way.
        time.sleep(1.0)
        n_before = c.count(r"Collective complete")
        c.menu("3\n16\n2\n1\n")           # run 2: the hook only fails run 1
        check("O rank error: the next RUN completes (class usable either way)",
              c.wait_count(r"Collective complete", n_before + 1, timeout=40),
              "w1 rc=%s w2 rc=%s" % (c.workers["w1"].poll(),
                                     c.workers["w2"].poll()))
        check("O rank error: no rank is left waiting / no watchdog wait",
              "[ABORTED] watchdog" not in c.log()
              and c.wait_count(r"World Size: \d+ \(capacity", n_before + 2,
                               timeout=45))
    finally:
        c.close()


# ---------------------------------------------------------------------------
# P. Ctrl-C at the MENU still leaves the session (and frees the workers)
def scenario_menu_interrupt(tmp):
    c = Class("P", 2, tmp)
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("P menu Ctrl-C: menu reached", False)
            return
        os.kill(c.teacher.pid, signal.SIGINT)
        end = time.time() + 20
        while time.time() < end and c.teacher.poll() is None:
            time.sleep(0.2)
        check("P menu Ctrl-C: the teacher exits (Ctrl-C is not swallowed)",
              c.teacher.poll() is not None, "rc=%s" % c.teacher.poll())
        w1 = c.workers["w1"]
        end = time.time() + 20
        while time.time() < end and w1.poll() is None:
            time.sleep(0.2)
        check("P menu Ctrl-C: the student worker is released too",
              w1.poll() is not None)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Q. a worker started with no teacher running must not print a traceback
def scenario_worker_no_teacher(tmp):
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               MINIMPI_JOIN_ATTEMPTS="2", MINIMPI_JOIN_RETRY_S="0.1")
    path = os.path.join(tmp, "Q_worker.log")
    with open(path, "w+") as f:
        p = subprocess.Popen([PY, os.path.join(ROOT, "worker.py"),
                              "--server", "127.0.0.1:1"],
                             stdin=subprocess.DEVNULL, stdout=f,
                             stderr=subprocess.STDOUT, text=True, env=env)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            kill(p)
    log = open(path).read()
    check("Q no teacher: the student gets words, not a traceback",
          "Traceback" not in log and "cannot reach the teacher" in log,
          log.strip().splitlines()[-1][:60] if log.strip() else "")
    check("Q no teacher: it gives up with a clear exit code", p.returncode == 3,
          "rc=%s" % p.returncode)


# ---------------------------------------------------------------------------
# R. a typo in MINIMPI_BENCH_SIZES must not kill the teacher
def scenario_bad_bench_env(tmp):
    c = Class("R", 3, tmp,
              teacher_env={"MINIMPI_BENCH_SIZES": "abc,49152,-8"})
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2")
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("R bad bench env: the teacher still starts", False)
            return
        check("R bad bench env: the bad sizes are reported and skipped",
              c.wait_log(r"\[warn\] MINIMPI_BENCH_SIZES ignored abc, -8",
                         timeout=15))
        c.menu("7\n1\n")
        check("R bad bench env: the benchmark still runs on the good size",
              c.wait_log(r"raw naive_allreduce 49152", timeout=40))
        check("R bad bench env: the session finishes normally",
              c.wait_log(r"Benchmark Complete", timeout=40))
    finally:
        c.close()


# ---------------------------------------------------------------------------
# S. "q" at the interrupt prompt aborts the RUN without dropping anybody
def scenario_interrupt_quit_run(tmp):
    c = Class("S", 3, tmp, teacher_env={"MINIMPI_RUN_TIMEOUT": "120"})
    try:
        c.start_teacher()
        c.start_worker("w1")
        c.wait_log(r"\[JOIN\] .* -> Rank 1 ", timeout=25)
        c.start_worker("w2")
        c.wait_log(r"\[JOIN\] .* -> Rank 2 ", timeout=25)
        if not c.wait_menu(1, timeout=45):
            check("S quit RUN: menu reached", False)
            return
        c.menu("3\n16\n1\n1\n")           # teaching run
        if not c.wait_log(r"Running\.\.\.  \(World Size 3\)", timeout=25):
            check("S quit RUN: run started", False)
            return
        c.freeze("w2")
        time.sleep(6.0)                      # let the class look stuck
        os.kill(c.teacher.pid, signal.SIGINT)
        check("S quit RUN: Ctrl-C offers the escape hatch",
              c.wait_log(r"q = abort this RUN", timeout=25))
        c.menu("q\n")
        check("S quit RUN: the RUN is aborted on request",
              c.wait_log(r"\[PAUSE\] RUN aborted", timeout=20))
        check("S quit RUN: nobody was dropped (World Size stays 3)",
              c.wait_count(r"World Size: 3 \(capacity 3, 2 student",
                           1, timeout=45))
    finally:
        c.close()


# ---------------------------------------------------------------------------
# T. classroom mistakes: a busy port and a wrong teacher address
def scenario_busy_port(tmp):
    """Starting the teacher twice (or on a port another program holds) must say
    what is wrong instead of dying with a traceback in front of the class."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        p = subprocess.run([PY, os.path.join(ROOT, "teacher.py"),
                            "--size", "4", "--port", str(port)],
                           capture_output=True, text=True, timeout=40, input="")
        out = p.stdout + p.stderr
        check("T busy port: the teacher explains, no traceback",
              "Traceback" not in out and "already in use" in out
              and p.returncode == 1, "rc=%s" % p.returncode)
        check("T busy port: a usable alternative is suggested",
              ("--port %d" % (port + 1)) in out)
    finally:
        srv.close()


def scenario_wrong_teacher_address(tmp):
    """A student typing the wrong IP:port may reach some other service; they
    must be told, not shown a JSON decode traceback."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    port = srv.getsockname()[1]

    def serve():
        for _ in range(6):
            try:
                c, _peer = srv.accept()
                c.sendall(b"hello, some other program\n")
                c.close()
            except OSError:
                return
    threading.Thread(target=serve, daemon=True).start()
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               MINIMPI_JOIN_ATTEMPTS="2", MINIMPI_JOIN_RETRY_S="0.1")
    try:
        p = subprocess.run([PY, os.path.join(ROOT, "worker.py"),
                            "--server", "127.0.0.1:%d" % port],
                           capture_output=True, text=True, timeout=40, env=env)
        out = p.stdout + p.stderr
        check("T wrong address: the student is told, no traceback",
              "Traceback" not in out and "not a MiniMPI teacher" in out
              and p.returncode == 3, "rc=%s" % p.returncode)
    finally:
        srv.close()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    tmp = tempfile.mkdtemp(prefix="minimpi_robustness_")
    print("Robustness logs: %s\n" % tmp)
    scenario_leave_when_idle(tmp)
    scenario_latecomer_join(tmp)
    scenario_leave_during_run(tmp)
    scenario_watchdog(tmp)
    scenario_teacher_gone(tmp)
    scenario_burst_join(tmp)
    scenario_leave_during_pause(tmp)
    scenario_back_to_back_leaves(tmp)
    scenario_dead_on_join(tmp)
    scenario_keepalive_options()
    scenario_frozen_worker(tmp)
    scenario_leave_during_benchmark(tmp)
    scenario_join_during_run(tmp)
    scenario_kick_idle(tmp)
    scenario_kick_stuck_run(tmp)
    scenario_rank_error(tmp)
    scenario_menu_interrupt(tmp)
    scenario_worker_no_teacher(tmp)
    scenario_bad_bench_env(tmp)
    scenario_interrupt_quit_run(tmp)
    scenario_busy_port(tmp)
    scenario_wrong_teacher_address(tmp)
    print("\nRobustness tests: %d passed, %d failed" % (len(_passed),
                                                        len(_failed)))
    if _failed:
        print("Failed: %s" % ", ".join(_failed))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
