"""_proc.py — subprocess helpers shared by local_demo.py / verify.py.

Spawns the real teacher.py and worker.py processes on 127.0.0.1, waits for the
teacher to exit, and cleans every child up on timeout.

Two things a classroom harness must get right (a lesson is not a batch job):

  * never start a worker before the teacher's control socket LISTENS — a
    worker that arrives too early dies with "connection refused" and the
    teacher then waits for a rank that can never arrive;
  * always DRAIN every child's stdout — a full 64 KB pipe blocks a worker in
    the middle of a collective.
"""
import os
import signal
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _tail(text, lines=25):
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


class Runner:
    def __init__(self, size, timeout=90):
        self.size = size
        self.timeout = timeout
        self.port = free_port()
        self.children = []
        self._buf = {}          # Popen -> [lines] (only when pumped)

    def _spawn(self, args, tag, pump=False):
        """`pump=False` keeps the plain PIPE that callers may read with
        communicate(); `pump=True` drains the pipe in a thread so long
        classroom logs can never block a child."""
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=env)
        if pump:
            buf = []
            self._buf[p] = buf
            threading.Thread(target=self._pump, args=(p, buf),
                             daemon=True).start()
        self.children.append(p)
        return p

    @staticmethod
    def _pump(p, buf):
        try:
            for line in p.stdout:
                buf.append(line)
        except (ValueError, OSError):
            pass

    def log_of(self, p):
        return "".join(self._buf.get(p, []))

    def teacher(self, extra, pump=False):
        args = [PY, os.path.join(ROOT, "teacher.py"),
                "--size", str(self.size), "--host", "127.0.0.1",
                "--port", str(self.port), "--advertise", "127.0.0.1",
                "--auto"] + extra
        return self._spawn(args, "teacher", pump=pump)

    def wait_listen(self, t, timeout=40):
        """Wait until the teacher really accepts joins (not merely 'started').
        Requires a pumped teacher."""
        needle = "Coordinator: 127.0.0.1:%d" % self.port
        end = time.time() + timeout
        while time.time() < end:
            if needle in self.log_of(t):
                return True
            if t.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    def workers(self, names=None, pump=False):
        ps = []
        for i in range(1, self.size):
            args = [PY, os.path.join(ROOT, "worker.py"),
                    "--server", "127.0.0.1:%d" % self.port]
            ps.append(self._spawn(args, "worker%d" % i, pump=pump))
        return ps

    def run_demo(self, extra, expect_exit=True):
        """Start teacher + (size-1) workers; return (teacher_log, ok, timed_out)."""
        t = self.teacher(extra, pump=True)
        self.wait_listen(t)
        self.workers(pump=True)

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if t.poll() is not None:
                break
            time.sleep(0.3)
        timed_out = t.poll() is None
        log = self.log_of(t)
        if timed_out:
            # a hang must be diagnosable from the artifact alone: keep every
            # rank's last words before the children are killed
            log += "\n----- TIMEOUT: last output of every rank -----\n"
            log += _tail(log)
            for i, w in enumerate(self.children[1:], start=1):
                log += "\n----- worker %d -----\n%s" % (i, _tail(self.log_of(w)))
            self.kill()
        return log, not timed_out, timed_out

    def kill(self):
        for p in self.children:
            try:
                p.kill()
            except Exception:
                pass
        for p in self.children:
            try:
                p.wait(timeout=3)
            except Exception:
                pass
        self.children = []

    def close(self):
        self.kill()
