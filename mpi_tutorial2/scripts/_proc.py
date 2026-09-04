"""_proc.py — subprocess helpers shared by local_demo.py / verify.py.

Spawns the real teacher.py and worker.py processes on 127.0.0.1, waits for
the teacher to exit, and cleans every child up on timeout.
"""
import os
import signal
import socket
import subprocess
import sys
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


class Runner:
    def __init__(self, size, timeout=90):
        self.size = size
        self.timeout = timeout
        self.port = free_port()
        self.children = []

    def _spawn(self, args, tag):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=env)
        self.children.append(p)
        return p

    def teacher(self, extra):
        args = [PY, os.path.join(ROOT, "teacher.py"),
                "--size", str(self.size), "--host", "127.0.0.1",
                "--port", str(self.port), "--advertise", "127.0.0.1",
                "--auto"] + extra
        return self._spawn(args, "teacher")

    def workers(self, names=None):
        ps = []
        for i in range(1, self.size):
            args = [PY, os.path.join(ROOT, "worker.py"),
                    "--server", "127.0.0.1:%d" % self.port,
                    "--name", (names[i - 1] if names else "W%d" % i)]
            ps.append(self._spawn(args, "worker%d" % i))
        return ps

    def run_demo(self, extra, expect_exit=True):
        """Start teacher + (size-1) workers; return (teacher_log, ok, timed_out)."""
        t = self.teacher(extra)
        time.sleep(2.0)          # let the teacher listen and print the banner
        self.workers()

        log = []
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if t.poll() is not None:
                break
            time.sleep(0.3)
        timed_out = t.poll() is None
        if timed_out:
            self.kill()
        out, _ = t.communicate(timeout=5) if not timed_out else (None, None)
        log.append(out or "")
        return "\n".join(log), not timed_out, timed_out

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
