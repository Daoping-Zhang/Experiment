#!/usr/bin/env python3
"""test_version_guard.py — worker/teacher must run the SAME MiniMPI version.

A worker whose version differs is refused at join (teacher prints
'[VERSION] rejected a worker ...') and exits with [VERSION MISMATCH] +
update instructions. A same-version worker joins normally.

Run:  python3 tests/test_version_guard.py
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))

from _proc import Runner  # noqa: E402
from minimpi import protocol as P  # noqa: E402

PASS = []
FAIL = []


def check(name, ok, extra=""):
    (PASS if ok else FAIL).append(name)
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  (%s)" % extra) if extra else ""))


def _worker_cmd(runner):
    return [sys.executable, os.path.join(os.path.dirname(HERE), "worker.py"),
            "--server", "127.0.0.1:%d" % runner.port]


def test_version_guard():
    r = Runner(4, timeout=60)
    t = r.teacher(["--auto", "--demo", "naive_allreduce", "--mode",
                   "performance", "--data-size", "4"])
    time.sleep(2)

    # 1) outdated worker -> refused, friendly message, exit code 2
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["MINIMPI_FAKE_VERSION"] = "0.0.0-old-copy"
    bad = subprocess.Popen(_worker_cmd(r), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env=env)
    bad_out, _ = bad.communicate(timeout=25)
    ok_bad = (bad.returncode == 2
              and "[VERSION MISMATCH]" in bad_out
              and "please update" in bad_out.lower())

    # 2) same-version worker joins normally (banner shows the version)
    good = subprocess.Popen(_worker_cmd(r), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env)
    time.sleep(3)
    good.kill()
    good_out, _ = good.communicate(timeout=5)
    ok_good = ("MiniMPI Worker version %s" % P.MINIMPI_VERSION) in good_out

    r.kill()
    t_out, _ = t.communicate(timeout=5)
    ok_teacher = "[VERSION] rejected a worker" in (t_out or "")

    check("worker with old version is refused + told to update",
          ok_bad, "rc=%s" % bad.returncode)
    check("teacher logs the rejection", ok_teacher)
    check("same-version worker joins normally", ok_good)


def main():
    test_version_guard()
    print("\nVersion-guard tests: %d passed, %d failed"
          % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
