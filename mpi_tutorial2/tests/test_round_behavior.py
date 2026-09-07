#!/usr/bin/env python3
"""test_round_behavior.py — behavior tests for the round-based refactor.

A. Start Barrier  : a deliberately slow rank must delay the collective start
                    (other ranks cannot start early).
B. Teaching Pause : teacher pause (ENTER / simulated sleep) must NOT inflate
                    the displayed Round time.
C. Round Blocking : (covered implicitly by B; teaching rounds block until all
                    ranks finish -> barrier gather present).
D. Performance no round barrier : sync_round is a no-op in performance mode.
E. Tag isolation  : algorithm tag and barrier tag never cross-match.

Run:  python3 tests/test_round_behavior.py
"""
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(HERE)))

from _proc import Runner  # noqa: E402

PASS = []
FAIL = []


def check(name, ok, extra=""):
    (PASS if ok else FAIL).append(name)
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  (%s)" % extra) if extra else ""))


# --------------------------------------------------------------------------
def test_a_start_barrier_slow_rank():
    import subprocess
    env = dict(os.environ)
    r = Runner(4, timeout=60)
    t = r.teacher(["--auto", "--demo", "naive_allreduce", "--mode", "performance",
                   "--data-size", "16"])
    time.sleep(2)
    workers = []
    for i in range(3):
        e = dict(env)
        if i == 0:
            e["MINIMPI_INPUT_DELAY"] = "3"     # one rank is 3 s late
        workers.append(subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(HERE), "worker.py"),
             "--server", "127.0.0.1:%d" % r.port],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=e))
    deadline = time.time() + 40
    while time.time() < deadline and t.poll() is None:
        time.sleep(0.3)
    timed_out = t.poll() is None
    out = ""
    if not timed_out:
        out, _ = t.communicate(timeout=5)
    r.kill()
    # teacher wall time is printed as "Collective complete. Total wall time"
    m = re.search(r"Total wall time:\s*([0-9.]+)\s*s", out or "")
    wall = float(m.group(1)) if m else 0.0
    ok = not timed_out and "Errors:" not in (out or "") and wall >= 2.5
    check("A. start barrier delays collective for slow rank", ok,
          "wall=%.2fs" % wall)


def test_b_teaching_pause_excluded():
    env = dict(os.environ)
    env["MINIMPI_TEACH_PAUSE"] = "2.0"     # 2 s pause after gather
    r = Runner(4, timeout=60)
    import subprocess
    t = r.teacher(["--auto", "--demo", "naive_allreduce", "--mode", "teaching",
                   "--data-size", "16"])
    time.sleep(2)
    ws = []
    for _ in range(3):
        ws.append(subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(HERE), "worker.py"),
             "--server", "127.0.0.1:%d" % r.port],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env))
    deadline = time.time() + 50
    while time.time() < deadline and t.poll() is None:
        time.sleep(0.3)
    timed_out = t.poll() is None
    out = ""
    if not timed_out:
        out, _ = t.communicate(timeout=5)
    r.kill()
    m = re.search(r"Round Finished At:\s*([0-9.]+)\s*ms", out or "")
    rms = float(m.group(1)) if m else None
    ok = (not timed_out) and (rms is not None) and rms < 1000 and \
         "Errors:" not in (out or "")
    check("B. teaching pause excluded from Round time", ok,
          "RoundFinished=%.2fms (pause=2s)" % (rms or -1))


def test_c_teaching_rounds_block_and_finish():
    r = Runner(4, timeout=90)
    log, ok, timed = r.run_demo(["--auto", "--demo", "tree_allreduce",
                                 "--mode", "teaching", "--data-size", "16"])
    r.close()
    rounds = len(re.findall(r"Round \d+ -", log))
    check("C. teaching rounds run & block", ok and not timed and rounds >= 2,
          "rounds=%d" % rounds)


def test_d_performance_no_round_barrier():
    # sync_round must be a no-op in performance mode: prove it never blocks
    from minimpi import mpi as M
    from minimpi.runtime import MiniRuntime

    rt = MiniRuntime(name="worker", mode="performance")
    called = {"n": 0}

    def boom(rnd):
        called["n"] += 1
        raise AssertionError("barrier called in performance mode")

    rt._barrier = boom
    rt._report = lambda *a: None
    rt.sync_round(3)
    check("D. performance sync_round is a no-op", called["n"] == 0)
    rt.close()


def test_e_tag_isolation():
    from minimpi.transport import PeerTransport
    a = PeerTransport("A", bind_host="127.0.0.1")
    b = PeerTransport("B", bind_host="127.0.0.1")
    a.set_peers(0, {1: (b.host, b.port)})
    b.set_peers(1, {0: (a.host, a.port)})

    def send():
        a.send_to(1, {"tag": 1001}, b"ALGO")     # algorithm tag
        a.send_to(1, {"tag": 7001}, b"BARR")     # barrier tag

    th = threading.Thread(target=send)
    th.start()
    h1, p1 = b.recv_match(tag=1001)              # algo recv gets algo message
    h2, p2 = b.recv_match(tag=7001)              # barrier recv gets barrier msg
    th.join(timeout=5)
    ok = p1 == b"ALGO" and p2 == b"BARR" and h1["tag"] == 1001 and h2["tag"] == 7001
    a.close(); b.close()
    check("E. algorithm/barrier tags never cross-match", ok)


def main():
    test_d_performance_no_round_barrier()
    test_e_tag_isolation()
    test_a_start_barrier_slow_rank()
    test_b_teaching_pause_excluded()
    test_c_teaching_rounds_block_and_finish()
    print("\nRound-behavior tests: %d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
