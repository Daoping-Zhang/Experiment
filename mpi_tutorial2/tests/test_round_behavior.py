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
F. Local UI / event report timing: a slow student terminal (arrival already
                    reported) must NOT inflate Round Finished At.

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
    # the slow rank's input delay pushes the Start Barrier late, which shows
    # up in the session wall time (input itself is excluded from Collective
    # Time by design). Teacher prints "Session wall time: X.XXX s".
    m = re.search(r"Session wall time:\s*([0-9.]+)\s*s", out or "")
    wall = float(m.group(1)) if m else 0.0
    ok = not timed_out and "Errors:" not in (out or "") and wall >= 2.5
    check("A. start barrier delays collective for slow rank", ok,
          "wall=%.2fs" % wall)


def test_b_teaching_pause_excluded():
    env = dict(os.environ)
    env["MINIMPI_TEACH_PAUSE"] = "2.0"     # 2 s pause after gather
    # Runner._spawn copies os.environ at spawn time -> the TEACHER child must
    # inherit the pause too, otherwise the pause never actually runs and the
    # test would pass vacuously.
    os.environ["MINIMPI_TEACH_PAUSE"] = "2.0"
    try:
        r = Runner(4, timeout=60)
        import subprocess
        t_start = time.time()
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
        elapsed = time.time() - t_start
        r.kill()
        m = re.search(r"Round Finished At:\s*([0-9.]+)\s*ms", out or "")
        rms = float(m.group(1)) if m else None
        # naive_allreduce = 2 teaching rounds x 2 s teacher pause -> the run
        # must really have taken >= ~3.5 s, while the displayed Round time
        # stays well below 1 s (pause excluded from round timing).
        ok = (not timed_out) and (rms is not None) and rms < 1000 and \
             elapsed >= 3.5 and "Errors:" not in (out or "")
        check("B. teaching pause excluded from Round time", ok,
              "RoundFinished=%.2fms elapsed=%.1fs (pause=2s)"
              % (rms or -1, elapsed))
    finally:
        os.environ.pop("MINIMPI_TEACH_PAUSE", None)


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
        # real project tags: tree_allreduce algorithm channel = 401,
        # round-1 teaching barrier = BARRIER_TAG_BASE(7000) + 1
        a.send_to(1, {"tag": 401}, b"ALGO")     # algorithm tag
        a.send_to(1, {"tag": 7001}, b"BARR")    # barrier tag

    th = threading.Thread(target=send)
    th.start()
    h1, p1 = b.recv_match(tag=401)              # algo recv gets algo message
    h2, p2 = b.recv_match(tag=7001)             # barrier recv gets barrier msg
    th.join(timeout=5)
    ok = p1 == b"ALGO" and p2 == b"BARR" and h1["tag"] == 401 and h2["tag"] == 7001
    a.close(); b.close()
    check("E. algorithm/barrier tags never cross-match", ok)


def test_f_local_ui_does_not_enter_round_time():
    # Arrival is reported BEFORE the student local view prints/upload runs.
    # Slow each worker's local UI by 1.5 s per round (inside the barrier's
    # on_arrived window, after arrival) — Round Finished At must stay tiny
    # while the run as a whole clearly takes >= 2 rounds x 1.5 s.
    env = dict(os.environ)
    env["MINIMPI_LOCAL_VIEW_DELAY"] = "1.5"
    r = Runner(4, timeout=60)
    import subprocess
    t_start = time.time()
    t = r.teacher(["--auto", "--demo", "naive_allreduce", "--mode", "teaching",
                   "--data-size", "16"])
    time.sleep(2)
    ws = []
    for _ in range(3):
        ws.append(subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(HERE), "worker.py"),
             "--server", "127.0.0.1:%d" % r.port],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env))
    deadline = time.time() + 55
    while time.time() < deadline and t.poll() is None:
        time.sleep(0.3)
    timed_out = t.poll() is None
    out = ""
    if not timed_out:
        out, _ = t.communicate(timeout=5)
    elapsed = time.time() - t_start
    r.kill()
    m = re.search(r"Round Finished At:\s*([0-9.]+)\s*ms", out or "")
    rms = float(m.group(1)) if m else None
    # naive_allreduce = 2 teaching rounds; each worker's UI sleeps 1.5 s per
    # round AFTER arrival -> run takes >= ~3 s, yet Round Finished At (gather
    # of arrivals) stays far below 1 s. If UI ran BEFORE arrival, the first
    # round's Round Finished At would be >= 1.5 s and this test would fail.
    ok = (not timed_out) and (rms is not None) and rms < 1000 and \
         elapsed >= 2.5 and "Errors:" not in (out or "")
    check("F. slow local UI does not inflate Round Finished At", ok,
          "RoundFinished=%.2fms elapsed=%.1fs (UI delay=1.5s x2 rounds)"
          % (rms or -1, elapsed))


def main():
    test_d_performance_no_round_barrier()
    test_e_tag_isolation()
    test_a_start_barrier_slow_rank()
    test_b_teaching_pause_excluded()
    test_c_teaching_rounds_block_and_finish()
    test_f_local_ui_does_not_enter_round_time()
    print("\nRound-behavior tests: %d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
