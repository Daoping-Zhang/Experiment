#!/usr/bin/env python3
"""verify.py — automated acceptance checks for MPI Tutorial 2.

Scenarios (each with a timeout; any hang == FAIL):
  1. import & module sanity
  2. in-process: tag matching + 4 MB payload over real TCP transports
  3. naive_reduce      (4 ranks, performance)  root correct
  4. naive_allreduce   (4 ranks, performance)  every rank correct
  5. tree_allreduce    (4 ranks, performance)  every rank correct
  6. ring_allreduce    (4 ranks, performance)  every rank correct
  7. naive_allreduce   (4 ranks, teaching)     finishes with correct result
  8. tree_allreduce    (4 ranks, teaching)     finishes with correct result
  9. large payload     (4 ranks, 4 MB naive_allreduce) completes
 10. clean shutdown (teacher exits; children reaped)
"""
import re
import sys
import threading

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, __import__("os").path.join(
    __import__("os").path.dirname(__file__), ".."))

EXPECTED = "[10,14,18,22]"
EXPECTED_128 = "[128,132,136,140]"

_passed = []
_failed = []


def check(name, ok, extra=""):
    if ok:
        _passed.append(name)
        print("[PASS] %s%s" % (name, ("  (%s)" % extra) if extra else ""))
    else:
        _failed.append(name)
        print("[FAIL] %s%s" % (name, ("  (%s)" % extra) if extra else ""))


# ---------------------------------------------------------------------------
def scenario_imports():
    try:
        import minimpi.protocol
        import minimpi.transport
        import minimpi.communicator
        import minimpi.metrics
        import minimpi.runtime
        import minimpi.collectives_dispatch
        for m in ("ping_pong", "naive_reduce", "naive_allreduce",
                  "tree_reduce", "tree_allreduce", "ring_allreduce"):
            __import__("collectives." + m)
        check("imports", True)
    except Exception as e:  # noqa: BLE001
        check("imports", False, str(e))


def scenario_p2p_tag_and_large():
    try:
        import time
        from minimpi.transport import PeerTransport
        a = PeerTransport("A", bind_host="127.0.0.1")
        b = PeerTransport("B", bind_host="127.0.0.1")
        a.set_peers(0, {1: (b.host, b.port)})
        b.set_peers(1, {0: (a.host, a.port)})
        big = bytes((i * 7) & 0xFF for i in range(4 * 1024 * 1024))

        def sender():
            a.send_to(1, {"tag": 1, "plen": len(big), "fmt": "raw"}, big)
            a.send_to(1, {"tag": 2, "plen": 4}, b"\x00\x00\x00\x07")
            a.send_to(1, {"tag": 1, "plen": 4}, b"\x00\x00\x00\x09")

        th = threading.Thread(target=sender)
        th.start()
        h1, p1 = b.recv_match(tag=1)          # first tag-1 message (big)
        h2, p2 = b.recv_match(tag=2)          # tag 2 arrives out of order
        h3, p3 = b.recv_match(tag=1)          # second tag-1 message
        th.join(timeout=10)
        ok = len(p1) == len(big) and p2 == b"\x00\x00\x00\x07" and \
             p3 == b"\x00\x00\x00\x09" and not th.is_alive()
        a.close(); b.close()
        check("p2p tag matching + 4MB payload", ok)
    except Exception as e:  # noqa: BLE001
        check("p2p tag matching + 4MB payload", False, str(e))


# ---------------------------------------------------------------------------
from _proc import Runner  # noqa: E402

FINALS_RE = re.compile(r"Rank (\d+) final = (\[[^\]]+\])")


def finals_from(log):
    return {int(m[0]): m[1] for m in FINALS_RE.findall(log)}


def e2e(name, demo, mode, expected_all, payload=None, size=4, expected_root=None):
    r = Runner(size)
    args = ["--demo", demo, "--mode", mode]
    if payload:
        args += ["--payload", str(payload)]
    log, ok, timed = r.run_demo(args)
    r.close()
    if timed:
        check(name, False, "timeout")
        return
    if payload:
        check(name, ok and "Errors:" not in log and "Traceback" not in log,
              "4MB payload demo completed")
        return
    finals = finals_from(log)
    if expected_root is not None:          # reduce: only the root changes
        good = len(finals) == size and finals.get(0) == expected_root
        check(name, good, "finals=%s" % sorted(finals.items()))
        return
    vals_ok = len(finals) == size and set(finals.values()) == {expected_all}
    check(name, vals_ok, "finals=%s" % sorted(finals.items()))


def main():
    print("========================================")
    print("MPI Tutorial 2 Verification")
    print("========================================")
    print()

    scenario_imports()
    scenario_p2p_tag_and_large()

    e2e("naive_reduce perf", "naive_reduce", "performance", "[10]",
        expected_root="[10]")
    e2e("naive_allreduce perf", "naive_allreduce", "performance", EXPECTED)
    e2e("tree_allreduce perf", "tree_allreduce", "performance", EXPECTED)
    e2e("ring_allreduce perf", "ring_allreduce", "performance", EXPECTED)
    e2e("naive_allreduce teaching", "naive_allreduce", "teaching", EXPECTED)
    e2e("tree_allreduce teaching", "tree_allreduce", "teaching", EXPECTED)
    e2e("ring_allreduce teaching", "ring_allreduce", "teaching", EXPECTED)
    e2e("naive_allreduce size8", "naive_allreduce", "performance",
        "[36,44,52,60]", size=8)
    e2e("4MB payload naive_allreduce", "naive_allreduce", "performance",
        None, payload=4 * 1024 * 1024)

    for mode in ("performance", "teaching"):
        name = "ping_pong %s" % mode
        r = Runner(4)
        log, ok, timed = r.run_demo(["--demo", "ping_pong", "--mode", mode])
        r.close()
        check(name, not timed and "Errors:" not in log and len(finals_from(log)) == 4,
              "ping_pong %s completed" % mode)


    print("\n========================================")
    print("Results: %d passed, %d failed" % (len(_passed), len(_failed)))
    print("========================================")
    if _failed:
        print("FAILED: %s" % ", ".join(_failed))
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
