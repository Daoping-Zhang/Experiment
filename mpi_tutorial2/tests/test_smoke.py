#!/usr/bin/env python3
"""test_smoke.py — quick in-process smoke tests (no subprocesses).

Covers: imports, P2P transport tag matching, a 1 MB raw payload round trip,
and correctness of combine (sum/xor). The heavy end-to-end suite lives in
scripts/verify.py.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from minimpi.communicator import combine, encode, decode  # noqa: E402

PASS = 0


def ok(name, cond):
    global PASS
    assert cond, "FAIL: %s" % name
    PASS += 1
    print("[PASS] %s" % name)


def test_imports():
    import minimpi.protocol
    import minimpi.transport
    import minimpi.communicator
    import minimpi.metrics
    import minimpi.runtime
    import minimpi.barrier
    import collectives.naive_reduce
    import collectives.naive_allreduce
    import collectives.tree_reduce
    import collectives.tree_allreduce
    import collectives.ring_allreduce
    ok("imports", True)


def test_combine():
    ok("sum i32", combine([1, 2], [3, 4], "sum", "i32") == [4, 6])
    ok("xor raw", combine(b"\x0f\x00", b"\x00\x0f", "xor", "raw") == b"\x0f\x0f")
    payload = encode([1, 2, 3, 4], "i32")
    ok("encode/decode round trip", decode(payload, "i32") == [1, 2, 3, 4])


def test_p2p_tag_and_1mb():
    from minimpi.transport import PeerTransport
    a = PeerTransport("A", bind_host="127.0.0.1")
    b = PeerTransport("B", bind_host="127.0.0.1")
    a.set_peers(0, {1: (b.host, b.port)})
    b.set_peers(1, {0: (a.host, a.port)})
    big = bytes((i * 13) & 0xFF for i in range(1024 * 1024))

    def sender():
        a.send_to(1, {"tag": 1}, big)
        a.send_to(1, {"tag": 2}, b"\x00\x00\x00\x05")
        a.send_to(1, {"tag": 1}, b"\x00\x00\x00\x06")

    th = threading.Thread(target=sender)
    th.start()
    _, p1 = b.recv_match(tag=1)
    _, p2 = b.recv_match(tag=2)
    _, p3 = b.recv_match(tag=1)
    th.join(timeout=10)
    ok("p2p tag matching", p1 == big and p2 == b"\x00\x00\x00\x05" and
       p3 == b"\x00\x00\x00\x06" and not th.is_alive())
    a.close()
    b.close()


def main():
    test_imports()
    test_combine()
    test_p2p_tag_and_1mb()
    print("\n%d smoke checks passed." % PASS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
