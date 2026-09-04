#!/usr/bin/env python3
"""check_env.py — environment prerequisites for MPI Tutorial 2.

Checks: Python version >= 3.8, stdlib socket/threading availability, and a
tiny loopback TCP listen/connect round trip. Exits 0 if ready.
"""
import socket
import sys
import threading


def main():
    fails = []
    print("========================================")
    print("MPI Tutorial 2 — environment check")
    print("========================================")
    print()

    if sys.version_info >= (3, 8):
        print("[PASS] python %d.%d (stdlib only, no third-party deps)" % sys.version_info[:2])
    else:
        print("[FAIL] python too old: %d.%d (>= 3.8 expected)" % sys.version_info[:2])
        fails.append("python")

    for mod in ("socket", "threading", "struct", "json", "uuid", "time"):
        try:
            __import__(mod)
            print("[PASS] module %s" % mod)
        except ImportError:
            print("[FAIL] module %s missing" % mod)
            fails.append(mod)

    # minimal loopback TCP round trip (data plane sanity)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def client():
        c = socket.create_connection(("127.0.0.1", port), timeout=5)
        c.sendall(b"hello")
        c.close()

    threading.Thread(target=client, daemon=True).start()
    conn, _ = srv.accept()
    data = conn.recv(32)
    conn.close()
    srv.close()
    if data == b"hello":
        print("[PASS] loopback TCP round trip")
    else:
        print("[FAIL] loopback TCP round trip")
        fails.append("tcp")

    print()
    if fails:
        print("Environment check FAILED: %s" % ", ".join(fails))
        return 1
    print("Environment OK. No third-party packages required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
