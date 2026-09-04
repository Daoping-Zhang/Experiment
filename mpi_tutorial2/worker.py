#!/usr/bin/env python3
"""worker.py — one student rank (a MiniMPI worker).

Usage:
    python3 worker.py --server <teacher-ip>:<port> [--name Alice]
"""
import argparse
import base64
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minimpi import protocol as P  # noqa: E402
from minimpi.runtime import MiniRuntime  # noqa: E402


def run_demo(rt, control, params):
    """Execute one collective; report per-round + final result to teacher."""
    # The welcome message only contained the peers joined at that moment;
    # the run command carries the full, final peer table — apply it first.
    peers = params.get("peers")
    if peers:
        table = {int(k): (v["host"], int(v["port"]))
                 for k, v in peers.items() if int(k) != rt.rank}
        rt.transport.set_peers(rt.rank, table)

    mode = params.get("mode", "performance")
    rt.mode = mode
    if mode == "teaching":
        from minimpi import barrier as BarrierMod
        rt._report = lambda rnd, evs: control.send(
            {"t": P.C_ROUND_DONE, "rnd": rnd,
             "events": [e.to_dict() for e in evs]})
        rt._barrier = lambda rnd: BarrierMod.barrier(rt.comm, rnd)
    else:
        rt._report = None
        rt._barrier = None

    try:
        result = rt.run_algorithm(params)
        # Do not ship multi-MB results back over the control channel — for
        # payload benchmarks the teacher only needs completion + no errors.
        final = None if params.get("payload") else _encode(result)
        control.send({"t": P.C_DONE, "rank": rt.rank, "final": final,
                      "events": len(rt.events.events)})
    except Exception as e:  # noqa: BLE001
        control.send({"t": P.C_DONE, "rank": rt.rank, "error": str(e),
                      "final": None, "events": 0})


def _encode(value):
    if isinstance(value, (bytes, bytearray)):
        return {"raw": base64.b64encode(bytes(value)).decode()}
    if isinstance(value, list):
        return {"vec": value}
    return {"vec": [value]}


class WorkerShell:
    """Control-plane reader: one blocking thread per worker that reacts to
    teacher commands while algorithm threads run independently."""

    def __init__(self, rt):
        self.rt = rt
        self.control = rt.control
        self.shutdown = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        while not self.shutdown.is_set():
            try:
                m = self.control.recv()
            except (ConnectionError, OSError):
                break
            self._handle(m)
        self.shutdown.set()

    def _handle(self, m):
        t = m.get("t")
        if t == P.C_RUN:
            threading.Thread(target=run_demo,
                             args=(self.rt, self.control, m["params"]),
                             daemon=True).start()
        elif t == P.C_CHECK:
            ok, fail = [], []
            for dst, ep in m.get("peers", {}).items():
                dst = int(dst)
                if dst == self.rt.rank:
                    continue
                try:
                    s = socket.create_connection((ep["host"], int(ep["port"])),
                                                 timeout=3)
                    s.close()
                    ok.append(dst)
                except OSError:
                    fail.append(dst)
            self.control.send({"t": P.C_CHECK_REPORT, "rank": self.rt.rank,
                               "pass": ok, "fail": fail})
        elif t == P.C_SHUTDOWN:
            self.shutdown.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:9000", help="teacher ip:port")
    ap.add_argument("--name", default="student", help="your display name")
    args = ap.parse_args()

    print("========================================\nMiniMPI Worker\n========================================")
    print("Connecting to coordinator (%s)..." % args.server)

    rt = MiniRuntime(name=args.name)
    rt.register_with_teacher(args.server, args.name)
    shell = WorkerShell(rt)

    print("[PASS] Connected")
    print("\nName: %s\nMy Rank: %d\nWorld Size: %d\nMy Endpoint: %s:%d\n" %
          (args.name, rt.rank, rt.size, rt.transport.host, rt.transport.port))
    print("Waiting for teacher commands...")

    while not shell.shutdown.wait(1.0):
        pass
    print("[shutdown]")
    rt.close()


if __name__ == "__main__":
    main()
