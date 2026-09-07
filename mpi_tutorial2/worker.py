#!/usr/bin/env python3
"""worker.py — one student rank (a MiniMPI worker).

Usage:
    python3 worker.py --server <teacher-ip>:<port>
"""
import argparse
import base64
import os
import socket
import sys
import threading

from minimpi.metrics import fmt_bytes

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
    rt.run_meta = {"algorithm": params.get("algorithm", ""),
                   "data_size": params.get("data_size", params.get("vector_len", 0)),
                   "vector_len": params.get("vector_len", 0),
                   "fmt": params.get("fmt", "i32")}

    # ---- student inputs ONE integer for this run ----------------------
    ds = rt.run_meta["data_size"]
    print("\nAlgorithm: %s\nData Size: %s elements\n" %
          (rt.run_meta["algorithm"], ds))
    value = _read_one_int(rt)          # [value] * data_size
    rt.typed_value = value

    if mode == "teaching":
        from minimpi import barrier as BarrierMod
        rt.show_ui = True
        rt._report = lambda rnd, evs: control.send(
            {"t": P.C_ROUND_DONE, "rnd": rnd,
             "events": [e.to_dict() for e in evs],
             "send_ms": rt._snap[0] if getattr(rt, "_snap", None) else None,
             "work_ms": rt._snap[1] if getattr(rt, "_snap", None) else None})
        rt._barrier = lambda rnd: BarrierMod.barrier(rt.comm, rnd)
        rt._on_round = lambda rnd: _show_round(rt, rnd)
    else:
        rt.show_ui = False
        rt._report = None
        rt._barrier = None
        rt._on_round = None

    try:
        result = rt.run_algorithm(params, value=value)
        if result is not None and not params.get("payload"):
            first = result[0] if isinstance(result, list) and result else result
            print("\nResult: %s\n" % first)
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


def _read_one_int(rt):
    """Each student types one integer per RUN; vector = [n] * data_size.
    Headless (non-tty) fallback: rank + 1 so automation stays deterministic."""
    import os as _os, time as _time
    if _os.environ.get("MINIMPI_INPUT_DELAY"):
        _time.sleep(float(_os.environ["MINIMPI_INPUT_DELAY"]))
    if sys.stdin.isatty():
        try:
            line = input("Input one integer:\n> ").strip()
            n = int(line)
        except (EOFError, ValueError):
            n = rt.rank + 1
        return n
    return rt.rank + 1


def _show_round(rt, rnd):
    """Student local view for one finished logical round (teaching mode)."""
    meta = rt.run_meta
    alg = meta.get("algorithm", "")
    ds = meta.get("data_size", "")
    evs = [e for e in rt.comm.events.by_round(rnd) if e.kind == "algorithm"]
    print("\nRound %d    (algorithm: %s, data size: %s)" % (rnd, alg, ds))
    if not evs:
        print("  (this rank does not communicate this round)")
    for e in evs:
        if e.side == "send":
            print("  Send:\n  Rank %d -> Rank %d\n  %s" %
                  (rt.rank, e.destination, fmt_bytes(e.payload_bytes)))
        else:
            print("  Receive:\n  Rank %d <- Rank %d\n  %s" %
                  (rt.rank, e.source, fmt_bytes(e.payload_bytes)))
    send_ms, work_ms = rt.comm_world.snapshot_ms()
    rt._snap = (send_ms, work_ms)
    send_txt = "N/A" if send_ms is None else "%.2f ms" % send_ms
    print("\nMy Send Finished At: %s" % send_txt)
    if send_ms is not None and work_ms is not None:
        print("Round Work Finished At: %.2f ms" % work_ms)
        print("Waiting After My Send: %.2f ms" % max(0.0, work_ms - send_ms))
    else:
        print("Round Work Finished At: %.2f ms" % (work_ms or 0.0))
    print("\nWaiting for other ranks...")


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
    ap.add_argument("--server", default="127.0.0.1:9000", help="rank 0 ip:port")
    args = ap.parse_args()

    from minimpi import mpi as M

    # ---- MPI session starts: Init does connect/join/rank/size + COMM_WORLD
    M.Init(server=args.server)

    comm = M.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("\nMiniMPI Worker")
    print("Rank: %d / %d\n" % (rank, size))
    print("Waiting for Rank 0...")

    # ---- wait/run loop (teacher commands drive collectives from here)
    rt = M._session["rt"]
    shell = WorkerShell(rt)
    while not shell.shutdown.wait(1.0):
        pass
    print("[shutdown]")

    # ---- MPI session ends -------------------------------------------------
    M.Finalize()


if __name__ == "__main__":
    main()
