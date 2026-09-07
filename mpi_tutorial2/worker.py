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
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minimpi import protocol as P  # noqa: E402
from minimpi import teaching as T  # noqa: E402
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
        rt._report = None
        rt._on_round = None

        # Per-run local-semantics state (real events + real initial data).
        alg = rt.run_meta["algorithm"]
        ds = int(rt.run_meta["data_size"] or rt.run_meta.get("vector_len") or 0)
        if ds > 0 and rt.run_meta.get("fmt") == "i32":
            initial = [value] * ds
        else:
            initial = [value]
        rt._local_ctx = T.RoundCtx(alg, rt.size, rt.rank, initial)

        # Start Barrier is about to run inside run_algorithm() -> show it.
        print("\nLocal data ready.\n\nEntering MPI Barrier...\n"
              "Waiting for all ranks...")

        def round_barrier(rnd):
            # ARRIVAL first: barrier() sends this rank's [1] to rank 0 the
            # moment its algorithm work finished. The student local view and
            # the C_ROUND_DONE upload then run in the barrier's on_arrived
            # window (while this rank waits for the release), so UI printing
            # and event report never enter Round Finished At.
            BarrierMod.barrier(rt.comm, rnd,
                               on_arrived=lambda _r: _show_and_report(
                                   rt, control, _r))
        rt._barrier = round_barrier
    else:
        rt.show_ui = False
        rt._report = None
        rt._barrier = None
        rt._on_round = None
        rt._local_ctx = None

    try:
        result = rt.run_algorithm(params, value=value)
        if result is not None and not params.get("payload"):
            first = result[0] if isinstance(result, list) and result else result
            if mode == "teaching":
                _print_final(rt, params, result, first)
            else:
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


def _print_final(rt, params, result, first):
    """Student-side 'Collective Complete' block (teaching mode)."""
    alg = params.get("algorithm", "")
    allreduce = alg in ("naive_allreduce", "tree_allreduce", "ring_allreduce")
    print("\n========================================")
    print("Collective Complete")
    print("========================================")
    print("\nResult:")
    if isinstance(result, list) and result:
        print(T.preview_vector(result))
        print("Summary: %s" % first)
        if allreduce:
            print("\nAll elements have the same value.")
            print("Every rank received the same reduced result.")
        elif rt.rank == 0:
            print("\nRank 0 (root) owns the final reduced result.")
        else:
            print("\nThis rank is not the root.")
            print("Its local data is not the global Reduce result.")
    elif result is not None:
        print(str(result))
    print()


def _show_and_report(rt, control, rnd):
    """Student side of a teaching round sync — runs INSIDE the round
    barrier's on_arrived window: this rank already reported its arrival to
    rank 0 (so Round Finished At is fixed), and is now waiting for the
    release. Here it prints its local view and uploads C_ROUND_DONE."""
    _show_round(rt, rnd)          # prints local view; sets rt._snap
    send_ms, work_ms = rt._snap
    control.send({"t": P.C_ROUND_DONE, "rnd": rnd,
                  "events": [e.light_dict()
                             for e in rt.comm.events.by_round(rnd)],
                  "send_ms": send_ms, "work_ms": work_ms})


def _show_round(rt, rnd):
    """Student local view for one finished logical round (teaching mode).

    Structure (spec §12-§14): Algorithm / Phase / Round x / y / Rank,
    My Role, BEFORE, SEND, RECEIVE, OPERATION, AFTER, TIMING.
    """
    # test-only: artificially slow down the LOCAL UI after arrival, to prove
    # UI printing never enters Round Finished At (see tests/test_round_behavior
    # F). Arrival already happened — this only delays the release wait.
    if os.environ.get("MINIMPI_LOCAL_VIEW_DELAY"):
        time.sleep(float(os.environ["MINIMPI_LOCAL_VIEW_DELAY"]))
    meta = rt.run_meta
    alg = meta.get("algorithm", "")
    evs = [e for e in rt.comm.events.by_round(rnd) if e.kind == "algorithm"]
    ctx = getattr(rt, "_local_ctx", None)
    if ctx is None:                       # safety: rebuild if unavailable
        ds = int(meta.get("data_size") or 0)
        base = getattr(rt, "typed_value", rt.rank + 1)
        ctx = T.RoundCtx(alg, rt.size, rt.rank, [base] * (ds or 1))
        rt._local_ctx = ctx
    view = T.describe_round(ctx, rnd, evs)
    lines = [T.local_view_text(view)]
    send_ms, work_ms = rt.comm_world.snapshot_ms()
    rt._snap = (send_ms, work_ms)      # uploaded with C_ROUND_DONE (teacher)
    send_txt = "N/A" if send_ms is None else "%.2f ms" % send_ms
    lines.append("\nTIMING")
    lines.append("My Send Finished: %s" % send_txt)
    # Debug/advanced only — NOT a "sync wait": it is this rank's own-clock
    # local work finish (no teacher clock, no other ranks involved).
    if os.environ.get("MINIMPI_SHOW_WORK"):
        work_txt = "N/A" if work_ms is None else "%.2f ms" % work_ms
        lines.append("My Round Work Finished: %s   (own clock)" % work_txt)
    lines.append("\nWaiting for the whole round...")
    print("\n" + "\n".join(lines))


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
