#!/usr/bin/env python3
"""worker.py — one student rank: a real MPI-style program.

Read the main flow top-down:

    MPI.Init(...)                 # 1  one MPI session for this process
    comm = MPI.COMM_WORLD         # 2  the world communicator
    rank = comm.Get_rank()        # 3  my identity
    size = comm.Get_size()

    while True:
        run = classroom.wait_for_run()    # 4  wait for the teacher's next RUN
        if run is None:                   #    (None == session shutdown)
            break
        value = read_one_integer(...)     # 5  I type one integer
        data = [value] * run.data_size    # 6  my local data vector
        comm.Barrier()                    # 7  Start Barrier: wait for everyone
        result = run_one_collective(...)  # 8  the real collective
        show_result(result)               # 9  my result

    MPI.Finalize()                # 10 end the MPI session

The classroom control plane (join / RUN / SHUTDOWN) is teaching
infrastructure, not fake MPI API — it stays inside ClassroomWorker.
"""
import argparse
import base64
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minimpi import mpi as M               # noqa: E402
from minimpi import protocol as P          # noqa: E402
from minimpi import teaching as T          # noqa: E402
from minimpi.classroom_worker import ClassroomWorker  # noqa: E402


def main():
    args = parse_args()

    MPI = M                                 # MPI.Init starts ONE session:
    MPI.Init(server=args.server)            # connects/joins + COMM_WORLD
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print_banner(rank, size)
        run_classroom(comm)
    finally:
        MPI.Finalize()                      # every exit path ends the session


def run_classroom(comm):
    """The student's run loop: wait -> input -> data -> barrier -> collective."""
    classroom = ClassroomWorker.attach_session()
    try:
        while True:
            run = classroom.wait_for_run()       # blocks until the teacher
            if run is None:                      # sends a RUN (None == end)
                print("[shutdown]")
                return

            # Performance Benchmark session setup: ONE input, all later cases
            # reuse that value — no per-case prompts, no zero Data Size.
            if run.kind == "benchmark_setup":
                print("\n========================================\n"
                      "Performance Benchmark Setup\n"
                      "========================================")
                print("\nThis value will be reused for all benchmark cases.")
                classroom.set_benchmark_value(read_one_integer(comm.Get_rank()))
                print("\nBenchmark value: %s\n" % classroom.benchmark_value())
                continue

            if run.kind == "benchmark_case":
                value = classroom.benchmark_value()
                if value is None:                # safety: never re-prompt
                    value = comm.Get_rank() + 1
                if not classroom.benchmark_announced():
                    print("\nPerformance Benchmark Running...\n\n"
                          "Local benchmark value: %s\n\nPlease wait." % value)
                    classroom.mark_benchmark_announced()
            else:
                value = read_one_integer(comm.Get_rank())

            data = None if run.payload else [value] * run.data_size

            if run.kind != "benchmark_case":
                if run.data_size > 0:
                    print("\nAlgorithm: %s\nData Size: %d elements\n"
                          % (run.algorithm, run.data_size))
                else:
                    print("\nAlgorithm: %s\nLocal Payload per Rank: %d B\n"
                          % (run.algorithm, run.payload))
                if run.mode == "teaching":
                    print("Local data ready.\n\nEntering MPI Barrier...\n"
                          "Waiting for all ranks...")

            comm.Barrier()               # Start Barrier (both modes): the
                                         # collective only starts when every
                                         # rank is ready

            result = run_one_collective(classroom, run, data)
            if run.kind != "benchmark_case":
                show_result(comm.Get_rank(), run, result)
    finally:
        classroom.close()


def run_one_collective(classroom, run, data):
    """Prepare one RUN and execute it on the session runtime.

    Returns the collective's real result (or None on error, which is
    reported to the teacher). This is where teaching hooks are installed —
    the algorithm files stay untouched.
    """
    rt = classroom.runtime
    control = rt.control
    params = run.params

    # The welcome only contained the peers joined at that moment; the RUN
    # command carries the full peer table — apply it first.
    peers = params.get("peers")
    if peers:
        table = {int(k): (v["host"], int(v["port"]))
                 for k, v in peers.items() if int(k) != rt.rank}
        rt.transport.set_peers(rt.rank, table)

    rt.mode = run.mode
    rt.run_meta = {"algorithm": run.algorithm, "data_size": run.data_size,
                   "vector_len": params.get("vector_len", 0),
                   "fmt": params.get("fmt", "i32")}

    if run.mode == "teaching":
        from minimpi import barrier as BarrierMod
        rt.show_ui = True
        rt._report = None
        rt._on_round = None

        # Per-run local-semantics state (real events + real initial data).
        if run.data_size > 0 and run.fmt == "i32":
            rt._local_ctx = T.RoundCtx(run.algorithm, rt.size, rt.rank, data)
        else:
            rt._local_ctx = None

        def round_barrier(rnd):
            # ARRIVAL first: barrier() sends this rank's [1] to rank 0 the
            # moment its algorithm work finished. The student local view +
            # C_ROUND_DONE upload run inside the barrier's on_arrived window
            # (while waiting for the release), so UI/upload never enter the
            # local-work / ready timings.
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
        result = rt.run_algorithm(params, value=data, barrier=False)
        final = None if params.get("payload") else _encode(result)
        control.send({"t": P.C_DONE, "rank": rt.rank, "final": final,
                      "events": len(rt.events.events)})
        return result
    except Exception as e:  # noqa: BLE001
        control.send({"t": P.C_DONE, "rank": rt.rank, "error": str(e),
                      "final": None, "events": 0})
        return None


# --------------------------------------------------------------------------
# helpers below — the run loop above is the student-facing program
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:9000",
                    help="rank 0 ip:port")
    return ap.parse_args()


def print_banner(rank, size):
    print("\nMiniMPI Worker")
    print("Rank: %d / %d\n" % (rank, size))
    print("Waiting for Rank 0...")


def read_one_integer(rank):
    """Each student types one integer per RUN; vector = [n] * data_size.
    Headless (non-tty) fallback: rank + 1 so automation stays deterministic."""
    if os.environ.get("MINIMPI_INPUT_DELAY"):
        time.sleep(float(os.environ["MINIMPI_INPUT_DELAY"]))
    if sys.stdin.isatty():
        try:
            line = input("Input one integer:\n> ").strip()
            n = int(line)
        except (EOFError, ValueError):
            n = rank + 1
        return n
    return rank + 1


def show_result(rank, run, result):
    if result is None or run.params.get("payload"):
        return
    first = result[0] if isinstance(result, list) and result else result
    if run.mode == "teaching":
        _print_final(rank, run, result, first)
    else:
        print("\nResult: %s\n" % first)


def _print_final(rank, run, result, first):
    """Student-side 'Collective Complete' block (teaching mode)."""
    alg = run.algorithm
    allreduce = alg in ("naive_allreduce", "recursive_doubling_allreduce", "ring_allreduce")
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
        elif rank == 0:
            print("\nRank 0 (root) owns the final reduced result.")
        else:
            print("\nThis rank is not the root.")
            print("Its local data is not the global Reduce result.")
    elif result is not None:
        print(str(result))
    print()


def _encode(value):
    if isinstance(value, (bytes, bytearray)):
        return {"raw": base64.b64encode(bytes(value)).decode()}
    if isinstance(value, list):
        return {"vec": value}
    return {"vec": [value]}


def _show_and_report(rt, control, rnd):
    """Student side of a teaching round sync — runs INSIDE the round
    barrier's on_arrived window: arrival already reported to rank 0 (so all
    round timings are fixed), now print the local view + upload events while
    waiting for the release."""
    _show_round(rt, rnd)
    control.send({"t": P.C_ROUND_DONE, "rnd": rnd,
                  "events": [e.light_dict()
                             for e in rt.comm.events.by_round(rnd)]})


def _show_round(rt, rnd):
    """Student local view for one finished logical round (teaching mode)."""
    # test-only: slow down the LOCAL UI after arrival (proves UI never
    # enters round timing; see tests/test_round_behavior F).
    if os.environ.get("MINIMPI_LOCAL_VIEW_DELAY"):
        time.sleep(float(os.environ["MINIMPI_LOCAL_VIEW_DELAY"]))
    evs = [e for e in rt.comm.events.by_round(rnd) if e.kind == "algorithm"]
    ctx = getattr(rt, "_local_ctx", None)
    if ctx is None:
        ds = int(rt.run_meta.get("data_size") or 0)
        base = getattr(rt, "typed_value", rt.rank + 1)
        ctx = T.RoundCtx(rt.run_meta.get("algorithm", ""), rt.size, rt.rank,
                         [base] * (ds or 1))
        rt._local_ctx = ctx
    view = T.describe_round(ctx, rnd, evs)
    lines = [T.local_view_text(view)]
    # Local Clock: every value on THIS rank's own clock, ms from Round Start
    lines.append("\nLOCAL TIMELINE")
    lines.extend(T.local_timeline_text(rt.comm_world.local_timings()))
    lines.append("\nWaiting at round synchronization...")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
